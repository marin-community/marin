# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protein distogram benchmark — loss-based eval wired into training.

Each benchmark target is a PDB entry (plus optional chain + assembly spec).
For every (target, seeded-contact count N in 0..5) we sample
`DEFAULT_PAIRS_PER_TARGET` random residue pairs (i < j) and emit one
pre-tokenized validation example per pair:

    <contacts-and-distances-v1>
    <begin_sequence> <AA_1> … <AA_n>
    <begin_statements>
    [N seeded GT long-range contacts]
    <distance> <p_i> <p_j> <CA> <CA>
    <d_{gt_bin}>

`loss_weights` is zero everywhere except the final `<d_...>` token, so
levanter's tagged-eval harness (`cb_tagged_lm_evaluate`, runs at every
`steps_per_eval`) computes cross-entropy on exactly that token.

## Metrics logged to wandb at every eval step

    eval/protein_dist/loss                   # micro-avg (token-weighted) over everything
    eval/protein_dist/macro_loss             # dataset-weighted macro across leaves
    eval/protein_dist/<pdb>/macro_loss       # macro across N for a single PDB
    eval/protein_dist/N<N>/macro_loss        # macro across PDBs for a single N
    eval/protein_dist_<pdb>_N<N>/loss        # leaf per-(pdb, N) loss

(Identical `/bpb` variants in bits-per-byte.)

## Adding a protein target

Append a `ProteinTarget` to `TARGETS` below and re-run the training
experiment. Marin's executor auto-builds the 6 eval-data parquets for your
new target on the next run and caches them thereafter.

Fields on `ProteinTarget`:
    pdb_id   — RCSB PDB ID, e.g. "1QYS". For FoldBench-style entries written
               as "5sbj-assembly1", just pass "5sbj" and set assembly=1.
    chain_id — None (default) means "first chain in the fetched file". For
               multi-chain deposited structures you want one specific chain of,
               set it explicitly.
    assembly — 0 (default) fetches the deposited asymmetric unit (.pdb.gz).
               1, 2, ... fetch biological assemblies (.pdb1.gz, .pdb2.gz, …).
               FoldBench entries are biological-assembly monomers → use 1.
    label    — optional display name; defaults to pdb_id.lower(). Used in
               metric paths and output directory names.

Currently only CA-CA distances are evaluated. Side-chain atom queries can
be added later by extending `_pair_tail_tokens` / `sample_eval_pairs`.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import logging
from dataclasses import dataclass, field
from urllib.request import urlopen

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from levanter.data.text import DatasetComponent, PrebuiltLmDatasetFormat, UrlDatasetSourceConfig
from marin.execution.executor import ExecutorStep, this_output_path, versioned

logger = logging.getLogger(__name__)

# ---- Constants ----

DISTANCE_BIN_WIDTH_A = 0.5
NUM_DISTANCE_BINS = 64  # tokenizer: <d0.5> .. <d32.0>
DISTANCE_MAX_A = NUM_DISTANCE_BINS * DISTANCE_BIN_WIDTH_A  # 32.0

# Which 3-letter residue names are "standard" amino acids (the ones our
# tokenizer has <AA> tokens for).
STANDARD_AA_3LETTER = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)

CB_CONTACT_ANGSTROMS = 8.0
LONG_RANGE_SEQ_SEP = 24  # matches the training-data <long-range-contact> definition


# ---- Target dataclass + registry ----


@dataclass(frozen=True)
class ProteinTarget:
    pdb_id: str
    chain_id: str | None = None  # None → first chain in the fetched file
    assembly: int = 0  # 0 = deposited asym unit, 1+ = biological assembly
    label: str | None = None
    # Optional sequence override. If set, the `<begin_sequence>` section of
    # the prompt uses these 3-letter residue names instead of the PDB's native
    # sequence. Ground-truth CA coordinates still come from the PDB structure,
    # so the eval asks "given this (redesigned) sequence, what's the CA-CA
    # distance in the native-like structure?". Length must match the parsed
    # chain length; all entries must be standard AAs.
    sequence_override: tuple[str, ...] | None = None
    # Metadata identifying the sequence source — "pdb" (default) or a
    # redesign method like "mpnn". Folded into the step name and metric paths.
    sequence_origin: str = "pdb"

    @property
    def name(self) -> str:
        return (self.label or self.pdb_id).lower()


# FoldBench monomer benchmark (first 10 entries of
# https://github.com/BEAM-Labs/FoldBench/blob/main/targets/monomer_protein.csv).
# CSV `chain_id=A` refers to the *assembly* chain labeling, which may differ
# from the deposited label — so we fetch the .pdb1 biological assembly and use
# the first chain encountered.
#
# The CSV's #1 entry (5sbj) is a 26-residue peptide with only 3 long-range
# GT contacts — too small to support seeded N=4/5 and too small for this
# benchmark in general. We substitute the next valid entry (7ur7, 63 residues,
# CSV #11) to keep a set of 10 monomer targets.
_FOLDBENCH_MONOMER_FIRST_10 = [
    # skipped: ProteinTarget(pdb_id="5SBJ", assembly=1, ...) — 26 residues, too small
    ProteinTarget(pdb_id="7PV5", assembly=1, label="foldbench-7pv5"),
    ProteinTarget(pdb_id="7QP5", assembly=1, label="foldbench-7qp5"),
    ProteinTarget(pdb_id="7QSJ", assembly=1, label="foldbench-7qsj"),
    ProteinTarget(pdb_id="7T9R", assembly=1, label="foldbench-7t9r"),
    ProteinTarget(pdb_id="7TLH", assembly=1, label="foldbench-7tlh"),
    ProteinTarget(pdb_id="7TJB", assembly=1, label="foldbench-7tjb"),
    ProteinTarget(pdb_id="7UBA", assembly=1, label="foldbench-7uba"),
    ProteinTarget(pdb_id="7UK8", assembly=1, label="foldbench-7uk8"),
    ProteinTarget(pdb_id="7UR2", assembly=1, label="foldbench-7ur2"),
    ProteinTarget(pdb_id="7UR7", assembly=1, label="foldbench-7ur7"),
]

TARGETS: list[ProteinTarget] = [
    ProteinTarget(pdb_id="1QYS", label="top7"),
    ProteinTarget(pdb_id="7BNY"),
    ProteinTarget(pdb_id="1UBQ", label="ubiquitin"),
    *_FOLDBENCH_MONOMER_FIRST_10,
]

N_VALUES: list[int] = [0, 1, 2, 3, 4, 5]
DEFAULT_PAIRS_PER_TARGET: int = 1000


# ---- PDB fetch + parse ----


def _fetch_pdb_text(pdb_id: str, assembly: int = 0) -> str:
    """Fetch the PDB file from RCSB. `assembly=0` is the deposited asym unit; 1+ are biological assemblies."""
    pdb_lc = pdb_id.lower()
    pdb_uc = pdb_id.upper()
    if assembly == 0:
        urls = (
            f"https://files.rcsb.org/download/{pdb_lc}.pdb.gz",
            f"https://files.rcsb.org/download/{pdb_uc}.pdb",
        )
    else:
        urls = (
            f"https://files.rcsb.org/download/{pdb_lc}.pdb{assembly}.gz",
            f"https://files.rcsb.org/download/{pdb_uc}.pdb{assembly}",
        )
    for url in urls:
        try:
            with urlopen(url, timeout=30) as response:
                raw = response.read()
            if url.endswith(".gz"):
                raw = gzip.decompress(raw)
            return raw.decode("utf-8")
        except Exception as exc:
            logger.warning("Fetch %s failed: %s", url, exc)
    raise RuntimeError(f"Could not download PDB {pdb_id} (assembly={assembly}) from RCSB.")


@dataclass(frozen=True)
class ParsedChain:
    sequence: list[str]  # 3-letter residue names, 0-indexed position = index
    ca_coords: dict[int, np.ndarray]  # 0-indexed → CA xyz
    cb_coords: dict[int, np.ndarray]  # 0-indexed → CB (or CA for GLY / missing CB)


def _parse_chain(pdb_text: str, chain_id: str | None) -> ParsedChain:
    """Parse a single chain from the PDB text. If `chain_id is None`, use the first chain present."""
    in_model_1 = True
    selected_chain = chain_id
    sequence: list[str] = []
    res_key_to_index: dict[tuple[str, int, str], int] = {}
    ca: dict[int, np.ndarray] = {}
    cb: dict[int, np.ndarray] = {}

    for line in pdb_text.splitlines():
        if line.startswith("MODEL "):
            in_model_1 = int(line[10:14].strip() or "1") == 1
            continue
        if line.startswith("ENDMDL"):
            in_model_1 = False
            continue
        if not in_model_1 or not line.startswith("ATOM  "):
            continue

        atom_name = line[12:16].strip()
        alt_loc = line[16].strip()
        res_name = line[17:20].strip()
        chain = line[21].strip()
        res_seq = int(line[22:26].strip())
        i_code = line[26].strip()

        if selected_chain is None:
            selected_chain = chain
        if chain != selected_chain:
            continue
        if alt_loc not in ("", "A"):
            continue
        if res_name not in STANDARD_AA_3LETTER:
            continue

        res_key = (chain, res_seq, i_code)
        if res_key not in res_key_to_index:
            res_key_to_index[res_key] = len(sequence)
            sequence.append(res_name)
        idx = res_key_to_index[res_key]

        if atom_name in ("CA", "CB"):
            coord = np.array((float(line[30:38]), float(line[38:46]), float(line[46:54])), dtype=np.float32)
            (ca if atom_name == "CA" else cb)[idx] = coord

    if not sequence:
        raise RuntimeError(f"No ATOM records parsed for chain {chain_id!r} — possibly a non-standard chain label.")

    # GLY has no CB → use CA; also backfill any residue missing CB.
    cb_final: dict[int, np.ndarray] = {}
    for i, res_name in enumerate(sequence):
        cb_here = cb.get(i)
        if res_name == "GLY" or cb_here is None:
            ca_here = ca.get(i)
            if ca_here is not None:
                cb_final[i] = ca_here
        else:
            cb_final[i] = cb_here

    return ParsedChain(sequence=sequence, ca_coords=ca, cb_coords=cb_final)


# ---- Ground truth helpers ----


def _distance_bin_idx(distance_A: float) -> int:
    """Bin `distance_A` into one of 64 0.5-Å bins (right-closed). Values > 32 Å clip to the ceiling bin."""
    bin_idx = int(np.ceil(distance_A / DISTANCE_BIN_WIDTH_A)) - 1
    return max(0, min(NUM_DISTANCE_BINS - 1, bin_idx))


def _ground_truth_long_range_contacts(chain: ParsedChain) -> list[tuple[int, int]]:
    """1-indexed (i, j) pairs with |i-j| >= 24 and CB-CB ≤ 8 Å, rank-ordered by (i, j)."""
    pairs: list[tuple[int, int]] = []
    indices = sorted(chain.cb_coords)
    for ii, i in enumerate(indices):
        for j in indices[ii + 1 :]:
            if j - i < LONG_RANGE_SEQ_SEP:
                continue
            d = float(np.linalg.norm(chain.cb_coords[i] - chain.cb_coords[j]))
            if d <= CB_CONTACT_ANGSTROMS:
                pairs.append((i + 1, j + 1))
    pairs.sort()
    return pairs


# ---- Pair sampling (CA-CA only) ----


@dataclass(frozen=True)
class SampledPair:
    i: int  # 1-indexed
    j: int  # 1-indexed, > i
    distance_A: float
    bin_idx: int


def sample_ca_pairs(chain: ParsedChain, n_pairs: int, seed: int) -> list[SampledPair]:
    """Uniformly sample up to `n_pairs` unique residue pairs (i<j) with CA coords available.

    If fewer than `n_pairs` unique pairs exist, returns all of them.
    """
    rng = np.random.default_rng(seed)
    valid_residues = np.array(sorted(chain.ca_coords), dtype=np.int64)
    n_valid = len(valid_residues)
    if n_valid < 2:
        raise RuntimeError(f"Fewer than 2 residues with CA coordinates ({n_valid}).")

    max_pairs = n_valid * (n_valid - 1) // 2
    target = min(n_pairs, max_pairs)
    sampled: list[SampledPair] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = target * 50

    while len(sampled) < target and attempts < max_attempts:
        attempts += 1
        a, b = rng.choice(valid_residues, size=2, replace=False)
        i0, j0 = (int(a), int(b)) if a < b else (int(b), int(a))
        key = (i0, j0)
        if key in seen:
            continue
        seen.add(key)
        d = float(np.linalg.norm(chain.ca_coords[i0] - chain.ca_coords[j0]))
        sampled.append(SampledPair(i=i0 + 1, j=j0 + 1, distance_A=d, bin_idx=_distance_bin_idx(d)))

    if len(sampled) < target:
        logger.warning(
            "Only sampled %d / %d pairs after %d attempts (n_valid=%d, max_pairs=%d)",
            len(sampled),
            target,
            attempts,
            n_valid,
            max_pairs,
        )
    return sampled


# ---- Prompt construction + tokenization ----


def _base_prompt_tokens(sequence_3letter: list[str], seeded: list[tuple[int, int]]) -> list[str]:
    toks = ["<contacts-and-distances-v1>", "<begin_sequence>"]
    toks.extend(f"<{aa}>" for aa in sequence_3letter)
    toks.append("<begin_statements>")
    for i, j in seeded:
        toks.extend(["<long-range-contact>", f"<p{i}>", f"<p{j}>"])
    return toks


def _pair_tail_tokens(pair: SampledPair) -> list[str]:
    bin_value = (pair.bin_idx + 1) * DISTANCE_BIN_WIDTH_A
    return [
        "<distance>",
        f"<p{pair.i}>",
        f"<p{pair.j}>",
        "<CA>",
        "<CA>",
        f"<d{bin_value:.1f}>",
    ]


def _encode(tokenizer, token_strs: list[str]) -> np.ndarray:
    ids = tokenizer.encode(" ".join(token_strs), add_special_tokens=False)
    if len(ids) != len(token_strs):
        raise ValueError(
            f"Tokenizer 1:1 check failed: {len(ids)} ids for {len(token_strs)} tokens. "
            f"first 8 in: {token_strs[:8]}  first 8 out: {list(ids[:8])}"
        )
    return np.asarray(ids, dtype=np.int32)


# ---- Parquet build (ExecutorStep function) ----


@dataclass(frozen=True)
class BuildDistogramEvalDataConfig:
    pdb_id: str
    chain_id: str | None
    assembly: int
    n_prompt_contacts: int
    tokenizer: str
    n_pairs: int
    seed: int
    # Every example is right-padded with `<pad>` to exactly this many tokens so
    # it matches the training Pos axis. levanter's PrebuiltLmDataset requires
    # the parquet rows to be Pos.size exactly (it doesn't honor DatasetComponent
    # pack=True for PrebuiltLmDatasetFormat). Keeps things simple at the cost of
    # 100 eval examples per target times 8192 tokens of mostly-pad (~3 MB per parquet).
    max_seq_len: int = 8192
    # Optional sequence override (see ProteinTarget.sequence_override). When
    # set, the prompt's <begin_sequence> uses these residues; GT distances
    # still come from the PDB's CA coordinates.
    sequence_override: tuple[str, ...] | None = None
    sequence_origin: str = "pdb"
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore[arg-type]


def build_distogram_eval_data(config: BuildDistogramEvalDataConfig) -> None:
    """Generate one parquet of pre-tokenized eval examples for (target, N)."""
    logger.info("Fetching PDB %s (assembly=%d)", config.pdb_id, config.assembly)
    chain = _parse_chain(
        _fetch_pdb_text(config.pdb_id, assembly=config.assembly),
        chain_id=config.chain_id,
    )
    logger.info("Parsed %s: %d residues, %d with CA", config.pdb_id, len(chain.sequence), len(chain.ca_coords))

    if config.sequence_override is not None:
        if len(config.sequence_override) != len(chain.sequence):
            raise ValueError(
                f"sequence_override length {len(config.sequence_override)} does not match "
                f"PDB chain length {len(chain.sequence)} for {config.pdb_id} "
                f"(chain={config.chain_id or 'first'}, assembly={config.assembly})."
            )
        bad = [aa for aa in config.sequence_override if aa not in STANDARD_AA_3LETTER]
        if bad:
            raise ValueError(f"sequence_override contains non-standard amino acids: {sorted(set(bad))[:5]}...")
        # Hamming distance to native for logging context
        hamming = sum(a != b for a, b in zip(chain.sequence, config.sequence_override, strict=True))
        logger.info(
            "Using %s sequence override (origin=%s); Hamming distance to native = %d / %d",
            config.pdb_id,
            config.sequence_origin,
            hamming,
            len(chain.sequence),
        )
        sequence_for_prompt = list(config.sequence_override)
    else:
        sequence_for_prompt = list(chain.sequence)

    gt_long_range = _ground_truth_long_range_contacts(chain)
    if config.n_prompt_contacts > len(gt_long_range):
        raise ValueError(
            f"Target {config.pdb_id} (chain {config.chain_id or 'first'}, assembly={config.assembly}) "
            f"has only {len(gt_long_range)} long-range GT contacts; requested {config.n_prompt_contacts} seeded."
        )
    seeded = gt_long_range[: config.n_prompt_contacts]

    # Seed the sample on pdb_id alone so the same 1000 pairs are reused across all N.
    pdb_hash = int(hashlib.sha256(config.pdb_id.encode()).hexdigest()[:8], 16)
    seed = (pdb_hash ^ config.seed) % (2**32)
    pairs = sample_ca_pairs(chain, config.n_pairs, seed=seed)
    logger.info(
        "Sampled %d CA-CA pairs (seed=%d). Distance stats: min=%.1fÅ  median=%.1fÅ  max=%.1fÅ  "
        "ceiling-hit=%d (%.1f%%)",
        len(pairs),
        seed,
        min(p.distance_A for p in pairs),
        float(np.median([p.distance_A for p in pairs])),
        max(p.distance_A for p in pairs),
        sum(1 for p in pairs if p.bin_idx == NUM_DISTANCE_BINS - 1),
        100 * sum(1 for p in pairs if p.bin_idx == NUM_DISTANCE_BINS - 1) / len(pairs),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    base_ids = _encode(tokenizer, _base_prompt_tokens(sequence_for_prompt, seeded))
    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    input_rows: list[list[int]] = []
    weight_rows: list[list[float]] = []
    for pair in pairs:
        tail_ids = _encode(tokenizer, _pair_tail_tokens(pair))
        real = np.concatenate([base_ids, tail_ids]).astype(np.int32)
        if real.shape[0] > config.max_seq_len:
            raise ValueError(
                f"Eval example for {config.pdb_id} has {real.shape[0]} tokens, exceeds "
                f"max_seq_len={config.max_seq_len}. Shorten the sequence or raise max_seq_len."
            )
        # Right-pad tokens to max_seq_len. Loss weight is nonzero only at the
        # position whose prediction IS the GT <d_value>. levanter's convention
        # is `loss_weight[i]` weights the loss of predicting `tokens[i+1]`, so
        # to score the <d_value> prediction (and <d_value> is the last real
        # token at index N-1) we set loss_weight[N-2] = 1 — not N-1, which
        # would be silently zeroed by the causal mask (predicts tokens[N], out
        # of bounds). Padding positions get weight 0 and have no effect on the
        # loss. Causal attention means pad positions see the prompt but the
        # prompt never sees the pads — no leakage.
        if real.shape[0] < 2:
            raise ValueError(f"Eval example too short ({real.shape[0]} tokens); need >=2")
        padded = np.full(config.max_seq_len, pad_id, dtype=np.int32)
        padded[: real.shape[0]] = real
        weights = np.zeros(config.max_seq_len, dtype=np.float32)
        weights[real.shape[0] - 2] = 1.0
        input_rows.append(padded.tolist())
        weight_rows.append(weights.tolist())

    table = pa.table(
        {
            "input_ids": pa.array(input_rows, type=pa.list_(pa.int32())),
            "loss_weights": pa.array(weight_rows, type=pa.list_(pa.float32())),
        }
    )
    out_path = config.output_path.rstrip("/") + "/data.parquet"
    logger.info("Writing %d examples to %s", table.num_rows, out_path)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    buf.seek(0)
    with fsspec.open(out_path, "wb") as f:
        f.write(buf.getvalue())

    # Pre-build the levanter tree-cache so training doesn't have to. During the
    # last run, training-time cache builds were spawning sequential zephyr
    # sub-pipelines (~37 s each, 72 datasets, 45 min total) that got killed every
    # time the preemptible TPU worker was reclaimed, and the job never made it
    # past the cache-build phase in 17 hours. By building the cache inside this
    # (CPU, non-preemptible) ExecutorStep, the result persists at the step's
    # versioned output_path and any training run just `TreeCache.load()`s it.
    from levanter.data.sharded_datasource import UrlDataSource
    from levanter.data.text.cache import build_lm_dataset_cache

    # Matches the path that training will look for: LmDataConfig.build_caches()
    # appends the split name to the component's cache_dir. Our DatasetComponent
    # sets cache_dir = step/"cache", so the validation cache lives at
    # {output_path}/cache/validation.
    cache_dir = config.output_path.rstrip("/") + "/cache/validation"
    logger.info("Building levanter tree-cache at %s", cache_dir)
    build_lm_dataset_cache(
        cache_dir=cache_dir,
        source=UrlDataSource([out_path]),
        format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
        tokenizer=tokenizer,
        # Pre-tokenized examples: no BOS/EOS enforcement needed; we wrote them exactly.
        enforce_eos=False,
    )
    logger.info("Cache build complete")


# ---- ExecutorStep factory + benchmark assembly ----


def build_distogram_eval_step(
    target: ProteinTarget,
    n_prompt_contacts: int,
    *,
    tokenizer: str,
    n_pairs: int = DEFAULT_PAIRS_PER_TARGET,
) -> ExecutorStep:
    """One ExecutorStep producing the eval-data parquet for (target, N)."""
    return ExecutorStep(
        name=f"protein-dist-eval/{target.name}-N{n_prompt_contacts}",
        description=(
            f"Build pre-tokenized distogram eval examples for PDB {target.pdb_id} "
            f"(chain={target.chain_id or 'first'}, assembly={target.assembly}, "
            f"origin={target.sequence_origin}, N={n_prompt_contacts}, n_pairs={n_pairs})"
        ),
        fn=build_distogram_eval_data,
        # versioned() is required for scalar fields to factor into the step's
        # output-path hash. Plain dataclass fields are IGNORED by marin's hasher
        # (only VersionedValue + InputName contribute), so without these wraps
        # changes to n_pairs / n_prompt_contacts / max_seq_len / etc. would
        # silently reuse stale caches under the same output-path suffix.
        config=BuildDistogramEvalDataConfig(
            pdb_id=versioned(target.pdb_id),
            chain_id=versioned(target.chain_id),
            assembly=versioned(target.assembly),
            n_prompt_contacts=versioned(n_prompt_contacts),
            tokenizer=versioned(tokenizer),
            n_pairs=versioned(n_pairs),
            seed=versioned(0),
            max_seq_len=versioned(8192),
            sequence_override=versioned(target.sequence_override),
            sequence_origin=target.sequence_origin,  # metadata only — fine to leave unversioned
        ),
    )


def _tags_for(target: ProteinTarget, n: int) -> list[str]:
    """Hierarchical tags so levanter aggregates per-PDB, per-N, and (for redesigns) per-method."""
    tags = [
        "protein_dist",
        f"protein_dist/{target.name}",
        f"protein_dist/N{n}",
        f"protein_dist/{target.name}/N{n}",
    ]
    if target.sequence_origin != "pdb":
        # Extra aggregation axis across all redesigned targets of a given method.
        tags.append(f"protein_dist/origin={target.sequence_origin}")
        tags.append(f"protein_dist/origin={target.sequence_origin}/N{n}")
    return tags


def load_redesigned_targets(
    source_jsonl: str,
    *,
    method_filter: str | None = None,
    base_targets: list[ProteinTarget] | None = None,
) -> list[ProteinTarget]:
    """Read a redesigns JSONL and produce one `ProteinTarget` per record.

    Expected JSONL record shape (one JSON object per line)::

        {
            "target_label": "foldbench-7pv5",
            "pdb_id": "7PV5",
            "chain_id": null,
            "assembly": 1,
            "method": "mpnn",
            "redesign_idx": 0,
            "sequence_3letter": ["THR", "TYR", ...],
            "native_sequence_3letter": ["THR", "TYR", ...],
            "hamming_distance": 27
        }

    Args:
        source_jsonl: Path (local or gs://) to the JSONL file. If it doesn't
            exist, returns [] and logs an info message (so the training
            config doesn't break before redesigns have been generated).
        method_filter: If set, only records with this method are used.
        base_targets: Ignored; kept for API symmetry with `distogram_eval_benchmark`.

    Returns:
        A list of `ProteinTarget`s, one per redesign record.
    """
    import json as _json

    del base_targets  # unused, see note above
    # We want config-build to succeed even if (a) the redesigns file hasn't been
    # generated yet, or (b) local GCS credentials are temporarily bad — neither
    # should block the training experiment from loading.
    try:
        with fsspec.open(source_jsonl, "r") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
    except FileNotFoundError:
        logger.info("No redesigns file at %s — skipping redesigned targets.", source_jsonl)
        return []
    except Exception as exc:
        logger.warning(
            "Could not read redesigns from %s (%s: %s) — skipping redesigned targets.",
            source_jsonl,
            type(exc).__name__,
            exc,
        )
        return []

    out: list[ProteinTarget] = []
    for line in lines:
        rec = _json.loads(line)
        if method_filter is not None and rec.get("method") != method_filter:
            continue
        base_label = rec["target_label"]
        method = rec["method"]
        idx = int(rec["redesign_idx"])
        redesign_label = f"{base_label}-{method}-{idx}"
        out.append(
            ProteinTarget(
                pdb_id=rec["pdb_id"],
                chain_id=rec.get("chain_id"),
                assembly=int(rec.get("assembly", 0)),
                label=redesign_label,
                sequence_override=tuple(rec["sequence_3letter"]),
                sequence_origin=method,
            )
        )
    logger.info("Loaded %d redesigned targets from %s", len(out), source_jsonl)
    return out


@dataclass(frozen=True)
class DistogramEvalBenchmark:
    """Bundle of ExecutorSteps and DatasetComponents to wire into training."""

    build_steps: list[ExecutorStep]
    components: dict[str, DatasetComponent] = field(default_factory=dict)


def distogram_eval_benchmark(
    tokenizer: str,
    *,
    targets: list[ProteinTarget] | None = None,
    n_values: list[int] | None = None,
    n_pairs: int = DEFAULT_PAIRS_PER_TARGET,
    redesigns_source: str | None = None,
) -> DistogramEvalBenchmark:
    """Build ExecutorSteps + DatasetComponents for the full benchmark.

    Usage in a training experiment::

        from experiments.protein.protein_distogram_eval import distogram_eval_benchmark
        bench = distogram_eval_benchmark(
            PROTEIN_TOKENIZER,
            redesigns_source="gs://marin-us-east5/protein-structure/protein-mpnn-redesigns/v1/redesigns.jsonl",
        )

        protein_docs_data = dataclasses.replace(
            protein_docs_data,
            components={**protein_docs_data.components, **bench.components},
            # train_weights already excludes the new keys (no eval weight needed)
        )

        executor_main(steps=[protein_model_1b, *bench.build_steps])

    Args:
        tokenizer: HF tokenizer path.
        targets: list of base `ProteinTarget`s. Defaults to `TARGETS`.
        n_values: list of seeded-contact counts. Defaults to `N_VALUES`.
        n_pairs: number of sampled residue pairs per (target, N).
        redesigns_source: optional JSONL path produced by
            `experiments/protein/redesign_sequences.py`. When set, one extra
            `ProteinTarget` per redesign record is added to the benchmark. If
            the path does not exist, redesigns are silently skipped (so
            configs can reference it before the redesign script has been run).
    """
    targets = targets or TARGETS
    if redesigns_source:
        targets = list(targets) + load_redesigned_targets(redesigns_source)
    n_values = n_values or N_VALUES

    build_steps: list[ExecutorStep] = []
    components: dict[str, DatasetComponent] = {}
    for target in targets:
        for n in n_values:
            step = build_distogram_eval_step(target, n, tokenizer=tokenizer, n_pairs=n_pairs)
            build_steps.append(step)
            component_name = f"protein_dist_{target.name}_N{n}"
            # `step / "..."` creates an InputName that marin's executor resolves
            # to the concrete GCS path once the step has an output_path. We put
            # the levanter tree-cache in a sibling subdir next to the parquet
            # so each (target, N) gets its own cache and there's no sharing.
            parquet_ref = step / "data.parquet"
            cache_dir_ref = step / "cache"
            components[component_name] = DatasetComponent(
                source=UrlDatasetSourceConfig(
                    validation_urls=[parquet_ref],  # type: ignore[list-item]
                    format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
                    tags=_tags_for(target, n),
                ),
                cache_dir=cache_dir_ref,  # type: ignore[arg-type]
                format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
                tags=_tags_for(target, n),
            )
    return DistogramEvalBenchmark(build_steps=build_steps, components=components)
