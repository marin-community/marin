# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predict a CB-CB distogram for a PDB entry, one residue pair at a time.

For each value of N in `--prompt-contact-counts` we construct a base prompt
containing the document header, the amino-acid sequence, `<begin_statements>`,
and the first N ground-truth long-range contacts (rank-ordered by position).
Then for every ordered pair (i, j), i ≠ j, we append

    <distance> <p_i> <p_j> <atom_i> <atom_j>

and query the model for *one* next token with top-K logprobs. We renormalize
the logprobs over the 64 distance-bin tokens (`<d0.5>`..`<d32.0>`) to get a
per-pair distribution, which can be plotted as a distogram and compared to
ground truth CB-CB distances (CA for GLY, same convention used at training).

Output per N value:
  - `distogram_n{N}.npz`: probs (n, n, 64) fp32, plus metadata
  - Combined `summary.json` with scalar metrics per N.

Usage::

    HF_TOKEN=... uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu -- \\
        python -m experiments.protein.eval_protein_distogram \\
            --model gs://marin-us-east5/checkpoints/protein-contacts-1b-2.5e-4-780930/hf \\
            --pdb-id 1QYS \\
            --prompt-contact-counts 0 1 2 3 4 5 \\
            --output-dir gs://marin-us-east5/eval/protein-distogram/1qys/run-01
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from urllib.request import urlopen

import fsspec
import numpy as np
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

# ---- Format constants ----

DISTANCE_BIN_WIDTH_A = 0.5
NUM_DISTANCE_BINS = 64  # <d0.5> .. <d32.0>
# Bin k (0-indexed) corresponds to token <d{(k+1)*0.5:.1f}>. We treat each bin
# as covering ((k*0.5), ((k+1)*0.5)], consistent with tokenize_by_half_angstrom.
DISTANCE_BIN_EDGES = (np.arange(1, NUM_DISTANCE_BINS + 1) * DISTANCE_BIN_WIDTH_A).astype(np.float32)
DISTANCE_BIN_MIDPOINTS = (DISTANCE_BIN_EDGES - DISTANCE_BIN_WIDTH_A / 2).astype(np.float32)
DISTANCE_MAX_A = float(DISTANCE_BIN_EDGES[-1])  # 32.0

CONTACT_TYPES_IN_ORDER = [
    ("<long-range-contact>", 24, float("inf")),
    ("<medium-range-contact>", 12, 24),
    ("<short-range-contact>", 6, 12),
]
CB_CONTACT_ANGSTROMS = 8.0

STANDARD_AA_3LETTER = {
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


# ---- PDB download + parsing ----


@dataclass
class ParsedStructure:
    sequence: list[str]  # 3-letter, 0-indexed → position = index+1
    cb_coords: dict[int, np.ndarray]  # 0-indexed → (CB, or CA for GLY)
    atom_per_residue: list[str]  # "CB" for all residues except GLY ("CA")


def _fetch_pdb_text(pdb_id: str) -> str:
    pdb_id = pdb_id.lower()
    for url in (
        f"https://files.rcsb.org/download/{pdb_id}.pdb.gz",
        f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
    ):
        try:
            with urlopen(url, timeout=30) as response:
                raw = response.read()
            if url.endswith(".gz"):
                raw = gzip.decompress(raw)
            return raw.decode("utf-8")
        except Exception as exc:
            logger.warning("Fetch %s failed: %s", url, exc)
    raise RuntimeError(f"Could not download PDB {pdb_id} from RCSB.")


def parse_pdb(pdb_text: str, chain_id: str | None = None) -> ParsedStructure:
    in_model_1 = True
    selected_chain: str | None = chain_id
    sequence: list[str] = []
    res_key_to_index: dict[tuple[str, int, str], int] = {}
    cb: dict[int, np.ndarray] = {}
    ca: dict[int, np.ndarray] = {}

    for line in pdb_text.splitlines():
        if line.startswith("MODEL "):
            model_num = int(line[10:14].strip() or "1")
            in_model_1 = model_num == 1
            continue
        if line.startswith("ENDMDL"):
            in_model_1 = False
            continue
        if not in_model_1:
            continue
        if not line.startswith("ATOM  "):
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
        res_idx = res_key_to_index[res_key]

        coord = np.array((float(line[30:38]), float(line[38:46]), float(line[46:54])), dtype=np.float32)
        if atom_name == "CB":
            cb[res_idx] = coord
        elif atom_name == "CA":
            ca[res_idx] = coord

    if not sequence:
        raise RuntimeError("No ATOM records parsed — is chain_id correct?")

    atom_per_residue: list[str] = []
    cb_coords: dict[int, np.ndarray] = {}
    for i, res_name in enumerate(sequence):
        if res_name == "GLY" or i not in cb:
            if i in ca:
                cb_coords[i] = ca[i]
                atom_per_residue.append("CA")
            else:
                logger.warning("Residue %d (%s) missing both CB and CA; excluding.", i + 1, res_name)
                atom_per_residue.append("CA")  # placeholder; pair will be skipped via GT missing
        else:
            cb_coords[i] = cb[i]
            atom_per_residue.append("CB")

    return ParsedStructure(sequence=sequence, cb_coords=cb_coords, atom_per_residue=atom_per_residue)


# ---- Ground truth ----


def gt_distance_matrix(structure: ParsedStructure) -> np.ndarray:
    """(n, n) CB-CB distances (CA for GLY). NaN where either residue lacks coords."""
    n = len(structure.sequence)
    out = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        if i not in structure.cb_coords:
            continue
        for j in range(n):
            if j not in structure.cb_coords:
                continue
            if i == j:
                out[i, j] = 0.0
                continue
            out[i, j] = float(np.linalg.norm(structure.cb_coords[i] - structure.cb_coords[j]))
    return out


def ground_truth_contacts_by_type(
    structure: ParsedStructure,
) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {t: [] for t, _, _ in CONTACT_TYPES_IN_ORDER}
    indices = sorted(structure.cb_coords)
    for ii, i in enumerate(indices):
        for j in indices[ii + 1 :]:
            sep = j - i
            if sep < 6:
                continue
            d = float(np.linalg.norm(structure.cb_coords[i] - structure.cb_coords[j]))
            if d > CB_CONTACT_ANGSTROMS:
                continue
            for type_tok, lo, hi in CONTACT_TYPES_IN_ORDER:
                if lo <= sep < hi:
                    result[type_tok].append((i + 1, j + 1))
                    break
    # Rank order by (i, j) to match training convention.
    for k in result:
        result[k].sort()
    return result


# ---- Prompt construction ----


def build_base_prompt_tokens(
    sequence_3letter: list[str],
    seeded_contacts: list[tuple[str, int, int]],
) -> list[str]:
    """Return the base prompt as a list of token strings (ready to join with spaces).

    seeded_contacts: list of (type_token, pos_i, pos_j) with 1-indexed positions.
    """
    toks: list[str] = ["<contacts-and-distances-v1>", "<begin_sequence>"]
    toks.extend(f"<{aa}>" for aa in sequence_3letter)
    toks.append("<begin_statements>")
    for type_tok, i, j in seeded_contacts:
        toks.extend([type_tok, f"<p{i}>", f"<p{j}>"])
    return toks


def pair_tail_tokens(i: int, j: int, atom_i: str, atom_j: str) -> list[str]:
    return ["<distance>", f"<p{i}>", f"<p{j}>", f"<{atom_i}>", f"<{atom_j}>"]


# ---- Distogram inference ----


def _distance_token_ids(tokenizer) -> list[int]:
    """Return the 64 distance-bin token IDs in bin order (k=0 → <d0.5>, ...)."""
    ids: list[int] = []
    for k in range(NUM_DISTANCE_BINS):
        tok = f"<d{(k + 1) * DISTANCE_BIN_WIDTH_A:.1f}>"
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"Unexpected encoding for {tok}: {enc!r}")
        ids.append(int(enc[0]))
    return ids


def _encode_tokens(tokenizer, token_strs: list[str]) -> list[int]:
    """Encode a whitespace-joined list of tokens to IDs (WordLevel tokenizer)."""
    ids = tokenizer.encode(" ".join(token_strs), add_special_tokens=False)
    if len(ids) != len(token_strs):
        # Surface trouble early (any <UNK> fallback would be catastrophic here).
        raise ValueError(
            f"Tokenizer did not produce 1:1 mapping: {len(token_strs)} in → {len(ids)} out.\n"
            f"first 10 in: {token_strs[:10]}\nfirst 10 out: {ids[:10]}"
        )
    return [int(x) for x in ids]


def run_distogram(
    llm,
    tokenizer,
    base_prompt_tokens: list[str],
    atom_per_residue: list[str],
    seq_len: int,
    *,
    top_k_logprobs: int,
    batch_size: int,
    canonical_order: bool = True,
) -> tuple[np.ndarray, dict]:
    """Compute (n, n, 64) probability tensor and coverage stats.

    Returns (probs, stats). probs[i-1, j-1, k] = P(bin k | pair (i, j)).
    Self-pair rows (i==j) are zeros.

    When `canonical_order=True` we only query pairs with i < j (matching the
    training-data invariant that 100% of `<distance>` statements have i < j),
    and mirror the result to fill probs[j-1, i-1] = probs[i-1, j-1]. This
    halves inference cost and makes the distogram symmetric by construction.
    When False, every ordered pair is queried independently, exposing the
    model's (physically nonsensical) order asymmetry.
    """
    from vllm import SamplingParams, TokensPrompt

    base_ids = _encode_tokens(tokenizer, base_prompt_tokens)
    distance_ids = _distance_token_ids(tokenizer)
    bin_of: dict[int, int] = {tok_id: k for k, tok_id in enumerate(distance_ids)}
    distance_id_set = set(distance_ids)

    pair_prompts: list = []
    pair_keys: list[tuple[int, int]] = []  # 1-indexed positions
    for i in range(1, seq_len + 1):
        atom_i = atom_per_residue[i - 1]
        for j in range(1, seq_len + 1):
            if i == j:
                continue
            if canonical_order and i >= j:
                continue
            atom_j = atom_per_residue[j - 1]
            tail_ids = _encode_tokens(tokenizer, pair_tail_tokens(i, j, atom_i, atom_j))
            pair_prompts.append(TokensPrompt(prompt_token_ids=base_ids + tail_ids))
            pair_keys.append((i, j))

    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        logprobs=top_k_logprobs,
        n=1,
    )

    probs = np.zeros((seq_len, seq_len, NUM_DISTANCE_BINS), dtype=np.float32)
    missing_bins_per_pair: list[int] = []
    non_distance_top_mass_per_pair: list[float] = []

    t0 = time.time()
    for chunk_start in range(0, len(pair_prompts), batch_size):
        chunk_prompts = pair_prompts[chunk_start : chunk_start + batch_size]
        chunk_keys = pair_keys[chunk_start : chunk_start + batch_size]

        outputs = llm.generate(chunk_prompts, sampling, use_tqdm=False)

        for (i, j), out in zip(chunk_keys, outputs, strict=True):
            # out.outputs[0].logprobs is a list of len == num_generated_tokens (1 here).
            # Each element is dict[token_id -> Logprob(logprob=..., rank=..., decoded_token=...)].
            lp_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            row = np.zeros(NUM_DISTANCE_BINS, dtype=np.float32)
            non_distance_mass = 0.0
            for tok_id, lp in lp_dict.items():
                tok_id = int(tok_id)
                logprob = float(lp.logprob)
                if tok_id in distance_id_set:
                    row[bin_of[tok_id]] = np.exp(logprob)
                else:
                    non_distance_mass += float(np.exp(logprob))
            total = float(row.sum())
            missing = NUM_DISTANCE_BINS - int((row > 0).sum())
            missing_bins_per_pair.append(missing)
            non_distance_top_mass_per_pair.append(non_distance_mass)
            if total > 0:
                row /= total
            probs[i - 1, j - 1] = row
            if canonical_order:
                probs[j - 1, i - 1] = row

        done = chunk_start + len(chunk_prompts)
        elapsed = time.time() - t0
        logger.info(
            "  pairs %d/%d (%.1f%%) — %.1fs elapsed (%.1fms/pair)",
            done,
            len(pair_prompts),
            100 * done / len(pair_prompts),
            elapsed,
            1000 * elapsed / done,
        )

    stats = {
        "num_pairs": len(pair_prompts),
        "canonical_order": canonical_order,
        "missing_bins_median": float(np.median(missing_bins_per_pair)),
        "missing_bins_max": int(np.max(missing_bins_per_pair)),
        "missing_bins_mean": float(np.mean(missing_bins_per_pair)),
        "non_distance_top_mass_median": float(np.median(non_distance_top_mass_per_pair)),
        "non_distance_top_mass_max": float(np.max(non_distance_top_mass_per_pair)),
    }
    return probs, stats


# ---- Metrics ----


def summarize(probs: np.ndarray, gt: np.ndarray) -> dict:
    """Per-N scalar metrics. Considers only pairs where GT is finite and <= 32 Å."""
    n = probs.shape[0]
    expected = (probs * DISTANCE_BIN_MIDPOINTS[None, None, :]).sum(axis=-1)  # (n, n)
    argmax_bin = probs.argmax(axis=-1)  # (n, n)
    argmax_distance = DISTANCE_BIN_MIDPOINTS[argmax_bin]  # (n, n)

    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    valid = (ii != jj) & np.isfinite(gt) & (gt <= DISTANCE_MAX_A)

    # Per-pair errors
    expected_err = expected - gt
    argmax_err = argmax_distance - gt

    # Contact accuracy: GT < 8A vs. predicted P(d <= 8A)
    gt_contact = (gt < CB_CONTACT_ANGSTROMS) & valid
    # P(d <= 8) = sum of bins covering (0, 8] = bins 0..15 (edges 0.5..8.0)
    p_contact_8a = probs[:, :, :16].sum(axis=-1)

    def _masked_abs_mean(arr):
        return float(np.abs(arr[valid]).mean())

    def _masked_signed_mean(arr):
        return float(arr[valid].mean())

    # Symmetry: the model is given ordered pairs, but distance is symmetric.
    # Compare the two orderings for each unordered pair.
    asym = np.abs(expected - expected.T)

    return {
        "num_valid_pairs": int(valid.sum()),
        "expected_mean_abs_err_A": _masked_abs_mean(expected_err),
        "expected_mean_signed_err_A": _masked_signed_mean(expected_err),
        "argmax_mean_abs_err_A": _masked_abs_mean(argmax_err),
        "expected_order_asymmetry_mean_A": float(asym[valid].mean()),
        "expected_order_asymmetry_max_A": float(asym[valid].max()),
        "contact_threshold_A": CB_CONTACT_ANGSTROMS,
        "contact_prob_auc_proxy_corr": float(
            np.corrcoef(p_contact_8a[valid].ravel(), gt_contact[valid].astype(np.float32).ravel())[0, 1]
        ),
    }


# ---- Model staging ----


def _looks_like_levanter_checkpoint(model_path: str) -> bool:
    """Detect a Levanter checkpoint dir.

    Levanter dirs contain a ``metadata.json`` (and ``state/``) at the
    checkpoint root, or live as ``step-N`` subdirs of a parent ``checkpoints/``
    dir. HF dirs always have ``config.json`` at the top level.

    Returns True only when the path doesn't look HF-shaped.
    """
    fs, root = url_to_fs(model_path.rstrip("/"))
    if not fs.exists(root):
        return False
    if fs.exists(f"{root}/config.json"):
        return False
    if fs.exists(f"{root}/metadata.json"):
        return True
    # Parent of step-N subdirs (`checkpoints/`-style) — `discover_latest` will
    # pick whichever is loadable.
    try:
        children = fs.ls(root, detail=False)
    except (FileNotFoundError, NotADirectoryError):
        return False
    for child in children:
        name = child.rstrip("/").rsplit("/", 1)[-1]
        if name.startswith("step-") and fs.exists(f"{child}/metadata.json"):
            return True
    return False


def _resolve_levanter_model_spec(spec: str):
    """Import ``module.path.attribute`` and return the attribute (an LmConfig)."""
    import importlib

    if "." not in spec:
        raise ValueError(f"--levanter-model-spec must be 'module.attribute'; got {spec!r}")
    mod_name, attr_name = spec.rsplit(".", 1)
    return getattr(importlib.import_module(mod_name), attr_name)


def stage_levanter_checkpoint_locally(
    model_path: str,
    *,
    levanter_model_spec: str,
    tokenizer: str,
) -> str:
    """Convert a Levanter checkpoint to a local HF dir for vLLM.

    Uses ``discover_latest_checkpoint`` to resolve a ``checkpoints/`` parent
    dir to the most recent loadable ``step-N``. Conversion runs on CPU to
    avoid contending with the TPU that vLLM is about to claim.

    Result is cached under
    ``/tmp/marin-protein-eval-levanter/<sha-of-model_path>/`` so re-running
    the eval skips the conversion.
    """
    from levanter.checkpoint import discover_latest_checkpoint
    from levanter.main.export_lm_to_hf import ConvertLmConfig
    from levanter.main.export_lm_to_hf import main as export_lm_to_hf_main
    from levanter.trainer import TrainerConfig

    cache_key = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:16]
    local_dir = os.path.join(tempfile.gettempdir(), "marin-protein-eval-levanter", cache_key)
    if os.path.exists(os.path.join(local_dir, "config.json")):
        logger.info("Reusing cached Levanter→HF conversion at %s", local_dir)
        return local_dir
    os.makedirs(local_dir, exist_ok=True)

    discovered = discover_latest_checkpoint(model_path)
    if discovered is None:
        raise FileNotFoundError(f"No Levanter checkpoint found under {model_path!r}")
    if discovered != model_path:
        logger.info("discover_latest_checkpoint resolved %s → %s", model_path, discovered)

    model_config = _resolve_levanter_model_spec(levanter_model_spec)

    logger.info("Converting Levanter checkpoint %s → local HF at %s (CPU)", discovered, local_dir)
    t0 = time.time()
    export_lm_to_hf_main(
        ConvertLmConfig(
            trainer=TrainerConfig(),
            checkpoint_path=discovered,
            output_dir=local_dir,
            model=model_config,
            tokenizer=tokenizer,
            save_tokenizer=True,
            use_cpu=True,
        )
    )
    logger.info("Levanter→HF conversion took %.1fs", time.time() - t0)

    if not os.path.exists(os.path.join(local_dir, "config.json")):
        raise RuntimeError(f"Conversion produced no config.json at {local_dir}")
    return local_dir


def stage_model_locally(
    model_path: str,
    *,
    levanter_model_spec: str | None = None,
    tokenizer: str | None = None,
) -> str:
    """Make ``model_path`` available locally as an HF checkpoint dir.

    * Local paths: returned unchanged.
    * gs://... HF dirs: mirrored to ``/tmp/marin-protein-eval-model/<sha>/``.
    * gs://... Levanter checkpoint dirs: converted in-place via
      ``levanter.main.export_lm_to_hf`` to ``/tmp/marin-protein-eval-levanter/<sha>/``.

    Auto-detect picks the right path based on the directory contents. Pass
    ``levanter_model_spec`` (an importable ``module.attribute`` for an
    ``LmConfig``) when forcing the Levanter path or when auto-detect picks
    Levanter — the conversion needs the model's architecture spec since
    Levanter checkpoints don't carry one. ``tokenizer`` defaults to the
    model spec's HF tokenizer; pass explicitly to override.
    """
    if not model_path.startswith(("gs://", "s3://")):
        return model_path

    if _looks_like_levanter_checkpoint(model_path):
        if levanter_model_spec is None:
            raise ValueError(f"{model_path} looks like a Levanter checkpoint but --levanter-model-spec was not set.")
        if tokenizer is None:
            raise ValueError("Levanter checkpoint requires an explicit --tokenizer.")
        return stage_levanter_checkpoint_locally(
            model_path,
            levanter_model_spec=levanter_model_spec,
            tokenizer=tokenizer,
        )

    fs, remote_root = url_to_fs(model_path.rstrip("/"))
    cache_key = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:16]
    local_dir = os.path.join(tempfile.gettempdir(), "marin-protein-eval-model", cache_key)
    os.makedirs(local_dir, exist_ok=True)

    found = fs.find(remote_root, detail=True, maxdepth=1)
    entries = list(found.values()) if isinstance(found, dict) else [fs.info(p) for p in found]
    logger.info("Staging %d entries from %s to %s", len(entries), model_path, local_dir)
    for entry in entries:
        if entry.get("type") != "file":
            continue
        name = os.path.basename(entry["name"].rstrip("/"))
        if not name:
            continue
        local_path = os.path.join(local_dir, name)
        remote_size = entry.get("size")
        if os.path.exists(local_path) and remote_size is not None and os.path.getsize(local_path) == remote_size:
            continue
        logger.info("  download %s (%s bytes)", name, remote_size)
        t0 = time.time()
        fs.get(entry["name"], local_path)
        logger.info("  done in %.1fs", time.time() - t0)

    if not os.path.exists(os.path.join(local_dir, "config.json")):
        raise FileNotFoundError(f"No config.json in {local_dir} (source: {model_path})")
    return local_dir


# ---- IO ----


def _write_json(path: str, obj: dict) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_npz(path: str, **arrays) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    with fsspec.open(path, "wb") as f:
        f.write(buf.getvalue())


# ---- Main ----


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pdb-id", default="1QYS")
    parser.add_argument("--chain-id", default=None)
    parser.add_argument(
        "--prompt-contact-counts",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Run one distogram per N in this list (N = number of seeded GT long-range contacts in the prompt).",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512, help="Prompts per vLLM generate() call.")
    parser.add_argument(
        "--top-k-logprobs",
        type=int,
        default=128,
        help=(
            "Top-K logprobs requested from vLLM. Must cover the 64 distance bins; "
            "missing bins are reported in the summary."
        ),
    )
    parser.add_argument(
        "--canonical-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set (default), only query pairs with i < j and mirror the "
            "probability distribution to (j, i). Matches the training-data "
            "invariant that 100%% of <distance> statements have i < j. "
            "Pass --no-canonical-order to query every ordered pair independently."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sequence-override-source",
        default=None,
        help=(
            "Optional JSONL path (local or gs://) to a redesigns file produced by "
            "`experiments.protein.redesign_sequences`. When set together with "
            "--sequence-override-target-label, --sequence-override-method, and "
            "--sequence-override-idx, the <begin_sequence> section of the prompt "
            "uses the matching redesigned sequence instead of the native PDB "
            "sequence. Ground-truth CA coordinates still come from the PDB."
        ),
    )
    parser.add_argument(
        "--sequence-override-target-label",
        default=None,
        help="target_label to match in the redesigns JSONL (e.g. 'top7').",
    )
    parser.add_argument(
        "--sequence-override-method", default=None, help="method field to match (e.g. 'soluble' or 'mpnn')."
    )
    parser.add_argument("--sequence-override-idx", type=int, default=None, help="redesign_idx to match.")
    parser.add_argument(
        "--levanter-model-spec",
        default=None,
        help=(
            "Importable ``module.attribute`` for an LmConfig (e.g. "
            "``experiments.protein.train_protein_30m_distance_masked.protein_llama_30m``). "
            "Required when --model points at a Levanter checkpoint dir; the "
            "conversion needs the architecture spec since Levanter checkpoints "
            "don't carry one. Ignored when --model is an HF dir."
        ),
    )
    parser.add_argument(
        "--levanter-tokenizer",
        default="timodonnell/protein-docs-tokenizer",
        help="HF tokenizer id used during Levanter→HF conversion.",
    )
    args = parser.parse_args(argv)

    override_fields = (
        args.sequence_override_source,
        args.sequence_override_target_label,
        args.sequence_override_method,
        args.sequence_override_idx,
    )
    if any(x is not None for x in override_fields) and not all(x is not None for x in override_fields):
        parser.error("--sequence-override-{source,target-label,method,idx} must all be set together.")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # --- 1. PDB + ground truth ---
    logger.info("Fetching PDB %s", args.pdb_id)
    structure = parse_pdb(_fetch_pdb_text(args.pdb_id), chain_id=args.chain_id)
    seq_len = len(structure.sequence)
    logger.info(
        "Parsed %s: %d residues; atoms %s",
        args.pdb_id,
        seq_len,
        {"CB": structure.atom_per_residue.count("CB"), "CA": structure.atom_per_residue.count("CA")},
    )

    gt_dist = gt_distance_matrix(structure)
    gt_contacts = ground_truth_contacts_by_type(structure)
    n_long = len(gt_contacts["<long-range-contact>"])

    # --- 1b. Optional sequence override (for MPNN / SolubleMPNN redesigns) ---
    sequence_for_prompt = list(structure.sequence)
    override_info: dict | None = None
    if args.sequence_override_source is not None:
        import json as _json

        with fsspec.open(args.sequence_override_source, "r") as f:
            lines = [_json.loads(l) for l in f.read().splitlines() if l.strip()]
        matches = [
            r
            for r in lines
            if r.get("target_label") == args.sequence_override_target_label
            and r.get("method") == args.sequence_override_method
            and int(r.get("redesign_idx", -1)) == args.sequence_override_idx
        ]
        if not matches:
            raise ValueError(
                f"No redesigns record matched target_label={args.sequence_override_target_label!r} "
                f"method={args.sequence_override_method!r} redesign_idx={args.sequence_override_idx} "
                f"in {args.sequence_override_source}"
            )
        if len(matches) > 1:
            raise ValueError(f"Ambiguous match ({len(matches)} records) for override.")
        rec = matches[0]
        redesigned = list(rec["sequence_3letter"])
        if len(redesigned) != seq_len:
            raise ValueError(f"Override sequence length {len(redesigned)} does not match PDB chain length {seq_len}.")
        hamming = sum(a != b for a, b in zip(structure.sequence, redesigned, strict=True))
        logger.info(
            "Using sequence override (%s/%s #%d): Hamming distance to native = %d/%d",
            args.sequence_override_target_label,
            args.sequence_override_method,
            args.sequence_override_idx,
            hamming,
            seq_len,
        )
        sequence_for_prompt = redesigned
        override_info = {
            "source": args.sequence_override_source,
            "target_label": rec["target_label"],
            "method": rec["method"],
            "redesign_idx": rec["redesign_idx"],
            "hamming_distance": hamming,
            "mpnn_score": rec.get("mpnn_score"),
        }
    max_n = max(args.prompt_contact_counts)
    if max_n > n_long:
        raise ValueError(
            f"--prompt-contact-counts has N={max_n} but only {n_long} long-range GT contacts are available."
        )

    # --- 2. Stage model + load vLLM once ---
    from vllm import LLM

    local_model_path = stage_model_locally(
        args.model,
        levanter_model_spec=args.levanter_model_spec,
        tokenizer=args.levanter_tokenizer,
    )
    logger.info("Loading model %s via vLLM", args.model)
    llm = LLM(
        model=local_model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        trust_remote_code=True,
        max_logprobs=max(args.top_k_logprobs, 128),
    )
    tokenizer = llm.get_tokenizer()

    # --- 3. For each N, build prompt + run distogram ---
    output_dir = args.output_dir.rstrip("/")
    per_n_summary = []
    t_total = time.time()

    for n_prompt in args.prompt_contact_counts:
        seeded = [("<long-range-contact>", i, j) for (i, j) in gt_contacts["<long-range-contact>"][:n_prompt]]
        base_tokens = build_base_prompt_tokens(sequence_for_prompt, seeded)
        logger.info(
            "--- N=%d (seeded %d long-range contacts) | base prompt = %d tokens ---",
            n_prompt,
            len(seeded),
            len(base_tokens),
        )
        t0 = time.time()
        probs, stats = run_distogram(
            llm,
            tokenizer,
            base_prompt_tokens=base_tokens,
            atom_per_residue=structure.atom_per_residue,
            seq_len=seq_len,
            top_k_logprobs=args.top_k_logprobs,
            batch_size=args.batch_size,
            canonical_order=args.canonical_order,
        )
        elapsed = time.time() - t0
        logger.info("N=%d distogram took %.1fs", n_prompt, elapsed)

        metrics = summarize(probs, gt_dist)
        logger.info(
            "  metrics: E[|d_pred - d_gt|]=%.2fA (expected), %.2fA (argmax); "
            "non-distance top-K mass median=%.2e; missing bins median=%.0f",
            metrics["expected_mean_abs_err_A"],
            metrics["argmax_mean_abs_err_A"],
            stats["non_distance_top_mass_median"],
            stats["missing_bins_median"],
        )

        _write_npz(
            f"{output_dir}/distogram_n{n_prompt}.npz",
            probs=probs,
            gt_distance=gt_dist,
            seeded_contacts=np.array(seeded if seeded else [], dtype=object),
            bin_midpoints=DISTANCE_BIN_MIDPOINTS,
            bin_edges=DISTANCE_BIN_EDGES,
            atom_per_residue=np.array(structure.atom_per_residue),
            sequence_3letter=np.array(structure.sequence),  # native
            sequence_3letter_used_in_prompt=np.array(sequence_for_prompt),
        )
        per_n_summary.append(
            {
                "n_prompt_contacts": n_prompt,
                "seeded_contacts": seeded,
                "elapsed_seconds": elapsed,
                "coverage_stats": stats,
                "metrics": metrics,
            }
        )

    total_elapsed = time.time() - t_total
    summary = {
        "pdb_id": args.pdb_id.upper(),
        "chain_id": args.chain_id,
        "sequence_length": seq_len,
        "sequence_3letter": structure.sequence,  # native PDB sequence
        "sequence_3letter_used_in_prompt": sequence_for_prompt,
        "sequence_override": override_info,
        "atom_per_residue": structure.atom_per_residue,
        "inference": {
            "model": args.model,
            "top_k_logprobs": args.top_k_logprobs,
            "batch_size": args.batch_size,
            "canonical_order": args.canonical_order,
            "total_elapsed_seconds": total_elapsed,
        },
        "ground_truth_counts": {t: len(v) for t, v in gt_contacts.items()},
        "per_n": per_n_summary,
    }
    _write_json(f"{output_dir}/summary.json", summary)
    logger.info("Wrote results to %s (total %.1fs)", output_dir, total_elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
