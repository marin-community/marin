# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the protein-docs-1b-v3 model on Top7 (PDB 1QYS) contact prediction.

The model's tokenizer (``WillHeld/contactdoc-tokenizer``) only includes tokens
for the ``deterministic-positives-only`` document format::

    <deterministic-positives-only>
    <begin_sequence>
    <MET> <LYS> <PHE> ...
    <begin_contacts>
    <p1> <p8> <SD> <CD1>
    <p1> <p7> <CG> <CA>
    <p2> <p8> <NZ> <O>
    ...
    <end_contacts>
    <end>

Each contact is a 4-token group ``<p_i> <p_j> <atom_i> <atom_j>``. Contacts are
sorted by decreasing sequence separation (training-time convention). All
contacts are true positives (closest heavy-atom pair per residue pair within
``--contact-cutoff`` Å); there are no distance bins or correction tokens.

Ground truth (matching the `LlamaFold-experiments` exp5 Top7 analysis):
  - one contact per residue pair, closest heavy-atom pair (both atoms valid for
    the residue's amino acid), within ``--contact-cutoff`` (default 4.0 Å),
  - adjacent residues (|i-j| < 2) excluded.

Residue-pair precision/recall is computed against this ground truth. Consensus
predictions are residue pairs appearing in >= ``consensus_threshold`` of the
sampled rollouts.

Usage (launched via iris on a v5p-8)::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu -- \\
        python -m experiments.protein.eval_protein_top7_v3 \\
            --model gs://marin-us-central1/checkpoints/protein-docs-1b-v3-7e87f7/hf \\
            --pdb-id 1QYS \\
            --num-rollouts 10 \\
            --tensor-parallel-size 2 \\
            --output-dir gs://marin-us-central1/eval/protein-top7-v3/1qys/run-04
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from urllib.request import urlopen

import fsspec
import numpy as np
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

# ---- Format constants (must match LlamaFold-experiments/exp5/src/data.py) ----

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
NONSTANDARD_AA_MAP = {
    "MSE": "MET",
    "CSE": "CYS",
    "SEC": "CYS",
    "HYP": "PRO",
    "TPO": "THR",
    "SEP": "SER",
    "PTR": "TYR",
}
_BACKBONE = {"N", "CA", "C", "O", "OXT"}
VALID_ATOMS: dict[str, set[str]] = {
    "ALA": _BACKBONE | {"CB"},
    "ARG": _BACKBONE | {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": _BACKBONE | {"CB", "CG", "OD1", "ND2"},
    "ASP": _BACKBONE | {"CB", "CG", "OD1", "OD2"},
    "CYS": _BACKBONE | {"CB", "SG"},
    "GLN": _BACKBONE | {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": _BACKBONE | {"CB", "CG", "CD", "OE1", "OE2"},
    "GLY": _BACKBONE,
    "HIS": _BACKBONE | {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": _BACKBONE | {"CB", "CG1", "CG2", "CD1"},
    "LEU": _BACKBONE | {"CB", "CG", "CD1", "CD2"},
    "LYS": _BACKBONE | {"CB", "CG", "CD", "CE", "NZ"},
    "MET": _BACKBONE | {"CB", "CG", "SD", "CE"},
    "PHE": _BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": _BACKBONE | {"CB", "CG", "CD"},
    "SER": _BACKBONE | {"CB", "OG"},
    "THR": _BACKBONE | {"CB", "OG1", "CG2"},
    "TRP": _BACKBONE | {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": _BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": _BACKBONE | {"CB", "CG1", "CG2"},
}

ATOM_NAMES = [
    "C",
    "CA",
    "CB",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CG",
    "CG1",
    "CG2",
    "CH2",
    "CZ",
    "CZ2",
    "CZ3",
    "N",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "O",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "SD",
    "SG",
    "OXT",
]
_ATOM_TOKEN_SET = {f"<{a}>" for a in ATOM_NAMES}
_POS_PATTERN = re.compile(r"^<p(\d+)>$")
_END_MARKERS = {"<end_contacts>", "<end>", "<eos>", "<pad>", "<end_of_document>"}

# Contact tuple: (pos1, pos2, atom1, atom2)
Contact = tuple[int, int, str, str]


# ---- PDB parsing ----


@dataclass
class ParsedStructure:
    sequence: list[str]  # 3-letter residue names in position order
    # For each (res_idx 0-based, atom_name): coord
    atom_coords: list[dict[str, np.ndarray]]


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
    """Parse heavy-atom ATOM records for model 1 / a single chain.

    Non-standard residues are remapped via NONSTANDARD_AA_MAP (e.g. MSE->MET).
    """
    in_model_1 = True
    selected_chain: str | None = chain_id
    sequence: list[str] = []
    res_key_to_index: dict[tuple[str, int, str], int] = {}
    atom_coords: list[dict[str, np.ndarray]] = []

    for line in pdb_text.splitlines():
        if line.startswith("MODEL "):
            try:
                model_num = int(line[10:14].strip() or "1")
            except ValueError:
                model_num = 1
            in_model_1 = model_num == 1
            continue
        if line.startswith("ENDMDL"):
            in_model_1 = False
            continue
        if not in_model_1:
            continue
        # Accept ATOM records and HETATM records for amino acids (e.g. MSE/SEP/SEC).
        record = line[:6]
        if record not in ("ATOM  ", "HETATM"):
            continue

        atom_name = line[12:16].strip()
        alt_loc = line[16].strip()
        res_name = line[17:20].strip()
        chain = line[21].strip()
        try:
            res_seq = int(line[22:26].strip())
        except ValueError:
            continue
        i_code = line[26].strip()
        element = line[76:78].strip() if len(line) >= 78 else ""

        if selected_chain is None:
            selected_chain = chain
        if chain != selected_chain:
            continue
        if alt_loc not in ("", "A"):
            continue
        # Heavy atoms only.
        if element == "H" or atom_name.startswith("H"):
            continue

        mapped_res_name = res_name
        if mapped_res_name not in STANDARD_AA_3LETTER:
            if mapped_res_name in NONSTANDARD_AA_MAP:
                mapped_res_name = NONSTANDARD_AA_MAP[mapped_res_name]
            else:
                continue

        res_key = (chain, res_seq, i_code)
        if res_key not in res_key_to_index:
            res_key_to_index[res_key] = len(sequence)
            sequence.append(mapped_res_name)
            atom_coords.append({})
        res_idx = res_key_to_index[res_key]

        try:
            coord = np.array(
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                dtype=np.float32,
            )
        except ValueError:
            continue
        atom_coords[res_idx][atom_name] = coord

    if not sequence:
        raise RuntimeError("No standard-residue ATOM records parsed.")

    return ParsedStructure(sequence=sequence, atom_coords=atom_coords)


# ---- Ground truth contacts ----


@dataclass(frozen=True)
class GroundTruthContact:
    pos1: int  # 1-indexed
    pos2: int  # 1-indexed, > pos1
    atom1: str
    atom2: str
    distance: float


def compute_ground_truth(
    structure: ParsedStructure, contact_cutoff: float, min_sep: int = 2
) -> list[GroundTruthContact]:
    """One contact per residue pair: the closest valid-atom pair within cutoff.

    Adjacent residues (|i-j| < min_sep) are excluded. Only atoms valid for the
    residue's amino acid are considered.
    """
    seq = structure.sequence
    coords = structure.atom_coords

    best_per_pair: dict[tuple[int, int], GroundTruthContact] = {}

    # Flatten atoms (with residue indices) for spatial query.
    points: list[np.ndarray] = []
    point_meta: list[tuple[int, str]] = []  # (res_idx_0, atom_name)
    for res_idx, aa in enumerate(seq):
        valid = VALID_ATOMS.get(aa, set())
        for atom_name, coord in coords[res_idx].items():
            if atom_name in valid:
                points.append(coord)
                point_meta.append((res_idx, atom_name))

    if not points:
        return []

    from scipy.spatial import cKDTree

    tree = cKDTree(np.stack(points))
    pairs = tree.query_pairs(r=contact_cutoff)

    for i, j in pairs:
        ri, ai = point_meta[i]
        rj, aj = point_meta[j]
        if abs(ri - rj) < min_sep:
            continue
        if ri > rj:
            ri, rj = rj, ri
            ai, aj = aj, ai
        dist = float(np.linalg.norm(points[i] - points[j]))
        key = (ri, rj)
        existing = best_per_pair.get(key)
        if existing is None or dist < existing.distance:
            best_per_pair[key] = GroundTruthContact(pos1=ri + 1, pos2=rj + 1, atom1=ai, atom2=aj, distance=dist)

    result = list(best_per_pair.values())
    result.sort(key=lambda c: (c.pos1, c.pos2))
    return result


# ---- Prompt construction ----


def build_prompt(sequence: list[str], doc_type_token: str) -> str:
    seq_tokens = " ".join(f"<{aa}>" for aa in sequence)
    return f"{doc_type_token} <begin_sequence> {seq_tokens} <begin_contacts>"


# ---- Parsing generated contacts ----

# The model was trained on random-3-bins documents tokenized with a v1
# (deterministic-positives-only) tokenizer. All tokens the v1 tokenizer didn't
# know — the ``<random-3-bins>`` header, ``<correction>``/``<non-correction>``
# markers, ``<bin_*>`` tokens, and ``<plddt_*>`` tokens — collapse to ``<UNK>``.
# The core "contact" payload that survives is the 4-token spine
# ``<p_i> <p_j> <atom_i> <atom_j>``. Parsing skips any run of tokens that isn't
# position/atom before trying to match the spine again.


def parse_generated_contacts(tokens: list[str]) -> tuple[list[Contact], bool]:
    """Parse 4-token contact spines, skipping non-spine filler (e.g. ``<UNK>``).

    A contact is four consecutive tokens ``<p_i> <p_j> <atom_i> <atom_j>``.
    Any other non-end token (e.g. ``<UNK>`` or an unrecognized filler token)
    is skipped silently — this handles the v1-tokenizer-on-v2-documents case
    where correction/bin/pLDDT markers became ``<UNK>``.
    """
    contacts: list[Contact] = []
    is_valid = True
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in _END_MARKERS:
            break
        # Try to match a 4-token contact spine starting at i.
        if i + 4 <= n:
            t1, t2, t3, t4 = tokens[i : i + 4]
            if not any(t in _END_MARKERS for t in (t1, t2, t3, t4)):
                m1 = _POS_PATTERN.match(t1)
                m2 = _POS_PATTERN.match(t2)
                if m1 and m2 and t3 in _ATOM_TOKEN_SET and t4 in _ATOM_TOKEN_SET:
                    contacts.append(
                        (
                            int(m1.group(1)),
                            int(m2.group(1)),
                            t3.strip("<>"),
                            t4.strip("<>"),
                        )
                    )
                    i += 4
                    continue
        # No spine at i: skip this filler token and keep going.
        is_valid = False
        i += 1
    return contacts, is_valid


# ---- Scoring ----


def _unique_pairs(contacts: list[Contact], seq_len: int) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for p1, p2, _a1, _a2 in contacts:
        if p1 == p2:
            continue
        if not (1 <= p1 <= seq_len and 1 <= p2 <= seq_len):
            continue
        pairs.add((min(p1, p2), max(p1, p2)))
    return pairs


def atom_validity(contacts: list[Contact], sequence: list[str]) -> tuple[int, int]:
    """Count atom references valid for the residue's amino acid."""
    valid = 0
    total = 0
    for p1, p2, a1, a2 in contacts:
        for pos, atom in ((p1, a1), (p2, a2)):
            total += 1
            idx = pos - 1
            if 0 <= idx < len(sequence):
                aa = sequence[idx]
                if aa in VALID_ATOMS and atom in VALID_ATOMS[aa]:
                    valid += 1
    return valid, total


def score_rollout(
    contacts: list[Contact],
    gt_pairs: set[tuple[int, int]],
    sequence: list[str],
    short_range_threshold: int = 6,
) -> dict:
    seq_len = len(sequence)
    pred = _unique_pairs(contacts, seq_len)
    tp = len(pred & gt_pairs)
    fp = len(pred - gt_pairs)
    fn = len(gt_pairs - pred)

    def _split(pairs, predicate):
        return {p for p in pairs if predicate(p)}

    is_short = lambda p: abs(p[1] - p[0]) < short_range_threshold  # noqa: E731
    is_long = lambda p: abs(p[1] - p[0]) >= short_range_threshold  # noqa: E731

    gt_short = _split(gt_pairs, is_short)
    gt_long = _split(gt_pairs, is_long)
    pred_short = _split(pred, is_short)
    pred_long = _split(pred, is_long)

    valid_atoms, total_atoms = atom_validity(contacts, sequence)

    return {
        "num_contacts": len(contacts),
        "num_lt4_pairs": len(pred),
        "num_correct": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / len(pred) if pred else 0.0,
        "recall": tp / len(gt_pairs) if gt_pairs else 0.0,
        "short_range": {
            "num_pred": len(pred_short),
            "num_correct": len(pred_short & gt_short),
            "num_gt": len(gt_short),
            "precision": len(pred_short & gt_short) / len(pred_short) if pred_short else 0.0,
            "recall": len(pred_short & gt_short) / len(gt_short) if gt_short else 0.0,
        },
        "long_range": {
            "num_pred": len(pred_long),
            "num_correct": len(pred_long & gt_long),
            "num_gt": len(gt_long),
            "precision": len(pred_long & gt_long) / len(pred_long) if pred_long else 0.0,
            "recall": len(pred_long & gt_long) / len(gt_long) if gt_long else 0.0,
        },
        "bin_counts": {},
        "atom_validity": {
            "valid": valid_atoms,
            "total": total_atoms,
            "fraction": valid_atoms / total_atoms if total_atoms else 0.0,
        },
        "predicted_pairs": sorted(pred),
    }


def score_consensus(
    rollouts_pairs: list[set[tuple[int, int]]],
    gt_pairs: set[tuple[int, int]],
    seq_len: int,
    threshold: float,
    short_range_threshold: int = 6,
) -> dict:
    freq: Counter[tuple[int, int]] = Counter()
    for pairs in rollouts_pairs:
        for p in pairs:
            freq[p] += 1
    n = len(rollouts_pairs) or 1
    consensus = {p for p, c in freq.items() if c / n >= threshold}
    tp = len(consensus & gt_pairs)

    def _split(pairs, predicate):
        return {p for p in pairs if predicate(p)}

    is_short = lambda p: abs(p[1] - p[0]) < short_range_threshold  # noqa: E731
    is_long = lambda p: abs(p[1] - p[0]) >= short_range_threshold  # noqa: E731
    gt_short = _split(gt_pairs, is_short)
    gt_long = _split(gt_pairs, is_long)
    cons_short = _split(consensus, is_short)
    cons_long = _split(consensus, is_long)

    top_freq = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:100]

    return {
        "threshold": threshold,
        "num_predicted": len(consensus),
        "num_correct": tp,
        "precision": tp / len(consensus) if consensus else 0.0,
        "recall": tp / len(gt_pairs) if gt_pairs else 0.0,
        "short_range": {
            "num_pred": len(cons_short),
            "num_correct": len(cons_short & gt_short),
            "num_gt": len(gt_short),
            "precision": len(cons_short & gt_short) / len(cons_short) if cons_short else 0.0,
            "recall": len(cons_short & gt_short) / len(gt_short) if gt_short else 0.0,
        },
        "long_range": {
            "num_pred": len(cons_long),
            "num_correct": len(cons_long & gt_long),
            "num_gt": len(gt_long),
            "precision": len(cons_long & gt_long) / len(cons_long) if cons_long else 0.0,
            "recall": len(cons_long & gt_long) / len(gt_long) if gt_long else 0.0,
        },
        "predicted_pairs": sorted(consensus),
        "top_pairs_by_freq": [
            {"i": int(p[0]), "j": int(p[1]), "freq": c / n, "is_ground_truth": p in gt_pairs} for p, c in top_freq
        ],
    }


# ---- Model staging (mirror HF checkpoint from gs:// to local tmp) ----


def stage_model_locally(model_path: str) -> str:
    if not model_path.startswith(("gs://", "s3://")):
        return model_path
    fs, remote_root = url_to_fs(model_path.rstrip("/"))
    cache_key = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:16]
    local_dir = os.path.join(tempfile.gettempdir(), "marin-protein-top7-v3", cache_key)
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

    config_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json in {local_dir} (source: {model_path})")

    # The Levanter exporter mis-labels Qwen3 checkpoints as LlamaForCausalLM
    # while setting model_type=qwen3, which sends vLLM to its Llama backend and
    # trips KeyErrors on the Qwen3-specific k_norm / q_norm weights. Rewrite
    # architectures in-place on the local staging copy.
    with open(config_path, "r+") as f:
        cfg = json.load(f)
        if cfg.get("model_type") == "qwen3" and cfg.get("architectures") != ["Qwen3ForCausalLM"]:
            cfg["architectures"] = ["Qwen3ForCausalLM"]
            f.seek(0)
            json.dump(cfg, f)
            f.truncate()
            logger.info("Patched %s architectures -> Qwen3ForCausalLM", config_path)
    return local_dir


# ---- Output ----


def _write_json(path: str, obj) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ---- Main ----


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF checkpoint path (gs://... or local).")
    parser.add_argument("--pdb-id", default="1QYS", help="PDB ID (Top7 = 1QYS).")
    parser.add_argument("--chain-id", default=None)
    parser.add_argument("--num-rollouts", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=3440)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="4 for v5p-8.")
    parser.add_argument("--consensus-threshold", type=float, default=0.5)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument(
        "--doc-type-token",
        default="<UNK>",
        help=(
            "Document-type token to prepend to the prompt. Defaults to "
            "'<UNK>' — matches the v1-tokenizer-on-v2-documents case where "
            "'<random-3-bins>' collapsed to UNK at training time. Use "
            "'<deterministic-positives-only>' if the tokenizer matches the "
            "training data."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # --- 1. Ground truth ---
    logger.info("Fetching PDB %s", args.pdb_id)
    pdb_text = _fetch_pdb_text(args.pdb_id)
    structure = parse_pdb(pdb_text, chain_id=args.chain_id)
    seq_len = len(structure.sequence)
    logger.info("Parsed %s: %d residues", args.pdb_id, seq_len)

    gt_contacts = compute_ground_truth(structure, contact_cutoff=args.contact_cutoff)
    gt_pairs = {(c.pos1, c.pos2) for c in gt_contacts}
    logger.info("Ground truth: %d bin_lt4 residue pairs (within %.1f Å)", len(gt_pairs), args.contact_cutoff)
    short_gt = sum(1 for p in gt_pairs if abs(p[1] - p[0]) < 6)
    logger.info("  short-range (|i-j|<6): %d; long-range: %d", short_gt, len(gt_pairs) - short_gt)

    if not gt_pairs:
        raise RuntimeError("No ground-truth contacts.")

    # --- 2. vLLM ---
    from vllm import LLM, SamplingParams

    local_model_path = stage_model_locally(args.model)
    logger.info("Loading model %s (staged at %s) via vLLM", args.model, local_model_path)
    llm = LLM(
        model=local_model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    end_id = tokenizer.convert_tokens_to_ids("<end>")

    prompt = build_prompt(structure.sequence, args.doc_type_token)
    logger.info("Prompt prefix: %s ...", args.doc_type_token)
    logger.info("Prompt length: %d tokens", len(prompt.split()))

    def _run(sampling: SamplingParams):
        t0 = time.time()
        [out] = llm.generate([prompt], sampling, use_tqdm=False)
        elapsed = time.time() - t0
        text = out.outputs[0].text
        return text, elapsed

    # --- 3. Greedy ---
    logger.info("Greedy generation...")
    greedy_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[end_id],
    )
    greedy_text, greedy_elapsed = _run(greedy_sampling)
    greedy_tokens = greedy_text.split()
    greedy_contacts, greedy_valid_grammar = parse_generated_contacts(greedy_tokens)
    greedy_score = score_rollout(greedy_contacts, gt_pairs, structure.sequence)
    greedy_score.update(
        {
            "elapsed_seconds": greedy_elapsed,
            "valid_grammar": greedy_valid_grammar,
            "plddt_token": None,
            "raw_text_prefix": greedy_text[:500],
        }
    )
    logger.info(
        "Greedy: %d contacts, pairs=%d, correct=%d, prec=%.1f%%, rec=%.1f%% (%.1fs)",
        greedy_score["num_contacts"],
        greedy_score["num_lt4_pairs"],
        greedy_score["num_correct"],
        100 * greedy_score["precision"],
        100 * greedy_score["recall"],
        greedy_elapsed,
    )
    logger.info("Greedy raw (first 500 chars): %s", greedy_text[:500])

    # --- 4. Sampled rollouts ---
    logger.info("Sampling %d rollouts (T=%.2f)...", args.num_rollouts, args.temperature)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=args.num_rollouts,
        stop_token_ids=[end_id],
    )
    # Ask vLLM for n samples in one call.
    t0 = time.time()
    [rollout_out] = llm.generate([prompt], sampling, use_tqdm=False)
    rollout_elapsed = time.time() - t0
    rollouts_scores: list[dict] = []
    rollouts_contacts: list[list[Contact]] = []
    rollouts_pairs: list[set[tuple[int, int]]] = []
    for i, completion in enumerate(rollout_out.outputs):
        tokens_i = completion.text.split()
        contacts_i, valid_i = parse_generated_contacts(tokens_i)
        score_i = score_rollout(contacts_i, gt_pairs, structure.sequence)
        score_i.update(
            {
                "index": i,
                "valid_grammar": valid_i,
                "plddt_token": None,
                "raw_text_prefix": completion.text[:200],
            }
        )
        rollouts_scores.append(score_i)
        rollouts_contacts.append(contacts_i)
        rollouts_pairs.append(_unique_pairs(contacts_i, seq_len))
        logger.info(
            "  rollout %2d: %d contacts, pairs=%d, correct=%d, prec=%.1f%%, rec=%.1f%%",
            i,
            score_i["num_contacts"],
            score_i["num_lt4_pairs"],
            score_i["num_correct"],
            100 * score_i["precision"],
            100 * score_i["recall"],
        )

    consensus = score_consensus(
        rollouts_pairs,
        gt_pairs,
        seq_len,
        threshold=args.consensus_threshold,
    )
    logger.info(
        "Consensus (>= %.0f%% of %d): %d pred, %d correct, prec=%.1f%%, rec=%.1f%%",
        100 * args.consensus_threshold,
        args.num_rollouts,
        consensus["num_predicted"],
        consensus["num_correct"],
        100 * consensus["precision"],
        100 * consensus["recall"],
    )
    logger.info(
        "  short-range: prec=%.1f%% rec=%.1f%%; long-range: prec=%.1f%% rec=%.1f%%",
        100 * consensus["short_range"]["precision"],
        100 * consensus["short_range"]["recall"],
        100 * consensus["long_range"]["precision"],
        100 * consensus["long_range"]["recall"],
    )

    # --- 5. Write outputs ---
    output_dir = args.output_dir.rstrip("/")
    summary = {
        "pdb_id": args.pdb_id.upper(),
        "chain_id": args.chain_id,
        "sequence": structure.sequence,
        "sequence_length": seq_len,
        "ground_truth": {
            "num_pairs": len(gt_pairs),
            "num_short_range": short_gt,
            "num_long_range": len(gt_pairs) - short_gt,
            "contact_cutoff": args.contact_cutoff,
            "contacts": [
                {
                    "pos1": c.pos1,
                    "pos2": c.pos2,
                    "atom1": c.atom1,
                    "atom2": c.atom2,
                    "distance": c.distance,
                }
                for c in gt_contacts
            ],
            "pairs": sorted(gt_pairs),
        },
        "model": args.model,
        "inference": {
            "num_rollouts": args.num_rollouts,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "rollout_elapsed_seconds": rollout_elapsed,
        },
        "greedy": greedy_score,
        "rollouts": rollouts_scores,
        "consensus": consensus,
    }
    _write_json(f"{output_dir}/summary.json", summary)

    rollouts_dump = []
    for i, (contacts, score) in enumerate(zip(rollouts_contacts, rollouts_scores, strict=True)):
        rollouts_dump.append(
            {
                "index": i,
                "valid_grammar": score.get("valid_grammar"),
                "raw_text_prefix": score.get("raw_text_prefix"),
                "contacts": [
                    {
                        "pos1": c[0],
                        "pos2": c[1],
                        "atom1": c[2],
                        "atom2": c[3],
                    }
                    for c in contacts
                ],
            }
        )
    _write_json(f"{output_dir}/rollouts.json", {"rollouts": rollouts_dump})
    _write_json(
        f"{output_dir}/greedy.json",
        {
            "valid_grammar": greedy_valid_grammar,
            "raw_text_prefix": greedy_text[:2000],
            "contacts": [
                {
                    "pos1": c[0],
                    "pos2": c[1],
                    "atom1": c[2],
                    "atom2": c[3],
                }
                for c in greedy_contacts
            ],
        },
    )

    logger.info("Wrote results to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
