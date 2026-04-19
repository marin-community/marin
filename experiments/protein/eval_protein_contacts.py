# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the protein-docs model on residue-contact prediction for a PDB entry.

Strategy: *forced-scaffold* generation. We first count ground-truth contacts by
sequence-separation class (x long-range, y medium-range, z short-range). Then we
prompt the model with the document header + sequence + `<begin_statements>`, and
for each of the (x+y+z) contact slots we:

1. Append the forced contact-type token (`<long-range-contact>` for the first x
   slots, then `<medium-range-contact>` for y slots, then `<short-range-contact>`
   for z slots).
2. Ask vLLM to sample exactly 2 tokens — the residue-position pair the model
   wants to emit for that contact.

This removes the pathology of open-ended generation (the previous version got
stuck in repetition loops and never emitted `<end>`) and makes scoring
unambiguous: each rollout predicts exactly k=GT pairs per type, so precision
= recall = F1 at that type.

Distances are intentionally not generated here.

Usage (launched via iris on a v5p-8)::

    HF_TOKEN=... uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu -- \\
        python -m experiments.protein.eval_protein_contacts \\
            --model gs://marin-us-east5/checkpoints/protein-contacts-1b-2.5e-4-780930/hf \\
            --pdb-id 1QYS \\
            --num-rollouts 16 \\
            --output-dir gs://marin-us-east5/eval/protein-contacts/1qys/run-04
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
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

# ---- Format constants (must match experiments/protein/create_protein_tokenizer.py) ----

# Contact types, in the generation order used by this script (long first, then
# medium, then short — matching the training data's rank ordering which puts
# long-range contacts first).
CONTACT_TYPES_IN_ORDER: list[tuple[str, int, float]] = [
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
    sequence: list[str]  # 3-letter residue names in position order (1-indexed externally)
    cb_coords: dict[int, np.ndarray]  # 0-indexed residue -> CB coord (CA for GLY)


def _fetch_pdb_text(pdb_id: str) -> str:
    """Fetch the .pdb file for `pdb_id` from RCSB (tries gzipped endpoint first)."""
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
    """Parse ATOM records for model 1 / a single chain; keep CA+CB only.

    For GLY (no CB) we record the CA as the "CB" representative.
    """
    in_model_1 = True
    selected_chain: str | None = chain_id
    sequence: list[str] = []
    res_key_to_index: dict[tuple[str, int, str], int] = {}
    cb_coords: dict[int, np.ndarray] = {}
    ca_coords: dict[int, np.ndarray] = {}

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
            cb_coords[res_idx] = coord
        elif atom_name == "CA":
            ca_coords[res_idx] = coord

    if not sequence:
        raise RuntimeError("No ATOM records parsed — is chain_id correct?")

    # Fill in CB coords for GLY (and any residue missing CB) using CA.
    for i, res_name in enumerate(sequence):
        if i not in cb_coords:
            if i in ca_coords:
                cb_coords[i] = ca_coords[i]
            else:
                logger.warning("Residue %d (%s) has neither CA nor CB; excluding from contacts.", i + 1, res_name)

    return ParsedStructure(sequence=sequence, cb_coords=cb_coords)


# ---- Ground truth contacts, partitioned by type ----


def ground_truth_contacts_by_type(
    structure: ParsedStructure,
) -> dict[str, set[tuple[int, int]]]:
    """Return {type_token: set[(i,j) 1-indexed]} for CB-CB <= 8 Å with the
    sequence-separation bins defined in CONTACT_TYPES_IN_ORDER.
    """
    result: dict[str, set[tuple[int, int]]] = {t: set() for t, _, _ in CONTACT_TYPES_IN_ORDER}
    indices = sorted(structure.cb_coords)
    for ii, i in enumerate(indices):
        for j in indices[ii + 1 :]:
            sep = j - i
            if sep < 6:
                continue
            dist = float(np.linalg.norm(structure.cb_coords[i] - structure.cb_coords[j]))
            if dist > CB_CONTACT_ANGSTROMS:
                continue
            for type_tok, lo, hi in CONTACT_TYPES_IN_ORDER:
                if lo <= sep < hi:
                    result[type_tok].add((i + 1, j + 1))
                    break
    return result


# ---- Prompt construction ----


def build_prompt(sequence_3letter: list[str]) -> str:
    toks = ["<contacts-and-distances-v1>", "<begin_sequence>"]
    toks.extend(f"<{aa}>" for aa in sequence_3letter)
    toks.append("<begin_statements>")
    return " ".join(toks)


# ---- Forced-scaffold generation ----


def _token_to_id(tokenizer, tok: str) -> int:
    ids = tokenizer.encode(tok, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single token id for {tok!r}, got {ids!r}")
    return int(ids[0])


def _position_from_token(tokenizer, token_id: int) -> int | None:
    tok = tokenizer.convert_ids_to_tokens(int(token_id))
    if not (isinstance(tok, str) and tok.startswith("<p") and tok.endswith(">")):
        return None
    try:
        return int(tok[2:-1])
    except ValueError:
        return None


@dataclass
class ForcedRollout:
    """One rollout's output: list of (type_token, pos_i, pos_j, raw_token_strs)."""

    entries: list[tuple[str, int | None, int | None, tuple[str, str]]]


def generate_forced_rollouts(
    llm,
    tokenizer,
    prompt: str,
    type_sequence: list[str],  # e.g. ["<long-range-contact>"]*x + ...
    *,
    n_rollouts: int,
    temperature: float,
    top_k: int,
) -> list[ForcedRollout]:
    """Iteratively sample 2 tokens per contact slot, across n rollouts in parallel."""
    from vllm import SamplingParams, TokensPrompt

    initial_ids = tokenizer.encode(prompt, add_special_tokens=False)
    type_token_ids = {t: _token_to_id(tokenizer, t) for t in set(type_sequence)}

    # Per-rollout prefix (grows as we generate) and accumulated predictions.
    prefixes: list[list[int]] = [list(initial_ids) for _ in range(n_rollouts)]
    predictions: list[list[tuple[str, int | None, int | None, tuple[str, str]]]] = [[] for _ in range(n_rollouts)]

    sampling = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        max_tokens=2,
        n=1,
    )

    total = len(type_sequence)
    report_every = max(1, total // 20)
    t_start = time.time()

    for slot_idx, contact_type in enumerate(type_sequence):
        forced_id = type_token_ids[contact_type]
        for p in prefixes:
            p.append(forced_id)

        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prefixes],
            sampling,
            use_tqdm=False,
        )

        for i, out in enumerate(outputs):
            gen_ids = list(out.outputs[0].token_ids)
            while len(gen_ids) < 2:
                gen_ids.append(tokenizer.unk_token_id or 0)
            pi_id, pj_id = int(gen_ids[0]), int(gen_ids[1])
            prefixes[i].extend([pi_id, pj_id])
            pi_tok = tokenizer.convert_ids_to_tokens(pi_id)
            pj_tok = tokenizer.convert_ids_to_tokens(pj_id)
            pi = _position_from_token(tokenizer, pi_id)
            pj = _position_from_token(tokenizer, pj_id)
            predictions[i].append((contact_type, pi, pj, (pi_tok, pj_tok)))

        if (slot_idx + 1) % report_every == 0 or slot_idx + 1 == total:
            elapsed = time.time() - t_start
            logger.info(
                "slot %d/%d (%.1f%%) — %.1fs elapsed (%.1fms/slot)",
                slot_idx + 1,
                total,
                100 * (slot_idx + 1) / total,
                elapsed,
                1000 * elapsed / (slot_idx + 1),
            )

    return [ForcedRollout(entries=p) for p in predictions]


# ---- Scoring ----


def _unique_pairs(rollout: ForcedRollout, type_tok: str, seq_len: int) -> set[tuple[int, int]]:
    """Residue pairs (1-indexed, i<j) predicted in this rollout for this type."""
    pairs: set[tuple[int, int]] = set()
    for t, pi, pj, _ in rollout.entries:
        if t != type_tok:
            continue
        if pi is None or pj is None:
            continue
        if pi == pj:
            continue
        if not (1 <= pi <= seq_len and 1 <= pj <= seq_len):
            continue
        pairs.add((min(pi, pj), max(pi, pj)))
    return pairs


def score_rollouts(
    rollouts: list[ForcedRollout],
    gt_by_type: dict[str, set[tuple[int, int]]],
    seq_len: int,
    *,
    consensus_threshold: float = 0.5,
) -> dict:
    """Per-type precision/recall per rollout + across-rollout consensus."""
    per_type_report: dict[str, dict] = {}

    for type_tok in gt_by_type:
        gt = gt_by_type[type_tok]

        per_rollout: list[dict] = []
        freq: Counter[tuple[int, int]] = Counter()
        for roll in rollouts:
            pred = _unique_pairs(roll, type_tok, seq_len)
            tp = len(pred & gt)
            per_rollout.append(
                {
                    "num_slots": sum(1 for e in roll.entries if e[0] == type_tok),
                    "num_unique_pairs": len(pred),
                    "num_invalid_slots": sum(
                        1 for e in roll.entries if e[0] == type_tok and (e[1] is None or e[2] is None)
                    ),
                    "num_out_of_range": sum(
                        1
                        for e in roll.entries
                        if e[0] == type_tok
                        and e[1] is not None
                        and e[2] is not None
                        and not (1 <= e[1] <= seq_len and 1 <= e[2] <= seq_len)
                    ),
                    "num_self_contacts": sum(
                        1
                        for e in roll.entries
                        if e[0] == type_tok and e[1] is not None and e[2] is not None and e[1] == e[2]
                    ),
                    "true_positives": tp,
                    "precision": tp / len(pred) if pred else 0.0,
                    "recall": tp / len(gt) if gt else 0.0,
                }
            )
            for p in pred:
                freq[p] += 1

        n = len(rollouts)
        consensus_pairs = {p for p, c in freq.items() if c / n >= consensus_threshold}
        tp = len(consensus_pairs & gt)
        top_freq = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
        per_type_report[type_tok] = {
            "num_ground_truth": len(gt),
            "per_rollout": per_rollout,
            "consensus": {
                "threshold": consensus_threshold,
                "num_predicted": len(consensus_pairs),
                "true_positives": tp,
                "precision": tp / len(consensus_pairs) if consensus_pairs else 0.0,
                "recall": tp / len(gt) if gt else 0.0,
                "top_pairs_by_freq": [
                    {"i": int(p[0]), "j": int(p[1]), "freq": c / n, "is_ground_truth": p in gt} for p, c in top_freq
                ],
            },
        }

    return per_type_report


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


# ---- Model staging (HF checkpoint mirrored from gs:// to local tmp) ----


def stage_model_locally(model_path: str) -> str:
    """Mirror an HF checkpoint directory from gs://... to a local tmpdir."""
    if not model_path.startswith(("gs://", "s3://")):
        return model_path

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


# ---- Output ----


def _write_json(path: str, obj: dict) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---- Main ----


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF checkpoint path (gs://... or HF repo id).")
    parser.add_argument("--pdb-id", default="1QYS")
    parser.add_argument("--chain-id", default=None)
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="4 for v5p-8.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--consensus-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # --- 1. Ground truth ---
    logger.info("Fetching PDB %s", args.pdb_id)
    pdb_text = _fetch_pdb_text(args.pdb_id)
    structure = parse_pdb(pdb_text, chain_id=args.chain_id)
    seq_len = len(structure.sequence)
    logger.info("Parsed %s: %d residues", args.pdb_id, seq_len)

    gt_by_type = ground_truth_contacts_by_type(structure)
    counts = {t: len(pairs) for t, pairs in gt_by_type.items()}
    logger.info("Ground truth counts: %s (total=%d)", counts, sum(counts.values()))

    if sum(counts.values()) == 0:
        raise RuntimeError("No ground-truth contacts found; aborting.")

    # --- 2. Build the forced type-token sequence: x long, y medium, z short ---
    type_sequence: list[str] = []
    for type_tok, _, _ in CONTACT_TYPES_IN_ORDER:
        type_sequence.extend([type_tok] * counts[type_tok])
    logger.info(
        "Total forced slots: %d (2 generated tokens each = %d sampled tokens per rollout)",
        len(type_sequence),
        2 * len(type_sequence),
    )

    # --- 3. Load vLLM ---
    from vllm import LLM

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

    # --- 4. Generate ---
    prompt = build_prompt(structure.sequence)
    logger.info("Generating %d rollouts (T=%.2f, top_k=%d)", args.num_rollouts, args.temperature, args.top_k)
    t0 = time.time()
    rollouts = generate_forced_rollouts(
        llm,
        tokenizer,
        prompt=prompt,
        type_sequence=type_sequence,
        n_rollouts=args.num_rollouts,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    elapsed = time.time() - t0
    logger.info("Forced rollout generation took %.1fs", elapsed)

    # --- 5. Score ---
    per_type_report = score_rollouts(
        rollouts,
        gt_by_type,
        seq_len,
        consensus_threshold=args.consensus_threshold,
    )

    # --- 6. Log a quick summary + write artifacts ---
    output_dir = args.output_dir.rstrip("/")
    summary = {
        "pdb_id": args.pdb_id.upper(),
        "chain_id": args.chain_id,
        "sequence_length": seq_len,
        "sequence_3letter": structure.sequence,
        "ground_truth_counts": counts,
        "ground_truth_pairs_by_type": {t: sorted(pairs) for t, pairs in gt_by_type.items()},
        "inference": {
            "model": args.model,
            "num_rollouts": args.num_rollouts,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "elapsed_seconds": elapsed,
            "total_forced_slots": len(type_sequence),
        },
        "per_type": per_type_report,
    }
    _write_json(f"{output_dir}/summary.json", summary)

    # Per-rollout raw predictions for downstream plotting.
    rollouts_dump = []
    for idx, roll in enumerate(rollouts):
        rollouts_dump.append(
            {
                "index": idx,
                "entries": [
                    {
                        "type": t,
                        "pos_i": pi,
                        "pos_j": pj,
                        "pi_token": tok_i,
                        "pj_token": tok_j,
                    }
                    for t, pi, pj, (tok_i, tok_j) in roll.entries
                ],
            }
        )
    _write_json(f"{output_dir}/rollouts.json", {"rollouts": rollouts_dump})

    # Log a pretty headline for each contact type.
    for type_tok, report in per_type_report.items():
        per_roll = report["per_rollout"]
        precs = [r["precision"] for r in per_roll]
        recs = [r["recall"] for r in per_roll]
        invalids = [r["num_invalid_slots"] for r in per_roll]
        oors = [r["num_out_of_range"] for r in per_roll]
        selfs = [r["num_self_contacts"] for r in per_roll]
        logger.info(
            "%s — gt=%d  per-rollout P/R median=%.3f/%.3f  consensus(@%.2f) P=%.3f R=%.3f "
            "(pred %d / tp %d)  invalid_slots med=%.0f  oor med=%.0f  self med=%.0f",
            type_tok,
            report["num_ground_truth"],
            _median(precs) or 0.0,
            _median(recs) or 0.0,
            args.consensus_threshold,
            report["consensus"]["precision"],
            report["consensus"]["recall"],
            report["consensus"]["num_predicted"],
            report["consensus"]["true_positives"],
            _median(invalids) or 0.0,
            _median(oors) or 0.0,
            _median(selfs) or 0.0,
        )

    logger.info("Wrote results to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
