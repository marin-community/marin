# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ProteinGym DMS-substitutions zero-shot evaluation for the protein-docs LM.

Treat the model as a (causal) protein language model. For each DMS dataset we
construct the document prefix

    <contacts-and-distances-v1> <begin_sequence> <AA_1> <AA_2> ... <AA_n>

Single-substitution variants are scored from the **wild-type forward pass**:

    score(i, ref->alt) = log P(<alt> | prefix up to i-1)
                       - log P(<ref> | prefix up to i-1)

(Both numerator and denominator share the same prefix — one WT pass gives
all single-mutant scores via lookup.)

Multi-substitution variants are scored by an **independent forward pass over
the mutated sequence** and a sequence-level joint log-likelihood difference

    score(variant) = sum_i [log P(<aa_var_i> | mut prefix up to i-1)
                          - log P(<aa_wt_i>  | wt  prefix up to i-1)]

(Single-mutant variants would give the same answer either way, but we use
the lookup path for speed.)

Position-bias note: this is a causal LM, so the model has no left context at
the N-terminus. We always emit per-position perplexity profiles and
position-stratified Spearman so the reader can see the bias directly.

Inference path: HuggingFace transformers on CPU. vLLM's TPU backend currently
does not implement `prompt_logprobs` (see vllm-project/tpu-inference#1333).
The 1B model is small enough that one forward pass per protein on 16 vCPU
takes a few seconds.

Usage::

    HF_TOKEN=... uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=32GB --disk=64GB --cpu=16 --extra=cpu --zone=us-east5-a -- \\
        python -m experiments.protein.eval_protein_proteingym \\
            --model gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e/hf/step-15049 \\
            --proteingym-dir gs://marin-us-east5/protein-structure/proteingym/v1.3 \\
            --datasets BLAT_ECOLX_Stiffler_2015 CALM1_HUMAN_Weile_2017 \\
            --output-dir gs://marin-us-east5/eval/protein-proteingym/<run>/step-15049
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np

logger = logging.getLogger(__name__)

# 1-letter -> 3-letter amino acid map (ProteinGym uses 1-letter; our tokenizer uses <ALA>, etc.).
ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}
STANDARD_AA = list(ONE_TO_THREE.keys())  # 20 letters


# ---- Data IO ----


def _read_text(path: str) -> str:
    with fsspec.open(path, "r") as f:
        return f.read()


def _write_json(path: str, obj) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_npz(path: str, **arrays) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    with fsspec.open(path, "wb") as f:
        f.write(buf.getvalue())


@dataclass(frozen=True)
class DatasetMeta:
    dms_id: str
    target_seq: str  # 1-letter WT sequence
    seq_len: int
    includes_multiple: bool
    raw_mut_offset: int  # mutation positions are 1-indexed; offset adjusts to target_seq indexing
    csv_path: str  # full GCS path to the per-dataset CSV


def _load_reference(reference_csv: str) -> dict[str, DatasetMeta]:
    text = _read_text(reference_csv)
    rows = list(csv.DictReader(io.StringIO(text)))
    out: dict[str, DatasetMeta] = {}
    for r in rows:
        # raw_mut_offset is "" for most rows (== 0 effectively).
        offset_str = r.get("raw_mut_offset", "")
        offset = int(offset_str) if offset_str.strip() else 0
        dms_id = r["DMS_id"]
        out[dms_id] = DatasetMeta(
            dms_id=dms_id,
            target_seq=r["target_seq"],
            seq_len=int(r["seq_len"]),
            includes_multiple=r["includes_multiple_mutants"].upper() == "TRUE",
            raw_mut_offset=offset,
            csv_path="",  # filled by caller
        )
    return out


def _read_dms_csv(path: str) -> list[dict]:
    text = _read_text(path)
    return list(csv.DictReader(io.StringIO(text)))


# ---- Mutation parsing ----


def _parse_mutant_field(mutant: str) -> list[tuple[int, str, str]]:
    """Parse a 'mutant' field like 'A23K' or 'A23K:G45L:M67P' into (pos, ref, alt) tuples.

    Position is 1-indexed (relative to the target_seq; offset already applied upstream).
    """
    if not mutant or mutant == "WT":
        return []
    out: list[tuple[int, str, str]] = []
    for tok in mutant.split(":"):
        tok = tok.strip()
        if not tok:
            continue
        ref = tok[0]
        alt = tok[-1]
        try:
            pos = int(tok[1:-1])
        except ValueError as err:
            raise ValueError(f"Cannot parse mutation token {tok!r} in {mutant!r}") from err
        out.append((pos, ref, alt))
    return out


# ---- Tokenizer / prompt construction ----


def _aa3_token(one_letter: str) -> str:
    return f"<{ONE_TO_THREE[one_letter]}>"


def _build_prompt_tokens(sequence_1letter: str) -> list[str]:
    """Document prefix the model expects for a sequence-only PLM-style query.

    Note: we deliberately stop after the AAs (no <begin_statements>); the LM
    head probabilities at AA positions are what we want.
    """
    toks = ["<contacts-and-distances-v1>", "<begin_sequence>"]
    toks.extend(_aa3_token(c) for c in sequence_1letter)
    return toks


def _encode_tokens(tokenizer, tokens: list[str]) -> list[int]:
    ids = tokenizer.encode(" ".join(tokens), add_special_tokens=False)
    if len(ids) != len(tokens):
        for t in tokens:
            single = tokenizer.encode(t, add_special_tokens=False)
            if len(single) != 1:
                raise ValueError(f"Token {t!r} encodes to {len(single)} ids: {single}")
        raise ValueError(f"Whitespace-joined encoding produced {len(ids)} ids for {len(tokens)} tokens.")
    return ids


# ---- Inference ----


def _aa_token_ids(tokenizer) -> dict[str, int]:
    """Map 1-letter AA -> token id."""
    return {one: tokenizer.encode(_aa3_token(one), add_special_tokens=False)[0] for one in STANDARD_AA}


def _wt_logprobs_per_position(
    model,
    tokenizer,
    wt_sequence: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single forward pass over the WT sequence and return:

    - aa_logprobs: (n, 20) — log P(aa | prefix up to pos-1) at each AA position, in STANDARD_AA order
    - wt_aa_logprobs: (n,) — log P(wt_aa | prefix up to pos-1), pulled from aa_logprobs at the WT residue
    """
    import torch

    prompt_tokens = _build_prompt_tokens(wt_sequence)
    prompt_ids = _encode_tokens(tokenizer, prompt_tokens)
    aa_id_arr = np.array([_aa_token_ids(tokenizer)[a] for a in STANDARD_AA], dtype=np.int64)

    # logits[t, v] = pre-softmax score for predicting token at t+1 given tokens[0..t].
    # i.e. log_softmax(logits[t]) = log P(. | tokens[0..t]).
    # AA at protein position p (1-indexed) sits at index p+1 in the prompt
    # (after <header>, <begin_sequence>). So to score AA at p we want the
    # distribution at logits index p (i.e. the predictor of token at p+1).
    t0 = time.time()
    device = next(model.parameters()).device
    with torch.no_grad():
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits = model(ids).logits[0]  # (T, V)
        # bf16 -> fp32 for softmax stability
        logp = torch.log_softmax(logits.to(torch.float32), dim=-1).cpu().numpy()
    logger.debug("WT pass: %d tokens in %.2fs", len(prompt_ids), time.time() - t0)

    n = len(wt_sequence)
    aa_logprobs = np.full((n, 20), -np.inf, dtype=np.float32)
    for pos1 in range(1, n + 1):
        idx = pos1  # logits[idx] predicts token at idx+1 = AA at position pos1
        if idx >= logp.shape[0]:
            continue
        aa_logprobs[pos1 - 1] = logp[idx, aa_id_arr]
    wt_aa_logprobs = np.array(
        [aa_logprobs[i, STANDARD_AA.index(c)] if c in ONE_TO_THREE else np.nan for i, c in enumerate(wt_sequence)],
        dtype=np.float32,
    )
    return aa_logprobs, wt_aa_logprobs


def _multi_mut_joint_logprob(
    model,
    tokenizer,
    sequences: list[str],
    *,
    batch_size: int = 16,
) -> np.ndarray:
    """Return (len(sequences),) array of joint log P(<AA_1> ... <AA_n>) for each sequence.

    For each sequence, compute log_softmax over the LM logits and gather the
    log-probabilities at each AA token position. Sum gives the joint LL
    (the model's probability that the AA sequence is generated under the LM,
    given the document header).

    Sequences of identical length are batched into a single forward pass —
    important for GPU throughput on multi-mutant-heavy datasets.
    """
    import torch

    if not sequences:
        return np.zeros(0, dtype=np.float32)

    out = np.zeros(len(sequences), dtype=np.float64)
    aa_id_full = _aa_token_ids(tokenizer)
    device = next(model.parameters()).device
    # All variants of the same target share length, so we can batch without padding.
    # Pre-encode prompt token-ids and the per-sequence AA token-ids vector
    # (length = seq_len, indexed by prompt position p1 = i+1).
    prompt_id_lists: list[list[int]] = []
    aa_target_id_lists: list[list[int]] = []  # token id at each position; None if non-standard
    seq_lens: list[int] = []
    for seq in sequences:
        prompt_tokens = _build_prompt_tokens(seq)
        prompt_id_lists.append(_encode_tokens(tokenizer, prompt_tokens))
        aa_target_id_lists.append([aa_id_full.get(c, -1) for c in seq])
        seq_lens.append(len(seq))

    # Group by length. Substitution variants of one target all have identical
    # length, but if the caller mixes targets we still degrade gracefully.
    from collections import defaultdict

    by_len: dict[int, list[int]] = defaultdict(list)
    for s, L in enumerate(seq_lens):
        by_len[L].append(s)

    t0 = time.time()
    done = 0
    for length, idxs in by_len.items():
        # Process this length group in chunks of `batch_size`.
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start : start + batch_size]
            batch_ids = torch.tensor([prompt_id_lists[s] for s in chunk], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(batch_ids).logits  # (B, T, V), bf16
                logp = torch.log_softmax(logits.to(torch.float32), dim=-1)
                # Gather log-prob of the target AA at each AA position.
                # AA at 1-indexed protein position p sits at prompt index p+1
                # and is predicted by logits[..., p, :].
                # We want, for each (B, p), logp[b, p, aa_target_id_lists[chunk[b]][p-1]].
                # Build target_ids tensor of shape (B, length); pad value 0 where non-standard.
                tgt = torch.tensor(
                    [aa_target_id_lists[s] for s in chunk],
                    dtype=torch.long,
                    device=device,
                )  # (B, length); -1 for non-standard
                valid = tgt >= 0
                tgt_clamped = tgt.clamp_min(0)
                T = logp.shape[1]
                # Gather at logits index 1..length (i.e. logp[:, 1:1+length, :])
                end_idx = min(T, 1 + length)
                logp_aa_positions = logp[:, 1:end_idx, :]  # (B, end_idx-1, V)
                # Trim valid + tgt_clamped to match
                tgt_clamped = tgt_clamped[:, : end_idx - 1]
                valid = valid[:, : end_idx - 1]
                gathered = logp_aa_positions.gather(2, tgt_clamped.unsqueeze(-1)).squeeze(-1)
                gathered = gathered * valid
                lls = gathered.sum(dim=1).cpu().numpy()
            for j, s in enumerate(chunk):
                out[s] = float(lls[j])
            done += len(chunk)
            if done % 200 == 0 or done == len(sequences):
                logger.info(
                    "  multi-mut: %d/%d (%.1fs elapsed, batch_size=%d on %s)",
                    done,
                    len(sequences),
                    time.time() - t0,
                    batch_size,
                    device,
                )
    elapsed = time.time() - t0
    logger.debug("multi-mut batched: %d sequences in %.2fs", len(sequences), elapsed)
    return out.astype(np.float32)


# ---- Per-dataset evaluation ----


@dataclass
class DatasetResult:
    dms_id: str
    seq_len: int
    num_variants: int
    num_multi_mutant: int
    spearman_overall: float
    spearman_by_position_bucket: dict[str, float]  # "0-25", "25-50", "50-75", "75-100"
    perplexity_profile_mean: list[float]  # per-position mean -log P(WT_AA), length seq_len
    elapsed_seconds: float


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, NaN-safe."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[mask]))
    ry = np.argsort(np.argsort(y[mask]))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _evaluate_dataset(
    model,
    tokenizer,
    meta: DatasetMeta,
    *,
    max_protein_len: int,
    multi_mut_batch_size: int,
) -> tuple[DatasetResult, np.ndarray, np.ndarray, list[dict]]:
    """Score every variant in the dataset. Returns (result, aa_logprobs, model_scores, per_variant_rows)."""
    rows = _read_dms_csv(meta.csv_path)
    seq = meta.target_seq
    n = len(seq)
    if n > max_protein_len:
        raise ValueError(f"Dataset {meta.dms_id} seq_len={n} > max_protein_len={max_protein_len}")

    t0 = time.time()
    logger.info("[%s] WT forward pass (n=%d)", meta.dms_id, n)
    aa_logprobs, wt_logprobs = _wt_logprobs_per_position(model, tokenizer, seq)

    # Single-mutant lookup table:
    #   score(pos, ref->alt) = aa_logprobs[pos-1, alt_idx] - aa_logprobs[pos-1, ref_idx]
    aa_idx = {aa: k for k, aa in enumerate(STANDARD_AA)}

    # Group variants: single -> direct lookup; multi -> joint forward pass.
    multi_variants: list[tuple[int, str]] = []  # (row_idx, mutated_sequence)
    per_variant_rows: list[dict] = []
    parse_errors = 0
    for row_idx, row in enumerate(rows):
        mutant = row["mutant"]
        try:
            muts = _parse_mutant_field(mutant)
        except ValueError as e:
            parse_errors += 1
            logger.warning("[%s] parse error: %s", meta.dms_id, e)
            per_variant_rows.append(
                {"row_idx": row_idx, "mutant": mutant, "model_score": float("nan"), "skipped": "parse_error"}
            )
            continue

        if len(muts) == 0:
            per_variant_rows.append({"row_idx": row_idx, "mutant": mutant, "model_score": float("nan"), "skipped": "WT"})
            continue

        # Apply offset and check bounds.
        valid = True
        for pos, ref, _alt in muts:
            true_pos = pos - meta.raw_mut_offset
            if not (1 <= true_pos <= n):
                valid = False
                break
            if seq[true_pos - 1] != ref:
                valid = False
                break
        if not valid:
            per_variant_rows.append(
                {"row_idx": row_idx, "mutant": mutant, "model_score": float("nan"), "skipped": "ref_mismatch_or_oob"}
            )
            continue

        if len(muts) == 1:
            pos, ref, alt = muts[0]
            true_pos = pos - meta.raw_mut_offset
            if alt not in aa_idx or ref not in aa_idx:
                per_variant_rows.append(
                    {"row_idx": row_idx, "mutant": mutant, "model_score": float("nan"), "skipped": "non_standard_aa"}
                )
                continue
            score = float(aa_logprobs[true_pos - 1, aa_idx[alt]] - aa_logprobs[true_pos - 1, aa_idx[ref]])
            per_variant_rows.append(
                {
                    "row_idx": row_idx,
                    "mutant": mutant,
                    "n_substitutions": 1,
                    "primary_position": true_pos,
                    "min_position": true_pos,
                    "max_position": true_pos,
                    "model_score": score,
                }
            )
        else:
            # Multi-mut: queue mutated sequence for joint forward pass.
            mut_seq_list = list(seq)
            non_standard = False
            for pos, _ref, alt in muts:
                true_pos = pos - meta.raw_mut_offset
                if alt not in aa_idx:
                    non_standard = True
                    break
                mut_seq_list[true_pos - 1] = alt
            if non_standard:
                per_variant_rows.append(
                    {"row_idx": row_idx, "mutant": mutant, "model_score": float("nan"), "skipped": "non_standard_aa"}
                )
                continue
            mut_seq = "".join(mut_seq_list)
            positions = [pos - meta.raw_mut_offset for pos, _, _ in muts]
            multi_variants.append((row_idx, mut_seq))
            per_variant_rows.append(
                {
                    "row_idx": row_idx,
                    "mutant": mutant,
                    "n_substitutions": len(muts),
                    "primary_position": positions[0],
                    "min_position": min(positions),
                    "max_position": max(positions),
                    "model_score": None,  # filled below
                }
            )

    # Compute WT joint LL once (sum of WT logprobs).
    wt_total_ll = float(np.nansum(wt_logprobs))

    # Score multi-mut variants in batches.
    if multi_variants:
        logger.info(
            "[%s] %d multi-mutant variants -> batched joint forward passes (batch_size=%d)",
            meta.dms_id,
            len(multi_variants),
            multi_mut_batch_size,
        )
        multi_seqs = [s for _, s in multi_variants]
        all_lls = _multi_mut_joint_logprob(
            model,
            tokenizer,
            multi_seqs,
            batch_size=multi_mut_batch_size,
        )
        # Map back into per_variant_rows.
        idx_to_row = {row_idx: i for i, (row_idx, _) in enumerate(multi_variants)}
        for r in per_variant_rows:
            ri = r["row_idx"]
            if r.get("model_score") is None and ri in idx_to_row:
                mut_ll = float(all_lls[idx_to_row[ri]])
                r["model_score"] = mut_ll - wt_total_ll

    # Pull DMS scores for correlation.
    dms_score = []
    pred = []
    positions = []
    for r in per_variant_rows:
        sc = r.get("model_score")
        if sc is None or not np.isfinite(sc):
            continue
        try:
            ds = float(rows[r["row_idx"]]["DMS_score"])
        except (ValueError, KeyError):
            continue
        if not np.isfinite(ds):
            continue
        dms_score.append(ds)
        pred.append(sc)
        positions.append(r.get("primary_position", -1))

    dms_arr = np.array(dms_score, dtype=np.float32)
    pred_arr = np.array(pred, dtype=np.float32)
    pos_arr = np.array(positions, dtype=np.int32)
    spearman_overall = _spearman(pred_arr, dms_arr)

    # Position-stratified Spearman.
    bucket_edges = [0, 0.25, 0.5, 0.75, 1.01]
    bucket_labels = ["0-25", "25-50", "50-75", "75-100"]
    sb: dict[str, float] = {}
    rel_pos = np.where(pos_arr > 0, pos_arr / max(n, 1), -1.0)
    for lo, hi, lbl in zip(bucket_edges[:-1], bucket_edges[1:], bucket_labels, strict=True):
        mask = (rel_pos >= lo) & (rel_pos < hi)
        if mask.sum() >= 5:
            sb[lbl] = _spearman(pred_arr[mask], dms_arr[mask])
        else:
            sb[lbl] = float("nan")

    # Per-position perplexity profile (negative log-likelihood of WT residue).
    profile = (-wt_logprobs).tolist()

    elapsed = time.time() - t0
    logger.info(
        "[%s] %d variants scored | overall Spearman=%.3f | %.1fs",
        meta.dms_id,
        int(np.isfinite(pred_arr).sum()),
        spearman_overall,
        elapsed,
    )

    result = DatasetResult(
        dms_id=meta.dms_id,
        seq_len=n,
        num_variants=len(rows),
        num_multi_mutant=len(multi_variants),
        spearman_overall=spearman_overall,
        spearman_by_position_bucket=sb,
        perplexity_profile_mean=profile,
        elapsed_seconds=elapsed,
    )
    return result, aa_logprobs, pred_arr, per_variant_rows


# ---- Model staging (mirror of eval_protein_distogram pattern) ----


def stage_model_locally(model_path: str) -> str:
    """Mirror an HF model dir from gs:// to a local cache for vLLM."""
    if not model_path.startswith(("gs://", "s3://")):
        return model_path
    import hashlib

    from rigging.filesystem import url_to_fs

    fs, remote_root = url_to_fs(model_path.rstrip("/"))
    cache_key = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:16]
    local_dir = Path(tempfile.gettempdir()) / "marin-protein-eval-model" / cache_key
    local_dir.mkdir(parents=True, exist_ok=True)

    found = fs.find(remote_root, detail=True, maxdepth=1)
    entries = list(found.values()) if isinstance(found, dict) else [fs.info(p) for p in found]
    logger.info("Staging %d entries from %s -> %s", len(entries), model_path, local_dir)
    for entry in entries:
        if entry.get("type") != "file":
            continue
        name = os.path.basename(entry["name"].rstrip("/"))
        if not name:
            continue
        local_path = local_dir / name
        size = entry.get("size")
        if local_path.exists() and size is not None and local_path.stat().st_size == size:
            continue
        logger.info("  download %s (%s bytes)", name, size)
        fs.get(entry["name"], str(local_path))
    if not (local_dir / "config.json").exists():
        raise FileNotFoundError(f"No config.json in {local_dir} (source: {model_path})")
    return str(local_dir)


# ---- Main ----


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--proteingym-dir", required=True, help="Root containing reference/ and DMS_ProteinGym_substitutions/"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of DMS_id values. Default: all 217.")
    parser.add_argument("--max-protein-len", type=int, default=2700)
    parser.add_argument(
        "--multi-mut-batch-size",
        type=int,
        default=64,
        help="Number of multi-mutant sequences to process in one log-progress chunk.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: 'auto' (cuda if available, else cpu), 'cpu', 'cuda'.",
    )
    args = parser.parse_args(argv)

    proteingym_dir = args.proteingym_dir.rstrip("/")
    reference_csv = f"{proteingym_dir}/reference/DMS_substitutions.csv"
    metas = _load_reference(reference_csv)

    if args.datasets:
        ids = list(args.datasets)
        unknown = [i for i in ids if i not in metas]
        if unknown:
            raise SystemExit(f"Unknown DMS_id(s): {unknown}")
    else:
        ids = sorted(metas.keys())

    logger.info("Eval %d datasets from %s", len(ids), proteingym_dir)

    # Stage model + load via HF transformers.
    # vLLM's TPU backend doesn't currently support prompt_logprobs (see
    # vllm-project/tpu-inference#1333), so we use plain torch. The 1B model
    # fits comfortably on a CPU-only iris worker (~2.6 GB bf16) and a single
    # forward pass over a few-hundred-residue prompt takes a few seconds on
    # 16 vCPU. On a single GPU it's roughly 100x faster.
    local_model = stage_model_locally(args.model)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Loading model %s via transformers (device=%s)", args.model, device)
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(local_model)
    model = AutoModelForCausalLM.from_pretrained(local_model, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    output_dir = args.output_dir.rstrip("/")
    summary_results: list[dict] = []

    for ds_idx, dms_id in enumerate(ids):
        meta = metas[dms_id]
        if meta.seq_len > args.max_protein_len:
            logger.warning("[%s] seq_len=%d > %d; skipping", dms_id, meta.seq_len, args.max_protein_len)
            summary_results.append({"dms_id": dms_id, "seq_len": meta.seq_len, "skipped": "too_long"})
            continue
        meta = DatasetMeta(
            dms_id=meta.dms_id,
            target_seq=meta.target_seq,
            seq_len=meta.seq_len,
            includes_multiple=meta.includes_multiple,
            raw_mut_offset=meta.raw_mut_offset,
            csv_path=f"{proteingym_dir}/DMS_ProteinGym_substitutions/{dms_id}.csv",
        )

        logger.info(
            "=== [%d/%d] %s (seq_len=%d, includes_multi=%s) ===",
            ds_idx + 1,
            len(ids),
            dms_id,
            meta.seq_len,
            meta.includes_multiple,
        )
        try:
            result, aa_logprobs, _pred, per_variant = _evaluate_dataset(
                model,
                tokenizer,
                meta,
                max_protein_len=args.max_protein_len,
                multi_mut_batch_size=args.multi_mut_batch_size,
            )
        except Exception as e:
            logger.exception("[%s] failed: %s", dms_id, e)
            summary_results.append({"dms_id": dms_id, "skipped": "exception", "error": str(e)})
            continue

        # Write per-dataset artifacts.
        ds_out = f"{output_dir}/per_dataset/{dms_id}"
        _write_json(f"{ds_out}/per_variant.json", per_variant)
        _write_npz(f"{ds_out}/aa_logprobs.npz", aa_logprobs=aa_logprobs)

        summary_results.append(
            {
                "dms_id": dms_id,
                "seq_len": result.seq_len,
                "num_variants": result.num_variants,
                "num_multi_mutant": result.num_multi_mutant,
                "spearman_overall": result.spearman_overall,
                "spearman_by_position_bucket": result.spearman_by_position_bucket,
                "perplexity_profile_mean": result.perplexity_profile_mean,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        # Stream-write the summary so partial progress is visible if a later dataset fails.
        _write_json(f"{output_dir}/summary.json", {"datasets": summary_results, "model": args.model})

    logger.info("Wrote summary to %s/summary.json", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
