# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distogram-style eval for the legacy `random-3-bins` document format.

This is a sibling of `eval_protein_distogram.py` for the LlamaFold-experiments
era model (e.g. `timodonnell/LlamaFold-experiments/exp5.ethereal-galaxy-3`),
whose document format has 3 coarse distance bins instead of 64:

    <random-3-bins>
    <begin_sequence> <AA_1> ... <AA_n>
    <begin_contacts>
    <non-correction> <p_i> <p_j> <atom_i> <atom_j> <bin_lt4|bin_4_12|bin_gt12>
    ...
    <plddt_*>
    <end_contacts>
    <end>

For each ordered residue pair (i, j) with i != j we append
`<non-correction> <p_i> <p_j> <CA> <CA>` to the base prompt and read the top-K
logprobs of the next token. We renormalize over the 3 bin tokens and compare
to the ground-truth bin (derived from the actual CA-CA distance in the PDB).

The HF checkpoint at exp5.ethereal-galaxy-3/checkpoint-125500 does NOT ship a
tokenizer, so we reconstruct one from the vocab definition in
`LlamaFold-experiments/experiments/exp5_contact_prediction/src/data.py`
(reproduced here verbatim) and assemble a full HF model dir locally for vLLM.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu -- \\
        python -m experiments.protein.eval_protein_distogram_3bins \\
            --hf-repo timodonnell/LlamaFold-experiments \\
            --hf-subdir exp5.ethereal-galaxy-3/checkpoint-125500 \\
            --pdb-id 1QYS \\
            --prompt-contact-counts 0 1 2 3 4 5 \\
            --output-dir gs://marin-us-east5/eval/protein-distogram-3bins/exp5-ethereal-galaxy-3-step-125500/1qys/run-01
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import fsspec
import numpy as np
from huggingface_hub import snapshot_download

from experiments.protein.eval_protein_distogram import (
    CB_CONTACT_ANGSTROMS,
    _fetch_pdb_text,
    parse_pdb,
)

# Sequence separation threshold for a "long-range" GT contact. Matches the
# convention used elsewhere in this codebase (contacts-and-distances-v1).
LONG_RANGE_SEQ_SEP = 24

logger = logging.getLogger(__name__)

# ---- Old-format vocabulary (verbatim from exp5_contact_prediction/src/data.py) ----

CONTROL_TOKENS = [
    "<random-3-bins>",
    "<begin_sequence>",
    "<begin_contacts>",
    "<end_contacts>",
    "<end>",
]
CORRECTION_TOKENS = ["<correction>", "<non-correction>"]
DISTANCE_BIN_TOKENS = ["<bin_lt4>", "<bin_4_12>", "<bin_gt12>"]
# Bin midpoints used for expected-distance metric. The last bin is unbounded
# on the upper side (d >= 12 Å); 22 Å is a compromise that roughly matches the
# typical long-range-but-not-crazy-far pair.
DISTANCE_BIN_MIDPOINTS_A = np.array([2.0, 8.0, 22.0], dtype=np.float32)

PLDDT_TOKENS = [
    "<plddt_lt70>",
    "<plddt_70_75>",
    "<plddt_75_80>",
    "<plddt_80_85>",
    "<plddt_85_90>",
    "<plddt_90_95>",
    "<plddt_95_100>",
]
EXTRA_TOKENS = ["<UNK>"]
AMINO_ACIDS = [
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
]
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
MAX_POSITION = 2700
UTILITY_TOKENS = ["<pad>", "<eos>", "\n"]  # pad=0, eos=1, newline=2


def _all_vocab_in_order() -> list[str]:
    domain = (
        CONTROL_TOKENS
        + CORRECTION_TOKENS
        + DISTANCE_BIN_TOKENS
        + PLDDT_TOKENS
        + EXTRA_TOKENS
        + [f"<{aa}>" for aa in AMINO_ACIDS]
        + [f"<{atom}>" for atom in ATOM_NAMES]
        + [f"<p{i}>" for i in range(MAX_POSITION + 1)]
    )
    return UTILITY_TOKENS + domain


def _build_legacy_tokenizer_into_dir(out_dir: Path) -> None:
    """Build a PreTrainedTokenizerFast matching exp5.data.get_all_tokens()
    and save tokenizer files into `out_dir`."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    all_tokens = _all_vocab_in_order()
    vocab = {tok: idx for idx, tok in enumerate(all_tokens)}
    tokenizer_model = WordLevel(vocab=vocab, unk_token="<pad>")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = WhitespaceSplit()
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<pad>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    hf_tok.save_pretrained(str(out_dir))
    logger.info("Built legacy tokenizer with %d tokens → %s", len(all_tokens), out_dir)


def _assemble_model_dir(hf_repo: str, hf_subdir: str) -> Path:
    """Download HF model files + build tokenizer into one local dir. Returns the dir."""
    cache_key = hashlib.sha256(f"{hf_repo}::{hf_subdir}".encode()).hexdigest()[:16]
    local_dir = Path(tempfile.gettempdir()) / "marin-legacy-model" / cache_key
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download just the model files we need (skip optimizer.pt / rng_state_*.pth).
    wanted = [
        f"{hf_subdir}/config.json",
        f"{hf_subdir}/generation_config.json",
        f"{hf_subdir}/model.safetensors",
    ]
    logger.info("Downloading %d files from %s:%s", len(wanted), hf_repo, hf_subdir)
    staged = snapshot_download(
        repo_id=hf_repo,
        allow_patterns=wanted,
        local_dir=str(local_dir / "_download"),
    )
    # Flatten the subdir structure — vLLM wants everything at the top of local_dir.
    src = Path(staged) / hf_subdir
    for fn in ("config.json", "generation_config.json", "model.safetensors"):
        srcp = src / fn
        dstp = local_dir / fn
        if srcp.exists() and (not dstp.exists() or dstp.stat().st_size != srcp.stat().st_size):
            dstp.write_bytes(srcp.read_bytes())

    _build_legacy_tokenizer_into_dir(local_dir)
    return local_dir


# ---- Ground-truth helpers ----


def _ca_coords_by_residue(structure) -> dict[int, np.ndarray]:
    """Pull CA coordinates out of the structure (uses cb_coords which is CA for GLY).

    ParsedStructure.cb_coords is what `parse_pdb` returns; for non-GLY it's CB,
    and for GLY it's CA. For a CA-CA distance eval we want *actual* CA coords,
    which the eval_protein_distogram.parse_pdb function keeps separately in
    `ca_coords`... except that path dropped `ca_coords` from the return type.
    Re-parse the raw PDB to pick them up fresh.
    """
    raise NotImplementedError("_ca_coords_by_residue is provided inline in main() by re-parsing the PDB.")


def _bin_for_distance_A(d_A: float) -> int:
    """Return 3-bin index for distance d (in Å): 0=lt4, 1=4_12, 2=gt12."""
    if d_A < 4.0:
        return 0
    if d_A < 12.0:
        return 1
    return 2


# ---- Prompt construction ----


def _build_base_prompt_tokens_3bins(
    sequence_3letter: list[str],
    seeded_contacts_6tok: list[tuple[int, int, str, str, str]],
) -> list[str]:
    """Header + sequence + `<begin_contacts>` + N seeded 6-token contact statements.

    Each seeded contact is (i, j, atom_i, atom_j, bin_token_no_brackets), with
    `<non-correction>` as the correction marker (we're asserting true contacts,
    not corrections of something).
    """
    toks = ["<random-3-bins>", "<begin_sequence>"]
    toks.extend(f"<{aa}>" for aa in sequence_3letter)
    toks.append("<begin_contacts>")
    for i, j, ai, aj, bin_tok in seeded_contacts_6tok:
        toks.extend(["<non-correction>", f"<p{i}>", f"<p{j}>", f"<{ai}>", f"<{aj}>", f"<{bin_tok}>"])
    return toks


def _pair_query_tail_3bins(i: int, j: int, atom_i: str, atom_j: str) -> list[str]:
    """5-token tail whose next token IS the predicted distance bin."""
    return ["<non-correction>", f"<p{i}>", f"<p{j}>", f"<{atom_i}>", f"<{atom_j}>"]


# ---- vLLM query loop ----


def _encode_token(tokenizer, tok: str) -> int:
    ids = tokenizer.encode(tok, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single-token encoding of {tok!r}; got {ids!r}")
    return int(ids[0])


def _encode_tokens(tokenizer, toks: list[str]) -> list[int]:
    ids = tokenizer.encode(" ".join(toks), add_special_tokens=False)
    if len(ids) != len(toks):
        # Try per-token fallback so we surface which token is the troublemaker.
        for t in toks:
            _encode_token(tokenizer, t)
        raise ValueError(f"Whitespace-joined encoding produced {len(ids)} ids for {len(toks)} tokens.")
    return ids


def _run_3bin_distogram(
    llm,
    tokenizer,
    base_tokens: list[str],
    atom_per_residue: list[str],
    ca_coords: dict[int, np.ndarray],
    *,
    batch_size: int,
    top_k_logprobs: int,
    canonical_order: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (probs, bin_argmax, stats).

    probs[i-1, j-1, b] = P(bin b | pair (i, j))  — renormalized over the 3 bins.
    bin_argmax[i-1, j-1] = argmax bin.

    Query atoms: CB-CB for non-GLY residues, CA where the residue is GLY (or
    missing CB). CA-CA is a bad choice here because the legacy training data
    bins per-atom-pair distances and CA-CA is always >= 3.8 Å, making
    `<bin_lt4>` essentially unreachable — the model trivially puts ~0 mass
    there and the whole eval collapses.
    """
    from vllm import SamplingParams, TokensPrompt

    n_res = len(atom_per_residue)
    bin_ids = [_encode_token(tokenizer, t) for t in DISTANCE_BIN_TOKENS]
    bin_id_to_idx = {tok_id: b for b, tok_id in enumerate(bin_ids)}
    bin_id_set = set(bin_ids)

    base_ids = _encode_tokens(tokenizer, base_tokens)

    prompts: list = []
    keys: list[tuple[int, int]] = []
    for i in range(1, n_res + 1):
        for j in range(1, n_res + 1):
            if i == j:
                continue
            if canonical_order and i >= j:
                continue
            if i not in ca_coords or j not in ca_coords:
                continue
            atom_i = atom_per_residue[i - 1]  # "CB" or "CA"
            atom_j = atom_per_residue[j - 1]
            tail = _pair_query_tail_3bins(i, j, atom_i, atom_j)
            tail_ids = _encode_tokens(tokenizer, tail)
            prompts.append(TokensPrompt(prompt_token_ids=base_ids + tail_ids))
            keys.append((i, j))

    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        logprobs=top_k_logprobs,
        n=1,
    )

    probs = np.zeros((n_res, n_res, 3), dtype=np.float32)
    missing_bins_per_pair: list[int] = []
    non_bin_top_mass_per_pair: list[float] = []

    t0 = time.time()
    for chunk_start in range(0, len(prompts), batch_size):
        chunk = prompts[chunk_start : chunk_start + batch_size]
        chunk_keys = keys[chunk_start : chunk_start + batch_size]
        outputs = llm.generate(chunk, sampling, use_tqdm=False)
        for (i, j), out in zip(chunk_keys, outputs, strict=True):
            lp_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            row = np.zeros(3, dtype=np.float32)
            non_bin_mass = 0.0
            for tok_id, lp in lp_dict.items():
                p = float(np.exp(lp.logprob))
                if int(tok_id) in bin_id_set:
                    row[bin_id_to_idx[int(tok_id)]] = p
                else:
                    non_bin_mass += p
            missing_bins_per_pair.append(3 - int((row > 0).sum()))
            non_bin_top_mass_per_pair.append(non_bin_mass)
            total = float(row.sum())
            if total > 0:
                row /= total
            probs[i - 1, j - 1] = row
            if canonical_order:
                probs[j - 1, i - 1] = row
        done = chunk_start + len(chunk)
        elapsed = time.time() - t0
        logger.info(
            "  pairs %d/%d (%.1f%%) — %.1fs elapsed (%.1fms/pair)",
            done,
            len(prompts),
            100 * done / len(prompts),
            elapsed,
            1000 * elapsed / done,
        )

    bin_argmax = probs.argmax(axis=-1)

    stats = {
        "num_pairs_queried": len(prompts),
        "canonical_order": canonical_order,
        "missing_bins_median": float(np.median(missing_bins_per_pair)),
        "non_bin_top_mass_median": float(np.median(non_bin_top_mass_per_pair)),
    }
    return probs, bin_argmax, stats


# ---- Metrics ----


def _compute_metrics(
    probs: np.ndarray,
    ca_coords: dict[int, np.ndarray],
    cb_coords_for_contacts: dict[int, np.ndarray],
) -> dict:
    """Compute 3-bin metrics.

    We query with CB-CB atoms, so GT distance / bin assignment uses CB-CB.
    Contact-probability proxy is P(d < 12 Å) = P(<bin_lt4>) + P(<bin_4_12>),
    compared against the standard CB-CB <= 8 Å contact indicator. Using
    P(<bin_lt4>) alone would be wrong — a CB-CB distance of 4-8 Å (typical
    for a real contact) lands in the middle bin, not the short one.
    """
    n_res = probs.shape[0]
    preds: list[tuple[int, int, int, int, float]] = []  # (i, j, gt_bin, pred_bin, p_contact<12)
    contact_gt: list[int] = []
    contact_score: list[float] = []
    exp_distance: list[float] = []
    abs_err: list[float] = []
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if not (i in cb_coords_for_contacts and j in cb_coords_for_contacts):
                continue
            d_cb = float(np.linalg.norm(cb_coords_for_contacts[i] - cb_coords_for_contacts[j]))
            gt_bin = _bin_for_distance_A(d_cb)
            p = probs[i, j]
            pred_bin = int(p.argmax())
            p_contact_under_12 = float(p[0] + p[1])
            preds.append((i + 1, j + 1, gt_bin, pred_bin, p_contact_under_12))
            exp_d = float((p * DISTANCE_BIN_MIDPOINTS_A).sum())
            exp_distance.append(exp_d)
            abs_err.append(abs(exp_d - d_cb))
            contact_gt.append(1 if d_cb <= CB_CONTACT_ANGSTROMS else 0)
            contact_score.append(p_contact_under_12)

    if not preds:
        return {"num_pairs": 0}

    gt_bins = np.array([p[2] for p in preds], dtype=np.int32)
    pred_bins = np.array([p[3] for p in preds], dtype=np.int32)
    correct = (gt_bins == pred_bins).astype(np.float32)
    per_bin_accuracy: dict[str, float] = {}
    for b, name in enumerate(DISTANCE_BIN_TOKENS):
        mask = gt_bins == b
        if int(mask.sum()) > 0:
            per_bin_accuracy[name] = float(correct[mask].mean())
            per_bin_accuracy[f"{name}_n_gt"] = int(mask.sum())

    contact_gt_arr = np.array(contact_gt, dtype=np.float32)
    contact_score_arr = np.array(contact_score, dtype=np.float32)
    if len(contact_gt_arr) > 1 and contact_gt_arr.std() > 0 and contact_score_arr.std() > 0:
        contact_corr = float(np.corrcoef(contact_gt_arr, contact_score_arr)[0, 1])
    else:
        contact_corr = float("nan")

    gt_cb_dists = np.array(
        [float(np.linalg.norm(cb_coords_for_contacts[i - 1] - cb_coords_for_contacts[j - 1])) for i, j, _, _, _ in preds]
    )
    return {
        "num_pairs": len(preds),
        "three_way_accuracy": float(correct.mean()),
        "per_bin_accuracy": per_bin_accuracy,
        "expected_mean_abs_err_A": float(np.mean(abs_err)),
        "expected_mean_signed_err_A": float(np.mean(np.array(exp_distance) - gt_cb_dists)),
        "contact_prob_auc_proxy_corr": contact_corr,  # P(d<12) vs 1[CB-CB <= 8Å]
    }


# ---- Ground-truth long-range contacts (for seeding) ----


def _seeded_contacts_3tok(
    structure,
    n: int,
) -> list[tuple[int, int, str, str, str]]:
    """Pick the first N long-range (|i-j|>=24) CB-CB <= 8 Å contacts, rank-ordered,
    and return them as 6-token contact statements with the same atom pair we'll
    query at eval time (CB-CB, or CA for GLY residues — matches
    `structure.atom_per_residue`).
    """
    cb_coords = structure.cb_coords  # CB (or CA for GLY/missing-CB)
    atom_per_residue = structure.atom_per_residue
    pairs: list[tuple[int, int]] = []
    indices = sorted(cb_coords)
    for ii, i in enumerate(indices):
        for j in indices[ii + 1 :]:
            if j - i < LONG_RANGE_SEQ_SEP:
                continue
            d_cb = float(np.linalg.norm(cb_coords[i] - cb_coords[j]))
            if d_cb <= CB_CONTACT_ANGSTROMS:
                pairs.append((i + 1, j + 1))
    pairs.sort()
    out: list[tuple[int, int, str, str, str]] = []
    for ip, jp in pairs[:n]:
        i0, j0 = ip - 1, jp - 1
        d_cb = float(np.linalg.norm(cb_coords[i0] - cb_coords[j0]))
        bin_idx = _bin_for_distance_A(d_cb)
        bin_tok = DISTANCE_BIN_TOKENS[bin_idx][1:-1]  # strip <>
        out.append((ip, jp, atom_per_residue[i0], atom_per_residue[j0], bin_tok))
    return out


# ---- Main ----


def _write_json(path: str, obj: dict) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_npz(path: str, **arrays) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    with fsspec.open(path, "wb") as f:
        f.write(buf.getvalue())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", required=True, help="HuggingFace repo, e.g. 'timodonnell/LlamaFold-experiments'")
    parser.add_argument(
        "--hf-subdir", required=True, help="Subdir within the repo, e.g. 'exp5.ethereal-galaxy-3/checkpoint-125500'"
    )
    parser.add_argument("--pdb-id", default="1QYS")
    parser.add_argument("--chain-id", default=None)
    parser.add_argument(
        "--prompt-contact-counts",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="N = # of seeded GT long-range contacts to prepend to the prompt.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k-logprobs", type=int, default=128)
    parser.add_argument(
        "--canonical-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror P(bin|(i,j)) to (j,i); only query i<j. Default True.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # --- 1. Assemble model dir (HF + reconstructed tokenizer) ---
    local_model_dir = _assemble_model_dir(args.hf_repo, args.hf_subdir)
    logger.info("Model dir ready at %s", local_model_dir)

    # --- 2. PDB + CA / CB ground truth ---
    structure = parse_pdb(_fetch_pdb_text(args.pdb_id), chain_id=args.chain_id)
    seq_len = len(structure.sequence)
    # parse_pdb fills cb_coords with CB (or CA for GLY). Extract CA directly from the raw PDB.
    ca_only: dict[int, np.ndarray] = {}
    for line in _fetch_pdb_text(args.pdb_id).splitlines():
        if not line.startswith("ATOM  "):
            continue
        if line[12:16].strip() != "CA":
            continue
        if line[17:20].strip() not in {
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
        }:
            continue
        chain_here = line[21].strip()
        if args.chain_id is None:
            # We don't know first-chain yet; set only if this chain matches structure's first
            if chain_here != (args.chain_id or chain_here):
                continue
        elif chain_here != args.chain_id:
            continue
        if line[16].strip() not in ("", "A"):
            continue
        # Map res_seq back to the 0-indexed position in structure.sequence. The
        # simplest correct mapping is by order of first occurrence.
        pass  # (we build ca_only properly below via the structure-aware approach)

    # Re-derive ca_only using the same filter logic as parse_pdb (guaranteed to agree on indexing).
    ca_only = _extract_ca_aligned_with_structure(_fetch_pdb_text(args.pdb_id), structure, args.chain_id)
    logger.info("Parsed %s: %d residues, CA coords for %d", args.pdb_id, seq_len, len(ca_only))

    # --- 3. Load vLLM ---
    from vllm import LLM

    logger.info("Loading model via vLLM...")
    llm = LLM(
        model=str(local_model_dir),
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        trust_remote_code=True,
        max_logprobs=max(args.top_k_logprobs, 128),
    )
    tokenizer = llm.get_tokenizer()

    # --- 4. Per-N distograms ---
    output_dir = args.output_dir.rstrip("/")
    per_n_summary: list[dict] = []
    t_total = time.time()
    for n_prompt in args.prompt_contact_counts:
        seeded = _seeded_contacts_3tok(structure, n_prompt)
        if len(seeded) < n_prompt:
            logger.warning("Asked for N=%d but only %d seedable contacts available.", n_prompt, len(seeded))
        base_tokens = _build_base_prompt_tokens_3bins(structure.sequence, seeded)
        logger.info(
            "--- N=%d (seeded %d contacts) | base prompt = %d tokens ---", n_prompt, len(seeded), len(base_tokens)
        )
        t0 = time.time()
        probs, bin_argmax, stats = _run_3bin_distogram(
            llm,
            tokenizer,
            base_tokens=base_tokens,
            atom_per_residue=structure.atom_per_residue,
            ca_coords=ca_only,
            batch_size=args.batch_size,
            top_k_logprobs=args.top_k_logprobs,
            canonical_order=args.canonical_order,
        )
        elapsed = time.time() - t0
        metrics = _compute_metrics(probs, ca_only, structure.cb_coords)
        logger.info(
            "  metrics: 3-way-acc=%.3f | exp_MAE=%.2fA | contact_corr=%.3f | non-bin mass median=%.2e",
            metrics.get("three_way_accuracy", float("nan")),
            metrics.get("expected_mean_abs_err_A", float("nan")),
            metrics.get("contact_prob_auc_proxy_corr", float("nan")),
            stats["non_bin_top_mass_median"],
        )
        _write_npz(
            f"{output_dir}/distogram_n{n_prompt}.npz",
            probs=probs,
            bin_argmax=bin_argmax,
            ca_coords_indices=np.array(sorted(ca_only)),
            ca_coords=np.stack([ca_only[i] for i in sorted(ca_only)]),
            seeded_contacts=np.array(seeded, dtype=object) if seeded else np.empty(0, dtype=object),
            bin_tokens=np.array(DISTANCE_BIN_TOKENS),
            bin_midpoints=DISTANCE_BIN_MIDPOINTS_A,
            sequence_3letter=np.array(structure.sequence),
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
        "sequence_3letter": structure.sequence,
        "format": "random-3-bins",
        "inference": {
            "hf_repo": args.hf_repo,
            "hf_subdir": args.hf_subdir,
            "top_k_logprobs": args.top_k_logprobs,
            "batch_size": args.batch_size,
            "canonical_order": args.canonical_order,
            "total_elapsed_seconds": total_elapsed,
        },
        "per_n": per_n_summary,
    }
    _write_json(f"{output_dir}/summary.json", summary)
    logger.info("Wrote results to %s (total %.1fs)", output_dir, total_elapsed)
    return 0


def _extract_ca_aligned_with_structure(pdb_text: str, structure, chain_id: str | None):
    """Produce a {0-indexed position → CA np.ndarray} dict whose indexing matches
    `structure.sequence` (same res_key → index mapping parse_pdb uses)."""
    in_model_1 = True
    selected_chain: str | None = chain_id
    res_key_to_index: dict[tuple[str, int, str], int] = {}
    ca: dict[int, np.ndarray] = {}
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
        from experiments.protein.eval_protein_distogram import STANDARD_AA_3LETTER

        if res_name not in STANDARD_AA_3LETTER:
            continue
        res_key = (chain, res_seq, i_code)
        if res_key not in res_key_to_index:
            res_key_to_index[res_key] = len(res_key_to_index)
        idx = res_key_to_index[res_key]
        if atom_name == "CA":
            ca[idx] = np.array(
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                dtype=np.float32,
            )
    # Sanity: should match structure.sequence length
    if len(res_key_to_index) != len(structure.sequence):
        logger.warning(
            "Reparse mapped %d residues but structure has %d; indexing may be offset.",
            len(res_key_to_index),
            len(structure.sequence),
        )
    return ca


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
