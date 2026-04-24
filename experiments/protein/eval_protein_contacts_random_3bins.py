# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout-style contact eval for the legacy `random-3-bins` format.

Mirrors the evaluation approach used in
LlamaFold-experiments/experiments/exp5_contact_prediction/visualize_contacts.ipynb:

1. Prompt: `<random-3-bins> <begin_sequence> <AA>* <begin_contacts> [<N seeded
   GT contacts>]`.
2. Generate: (a) one greedy completion and (b) `--num-rollouts` sampled
   completions (T=1.0, top_k=0) until `<end>`.
3. Parse each completion into (correction, p1, p2, atom1, atom2, bin_token)
   contact tuples plus one pLDDT token.
4. Score `<bin_lt4>` predicted pairs against ground-truth heavy-atom contacts
   at 4 Å (closest atom pair per residue pair). Compute:
   - greedy / consensus precision & recall (total, short-range <6, long-range >=6)
   - rollout frequency matrix (fraction of rollouts emitting each pair)
   - atom validity (% of atom tokens compatible with their AA)
   - per-atom-name emission frequency (rollouts vs GT)
   - pLDDT token distribution

Also supports `--sequence-override-*` (same schema as
`eval_protein_distogram.py`) so SolubleMPNN redesigns can be scored.

Outputs `summary.json` + `rollouts.json` + `matrices.npz` into
`--output-dir`. Use `plot_protein_contacts_random_3bins.py` to render a PDF
report.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu -- \\
        python -m experiments.protein.eval_protein_contacts_random_3bins \\
            --hf-repo timodonnell/LlamaFold-experiments \\
            --hf-subdir exp5.ethereal-galaxy-3/checkpoint-125500 \\
            --pdb-id 1QYS \\
            --num-rollouts 10 \\
            --prompt-contact-counts 0 \\
            --output-dir gs://marin-us-east5/eval/protein-contacts-random-3bins/exp5-ethereal-galaxy-3-step-125500/1qys/run-01
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass

import fsspec
import numpy as np

from experiments.protein.eval_protein_distogram import _fetch_pdb_text
from experiments.protein.eval_protein_distogram_3bins import (
    AMINO_ACIDS,
    DISTANCE_BIN_TOKENS,
    PLDDT_TOKENS,
    _assemble_model_dir,
    _build_base_prompt_tokens_3bins,  # reused for prompt header
    _bin_for_distance_A,
    _encode_token,
    _encode_tokens,
)

logger = logging.getLogger(__name__)

# Per-AA valid atoms (copied from exp5 data.py so we can score atom validity).
_BACKBONE = {"N", "CA", "C", "O", "OXT"}
VALID_ATOMS: dict[str, frozenset[str]] = {
    "ALA": frozenset(_BACKBONE | {"CB"}),
    "ARG": frozenset(_BACKBONE | {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}),
    "ASN": frozenset(_BACKBONE | {"CB", "CG", "OD1", "ND2"}),
    "ASP": frozenset(_BACKBONE | {"CB", "CG", "OD1", "OD2"}),
    "CYS": frozenset(_BACKBONE | {"CB", "SG"}),
    "GLN": frozenset(_BACKBONE | {"CB", "CG", "CD", "OE1", "NE2"}),
    "GLU": frozenset(_BACKBONE | {"CB", "CG", "CD", "OE1", "OE2"}),
    "GLY": frozenset(_BACKBONE),
    "HIS": frozenset(_BACKBONE | {"CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "ILE": frozenset(_BACKBONE | {"CB", "CG1", "CG2", "CD1"}),
    "LEU": frozenset(_BACKBONE | {"CB", "CG", "CD1", "CD2"}),
    "LYS": frozenset(_BACKBONE | {"CB", "CG", "CD", "CE", "NZ"}),
    "MET": frozenset(_BACKBONE | {"CB", "CG", "SD", "CE"}),
    "PHE": frozenset(_BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "PRO": frozenset(_BACKBONE | {"CB", "CG", "CD"}),
    "SER": frozenset(_BACKBONE | {"CB", "OG"}),
    "THR": frozenset(_BACKBONE | {"CB", "OG1", "CG2"}),
    "TRP": frozenset(_BACKBONE | {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}),
    "TYR": frozenset(_BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}),
    "VAL": frozenset(_BACKBONE | {"CB", "CG1", "CG2"}),
}
STANDARD_AAS = frozenset(AMINO_ACIDS)
_END_MARKERS = frozenset({"<end_contacts>", "<end>", "<eos>", "<pad>"})
_PLDDT_TOKEN_SET = frozenset(PLDDT_TOKENS)

BIN_EDGES = [4.0, 12.0]
CONTACT_DISTANCE_CUTOFF = BIN_EDGES[0]  # 4 Å
MIN_SEQ_SEP_FOR_GT = 2  # notebook excludes |i-j| < 2


# ---- PDB parsing (all heavy atoms per residue) ----


@dataclass
class FullChain:
    sequence: list[str]  # 3-letter residue names
    # Per-residue heavy-atom coords: list indexed by position (0-based), each
    # entry is {atom_name: np.ndarray[3]}. Empty if a residue has no parsed atoms.
    atoms_by_residue: list[dict[str, np.ndarray]]
    # Flat arrays useful for KDTree:
    all_coords: np.ndarray  # (N_atoms, 3)
    all_atom_names: list[str]
    all_atom_res_idx: list[int]  # 0-indexed residue


def _parse_chain_all_atoms(pdb_text: str, chain_id: str | None) -> FullChain:
    in_model_1 = True
    selected_chain: str | None = chain_id
    sequence: list[str] = []
    res_key_to_index: dict[tuple[str, int, str], int] = {}
    atoms_by_residue: list[dict[str, np.ndarray]] = []

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
        if res_name not in STANDARD_AAS:
            continue
        # Only keep atoms that are valid for this AA (same filter exp5 data.py applies).
        if atom_name not in VALID_ATOMS[res_name]:
            continue

        res_key = (chain, res_seq, i_code)
        if res_key not in res_key_to_index:
            res_key_to_index[res_key] = len(sequence)
            sequence.append(res_name)
            atoms_by_residue.append({})
        idx = res_key_to_index[res_key]
        atoms_by_residue[idx][atom_name] = np.array(
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            dtype=np.float32,
        )

    if not sequence:
        raise RuntimeError(f"No ATOM records parsed (chain_id={chain_id!r}).")

    all_coords: list[np.ndarray] = []
    all_atom_names: list[str] = []
    all_atom_res_idx: list[int] = []
    for ri, atoms in enumerate(atoms_by_residue):
        for name, coord in atoms.items():
            all_coords.append(coord)
            all_atom_names.append(name)
            all_atom_res_idx.append(ri)

    return FullChain(
        sequence=sequence,
        atoms_by_residue=atoms_by_residue,
        all_coords=np.stack(all_coords) if all_coords else np.zeros((0, 3), dtype=np.float32),
        all_atom_names=all_atom_names,
        all_atom_res_idx=all_atom_res_idx,
    )


# ---- Ground-truth contacts (heavy-atom 4 Å, one per residue pair) ----


@dataclass(frozen=True)
class GTContact:
    p1: int  # 1-indexed residue
    p2: int  # 1-indexed, > p1
    atom1: str
    atom2: str
    distance_A: float
    bin_token: str  # "bin_lt4" / "bin_4_12" / "bin_gt12" (bin of the reported atom-pair distance)


def _ground_truth_contacts(chain: FullChain) -> list[GTContact]:
    from scipy.spatial import KDTree

    if len(chain.all_coords) == 0:
        return []
    tree = KDTree(chain.all_coords)
    close_pairs = tree.query_pairs(r=CONTACT_DISTANCE_CUTOFF)
    best_per_pair: dict[tuple[int, int], GTContact] = {}
    for i, j in close_pairs:
        ri = chain.all_atom_res_idx[i]
        rj = chain.all_atom_res_idx[j]
        if abs(ri - rj) < MIN_SEQ_SEP_FOR_GT:
            continue
        ai = chain.all_atom_names[i]
        aj = chain.all_atom_names[j]
        d = float(np.linalg.norm(chain.all_coords[i] - chain.all_coords[j]))
        if ri > rj:
            ri, rj = rj, ri
            ai, aj = aj, ai
        key = (ri, rj)
        prev = best_per_pair.get(key)
        if prev is None or d < prev.distance_A:
            best_per_pair[key] = GTContact(
                p1=ri + 1,
                p2=rj + 1,
                atom1=ai,
                atom2=aj,
                distance_A=d,
                bin_token=DISTANCE_BIN_TOKENS[_bin_for_distance_A(d)][1:-1],
            )
    return sorted(best_per_pair.values(), key=lambda c: (c.p1, c.p2))


# ---- Rollout generation + parsing ----


@dataclass
class ParsedRollout:
    contacts: list[tuple[bool, int, int, str, str, str]]  # (is_correction, p1, p2, a1, a2, bin_tok)
    plddt: str | None
    valid_grammar: bool
    num_tokens: int
    finish_reason: str


def _parse_generated_contacts(tokens: list[str]) -> ParsedRollout:
    """Mirror of exp5 parse_generated_contacts (6-token groups, inline pLDDT, end markers)."""
    contacts: list[tuple[bool, int, int, str, str, str]] = []
    plddt: str | None = None
    valid = True
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _END_MARKERS:
            break
        if tok in _PLDDT_TOKEN_SET:
            plddt = tok.strip("<>")
            i += 1
            continue
        # Expect correction/non-correction start of 6-token contact group.
        if tok not in ("<correction>", "<non-correction>"):
            valid = False
            i += 1
            continue
        if i + 5 >= len(tokens):
            valid = False
            break
        try:
            is_corr = tok == "<correction>"
            p1 = int(tokens[i + 1][2:-1])
            p2 = int(tokens[i + 2][2:-1])
            atom1 = tokens[i + 3][1:-1]
            atom2 = tokens[i + 4][1:-1]
            bin_tok_raw = tokens[i + 5]
            if bin_tok_raw not in ("<bin_lt4>", "<bin_4_12>", "<bin_gt12>"):
                valid = False
                i += 1
                continue
            bin_tok = bin_tok_raw[1:-1]
            contacts.append((is_corr, p1, p2, atom1, atom2, bin_tok))
            i += 6
        except (ValueError, IndexError):
            valid = False
            i += 1
    return ParsedRollout(contacts=contacts, plddt=plddt, valid_grammar=valid, num_tokens=len(tokens), finish_reason="")


def _run_rollouts(
    llm,
    tokenizer,
    base_tokens: list[str],
    *,
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[ParsedRollout]:
    from vllm import SamplingParams, TokensPrompt

    base_ids = _encode_tokens(tokenizer, base_tokens)
    end_id = _encode_token(tokenizer, "<end>")
    prompt = TokensPrompt(prompt_token_ids=base_ids)

    is_greedy = num_rollouts == 0 or temperature <= 0
    # `ignore_eos=True` is critical: the legacy model's config.json declares
    # eos_token_id=1 (which is `<eos>` in our reconstructed vocab), but the
    # training format actually terminates documents with `<end>` (a control
    # token). Without `ignore_eos`, vLLM stops whenever the model emits `<eos>`
    # as a stray prediction, which happens within ~60-70 contact statements and
    # truncates rollout diversity. The exp5 notebook avoided this by passing
    # `eos_token_id=end_token_id` to HF .generate(), effectively the same fix.
    sampling = SamplingParams(
        temperature=max(temperature, 0.0),
        top_k=top_k if not is_greedy else -1,
        max_tokens=max_new_tokens,
        stop_token_ids=[end_id],
        ignore_eos=True,
        n=max(num_rollouts, 1),
    )
    t0 = time.time()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    elapsed = time.time() - t0
    logger.info("vLLM generate produced %d completions in %.1fs", len(outputs[0].outputs), elapsed)

    parsed: list[ParsedRollout] = []
    for completion in outputs[0].outputs:
        text = tokenizer.decode(completion.token_ids, skip_special_tokens=False)
        pr = _parse_generated_contacts(text.split())
        pr.num_tokens = len(completion.token_ids)
        pr.finish_reason = completion.finish_reason or ""
        parsed.append(pr)
    return parsed


# ---- Metrics / matrices ----


def _contacts_to_matrix(pairs: set[tuple[int, int]], seq_len: int) -> np.ndarray:
    m = np.zeros((seq_len, seq_len), dtype=np.float32)
    for p1, p2 in pairs:
        if 1 <= p1 <= seq_len and 1 <= p2 <= seq_len:
            m[p1 - 1, p2 - 1] = 1
            m[p2 - 1, p1 - 1] = 1
    return m


def _lt4_pairs(contacts: list[tuple[bool, int, int, str, str, str]], skip_first: int = 0) -> set[tuple[int, int]]:
    return {
        (min(p1, p2), max(p1, p2))
        for (_is_corr, p1, p2, _a1, _a2, bin_tok) in contacts[skip_first:]
        if bin_tok == "bin_lt4"
    }


def _precision_recall(pred: set[tuple[int, int]], gt: set[tuple[int, int]]) -> dict[str, float]:
    tp = len(pred & gt)
    return {
        "n_predicted": len(pred),
        "n_gt": len(gt),
        "n_correct": tp,
        "precision": tp / len(pred) if pred else 0.0,
        "recall": tp / len(gt) if gt else 0.0,
    }


def _accuracy_by_range(
    pred_pairs: set[tuple[int, int]],
    gt_pairs: set[tuple[int, int]],
    threshold: int = 6,
) -> dict[str, dict[str, float]]:
    short_pred = {p for p in pred_pairs if abs(p[1] - p[0]) < threshold}
    long_pred = {p for p in pred_pairs if abs(p[1] - p[0]) >= threshold}
    short_gt = {p for p in gt_pairs if abs(p[1] - p[0]) < threshold}
    long_gt = {p for p in gt_pairs if abs(p[1] - p[0]) >= threshold}
    return {"short": _precision_recall(short_pred, short_gt), "long": _precision_recall(long_pred, long_gt)}


def _check_atom_validity(
    contacts: list[tuple[bool, int, int, str, str, str]],
    sequence: list[str],
) -> tuple[int, int]:
    """Return (valid_atom_count, total_atom_count) across both atom slots of each contact."""
    valid = 0
    total = 0
    for _is_corr, p1, p2, a1, a2, _bin in contacts:
        for pos, atom in ((p1 - 1, a1), (p2 - 1, a2)):
            total += 1
            if 0 <= pos < len(sequence):
                aa = sequence[pos]
                if aa in VALID_ATOMS and atom in VALID_ATOMS[aa]:
                    valid += 1
    return valid, total


# ---- Base-prompt construction with optional seeded contacts ----


def _seeded_contacts_6tok(
    gt_contacts: list[GTContact],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, str, str, str]]:
    """Pick N seeded contacts in random order (matching the training-data convention)."""
    idx = rng.permutation(len(gt_contacts))[:n]
    out: list[tuple[int, int, str, str, str]] = []
    for k in idx:
        c = gt_contacts[int(k)]
        out.append((c.p1, c.p2, c.atom1, c.atom2, c.bin_token))
    return out


# ---- Main ----


def _write_json(path: str, obj) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_npz(path: str, **arrays) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    with fsspec.open(path, "wb") as f:
        f.write(buf.getvalue())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-subdir", required=True)
    parser.add_argument("--pdb-id", required=True)
    parser.add_argument("--chain-id", default=None)
    parser.add_argument("--prompt-contact-counts", type=int, nargs="+", default=[0])
    parser.add_argument("--num-rollouts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="top_k=0 disables (full softmax), matching the notebook")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-override-source", default=None)
    parser.add_argument("--sequence-override-target-label", default=None)
    parser.add_argument("--sequence-override-method", default=None)
    parser.add_argument("--sequence-override-idx", type=int, default=None)
    args = parser.parse_args(argv)

    override_fields = (
        args.sequence_override_source,
        args.sequence_override_target_label,
        args.sequence_override_method,
        args.sequence_override_idx,
    )
    if any(x is not None for x in override_fields) and not all(x is not None for x in override_fields):
        parser.error("--sequence-override-* flags must all be set together.")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # --- 1. Model + tokenizer ---
    local_model_dir = _assemble_model_dir(args.hf_repo, args.hf_subdir)
    logger.info("Model staged at %s", local_model_dir)

    # --- 2. PDB + GT contacts ---
    pdb_text = _fetch_pdb_text(args.pdb_id)
    chain = _parse_chain_all_atoms(pdb_text, args.chain_id)
    seq_len = len(chain.sequence)
    gt_contacts = _ground_truth_contacts(chain)
    gt_pairs = {(c.p1, c.p2) for c in gt_contacts}
    logger.info(
        "Parsed %s: %d residues, %d heavy atoms, %d GT heavy-atom contacts (<%.0fÅ, |i-j|≥%d)",
        args.pdb_id,
        seq_len,
        len(chain.all_coords),
        len(gt_contacts),
        CONTACT_DISTANCE_CUTOFF,
        MIN_SEQ_SEP_FOR_GT,
    )

    # --- 2b. Optional sequence override (for MPNN / SolubleMPNN redesigns) ---
    sequence_for_prompt = list(chain.sequence)
    override_info: dict | None = None
    if args.sequence_override_source is not None:
        with fsspec.open(args.sequence_override_source, "r") as f:
            lines = [json.loads(l) for l in f.read().splitlines() if l.strip()]
        matches = [
            r
            for r in lines
            if r.get("target_label") == args.sequence_override_target_label
            and r.get("method") == args.sequence_override_method
            and int(r.get("redesign_idx", -1)) == args.sequence_override_idx
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected 1 override match; got {len(matches)} for "
                f"{args.sequence_override_target_label}/{args.sequence_override_method}#{args.sequence_override_idx}"
            )
        rec = matches[0]
        redesigned = list(rec["sequence_3letter"])
        if len(redesigned) != seq_len:
            raise ValueError(f"Override sequence length {len(redesigned)} != PDB chain length {seq_len}")
        hamming = sum(a != b for a, b in zip(chain.sequence, redesigned, strict=True))
        logger.info(
            "Sequence override (%s/%s #%d): Hamming = %d/%d",
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

    # --- 3. vLLM ---
    from vllm import LLM

    logger.info("Loading model via vLLM...")
    llm = LLM(
        model=str(local_model_dir),
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # --- 4. Per-N generation loop ---
    output_dir = args.output_dir.rstrip("/")
    rng = np.random.default_rng(args.seed)

    per_n_summary: list[dict] = []
    rollouts_dump: dict[str, list[dict]] = {}
    matrices: dict[str, np.ndarray] = {
        "gt_pair_matrix": _contacts_to_matrix(gt_pairs, seq_len),
    }
    t_total = time.time()

    for n_prompt in args.prompt_contact_counts:
        seeded = _seeded_contacts_6tok(gt_contacts, n_prompt, rng)
        base_tokens = _build_base_prompt_tokens_3bins(sequence_for_prompt, seeded)
        logger.info(
            "--- N=%d (seeded %d GT contacts) | base prompt = %d tokens ---", n_prompt, len(seeded), len(base_tokens)
        )

        # Greedy (1 completion, T=0). Use n=1 and temperature=0 via a second call.
        logger.info("Greedy (T=0)")
        greedy_parsed = _run_rollouts(
            llm,
            tokenizer,
            base_tokens,
            num_rollouts=1,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
        )[0]

        logger.info("Sampling %d rollouts (T=%.2f, top_k=%d)", args.num_rollouts, args.temperature, args.top_k)
        rollouts = _run_rollouts(
            llm,
            tokenizer,
            base_tokens,
            num_rollouts=args.num_rollouts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Metrics
        # Greedy: prepend seeded contacts (they don't count toward generated pairs for scoring,
        # matching notebook behavior which excludes the prefix from generation scoring).
        greedy_gen_only = greedy_parsed.contacts  # already excludes the prompt
        greedy_lt4 = _lt4_pairs(greedy_gen_only)
        greedy_range = _accuracy_by_range(greedy_lt4, gt_pairs)

        # Rollout consensus (fraction of rollouts emitting each pair; >50% threshold)
        freq = np.zeros((seq_len, seq_len), dtype=np.float32)
        for r in rollouts:
            m = _contacts_to_matrix(_lt4_pairs(r.contacts), seq_len)
            freq += m
        freq /= max(len(rollouts), 1)

        consensus_50 = {(i + 1, j + 1) for i in range(seq_len) for j in range(i + 1, seq_len) if freq[i, j] >= 0.5}
        consensus_10 = {(i + 1, j + 1) for i in range(seq_len) for j in range(i + 1, seq_len) if freq[i, j] >= 0.1}
        consensus_50_overall = _precision_recall(consensus_50, gt_pairs)
        consensus_10_overall = _precision_recall(consensus_10, gt_pairs)
        consensus_50_range = _accuracy_by_range(consensus_50, gt_pairs)

        # Atom validity + atom-name frequency across rollouts
        rollout_atom_counts: Counter[str] = Counter()
        valid_counts: list[int] = []
        total_counts: list[int] = []
        for r in rollouts:
            v, t = _check_atom_validity(r.contacts, chain.sequence)
            valid_counts.append(v)
            total_counts.append(t)
            for _ic, _p1, _p2, a1, a2, _bt in r.contacts:
                rollout_atom_counts[a1] += 1
                rollout_atom_counts[a2] += 1

        gt_atom_counts: Counter[str] = Counter()
        for c in gt_contacts:
            gt_atom_counts[c.atom1] += 1
            gt_atom_counts[c.atom2] += 1

        # pLDDT distribution
        plddt_counts = Counter(r.plddt for r in rollouts)

        # Greedy bin distribution
        greedy_bin_counter = Counter(c[5] for c in greedy_gen_only)

        per_n_summary.append(
            {
                "n_prompt_contacts": n_prompt,
                "seeded_contacts": [
                    {"p1": p1, "p2": p2, "atom1": a1, "atom2": a2, "bin_token": bt} for (p1, p2, a1, a2, bt) in seeded
                ],
                "greedy": {
                    "num_contacts": len(greedy_gen_only),
                    "bin_counts": dict(greedy_bin_counter),
                    "plddt": greedy_parsed.plddt,
                    "valid_grammar": greedy_parsed.valid_grammar,
                    "finish_reason": greedy_parsed.finish_reason,
                    "num_tokens": greedy_parsed.num_tokens,
                    "lt4_precision_recall": _precision_recall(greedy_lt4, gt_pairs),
                    "by_seq_sep": greedy_range,
                },
                "rollouts": {
                    "num": len(rollouts),
                    "median_num_contacts": float(np.median([len(r.contacts) for r in rollouts])),
                    "median_num_tokens": float(np.median([r.num_tokens for r in rollouts])),
                    "finish_reasons": dict(Counter(r.finish_reason for r in rollouts)),
                    "valid_grammar_frac": float(np.mean([r.valid_grammar for r in rollouts])),
                    "atom_validity_mean_pct": (
                        float(100 * sum(valid_counts) / sum(total_counts)) if sum(total_counts) else 0.0
                    ),
                    "plddt_counts": {str(k): v for k, v in plddt_counts.items()},
                    "consensus_50_overall": consensus_50_overall,
                    "consensus_50_by_seq_sep": consensus_50_range,
                    "consensus_10_overall": consensus_10_overall,
                },
            }
        )

        # Raw rollout dump for plotting
        rollouts_dump[f"N{n_prompt}"] = [
            {
                "contacts": r.contacts,
                "plddt": r.plddt,
                "valid_grammar": r.valid_grammar,
                "finish_reason": r.finish_reason,
                "num_tokens": r.num_tokens,
            }
            for r in rollouts
        ]
        # Matrices (per-N)
        matrices[f"freq_N{n_prompt}"] = freq
        matrices[f"greedy_pair_matrix_N{n_prompt}"] = _contacts_to_matrix(greedy_lt4, seq_len)
        # also: bin-colored greedy matrix
        bin_val = {"bin_lt4": 1, "bin_4_12": 2, "bin_gt12": 3}
        greedy_bin_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
        for _ic, p1, p2, _a1, _a2, bt in greedy_gen_only:
            if 1 <= p1 <= seq_len and 1 <= p2 <= seq_len:
                v = bin_val.get(bt, 0)
                greedy_bin_matrix[p1 - 1, p2 - 1] = v
                greedy_bin_matrix[p2 - 1, p1 - 1] = v
        matrices[f"greedy_bin_matrix_N{n_prompt}"] = greedy_bin_matrix

    total_elapsed = time.time() - t_total

    # --- 5. Write outputs ---
    _write_npz(
        f"{output_dir}/matrices.npz",
        **matrices,
        gt_atom_freq_keys=np.array(list(gt_atom_counts.keys())),
        gt_atom_freq_values=np.array(list(gt_atom_counts.values())),
        rollout_atom_freq_keys=np.array(list(rollout_atom_counts.keys())),
        rollout_atom_freq_values=np.array(list(rollout_atom_counts.values())),
    )
    _write_json(f"{output_dir}/rollouts.json", rollouts_dump)

    summary = {
        "pdb_id": args.pdb_id.upper(),
        "chain_id": args.chain_id,
        "sequence_length": seq_len,
        "native_sequence_3letter": chain.sequence,
        "sequence_used_in_prompt": sequence_for_prompt,
        "sequence_override": override_info,
        "format": "random-3-bins",
        "inference": {
            "hf_repo": args.hf_repo,
            "hf_subdir": args.hf_subdir,
            "num_rollouts": args.num_rollouts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "tensor_parallel_size": args.tensor_parallel_size,
            "total_elapsed_seconds": total_elapsed,
        },
        "ground_truth": {
            "num_contacts": len(gt_contacts),
            "distance_cutoff_A": CONTACT_DISTANCE_CUTOFF,
            "min_seq_sep": MIN_SEQ_SEP_FOR_GT,
        },
        "per_n": per_n_summary,
    }
    _write_json(f"{output_dir}/summary.json", summary)
    logger.info("Wrote results to %s (total %.1fs)", output_dir, total_elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
