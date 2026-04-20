# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redesign benchmark protein sequences using ProteinMPNN.

Local script. Not integrated with the executor / iris / TPU. Run once on a
machine with PyTorch (CPU is fine; ~30 s per target); outputs a JSONL in GCS
that the training experiment picks up via `distogram_eval_benchmark(...,
redesigns_source=...)`.

Setup (once)::

    # 1. Clone ProteinMPNN somewhere stable. Either:
    #    (a) set PROTEINMPNN_DIR env var to point at an existing clone, OR
    #    (b) let this script clone it to third_party/ProteinMPNN.
    git clone https://github.com/dauparas/ProteinMPNN.git third_party/ProteinMPNN

    # 2. Install the runtime deps (not in marin's main env):
    pip install torch biopython

Usage::

    uv run python -m experiments.protein.redesign_sequences \\
        --output gs://marin-us-east5/protein-structure/protein-mpnn-redesigns/v1/redesigns.jsonl \\
        --temperature 0.1 \\
        --seed 0

One JSONL record per (target, redesign_idx). Schema is matched to what
`protein_distogram_eval.load_redesigned_targets()` expects.

Each record::

    {
        "target_label": "foldbench-7pv5",
        "pdb_id": "7PV5",
        "chain_id": null,
        "assembly": 1,
        "method": "mpnn",
        "temperature": 0.1,
        "redesign_idx": 0,
        "sequence_3letter": ["THR", "TYR", ...],
        "native_sequence_3letter": ["THR", "TYR", ...],
        "hamming_distance": 27,
        "mpnn_score": 0.82
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fsspec

from experiments.protein.protein_distogram_eval import (
    ProteinTarget,
    STANDARD_AA_3LETTER,
    TARGETS,
    _fetch_pdb_text,
    _parse_chain,
)

logger = logging.getLogger(__name__)

PROTEINMPNN_REPO = "https://github.com/dauparas/ProteinMPNN.git"
DEFAULT_PROTEINMPNN_DIR = "third_party/ProteinMPNN"

_ONE_LETTER_TO_THREE = {
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


def _three_to_one(aa3: str) -> str:
    # inverse, for converting native sequence to one-letter for MPNN if needed
    for one, three in _ONE_LETTER_TO_THREE.items():
        if three == aa3:
            return one
    raise KeyError(aa3)


def _ensure_proteinmpnn(explicit_dir: str | None) -> Path:
    """Return a Path to a working ProteinMPNN checkout. Clones if not present."""
    if explicit_dir:
        path = Path(explicit_dir).expanduser().resolve()
        if not (path / "protein_mpnn_run.py").exists():
            raise FileNotFoundError(f"PROTEINMPNN_DIR={path} is set but does not contain protein_mpnn_run.py")
        return path

    path = Path(DEFAULT_PROTEINMPNN_DIR).resolve()
    if (path / "protein_mpnn_run.py").exists():
        logger.info("Using existing ProteinMPNN clone at %s", path)
        return path

    logger.info("Cloning ProteinMPNN into %s ...", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", PROTEINMPNN_REPO, str(path)], check=True)
    if not (path / "protein_mpnn_run.py").exists():
        raise RuntimeError(f"Clone succeeded but {path}/protein_mpnn_run.py is missing.")
    return path


@dataclass(frozen=True)
class RedesignResult:
    target: ProteinTarget
    native_sequence_3letter: list[str]
    redesigned_sequence_3letter: list[str]
    hamming_distance: int
    mpnn_score: float | None


def _filter_pdb_for_mpnn(pdb_text: str, chain_letter: str) -> str:
    """Reduce a PDB to exactly the residues `_parse_chain` would return, and renumber
    them contiguously starting at 1.

    Matches `_parse_chain`'s filters: only ATOM records (drops HETATMs such as
    MSE/selenomethionine), model 1 only, alt_loc blank or "A", residue in
    STANDARD_AA_3LETTER, and restricted to `chain_letter`. Renumbering is
    required because ProteinMPNN uses PDB residue numbers as positional indices
    and inserts `X` into output sequences for any numbering gap — renumbering
    eliminates gaps so MPNN's output length matches `_parse_chain`'s residue list.
    """
    kept: list[str] = []
    in_model_1 = True
    residue_key_to_new_num: dict[tuple[str, int, str], int] = {}
    for line in pdb_text.splitlines():
        if line.startswith("MODEL "):
            in_model_1 = int(line[10:14].strip() or "1") == 1
            continue
        if line.startswith("ENDMDL"):
            in_model_1 = False
            continue
        if not in_model_1 or not line.startswith("ATOM  "):
            continue
        alt_loc = line[16].strip()
        res_name = line[17:20].strip()
        if line[21] != chain_letter or alt_loc not in ("", "A") or res_name not in STANDARD_AA_3LETTER:
            continue
        res_seq = int(line[22:26].strip())
        i_code = line[26]
        res_key = (line[21], res_seq, i_code.strip())
        if res_key not in residue_key_to_new_num:
            residue_key_to_new_num[res_key] = len(residue_key_to_new_num) + 1
        new_num = residue_key_to_new_num[res_key]
        # Rewrite columns 23-26 (1-indexed) with the new residue number and blank the insertion code.
        rewritten = line[:22] + f"{new_num:>4d}" + " " + line[27:]
        kept.append(rewritten)
    kept.append("END")
    return "\n".join(kept) + "\n"


def _run_proteinmpnn_one_target(
    target: ProteinTarget,
    *,
    proteinmpnn_dir: Path,
    temperature: float,
    seed: int,
    num_seq_per_target: int = 1,
) -> list[RedesignResult]:
    """Run ProteinMPNN on a single target; return `num_seq_per_target` redesigns.

    Implementation: write the fetched PDB to a temp file, invoke the upstream
    `protein_mpnn_run.py` CLI as a subprocess, parse the output FASTA.
    """
    pdb_text = _fetch_pdb_text(target.pdb_id, assembly=target.assembly)
    native_chain = _parse_chain(pdb_text, chain_id=target.chain_id)

    # ProteinMPNN `--chain_id` selects chains to *design*. If None, we
    # design the first chain actually present (matching `_parse_chain`'s
    # behavior). We identify that by re-reading the PDB's first chain.
    first_chain_letter = next(
        (line[21] for line in pdb_text.splitlines() if line.startswith("ATOM  ")),
        None,
    )
    if first_chain_letter is None:
        raise RuntimeError(f"No ATOM records in {target.pdb_id} — cannot run ProteinMPNN.")
    chain_for_design = (target.chain_id or first_chain_letter).strip()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pdb_path = tmp / f"{target.name}.pdb"
        pdb_path.write_text(_filter_pdb_for_mpnn(pdb_text, chain_for_design), encoding="utf-8")
        out_folder = tmp / "mpnn_out"
        out_folder.mkdir()

        cmd = [
            sys.executable,
            str(proteinmpnn_dir / "protein_mpnn_run.py"),
            "--pdb_path",
            str(pdb_path),
            "--pdb_path_chains",
            chain_for_design,
            "--out_folder",
            str(out_folder),
            "--num_seq_per_target",
            str(num_seq_per_target),
            "--sampling_temp",
            str(temperature),
            "--seed",
            str(seed),
            "--batch_size",
            "1",
        ]
        logger.info("Running ProteinMPNN for %s: %s", target.name, " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=proteinmpnn_dir)

        fasta_path = out_folder / "seqs" / f"{target.name}.fa"
        if not fasta_path.exists():
            raise RuntimeError(f"ProteinMPNN did not produce {fasta_path}")

        return _parse_mpnn_fasta(
            fasta_path.read_text(encoding="utf-8"),
            target=target,
            native_chain=native_chain,
        )


def _parse_mpnn_fasta(fasta_text: str, *, target: ProteinTarget, native_chain) -> list[RedesignResult]:
    """Parse a ProteinMPNN output FASTA. First record is native (T=0); subsequent
    are samples with header fields like `T=0.1, sample=1, score=0.82, seq_recovery=0.40`.
    """
    records: list[tuple[dict[str, str], str]] = []
    current_header: str | None = None
    current_seq: list[str] = []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            if current_header is not None:
                records.append((_parse_mpnn_header(current_header), "".join(current_seq)))
            current_header = line[1:].strip()
            current_seq = []
        elif line.strip():
            current_seq.append(line.strip())
    if current_header is not None:
        records.append((_parse_mpnn_header(current_header), "".join(current_seq)))

    # First record is the native (T=0.0 header); skip it.
    redesigns = records[1:]
    results: list[RedesignResult] = []
    for header, seq_one_letter in redesigns:
        if len(seq_one_letter) != len(native_chain.sequence):
            raise RuntimeError(
                f"ProteinMPNN returned seq of length {len(seq_one_letter)} but native chain "
                f"for {target.pdb_id} has length {len(native_chain.sequence)}"
            )
        seq_three = [_ONE_LETTER_TO_THREE[c] for c in seq_one_letter]
        for aa in seq_three:
            assert aa in STANDARD_AA_3LETTER, f"Unexpected AA '{aa}' from MPNN"
        hamming = sum(a != b for a, b in zip(native_chain.sequence, seq_three, strict=True))
        score_raw = header.get("score")
        mpnn_score = float(score_raw) if score_raw is not None else None
        results.append(
            RedesignResult(
                target=target,
                native_sequence_3letter=list(native_chain.sequence),
                redesigned_sequence_3letter=seq_three,
                hamming_distance=hamming,
                mpnn_score=mpnn_score,
            )
        )
    return results


_MPNN_HEADER_KV_RE = re.compile(r"(\w+)=([^,\s]+)")


def _parse_mpnn_header(header: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _MPNN_HEADER_KV_RE.finditer(header)}


def _write_jsonl(path: str, records: list[dict]) -> None:
    body = "\n".join(json.dumps(rec) for rec in records) + "\n"
    with fsspec.open(path, "w") as f:
        f.write(body)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSONL output path (local or gs://).")
    parser.add_argument(
        "--proteinmpnn-dir",
        default=os.environ.get("PROTEINMPNN_DIR"),
        help=f"Path to a ProteinMPNN clone. Default: env PROTEINMPNN_DIR, else clone to {DEFAULT_PROTEINMPNN_DIR}.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-redesigns", type=int, default=1, help="Redesigns per target.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--targets",
        nargs="*",
        help="Optional subset of target names (from TARGETS). Default: all targets.",
    )
    args = parser.parse_args(argv)

    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone ProteinMPNN; install git and retry.")

    proteinmpnn_dir = _ensure_proteinmpnn(args.proteinmpnn_dir)

    selected_targets = TARGETS
    if args.targets:
        want = set(args.targets)
        selected_targets = [t for t in TARGETS if t.name in want]
        missing = want - {t.name for t in selected_targets}
        if missing:
            raise ValueError(f"Unknown target names: {sorted(missing)}")

    records: list[dict] = []
    for target in selected_targets:
        logger.info("=== Redesigning %s ===", target.name)
        try:
            results = _run_proteinmpnn_one_target(
                target,
                proteinmpnn_dir=proteinmpnn_dir,
                temperature=args.temperature,
                seed=args.seed,
                num_seq_per_target=args.num_redesigns,
            )
        except Exception:
            logger.exception("Failed to redesign %s; skipping.", target.name)
            continue
        for idx, r in enumerate(results):
            logger.info(
                "  %s redesign %d: hamming=%d/%d mpnn_score=%s",
                target.name,
                idx,
                r.hamming_distance,
                len(r.native_sequence_3letter),
                f"{r.mpnn_score:.3f}" if r.mpnn_score is not None else "NA",
            )
            records.append(
                {
                    "target_label": target.name,
                    "pdb_id": target.pdb_id,
                    "chain_id": target.chain_id,
                    "assembly": target.assembly,
                    "method": "mpnn",
                    "temperature": args.temperature,
                    "redesign_idx": idx,
                    "sequence_3letter": r.redesigned_sequence_3letter,
                    "native_sequence_3letter": r.native_sequence_3letter,
                    "hamming_distance": r.hamming_distance,
                    "mpnn_score": r.mpnn_score,
                }
            )

    _write_jsonl(args.output, records)
    logger.info("Wrote %d redesign records to %s", len(records), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
