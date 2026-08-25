# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Freeze and deduplicate the cap-4/cap-6 Delphi prefix candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_ROOT = SCRIPT_DIR / "reference_outputs"
DEFAULT_INPUTS = {
    4: REFERENCE_ROOT / "delphi_phase0_prefix_candidates_cap4_20260825",
    6: REFERENCE_ROOT / "delphi_phase0_prefix_candidates_cap6_20260825",
}
DEFAULT_OUTPUT_DIR = REFERENCE_ROOT / "delphi_phase0_harsh_cap_candidates_20260825"
MIXTURE_BLOCK_SIZE = 2_048
KL_LABELS = ("kl0", "kl0p05", "kl0p2", "kl0p5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap4-dir", type=Path, default=DEFAULT_INPUTS[4])
    parser.add_argument("--cap6-dir", type=Path, default=DEFAULT_INPUTS[6])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cap(directory: Path, cap: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest_path = directory / "manifest.json"
    summary_path = directory / "candidate_summary.csv"
    weights_path = directory / "candidate_weights.csv"
    manifest = json.loads(manifest_path.read_text())
    if float(manifest["phase_0_epoch_cap"]) != cap:
        raise ValueError(f"Unexpected cap in {manifest_path}: {manifest['phase_0_epoch_cap']}")
    expected = (
        *(f"shared_bounded_ensemble_{label}" for label in KL_LABELS),
        f"observed_cap{cap}_best",
        "proportional_control",
    )
    summary = pd.read_csv(summary_path)
    weights = pd.read_csv(weights_path)
    if tuple(summary.candidate_id) != expected:
        raise ValueError(f"Candidate identities changed for cap {cap}: {tuple(summary.candidate_id)}")
    if tuple(weights.candidate_id.drop_duplicates()) != expected:
        raise ValueError(f"Candidate weight order changed for cap {cap}")
    summary.insert(0, "cap_epochs", cap)
    weights.insert(0, "cap_epochs", cap)
    return (
        summary,
        weights,
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "summary_sha256": file_sha256(summary_path),
            "weights_sha256": file_sha256(weights_path),
        },
    )


def alias_id(cap: int, candidate_id: str) -> str:
    return f"cap{cap}_{candidate_id}"


def freeze_candidates(
    cap_inputs: dict[int, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    summaries = []
    weights = []
    sources = []
    for cap, directory in cap_inputs.items():
        summary, cap_weights, source = load_cap(directory, cap)
        summaries.append(summary)
        weights.append(cap_weights)
        sources.append({"cap_epochs": cap, **source})
    alias_summary = pd.concat(summaries, ignore_index=True)
    alias_weights = pd.concat(weights, ignore_index=True)
    alias_summary["alias_id"] = [
        alias_id(int(cap), candidate_id)
        for cap, candidate_id in zip(alias_summary.cap_epochs, alias_summary.candidate_id, strict=True)
    ]
    alias_summary["selection_eligible"] = alias_summary.candidate_id.isin(
        tuple(f"shared_bounded_ensemble_{label}" for label in KL_LABELS)
    )
    alias_weights["alias_id"] = [
        alias_id(int(cap), candidate_id)
        for cap, candidate_id in zip(alias_weights.cap_epochs, alias_weights.candidate_id, strict=True)
    ]

    count_vectors: dict[tuple[int, ...], str] = {}
    alias_to_canonical = {}
    training_frames = []
    for current_alias in alias_summary.alias_id:
        rows = alias_weights[alias_weights.alias_id.eq(current_alias)]
        counts = tuple(rows.phase_0_count.astype(int))
        if sum(counts) != MIXTURE_BLOCK_SIZE:
            raise ValueError(f"Runtime counts do not sum to {MIXTURE_BLOCK_SIZE}: {current_alias}")
        canonical = count_vectors.get(counts)
        if canonical is None:
            canonical = current_alias
            count_vectors[counts] = canonical
            canonical_rows = rows.copy()
            canonical_rows["candidate_id"] = canonical
            training_frames.append(canonical_rows)
        alias_to_canonical[current_alias] = canonical

    alias_summary["canonical_candidate_id"] = alias_summary.alias_id.map(alias_to_canonical)
    alias_weights["canonical_candidate_id"] = alias_weights.alias_id.map(alias_to_canonical)
    aliases = alias_summary[
        ["cap_epochs", "candidate_id", "alias_id", "canonical_candidate_id", "selection_eligible"]
    ].copy()
    training_weights = pd.concat(training_frames, ignore_index=True)
    training_weights = training_weights[
        [
            "candidate_id",
            "bucket",
            "phase_0_weight",
            "phase_0_count",
            "phase_0_materialized_epochs",
        ]
    ]
    return alias_summary, alias_weights, aliases, training_weights, sources


def main() -> None:
    args = parse_args()
    cap_inputs = {4: args.cap4_dir, 6: args.cap6_dir}
    alias_summary, alias_weights, aliases, training_weights, sources = freeze_candidates(cap_inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "alias_summary.csv": alias_summary,
        "alias_weights.csv": alias_weights,
        "candidate_aliases.csv": aliases,
        "training_candidate_weights.csv": training_weights,
    }
    hashes = {}
    for name, frame in outputs.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        hashes[name] = file_sha256(path)
    manifest = {
        "caps": list(cap_inputs),
        "kl_penalties": [0.0, 0.05, 0.2, 0.5],
        "selection_target": "mean exact-boundary Uncheatable BPB over paired seeds 0, 1, and 2",
        "selection_rule": (
            "Within each cap, select the eligible KL candidate with the lowest three-seed mean boundary "
            "Uncheatable BPB; break an exact tie by the lower KL penalty. Controls are diagnostic only."
        ),
        "mixture_block_size": MIXTURE_BLOCK_SIZE,
        "alias_count": len(aliases),
        "unique_training_candidate_count": training_weights.candidate_id.nunique(),
        "source_artifacts": sources,
        "output_sha256": hashes,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(aliases.to_string(index=False))
    print(f"\nUnique training candidates: {manifest['unique_training_candidate_count']}")


if __name__ == "__main__":
    main()
