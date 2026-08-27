# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec==2026.1.0",
#   "gcsfs==2026.1.0",
#   "numpy==2.3.5",
#   "pandas==2.2.2",
# ]
# ///
"""Freeze a matched graft panel across proportional and cap-4 prefix states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_common_branches_20260824 as common_design,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as harsh_design,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_proportional_prefix_confirmation_20260826 as proportional_confirmation,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_PROPORTIONAL_CANDIDATE_WEIGHTS = (
    REFERENCE_OUTPUTS / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
)
DEFAULT_CAP4_CANDIDATE_WEIGHTS = (
    REFERENCE_OUTPUTS / "delphi_phase0_harsh_cap_candidates_20260825" / "training_candidate_weights.csv"
)
DEFAULT_PROPORTIONAL_PREFIXES = (
    REFERENCE_OUTPUTS / "delphi_phase0_proportional_prefix_confirmation_20260826" / "selected_prefixes.json"
)
DEFAULT_CAP4_PREFIXES = (
    REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap4_branches_20260825" / "confirmation_selected_prefixes.json"
)
DEFAULT_COMBINED_RESULTS = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_combined_results_20260826"
DEFAULT_COMBINED_SUMMARY = DEFAULT_COMBINED_RESULTS / "continuation_summary.csv"
DEFAULT_COMBINED_WEIGHTS = DEFAULT_COMBINED_RESULTS / "continuation_weights.csv"
DEFAULT_CANDIDATE_PREDICTIONS = (
    REFERENCE_OUTPUTS
    / "delphi_phase1_proportional_prefix_combined_fit_20260826"
    / "proportional_control"
    / "candidate_predictions.csv"
)
DEFAULT_FRONTIER_CONTRACT = (
    REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_20260825" / "validated_frontier_contract.json"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_cross_prefix_graft_v2_20260827"

PROPORTIONAL_PREFIX = "proportional_control"
CAP4_PREFIX = "cap4_shared_bounded_ensemble_kl0"
PREFIX_IDS = (PROPORTIONAL_PREFIX, CAP4_PREFIX)
PREFIX_SEEDS = (0, 1, 2)
DATA_SEEDS = (973_000, 973_001, 973_002)
IMPORTED_CONTINUATION_ID = "fit_079"
NOVEL_CONTINUATION_ID = "novel_rank0"
MIXTURE_BLOCK_SIZE = common_design.MIXTURE_BLOCK_SIZE
COUNT_PREFIX = proportional_confirmation.COUNT_PREFIX
CONTRACT_VERSION = "delphi_phase1_cross_prefix_graft_20260827_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proportional-candidate-weights", type=Path, default=DEFAULT_PROPORTIONAL_CANDIDATE_WEIGHTS)
    parser.add_argument("--cap4-candidate-weights", type=Path, default=DEFAULT_CAP4_CANDIDATE_WEIGHTS)
    parser.add_argument("--proportional-prefixes", type=Path, default=DEFAULT_PROPORTIONAL_PREFIXES)
    parser.add_argument("--cap4-prefixes", type=Path, default=DEFAULT_CAP4_PREFIXES)
    parser.add_argument("--combined-summary", type=Path, default=DEFAULT_COMBINED_SUMMARY)
    parser.add_argument("--combined-weights", type=Path, default=DEFAULT_COMBINED_WEIGHTS)
    parser.add_argument("--candidate-predictions", type=Path, default=DEFAULT_CANDIDATE_PREDICTIONS)
    parser.add_argument("--frontier-contract", type=Path, default=DEFAULT_FRONTIER_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different frozen artifact: {path}")
        return
    path.write_bytes(payload)


def candidate_center(path: Path, candidate_id: str, buckets: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(path)
    rows = frame[frame.candidate_id.eq(candidate_id)]
    if tuple(rows.bucket) != buckets:
        raise ValueError(f"Candidate bucket order changed for {candidate_id}")
    weights = rows.phase_0_weight.to_numpy(dtype=float)
    counts = common_design.runtime_counts(weights)
    if not np.array_equal(counts, rows.phase_0_count.to_numpy(dtype=int)):
        raise ValueError(f"Candidate {candidate_id} is not runtime exact")
    return counts / MIXTURE_BLOCK_SIZE


def prefix_manifest(path: Path, candidate_id: str, candidate_weights_sha256: str) -> dict[str, object]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    if payload.get("candidate_weights_sha256") != candidate_weights_sha256:
        raise ValueError(f"Prefix manifest references different candidate weights: {path}")
    prefixes = cast(list[dict[str, object]], payload.get("prefixes", []))
    observed = sorted(
        int(row["repeat_seed"])
        for row in prefixes
        if str(row.get("candidate_id", row.get("canonical_candidate_id"))) == candidate_id
    )
    if observed != list(PREFIX_SEEDS):
        raise ValueError(f"Prefix manifest does not contain seeds {PREFIX_SEEDS}: {path}")
    return payload


def imported_frontier_counts(
    combined_weights_path: Path,
    frontier_contract_path: Path,
    buckets: tuple[str, ...],
) -> np.ndarray:
    weights = pd.read_csv(combined_weights_path)
    rows = weights[weights.continuation_id.eq(IMPORTED_CONTINUATION_ID)]
    if tuple(rows.bucket) != buckets:
        raise ValueError("Imported frontier bucket order changed")
    counts = rows.phase_1_count.to_numpy(dtype=int)
    contract = cast(dict[str, object], json.loads(frontier_contract_path.read_text()))
    expected_counts = cast(dict[str, int], contract.get("runtime_counts", {}))
    if dict(zip(buckets, counts, strict=True)) != expected_counts:
        raise ValueError("Imported fit_079 no longer matches the validated cap-4 frontier contract")
    return counts


def novel_counts(
    predictions_path: Path,
    combined_summary_path: Path,
    combined_weights_path: Path,
    proportional_center: np.ndarray,
    buckets: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    previous = proportional_confirmation.previous_design_counts(combined_summary_path, combined_weights_path, buckets)
    selected = proportional_confirmation.selected_counts(
        predictions_path,
        buckets,
        proportional_center,
        candidate_count=1,
        previous_counts=previous,
    )
    row = selected.iloc[0]
    counts = np.asarray([int(row[f"{COUNT_PREFIX}{bucket}"]) for bucket in buckets], dtype=int)
    metadata: dict[str, object] = {
        "candidate_rank": int(row.candidate_rank),
        "source": str(row.source),
        "predicted_expected_endpoint_bpb": float(row.predicted_expected_endpoint_bpb),
        "stability_score_bpb": float(row.stability_score_bpb),
        "fold_fraction_predicting_improvement": float(row.fold_fraction_predicting_improvement),
        "hellinger_to_proportional_tied": float(row.hellinger_to_tied),
    }
    return counts, metadata


def append_design_row(
    summary_rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
    *,
    prefix_id: str,
    prefix_seed: int,
    data_seed: int,
    action_id: str,
    role: str,
    source: str,
    counts: np.ndarray,
    prefix_center: np.ndarray,
    buckets: tuple[str, ...],
    c0: np.ndarray,
    c1: np.ndarray,
) -> None:
    weights = counts / MIXTURE_BLOCK_SIZE
    phase0_exposure = prefix_center * c0
    phase1_exposure = weights * c1
    total_exposure = phase0_exposure + phase1_exposure
    if float(total_exposure.max()) > harsh_design.TOTAL_MATERIALIZED_EPOCH_CAP + 1e-12:
        raise ValueError(f"Cross-prefix action violates the epoch cap: {prefix_id}/{action_id}")
    data_position = DATA_SEEDS.index(data_seed)
    continuation_id = f"cross_{action_id}_seed{prefix_seed}_data{data_position}"
    summary_rows.append(
        {
            "prefix_candidate_id": prefix_id,
            "continuation_id": continuation_id,
            "role": role,
            "fit_budget": False,
            "prefix_repeat_seed": prefix_seed,
            "data_seed": data_seed,
            "source": source,
            "hellinger_to_tied": common_design.hellinger(weights, prefix_center),
            "max_phase0_materialized_epoch": float(phase0_exposure.max()),
            "max_phase1_materialized_epoch": float(phase1_exposure.max()),
            "max_total_materialized_epoch": float(total_exposure.max()),
        }
    )
    for position, bucket in enumerate(buckets):
        weight_rows.append(
            {
                "prefix_candidate_id": prefix_id,
                "continuation_id": continuation_id,
                "bucket": bucket,
                "phase_1_count": int(counts[position]),
                "phase_1_weight": float(weights[position]),
                "phase_1_materialized_epochs": float(phase1_exposure[position]),
                "total_materialized_epochs": float(total_exposure[position]),
            }
        )


def build_design(args: argparse.Namespace) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], dict[str, object]]:
    panel = common_design.load_canonical_panel_geometry()
    buckets = panel.buckets
    proportional_center = candidate_center(args.proportional_candidate_weights, PROPORTIONAL_PREFIX, buckets)
    cap4_center = candidate_center(args.cap4_candidate_weights, CAP4_PREFIX, buckets)
    proportional_weights_sha256 = file_sha256(args.proportional_candidate_weights)
    cap4_weights_sha256 = file_sha256(args.cap4_candidate_weights)
    prefix_manifest(args.proportional_prefixes, PROPORTIONAL_PREFIX, proportional_weights_sha256)
    prefix_manifest(args.cap4_prefixes, CAP4_PREFIX, cap4_weights_sha256)
    imported_counts = imported_frontier_counts(args.combined_weights, args.frontier_contract, buckets)
    selected_novel_counts, novel_metadata = novel_counts(
        args.candidate_predictions,
        args.combined_summary,
        args.combined_weights,
        proportional_center,
        buckets,
    )
    if np.array_equal(imported_counts, selected_novel_counts):
        raise ValueError("Imported and novel graft actions collapsed to the same runtime coordinate")

    centers = {PROPORTIONAL_PREFIX: proportional_center, CAP4_PREFIX: cap4_center}
    designs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for prefix_id, center in centers.items():
        summary_rows: list[dict[str, object]] = []
        weight_rows: list[dict[str, object]] = []
        actions = (
            ("tied", "paired_tied_control", "prefix_tied", common_design.runtime_counts(center)),
            ("fit079", "imported_frontier_graft", "validated_cap4_frontier:fit_079", imported_counts),
            ("novel", "novel_model_graft", "proportional_model:best_untrained", selected_novel_counts),
        )
        for prefix_seed in PREFIX_SEEDS:
            for data_seed in DATA_SEEDS:
                for action_id, role, source, counts in actions:
                    append_design_row(
                        summary_rows,
                        weight_rows,
                        prefix_id=prefix_id,
                        prefix_seed=prefix_seed,
                        data_seed=data_seed,
                        action_id=action_id,
                        role=role,
                        source=source,
                        counts=counts,
                        prefix_center=center,
                        buckets=buckets,
                        c0=panel.c0,
                        c1=panel.c1,
                    )
        summary = pd.DataFrame(summary_rows)
        weights = pd.DataFrame(weight_rows)
        if len(summary) != 27 or summary.continuation_id.nunique() != 27:
            raise ValueError(f"Cross-prefix allocation changed for {prefix_id}")
        designs[prefix_id] = (summary, weights)

    primary_estimand = " ".join(
        (
            "[Y(proportional, fit_079) - Y(proportional, novel)]",
            "-",
            "[Y(cap4, fit_079) - Y(cap4, novel)]",
        )
    )
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "outcome_informed": True,
        "research_question": (
            "Does a frontier-relevant phase-1 action retain its gain after grafting across prefix states?"
        ),
        "prefix_ids": list(PREFIX_IDS),
        "prefix_seeds": list(PREFIX_SEEDS),
        "data_seeds": list(DATA_SEEDS),
        "actions": ["prefix_tied", "validated_cap4_frontier_fit_079", "best_untrained_proportional_model_candidate"],
        "rows_per_prefix": 27,
        "total_rows": 54,
        "novel_candidate": novel_metadata,
        "inputs": {
            "proportional_candidate_weights_sha256": proportional_weights_sha256,
            "cap4_candidate_weights_sha256": cap4_weights_sha256,
            "proportional_prefixes_sha256": file_sha256(args.proportional_prefixes),
            "cap4_prefixes_sha256": file_sha256(args.cap4_prefixes),
            "combined_summary_sha256": file_sha256(args.combined_summary),
            "combined_weights_sha256": file_sha256(args.combined_weights),
            "candidate_predictions_sha256": file_sha256(args.candidate_predictions),
            "frontier_contract_sha256": file_sha256(args.frontier_contract),
        },
        "analysis": {
            "additive_null": "Y(prefix, action) = alpha_prefix + beta_action",
            "primary_estimand": primary_estimand,
            "primary_interpretation": (
                "The two shared actions are runtime-identical across prefixes, so zero is implied by the additive null. "
                "A nonzero contrast identifies an interaction for these prefixes and actions."
            ),
            "secondary_estimands": [
                "raw endpoint BPB",
                "within-prefix candidate-minus-prefix-specific-tied BPB",
                "tied-anchored difference-in-differences, descriptive only",
            ],
            "pairing": "Each prefix seed and fresh data seed block contains tied, fit_079, and novel actions.",
        },
        "selection_caveat": (
            "fit_079 was selected from cap-4 outcomes and novel from a proportional-prefix model. Fresh data seeds "
            "remove realized continuation-noise reuse, but prefix-asymmetric selection can inflate a nonzero "
            "shared-action interaction. The panel can refute strict additivity for these cells; it cannot establish a "
            "general transfer law."
        ),
    }
    return designs, manifest


def write_design(args: argparse.Namespace) -> dict[str, object]:
    designs, manifest = build_design(args)
    artifacts: dict[str, str] = {}
    for prefix_id, (summary, weights) in designs.items():
        prefix_dir = args.output_dir / prefix_id
        summary_path = prefix_dir / "continuation_summary.csv"
        weights_path = prefix_dir / "continuation_weights.csv"
        manifest_path = prefix_dir / "manifest.json"
        write_bytes_exact(summary_path, summary.to_csv(index=False).encode())
        write_bytes_exact(
            weights_path,
            weights.loc[:, list(harsh_design.WEIGHT_ARTIFACT_COLUMNS)].to_csv(index=False).encode(),
        )
        prefix_manifest_payload = {
            "contract_version": CONTRACT_VERSION,
            "selected_candidate_ids": [prefix_id],
            "rows": {"controls_per_prefix": 27, "fit_per_prefix": 0, "sealed_referees_per_prefix": 0, "total": 27},
            "role_counts_per_prefix": summary.role.value_counts().to_dict(),
            "analysis_contract": manifest["analysis"],
            "selection_caveat": manifest["selection_caveat"],
            "parent_manifest_inputs": manifest["inputs"],
            "artifacts": {
                summary_path.name: file_sha256(summary_path),
                weights_path.name: file_sha256(weights_path),
            },
        }
        write_bytes_exact(manifest_path, (json.dumps(prefix_manifest_payload, indent=2, sort_keys=True) + "\n").encode())
        for path in (summary_path, weights_path, manifest_path):
            artifacts[str(path.relative_to(args.output_dir))] = file_sha256(path)

    copied_inputs = {
        "proportional_candidate_weights.csv": args.proportional_candidate_weights,
        "cap4_candidate_weights.csv": args.cap4_candidate_weights,
        "proportional_selected_prefixes.json": args.proportional_prefixes,
        "cap4_selected_prefixes.json": args.cap4_prefixes,
    }
    for filename, source in copied_inputs.items():
        destination = args.output_dir / filename
        write_bytes_exact(destination, source.read_bytes())
        artifacts[filename] = file_sha256(destination)

    payload = {**manifest, "artifacts": artifacts}
    write_bytes_exact(args.output_dir / "manifest.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    return payload


def main() -> None:
    args = parse_args()
    payload = write_design(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
