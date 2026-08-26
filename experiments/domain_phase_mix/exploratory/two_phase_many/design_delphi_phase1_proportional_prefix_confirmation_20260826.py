# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec==2026.1.0",
#   "gcsfs==2026.1.0",
#   "numpy==2.3.5",
#   "pandas==2.2.2",
#   "scipy==1.17.0",
# ]
# ///
"""Freeze paired confirmation rows for proportional-prefix branch optima."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_COMBINED_FIT = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_combined_fit_20260826"
DEFAULT_COMBINED_RESULTS = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_combined_results_20260826"
DEFAULT_COMBINED_SUMMARY = DEFAULT_COMBINED_RESULTS / "continuation_summary.csv"
DEFAULT_COMBINED_WEIGHTS = DEFAULT_COMBINED_RESULTS / "continuation_weights.csv"
DEFAULT_CANDIDATE_PREDICTIONS = DEFAULT_COMBINED_FIT / "proportional_control" / "candidate_predictions.csv"
DEFAULT_MODEL_CONTRACT = DEFAULT_COMBINED_FIT / "frozen_model_contract.json"
DEFAULT_COVERAGE = DEFAULT_COMBINED_RESULTS / "coverage.json"
DEFAULT_CANDIDATE_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
DEFAULT_SELECTED_PREFIXES = (
    REFERENCE_OUTPUTS / "delphi_phase0_proportional_prefix_confirmation_20260826" / "selected_prefixes.json"
)
DEFAULT_WAVE2_CONTRACT = (
    REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave2_contract_20260826" / "contract.json"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_confirmation_20260826"
TARGET_PREFIX = "proportional_control"
COUNT_PREFIX = "phase_1_count::"
EXPECTED_FIT_ROWS = 160
MINIMUM_PAIRWISE_HELLINGER = 0.05
MINIMUM_HELLINGER_TO_TIED = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-predictions", type=Path, default=DEFAULT_CANDIDATE_PREDICTIONS)
    parser.add_argument("--model-contract", type=Path, default=DEFAULT_MODEL_CONTRACT)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--combined-summary", type=Path, default=DEFAULT_COMBINED_SUMMARY)
    parser.add_argument("--combined-weights", type=Path, default=DEFAULT_COMBINED_WEIGHTS)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--selected-prefixes", type=Path, default=DEFAULT_SELECTED_PREFIXES)
    parser.add_argument("--wave2-contract", type=Path, default=DEFAULT_WAVE2_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different frozen artifact: {path}")
        return
    path.write_bytes(payload)


def confirmation_contract(path: Path) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    contract = cast(dict[str, object], json.loads(path.read_text()))
    confirmation = cast(dict[str, object], contract.get("confirmation", {}))
    prefix_seeds = tuple(int(seed) for seed in cast(list[int], confirmation.get("prefix_seeds", [])))
    data_seeds = tuple(int(seed) for seed in cast(list[int], confirmation.get("data_seeds", [])))
    candidate_count = int(confirmation.get("candidate_count", -1))
    required = {
        "outside_fit_budget": True,
        "crossed_prefix_data_blocks": 9,
        "minimum_pairwise_hellinger": MINIMUM_PAIRWISE_HELLINGER,
        "minimum_hellinger_to_tied": MINIMUM_HELLINGER_TO_TIED,
        "measured_fit_coordinates_excluded": True,
        "previously_trained_coordinates_excluded": True,
        "primary_candidate_index": 0,
    }
    if prefix_seeds != (0, 1, 2) or data_seeds != (972_000, 972_001, 972_002) or candidate_count != 3:
        raise ValueError("Proportional-prefix confirmation allocation changed")
    for key, value in required.items():
        if confirmation.get(key) != value:
            raise ValueError(f"Proportional-prefix confirmation contract changed: {key}")
    return prefix_seeds, data_seeds, candidate_count


def validate_model_freeze(
    contract_path: Path,
    coverage_path: Path,
    predictions_path: Path,
    combined_summary_path: Path,
    combined_weights_path: Path,
) -> None:
    contract = cast(dict[str, object], json.loads(contract_path.read_text()))
    coverage = cast(dict[str, object], json.loads(coverage_path.read_text()))
    if coverage.get("status") != "complete" or coverage.get("referee_outcomes_opened") is not False:
        raise ValueError("Confirmation requires a complete fit with unopened referee outcomes")
    if contract.get("expected_fit_rows") != EXPECTED_FIT_ROWS:
        raise ValueError("Confirmation requires the combined 160-row fit")
    seal = cast(dict[str, object], contract.get("seal", {}))
    if seal.get("referee_outcomes_present_in_fit_input") is not False:
        raise ValueError("Referee outcomes leaked into the confirmation model")
    inputs = cast(dict[str, object], contract.get("inputs", {}))
    if inputs.get("coverage_sha256") != file_sha256(coverage_path):
        raise ValueError("Frozen model contract references different coverage")
    if inputs.get("design_summary_sha256") != file_sha256(combined_summary_path):
        raise ValueError("Frozen model contract references a different combined design summary")
    if inputs.get("design_weights_sha256") != file_sha256(combined_weights_path):
        raise ValueError("Frozen model contract references different combined design weights")
    frozen = cast(dict[str, object], contract.get("frozen_candidates", {}))
    candidate = cast(dict[str, object], frozen.get(TARGET_PREFIX, {}))
    if candidate.get("eligible_for_measurement") is not True:
        raise ValueError("Combined response model is not eligible for confirmation")
    artifacts = cast(dict[str, object], contract.get("artifacts", {}))
    expected_prediction_sha256 = artifacts.get(f"{TARGET_PREFIX}/candidate_predictions.csv")
    if expected_prediction_sha256 != file_sha256(predictions_path):
        raise ValueError("Candidate predictions are not bound to the frozen model contract")


def bucket_order(candidate_weights_path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    frame = pd.read_csv(candidate_weights_path)
    rows = frame[frame.candidate_id.eq(TARGET_PREFIX)]
    if len(rows) != 39:
        raise ValueError(f"Expected 39 proportional-prefix buckets, got {len(rows)}")
    return tuple(rows.bucket), rows.phase_0_weight.to_numpy(dtype=float)


def previous_design_counts(
    summary_path: Path,
    weights_path: Path,
    buckets: tuple[str, ...],
) -> set[tuple[int, ...]]:
    summary = pd.read_csv(summary_path)
    weights = pd.read_csv(weights_path)
    fit_ids = set(summary[summary.fit_budget.astype(bool)].continuation_id)
    if len(fit_ids) != EXPECTED_FIT_ROWS:
        raise ValueError(f"Expected {EXPECTED_FIT_ROWS} measured fit coordinates, got {len(fit_ids)}")
    observed_buckets = tuple(weights.bucket.drop_duplicates())
    if observed_buckets != buckets:
        raise ValueError("Combined design bucket order changed")
    fit_counts = {
        tuple(group.phase_1_count.to_numpy(dtype=int))
        for continuation_id, group in weights.groupby("continuation_id", sort=False)
        if continuation_id in fit_ids
    }
    if len(fit_counts) != EXPECTED_FIT_ROWS:
        raise ValueError("Combined fit contains duplicate runtime coordinates")
    design_ids = set(summary.continuation_id)
    counts = {
        tuple(group.phase_1_count.to_numpy(dtype=int))
        for continuation_id, group in weights.groupby("continuation_id", sort=False)
        if continuation_id in design_ids
    }
    if set(weights.continuation_id) != design_ids:
        raise ValueError("Combined design summary and weights disagree")
    return counts


def selected_counts(
    predictions_path: Path,
    buckets: tuple[str, ...],
    center: np.ndarray,
    candidate_count: int,
    previous_counts: set[tuple[int, ...]],
) -> pd.DataFrame:
    frame = pd.read_csv(predictions_path)
    count_columns = [f"{COUNT_PREFIX}{bucket}" for bucket in buckets]
    required = {
        "stability_score_bpb",
        "fold_fraction_predicting_improvement",
        "hellinger_to_tied",
        "source",
        *count_columns,
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Candidate predictions are missing columns: {sorted(required - set(frame.columns))}")
    eligible = frame[frame.hellinger_to_tied.ge(MINIMUM_HELLINGER_TO_TIED)].copy()
    eligible = eligible.sort_values(
        [
            "stability_score_bpb",
            "fold_fraction_predicting_improvement",
            "hellinger_to_tied",
            *count_columns,
        ],
        ascending=[True, False, True, *([True] * len(count_columns))],
    )
    unique_rows = []
    seen: set[tuple[int, ...]] = set()
    for _, row in eligible.iterrows():
        counts = tuple(int(row[column]) for column in count_columns)
        if counts in seen or counts in previous_counts:
            continue
        if sum(counts) != design.MIXTURE_BLOCK_SIZE or min(counts) < 0:
            raise ValueError("Predicted confirmation coordinate is off the runtime lattice")
        weights = np.asarray(counts, dtype=float) / design.MIXTURE_BLOCK_SIZE
        actual_hellinger = float(design.hellinger(weights[None, :], center[None, :])[0])
        if actual_hellinger < MINIMUM_HELLINGER_TO_TIED:
            continue
        if not np.isclose(actual_hellinger, float(row.hellinger_to_tied), atol=1e-12, rtol=0.0):
            raise ValueError("Candidate prediction Hellinger distance does not match its runtime counts")
        selected_weights = [
            np.asarray(selected_counts, dtype=float) / design.MIXTURE_BLOCK_SIZE for selected_counts in seen
        ]
        if selected_weights and any(
            float(design.hellinger(weights[None, :], previous[None, :])[0]) < MINIMUM_PAIRWISE_HELLINGER
            for previous in selected_weights
        ):
            continue
        seen.add(counts)
        unique_rows.append(row)
        if len(unique_rows) == candidate_count:
            break
    if len(unique_rows) != candidate_count:
        raise ValueError(f"Could select only {len(unique_rows)} distinct predicted confirmation coordinates")
    return pd.DataFrame(unique_rows).reset_index(drop=True)


def build_design(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    prefix_seeds, data_seeds, candidate_count = confirmation_contract(args.wave2_contract)
    validate_model_freeze(
        args.model_contract,
        args.coverage,
        args.candidate_predictions,
        args.combined_summary,
        args.combined_weights,
    )
    selected_prefixes = cast(dict[str, object], json.loads(args.selected_prefixes.read_text()))
    if selected_prefixes.get("candidate_weights_sha256") != file_sha256(args.candidate_weights):
        raise ValueError("Confirmation prefix manifest references different candidate weights")
    observed_prefix_seeds = tuple(
        sorted(int(row["repeat_seed"]) for row in cast(list[dict[str, object]], selected_prefixes.get("prefixes", [])))
    )
    if observed_prefix_seeds != prefix_seeds:
        raise ValueError("Confirmation prefix manifest does not contain seeds 0, 1, and 2")
    buckets, center = bucket_order(args.candidate_weights)
    previous_counts = previous_design_counts(args.combined_summary, args.combined_weights, buckets)
    selected = selected_counts(args.candidate_predictions, buckets, center, candidate_count, previous_counts)
    panel = design.common_design.load_canonical_panel_geometry()
    if panel.buckets != buckets:
        raise ValueError("Confirmation bucket order changed")
    phase0_exposure = center * panel.c0

    summary_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    def append_row(
        continuation_id: str,
        counts: np.ndarray,
        role: str,
        prefix_seed: int,
        data_seed: int,
        source: str,
    ) -> None:
        weights = counts / design.MIXTURE_BLOCK_SIZE
        phase1_exposure = weights * panel.c1
        if float((phase0_exposure + phase1_exposure).max()) > design.TOTAL_MATERIALIZED_EPOCH_CAP + 1e-12:
            raise ValueError(f"Confirmation coordinate violates the epoch cap: {continuation_id}")
        summary_rows.append(
            {
                "prefix_candidate_id": TARGET_PREFIX,
                "continuation_id": continuation_id,
                "role": role,
                "fit_budget": False,
                "prefix_repeat_seed": prefix_seed,
                "data_seed": data_seed,
                "source": source,
                "hellinger_to_tied": float(design.hellinger(weights[None, :], center[None, :])[0]),
                "max_phase0_materialized_epoch": float(phase0_exposure.max()),
                "max_phase1_materialized_epoch": float(phase1_exposure.max()),
                "max_total_materialized_epoch": float((phase0_exposure + phase1_exposure).max()),
            }
        )
        for position, bucket in enumerate(buckets):
            weight_rows.append(
                {
                    "prefix_candidate_id": TARGET_PREFIX,
                    "continuation_id": continuation_id,
                    "bucket": bucket,
                    "phase_1_count": int(counts[position]),
                    "phase_1_weight": float(weights[position]),
                    "phase_1_materialized_epochs": float(phase1_exposure[position]),
                    "total_materialized_epochs": float(phase0_exposure[position] + phase1_exposure[position]),
                }
            )

    tied_counts = design.common_design.runtime_counts(center)
    for prefix_seed in prefix_seeds:
        for data_position, data_seed in enumerate(data_seeds):
            append_row(
                f"confirm_tied_seed{prefix_seed}_data{data_position}",
                tied_counts,
                "paired_tied_confirmation",
                prefix_seed,
                data_seed,
                "tied",
            )
    count_columns = [f"{COUNT_PREFIX}{bucket}" for bucket in buckets]
    for candidate_position, (_, row) in enumerate(selected.iterrows()):
        counts = np.asarray([int(row[column]) for column in count_columns], dtype=int)
        for prefix_seed in prefix_seeds:
            for data_position, data_seed in enumerate(data_seeds):
                append_row(
                    f"confirm_candidate{candidate_position}_seed{prefix_seed}_data{data_position}",
                    counts,
                    "predicted_branch_confirmation",
                    prefix_seed,
                    data_seed,
                    f"spatial_crossfit_rank:{candidate_position}",
                )

    summary = pd.DataFrame(summary_rows)
    weights = pd.DataFrame(weight_rows)
    expected_rows = len(prefix_seeds) * len(data_seeds) * (candidate_count + 1)
    if len(summary) != expected_rows or summary.continuation_id.nunique() != expected_rows:
        raise ValueError("Confirmation row allocation changed")
    manifest: dict[str, object] = {
        "contract_version": "delphi_phase1_proportional_prefix_confirmation_20260826_v1",
        "selected_candidate_ids": [TARGET_PREFIX],
        "rows": {
            "controls_per_prefix": expected_rows,
            "fit_per_prefix": 0,
            "sealed_referees_per_prefix": 0,
            "total": expected_rows,
        },
        "role_counts_per_prefix": summary.role.value_counts().to_dict(),
        "inputs": {
            "candidate_predictions_sha256": file_sha256(args.candidate_predictions),
            "model_contract_sha256": file_sha256(args.model_contract),
            "coverage_sha256": file_sha256(args.coverage),
            "combined_summary_sha256": file_sha256(args.combined_summary),
            "combined_weights_sha256": file_sha256(args.combined_weights),
            "candidate_weights_sha256": file_sha256(args.candidate_weights),
            "selected_prefixes_sha256": file_sha256(args.selected_prefixes),
            "wave2_contract_sha256": file_sha256(args.wave2_contract),
        },
        "selected_predictions": (
            selected[
                [
                    "stability_score_bpb",
                    "fold_fraction_predicting_improvement",
                    "hellinger_to_tied",
                    *count_columns,
                ]
            ].to_dict(orient="records")
        ),
        "pairing": {
            "prefix_seeds": list(prefix_seeds),
            "data_seeds": list(data_seeds),
            "crossed_prefix_data_blocks": len(prefix_seeds) * len(data_seeds),
            "shared_tied_control_per_prefix_data_block": True,
            "minimum_pairwise_hellinger": MINIMUM_PAIRWISE_HELLINGER,
            "minimum_hellinger_to_tied": MINIMUM_HELLINGER_TO_TIED,
            "measured_fit_coordinates_excluded": True,
            "previously_trained_coordinates_excluded": True,
        },
    }
    return summary, weights, manifest


def main() -> None:
    args = parse_args()
    summary, weights, manifest = build_design(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "continuation_summary.csv"
    weights_path = args.output_dir / "continuation_weights.csv"
    write_bytes_exact(summary_path, summary.to_csv(index=False).encode())
    write_bytes_exact(
        weights_path,
        weights.loc[:, list(design.WEIGHT_ARTIFACT_COLUMNS)].to_csv(index=False).encode(),
    )
    payload = {
        **manifest,
        "artifacts": {
            summary_path.name: file_sha256(summary_path),
            weights_path.name: file_sha256(weights_path),
        },
    }
    write_bytes_exact(args.output_dir / "manifest.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
