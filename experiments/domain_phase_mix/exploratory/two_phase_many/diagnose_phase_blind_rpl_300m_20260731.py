# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Diagnose whether repaired RPL's temporal block adds signal or variance.

The diagnostic refits the same retained-power-law aggregate response after
forcing retention to zero, the late multiplier to one, and all explicit phase
features out of the design. It uses the frozen expanded-300M rows and
correspondence folds, then compares the restriction with full repaired RPL,
parent RPL, and hierarchical phase replay under the corrected paired bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as expanded,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_repaired_rpl_300m_20260731 as full_candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    bootstrap_expanded_300m_pareto_baseline_20260731 as bootstrap,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_estimator_repair_20260731 as repaired,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_phase_blind_diagnostic_20260731"
BASELINE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731"
FULL_CANDIDATE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_20260731"
MODEL_ID = "retained_power_law_phase_blind"
COMPARISON_MODELS = (
    MODEL_ID,
    full_candidate.MODEL_ID,
    "retained_power_law",
    "hierarchical_phase_replay",
)
PROTOCOL_VERSION = "repaired-rpl-phase-blind-diagnostic-v1"
BOOTSTRAP_DRAWS = 4_000
BOOTSTRAP_SEED = 731_312
AGGREGATE_MATCH_TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=",".join(baseline.TARGETS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def protocol_payload() -> dict[str, Any]:
    baseline_protocol = json.loads((BASELINE_OUTPUT_DIR / "protocol.json").read_text())
    full_candidate_protocol = json.loads((FULL_CANDIDATE_OUTPUT_DIR / "protocol.json").read_text())
    sources = (
        Path(__file__),
        Path(repaired.__file__),
        Path(bootstrap.__file__),
    )
    payload = {
        "version": PROTOCOL_VERSION,
        "model": MODEL_ID,
        "targets": list(baseline.TARGETS),
        "baseline_protocol_hash": baseline_protocol["protocol_hash"],
        "full_candidate_protocol_hash": full_candidate_protocol["protocol_hash"],
        "restriction": {
            "retention": 0.0,
            "late_multiplier": 1.0,
            "ordering_channel": False,
            "explicit_phase_features": False,
        },
        "outer_splits": baseline.OUTER_SPLITS,
        "inner_splits": baseline.INNER_SPLITS,
        "selection": {
            "core_rmse_ratio_limit": repaired.CORE_RMSE_RATIO_LIMIT,
            "regret_at_1_slack": repaired.REGRET_AT_1_SLACK,
            "lower_tail_fraction": repaired.LOWER_TAIL_FRACTION,
            "lower_tail_min_count": repaired.LOWER_TAIL_MIN_COUNT,
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "smooth_metric_unit": "phase_correspondence_key within outer fold",
            "regret_unit": "outer fold with fixed candidate population",
            "tie_tolerance": bootstrap.TIE_TOLERANCE,
        },
        "aggregate_match_tolerance": AGGREGATE_MATCH_TOLERANCE,
        "source_hashes": {str(path.relative_to(REPO_ROOT)): baseline.file_hash(path) for path in sources},
    }
    encoded = json.dumps(baseline.json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def cell_dir(output_dir: Path, target: str) -> Path:
    return output_dir / "cells" / target / MODEL_ID


def cell_complete(path: Path, protocol_hash: str) -> bool:
    required = (
        path / "complete.json",
        path / "predictions.csv",
        path / "metrics.json",
        path / "pair_metrics.json",
        path / "pair_predictions.csv",
        path / "fold_selections.json",
        path / "parameter_diagnostics.csv",
        path / "fold_coefficients.csv",
        path / "coefficient_stability.csv",
    )
    if any(not item.exists() for item in required):
        return False
    marker = json.loads((path / "complete.json").read_text())
    return marker.get("protocol_hash") == protocol_hash


def assert_exact_aggregate_pairs(dataset: expanded.Dataset) -> float:
    tied, asymmetric, _keys = baseline.pair_indices(dataset)
    phase_0_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    aggregate = phase_0_fraction * dataset.weights[:, 0, :] + (1.0 - phase_0_fraction) * dataset.weights[:, 1, :]
    maximum_error = float(np.max(np.abs(aggregate[tied] - aggregate[asymmetric])))
    if maximum_error > AGGREGATE_MATCH_TOLERANCE:
        raise ValueError(
            f"exact-pair aggregate mismatch {maximum_error:.3e} exceeds " f"{AGGREGATE_MATCH_TOLERANCE:.3e}"
        )
    return maximum_error


def fit_model(
    dataset: baseline.pooled.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    family_index: np.ndarray,
    workers: int,
) -> repaired.Fitted:
    return repaired.fit(
        dataset.weights,
        dataset.y,
        baseline.retained_geometry(dataset, family_index),
        folds,
        full_candidate.selection_context(dataset),
        workers=workers,
        phase_blind=True,
    )


def parameter_diagnostics(model: repaired.Fitted) -> dict[str, int | float | str]:
    aggregate_nominal = len(model.aggregate_coefficients)
    aggregate_active = baseline.active_count(model.aggregate_coefficients)
    selected_shape_scalars = baseline.numeric_scalar_count(model.shape)
    shape_search_dimensions = 3
    return {
        "nominal_parameter_count": 1 + selected_shape_scalars + aggregate_nominal,
        "selected_shape_scalar_count": selected_shape_scalars,
        "shape_search_dimensions": shape_search_dimensions,
        "aggregate_linear_parameter_count": aggregate_nominal,
        "phase_linear_parameter_count": 0,
        "active_aggregate_parameter_count": aggregate_active,
        "active_phase_parameter_count": 0,
        "effective_df_active_set_proxy": 1 + shape_search_dimensions + aggregate_active,
        "effective_df_note": "intercept + three selected aggregate-shape dimensions + active aggregate coefficients",
    }


def enforce_pair_invariance(dataset: expanded.Dataset, predicted: np.ndarray) -> float:
    """Remove numerical roundoff from algebraically identical pair predictions."""

    tied, asymmetric, _keys = baseline.pair_indices(dataset)
    maximum_error = float(np.max(np.abs(predicted[tied] - predicted[asymmetric])))
    if maximum_error > AGGREGATE_MATCH_TOLERANCE:
        raise ValueError(
            f"phase-blind pair prediction mismatch {maximum_error:.3e} exceeds " f"{AGGREGATE_MATCH_TOLERANCE:.3e}"
        )
    pair_mean = 0.5 * (predicted[tied] + predicted[asymmetric])
    predicted[tied] = pair_mean
    predicted[asymmetric] = pair_mean
    return maximum_error


def coefficient_rows(model: repaired.Fitted, fold: int | str) -> list[dict[str, int | float | str]]:
    names = repaired.feature_names(model.geometry, model.shape, include_phase=False)
    if len(names) != len(model.aggregate_coefficients):
        raise ValueError("aggregate coefficient names do not match the phase-blind design")
    return [
        {
            "fold": fold,
            "feature": name,
            "coefficient": float(coefficient),
            "channel": "aggregate",
        }
        for name, coefficient in zip(names, model.aggregate_coefficients, strict=True)
    ]


def run_cell(
    output_dir: Path,
    protocol: dict[str, Any],
    target: str,
    dataset: expanded.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    workers: int,
    force: bool,
) -> None:
    path = cell_dir(output_dir, target)
    if not force and cell_complete(path, str(protocol["protocol_hash"])):
        print(f"skip complete {target}/{MODEL_ID}", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    maximum_pair_aggregate_error = assert_exact_aggregate_pairs(dataset)
    pooled_dataset = baseline.as_pooled(dataset)
    predicted = np.full(dataset.n, np.nan, dtype=float)
    outer_fold = np.full(dataset.n, -1, dtype=int)
    nearest_policy_tv = np.full(dataset.n, np.nan, dtype=float)
    nearest_aggregate_tv = np.full(dataset.n, np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, int | float | str]] = []
    coefficients: list[dict[str, int | float | str]] = []

    for fold_id, (train, test) in enumerate(folds):
        print(
            f"{target}/{MODEL_ID}: outer fold {fold_id + 1}/{len(folds)} " f"({len(train)} train, {len(test)} test)",
            flush=True,
        )
        local = baseline.subset_dataset(pooled_dataset, train, f"outer{fold_id}_train")
        inner = baseline.correspondence_folds(
            local.frame,
            baseline.INNER_SEED_BASE + fold_id,
            baseline.INNER_SPLITS,
        )
        fitted = fit_model(local, inner, dataset.family_index, workers)
        predicted[test] = fitted.predict(dataset.weights[test])
        outer_fold[test] = fold_id
        nearest_policy_tv[test], nearest_aggregate_tv[test] = baseline.test_support_columns(
            dataset,
            train,
            test,
        )
        selections.append(
            {
                "outer_fold": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                "selection": asdict(fitted.selection),
                "shape": asdict(fitted.shape),
                "ridge": fitted.ridge,
            }
        )
        parameter_rows.append({"outer_fold": fold_id, **parameter_diagnostics(fitted)})
        coefficients.extend(coefficient_rows(fitted, fold_id))

    if not np.isfinite(predicted).all() or np.any(outer_fold < 0):
        raise RuntimeError(f"incomplete OOF predictions for {target}/{MODEL_ID}")
    maximum_pair_prediction_error = enforce_pair_invariance(dataset, predicted)
    tied = expanded.replay_control.tied_rows(dataset.weights)
    phase_0_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    aggregate = phase_0_fraction * dataset.weights[:, 0, :] + (1.0 - phase_0_fraction) * dataset.weights[:, 1, :]
    phase_tv = 0.5 * np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]).sum(axis=1)
    pd.DataFrame(
        {
            "row_index": np.arange(dataset.n),
            "run_name": dataset.frame["run_name"].astype(str),
            "phase_correspondence_key": dataset.frame["phase_correspondence_key"].astype(str),
            "policy_family": dataset.frame["policy_family"].astype(str),
            "physical_tied": tied,
            "outer_fold": outer_fold,
            "observed": dataset.y,
            "predicted": predicted,
            "residual": predicted - dataset.y,
            "optimism": dataset.y - predicted,
            "phase_tv": phase_tv,
            "aggregate_hhi": np.sum(aggregate**2, axis=1),
            "nearest_train_policy_tv": nearest_policy_tv,
            "nearest_train_aggregate_tv": nearest_aggregate_tv,
        }
    ).to_csv(path / "predictions.csv", index=False)
    metrics = {
        "target": target,
        "model": MODEL_ID,
        "reference_only": True,
        **baseline.metric_summary(dataset, predicted, folds),
    }
    pair_metrics, pair_predictions = baseline.pair_summary(dataset, predicted)
    baseline.write_json(path / "metrics.json", metrics)
    baseline.write_json(path / "pair_metrics.json", pair_metrics)
    pair_predictions.to_csv(path / "pair_predictions.csv", index=False)
    baseline.write_json(path / "fold_selections.json", selections)
    pd.DataFrame(parameter_rows).to_csv(path / "parameter_diagnostics.csv", index=False)

    print(f"{target}/{MODEL_ID}: fitting full 520-row restriction", flush=True)
    full_inner = baseline.correspondence_folds(
        pooled_dataset.frame,
        baseline.FULL_FIT_SEED,
        baseline.INNER_SPLITS,
    )
    full_fit = fit_model(pooled_dataset, full_inner, dataset.family_index, workers)
    coefficients.extend(coefficient_rows(full_fit, "full"))
    coefficient_frame = pd.DataFrame(coefficients)
    coefficient_frame.to_csv(path / "fold_coefficients.csv", index=False)
    full_candidate.coefficient_stability(coefficient_frame).to_csv(
        path / "coefficient_stability.csv",
        index=False,
    )
    baseline.write_json(
        path / "full_fit.json",
        {
            "shape": asdict(full_fit.shape),
            "ridge": full_fit.ridge,
            "selection": asdict(full_fit.selection),
            "parameter_diagnostics": parameter_diagnostics(full_fit),
        },
    )
    baseline.write_json(
        path / "complete.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "target": target,
            "model": MODEL_ID,
            "maximum_pair_aggregate_error": maximum_pair_aggregate_error,
            "maximum_pair_prediction_error_before_roundoff_correction": maximum_pair_prediction_error,
        },
    )
    print(f"completed {target}/{MODEL_ID}", flush=True)


def aligned_predictions(reference: pd.DataFrame, path: Path, model_id: str) -> np.ndarray:
    candidate = pd.read_csv(path).sort_values("row_index").reset_index(drop=True)
    bootstrap._aligned_frame(reference, candidate, model_id)
    return candidate["predicted"].to_numpy(dtype=float)


def comparison_data(output_dir: Path, target: str) -> bootstrap.TargetData:
    base = bootstrap.load_target(
        BASELINE_OUTPUT_DIR,
        json.loads((BASELINE_OUTPUT_DIR / "protocol.json").read_text())["protocol_hash"],
        target,
    )
    baseline_indices = {model_id: index for index, model_id in enumerate(base.model_ids)}
    predictions = [
        aligned_predictions(
            base.frame,
            cell_dir(output_dir, target) / "predictions.csv",
            MODEL_ID,
        ),
        aligned_predictions(
            base.frame,
            FULL_CANDIDATE_OUTPUT_DIR / "cells" / target / full_candidate.MODEL_ID / "predictions.csv",
            full_candidate.MODEL_ID,
        ),
        base.predictions[baseline_indices["retained_power_law"]],
        base.predictions[baseline_indices["hierarchical_phase_replay"]],
    ]
    return replace(
        base,
        model_ids=COMPARISON_MODELS,
        predictions=np.stack(predictions),
    )


def collect_results(output_dir: Path, protocol: dict[str, Any]) -> None:
    metrics = []
    pair_metrics = []
    bootstrap_summaries = []
    bootstrap_pairwise = []
    complete_targets = []
    for target_index, target in enumerate(baseline.TARGETS):
        path = cell_dir(output_dir, target)
        if not cell_complete(path, str(protocol["protocol_hash"])):
            continue
        complete_targets.append(target)
        metrics.append(json.loads((path / "metrics.json").read_text()))
        pair_metrics.append(
            {
                "target": target,
                "model": MODEL_ID,
                **json.loads((path / "pair_metrics.json").read_text()),
            }
        )
        data = comparison_data(output_dir, target)
        summary, pairwise = bootstrap.bootstrap_target(
            data,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED + target_index,
        )
        bootstrap_summaries.append(summary)
        bootstrap_pairwise.append(pairwise)

    metric_frame = pd.DataFrame(metrics)
    pair_frame = pd.DataFrame(pair_metrics)
    summary_frame = pd.concat(bootstrap_summaries, ignore_index=True) if bootstrap_summaries else pd.DataFrame()
    pairwise_frame = pd.concat(bootstrap_pairwise, ignore_index=True) if bootstrap_pairwise else pd.DataFrame()
    metric_frame.to_csv(output_dir / "phase_blind_metrics.csv", index=False)
    pair_frame.to_csv(output_dir / "phase_blind_pair_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "comparison_bootstrap_summary.csv", index=False)
    pairwise_frame.to_csv(output_dir / "comparison_bootstrap_pairwise.csv", index=False)
    baseline.write_json(
        output_dir / "status.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "complete_targets": complete_targets,
            "complete": len(complete_targets) == len(baseline.TARGETS),
        },
    )

    report = [
        "# Phase-Blind Retained-Power-Law Diagnostic",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Complete targets: {len(complete_targets)}/{len(baseline.TARGETS)}",
        "- This is a refitted diagnostic restriction, not a promoted candidate.",
        "- Retention is zero, late multiplier is one, and all explicit phase features are absent.",
        "- Smooth metrics resample correspondence groups; Regret@k resamples outer folds over fixed candidate sets.",
        "",
    ]
    if not metric_frame.empty:
        report.extend(
            [
                "## Correspondence-Grouped OOF",
                "",
                metric_frame[
                    [
                        "target",
                        "all_rmse",
                        "tied_rmse",
                        "asymmetric_rmse",
                        "asymmetric_regret_at_1",
                        "all_low_tail_rmse",
                        "all_calibration_slope",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    if not pair_frame.empty:
        report.extend(
            [
                "## Exact Aggregate-Matched Contrasts",
                "",
                pair_frame[["target", "delta_rmse", "delta_spearman", "delta_bias", "sign_accuracy"]].to_markdown(
                    index=False, floatfmt=".6f"
                ),
                "",
            ]
        )
    if not pairwise_frame.empty:
        selected = pairwise_frame.loc[
            pairwise_frame["candidate"].eq(MODEL_ID)
            & pairwise_frame["comparator"].isin(COMPARISON_MODELS[1:])
            & pairwise_frame["metric"].isin(("all_rmse", "asymmetric_rmse", "pair_delta_rmse", "asymmetric_regret_at_1"))
        ].copy()
        report.extend(
            [
                "## Paired Comparisons",
                "",
                selected[
                    [
                        "target",
                        "comparator",
                        "metric",
                        "point_loss_difference",
                        "ci_lower",
                        "ci_upper",
                        "probability_candidate_better",
                        "probability_candidate_tied",
                        "probability_candidate_worse",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(report))


def main() -> None:
    args = parse_args()
    targets = baseline.parse_csv(args.targets)
    unknown = sorted(set(targets) - set(baseline.TARGETS))
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload()
    baseline.write_json(args.output_dir / "protocol.json", protocol)
    for target in baseline.TARGETS:
        dataset, folds = baseline.prepare_target(args.output_dir, target, baseline.OUTER_SPLITS)
        if target in targets:
            run_cell(
                args.output_dir,
                protocol,
                target,
                dataset,
                folds,
                args.workers,
                args.force,
            )
            collect_results(args.output_dir, protocol)
    collect_results(args.output_dir, protocol)


if __name__ == "__main__":
    main()
