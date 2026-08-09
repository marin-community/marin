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
"""Evaluate the preregistered repaired RPL head on the expanded 300M panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
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
    retained_power_law_estimator_repair_20260731 as repaired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as parent,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_20260731"
BASELINE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731"
TARGETS = baseline.TARGETS
MODEL_ID = "retained_power_law_repaired"
PROTOCOL_VERSION = "repaired-rpl-300m-v1-correspondence-nested"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--optimizer-starts", type=int, default=baseline.OPTIMIZER_STARTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-optimum", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def protocol_payload(optimizer_starts: int) -> dict[str, Any]:
    baseline_protocol = baseline.protocol_payload(
        baseline.OUTER_SPLITS,
        baseline.INNER_SPLITS,
        optimizer_starts,
    )
    sources = (Path(__file__), Path(repaired.__file__), Path(parent.__file__))
    payload = {
        "version": PROTOCOL_VERSION,
        "model": MODEL_ID,
        "targets": TARGETS,
        "baseline_protocol_hash": baseline_protocol["protocol_hash"],
        "outer_splits": baseline.OUTER_SPLITS,
        "inner_splits": baseline.INNER_SPLITS,
        "optimizer_starts": optimizer_starts,
        "selection": {
            "core_rmse_ratio_limit": repaired.CORE_RMSE_RATIO_LIMIT,
            "regret_at_1_slack": repaired.REGRET_AT_1_SLACK,
            "lower_tail_fraction": repaired.LOWER_TAIL_FRACTION,
            "lower_tail_min_count": repaired.LOWER_TAIL_MIN_COUNT,
            "phase_penalty_multiplier": repaired.PHASE_PENALTY_MULTIPLIER,
        },
        "source_hashes": {str(path.relative_to(REPO_ROOT)): baseline.file_hash(path) for path in sources},
    }
    encoded = json.dumps(baseline.json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def selection_context(dataset: baseline.pooled.Dataset) -> repaired.SelectionContext:
    frame = dataset.frame.reset_index(drop=True)
    indexed = frame.reset_index().set_index(["phase_correspondence_key", "policy_family"])["index"]
    keys = sorted(
        set(frame.loc[frame["policy_family"].eq("single_phase"), "phase_correspondence_key"].astype(str))
        & set(frame.loc[frame["policy_family"].eq("two_phase"), "phase_correspondence_key"].astype(str))
    )
    pair_tied = np.asarray([indexed.loc[(key, "single_phase")] for key in keys], dtype=int)
    pair_asymmetric = np.asarray([indexed.loc[(key, "two_phase")] for key in keys], dtype=int)
    genuinely_asymmetric = ~expanded.replay_control.tied_rows(dataset.weights[pair_asymmetric])
    return repaired.SelectionContext(
        tied=expanded.replay_control.tied_rows(dataset.weights),
        pair_tied=pair_tied[genuinely_asymmetric],
        pair_asymmetric=pair_asymmetric[genuinely_asymmetric],
    )


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
        selection_context(dataset),
        workers=workers,
    )


def parameter_diagnostics(model: repaired.Fitted) -> dict[str, int | float | str]:
    nonlinear = baseline.numeric_scalar_count(model.shape)
    aggregate_nominal = len(model.aggregate_coefficients)
    phase_nominal = len(model.phase_coefficients)
    aggregate_active = baseline.active_count(model.aggregate_coefficients)
    phase_active = baseline.active_count(model.phase_coefficients)
    return {
        "nominal_parameter_count": 1 + nonlinear + aggregate_nominal + phase_nominal,
        "nonlinear_parameter_count": nonlinear,
        "aggregate_linear_parameter_count": aggregate_nominal,
        "phase_linear_parameter_count": phase_nominal,
        "active_aggregate_parameter_count": aggregate_active,
        "active_phase_parameter_count": phase_active,
        "effective_df_active_set_proxy": 1 + nonlinear + aggregate_active + phase_active,
        "effective_df_note": "intercept + selected shape scalars + active aggregate and signed phase coefficients",
    }


def coefficient_rows(model: repaired.Fitted, fold: int | str) -> list[dict[str, int | float | str]]:
    names = repaired.feature_names(model.geometry, model.shape)
    if len(names) != len(model.coefficients):
        raise ValueError("coefficient names do not match the repaired RPL design")
    return [
        {
            "fold": fold,
            "feature": name,
            "coefficient": float(coefficient),
            "channel": "phase" if name.startswith("phase_") else "aggregate",
        }
        for name, coefficient in zip(names, model.coefficients, strict=True)
    ]


def coefficient_stability(rows: pd.DataFrame) -> pd.DataFrame:
    outer = rows.loc[rows["fold"].astype(str).ne("full")].copy()
    summaries = []
    for (channel, feature), block in outer.groupby(["channel", "feature"], sort=True):
        values = block["coefficient"].to_numpy(dtype=float)
        nonzero = np.abs(values) > baseline.ACTIVE_COEFFICIENT_TOLERANCE
        signs = np.sign(values[nonzero])
        summaries.append(
            {
                "channel": channel,
                "feature": feature,
                "fold_count": len(block),
                "nonzero_fold_count": int(nonzero.sum()),
                "mean_coefficient": float(np.mean(values)),
                "coefficient_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "sign_stable_when_nonzero": bool(not len(signs) or np.all(signs == signs[0])),
            }
        )
    return pd.DataFrame(summaries)


def cell_dir(output_dir: Path, target: str) -> Path:
    return output_dir / "cells" / target / MODEL_ID


def cell_complete(path: Path, protocol_hash: str, require_optimum: bool) -> bool:
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
    if marker.get("protocol_hash") != protocol_hash:
        return False
    return not require_optimum or bool(marker.get("has_raw_optimum"))


def run_cell(
    output_dir: Path,
    protocol: dict[str, Any],
    target: str,
    dataset: expanded.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    workers: int,
    optimizer_starts: int,
    skip_optimum: bool,
    force: bool,
) -> None:
    path = cell_dir(output_dir, target)
    if not force and cell_complete(path, str(protocol["protocol_hash"]), not skip_optimum):
        print(f"skip complete {target}/{MODEL_ID}", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    pooled_dataset = baseline.as_pooled(dataset)
    predicted = np.full(dataset.n, np.nan, dtype=float)
    outer_fold = np.full(dataset.n, -1, dtype=int)
    nearest_policy_tv = np.full(dataset.n, np.nan, dtype=float)
    nearest_aggregate_tv = np.full(dataset.n, np.nan, dtype=float)
    selections = []
    parameter_rows = []
    coefficients = []

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
        nearest_policy_tv[test], nearest_aggregate_tv[test] = baseline.test_support_columns(dataset, train, test)
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
    tied = expanded.replay_control.tied_rows(dataset.weights)
    beta0 = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    phase_tv = 0.5 * np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]).sum(axis=1)
    aggregate = beta0 * dataset.weights[:, 0, :] + (1.0 - beta0) * dataset.weights[:, 1, :]
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
        "reference_only": False,
        **baseline.metric_summary(dataset, predicted, folds),
    }
    pair_metrics, pair_predictions = baseline.pair_summary(dataset, predicted)
    baseline.write_json(path / "metrics.json", metrics)
    baseline.write_json(path / "pair_metrics.json", pair_metrics)
    pair_predictions.to_csv(path / "pair_predictions.csv", index=False)
    baseline.write_json(path / "fold_selections.json", selections)
    pd.DataFrame(parameter_rows).to_csv(path / "parameter_diagnostics.csv", index=False)

    print(f"{target}/{MODEL_ID}: fitting full 520-row model", flush=True)
    full_inner = baseline.correspondence_folds(
        pooled_dataset.frame,
        baseline.FULL_FIT_SEED,
        baseline.INNER_SPLITS,
    )
    full_fit = fit_model(pooled_dataset, full_inner, dataset.family_index, workers)
    coefficients.extend(coefficient_rows(full_fit, "full"))
    coefficient_frame = pd.DataFrame(coefficients)
    coefficient_frame.to_csv(path / "fold_coefficients.csv", index=False)
    coefficient_stability(coefficient_frame).to_csv(path / "coefficient_stability.csv", index=False)
    baseline.write_json(
        path / "full_fit.json",
        {
            "shape": asdict(full_fit.shape),
            "ridge": full_fit.ridge,
            "selection": asdict(full_fit.selection),
            "parameter_diagnostics": parameter_diagnostics(full_fit),
        },
    )
    if not skip_optimum:
        print(f"{target}/{MODEL_ID}: raw optimum audit", flush=True)
        optimum, policy = baseline.raw_optimum(
            MODEL_ID,
            baseline.FitResult(
                model=full_fit,
                selection={},
                parameter_diagnostics=parameter_diagnostics(full_fit),
            ),
            pooled_dataset,
            optimizer_starts,
            baseline.FULL_FIT_SEED,
        )
        baseline.write_json(path / "raw_optimum.json", optimum)
        policy.to_csv(path / "raw_optimum_policy.csv", index=False)

    baseline.write_json(
        path / "complete.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "target": target,
            "model": MODEL_ID,
            "has_raw_optimum": (path / "raw_optimum.json").exists(),
        },
    )
    print(f"completed {target}/{MODEL_ID}", flush=True)


def collect_results(output_dir: Path, protocol: dict[str, Any]) -> None:
    rows = []
    pairs = []
    optima = []
    complete = []
    for target in TARGETS:
        path = cell_dir(output_dir, target)
        if not cell_complete(path, str(protocol["protocol_hash"]), False):
            continue
        complete.append(target)
        rows.append(json.loads((path / "metrics.json").read_text()))
        pairs.append(
            {
                "target": target,
                "model": MODEL_ID,
                **json.loads((path / "pair_metrics.json").read_text()),
            }
        )
        optimum_path = path / "raw_optimum.json"
        if optimum_path.exists():
            optima.append(
                {
                    "target": target,
                    "model": MODEL_ID,
                    **json.loads(optimum_path.read_text()),
                }
            )
    metrics = pd.DataFrame(rows)
    pair_metrics = pd.DataFrame(pairs)
    raw_optima = pd.DataFrame(optima)
    metrics.to_csv(output_dir / "candidate_metrics.csv", index=False)
    pair_metrics.to_csv(output_dir / "candidate_pair_metrics.csv", index=False)
    raw_optima.to_csv(output_dir / "candidate_raw_optima.csv", index=False)
    baseline.write_json(
        output_dir / "status.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "complete_targets": complete,
            "frozen_complete": len(complete) == len(TARGETS),
        },
    )
    report = [
        "# Repaired Retained-Power-Law Head",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Parent baseline protocol: `{protocol['baseline_protocol_hash']}`",
        f"- Complete targets: {len(complete)}/{len(TARGETS)}",
        "- This is estimator route `WSD80-SUR-046`; the RPL state and response are unchanged.",
        "",
    ]
    if not metrics.empty:
        report.extend(
            [
                "## Correspondence-Grouped OOF",
                "",
                metrics[
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
    if not pair_metrics.empty:
        report.extend(
            [
                "## Exact Aggregate-Matched Contrasts",
                "",
                pair_metrics[["target", "delta_rmse", "delta_spearman", "delta_bias", "sign_accuracy"]].to_markdown(
                    index=False, floatfmt=".6f"
                ),
                "",
            ]
        )
    if not raw_optima.empty:
        report.extend(
            [
                "## Raw Optima",
                "",
                raw_optima[
                    [
                        "target",
                        "predicted_bpb",
                        "predicted_phase_gain_on_fiber",
                        "phase_tv",
                        "max_bucket_weight",
                        "nearest_policy_tv",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(report))


def main() -> None:
    args = parse_args()
    targets = baseline.parse_csv(args.targets)
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.optimizer_starts < 8:
        raise ValueError("optimizer starts must be at least eight")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(args.optimizer_starts)
    baseline.write_json(args.output_dir / "protocol.json", protocol)
    baseline.write_json(args.output_dir / "acceptance_gate.json", baseline.acceptance_gate())
    prepared: dict[
        str,
        tuple[expanded.Dataset, tuple[tuple[np.ndarray, np.ndarray], ...]],
    ] = {}
    for target in TARGETS:
        prepared[target] = baseline.prepare_target(args.output_dir, target, baseline.OUTER_SPLITS)
    if args.prepare_only:
        collect_results(args.output_dir, protocol)
        print(f"prepared protocol {protocol['protocol_hash']} in {args.output_dir}", flush=True)
        return

    for target in TARGETS:
        if target not in targets:
            continue
        dataset, folds = prepared[target]
        run_cell(
            args.output_dir,
            protocol,
            target,
            dataset,
            folds,
            args.workers,
            args.optimizer_starts,
            args.skip_optimum,
            args.force,
        )
        if not args.no_collect:
            collect_results(args.output_dir, protocol)
    if not args.no_collect:
        collect_results(args.output_dir, protocol)


if __name__ == "__main__":
    main()
