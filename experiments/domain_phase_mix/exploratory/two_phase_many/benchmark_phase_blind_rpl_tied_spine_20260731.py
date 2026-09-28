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
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit repaired RPL as a physically tied-only 300M aggregate spine.

This is an identification audit, not a new surrogate. It removes every RPL
phase channel, fits only genuinely tied policies, selects nonlinear response
shape and ridge inside each outer training fold, and then optimizes the raw
tied response without deployment regularization.

The aggregate response is

    A(w) = b + sum_i a_i (w_i + e0)^(-p)
             + sum_i d_i max(E_i(w) - tau, 0)^q,

with nonnegative family-pooled amplitudes and shrunk bucket departures. The
benefit is finite at zero share because e0 > 0; physical materialized epochs
enter only the overload term. No asymmetric endpoint can change this spine.
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
from scipy.stats import spearmanr

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
    benchmark_physical_hpr_tied_spine_20260731 as physical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_estimator_repair_20260731 as repaired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as parent,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "phase_blind_rpl_tied_spine_20260731"
MODEL_ID = "phase_blind_rpl_tied_spine"
TARGETS = ("uncheatable", "table9")
PROTOCOL_VERSION = "phase-blind-rpl-tied-spine-v1"
OUTER_SPLITS = 5
INNER_SPLITS = 3
OUTER_SEED = 731_630
INNER_SEED_BASE = 7_316_300
FULL_FIT_SEED = 7_316_399
OPTIMIZER_SEED = 7_316_500
BOOTSTRAP_SEED = 7_316_600
BOOTSTRAP_REPLICATES = 20
LOWER_TAIL_FRACTION = 0.15
OPTIMUM_ZERO_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--outer-splits", type=int, default=OUTER_SPLITS)
    parser.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-optimum", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_targets(raw: str) -> tuple[str, ...]:
    targets = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    if not targets:
        raise ValueError("At least one target is required")
    return targets


def protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    sources = (
        Path(__file__),
        Path(expanded.__file__),
        Path(baseline.__file__),
        Path(physical.__file__),
        Path(repaired.__file__),
        Path(parent.__file__),
        expanded.PACKET,
        expanded.ONE_PHASE_SOURCE,
    )
    payload = {
        "version": PROTOCOL_VERSION,
        "model": MODEL_ID,
        "data_role": "282 physically tied 300M policies only",
        "targets": list(parse_targets(args.targets)),
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "outer_seed": OUTER_SEED,
        "inner_seed_base": INNER_SEED_BASE,
        "full_fit_seed": FULL_FIT_SEED,
        "optimizer_seed": OPTIMIZER_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "phase_blind": True,
        "deployment_regularization": None,
        "shape_grid": {
            "benefit_exponents": list(parent.BENEFIT_EXPONENTS),
            "base_benefit_offsets": list(parent.BENEFIT_OFFSETS),
            "damage_exponents": list(parent.DAMAGE_EXPONENTS),
            "damage_thresholds": list(parent.DAMAGE_THRESHOLDS),
            "ridge": list(parent.RIDGE_GRID),
        },
        "selection": {
            "core_rmse_ratio_limit": repaired.CORE_RMSE_RATIO_LIMIT,
            "regret_at_1_slack": repaired.REGRET_AT_1_SLACK,
            "lower_tail_fraction": repaired.LOWER_TAIL_FRACTION,
            "lower_tail_min_count": repaired.LOWER_TAIL_MIN_COUNT,
        },
        "frozen_reference_rmse": physical.FROZEN_REFERENCE_RMSE,
        "source_hashes": {str(path.relative_to(REPO_ROOT)): baseline.file_hash(path) for path in sources},
    }
    encoded = json.dumps(
        baseline.json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def tied_panel(target: str) -> tuple[baseline.pooled.Dataset, np.ndarray]:
    source = expanded.load_300m(target)
    tied = expanded.replay_control.tied_rows(source.weights)
    if (int(tied.sum()), int((~tied).sum())) != (282, 238):
        raise ValueError("Expected 282 physically tied and 238 asymmetric policies")
    dataset = baseline.subset_dataset(
        baseline.as_pooled(source),
        np.flatnonzero(tied),
        "physical_tied",
    )
    if not np.allclose(
        dataset.weights[:, 0, :],
        dataset.weights[:, 1, :],
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError("Asymmetric policy leaked into the tied aggregate audit")
    if tuple(dataset.domain_names) != source.domain_names:
        raise AssertionError("Domain order changed while subsetting the tied panel")
    return dataset, np.asarray(source.family_index, dtype=int)


def tied_context(rows: int) -> repaired.SelectionContext:
    return repaired.SelectionContext(
        tied=np.ones(rows, dtype=bool),
        pair_tied=np.empty(0, dtype=int),
        pair_asymmetric=np.empty(0, dtype=int),
    )


def fit_model(
    dataset: baseline.pooled.Dataset,
    family_index: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    workers: int,
) -> repaired.Fitted:
    return repaired.fit(
        dataset.weights,
        dataset.y,
        baseline.retained_geometry(dataset, family_index),
        folds,
        tied_context(dataset.n),
        workers=workers,
        phase_blind=True,
    )


def fit_pinned_shape(
    dataset: baseline.pooled.Dataset,
    template: repaired.Fitted,
) -> repaired.Fitted:
    design, layout = repaired.phase_blind_design_matrix(
        dataset.weights,
        template.geometry,
        template.shape,
    )
    intercept, aggregate, phase = repaired.solve_head(
        design,
        dataset.y,
        template.ridge,
        repaired.penalty_multipliers(template.geometry, layout),
        layout,
    )
    return replace(
        template,
        intercept=intercept,
        aggregate_coefficients=aggregate,
        phase_coefficients=phase,
    )


def metric_summary(
    observed: np.ndarray,
    predicted: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> dict[str, float | int]:
    residual = predicted - observed
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    count = max(5, int(np.ceil(LOWER_TAIL_FRACTION * len(observed))))
    lower = np.argsort(predicted)[:count]
    metrics: dict[str, float | int] = {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "observed_on_predicted_slope": float(slope),
        "observed_on_predicted_intercept": float(intercept),
        "lower_tail_rmse": float(np.sqrt(np.mean(residual[lower] ** 2))),
        "lower_tail_optimism": float(np.mean(observed[lower] - predicted[lower])),
        "optimism_above_0p05_count": int(np.sum(observed - predicted > 0.05)),
        "worst_optimism": float(np.max(observed - predicted)),
    }
    for count_selected in (1, 3, 5):
        regrets = []
        for _train, test in folds:
            selected = test[np.argsort(predicted[test])[: min(count_selected, len(test))]]
            regrets.append(float(np.min(observed[selected]) - np.min(observed[test])))
        metrics[f"regret_at_{count_selected}"] = float(np.mean(regrets))
    return metrics


def parameter_diagnostics(model: repaired.Fitted) -> dict[str, int]:
    active = baseline.active_count(model.aggregate_coefficients)
    return {
        "selected_nonlinear_parameter_count": 3,
        "aggregate_linear_parameter_count": len(model.aggregate_coefficients),
        "active_aggregate_parameter_count": active,
        "nominal_parameter_count": 1 + 3 + len(model.aggregate_coefficients),
        "effective_df_active_set_proxy": 1 + 3 + active,
    }


def coefficient_rows(
    model: repaired.Fitted,
    fold: int | str,
) -> list[dict[str, int | float | str]]:
    names = repaired.feature_names(model.geometry, model.shape, include_phase=False)
    if len(names) != len(model.aggregate_coefficients):
        raise ValueError("Coefficient names do not match the phase-blind RPL design")
    return [
        {
            "fold": fold,
            "feature": name,
            "coefficient": float(coefficient),
        }
        for name, coefficient in zip(
            names,
            model.aggregate_coefficients,
            strict=True,
        )
    ]


def optimum_record(
    model: repaired.Fitted,
    dataset: baseline.pooled.Dataset,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    optimum, prediction, successful = physical.optimize_tied(model, dataset, seed)
    tied_weights = np.stack([optimum, optimum])
    epochs = optimum * (dataset.c0 + dataset.c1)
    observed_best_index = int(np.argmin(dataset.y))
    observed_best = dataset.weights[observed_best_index, 0, :]
    observed_epochs = dataset.weights[:, 0, :] * (dataset.c0 + dataset.c1)
    beta0 = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    diagnostics: dict[str, float | int] = {
        "predicted_bpb": prediction,
        "observed_best_tied_bpb": float(dataset.y[observed_best_index]),
        "l1_to_observed_best": float(np.abs(optimum - observed_best).sum()),
        "maximum_bucket_weight": float(np.max(optimum)),
        "maximum_materialized_epochs": float(np.max(epochs)),
        "near_zero_bucket_count": int(np.sum(optimum <= OPTIMUM_ZERO_TOLERANCE)),
        "observed_maximum_bucket_weight": float(np.max(dataset.weights[:, 0, :])),
        "observed_maximum_materialized_epochs": float(np.max(observed_epochs)),
        "successful_optimizer_starts": successful,
        **baseline.support_distances(tied_weights, dataset.weights, beta0),
    }
    policy = pd.DataFrame(
        {
            "domain": dataset.domain_names,
            "weight": optimum,
            "materialized_epochs": epochs,
            "observed_best_weight": observed_best,
        }
    )
    return diagnostics, policy


def bootstrap_indices(
    frame: pd.DataFrame,
    generator: np.random.Generator,
) -> np.ndarray:
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()
    unique = np.unique(groups)
    sampled = generator.choice(unique, size=len(unique), replace=True)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    return np.concatenate([rows_by_group[group] for group in sampled])


def bootstrap_optima(
    dataset: baseline.pooled.Dataset,
    template: repaired.Fitted,
    full_optimum: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        indices = bootstrap_indices(dataset.frame, generator)
        sampled = baseline.subset_dataset(
            dataset,
            indices,
            f"bootstrap_{replicate}",
        )
        fitted = fit_pinned_shape(sampled, template)
        optimum, prediction, successful = physical.optimize_tied(
            fitted,
            sampled,
            seed + 1000 + replicate,
        )
        row: dict[str, float | int] = {
            "replicate": replicate,
            "predicted_optimum_bpb": prediction,
            "l1_to_full_optimum": float(np.abs(optimum - full_optimum).sum()),
            "maximum_bucket_weight": float(np.max(optimum)),
            "near_zero_bucket_count": int(np.sum(optimum <= OPTIMUM_ZERO_TOLERANCE)),
            "successful_optimizer_starts": successful,
        }
        row.update(
            {
                f"weight_{domain}": float(value)
                for domain, value in zip(
                    dataset.domain_names,
                    optimum,
                    strict=True,
                )
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def target_dir(output_dir: Path, target: str) -> Path:
    return output_dir / "cells" / target


def target_complete(
    path: Path,
    protocol_hash: str,
    require_optimum: bool,
) -> bool:
    marker = path / "complete.json"
    required = (
        path / "metrics.json",
        path / "predictions.csv",
        path / "full_fit.json",
    )
    if not marker.exists() or any(not item.exists() for item in required):
        return False
    payload = json.loads(marker.read_text())
    if payload.get("protocol_hash") != protocol_hash:
        return False
    return not require_optimum or bool(payload.get("has_raw_optimum"))


def prepare_target(
    output_dir: Path,
    target: str,
    outer_splits: int,
) -> tuple[
    baseline.pooled.Dataset,
    np.ndarray,
    tuple[tuple[np.ndarray, np.ndarray], ...],
]:
    dataset, family_index = tied_panel(target)
    folds = baseline.correspondence_folds(
        dataset.frame,
        OUTER_SEED,
        outer_splits,
    )
    assignment = np.full(dataset.n, -1, dtype=int)
    for fold, (_train, test) in enumerate(folds):
        assignment[test] = fold
    manifest = dataset.frame.copy()
    manifest.insert(0, "row_index", np.arange(dataset.n))
    manifest["outer_fold"] = assignment
    manifest["target_value"] = dataset.y
    manifest.to_csv(output_dir / f"rows_{target}.csv", index=False)
    return dataset, family_index, folds


def run_target(
    output_dir: Path,
    protocol: dict[str, Any],
    target: str,
    dataset: baseline.pooled.Dataset,
    family_index: np.ndarray,
    outer_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    args: argparse.Namespace,
) -> None:
    path = target_dir(output_dir, target)
    require_optimum = not args.skip_optimum
    if not args.force and target_complete(
        path,
        str(protocol["protocol_hash"]),
        require_optimum,
    ):
        print(f"skip complete {target}", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    prediction = np.full(dataset.n, np.nan)
    outer_fold = np.full(dataset.n, -1, dtype=int)
    selections = []
    fold_optima = []
    coefficient_records = []

    for fold_id, (train, test) in enumerate(outer_folds):
        print(
            f"{target}: outer fold {fold_id + 1}/{len(outer_folds)}",
            flush=True,
        )
        local = baseline.subset_dataset(
            dataset,
            train,
            f"outer_{fold_id}_train",
        )
        inner = baseline.correspondence_folds(
            local.frame,
            INNER_SEED_BASE + fold_id,
            args.inner_splits,
        )
        model = fit_model(local, family_index, inner, args.workers)
        prediction[test] = model.predict(dataset.weights[test])
        outer_fold[test] = fold_id
        selections.append(
            {
                "outer_fold": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                "shape": asdict(model.shape),
                "ridge": model.ridge,
                "selection": asdict(model.selection),
                "parameters": parameter_diagnostics(model),
            }
        )
        coefficient_records.extend(coefficient_rows(model, fold_id))
        if not args.skip_optimum:
            diagnostics, policy = optimum_record(
                model,
                local,
                OPTIMIZER_SEED + 100 * (fold_id + 1),
            )
            row: dict[str, float | int | str] = {
                "fold": fold_id,
                **diagnostics,
            }
            row.update(
                {
                    f"weight_{domain}": float(value)
                    for domain, value in zip(
                        local.domain_names,
                        policy["weight"],
                        strict=True,
                    )
                }
            )
            fold_optima.append(row)

    if not np.isfinite(prediction).all() or np.any(outer_fold < 0):
        raise RuntimeError(f"Incomplete tied-only OOF predictions for {target}")
    predictions = dataset.frame[
        [
            "run_name",
            "phase_correspondence_key",
            "policy_family",
            "source_panel",
        ]
    ].copy()
    predictions.insert(0, "row_index", np.arange(dataset.n))
    predictions["outer_fold"] = outer_fold
    predictions["observed"] = dataset.y
    predictions["predicted"] = prediction
    predictions["residual"] = prediction - dataset.y
    predictions["optimism"] = dataset.y - prediction
    predictions.to_csv(path / "predictions.csv", index=False)

    metrics = {
        "target": target,
        "model": MODEL_ID,
        "n_tied": dataset.n,
        **metric_summary(dataset.y, prediction, outer_folds),
    }
    reference = physical.FROZEN_REFERENCE_RMSE[target]
    metrics.update(
        {
            "frozen_reference_rmse": reference,
            "relative_rmse_to_reference": float(metrics["rmse"]) / reference - 1.0,
            "passes_five_percent_oof_gate": bool(float(metrics["rmse"]) <= 1.05 * reference),
        }
    )
    baseline.write_json(path / "metrics.json", metrics)
    baseline.write_json(path / "fold_selections.json", selections)

    print(f"{target}: fitting full tied-only model", flush=True)
    full_inner = baseline.correspondence_folds(
        dataset.frame,
        FULL_FIT_SEED,
        args.inner_splits,
    )
    full_model = fit_model(dataset, family_index, full_inner, args.workers)
    coefficient_records.extend(coefficient_rows(full_model, "full"))
    coefficient_frame = pd.DataFrame(coefficient_records)
    coefficient_frame.to_csv(path / "coefficients.csv", index=False)
    outer_coefficients = coefficient_frame.loc[coefficient_frame["fold"].astype(str).ne("full")]
    stability = (
        outer_coefficients.groupby("feature", sort=True)["coefficient"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    stability["sign_stable"] = stability["min"] * stability["max"] >= 0.0
    stability.to_csv(path / "coefficient_stability.csv", index=False)
    full_fit_payload: dict[str, Any] = {
        "shape": asdict(full_model.shape),
        "ridge": full_model.ridge,
        "selection": asdict(full_model.selection),
        "parameters": parameter_diagnostics(full_model),
    }

    if not args.skip_optimum:
        full_diagnostics, full_policy = optimum_record(
            full_model,
            dataset,
            OPTIMIZER_SEED,
        )
        full_fit_payload["raw_optimum"] = full_diagnostics
        full_policy.to_csv(path / "raw_optimum_policy.csv", index=False)
        if fold_optima:
            fold_frame = pd.DataFrame(fold_optima)
            full_weights = full_policy["weight"].to_numpy(dtype=float)
            weight_columns = [f"weight_{domain}" for domain in dataset.domain_names]
            fold_frame["l1_to_full_optimum"] = np.abs(
                fold_frame[weight_columns].to_numpy(dtype=float) - full_weights
            ).sum(axis=1)
            fold_frame.to_csv(path / "fold_optima.csv", index=False)
            full_fit_payload["fold_optimum_stability"] = {
                "median_l1_to_full": float(fold_frame["l1_to_full_optimum"].median()),
                "maximum_l1_to_full": float(fold_frame["l1_to_full_optimum"].max()),
            }
        if args.bootstrap_replicates:
            bootstraps = bootstrap_optima(
                dataset,
                full_model,
                full_policy["weight"].to_numpy(dtype=float),
                args.bootstrap_replicates,
                BOOTSTRAP_SEED,
            )
            bootstraps.to_csv(path / "bootstrap_optima.csv", index=False)
            full_fit_payload["conditional_bootstrap_stability"] = {
                "replicates": len(bootstraps),
                "shape_and_ridge_reselected": False,
                "median_l1_to_full": float(bootstraps["l1_to_full_optimum"].median()),
                "maximum_l1_to_full": float(bootstraps["l1_to_full_optimum"].max()),
            }
    baseline.write_json(path / "full_fit.json", full_fit_payload)
    baseline.write_json(
        path / "complete.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "has_raw_optimum": not args.skip_optimum,
            "bootstrap_replicates": 0 if args.skip_optimum else args.bootstrap_replicates,
        },
    )


def collect(output_dir: Path, targets: tuple[str, ...]) -> None:
    metric_rows = []
    optimum_rows = []
    for target in targets:
        path = target_dir(output_dir, target)
        metric_rows.append(json.loads((path / "metrics.json").read_text()))
        full = json.loads((path / "full_fit.json").read_text())
        if "raw_optimum" in full:
            optimum_rows.append(
                {
                    "target": target,
                    **full["raw_optimum"],
                    **full.get("fold_optimum_stability", {}),
                    **{
                        f"bootstrap_{key}": value
                        for key, value in full.get(
                            "conditional_bootstrap_stability",
                            {},
                        ).items()
                    },
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    optima = pd.DataFrame(optimum_rows)
    if len(optima):
        optima.to_csv(output_dir / "optimum_diagnostics.csv", index=False)

    lines = [
        "# Physically tied phase-blind RPL aggregate-spine audit",
        "",
        "The repaired RPL aggregate restriction was fit only to 282 physically tied policies.",
        "Nonlinear shape and ridge were selected inside every correspondence-grouped outer training fold.",
        "No asymmetric endpoint, temporal state, KL penalty, trust region, or output calibration was used.",
        "",
        metrics.to_markdown(index=False),
    ]
    if len(optima):
        lines.extend(
            [
                "",
                "## Raw tied optimum",
                "",
                optima.to_markdown(index=False),
                "",
                "Bootstrap stability is conditional on the full-data selected shape and ridge; "
                "outer-fold stability includes complete nested reselection.",
            ]
        )
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.outer_splits < 2 or args.inner_splits < 2:
        raise ValueError("workers must be positive and fold counts must be at least two")
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    targets = parse_targets(args.targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(args)
    baseline.write_json(args.output_dir / "protocol.json", protocol)
    prepared = {target: prepare_target(args.output_dir, target, args.outer_splits) for target in targets}
    if args.prepare_only:
        print(f"Prepared protocol {protocol['protocol_hash']}", flush=True)
        return
    for target in targets:
        dataset, family_index, folds = prepared[target]
        run_target(
            args.output_dir,
            protocol,
            target,
            dataset,
            family_index,
            folds,
            args,
        )
    collect(args.output_dir, targets)
    print(f"Wrote {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
