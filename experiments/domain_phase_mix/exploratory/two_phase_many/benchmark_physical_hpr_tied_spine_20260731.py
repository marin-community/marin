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
"""Audit phase-blind physical-exposure aggregate spines.

Each aggregate spine is fit only to physically tied 300M policies. Its latent
state is observed total materialized epochs,

    E_i(a) = (c_i^(0) + c_i^(1)) a_i,

and retention, forgetting, late utility, and phase-shift terms are fixed to
their phase-blind values. The maintained HPR response is compared with a
family-composition response that lets family total exposure control saturation
and within-family exposure shares allocate quality. This isolates aggregate
modeling from the temporal-state problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_compact_tied_backbone_20260730 as compact_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    corrected_hpr_model_20260727 as corrected_hpr,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "physical_hpr_tied_spine_20260731"
TARGETS = ("uncheatable", "table9")
OPTIMIZER_STARTS = 12
OPTIMIZER_SEED = 20260731
WEIGHT_ZERO_TOLERANCE = 1e-6
AGGREGATE_SPECS = {
    "physical_hpr_tied_spine": (
        hierarchical_grp.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
        None,
    ),
    "family_composition_tied_spine": (
        hierarchical_grp.Variant.FAMILY_COMPOSITION_PHASE_REPLAY,
        None,
    ),
    "bounded_physical_hpr_tied_spine": (
        hierarchical_grp.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
        corrected_hpr.Corrections(bounded_link=True),
    ),
}

# Strongest maintained independently fitted one-phase references in the
# Observatory. These are not refit on the 282-row panel and are therefore used
# only as a frozen acceptance threshold, not as rowwise paired predictions.
FROZEN_REFERENCE_RMSE = {
    "uncheatable": 0.004713404694656708,
    "table9": 0.010357273218151471,
}


class Predictor(Protocol):
    """Predict BPB for a batch of two-column policies."""

    def predict(self, weights: np.ndarray) -> np.ndarray: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def tied_dataset(target: str) -> pooled.Dataset:
    """Return the 282 physically tied rows from the matched 300M panel."""
    source = benchmark.load_300m(target)
    tied = benchmark.replay_control.tied_rows(source.weights)
    if (int(tied.sum()), int((~tied).sum())) != (282, 238):
        raise ValueError("Expected 282 tied and 238 asymmetric 300M policies")
    return compact_audit.as_pooled_dataset(
        name=f"300m_{target}_physical_tied",
        frame=source.frame.loc[tied].reset_index(drop=True),
        target=source.y[tied],
        weights=source.weights[tied],
        c0=source.c0,
        c1=source.c1,
        domain_names=source.domain_names,
    )


def assert_physical_restriction(
    dataset: pooled.Dataset,
    config: hierarchical_grp.Config,
) -> None:
    """Verify that the aggregate state is exactly observed total materialized epochs."""
    shape = config.shape
    if shape.late_multiplier != 1.0 or shape.forgetting_rate != 0.0:
        raise AssertionError(f"Temporal parameters leaked into aggregate spine: {shape}")
    if config.variant not in {variant for variant, _corrections in AGGREGATE_SPECS.values()}:
        raise AssertionError(f"Unexpected aggregate variant: {config.variant}")
    if not np.allclose(dataset.weights[:, 0, :], dataset.weights[:, 1, :], atol=1e-12, rtol=0.0):
        raise AssertionError("Aggregate spine received an asymmetric policy")


def selected_config(
    dataset: pooled.Dataset,
    variant: hierarchical_grp.Variant,
    corrections: corrected_hpr.Corrections | None,
) -> tuple[hierarchical_grp.Config, dict]:
    structured = observatory.family_dataset(dataset)
    shapes = observatory.hierarchical_phase_replay_shape_candidates(observatory.SINGLE_PHASE)
    splits = observatory.folds(dataset, hierarchical_grp.SCREEN_SEED)
    _baseline, _baseline_prediction, baseline_rows = hierarchical_grp.score_configs(
        structured,
        hierarchical_grp.baseline_configs(shapes),
        splits,
    )
    best_by_shape: dict[int, float] = {}
    for row in baseline_rows:
        shape_index = int(row["shape_index"])
        best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), float(row["rmse"]))
    shape_indices = [
        shape_index
        for shape_index, _rmse in sorted(best_by_shape.items(), key=lambda item: item[1])[
            : observatory.HIERARCHICAL_PHASE_REPLAY_TOP_SHAPES
        ]
    ]
    if corrections is None:
        config, _prediction, candidate_rows = hierarchical_grp.score_configs(
            structured,
            hierarchical_grp.structural_configs(variant, shapes, shape_indices),
            splits,
        )
    else:
        # A nonlinear output link changes which latent-response shape is best.
        # Score the complete frozen grid so the unbounded HPR shortlist cannot
        # silently determine the bounded candidate.
        shape_indices = list(range(len(shapes)))
        candidate_rows = []
        best: tuple[float, float, hierarchical_grp.Config] | None = None
        for candidate in hierarchical_grp.structural_configs(variant, shapes, shape_indices):
            prediction = corrected_hpr.corrected_oof_prediction(
                structured,
                candidate,
                corrections,
                splits,
            )
            metrics = hierarchical_grp.metric_summary(structured.target, prediction)
            candidate_rows.append(hierarchical_grp.config_record(candidate, metrics))
            key = (float(metrics["rmse"]), -float(metrics["spearman"]), candidate)
            if best is None or key[:2] < best[:2]:
                best = key
        if best is None:
            raise RuntimeError(f"No bounded aggregate configurations for {dataset.name}")
        config = best[2]
    assert_physical_restriction(dataset, config)
    return config, {
        "baselineShapeScreen": baseline_rows,
        "candidateSweep": candidate_rows,
        "corrections": None if corrections is None else asdict(corrections),
        "screenSeed": hierarchical_grp.SCREEN_SEED,
        "topShapeIndices": shape_indices,
    }


def fit_spine(
    dataset: pooled.Dataset,
    config: hierarchical_grp.Config,
    indices: np.ndarray,
    corrections: corrected_hpr.Corrections | None,
) -> Predictor:
    structured = observatory.family_dataset(dataset)
    if corrections is None:
        return hierarchical_grp.fit_model(structured, config, indices)
    return corrected_hpr.fit_corrected(structured, config, corrections, indices)


def grouped_oof(
    dataset: pooled.Dataset,
    config: hierarchical_grp.Config,
    corrections: corrected_hpr.Corrections | None,
    seed: int,
) -> tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Refit the nonnegative head in grouped folds at the frozen shape."""
    groups = dataset.frame["phase_correspondence_key"].astype(str).to_numpy()
    indices = np.arange(dataset.n)
    folds = compact_audit.group_folds(indices, groups, 5, seed)
    prediction = np.full(dataset.n, np.nan)
    for train, test in folds:
        model = fit_spine(dataset, config, train, corrections)
        prediction[test] = model.predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete physical-HPR OOF prediction")
    return prediction, folds


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "observed_on_predicted_slope": float(slope),
        "observed_on_predicted_intercept": float(intercept),
    }


def weights_from_logits(logits: np.ndarray, bucket_count: int) -> np.ndarray:
    reduced = np.asarray(logits, dtype=float).reshape(bucket_count - 1)
    tied = softmax(np.concatenate([reduced, np.zeros(1)]))
    return np.stack([tied, tied], axis=0)


def logits_from_weights(weights: np.ndarray) -> np.ndarray:
    safe = np.log(np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0))
    return np.clip(safe[:-1] - safe[-1], -12.0, 12.0)


def optimization_starts(dataset: pooled.Dataset, seed: int) -> tuple[np.ndarray, ...]:
    observed_best = dataset.weights[int(np.argmin(dataset.y)), 0, :]
    proportional = 1.0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)
    proportional /= proportional.sum()
    starts = [
        logits_from_weights(observed_best),
        logits_from_weights(proportional),
        logits_from_weights(np.full(dataset.m, 1.0 / dataset.m)),
    ]
    generator = np.random.default_rng(seed)
    concentrations = (0.25, 1.0, 4.0)
    while len(starts) < OPTIMIZER_STARTS:
        concentration = concentrations[(len(starts) - 3) % len(concentrations)]
        starts.append(logits_from_weights(generator.dirichlet(np.full(dataset.m, concentration))))
    return tuple(starts)


def optimize_tied(
    model: Predictor,
    dataset: pooled.Dataset,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    """Optimize the raw aggregate response with no deployment regularizer."""

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits, dataset.m)
        return float(model.predict(weights[None, :, :])[0])

    candidates: list[tuple[float, np.ndarray]] = []
    successful = 0
    for start in optimization_starts(dataset, seed):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0)] * len(start),
            options={
                "maxiter": 2000,
                "maxfun": 250000,
                "ftol": 1e-12,
                "gtol": 1e-7,
                "maxls": 50,
            },
        )
        if result.success:
            successful += 1
        if not np.isfinite(result.fun):
            continue
        weights = weights_from_logits(np.asarray(result.x, dtype=float), dataset.m)[0]
        candidates.append((float(result.fun), weights))
    if not candidates:
        raise RuntimeError(f"No finite raw tied optimum for {dataset.name}")
    value, weights = min(candidates, key=lambda item: item[0])
    return weights, value, successful


def audit_target(
    model_name: str,
    variant: hierarchical_grp.Variant,
    corrections: corrected_hpr.Corrections | None,
    target: str,
    seed: int,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame, pd.DataFrame, dict]:
    dataset = tied_dataset(target)
    config, sweep = selected_config(dataset, variant, corrections)
    prediction, folds = grouped_oof(dataset, config, corrections, seed)
    metrics = metric_summary(dataset.y, prediction)
    full_model = fit_spine(dataset, config, np.arange(dataset.n), corrections)
    optimum, optimum_bpb, successful = optimize_tied(full_model, dataset, OPTIMIZER_SEED + seed)
    fold_rows = []
    distances = []
    for fold_index, (train, _test) in enumerate(folds):
        fold_model = fit_spine(dataset, config, train, corrections)
        fold_optimum, fold_bpb, fold_successful = optimize_tied(
            fold_model,
            dataset,
            OPTIMIZER_SEED + seed + 100 * (fold_index + 1),
        )
        distance = float(np.abs(fold_optimum - optimum).sum())
        distances.append(distance)
        fold_rows.append(
            {
                "model": model_name,
                "target": target,
                "fold": fold_index,
                "fold_to_full_optimum_l1": distance,
                "predicted_optimum_bpb": fold_bpb,
                "successful_optimizer_starts": fold_successful,
                **{f"weight_{name}": value for name, value in zip(dataset.domain_names, fold_optimum, strict=True)},
            }
        )

    exposure = optimum * (dataset.c0 + dataset.c1)
    observed_best = dataset.weights[int(np.argmin(dataset.y)), 0, :]
    reference = FROZEN_REFERENCE_RMSE[target]
    row: dict[str, float | int | str | bool] = {
        "target": target,
        "model": model_name,
        "n_tied": dataset.n,
        "oof_rmse": metrics["rmse"],
        "oof_mae": metrics["mae"],
        "oof_bias": metrics["bias"],
        "oof_spearman": metrics["spearman"],
        "observed_on_predicted_slope": metrics["observed_on_predicted_slope"],
        "frozen_reference_rmse": reference,
        "relative_rmse_to_reference": metrics["rmse"] / reference - 1.0,
        "passes_five_percent_oof_gate": bool(metrics["rmse"] <= 1.05 * reference),
        "selected_power": config.shape.exponent,
        "selected_penalty_threshold": config.shape.penalty_threshold,
        "selected_l2": config.l2,
        "selected_residual_shrink": config.residual_shrink,
        "bounded_link": corrections is not None and corrections.bounded_link,
        "predicted_optimum_bpb": optimum_bpb,
        "observed_best_tied_bpb": float(np.min(dataset.y)),
        "optimum_l1_to_observed_best": float(np.abs(optimum - observed_best).sum()),
        "median_fold_to_full_optimum_l1": float(np.median(distances)),
        "maximum_fold_to_full_optimum_l1": float(np.max(distances)),
        "maximum_optimum_weight": float(np.max(optimum)),
        "maximum_optimum_epochs": float(np.max(exposure)),
        "near_zero_optimum_weights": int(np.sum(optimum <= WEIGHT_ZERO_TOLERANCE)),
        "successful_optimizer_starts": successful,
    }
    prediction_frame = dataset.frame[["run_name", "phase_correspondence_key", "policy_family", "source_panel"]].copy()
    prediction_frame["model"] = model_name
    prediction_frame["target"] = target
    prediction_frame["observed"] = dataset.y
    prediction_frame["oof_prediction"] = prediction
    optimum_record = {
        "model": model_name,
        "target": target,
        "domain_names": dataset.domain_names,
        "weights": optimum.tolist(),
        "epochs": exposure.tolist(),
    }
    sweep_record = {
        "selected_config": {
            "variant": config.variant.value,
            "shape": asdict(config.shape),
            "l2": config.l2,
            "residual_shrink": config.residual_shrink,
            "corrections": None if corrections is None else asdict(corrections),
        },
        "selection": sweep,
    }
    return (
        row,
        pd.DataFrame(fold_rows),
        prediction_frame,
        {
            "optimum": optimum_record,
            "sweep": sweep_record,
        },
    )


def write_report(metrics: pd.DataFrame, output_dir: Path) -> Path:
    columns = [
        "model",
        "target",
        "n_tied",
        "oof_rmse",
        "frozen_reference_rmse",
        "relative_rmse_to_reference",
        "passes_five_percent_oof_gate",
        "oof_spearman",
        "observed_on_predicted_slope",
        "selected_power",
        "selected_penalty_threshold",
        "selected_l2",
        "selected_residual_shrink",
        "bounded_link",
        "predicted_optimum_bpb",
        "observed_best_tied_bpb",
        "optimum_l1_to_observed_best",
        "median_fold_to_full_optimum_l1",
        "maximum_fold_to_full_optimum_l1",
        "maximum_optimum_weight",
        "maximum_optimum_epochs",
        "near_zero_optimum_weights",
    ]
    lines = [
        "# Physical aggregate-spine audit",
        "",
        "Each model sees only physically tied policies. Its state is observed total materialized epochs;",
        "retention, forgetting, late utility, and phase shift are absent by construction.",
        "The family-composition variant replaces additive bucket and family benefits with family saturation",
        "and within-family exposure shares while retaining family overexposure and member-replay harm.",
        "The bounded-HPR variant uses the existing log-deficit response link with a training-fold floor;",
        "it changes the response geometry but not the physical epoch state.",
        "",
        metrics[columns].to_markdown(index=False),
        "",
        "The frozen references are the strongest maintained independent one-phase Observatory RMSEs.",
        "They were fit on the 280-row one-phase panel rather than this 282-row matched panel, so the",
        "ratio is an acceptance threshold rather than a paired rowwise comparison.",
        "",
        "Raw optima contain no KL, trust region, epoch cap, or other deployment regularization.",
    ]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    folds = []
    predictions = []
    details = {}
    for model_name, (variant, corrections) in AGGREGATE_SPECS.items():
        for target_index, target in enumerate(TARGETS):
            row, fold_frame, prediction_frame, detail = audit_target(
                model_name,
                variant,
                corrections,
                target,
                args.seed + target_index,
            )
            rows.append(row)
            folds.append(fold_frame)
            predictions.append(prediction_frame)
            details[f"{model_name}:{target}"] = detail
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(args.output_dir / "fold_optima.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "details.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")
    report = write_report(metrics, args.output_dir)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
