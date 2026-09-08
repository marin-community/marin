# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "joblib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Falsify an adaptive-optimizer-memory explanation of phase order.

The phase-0 policy initializes a per-bucket second-moment state. During phase
1, that state relaxes toward the phase-1 policy:

    s_i(u) = w1_i + (w0_i - w1_i) exp(-u / tau).

The transient effective acquisition is the integral of
``w1_i / sqrt(s_i(u) + epsilon)``. A frozen one-phase aggregate model supplies
the positive marginal value of each bucket. One nonnegative amplitude maps the
resulting acquisition difference to BPB. The exact tied-policy correction is
zero.

The model is selected within one exposed contrast design and evaluated on the
other. The sealed targeted-pairwise panel is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_aggregate_comparators_20260724 as comparators,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_frontier_control_aggregate_identification_20260724 as aggregate_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_marginal_acquisition_phase_potential_20260724 as marginal,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/optimizer_memory_phase_transition_20260724"
DEFAULT_SEEDS = (20260724, 20260725, 20260726)
MEMORY_TIMES = (0.002, 0.01, 0.05, 0.2, 1.0, 5.0)
STATE_FLOORS = (1e-4, 1e-3, 1e-2, 5e-2)
HUBER_THRESHOLD = 1.345
QUADRATURE_POINTS = 96
NUMERICAL_FLOOR = 1e-12
_GAUSS_NODES, _GAUSS_WEIGHTS = leggauss(QUADRATURE_POINTS)
QUADRATURE_LOCATIONS = 0.5 * (_GAUSS_NODES + 1.0)
QUADRATURE_WEIGHTS = 0.5 * _GAUSS_WEIGHTS


@dataclass(frozen=True)
class Configuration:
    """One frozen optimizer-state transition law."""

    memory_time: float
    state_floor: float


@dataclass(frozen=True)
class FeatureSet:
    """Predicted plus/minus BPB features for one configuration."""

    plus: np.ndarray
    minus: np.ndarray

    @property
    def odd(self) -> np.ndarray:
        return 0.5 * (self.plus - self.minus)

    @property
    def even(self) -> np.ndarray:
        return 0.5 * (self.plus + self.minus)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    return parser.parse_args()


def huber(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    quadratic = absolute <= HUBER_THRESHOLD
    return np.where(
        quadratic,
        0.5 * values**2,
        HUBER_THRESHOLD * (absolute - 0.5 * HUBER_THRESHOLD),
    )


def fit_amplitude(
    design: np.ndarray,
    target: np.ndarray,
    noise: np.ndarray,
) -> float:
    """Fit the single physically nonnegative response amplitude."""

    design_scale = float(np.sqrt(np.mean(design**2)))
    target_scale = float(np.sqrt(np.mean(target**2)))
    if design_scale <= NUMERICAL_FLOOR:
        return 0.0
    upper = max(10.0 * target_scale / design_scale, 1.0)

    def objective(value: float) -> float:
        residual = (value * design - target) / noise
        return float(np.sum(huber(residual)))

    result = minimize_scalar(
        objective,
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Amplitude fit failed: {result.message}")
    return float(result.x)


def aggregate_model(target: str, seed: int) -> Any:
    (
        _reference,
        _heldout_frame,
        _heldout_weights,
        single,
        controls,
        _evaluation_frame,
        _evaluation_weights,
        _observed,
        _clusters,
    ) = comparators.target_data(target)
    training = aggregate_audit.training_dataset(
        target,
        single,
        controls,
        "tied_272_plus_controls",
        seed,
    )
    folds = comparators.strict_protocol.grouped_stratified_folds(training, seed)
    return aggregate_audit.frozen_pooled_fit(training, folds).model


def transient_acquisition(
    phase0: np.ndarray,
    phase1: np.ndarray,
    configuration: Configuration,
) -> np.ndarray:
    """Integrate the phase-1 update magnitude under decaying phase-0 memory."""

    decay = np.exp(-QUADRATURE_LOCATIONS / configuration.memory_time)
    state = phase1[:, None, :] + (phase0 - phase1)[:, None, :] * decay[None, :, None]
    denominator = np.sqrt(np.maximum(state + configuration.state_floor, NUMERICAL_FLOOR))
    integrand = phase1[:, None, :] / denominator
    return np.sum(integrand * QUADRATURE_WEIGHTS[None, :, None], axis=1)


def correction_feature(
    aggregate: np.ndarray,
    contrast: np.ndarray,
    model: Any,
    configuration: Configuration,
) -> np.ndarray:
    """Return the unscaled BPB correction for one signed phase contrast."""

    alpha0 = float(model.phase_fraction)
    alpha1 = 1.0 - alpha0
    phase0 = aggregate - alpha1 * contrast
    phase1 = aggregate + alpha0 * contrast
    if np.min(phase0) < -1e-8 or np.min(phase1) < -1e-8:
        raise ValueError("Infeasible phase policy reconstructed from aggregate and contrast")
    phase0 = np.maximum(phase0, 0.0)
    phase1 = np.maximum(phase1, 0.0)
    transient = transient_acquisition(phase0, phase1, configuration)
    tied = aggregate / np.sqrt(aggregate + configuration.state_floor)
    marginal_value = marginal.marginal_bucket_value(model, aggregate)
    return -alpha1 * np.sum(marginal_value * (transient - tied), axis=1)


def feature_set(
    dataset: marginal.PairDataset,
    model: Any,
    configuration: Configuration,
) -> FeatureSet:
    aggregate, contrast = marginal.aligned_pair_arrays(dataset)
    return FeatureSet(
        plus=correction_feature(aggregate, contrast, model, configuration),
        minus=correction_feature(aggregate, -contrast, model, configuration),
    )


def stacked_arrays(
    dataset: marginal.PairDataset,
    features: FeatureSet,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.concatenate([features.plus[indices], features.minus[indices]])
    target = np.concatenate(
        [
            dataset.odd[indices] + dataset.even[indices],
            -dataset.odd[indices] + dataset.even[indices],
        ]
    )
    noise = np.concatenate([dataset.noise[indices], dataset.noise[indices]])
    return design, target, noise


def group_folds(dataset: marginal.PairDataset, indices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = dataset.frame.iloc[indices]["direction_group"].to_numpy()
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    result = []
    local = np.arange(len(indices))
    for train, test in splitter.split(local, groups=groups):
        result.append((indices[train], indices[test]))
    return result


def select_configuration(
    dataset: marginal.PairDataset,
    features: dict[Configuration, FeatureSet],
    indices: np.ndarray,
) -> Configuration:
    candidates = []
    for configuration, values in features.items():
        prediction = np.full((dataset.n, 2), np.nan, dtype=float)
        scores = []
        for train, test in group_folds(dataset, indices):
            design, target, noise = stacked_arrays(dataset, values, train)
            amplitude = fit_amplitude(design, target, noise)
            prediction[test, 0] = amplitude * values.odd[test]
            prediction[test, 1] = amplitude * values.even[test]
            test_design, test_target, test_noise = stacked_arrays(dataset, values, test)
            scores.append(float(np.mean(huber((amplitude * test_design - test_target) / test_noise))))
        if not np.isfinite(prediction[indices]).all():
            raise ValueError("Incomplete grouped-CV prediction")
        candidates.append((float(np.mean(scores)), configuration))
    return min(candidates, key=lambda value: (value[0], value[1].memory_time, value[1].state_floor))[1]


def metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    zero_rmse = float(np.sqrt(np.mean(observed**2)))
    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    correlation = spearmanr(observed, predicted).statistic
    slope = 0.0
    if np.std(predicted) > NUMERICAL_FLOOR:
        slope = float(np.polyfit(predicted, observed, 1)[0])
    return {
        "n": len(observed),
        "rmse": rmse,
        "zero_rmse": zero_rmse,
        "rmse_ratio": rmse / max(zero_rmse, NUMERICAL_FLOOR),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "calibration_slope": slope,
        "bias": float(np.mean(predicted - observed)),
    }


def evaluate_seed(
    dataset: marginal.PairDataset,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = aggregate_model(dataset.target, seed)
    configurations = [
        Configuration(memory_time, state_floor) for memory_time in MEMORY_TIMES for state_floor in STATE_FLOORS
    ]
    features = {configuration: feature_set(dataset, model, configuration) for configuration in configurations}
    prediction_odd = np.full(dataset.n, np.nan, dtype=float)
    prediction_even = np.full(dataset.n, np.nan, dtype=float)
    selections = []
    panels = dataset.frame["panel"].to_numpy()
    for heldout_panel in sorted(np.unique(panels)):
        train = np.flatnonzero(panels != heldout_panel)
        test = np.flatnonzero(panels == heldout_panel)
        selected = select_configuration(dataset, features, train)
        design, target, noise = stacked_arrays(dataset, features[selected], train)
        amplitude = fit_amplitude(design, target, noise)
        prediction_odd[test] = amplitude * features[selected].odd[test]
        prediction_even[test] = amplitude * features[selected].even[test]
        selections.append(
            {
                "target": dataset.target,
                "seed": seed,
                "heldout_panel": heldout_panel,
                "memory_time": selected.memory_time,
                "state_floor": selected.state_floor,
                "amplitude": amplitude,
                "amplitude_at_boundary": amplitude <= 1e-8,
            }
        )
    if not np.isfinite(prediction_odd).all() or not np.isfinite(prediction_even).all():
        raise ValueError("Incomplete cross-design prediction")

    metrics = []
    for response, observed, predicted in (
        ("odd", dataset.odd, prediction_odd),
        ("even", dataset.even, prediction_even),
        (
            "signed_policy",
            np.concatenate([dataset.odd + dataset.even, -dataset.odd + dataset.even]),
            np.concatenate(
                [
                    prediction_odd + prediction_even,
                    -prediction_odd + prediction_even,
                ]
            ),
        ),
    ):
        metrics.append(
            {
                "target": dataset.target,
                "seed": seed,
                "response": response,
                **metric_row(observed, predicted),
            }
        )
    return metrics, selections


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    summary = (
        metrics.groupby(["target", "response"], as_index=False)
        .agg(
            rmse_ratio_mean=("rmse_ratio", "mean"),
            rmse_ratio_min=("rmse_ratio", "min"),
            rmse_ratio_max=("rmse_ratio", "max"),
            spearman_mean=("spearman", "mean"),
            calibration_slope_mean=("calibration_slope", "mean"),
        )
        .sort_values(["target", "response"])
    )
    boundary_count = int(selections["amplitude_at_boundary"].sum())
    minimum_memory_count = int(np.isclose(selections["memory_time"], min(MEMORY_TIMES)).sum())
    maximum_floor_count = int(np.isclose(selections["state_floor"], max(STATE_FLOORS)).sum())
    odd = summary[summary["response"].eq("odd")]
    signed = summary[summary["response"].eq("signed_policy")]
    gate_pass = bool((odd["rmse_ratio_mean"] < 1.0).all() and (signed["rmse_ratio_mean"] < 1.0).all())
    decision = (
        "**Promote for full evaluation.**"
        if gate_pass
        else ("**Blocked.** The transition does not transfer as an odd phase-order " "mechanism on both targets.")
    )
    report = [
        "# Adaptive-optimizer-memory phase transition",
        "",
        "## Decision",
        "",
        decision,
        "",
        "This is a local exposed-data falsification screen. The sealed targeted-pairwise panel was not loaded.",
        "",
        "## Mechanism",
        "",
        r"The phase-0 mixture initializes a per-bucket second-moment state. During phase 1,",
        "",
        r"$$s_i(u)=w_i^{(1)}+(w_i^{(0)}-w_i^{(1)})e^{-u/\tau},$$",
        "",
        r"and transient acquisition is proportional to",
        r"$\int_0^1 w_i^{(1)}/\sqrt{s_i(u)+\epsilon}\,du$.",
        r"A frozen tied-policy aggregate model supplies bucket marginal values and one",
        r"nonnegative amplitude maps the acquisition difference to BPB. The correction",
        r"is exactly zero when the phases tie.",
        "",
        (
            f"The integral uses {QUADRATURE_POINTS}-point Gauss-Legendre quadrature "
            f"on $[0,1]$; its first node is {QUADRATURE_LOCATIONS.min():.6g}."
        ),
        "",
        "## Cross-design metrics",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Selected transition parameters",
        "",
        selections.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Gate",
        "",
        f"- Positive-amplitude boundary selections: {boundary_count}/{len(selections)}.",
        f"- Minimum-memory-time selections: {minimum_memory_count}/{len(selections)}.",
        f"- Maximum-state-floor selections: {maximum_floor_count}/{len(selections)}.",
        "- Promotion requires cross-design RMSE below the zero-effect baseline for both odd and "
        "signed-policy responses on both targets.",
        "- A gain confined to the even response would duplicate the already retained Fisher "
        "asymmetry-cost term and does not identify phase order.",
        f"- Gate result: {'pass' if gate_pass else 'fail'}.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    datasets = marginal.pair_datasets()
    metric_rows = []
    selection_rows = []
    for dataset in datasets.values():
        for seed in seeds:
            metrics, selections = evaluate_seed(dataset, seed)
            metric_rows.extend(metrics)
            selection_rows.extend(selections)
    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    metrics.to_csv(output_dir / "cross_design_metrics.csv", index=False)
    selections.to_csv(output_dir / "selected_parameters.csv", index=False)
    write_report(output_dir, metrics, selections)
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "memory_times": MEMORY_TIMES,
                "state_floors": STATE_FLOORS,
                "quadrature_points": QUADRATURE_POINTS,
                "quadrature_rule": "Gauss-Legendre on [0, 1]",
                "minimum_quadrature_location": float(QUADRATURE_LOCATIONS.min()),
                "sealed_targeted_pairwise_panel_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
