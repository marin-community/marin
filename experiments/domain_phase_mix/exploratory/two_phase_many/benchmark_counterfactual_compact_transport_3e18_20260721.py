# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.7",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Test finite counterfactual retained-state transport on Delphi 3e18.

The independently trained one-phase swarm identifies aggregate-mixture quality.
The original two-phase fit swarm identifies only the change from an exact
aggregate-matched tied policy. This avoids asking one response head to infer
aggregate quality and phase ordering from the same sparse two-phase design.

For aggregate mixture ``a = alpha0*w0 + alpha1*w1``, define

    z_i(w) = exp(-lambda * (1 - w1_i)) * e0_i + eta * e1_i
    s_i(w) = 1 - exp(-(rho * z_i(w))**p)

and let ``w_tied=(a,a)``. The model is

    L(w) = F_1p(a) - gamma * sum_i A_i [s_i(w) - s_i(w_tied)].

``F_1p`` and ``A_i,rho,p`` are fitted only on the 238 coordinate-independent
one-phase checkpoints. ``lambda,eta,gamma`` are fitted only on the original
280-row two-phase swarm. The correction is exactly zero for a tied policy.
Physical replay stays in ``F_1p`` because actual and tied schedules have the
same aggregate simulated exposure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_phase_policy_sample_efficiency_20260721 as sample_eff,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as compact,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/counterfactual_compact_transport_3e18_20260721"
TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
SEEDS = (0, 1, 2)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}
LOG_ETA_BOUNDS = (math.log(0.1), math.log(20.0))
LAMBDA_BOUNDS = (0.0, 8.0)
SHAPE_STARTS = tuple(
    np.asarray([math.log(eta), forgetting], dtype=float)
    for eta in (0.5, 1.0, 4.0, 10.0)
    for forgetting in (0.0, 1.0, 3.0)
)
HYBRID_SERIES = "delphi_3e18_hybrid_phase_ordering_validation_20260720"


@dataclass(frozen=True)
class TransportShape:
    """Dimensionless phase-transition parameters."""

    late_multiplier: float
    forgetting_rate: float
    transport_strength: float


@dataclass(frozen=True)
class TransportModel:
    """One-phase Compact spine plus zero-at-tie state transport."""

    aggregate_model: compact.FittedModel
    shape: TransportShape
    alpha0: float
    alpha1: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        tied = aggregate_tied_policy(weights, self.alpha0, self.alpha1)
        aggregate_prediction = self.aggregate_model.predict(tied)
        delta = transport_basis(
            self.aggregate_model,
            np.asarray(weights, dtype=float),
            self.alpha0,
            self.alpha1,
            self.shape.late_multiplier,
            self.shape.forgetting_rate,
        )
        return aggregate_prediction + self.shape.transport_strength * delta


@dataclass(frozen=True)
class RawOptimum:
    """Unregularized optimum and basic solver diagnostics."""

    weights: np.ndarray
    predicted_bpb: float
    converged: bool
    finite_starts: int
    successful_starts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--optimizer-starts", type=int, default=12)
    return parser.parse_args()


def aggregate_tied_policy(weights: np.ndarray, alpha0: float, alpha1: float) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    aggregate = alpha0 * values[:, 0, :] + alpha1 * values[:, 1, :]
    return np.stack([aggregate, aggregate], axis=1)


def retained_signal(
    aggregate_model: compact.FittedModel,
    weights: np.ndarray,
    late_multiplier: float,
    forgetting_rate: float,
) -> np.ndarray:
    e0 = weights[:, 0, :] * aggregate_model.c0[None, :]
    e1 = weights[:, 1, :] * aggregate_model.c1[None, :]
    retained = np.exp(-forgetting_rate * (1.0 - weights[:, 1, :])) * e0
    retained += late_multiplier * e1
    return compact.response_link(
        retained,
        aggregate_model.shape.rate,
        aggregate_model.shape.power,
        compact.ResponseKind.WEIBULL,
    )


def transport_basis(
    aggregate_model: compact.FittedModel,
    weights: np.ndarray,
    alpha0: float,
    alpha1: float,
    late_multiplier: float,
    forgetting_rate: float,
) -> np.ndarray:
    """Return BPB-valued finite state transport relative to the tied counterfactual."""
    tied = aggregate_tied_policy(weights, alpha0, alpha1)
    actual_signal = retained_signal(aggregate_model, weights, late_multiplier, forgetting_rate)
    tied_signal = retained_signal(aggregate_model, tied, late_multiplier, forgetting_rate)
    return -((actual_signal - tied_signal) @ aggregate_model.signal_coef)


def profiled_strength(delta: np.ndarray, residual_target: np.ndarray) -> float:
    denominator = float(delta @ delta)
    if denominator < 1e-15:
        return 0.0
    return max(0.0, float(delta @ residual_target / denominator))


def fit_transport(
    aggregate_model: compact.FittedModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    *,
    allow_forgetting: bool,
    fixed_transport_strength: float | None = None,
) -> TransportModel:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    weights = dataset.weights[indices]
    target = dataset.y[indices]
    tied = aggregate_tied_policy(weights, alpha0, alpha1)
    aggregate_prediction = aggregate_model.predict(tied)
    residual_target = target - aggregate_prediction

    def decode(theta: np.ndarray) -> tuple[float, float]:
        late_multiplier = float(np.exp(theta[0]))
        forgetting_rate = float(theta[1]) if allow_forgetting else 0.0
        return late_multiplier, forgetting_rate

    def objective(theta: np.ndarray) -> float:
        late_multiplier, forgetting_rate = decode(theta)
        delta = transport_basis(
            aggregate_model,
            weights,
            alpha0,
            alpha1,
            late_multiplier,
            forgetting_rate,
        )
        strength = (
            profiled_strength(delta, residual_target) if fixed_transport_strength is None else fixed_transport_strength
        )
        residual = aggregate_prediction + strength * delta - target
        return float(np.mean(residual**2))

    bounds = [LOG_ETA_BOUNDS, LAMBDA_BOUNDS if allow_forgetting else (0.0, 0.0)]
    starts = SHAPE_STARTS if allow_forgetting else tuple(start * np.asarray([1.0, 0.0]) for start in SHAPE_STARTS)
    best_value = float("inf")
    best_theta = starts[0]
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
        value = float(result.fun) if np.isfinite(result.fun) else objective(start)
        theta = np.asarray(result.x, dtype=float) if np.isfinite(result.x).all() else start
        if value < best_value:
            best_value = value
            best_theta = theta
    late_multiplier, forgetting_rate = decode(best_theta)
    delta = transport_basis(
        aggregate_model,
        weights,
        alpha0,
        alpha1,
        late_multiplier,
        forgetting_rate,
    )
    strength = (
        profiled_strength(delta, residual_target) if fixed_transport_strength is None else fixed_transport_strength
    )
    return TransportModel(
        aggregate_model=aggregate_model,
        shape=TransportShape(late_multiplier, forgetting_rate, strength),
        alpha0=alpha0,
        alpha1=alpha1,
    )


def independent_one_phase_dataset(
    target: str,
    reference: pooled.Dataset,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
) -> pooled.Dataset:
    single, _indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    mask = single.frame["disposition"].eq("scheduled_new_training").to_numpy()
    if int(mask.sum()) != 238:
        raise ValueError(f"Expected 238 independent one-phase policies, found {int(mask.sum())}")
    return pooled.Dataset(
        name=f"delphi_3e18_independent_single_{target}",
        frame=single.frame.loc[mask].reset_index(drop=True),
        y=single.y[mask],
        weights=single.weights[mask],
        c0=single.c0,
        c1=single.c1,
        domain_names=list(single.domain_names),
    )


def fit_aggregate_model(target: str, single: pooled.Dataset) -> compact.FittedModel:
    spec = sample_eff.frozen_spec(target, observatory.SINGLE_PHASE, "compact_retained_state")
    return observatory.compact_fit(
        single,
        np.arange(single.n),
        float(spec.tuning["l2"]),
        observatory.SINGLE_PHASE,
    )


def cv_predictions(
    target: str,
    reference: pooled.Dataset,
    aggregate_model: compact.FittedModel,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    averaged: dict[str, list[np.ndarray]] = {
        "aggregate_only": [],
        "counterfactual_eta": [],
        "counterfactual_revisit": [],
        "counterfactual_revisit_unit": [],
    }
    for seed in seeds:
        predictions = {name: np.full(reference.n, np.nan) for name in averaged}
        for fold_id, (train, test) in enumerate(observatory.folds(reference, seed)):
            aggregate_only = TransportModel(
                aggregate_model,
                TransportShape(1.0, 0.0, 0.0),
                *observatory.phase_fractions(reference),
            )
            fitted = {
                "aggregate_only": aggregate_only,
                "counterfactual_eta": fit_transport(
                    aggregate_model,
                    reference,
                    train,
                    allow_forgetting=False,
                ),
                "counterfactual_revisit": fit_transport(
                    aggregate_model,
                    reference,
                    train,
                    allow_forgetting=True,
                ),
                "counterfactual_revisit_unit": fit_transport(
                    aggregate_model,
                    reference,
                    train,
                    allow_forgetting=True,
                    fixed_transport_strength=1.0,
                ),
            }
            for name, model in fitted.items():
                predictions[name][test] = model.predict(reference.weights[test])
                parameter_rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "fold": fold_id,
                        "model": name,
                        **asdict(model.shape),
                    }
                )
        for name, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise ValueError(f"Incomplete OOF prediction for {target}/{name}/seed={seed}")
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "model": name,
                    **sample_eff.metrics(reference.y, prediction),
                }
            )
            averaged[name].append(prediction)
    return (
        pd.DataFrame(rows),
        pd.DataFrame(parameter_rows),
        {name: np.mean(values, axis=0) for name, values in averaged.items()},
    )


def two_phase_evaluation_pool(
    reference: pooled.Dataset,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
    target: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mask = (
        heldout_frame["policy_class"].eq("two_phase")
        & heldout_frame["fit_panel_overlap"].eq("coordinate_disjoint")
        & heldout_frame["training_state"].eq("finished")
        & heldout_frame["checkpoint_declared_complete"].eq(1)
    ).to_numpy()
    pool = sample_eff.coordinate_pool(
        heldout_frame.loc[mask].reset_index(drop=True),
        heldout_weights[mask],
    )
    return (
        pool.frame,
        pool.weights,
        pool.frame[TARGET_COLUMNS[target]].to_numpy(dtype=float),
    )


def evaluation_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    hybrid = frame["training_series"].str.contains(HYBRID_SERIES, regex=False).to_numpy()
    adversarial = frame["training_series"].str.contains("adversarial_stress", regex=False).to_numpy()
    random_population = frame["training_series"].str.contains("frontier_random_phase_population", regex=False).to_numpy()
    fibers = frame["training_series"].str.contains("frontier_phase_fiber", regex=False).to_numpy()
    return {
        "all_two_phase": np.ones(len(frame), dtype=bool),
        "historical_without_hybrid": ~hybrid,
        "hybrid_phase_ordering": hybrid,
        "adversarial_stress": adversarial,
        "frontier_random_population": random_population,
        "frontier_phase_fibers": fibers,
    }


def full_models(
    target: str,
    reference: pooled.Dataset,
    aggregate_model: compact.FittedModel,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "aggregate_only": TransportModel(
            aggregate_model,
            TransportShape(1.0, 0.0, 0.0),
            *observatory.phase_fractions(reference),
        ),
        "counterfactual_eta": fit_transport(
            aggregate_model,
            reference,
            np.arange(reference.n),
            allow_forgetting=False,
        ),
        "counterfactual_revisit": fit_transport(
            aggregate_model,
            reference,
            np.arange(reference.n),
            allow_forgetting=True,
        ),
        "counterfactual_revisit_unit": fit_transport(
            aggregate_model,
            reference,
            np.arange(reference.n),
            allow_forgetting=True,
            fixed_transport_strength=1.0,
        ),
    }
    for model_id in ("compact_retained_state", "effective_exposure", "separate_heads"):
        spec = sample_eff.frozen_spec(target, observatory.TWO_PHASE, model_id)
        models[model_id] = (sample_eff.fit_frozen_model(reference, spec), spec)
    return models


def predict_named(model: Any, reference: pooled.Dataset, weights: np.ndarray) -> np.ndarray:
    if isinstance(model, tuple):
        fitted, spec = model
        return sample_eff.predict_frozen_model(fitted, reference, spec, weights)
    return np.asarray(model.predict(weights), dtype=float)


def heldout_metrics(
    target: str,
    reference: pooled.Dataset,
    frame: pd.DataFrame,
    weights: np.ndarray,
    observed: np.ndarray,
    models: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    masks = evaluation_masks(frame)
    for name, model in models.items():
        prediction = predict_named(model, reference, weights)
        local = frame[["heldout_id", "training_series", "objective", "policy_class"]].copy()
        local["target"] = target
        local["model"] = name
        local["observed"] = observed
        local["predicted"] = prediction
        local["residual"] = prediction - observed
        prediction_rows.append(local)
        for stratum, mask in masks.items():
            if int(mask.sum()) < 3:
                continue
            metric_rows.append(
                {
                    "target": target,
                    "model": name,
                    "stratum": stratum,
                    **sample_eff.metrics(observed[mask], prediction[mask]),
                }
            )
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def logits_to_weights(logits: np.ndarray, domains: int) -> np.ndarray:
    values = np.asarray(logits, dtype=float).reshape(2, domains)
    values -= values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=1, keepdims=True)


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    values = np.log(np.maximum(np.asarray(weights, dtype=float), 1e-12))
    return (values - values.mean(axis=1, keepdims=True)).ravel()


def optimize_raw(
    model: TransportModel,
    reference: pooled.Dataset,
    seed: int,
    starts: int,
) -> RawOptimum:
    alpha0, _alpha1 = observatory.phase_fractions(reference)
    proportional = observatory.natural_weights(reference, alpha0)
    initial = [
        np.stack([proportional, proportional]),
        reference.weights[int(np.argmin(reference.y))],
        reference.weights[int(np.argmin(model.predict(reference.weights)))],
    ]
    rng = np.random.default_rng(seed)
    while len(initial) < starts:
        concentration = (0.25, 1.0, 4.0)[len(initial) % 3]
        initial.append(
            np.stack(
                [
                    rng.dirichlet(np.full(reference.m, concentration)),
                    rng.dirichlet(np.full(reference.m, concentration)),
                ]
            )
        )

    def objective(logits: np.ndarray) -> float:
        weights = logits_to_weights(logits, reference.m)
        return float(model.predict(weights[None, :, :])[0])

    candidates: list[tuple[float, np.ndarray, bool]] = []
    for weights in initial[:starts]:
        result = minimize(
            objective,
            weights_to_logits(weights),
            method="L-BFGS-B",
            options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            candidates.append((float(result.fun), np.asarray(result.x), bool(result.success)))
    if not candidates:
        raise RuntimeError("No finite raw optimum")
    best = min(candidates, key=lambda item: item[0])
    return RawOptimum(
        weights=logits_to_weights(best[1], reference.m),
        predicted_bpb=best[0],
        converged=best[2],
        finite_starts=len(candidates),
        successful_starts=sum(candidate[2] for candidate in candidates),
    )


def optimum_record(
    target: str,
    name: str,
    model: TransportModel,
    optimum: RawOptimum,
    reference: pooled.Dataset,
) -> dict[str, Any]:
    alpha0, alpha1 = observatory.phase_fractions(reference)
    aggregate = alpha0 * optimum.weights[0] + alpha1 * optimum.weights[1]
    exposure = optimum.weights[0] * reference.c0 + optimum.weights[1] * reference.c1
    proportional = observatory.natural_weights(reference, alpha0)
    return {
        "target": target,
        "model": name,
        "predicted_bpb": optimum.predicted_bpb,
        "converged": optimum.converged,
        "finite_starts": optimum.finite_starts,
        "successful_starts": optimum.successful_starts,
        "late_multiplier": model.shape.late_multiplier,
        "forgetting_rate": model.shape.forgetting_rate,
        "transport_strength": model.shape.transport_strength,
        "max_bucket_weight": float(optimum.weights.max()),
        "max_simulated_epochs": float(exposure.max()),
        "phase_total_variation": float(0.5 * np.abs(optimum.weights[0] - optimum.weights[1]).sum()),
        "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - proportional).sum()),
        "support_distance": sample_eff.standardized_support_distance(reference, optimum.weights),
        "phase_0_weights_json": json.dumps(optimum.weights[0].tolist(), separators=(",", ":")),
        "phase_1_weights_json": json.dumps(optimum.weights[1].tolist(), separators=(",", ":")),
    }


def plot_calibration(predictions: pd.DataFrame, output_dir: Path) -> None:
    models = [
        "compact_retained_state",
        "effective_exposure",
        "aggregate_only",
        "counterfactual_eta",
        "counterfactual_revisit",
    ]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable · all two-phase",
            "Table-9 · all two-phase",
            "Uncheatable · hybrid panel",
            "Table-9 · hybrid panel",
        ),
    )
    colors = dict(zip(models, ["#a50026", "#d73027", "#fdae61", "#66bd63", "#006837"], strict=True))
    panels = (("uncheatable", False), ("table9", False), ("uncheatable", True), ("table9", True))
    for panel_index, (target, hybrid_only) in enumerate(panels):
        row = panel_index // 2 + 1
        column = panel_index % 2 + 1
        target_frame = predictions.loc[predictions["target"].eq(target)]
        if hybrid_only:
            target_frame = target_frame.loc[target_frame["training_series"].str.contains(HYBRID_SERIES, regex=False)]
        for model in models:
            local = target_frame.loc[target_frame["model"].eq(model)]
            figure.add_trace(
                go.Scatter(
                    x=local["observed"],
                    y=local["predicted"],
                    mode="markers",
                    name=model,
                    legendgroup=model,
                    showlegend=panel_index == 0,
                    marker={"size": 6, "opacity": 0.55, "color": colors[model]},
                    hovertemplate=(
                        "%{customdata[0]}<br>observed=%{x:.5f}<br>predicted=%{y:.5f}<extra>%{fullData.name}</extra>"
                    ),
                    customdata=local[["heldout_id"]].to_numpy(),
                ),
                row=row,
                col=column,
            )
        minimum = float(min(target_frame["observed"].min(), target_frame["predicted"].min()))
        maximum = float(max(target_frame["observed"].max(), target_frame["predicted"].max()))
        identity = go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            line={"color": "#64748b", "dash": "dash"},
            showlegend=False,
        )
        figure.add_trace(
            identity,
            row=row,
            col=column,
        )
    figure.update_xaxes(title_text="Observed BPB")
    figure.update_yaxes(title_text="Predicted BPB")
    figure.update_layout(
        template="plotly_white",
        title="Counterfactual retained-state transport: frozen 3e18 development calibration",
        width=1450,
        height=1050,
        legend={"orientation": "h", "y": -0.08},
    )
    figure.write_html(output_dir / "heldout_calibration.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_registry(output_dir: Path) -> None:
    rows = [
        {
            "family": "counterfactual_retained_state_transport",
            "variant": "aggregate_only",
            "materially_new_mechanism": "none; exact fitted single-phase spine",
            "additional_degrees_of_freedom": 0,
            "status": "control",
        },
        {
            "family": "counterfactual_retained_state_transport",
            "variant": "counterfactual_eta",
            "materially_new_mechanism": (
                "finite bounded state transport from aggregate-matched tied policy using relative late evidence"
            ),
            "additional_degrees_of_freedom": 2,
            "status": "frozen_local_screen",
        },
        {
            "family": "counterfactual_retained_state_transport",
            "variant": "counterfactual_revisit",
            "materially_new_mechanism": (
                "finite bounded state transport with revisit-dependent survival of phase-0 evidence"
            ),
            "additional_degrees_of_freedom": 3,
            "status": "frozen_local_screen",
        },
        {
            "family": "counterfactual_retained_state_transport",
            "variant": "counterfactual_revisit_unit",
            "materially_new_mechanism": "same transition with transport strength fixed to the physical unit scale",
            "additional_degrees_of_freedom": 2,
            "status": "nested_ablation",
        },
    ]
    pd.DataFrame(rows).to_csv(output_dir / "approach_registry.csv", index=False)


def algebraic_audit(
    target: str,
    reference: pooled.Dataset,
    models: dict[str, Any],
) -> pd.DataFrame:
    """Verify the exact tied restriction and conserved physical exposure."""
    alpha0, alpha1 = observatory.phase_fractions(reference)
    rng = np.random.default_rng(20260721)
    weights = np.stack(
        [
            rng.dirichlet(np.ones(reference.m), size=64),
            rng.dirichlet(np.ones(reference.m), size=64),
        ],
        axis=1,
    )
    tied = aggregate_tied_policy(weights, alpha0, alpha1)
    actual_exposure = weights[:, 0, :] * reference.c0 + weights[:, 1, :] * reference.c1
    tied_exposure = tied[:, 0, :] * reference.c0 + tied[:, 1, :] * reference.c1
    rows: list[dict[str, Any]] = [
        {
            "target": target,
            "model": "physical_exposure_conservation",
            "max_abs_tied_difference": float(np.max(np.abs(actual_exposure - tied_exposure))),
        }
    ]
    for name in (
        "aggregate_only",
        "counterfactual_eta",
        "counterfactual_revisit",
        "counterfactual_revisit_unit",
    ):
        model = models[name]
        aggregate_prediction = model.aggregate_model.predict(tied)
        rows.append(
            {
                "target": target,
                "model": name,
                "max_abs_tied_difference": float(np.max(np.abs(model.predict(tied) - aggregate_prediction))),
            }
        )
    return pd.DataFrame(rows)


def acceptance_gate(
    heldout: pd.DataFrame,
    optima: pd.DataFrame,
    full_parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the frozen local gate against ordinary two-phase Compact."""
    rows: list[dict[str, Any]] = []
    all_two_phase = heldout.loc[heldout["stratum"].eq("all_two_phase")]
    for target in TARGETS:
        baseline = all_two_phase.loc[
            all_two_phase["target"].eq(target) & all_two_phase["model"].eq("compact_retained_state")
        ].iloc[0]
        for model in (
            "counterfactual_eta",
            "counterfactual_revisit",
            "counterfactual_revisit_unit",
        ):
            metric = all_two_phase.loc[all_two_phase["target"].eq(target) & all_two_phase["model"].eq(model)].iloc[0]
            optimum = optima.loc[optima["target"].eq(target) & optima["model"].eq(model)].iloc[0]
            parameters = full_parameters.loc[
                full_parameters["target"].eq(target) & full_parameters["model"].eq(model)
            ].iloc[0]
            rmse_preserved = bool(metric["rmse"] <= baseline["rmse"] * 1.05)
            regret_preserved = bool(metric["regret_at_1"] <= baseline["regret_at_1"] + 0.002)
            optimism_preserved = bool(metric["optimism_gt_0p05"] <= baseline["optimism_gt_0p05"])
            calibration_improved = bool(metric["calibration_error"] <= baseline["calibration_error"])
            parameters_interior = bool(
                parameters["late_multiplier"] < math.exp(LOG_ETA_BOUNDS[1]) * 0.999
                and parameters["forgetting_rate"] < LAMBDA_BOUNDS[1] * 0.999
            )
            plausible_raw_optimum = bool(
                optimum["support_distance"] <= 5.0
                and optimum["aggregate_tv_to_proportional"] <= 0.5
                and optimum["max_simulated_epochs"] <= 20.0
            )
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "heldout_rmse_ratio_to_compact": metric["rmse"] / baseline["rmse"],
                    "heldout_regret_delta_to_compact": (metric["regret_at_1"] - baseline["regret_at_1"]),
                    "optimism_count_delta_to_compact": (metric["optimism_gt_0p05"] - baseline["optimism_gt_0p05"]),
                    "calibration_error_delta_to_compact": (metric["calibration_error"] - baseline["calibration_error"]),
                    "rmse_preserved": rmse_preserved,
                    "regret_preserved": regret_preserved,
                    "optimism_preserved": optimism_preserved,
                    "calibration_improved": calibration_improved,
                    "parameters_interior": parameters_interior,
                    "plausible_raw_optimum": plausible_raw_optimum,
                    "passed": all(
                        (
                            rmse_preserved,
                            regret_preserved,
                            optimism_preserved,
                            calibration_improved,
                            parameters_interior,
                            plausible_raw_optimum,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_data_use_ledger(output_dir: Path) -> None:
    rows = []
    for variant in (
        "counterfactual_eta",
        "counterfactual_revisit",
        "counterfactual_revisit_unit",
    ):
        rows.append(
            {
                "round": "20260721_counterfactual_compact_transport",
                "candidate": variant,
                "hybrid_outcomes_inspected_before_proposal": True,
                "directly_tuned_on_hybrid_or_adversarial_targets": False,
                "mechanism_inspiration": (
                    "effective-exposure phase ordering transferred locally on Uncheatable while Compact "
                    "retained state had the simplest plausible raw optimum"
                ),
                "frozen_inputs": (
                    "238 independent one-phase policies for aggregate; original 280 two-phase fit rows for transition"
                ),
                "evaluation_boundary": (
                    "all coordinate-disjoint two-phase development policies, including hybrid, read only "
                    "after equations and bounds were frozen"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "data_use_ledger.csv", index=False)


def write_report(
    cv_summary: pd.DataFrame,
    parameters: pd.DataFrame,
    heldout: pd.DataFrame,
    optima: pd.DataFrame,
    gate: pd.DataFrame,
    algebraic: pd.DataFrame,
    output_dir: Path,
) -> None:
    all_heldout = heldout.loc[heldout["stratum"].eq("all_two_phase")].copy()
    hybrid = heldout.loc[heldout["stratum"].eq("hybrid_phase_ordering")].copy()
    parameter_summary = parameters.groupby(["target", "model"], as_index=False).agg(
        late_multiplier_median=("late_multiplier", "median"),
        late_multiplier_min=("late_multiplier", "min"),
        late_multiplier_max=("late_multiplier", "max"),
        forgetting_rate_median=("forgetting_rate", "median"),
        forgetting_rate_min=("forgetting_rate", "min"),
        forgetting_rate_max=("forgetting_rate", "max"),
        transport_strength_median=("transport_strength", "median"),
        transport_strength_min=("transport_strength", "min"),
        transport_strength_max=("transport_strength", "max"),
    )
    report = "\n".join(
        [
            "# Counterfactual Compact state-transport screen",
            "",
            "## Verdict",
            "",
            "**Rejected.** Finite retained-state transport improves fit-swarm OOF but does not preserve the "
            "Compact heldout frontier. The revisit rate repeatedly reaches its upper bound, all candidate raw "
            "optima lie far outside empirical support, and no variant passes the frozen local gate on either "
            "target. This is evidence that the locally validated effective-exposure ordering signal cannot be "
            "turned into a globally additive state credit without reviving optimum-region optimism.",
            "",
            "## Boundary",
            "",
            "This is exposed local development evidence. The 238 coordinate-independent one-phase checkpoints "
            "fit the aggregate spine. The original 280 two-phase fit checkpoints fit the transition. No "
            "heldout target value selects a form, bound, or hyperparameter. The newly appended hybrid panel is "
            "evaluated only after the batch is frozen.",
            "",
            "## Mechanism",
            "",
            r"$$a=\alpha_0w^{(0)}+\alpha_1w^{(1)},\quad z_i=e^{-\lambda(1-w_i^{(1)})}e_i^{(0)}+\eta e_i^{(1)},$$",
            "",
            r"$$\widehat L=F_{1p}(a)-\gamma\sum_i A_i\left[S_i(z_i(w))-S_i(z_i(a,a))\right].$$",
            "",
            "The one-phase restriction is exact because the bracket is zero when the phases are tied. The "
            "response is a finite state difference, not a Taylor phase head or output calibrator. Rates are "
            "dimensionless except the inherited Compact Weibull rate, which has inverse-epoch units.",
            "",
            "## Algebraic audit",
            "",
            algebraic.to_markdown(index=False, floatfmt=".3e"),
            "",
            "## Grouped OOF on the original two-phase fit swarm",
            "",
            cv_summary.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Parameter stability",
            "",
            parameter_summary.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Coordinate-disjoint two-phase development archive",
            "",
            all_heldout.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Hybrid phase-ordering panel",
            "",
            hybrid.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Raw optimum audit",
            "",
            optima.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Frozen acceptance gate",
            "",
            gate.to_markdown(index=False, floatfmt=".6f"),
            "",
            "The transition is promoted only if it preserves Compact heldout RMSE and Regret@1, adds no "
            "optimism errors, improves calibration, keeps its fitted transition interior, and produces a "
            "plausible raw optimum. A deployment regularizer is not allowed to rescue a failed raw surface.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = tuple(part.strip() for part in args.targets.split(",") if part.strip())
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    unknown = sorted(set(targets).difference(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets {unknown}")
    write_registry(args.output_dir)

    cv_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    heldout_metric_frames: list[pd.DataFrame] = []
    heldout_prediction_frames: list[pd.DataFrame] = []
    optimum_rows: list[dict[str, Any]] = []
    full_parameter_rows: list[dict[str, Any]] = []
    algebraic_frames: list[pd.DataFrame] = []
    for target in targets:
        print(f"Loading {target}", flush=True)
        reference = observatory.load_delphi_3e18_fit_dataset(target)
        heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
        single = independent_one_phase_dataset(target, reference, heldout_frame, heldout_weights)
        aggregate_model = fit_aggregate_model(target, single)
        cv, parameters, _predictions = cv_predictions(target, reference, aggregate_model, seeds)
        cv_frames.append(cv)
        parameter_frames.append(parameters)
        frame, weights, observed = two_phase_evaluation_pool(
            reference,
            heldout_frame,
            heldout_weights,
            target,
        )
        models = full_models(target, reference, aggregate_model)
        algebraic_frames.append(algebraic_audit(target, reference, models))
        heldout_metrics_frame, heldout_predictions = heldout_metrics(
            target,
            reference,
            frame,
            weights,
            observed,
            models,
        )
        heldout_metric_frames.append(heldout_metrics_frame)
        heldout_prediction_frames.append(heldout_predictions)
        for name in ("aggregate_only", "counterfactual_eta", "counterfactual_revisit", "counterfactual_revisit_unit"):
            model = models[name]
            full_parameter_rows.append({"target": target, "model": name, **asdict(model.shape)})
            optimum = optimize_raw(model, reference, seed=20260721, starts=args.optimizer_starts)
            optimum_rows.append(optimum_record(target, name, model, optimum, reference))

    cv = pd.concat(cv_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    heldout_metrics_frame = pd.concat(heldout_metric_frames, ignore_index=True)
    heldout_predictions = pd.concat(heldout_prediction_frames, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    full_parameters = pd.DataFrame(full_parameter_rows)
    algebraic = pd.concat(algebraic_frames, ignore_index=True)
    cv_summary = cv.groupby(["target", "model"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rmse=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        spearman=("spearman", "mean"),
        regret_at_1=("regret_at_1", "mean"),
        regret_at_3=("regret_at_3", "mean"),
        regret_at_5=("regret_at_5", "mean"),
        calibration_slope=("calibration_slope", "mean"),
        optimism_gt_0p05=("optimism_gt_0p05", "mean"),
        worst_optimism=("worst_optimism", "mean"),
    )
    cv.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    cv_summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    full_parameters.to_csv(args.output_dir / "full_fit_parameters.csv", index=False)
    heldout_metrics_frame.to_csv(args.output_dir / "heldout_metrics.csv", index=False)
    heldout_predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)
    algebraic.to_csv(args.output_dir / "algebraic_audit.csv", index=False)
    gate = acceptance_gate(heldout_metrics_frame, optima, full_parameters)
    gate.to_csv(args.output_dir / "acceptance_gate.csv", index=False)
    write_data_use_ledger(args.output_dir)
    plot_calibration(heldout_predictions, args.output_dir)
    write_report(
        cv_summary,
        parameters,
        heldout_metrics_frame,
        optima,
        gate,
        algebraic,
        args.output_dir,
    )
    print(cv_summary.to_string(index=False), flush=True)
    print(
        heldout_metrics_frame.loc[heldout_metrics_frame["stratum"].eq("all_two_phase")].to_string(index=False),
        flush=True,
    )
    print(optima.to_string(index=False), flush=True)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
