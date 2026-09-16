# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "plotly", "scikit-learn", "scipy"]
# ///
"""Materialize a Delphi 3e18 one-phase DSP whole-run epoch-cap sweep.

The fitted response is either shared-shape or full per-bucket canonical DSP on
the complete 280-row single-phase panel. Uncheatable and Table-9 are fitted
independently. Each candidate minimizes the unregularized fitted response
subject only to the simplex and a per-bucket cap on materialized epochs over
the full run.

Continuous optima are projected onto the exact 1/2048 runtime mixture grid and
then locally refined while preserving the cap. This script never launches
training; it emits frozen candidate tables, model provenance, and a standalone
review artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import benchmark_dsp_single_phase_ladder_20260824 as dsp  # noqa: E402
import benchmark_single_phase_surrogates_20260824 as base  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import select_delphi_phase0_prefix_candidates_20260824 as prefix_materializer  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_one_phase_dsp_epoch_cap_sweep_20260828"
CAPS = tuple(range(2, 21, 2))
MODEL_VARIANTS = ("shared_shape", "canonical")
TARGETS = (swarm39.UNCHEATABLE, swarm39.TABLE9)
TARGET_LABELS = {
    swarm39.UNCHEATABLE: "Uncheatable",
    swarm39.TABLE9: "Table-9 macro",
}
MODEL_PARTITION_SEEDS = (0, 1, 2)
MODEL_RESTARTS = 8
MODEL_MAXITER = 160
MIXTURE_BLOCK_SIZE = prefix_materializer.MIXTURE_BLOCK_SIZE
ACTIVE_COUNT_SLACK = 1
MONOTONIC_TOLERANCE = 1e-10
REFINE_TOLERANCE = 1e-13
MAX_EXCHANGE_STEPS = 4 * MIXTURE_BLOCK_SIZE
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


@dataclass(frozen=True)
class DspFit:
    """One full-panel canonical DSP fit."""

    partition_seed: int
    rung_name: str
    shape: np.ndarray
    intercept: float
    coefficients: np.ndarray


@dataclass(frozen=True)
class MaterializedCandidate:
    """Continuous and exact-runtime forms of one constrained optimum."""

    target: str
    cap: int
    continuous_weights: np.ndarray
    runtime_counts: np.ndarray
    continuous_prediction: float
    runtime_prediction: float
    exchange_steps: int
    optimizer_successes: int

    @property
    def runtime_weights(self) -> np.ndarray:
        return self.runtime_counts / MIXTURE_BLOCK_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="shared_shape")
    parser.add_argument("--max-cap", type=int, default=max(CAPS))
    parser.add_argument("--experiment-name", default="Delphi 3e18 one-phase DSP whole-run epoch-cap sweep")
    parser.add_argument("--model-restarts", type=int, default=MODEL_RESTARTS)
    parser.add_argument("--model-maxiter", type=int, default=MODEL_MAXITER)
    parser.add_argument("--model-workers", type=int, default=min(2 * MODEL_RESTARTS, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.max_cap < min(CAPS) or args.max_cap % 2:
        parser.error("--max-cap must be an even integer of at least 2")
    if args.model_workers < 1:
        parser.error("--model-workers must be positive")
    return args


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_rung(name: str) -> dsp.Rung:
    return next(rung for rung in dsp.LADDER if rung.name == name)


def fit_models(
    panel: swarm39.Panel,
    target: str,
    *,
    rung_name: str,
    restarts: int,
    maxiter: int,
    workers: int,
    executor: Executor | None = None,
) -> tuple[DspFit, ...]:
    usable = np.isfinite(panel.targets[target])
    train = panel.subset(usable)
    exposure = (train.c0 + train.c1)[None, :] * train.phase0
    response = train.targets[target]
    rung = model_rung(rung_name)
    fits = []
    for partition_seed in MODEL_PARTITION_SEEDS:
        folds = swarm39.mixture_blocked_splits(train, dsp.N_FOLDS, seed=partition_seed)
        shape, intercept, coefficients = dsp.fit_rung(
            exposure,
            response,
            rung,
            folds,
            (),
            seed=partition_seed,
            maxiter=maxiter,
            restarts=restarts,
            workers=workers,
            executor=executor,
        )
        fits.append(
            DspFit(
                partition_seed=partition_seed,
                rung_name=rung_name,
                shape=shape,
                intercept=intercept,
                coefficients=coefficients,
            )
        )
    return tuple(fits)


def model_predictions(models: tuple[DspFit, ...], exposure: np.ndarray) -> np.ndarray:
    exposure = np.atleast_2d(np.asarray(exposure, dtype=float))
    rung_names = {model.rung_name for model in models}
    if len(rung_names) != 1:
        raise ValueError(f"Model ensemble mixes DSP rungs: {sorted(rung_names)}")
    rung = model_rung(next(iter(rung_names)))
    rows = []
    for model in models:
        design = dsp.rung_design(exposure, model.shape, rung, exposure.shape[1])
        rows.append(model.intercept + design @ model.coefficients)
    return np.asarray(rows)


def ensemble_prediction(models: tuple[DspFit, ...], exposure: np.ndarray) -> np.ndarray:
    return model_predictions(models, exposure).mean(axis=0)


def project_capped_simplex(values: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``sum(w)=1`` and ``0 <= w <= upper``."""
    values = np.asarray(values, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if values.shape != upper.shape:
        raise ValueError(f"Projection shape mismatch: {values.shape} != {upper.shape}")
    if np.any(upper < 0.0) or float(upper.sum()) < 1.0 - 1e-12:
        raise ValueError("Epoch cap does not leave a feasible simplex")
    low = float(np.min(values - upper))
    high = float(np.max(values))
    for _ in range(120):
        threshold = 0.5 * (low + high)
        projected = np.clip(values - threshold, 0.0, upper)
        if float(projected.sum()) > 1.0:
            low = threshold
        else:
            high = threshold
    projected = np.clip(values - high, 0.0, upper)
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-10:
        room = upper - projected if residual > 0.0 else projected
        index = int(np.argmax(room))
        projected[index] += residual
    if not np.isclose(projected.sum(), 1.0, atol=1e-10):
        raise ValueError("Capped-simplex projection did not sum to one")
    if np.any(projected < -1e-12) or np.any(projected > upper + 1e-12):
        raise ValueError("Capped-simplex projection violated a bound")
    return projected


def optimization_starts(
    panel: swarm39.Panel,
    target: str,
    models: tuple[DspFit, ...],
    scales: np.ndarray,
    upper: np.ndarray,
    previous: np.ndarray | None,
) -> list[np.ndarray]:
    weights = panel.phase0
    response = panel.targets[target]
    predictions = ensemble_prediction(models, scales[None, :] * weights)
    candidates = [panel.proportional]
    candidates.extend(weights[index] for index in np.argsort(response)[:24])
    candidates.extend(weights[index] for index in np.argsort(predictions)[:24])
    if previous is not None:
        candidates.append(previous)

    rng = np.random.default_rng(20260828)
    for concentration in (8.0, 32.0, 128.0):
        alpha = 1.0 + concentration * panel.proportional
        candidates.extend(rng.dirichlet(alpha) for _ in range(12))

    starts = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        projected = project_capped_simplex(candidate, upper)
        key = tuple(np.round(projected, 12))
        if key in seen:
            continue
        seen.add(key)
        starts.append(projected)
    return starts


def optimize_continuous_start(
    start: np.ndarray,
    *,
    models: tuple[DspFit, ...],
    scales: np.ndarray,
    upper: np.ndarray,
) -> tuple[bool, float, np.ndarray]:
    """Run one independent capped-simplex SLSQP start."""

    def objective(weights: np.ndarray) -> float:
        return float(ensemble_prediction(models, scales[None, :] * weights[None, :])[0])

    constraint = {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, float(limit)) for limit in upper],
        constraints=[constraint],
        options={"ftol": 1e-13, "maxiter": 1_500},
    )
    return bool(result.success), float(result.fun), np.asarray(result.x, dtype=float)


def continuous_optimum(
    models: tuple[DspFit, ...],
    scales: np.ndarray,
    upper: np.ndarray,
    starts: list[np.ndarray],
    executor: Executor | None = None,
) -> tuple[np.ndarray, float, int]:
    def objective(weights: np.ndarray) -> float:
        return float(ensemble_prediction(models, scales[None, :] * weights[None, :])[0])

    best_weights = starts[0]
    best_value = objective(best_weights)
    successes = 0
    optimize = partial(optimize_continuous_start, models=models, scales=scales, upper=upper)
    results = [optimize(start) for start in starts] if executor is None else list(executor.map(optimize, starts))
    for success, value, weights in results:
        if not success:
            continue
        successes += 1
        if value < best_value:
            best_weights = weights
            best_value = value
    if successes == 0:
        raise RuntimeError("No continuous optimization start converged")
    if not np.isclose(best_weights.sum(), 1.0, atol=1e-9):
        raise ValueError("Continuous optimum does not sum to one")
    if np.any(best_weights > upper + 1e-9):
        raise ValueError("Continuous optimum violates the epoch cap")
    return best_weights, best_value, successes


def refine_runtime_counts(
    models: tuple[DspFit, ...],
    scales: np.ndarray,
    initial_counts: np.ndarray,
    maximum_counts: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Apply improving one-count exchanges on the exact runtime grid."""
    current = initial_counts.copy()
    current_value = float(ensemble_prediction(models, scales[None, :] * (current / MIXTURE_BLOCK_SIZE))[0])
    for step in range(MAX_EXCHANGE_STEPS):
        donors = np.flatnonzero(current > 0)
        receivers = np.flatnonzero(current < maximum_counts)
        proposals = []
        moves = []
        for donor in donors:
            for receiver in receivers:
                if donor == receiver:
                    continue
                proposal = current.copy()
                proposal[donor] -= 1
                proposal[receiver] += 1
                proposals.append(proposal)
                moves.append((donor, receiver))
        if not proposals:
            return current, step
        proposal_array = np.asarray(proposals, dtype=float) / MIXTURE_BLOCK_SIZE
        values = ensemble_prediction(models, scales[None, :] * proposal_array)
        choice = int(np.argmin(values))
        if float(values[choice]) >= current_value - REFINE_TOLERANCE:
            return current, step
        donor, receiver = moves[choice]
        current[donor] -= 1
        current[receiver] += 1
        current_value = float(values[choice])
    raise RuntimeError(f"Runtime exchange refinement exceeded {MAX_EXCHANGE_STEPS} steps")


def materialize_target(
    panel: swarm39.Panel,
    target: str,
    models: tuple[DspFit, ...],
    scales: np.ndarray,
    caps: tuple[int, ...],
    executor: Executor | None = None,
) -> list[MaterializedCandidate]:
    candidates = []
    previous_runtime_counts: np.ndarray | None = None
    previous_continuous: np.ndarray | None = None
    previous_runtime_value = np.inf
    for cap in caps:
        upper = np.minimum(1.0, cap / scales)
        starts = optimization_starts(panel, target, models, scales, upper, previous_continuous)
        continuous, continuous_value, successes = continuous_optimum(models, scales, upper, starts, executor)
        maximum_counts = np.floor(upper * MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64)
        if int(maximum_counts.sum()) < MIXTURE_BLOCK_SIZE:
            raise ValueError(f"Cap {cap} is infeasible after runtime quantization")
        initial = prefix_materializer.constrained_counts(continuous, maximum_counts)
        if previous_runtime_counts is not None:
            previous_value = float(
                ensemble_prediction(models, scales[None, :] * (previous_runtime_counts / MIXTURE_BLOCK_SIZE))[0]
            )
            initial_value = float(ensemble_prediction(models, scales[None, :] * (initial / MIXTURE_BLOCK_SIZE))[0])
            if previous_value < initial_value:
                initial = previous_runtime_counts.copy()
        runtime_counts, exchange_steps = refine_runtime_counts(models, scales, initial, maximum_counts)
        runtime_weights = runtime_counts / MIXTURE_BLOCK_SIZE
        runtime_value = float(ensemble_prediction(models, scales[None, :] * runtime_weights[None, :])[0])
        if runtime_value > previous_runtime_value + MONOTONIC_TOLERANCE:
            raise ValueError(
                f"Runtime optimum worsened as cap expanded for {target}: "
                f"{runtime_value:.12f} > {previous_runtime_value:.12f}"
            )
        if not np.array_equal(prefix_materializer.runtime_counts(runtime_weights), runtime_counts):
            raise ValueError("Materialized candidate is unstable under runtime realization")
        realized_epochs = scales * runtime_weights
        if float(realized_epochs.max()) > cap + 1e-10:
            raise ValueError(f"Runtime candidate exceeds cap {cap}: {realized_epochs.max()}")
        candidates.append(
            MaterializedCandidate(
                target=target,
                cap=cap,
                continuous_weights=continuous,
                runtime_counts=runtime_counts,
                continuous_prediction=continuous_value,
                runtime_prediction=runtime_value,
                exchange_steps=exchange_steps,
                optimizer_successes=successes,
            )
        )
        previous_continuous = continuous
        previous_runtime_counts = runtime_counts
        previous_runtime_value = runtime_value
    return candidates


def hellinger(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.square(np.sqrt(first) - np.sqrt(second)).sum()))


def candidate_tables(
    panel: swarm39.Panel,
    candidates: list[MaterializedCandidate],
    models_by_target: dict[str, tuple[DspFit, ...]],
    scales: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    weight_rows = []
    panel_weights = panel.phase0
    for candidate in candidates:
        weights = candidate.runtime_weights
        epochs = scales * weights
        maximum_counts = np.floor(np.minimum(1.0, candidate.cap / scales) * MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64)
        active = candidate.runtime_counts >= maximum_counts - ACTIVE_COUNT_SLACK
        distances = 0.5 * np.abs(panel_weights - weights).sum(axis=1)
        nearest = int(np.argmin(distances))
        per_model = model_predictions(models_by_target[candidate.target], epochs[None, :])[:, 0]
        candidate_id = f"{candidate.target.removesuffix('_bpb')}_cap{candidate.cap:02d}"
        positive = weights > 0.0
        summary = {
            "candidate_id": candidate_id,
            "target": candidate.target,
            "target_label": TARGET_LABELS[candidate.target],
            "epoch_cap": candidate.cap,
            "continuous_predicted_bpb": candidate.continuous_prediction,
            "runtime_predicted_bpb": candidate.runtime_prediction,
            "runtime_minus_continuous_bpb": candidate.runtime_prediction - candidate.continuous_prediction,
            "partition_prediction_sd": float(np.std(per_model, ddof=1)),
            "max_materialized_epoch": float(epochs.max()),
            "cap_slack": float(candidate.cap - epochs.max()),
            "cap_active_buckets": int(active.sum()),
            "support_buckets": int(positive.sum()),
            "effective_buckets": float(np.exp(-np.sum(weights[positive] * np.log(weights[positive])))),
            "tv_to_proportional": float(0.5 * np.abs(weights - panel.proportional).sum()),
            "hellinger_to_proportional": hellinger(weights, panel.proportional),
            "nearest_panel_row_id": str(panel.row_id[nearest]),
            "nearest_panel_tv": float(distances[nearest]),
            "nearest_panel_observed_bpb": float(panel.targets[candidate.target][nearest]),
            "fit_panel_best_observed_bpb": float(np.min(panel.targets[candidate.target])),
            "predicted_minus_fit_panel_best_bpb": float(
                candidate.runtime_prediction - np.min(panel.targets[candidate.target])
            ),
            "largest_bucket": str(panel.buckets[int(np.argmax(weights))]),
            "largest_weight": float(weights.max()),
            "exchange_steps": candidate.exchange_steps,
            "optimizer_successes": candidate.optimizer_successes,
        }
        for model, prediction in zip(models_by_target[candidate.target], per_model, strict=True):
            summary[f"prediction_partition_seed_{model.partition_seed}"] = float(prediction)
        summary_rows.append(summary)
        for index, bucket in enumerate(panel.buckets):
            weight_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": candidate.target,
                    "target_label": TARGET_LABELS[candidate.target],
                    "epoch_cap": candidate.cap,
                    "domain": bucket,
                    "runtime_count": int(candidate.runtime_counts[index]),
                    "weight": float(weights[index]),
                    "proportional_weight": float(panel.proportional[index]),
                    "weight_ratio_to_proportional": float(weights[index] / panel.proportional[index]),
                    "materialized_epochs": float(epochs[index]),
                    "cap_fraction": float(epochs[index] / candidate.cap),
                    "cap_active": bool(active[index]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(weight_rows)


def model_audit(
    panel: swarm39.Panel,
    models_by_target: dict[str, tuple[DspFit, ...]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    heldout = base.single_phase_heldout("delphi_3e18")
    model_rows = []
    scores = {}
    for target, models in models_by_target.items():
        query = heldout.subset(np.isfinite(heldout.targets[target]))
        exposure = (query.c0 + query.c1)[None, :] * query.phase0
        predictions = ensemble_prediction(models, exposure)
        scores[target] = base.score(predictions, query.targets[target])
        for model in models:
            n_buckets = len(panel.buckets)
            if model.rung_name == "canonical":
                log_rates = model.shape[:n_buckets]
                thresholds = model.shape[n_buckets:]
            else:
                log_rates = model.shape[:1]
                thresholds = model.shape[1:2]
            model_rows.append(
                {
                    "target": target,
                    "partition_seed": model.partition_seed,
                    "model_variant": model.rung_name,
                    "shape_parameter_count": len(model.shape),
                    "benefit_rate_min": float(np.exp(log_rates).min()),
                    "benefit_rate_median": float(np.median(np.exp(log_rates))),
                    "benefit_rate_max": float(np.exp(log_rates).max()),
                    "repetition_threshold_min": float(thresholds.min()),
                    "repetition_threshold_median": float(np.median(thresholds)),
                    "repetition_threshold_max": float(thresholds.max()),
                    "log_benefit_rates_json": json.dumps(log_rates.tolist()),
                    "repetition_thresholds_json": json.dumps(thresholds.tolist()),
                    "intercept": model.intercept,
                    "benefit_amplitude_sum": float(model.coefficients[:n_buckets].sum()),
                    "damage_amplitude_sum": float(model.coefficients[n_buckets:].sum()),
                }
            )
    return pd.DataFrame(model_rows), scores


def short_bucket_name(bucket: str) -> str:
    value = bucket.replace("dolma3_cc/", "CC ").replace("dolma3_", "").replace("dolmino_", "")
    return value.replace("_high", " H").replace("_low", " L").replace("_", " ")


def base_layout(figure: go.Figure, *, height: int) -> None:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 72, "r": 36, "t": 72, "b": 72},
        paper_bgcolor="#fbf7ef",
        plot_bgcolor="#fbf7ef",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#17324a", "size": 15},
        hoverlabel={"font": {"family": "Avenir Next, Avenir, sans-serif"}},
    )


def build_figures(
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    buckets: tuple[str, ...],
    caps: tuple[int, ...],
) -> list[go.Figure]:
    objective = make_subplots(rows=1, cols=2, subplot_titles=[TARGET_LABELS[target] for target in TARGETS])
    for column, target in enumerate(TARGETS, start=1):
        frame = summary[summary.target == target]
        objective.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.runtime_predicted_bpb,
                mode="lines+markers",
                marker={"size": 9, "color": "#d9542d"},
                line={"width": 3, "color": "#d9542d"},
                error_y={"type": "data", "array": frame.partition_prediction_sd, "visible": True},
                customdata=np.column_stack([frame.support_buckets, frame.cap_active_buckets, frame.nearest_panel_tv]),
                hovertemplate=(
                    "Cap %{x}<br>Predicted BPB %{y:.6f}<br>Support %{customdata[0]:.0f} buckets"
                    "<br>At cap %{customdata[1]:.0f}<br>Nearest observed TV %{customdata[2]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        objective.update_xaxes(
            title_text="Whole-run epoch cap", tickmode="array", tickvals=list(caps), row=1, col=column
        )
        objective.update_yaxes(title_text="DSP-predicted BPB (lower is better)", row=1, col=column)
    objective.update_layout(title="Predicted constrained optimum along the cap path")
    base_layout(objective, height=520)

    labels = [short_bucket_name(bucket) for bucket in buckets]
    cap_fraction = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[f"{TARGET_LABELS[target]}: fraction of allowed epochs used" for target in TARGETS],
    )
    weight_ratio = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[f"{TARGET_LABELS[target]}: weight relative to proportional" for target in TARGETS],
    )
    for row, target in enumerate(TARGETS, start=1):
        frame = weights[weights.target == target]
        epoch_matrix = frame.pivot(index="epoch_cap", columns="domain", values="cap_fraction").reindex(
            index=caps, columns=buckets
        )
        ratio_matrix = frame.pivot(index="epoch_cap", columns="domain", values="weight_ratio_to_proportional").reindex(
            index=caps, columns=buckets
        )
        cap_fraction.add_trace(
            go.Heatmap(
                z=epoch_matrix.to_numpy(),
                x=labels,
                y=list(caps),
                zmin=0.0,
                zmax=1.0,
                colorscale="RdYlGn_r",
                colorbar={"title": "cap used"} if row == 1 else None,
                showscale=row == 1,
                customdata=frame.pivot(index="epoch_cap", columns="domain", values="materialized_epochs")
                .reindex(index=caps, columns=buckets)
                .to_numpy(),
                hovertemplate="%{x}<br>Cap %{y}<br>%{customdata:.3f} epochs<br>%{z:.1%} of cap<extra></extra>",
            ),
            row=row,
            col=1,
        )
        log_ratio = np.log2(np.clip(ratio_matrix.to_numpy(), 1 / 32, 32))
        weight_ratio.add_trace(
            go.Heatmap(
                z=log_ratio,
                x=labels,
                y=list(caps),
                zmin=-5,
                zmax=5,
                zmid=0,
                colorscale="RdYlGn_r",
                colorbar={"title": "log2 ratio"} if row == 1 else None,
                showscale=row == 1,
                customdata=ratio_matrix.to_numpy(),
                hovertemplate="%{x}<br>Cap %{y}<br>%{customdata:.2f}x proportional<extra></extra>",
            ),
            row=row,
            col=1,
        )
    cap_fraction.update_yaxes(title_text="Epoch cap", tickmode="array", tickvals=list(caps))
    cap_fraction.update_xaxes(tickangle=-55, row=2, col=1)
    cap_fraction.update_layout(title="Which bucket constraints bind?")
    base_layout(cap_fraction, height=900)
    weight_ratio.update_yaxes(title_text="Epoch cap", tickmode="array", tickvals=list(caps))
    weight_ratio.update_xaxes(tickangle=-55, row=2, col=1)
    weight_ratio.update_layout(title="How the learned mixture changes as the cap relaxes")
    base_layout(weight_ratio, height=900)

    geometry = make_subplots(
        rows=1, cols=2, subplot_titles=["Support and active constraints", "Distance from proportional"]
    )
    colors = {TARGETS[0]: "#147d6f", TARGETS[1]: "#d9542d"}
    for target in TARGETS:
        frame = summary[summary.target == target]
        label = TARGET_LABELS[target]
        geometry.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.support_buckets,
                mode="lines+markers",
                name=f"{label} support",
                line={"color": colors[target], "width": 3},
            ),
            row=1,
            col=1,
        )
        geometry.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.cap_active_buckets,
                mode="lines+markers",
                name=f"{label} at cap",
                line={"color": colors[target], "width": 2, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
        geometry.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.tv_to_proportional,
                mode="lines+markers",
                name=label,
                line={"color": colors[target], "width": 3},
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    geometry.update_xaxes(title_text="Whole-run epoch cap", tickmode="array", tickvals=list(caps))
    geometry.update_yaxes(title_text="Number of buckets", row=1, col=1)
    geometry.update_yaxes(title_text="Total variation", row=1, col=2)
    geometry.update_layout(title="Candidate geometry")
    base_layout(geometry, height=520)
    return [objective, cap_fraction, weight_ratio, geometry]


def audit_findings(summary: pd.DataFrame, weights: pd.DataFrame) -> list[str]:
    findings = []
    unique = summary.groupby("target").candidate_id.count().sum()
    weight_hashes = weights.groupby("candidate_id").apply(
        lambda frame: hashlib.sha256(frame.runtime_count.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
        include_groups=False,
    )
    duplicate_count = int(len(weight_hashes) - weight_hashes.nunique())
    findings.append(f"All {unique} requested target-cap candidates are feasible on the exact runtime grid.")
    findings.append(
        f"The path contains {duplicate_count} exact duplicate mixture(s) across target-cap cells; "
        "duplicates should be reused rather than retrained."
    )
    findings.append(
        f"Every optimum is extrapolative: nearest observed-policy TV ranges from "
        f"{summary.nearest_panel_tv.min():.3f} to {summary.nearest_panel_tv.max():.3f}."
    )
    for target in TARGETS:
        frame = summary[summary.target == target].sort_values("epoch_cap")
        first, last = frame.iloc[0], frame.iloc[-1]
        findings.append(
            f"{TARGET_LABELS[target]} moves from {int(first.support_buckets)} supported buckets at cap 2 "
            f"to {int(last.support_buckets)} at cap {int(last.epoch_cap)}; TV from proportional changes "
            f"{first.tv_to_proportional:.3f} to {last.tv_to_proportional:.3f}."
        )
        findings.append(
            f"At the largest effective cap, {TARGET_LABELS[target]} assigns {last.largest_weight:.1%} to "
            f"{last.largest_bucket}; the predicted value is "
            f"{abs(last.predicted_minus_fit_panel_best_bpb):.3f} BPB below the best observed fit-panel endpoint."
        )
        if np.any(np.diff(frame.runtime_predicted_bpb) > MONOTONIC_TOLERANCE):
            raise ValueError(f"Predicted objective is not monotone for {target}")
    return findings


def render_report(
    output_path: Path,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    scores: dict[str, dict[str, float]],
    panel_path: Path,
    panel_hash: str,
    findings: list[str],
    figures: list[go.Figure],
    *,
    caps: tuple[int, ...],
    model_variant: str,
) -> None:
    fragments = []
    for index, figure in enumerate(figures):
        fragments.append(
            pio.to_html(
                figure,
                include_plotlyjs=index == 0,
                full_html=False,
                config=PLOT_CONFIG,
                div_id=f"figure-{index}",
            )
        )
    score_cards = "".join(
        f"""
        <article class="metric">
          <span>{html.escape(TARGET_LABELS[target])}</span>
          <strong>{score["spearman"]:.3f}</strong>
          <small>held-out Spearman</small>
          <p>Top-1 regret {score["regret@1"]:.5f} BPB; top-3 regret {score["regret@3"]:.5f}.</p>
        </article>
        """
        for target, score in scores.items()
    )
    finding_items = "".join(f"<li>{html.escape(item)}</li>" for item in findings)
    summary_columns = [
        "candidate_id",
        "runtime_predicted_bpb",
        "partition_prediction_sd",
        "max_materialized_epoch",
        "support_buckets",
        "cap_active_buckets",
        "tv_to_proportional",
        "nearest_panel_tv",
        "largest_bucket",
        "largest_weight",
    ]
    table = summary[summary_columns].copy()
    table.columns = [
        "candidate",
        "predicted BPB",
        "partition SD",
        "max epochs",
        "support",
        "at cap",
        "TV to prop.",
        "nearest observed TV",
        "largest bucket",
        "largest weight",
    ]
    table_html = table.to_html(index=False, classes="audit-table", float_format=lambda value: f"{value:.6f}")
    model_copy = {
        "shared_shape": (
            "Shared-shape canonical DSP",
            "one global benefit rate and one global repetition threshold",
        ),
        "canonical": (
            "Full per-bucket canonical DSP",
            "one benefit rate and one repetition threshold per bucket",
        ),
    }[model_variant]
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delphi 3e18 one-phase DSP epoch-cap sweep</title>
  <style>
    :root {{ --ink:#17324a; --muted:#5d7080; --paper:#fbf7ef; --card:#fffdf8; --line:#d8cdbd;
      --accent:#d9542d; --teal:#147d6f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:"Avenir Next",Avenir,sans-serif; }}
    main {{ max-width:1500px; margin:0 auto; padding:56px 34px 90px; }}
    h1,h2 {{ font-family:Georgia,"Times New Roman",serif; letter-spacing:-0.025em; }}
    h1 {{ max-width:1080px; font-size:clamp(42px,6vw,78px); line-height:.98; margin:0 0 24px; }}
    h2 {{ font-size:36px; margin:64px 0 14px; }}
    .dek {{ max-width:1000px; color:var(--muted); font-size:21px; line-height:1.55; }}
    .warning {{ margin:30px 0; padding:20px 24px; border-left:6px solid var(--accent); background:#fff4e9;
      font-size:18px; line-height:1.5; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin:34px 0; }}
    .metric {{ background:var(--card); border:1px solid var(--line); padding:22px; }}
    .metric span,.metric small {{ display:block; color:var(--muted); }}
    .metric strong {{ display:block; font:52px Georgia,serif; color:var(--teal); margin:8px 0; }}
    .metric p {{ margin-bottom:0; line-height:1.45; }}
    .method {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
    .method > div {{ background:var(--card); border-top:4px solid var(--teal); padding:24px; }}
    code {{ background:#efe8dc; padding:2px 5px; }}
    .equation {{ overflow:auto; font:17px "SFMono-Regular",Consolas,monospace; line-height:1.55; }}
    ul {{ line-height:1.65; font-size:18px; }}
    .figure {{ margin:28px -12px 58px; }}
    .table-wrap {{ overflow:auto; background:var(--card); border:1px solid var(--line); padding:8px; }}
    .audit-table {{ border-collapse:collapse; width:100%; font-size:14px; }}
    .audit-table th,.audit-table td {{ padding:10px 12px; border-bottom:1px solid #e7ded1; text-align:right; }}
    .audit-table th:first-child,.audit-table td:first-child {{ text-align:left; position:sticky; left:0;
      background:var(--card); }}
    .provenance {{ color:var(--muted); font-size:14px; overflow-wrap:anywhere; }}
    @media (max-width:800px) {{ main {{ padding:34px 16px 64px; }} .method {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <p class="provenance">Pre-training materialization audit</p>
  <h1>One-phase DSP optima under whole-run epoch caps</h1>
  <p class="dek">{len(summary)} Delphi 3e18 candidate policies: caps {caps[0]} through {caps[-1]} epochs,
  fitted separately
  for Uncheatable and Table-9. Every policy is single-phase: the same 39-bucket mixture is used throughout
  training. The cap limits each bucket's total materialized exposure over the complete run, not only the
  first 80%.</p>
  <div class="warning"><strong>What this is not:</strong> predicted BPB is not a frontier claim. The model
  ranks held-out one-phase policies well, but its point optimum is imperfect, especially for Table-9.
  This artifact audits whether the proposed training sweep is coherent before spending accelerator time.</div>
  <section class="metrics">{score_cards}</section>
  <section>
    <h2>Method</h2>
    <div class="method">
      <div><h3>Response model</h3><p>{model_copy[0]}, fitted on all 280 one-phase endpoints. It
      uses {model_copy[1]}, with nonnegative benefit and
      damage amplitudes for each bucket. No semantic grouping and no KL penalty are used.</p>
      <p class="equation">y_hat(w) = b - sum_i a_i [1-exp(-rho E_i)]<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ sum_i p_i softplus(log(1+E_i)-tau)^2</p></div>
      <div><h3>Feasible policy</h3><p><code>E_i = s_i w_i</code> is total simulated epochs in bucket i.
      For cap C, the solve enforces <code>w_i >= 0</code>, <code>sum_i w_i = 1</code>, and
      <code>E_i <= C</code> for all buckets. The result is quantized to integer counts summing to 2048,
      then improved by cap-safe one-count exchanges.</p></div>
    </div>
  </section>
  <section><h2>Mechanical audit</h2><ul>{finding_items}</ul></section>
  <section class="figure">{fragments[0]}</section>
  <section class="figure">{fragments[1]}</section>
  <section class="figure">{fragments[2]}</section>
  <section class="figure">{fragments[3]}</section>
  <section><h2>Exact candidate audit</h2><div class="table-wrap">{table_html}</div></section>
  <section><h2>Provenance</h2>
    <p class="provenance">Panel: {html.escape(str(panel_path.relative_to(REPO_ROOT)))}<br>
    Panel SHA-256: {panel_hash}<br>Runtime mixture block: {MIXTURE_BLOCK_SIZE}<br>
    Model partition seeds: {", ".join(map(str, MODEL_PARTITION_SEEDS))}</p>
  </section>
</main></body></html>"""
    output_path.write_text(document)


def main() -> None:
    args = parse_args()
    caps = tuple(range(2, args.max_cap + 1, 2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = swarm39.CANONICAL / f"{base.ONE_PHASE_DATASET['delphi_3e18']}.csv"
    panel = base.one_phase_panel("delphi_3e18")
    if len(panel) != 280 or any(np.isfinite(panel.targets[target]).sum() != 280 for target in TARGETS):
        raise ValueError("The complete 280-row one-phase panel is not available for both targets")
    scales = panel.c0 + panel.c1

    def fit_target(target: str, restart_executor: Executor | None) -> tuple[str, tuple[DspFit, ...]]:
        return target, fit_models(
            panel,
            target,
            rung_name=args.model_variant,
            restarts=args.model_restarts,
            maxiter=args.model_maxiter,
            workers=args.model_workers,
            executor=restart_executor,
        )

    if args.model_workers == 1:
        models_by_target = dict(fit_target(target, None) for target in TARGETS)
        candidates = [
            candidate
            for target in TARGETS
            for candidate in materialize_target(panel, target, models_by_target[target], scales, caps)
        ]
    else:
        target_workers = min(len(TARGETS), args.model_workers)
        with ProcessPoolExecutor(max_workers=args.model_workers) as restart_executor:

            def fit_target_parallel(target: str) -> tuple[str, tuple[DspFit, ...]]:
                return fit_target(target, restart_executor)

            with ThreadPoolExecutor(max_workers=target_workers) as target_executor:
                models_by_target = dict(target_executor.map(fit_target_parallel, TARGETS))

            def materialize_target_parallel(target: str) -> list[MaterializedCandidate]:
                return materialize_target(
                    panel,
                    target,
                    models_by_target[target],
                    scales,
                    caps,
                    restart_executor,
                )

            with ThreadPoolExecutor(max_workers=target_workers) as target_executor:
                target_candidates = list(target_executor.map(materialize_target_parallel, TARGETS))
            candidates = [candidate for group in target_candidates for candidate in group]
    summary, weights = candidate_tables(panel, candidates, models_by_target, scales)
    model_frame, scores = model_audit(panel, models_by_target)
    findings = audit_findings(summary, weights)

    summary_path = args.output_dir / "candidate_summary.csv"
    weights_path = args.output_dir / "candidate_weights.csv"
    models_path = args.output_dir / "model_fits.csv"
    report_path = args.output_dir / "index.html"
    summary.to_csv(summary_path, index=False)
    weights.to_csv(weights_path, index=False)
    model_frame.to_csv(models_path, index=False)
    figures = build_figures(summary, weights, panel.buckets, caps)
    render_report(
        report_path,
        summary,
        weights,
        scores,
        panel_path,
        file_sha256(panel_path),
        findings,
        figures,
        caps=caps,
        model_variant=args.model_variant,
    )

    model_name = {
        "shared_shape": "shared-shape canonical DSP partition ensemble",
        "canonical": "full per-bucket canonical DSP partition ensemble",
    }[args.model_variant]
    manifest = {
        "experiment": args.experiment_name,
        "training_status": "not_submitted",
        "fit_panel": str(panel_path.relative_to(REPO_ROOT)),
        "fit_panel_sha256": file_sha256(panel_path),
        "fit_rows": len(panel),
        "targets": list(TARGETS),
        "caps": list(caps),
        "model": model_name,
        "model_partition_seeds": list(MODEL_PARTITION_SEEDS),
        "model_restarts": args.model_restarts,
        "model_maxiter": args.model_maxiter,
        "model_workers": args.model_workers,
        "semantic_families": False,
        "kl_penalty": 0.0,
        "cap_scope": "whole training run",
        "mixture_block_size": MIXTURE_BLOCK_SIZE,
        "candidate_count": len(summary),
        "unique_runtime_mixtures": int(
            weights.groupby("candidate_id")
            .apply(
                lambda frame: hashlib.sha256(frame.runtime_count.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
                include_groups=False,
            )
            .nunique()
        ),
        "heldout_scores": scores,
        "output_sha256": {
            summary_path.name: file_sha256(summary_path),
            weights_path.name: file_sha256(weights_path),
            models_path.name: file_sha256(models_path),
            report_path.name: file_sha256(report_path),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
