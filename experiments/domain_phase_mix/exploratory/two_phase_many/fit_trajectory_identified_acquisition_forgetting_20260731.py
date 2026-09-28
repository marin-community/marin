# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
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
"""Identify a bounded acquisition-forgetting state from intermediate trajectories.

For bucket ``i`` and normalized training progress ``u``, the latent mastery
state follows

    dz_i / du = alpha * q_i(u) * (1 - z_i) - lambda * z_i,

where ``q_i`` is the realized materialized-epoch rate. The state is bounded in
``[0, 1]`` and has an exact transition under each constant-mixture phase.

Only paired 300M Uncheatable increments ending by step 21,000 select ``alpha``,
``lambda``, and the response ridge. The 21,000-to-22,000 increment and final
step-22,887 pair delta are held out. Table-9 may fit only three predeclared
family response amplitudes with the transition and ridge frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_physical_hpr_tied_spine_20260731 as physical_spine,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "trajectory_identified_acquisition_forgetting_20260731"
HISTORY_PATH = SCRIPT_DIR / "reference_outputs" / "tied_two_phase_trajectory_audit_20260726" / "wandb_histories.csv"
TARGET_COLUMN = "eval/uncheatable_eval/bpb"
TRAINING_SCALE = "300m"
PHASE_BOUNDARY_STEP = 18_310
TRANSITION_TRAIN_END_STEP = 21_000
TRANSITION_HOLDOUT_END_STEP = 22_000
FINAL_STEP = 22_887
N_SPLITS = 5
SPLIT_SEED = 20260731

ACQUISITION_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FORGETTING_GRID = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
RIDGE_GRID = (0.0, 0.1, 1.0, 10.0, 100.0)

HPR_PAIR_RMSE = {
    "uncheatable": 0.007850,
    "table9": 0.016902,
}


@dataclass(frozen=True)
class Candidate:
    """Nonlinear state parameters and response regularization."""

    acquisition_rate: float
    forgetting_rate: float
    ridge: float


@dataclass(frozen=True)
class PairData:
    """Physical policies and observed trajectories for exact matched pairs."""

    keys: tuple[str, ...]
    asymmetric_weights: np.ndarray
    tied_weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    steps: np.ndarray
    progress: np.ndarray
    observed_delta: np.ndarray
    endpoint_delta: np.ndarray
    family_names: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    table9_delta: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def paired_indices(dataset: benchmark.Dataset) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return physically asymmetric rows and their tied counterparts."""
    frame = dataset.frame.reset_index()
    indexed = frame.set_index(["phase_correspondence_key", "policy_family"])["index"]
    keys = sorted(
        set(frame.loc[frame["policy_family"].eq("single_phase"), "phase_correspondence_key"].astype(str))
        & set(frame.loc[frame["policy_family"].eq("two_phase"), "phase_correspondence_key"].astype(str))
    )
    tied = np.asarray([indexed.loc[(key, "single_phase")] for key in keys], dtype=int)
    asymmetric = np.asarray([indexed.loc[(key, "two_phase")] for key in keys], dtype=int)
    genuinely_asymmetric = ~benchmark.replay_control.tied_rows(dataset.weights[asymmetric])
    return (
        tied[genuinely_asymmetric],
        asymmetric[genuinely_asymmetric],
        tuple(key for key, keep in zip(keys, genuinely_asymmetric, strict=True) if keep),
    )


def trajectory_matrix(keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical steps, progress, and possibly sparse paired BPB."""
    history = pd.read_csv(HISTORY_PATH)
    history = history.loc[history["scale_key"].eq(TRAINING_SCALE) & history["pair_id"].astype(str).isin(keys)].copy()
    values = history.pivot_table(
        index=["pair_id", "global_step"],
        columns="policy_class",
        values=TARGET_COLUMN,
        aggfunc="last",
    ).dropna(subset=["one_phase", "two_phase"])
    steps = np.asarray([*range(1_000, TRANSITION_HOLDOUT_END_STEP + 1, 1_000), FINAL_STEP], dtype=int)
    observed = np.vstack(
        [
            (values.loc[key].reindex(steps)["two_phase"] - values.loc[key].reindex(steps)["one_phase"]).to_numpy(float)
            for key in keys
        ]
    )
    progress_by_step = (
        history.loc[history["global_step"].isin(steps)]
        .groupby("global_step")["run_progress"]
        .median()
        .reindex(steps)
        .to_numpy(float)
    )
    if not np.all(np.diff(progress_by_step) > 0.0):
        raise ValueError("Training progress is not strictly increasing")
    if int(np.isfinite(observed[:, steps <= TRANSITION_TRAIN_END_STEP]).sum()) < 3_000:
        raise ValueError("Too few paired pre-final trajectory observations")
    return steps, progress_by_step, observed


def load_pair_data() -> PairData:
    uncheatable = benchmark.load_300m("uncheatable")
    tied, asymmetric, keys = paired_indices(uncheatable)
    if len(keys) != 238:
        raise ValueError(f"Expected 238 exact asymmetric pairs, found {len(keys)}")
    steps, progress, observed_delta = trajectory_matrix(keys)

    table9 = benchmark.load_300m("table9")
    table9_tied, table9_asymmetric, table9_keys = paired_indices(table9)
    if table9_keys != keys:
        raise ValueError("Uncheatable and Table-9 pair ordering differs")
    table9_delta = table9.y[table9_asymmetric] - table9.y[table9_tied]

    tied_dataset = physical_spine.tied_dataset("uncheatable")
    family_names, family_members, _quality = observatory.family_partition(tied_dataset)
    if tuple(tied_dataset.domain_names) != tuple(uncheatable.domain_names):
        raise ValueError("Family partition domain order differs from trajectory policies")

    return PairData(
        keys=keys,
        asymmetric_weights=uncheatable.weights[asymmetric],
        tied_weights=uncheatable.weights[tied],
        c0=np.asarray(uncheatable.c0, dtype=float),
        c1=np.asarray(uncheatable.c1, dtype=float),
        steps=steps,
        progress=progress,
        observed_delta=observed_delta,
        endpoint_delta=np.asarray(uncheatable.y[asymmetric] - uncheatable.y[tied], dtype=float),
        family_names=family_names,
        family_members=family_members,
        table9_delta=np.asarray(table9_delta, dtype=float),
    )


def advance_state(
    state: np.ndarray,
    exposure_rate: np.ndarray,
    duration: float,
    acquisition_rate: float,
    forgetting_rate: float,
) -> np.ndarray:
    """Apply the exact constant-rate acquisition-forgetting transition."""
    total_rate = acquisition_rate * exposure_rate + forgetting_rate
    equilibrium = np.divide(
        acquisition_rate * exposure_rate,
        total_rate,
        out=np.zeros_like(total_rate),
        where=total_rate > 0.0,
    )
    return equilibrium + (state - equilibrium) * np.exp(-total_rate * duration)


def states_at_progress(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    progress: np.ndarray,
    acquisition_rate: float,
    forgetting_rate: float,
) -> np.ndarray:
    """Evaluate the bounded state under the exact two-phase schedule."""
    phase_fraction = float(np.median(c0 / np.maximum(c0 + c1, 1e-12)))
    phase0_rate = c0[None, :] * weights[:, 0, :] / phase_fraction
    phase1_rate = c1[None, :] * weights[:, 1, :] / (1.0 - phase_fraction)
    initial = np.zeros_like(phase0_rate)
    boundary = advance_state(
        initial,
        phase0_rate,
        phase_fraction,
        acquisition_rate,
        forgetting_rate,
    )
    states = []
    for time in progress:
        if time <= phase_fraction:
            state = advance_state(
                initial,
                phase0_rate,
                float(time),
                acquisition_rate,
                forgetting_rate,
            )
        else:
            state = advance_state(
                boundary,
                phase1_rate,
                float(time - phase_fraction),
                acquisition_rate,
                forgetting_rate,
            )
        states.append(state)
    return np.stack(states, axis=1)


def state_features(
    data: PairData,
    acquisition_rate: float,
    forgetting_rate: float,
) -> np.ndarray:
    """Mean asymmetric-minus-tied mastery in each predeclared family."""
    asymmetric = states_at_progress(
        data.asymmetric_weights,
        data.c0,
        data.c1,
        data.progress,
        acquisition_rate,
        forgetting_rate,
    )
    tied = states_at_progress(
        data.tied_weights,
        data.c0,
        data.c1,
        data.progress,
        acquisition_rate,
        forgetting_rate,
    )
    delta = asymmetric - tied
    return np.stack(
        [delta[:, :, members].mean(axis=2) for members in data.family_members],
        axis=2,
    )


def pair_splits(pair_count: int, seed: int, n_splits: int = N_SPLITS) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(pair_count)
    return tuple((train, test) for train, test in splitter.split(indices))


def fit_nonnegative_response(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """Fit nonnegative family values with an RMS-scaled ridge."""
    scale = np.sqrt(np.mean(design**2, axis=0))
    scale = np.maximum(scale, 1e-10)
    normalized = design / scale[None, :]
    response = np.asarray(target, dtype=float)
    if ridge > 0.0:
        normalized = np.vstack([normalized, np.sqrt(ridge) * np.eye(normalized.shape[1])])
        response = np.concatenate([response, np.zeros(normalized.shape[1], dtype=float)])
    coefficients, _residual = nnls(normalized, response)
    return coefficients / scale


def interval_arrays(
    data: PairData,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pre-final increment designs and phase-1 indicator."""
    training_steps = data.steps[data.steps <= TRANSITION_TRAIN_END_STEP]
    positions = np.asarray([int(np.flatnonzero(data.steps == step)[0]) for step in training_steps])
    starts = positions[:-1]
    ends = positions[1:]
    # More mastery lowers BPB, hence the negative state difference.
    design = -(features[:, ends, :] - features[:, starts, :])
    target = data.observed_delta[:, ends] - data.observed_delta[:, starts]
    phase1 = data.steps[ends] > PHASE_BOUNDARY_STEP
    return design, target, phase1, data.steps[ends]


def candidate_oof(
    data: PairData,
    features: np.ndarray,
    ridge: float,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    design, target, phase1, _end_steps = interval_arrays(data, features)
    prediction = np.full_like(target, np.nan)
    coefficients = []
    for train, test in splits:
        train_design = design[train].reshape(-1, design.shape[2])
        train_target = target[train].reshape(-1)
        finite_train = np.isfinite(train_target)
        fitted = fit_nonnegative_response(
            train_design[finite_train],
            train_target[finite_train],
            ridge,
        )
        prediction[test] = design[test] @ fitted
        coefficients.append(fitted)
    covered = np.unique(np.concatenate([test for _train, test in splits]))
    if not np.isfinite(prediction[covered][np.isfinite(target[covered])]).all():
        raise ValueError("Incomplete interval OOF prediction")
    return prediction, target, phase1, np.vstack(coefficients)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((predicted[finite] - observed[finite]) ** 2)))


def select_candidate(
    data: PairData,
    pair_indices: np.ndarray | None = None,
    seed: int = SPLIT_SEED,
) -> tuple[Candidate, pd.DataFrame]:
    local = np.arange(len(data.keys)) if pair_indices is None else np.asarray(pair_indices, dtype=int)
    splits_local = pair_splits(len(local), seed)
    splits = tuple((local[train], local[test]) for train, test in splits_local)
    rows = []
    best: tuple[float, float, Candidate] | None = None
    for acquisition_rate in ACQUISITION_GRID:
        for forgetting_rate in FORGETTING_GRID:
            features = state_features(data, acquisition_rate, forgetting_rate)
            for ridge in RIDGE_GRID:
                prediction, target, phase1, _coefficients = candidate_oof(data, features, ridge, splits)
                mask = np.zeros(len(data.keys), dtype=bool)
                mask[local] = True
                phase1_rmse = rmse(target[mask][:, phase1], prediction[mask][:, phase1])
                all_rmse = rmse(target[mask], prediction[mask])
                candidate = Candidate(acquisition_rate, forgetting_rate, ridge)
                rows.append(
                    {
                        **asdict(candidate),
                        "phase1_interval_oof_rmse": phase1_rmse,
                        "all_interval_oof_rmse": all_rmse,
                    }
                )
                key = (phase1_rmse, all_rmse, candidate)
                if best is None or key[:2] < best[:2]:
                    best = key
    if best is None:
        raise RuntimeError("No acquisition-forgetting candidate was scored")
    return best[2], pd.DataFrame(rows)


def safe_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    if float(np.std(predicted)) <= 1e-12:
        return float("nan")
    return float(np.polyfit(predicted, observed, deg=1)[0])


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = np.asarray(observed, dtype=float)[finite]
    predicted = np.asarray(predicted, dtype=float)[finite]
    if len(observed) == 0:
        raise ValueError("No finite observations for metrics")
    observed_scale = float(np.std(observed))
    return {
        "rmse": rmse(observed, predicted),
        "bias": float(np.mean(predicted - observed)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "observed_on_predicted_slope": safe_slope(observed, predicted),
        "amplitude_ratio": float(np.std(predicted) / observed_scale) if observed_scale > 0.0 else float("nan"),
        "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
        "zero_delta_null_rmse": float(np.sqrt(np.mean(observed**2))),
    }


def endpoint_oof(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full(len(target), np.nan)
    coefficients = []
    for train, test in splits:
        fitted = fit_nonnegative_response(design[train], target[train], ridge)
        prediction[test] = design[test] @ fitted
        coefficients.append(fitted)
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete endpoint OOF prediction")
    return prediction, np.vstack(coefficients)


def write_plots(
    data: PairData,
    predicted_trajectory: np.ndarray,
    endpoint_prediction: np.ndarray,
    output_dir: Path,
) -> None:
    means_observed = np.nanmean(data.observed_delta, axis=0)
    means_predicted = predicted_trajectory.mean(axis=0)
    q10_observed, q90_observed = np.nanquantile(data.observed_delta, [0.1, 0.9], axis=0)
    q10_predicted, q90_predicted = np.quantile(predicted_trajectory, [0.1, 0.9], axis=0)

    trajectory = go.Figure()
    trajectory.add_trace(
        go.Scatter(
            x=np.concatenate([data.steps, data.steps[::-1]]),
            y=np.concatenate([q10_observed, q90_observed[::-1]]),
            fill="toself",
            fillcolor="rgba(215,48,39,0.12)",
            line={"color": "rgba(0,0,0,0)"},
            name="Observed 10-90%",
        )
    )
    trajectory.add_trace(
        go.Scatter(
            x=data.steps,
            y=means_observed,
            mode="lines+markers",
            line={"color": "#d73027", "width": 3},
            name="Observed mean",
        )
    )
    trajectory.add_trace(
        go.Scatter(
            x=np.concatenate([data.steps, data.steps[::-1]]),
            y=np.concatenate([q10_predicted, q90_predicted[::-1]]),
            fill="toself",
            fillcolor="rgba(26,152,80,0.12)",
            line={"color": "rgba(0,0,0,0)"},
            name="Predicted 10-90%",
        )
    )
    trajectory.add_trace(
        go.Scatter(
            x=data.steps,
            y=means_predicted,
            mode="lines+markers",
            line={"color": "#1a9850", "width": 3},
            name="Predicted mean",
        )
    )
    trajectory.add_vline(x=PHASE_BOUNDARY_STEP, line_dash="dash", annotation_text="phase boundary")
    trajectory.add_vrect(
        x0=TRANSITION_TRAIN_END_STEP,
        x1=FINAL_STEP,
        fillcolor="rgba(69,117,180,0.08)",
        line_width=0,
        annotation_text="held out",
        annotation_position="top left",
    )
    trajectory.update_layout(
        title="Trajectory-identified state: paired Uncheatable BPB",
        xaxis_title="Training step",
        yaxis_title="Two-phase minus tied BPB",
        template="plotly_white",
        height=700,
        width=1100,
    )
    trajectory.write_html(output_dir / "trajectory_fit.html", include_plotlyjs="cdn")

    endpoint = make_subplots(rows=1, cols=2, subplot_titles=("Final endpoint", "Endpoint residual"))
    endpoint.add_trace(
        go.Scatter(
            x=data.endpoint_delta,
            y=endpoint_prediction,
            mode="markers",
            marker={
                "color": data.endpoint_delta - endpoint_prediction,
                "colorscale": "RdYlGn_r",
                "showscale": True,
                "colorbar": {"title": "optimism"},
            },
            text=data.keys,
            name="pairs",
        ),
        row=1,
        col=1,
    )
    endpoint.add_trace(
        go.Scatter(
            x=data.endpoint_delta,
            y=endpoint_prediction - data.endpoint_delta,
            mode="markers",
            marker={"color": "#4575b4"},
            text=data.keys,
            name="residual",
        ),
        row=1,
        col=2,
    )
    low = float(min(data.endpoint_delta.min(), endpoint_prediction.min()))
    high = float(max(data.endpoint_delta.max(), endpoint_prediction.max()))
    endpoint.add_trace(
        go.Scatter(x=[low, high], y=[low, high], mode="lines", line={"dash": "dash", "color": "#555"}),
        row=1,
        col=1,
    )
    endpoint.add_hline(y=0.0, line_dash="dash", line_color="#555", row=1, col=2)
    endpoint.update_xaxes(title_text="Observed two-phase minus tied BPB")
    endpoint.update_yaxes(title_text="Predicted two-phase minus tied BPB", row=1, col=1)
    endpoint.update_yaxes(title_text="Predicted minus observed BPB", row=1, col=2)
    endpoint.update_layout(
        title="Strict final-endpoint falsification",
        template="plotly_white",
        height=650,
        width=1200,
        showlegend=False,
    )
    endpoint.write_html(output_dir / "endpoint_falsification.html", include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_pair_data()
    selected, sweep = select_candidate(data)
    sweep.to_csv(args.output_dir / "candidate_sweep.csv", index=False)

    features = state_features(data, selected.acquisition_rate, selected.forgetting_rate)
    design, interval_target, phase1, interval_end_steps = interval_arrays(data, features)
    splits = pair_splits(len(data.keys), SPLIT_SEED)
    interval_prediction, _target, _phase1, fold_coefficients = candidate_oof(
        data,
        features,
        selected.ridge,
        splits,
    )
    full_design = design.reshape(-1, design.shape[2])
    full_target = interval_target.reshape(-1)
    finite_training = np.isfinite(full_target)
    full_coefficients = fit_nonnegative_response(
        full_design[finite_training],
        full_target[finite_training],
        selected.ridge,
    )
    predicted_trajectory = -features @ full_coefficients

    heldout_start = int(np.flatnonzero(data.steps == TRANSITION_TRAIN_END_STEP)[0])
    heldout_end = int(np.flatnonzero(data.steps == TRANSITION_HOLDOUT_END_STEP)[0])
    heldout_design = -(features[:, heldout_end, :] - features[:, heldout_start, :])
    heldout_observed = data.observed_delta[:, heldout_end] - data.observed_delta[:, heldout_start]
    heldout_prediction = heldout_design @ full_coefficients

    endpoint_index = int(np.flatnonzero(data.steps == FINAL_STEP)[0])
    endpoint_observed = data.endpoint_delta
    endpoint_prediction = predicted_trajectory[:, endpoint_index]

    endpoint_design = -features[:, endpoint_index, :]
    table9_prediction, table9_coefficients = endpoint_oof(
        endpoint_design,
        data.table9_delta,
        selected.ridge,
        splits,
    )

    metrics_rows = [
        {
            "evaluation": "uncheatable_pre_final_interval_oof_all",
            **prediction_metrics(interval_target.ravel(), interval_prediction.ravel()),
            "hpr_reference_rmse": float("nan"),
        },
        {
            "evaluation": "uncheatable_pre_final_interval_oof_phase1",
            **prediction_metrics(interval_target[:, phase1].ravel(), interval_prediction[:, phase1].ravel()),
            "hpr_reference_rmse": float("nan"),
        },
        {
            "evaluation": "uncheatable_21000_to_22000_holdout",
            **prediction_metrics(heldout_observed, heldout_prediction),
            "hpr_reference_rmse": float("nan"),
        },
        {
            "evaluation": "uncheatable_final_endpoint_strict_holdout",
            **prediction_metrics(endpoint_observed, endpoint_prediction),
            "hpr_reference_rmse": HPR_PAIR_RMSE["uncheatable"],
        },
        {
            "evaluation": "table9_final_endpoint_frozen_state_oof",
            **prediction_metrics(data.table9_delta, table9_prediction),
            "hpr_reference_rmse": HPR_PAIR_RMSE["table9"],
        },
    ]
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)

    interval_rows = []
    for pair_index, key in enumerate(data.keys):
        for interval_index, end_step in enumerate(interval_end_steps):
            interval_rows.append(
                {
                    "phase_correspondence_key": key,
                    "end_step": int(end_step),
                    "phase1": bool(phase1[interval_index]),
                    "observed_increment": float(interval_target[pair_index, interval_index]),
                    "oof_predicted_increment": float(interval_prediction[pair_index, interval_index]),
                }
            )
    pd.DataFrame(interval_rows).to_csv(args.output_dir / "interval_oof_predictions.csv", index=False)

    endpoint_frame = pd.DataFrame(
        {
            "phase_correspondence_key": data.keys,
            "uncheatable_observed_delta": endpoint_observed,
            "uncheatable_predicted_delta_strict_holdout": endpoint_prediction,
            "table9_observed_delta": data.table9_delta,
            "table9_predicted_delta_frozen_state_oof": table9_prediction,
        }
    )
    endpoint_frame.to_csv(args.output_dir / "endpoint_predictions.csv", index=False)

    parameter_rows = []
    for family, value in zip(data.family_names, full_coefficients, strict=True):
        parameter_rows.append(
            {
                "fit": "uncheatable_pre_final_full",
                "fold": -1,
                "family": family,
                "response_bpb_per_mean_mastery": value,
            }
        )
    for fold, coefficients in enumerate(fold_coefficients):
        for family, value in zip(data.family_names, coefficients, strict=True):
            parameter_rows.append(
                {
                    "fit": "uncheatable_pre_final_oof",
                    "fold": fold,
                    "family": family,
                    "response_bpb_per_mean_mastery": value,
                }
            )
    for fold, coefficients in enumerate(table9_coefficients):
        for family, value in zip(data.family_names, coefficients, strict=True):
            parameter_rows.append(
                {
                    "fit": "table9_endpoint_oof",
                    "fold": fold,
                    "family": family,
                    "response_bpb_per_mean_mastery": value,
                }
            )
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / "response_parameters.csv", index=False)

    stability_rows = []
    all_pairs = np.arange(len(data.keys))
    for fold, (_train, excluded) in enumerate(splits):
        retained = np.setdiff1d(all_pairs, excluded)
        fold_selected, _fold_sweep = select_candidate(
            data,
            retained,
            seed=SPLIT_SEED + 100 * (fold + 1),
        )
        stability_rows.append({"excluded_fold": fold, **asdict(fold_selected)})
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(args.output_dir / "transition_stability.csv", index=False)

    write_plots(data, predicted_trajectory, endpoint_prediction, args.output_dir)

    selected_record = {
        **asdict(selected),
        "family_names": data.family_names,
        "phase_boundary_step": PHASE_BOUNDARY_STEP,
        "transition_training_end_step": TRANSITION_TRAIN_END_STEP,
        "transition_holdout_end_step": TRANSITION_HOLDOUT_END_STEP,
        "final_endpoint_step": FINAL_STEP,
    }
    (args.output_dir / "selected_model.json").write_text(
        json.dumps(selected_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = f"""# Trajectory-identified acquisition-forgetting state

## Frozen model

The bounded bucket state obeys `dz/du = alpha*q*(1-z) - lambda*z`, with physical
materialized-epoch rate `q`. Three nonnegative response amplitudes correspond to
the predeclared families `{", ".join(data.family_names)}`. The terminal phase
correction is the asymmetric state minus its tied counterfactual.

- Selected acquisition rate: `{selected.acquisition_rate}`
- Selected forgetting rate: `{selected.forgetting_rate}`
- Selected response ridge: `{selected.ridge}`
- Transition selection used only increments ending by step {TRANSITION_TRAIN_END_STEP}.
- The {TRANSITION_TRAIN_END_STEP}-to-{TRANSITION_HOLDOUT_END_STEP} increment and
  final step {FINAL_STEP} were not used for transition or response selection.
- Table-9 used the frozen transition and ridge; only three response amplitudes
  were refit inside correspondence-grouped folds.

## Falsification metrics

{metrics.to_markdown(index=False)}

## Transition stability

{stability.to_markdown(index=False)}

HPR's persisted exact-pair endpoint RMSE is {HPR_PAIR_RMSE["uncheatable"]:.6f}
on Uncheatable and {HPR_PAIR_RMSE["table9"]:.6f} on Table-9. The zero-delta
column is the no-phase-effect null on exactly the same pairs.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
