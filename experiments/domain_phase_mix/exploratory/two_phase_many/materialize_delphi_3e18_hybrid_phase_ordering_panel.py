# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Materialize 3e18-fit hybrid aggregate/phase-order validation paths.

Every surrogate sees the same 280-row Delphi 3e18 two-phase fit panel. A
CV-selected separate-heads model chooses a phase-tied aggregate. Four models
then choose only phase order while holding that aggregate exactly fixed:
effective-exposure DSP, Bucket-resolved family GRP, Compact retained state,
and Hierarchical phase replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_decoupled_phase_information_constraints_300m as phase_information,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_split_dsp_300m as phase_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_hierarchical_phase_replay_validation_panel_3e18 as hpr_panel,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_original_style_matched_sepheads_ablation_300m as separate_heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import (  # noqa: E402
    dsp_exact,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / ("reference_outputs/delphi_3e18_hybrid_phase_ordering_panel_20260720")
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_hybrid_phase_ordering_panel_20260720"
)
TARGETS = ("uncheatable", "table9")
TARGET_TAGS = {"uncheatable": "unch", "table9": "t9"}
MODELS = (
    "effective_exposure",
    "bucket_family_grp",
    "compact_retained_state",
    "hierarchical_phase_replay",
)
MODEL_TAGS = {
    "effective_exposure": "eff",
    "bucket_family_grp": "bfgrp",
    "compact_retained_state": "compact",
    "hierarchical_phase_replay": "hpr",
}
AGGREGATE_KL_COEFFICIENTS = (0.025, 0.05, 0.075, 0.1)
PHASE_INFORMATION_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.025)
FIT_ROWS = 280
SEPARATE_L2_VALUES = separate_heads.DEFAULT_L2_VALUES
CV_SEEDS = separate_heads.CV_SEEDS
N_SPLITS = separate_heads.N_SPLITS
EFFECTIVE_EXPOSURE_LINEAR_REG = 0.01
EFFECTIVE_EXPOSURE_MAXITER = 40
EFFECTIVE_EXPOSURE_TOP_K = 3
OPTIMIZER_STARTS = 16
COORDINATE_DECIMALS = 12
EXACT_POLICY_TV = 1e-9
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Predictor:
    """A fitted phase-ordering surrogate with frozen selection metadata."""

    model_id: str
    predict: Callable[[np.ndarray], float]
    selection: dict[str, Any]


@dataclass(frozen=True)
class SolveResult:
    """One finite multistart policy solution."""

    weights: np.ndarray
    prediction: float
    regularized_objective: float
    successful_starts: int


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one numeric sweep value")
    return values


def float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def policy_hash(weights: np.ndarray) -> str:
    rounded = np.round(np.asarray(weights, dtype=np.float64), decimals=COORDINATE_DECIMALS)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def json_clean(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return json_clean(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_clean(value.to_dict())
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if hasattr(value, "value"):
        return json_clean(value.value)
    return value


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=float) - float(np.max(logits))
    values = np.exp(shifted)
    return values / values.sum()


def scalar_prediction(predict: Callable[[np.ndarray], float], weights: np.ndarray) -> float:
    value = float(predict(np.asarray(weights, dtype=float)))
    if not np.isfinite(value):
        raise ValueError("Surrogate returned a non-finite prediction")
    return value


def separate_cv_selection(dataset, target: str) -> tuple[float, pd.DataFrame]:
    """Select separate-heads ridge with the original three-seed protocol."""
    rows: list[dict[str, float | int | str]] = []
    for l2 in SEPARATE_L2_VALUES:
        for seed in CV_SEEDS:
            folds = component_dsp.panel_stratified_folds(dataset.frame, n_splits=N_SPLITS, seed=seed)
            prediction = np.full(dataset.n, np.nan, dtype=float)
            fold_regrets: list[float] = []
            for train, test in folds:
                model = separate_heads.fit_model(dataset, train, "2p", l2)
                prediction[test] = separate_heads.predict_model(model, dataset, dataset.weights[test])
                selected = int(test[int(np.argmin(prediction[test]))])
                fold_regrets.append(float(dataset.y[selected] - np.min(dataset.y[test])))
            residual = prediction - dataset.y
            rows.append(
                {
                    "target": target,
                    "l2": float(l2),
                    "seed": int(seed),
                    "oof_rmse": float(np.sqrt(np.mean(residual**2))),
                    "fold_mean_regret_at_1": float(np.mean(fold_regrets)),
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("l2", as_index=False)
        .agg(
            oof_rmse=("oof_rmse", "mean"),
            oof_rmse_sd=("oof_rmse", "std"),
            fold_mean_regret_at_1=("fold_mean_regret_at_1", "mean"),
        )
        .sort_values(["oof_rmse", "fold_mean_regret_at_1", "l2"])
        .reset_index(drop=True)
    )
    return float(summary.iloc[0]["l2"]), summary


def fit_separate_anchor(dataset, target: str) -> tuple[Predictor, pd.DataFrame]:
    l2, sweep = separate_cv_selection(dataset, target)
    model = separate_heads.fit_model(dataset, np.arange(dataset.n), "2p", l2)

    def predict(weights: np.ndarray) -> float:
        return float(separate_heads.predict_model(model, dataset, weights[None, :, :])[0])

    return Predictor("separate_heads", predict, {"l2": l2}), sweep


def fit_phase_predictors(dataset) -> dict[str, Predictor]:
    indices = np.arange(dataset.n)
    packet = coverage_dsp.packet(dataset, indices)
    effective_model, effective_tuning = phase_dsp.fit_variant_with_l2(
        packet,
        "effective_exposure",
        EFFECTIVE_EXPOSURE_LINEAR_REG,
        maxiter=EFFECTIVE_EXPOSURE_MAXITER,
        coarse_top_k=EFFECTIVE_EXPOSURE_TOP_K,
        basin_hopping_iters=0,
    )

    def effective_predict(weights: np.ndarray) -> float:
        return float(dsp_exact.predict(effective_model, weights[None, :, :])[0])

    bucket_shape, bucket_l2, bucket_sweep = observatory.select_bucket_hyperparameters(
        dataset,
        observatory.TWO_PHASE,
    )
    bucket_model = observatory.bucket_fit(dataset, indices, bucket_shape, bucket_l2)

    def bucket_predict(weights: np.ndarray) -> float:
        return float(bucket_model.predict(weights[None, :, :])[0])

    compact_l2, compact_sweep = observatory.select_compact_l2(dataset, observatory.TWO_PHASE)
    compact_model = observatory.compact_fit(dataset, indices, compact_l2, observatory.TWO_PHASE)

    def compact_predict(weights: np.ndarray) -> float:
        return float(compact_model.predict(weights[None, :, :])[0])

    hpr_config, hpr_sweep = observatory.select_hierarchical_phase_replay_config(
        dataset,
        observatory.TWO_PHASE,
    )
    hpr_model = observatory.hierarchical_phase_replay_fit(dataset, indices, hpr_config)

    def hpr_predict(weights: np.ndarray) -> float:
        return float(hpr_model.predict(weights[None, :, :])[0])

    return {
        "effective_exposure": Predictor(
            "effective_exposure",
            effective_predict,
            {
                "linear_reg": EFFECTIVE_EXPOSURE_LINEAR_REG,
                "maxiter": EFFECTIVE_EXPOSURE_MAXITER,
                "top_k": EFFECTIVE_EXPOSURE_TOP_K,
                "fit_tuning": effective_tuning,
            },
        ),
        "bucket_family_grp": Predictor(
            "bucket_family_grp",
            bucket_predict,
            {
                "shape": asdict(bucket_shape),
                "l2": bucket_l2,
                "cv_winner": min(bucket_sweep, key=lambda row: (row["oofRmse"], -row["oofSpearman"])),
            },
        ),
        "compact_retained_state": Predictor(
            "compact_retained_state",
            compact_predict,
            {
                "config": asdict(observatory.compact_config(observatory.TWO_PHASE)),
                "l2": compact_l2,
                "l2_sweep": compact_sweep,
                "shape": asdict(compact_model.shape),
            },
        ),
        "hierarchical_phase_replay": Predictor(
            "hierarchical_phase_replay",
            hpr_predict,
            {
                "config": {
                    **asdict(hpr_config),
                    "variant": hpr_config.variant.value,
                },
                "candidate_sweep": hpr_sweep["candidateSweep"],
            },
        ),
    }


def policy_starts(dataset, natural: np.ndarray, *, tied: bool, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)

    def logits(weights: np.ndarray) -> np.ndarray:
        if tied:
            alpha0, alpha1 = observatory.phase_fractions(dataset)
            aggregate = alpha0 * weights[0] + alpha1 * weights[1]
            values = aggregate
        else:
            values = weights
        logged = np.log(np.clip(values, 1e-12, 1.0))
        return logged.reshape(-1)

    starts = [logits(np.stack([natural, natural]))]
    starts.extend(logits(dataset.weights[index]) for index in np.argsort(dataset.y)[:8])
    while len(starts) < count:
        concentration = (0.25, 1.0, 4.0)[len(starts) % 3]
        if tied:
            sample = rng.dirichlet(np.full(dataset.m, concentration))
            weights = np.stack([sample, sample])
        else:
            weights = np.stack(
                [
                    rng.dirichlet(np.full(dataset.m, concentration)),
                    rng.dirichlet(np.full(dataset.m, concentration)),
                ]
            )
        starts.append(logits(weights))
    return starts[:count]


def optimize_policy(
    predictor: Predictor,
    dataset,
    natural: np.ndarray,
    *,
    tied: bool,
    aggregate_kl_coefficient: float,
    seed: int,
) -> SolveResult:
    def weights_from_logits(logits: np.ndarray) -> np.ndarray:
        if tied:
            values = softmax(logits)
            return np.stack([values, values])
        midpoint = dataset.m
        return np.stack([softmax(logits[:midpoint]), softmax(logits[midpoint:])])

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits)
        alpha0, alpha1 = observatory.phase_fractions(dataset)
        aggregate = alpha0 * weights[0] + alpha1 * weights[1]
        return scalar_prediction(predictor.predict, weights) + aggregate_kl_coefficient * hpr_panel.categorical_kl(
            aggregate,
            natural,
        )

    best: tuple[float, np.ndarray] | None = None
    successful = 0
    for start in policy_starts(dataset, natural, tied=tied, count=OPTIMIZER_STARTS, seed=seed):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-11, "maxls": 40},
        )
        if result.success:
            successful += 1
        weights = weights_from_logits(np.asarray(result.x, dtype=float))
        candidate = (float(result.fun), weights)
        if np.isfinite(candidate[0]) and (best is None or candidate[0] < best[0]):
            best = candidate
    if best is None:
        raise RuntimeError(f"No finite {'tied' if tied else 'two-phase'} optimum")
    return SolveResult(
        weights=best[1],
        prediction=scalar_prediction(predictor.predict, best[1]),
        regularized_objective=best[0],
        successful_starts=successful,
    )


def feasible_phase_start(
    delta: np.ndarray,
    aggregate: np.ndarray,
    active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: float,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    scale = 1.0
    for _attempt in range(60):
        candidate = scale * delta
        full = np.zeros_like(aggregate)
        full[active] = candidate
        weights = phase_information.fixed_aggregate.weights_from_delta(aggregate, full, alpha0, alpha1)
        information = phase_information.fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)
        if np.all(candidate >= lower) and np.all(candidate <= upper) and information <= 0.8 * budget:
            return candidate
        scale *= 0.5
    return np.zeros_like(delta)


def optimize_fixed_aggregate(
    predictor: Predictor,
    aggregate: np.ndarray,
    budget: float,
    alpha0: float,
    alpha1: float,
    seed: int,
) -> SolveResult:
    active = np.flatnonzero(aggregate > 1e-12)
    lower = -aggregate[active] / alpha1
    upper = aggregate[active] / alpha0

    def full_delta(active_delta: np.ndarray) -> np.ndarray:
        delta = np.zeros_like(aggregate)
        delta[active] = active_delta
        return delta

    def weights_from_delta(active_delta: np.ndarray) -> np.ndarray:
        return phase_information.fixed_aggregate.weights_from_delta(
            aggregate,
            full_delta(active_delta),
            alpha0,
            alpha1,
        )

    def information(active_delta: np.ndarray) -> float:
        weights = weights_from_delta(active_delta)
        return phase_information.fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)

    rng = np.random.default_rng(seed)
    starts = [np.zeros(len(active), dtype=float)]
    for _index in range(12):
        random_delta = phase_information.fixed_aggregate.random_start(
            aggregate,
            -aggregate / alpha1,
            aggregate / alpha0,
            rng,
        )[active]
        starts.append(
            feasible_phase_start(
                random_delta,
                aggregate,
                active,
                lower,
                upper,
                budget,
                alpha0,
                alpha1,
            )
        )
    constraints = [
        {"type": "eq", "fun": lambda delta: float(np.sum(delta))},
        {"type": "ineq", "fun": lambda delta: budget - information(np.asarray(delta, dtype=float))},
    ]
    bounds = list(zip(lower, upper, strict=True))
    tied = np.stack([aggregate, aggregate])
    best = (scalar_prediction(predictor.predict, tied), tied)
    successful = 0
    for start in starts:
        result = minimize(
            lambda delta: scalar_prediction(predictor.predict, weights_from_delta(np.asarray(delta, dtype=float))),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-11},
        )
        if result.success:
            successful += 1
        weights = weights_from_delta(np.asarray(result.x, dtype=float))
        prediction = scalar_prediction(predictor.predict, weights)
        if (
            np.isfinite(prediction)
            and information(np.asarray(result.x, dtype=float)) <= budget + 1e-7
            and float(weights.min()) >= -1e-7
            and prediction < best[0]
        ):
            best = (prediction, weights)
    return SolveResult(best[1], best[0], best[0], successful)


def model_geometry(dataset, weights: np.ndarray, natural: np.ndarray) -> dict[str, float]:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    epochs = weights[0] * dataset.c0 + weights[1] * dataset.c1
    return {
        "aggregate_kl_to_proportional": hpr_panel.categorical_kl(aggregate, natural),
        "phase_information_kl": phase_information.fixed_aggregate.phase_order_kl(
            weights,
            aggregate,
            alpha0,
            alpha1,
        ),
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "max_bucket_weight": float(weights.max()),
        "max_simulated_epoch": float(epochs.max()),
    }


def candidate_record(
    *,
    candidate_id: str,
    target: str,
    policy_class: str,
    candidate_kind: str,
    model_id: str,
    weights: np.ndarray,
    selected_prediction: float,
    aggregate_predictor: Predictor,
    phase_predictor: Predictor | None,
    dataset,
    natural: np.ndarray,
    aggregate_kl_coefficient: float | None,
    phase_information_budget: float | None,
    successful_starts: int,
) -> dict[str, Any]:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    tied = np.stack([aggregate, aggregate])
    tied_phase_prediction = scalar_prediction(phase_predictor.predict, tied) if phase_predictor is not None else None
    return {
        "candidate_id": candidate_id,
        "target": target,
        "policy_class": policy_class,
        "candidate_kind": candidate_kind,
        "model": model_id,
        "fit_source": "delphi_3e18",
        "fit_rows": FIT_ROWS,
        "aggregate_kl_coefficient": aggregate_kl_coefficient,
        "phase_information_budget": phase_information_budget,
        "selected_model_prediction": selected_prediction,
        "aggregate_model_tied_prediction": scalar_prediction(aggregate_predictor.predict, tied),
        "phase_model_tied_prediction": tied_phase_prediction,
        "phase_model_predicted_gain": (
            tied_phase_prediction - selected_prediction if tied_phase_prediction is not None else 0.0
        ),
        "successful_starts": successful_starts,
        "coordinate_hash": policy_hash(weights),
        **model_geometry(dataset, weights, natural),
        "weights": weights,
    }


def existing_coordinate(
    weights: np.ndarray,
    fit_weights: np.ndarray,
    heldout_weights: np.ndarray,
    heldout_frame: pd.DataFrame,
    alpha0: float,
    alpha1: float,
) -> tuple[str, str | None]:
    fit_distance = hpr_panel.weighted_policy_tv(weights[None], fit_weights, alpha0, alpha1)
    if float(fit_distance.min()) <= EXACT_POLICY_TV:
        return "fit", None
    heldout_distance = hpr_panel.weighted_policy_tv(weights[None], heldout_weights, alpha0, alpha1)
    index = int(np.argmin(heldout_distance))
    if float(heldout_distance[index]) <= EXACT_POLICY_TV:
        return "heldout", str(heldout_frame.iloc[index]["run_name"])
    return "new", None


def render_diagnostics(manifest: pd.DataFrame, output_dir: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: prediction vs max epochs",
            "Table-9: prediction vs max epochs",
            "Uncheatable: phase information path",
            "Table-9: phase information path",
        ),
    )
    colors = {
        "separate_heads": "#64748B",
        "effective_exposure": "#E76F2E",
        "bucket_family_grp": "#2A9D8F",
        "compact_retained_state": "#457B9D",
        "hierarchical_phase_replay": "#B23A48",
    }
    for column, target in enumerate(TARGETS, start=1):
        selected = manifest.loc[manifest["target"].eq(target)]
        for model_id, group in selected.groupby("model", sort=False):
            figure.add_trace(
                go.Scatter(
                    x=group["max_simulated_epoch"],
                    y=group["selected_model_prediction"],
                    mode="markers",
                    name=model_id,
                    legendgroup=model_id,
                    showlegend=column == 1,
                    marker={"color": colors[model_id], "size": 8},
                    customdata=np.column_stack([group["candidate_id"], group["phase_information_budget"]]),
                    hovertemplate=(
                        "%{customdata[0]}<br>prediction=%{y:.6f}<br>max epochs=%{x:.3f}"
                        "<br>epsilon=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            phase_rows = group.loc[group["candidate_kind"].str.startswith("fixed_aggregate")]
            figure.add_trace(
                go.Scatter(
                    x=phase_rows["phase_information_budget"],
                    y=phase_rows["selected_model_prediction"],
                    mode="markers+lines",
                    name=model_id,
                    legendgroup=model_id,
                    showlegend=False,
                    marker={"color": colors[model_id], "size": 7},
                    line={"color": colors[model_id]},
                    customdata=phase_rows[["candidate_id", "aggregate_kl_coefficient"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>aggregate KL=%{customdata[1]}"
                        "<br>epsilon=%{x}<br>prediction=%{y:.6f}<extra></extra>"
                    ),
                ),
                row=2,
                col=column,
            )
    figure.update_xaxes(type="log", title_text="maximum simulated epochs", row=1)
    figure.update_yaxes(title_text="predicted BPB", row=1)
    figure.update_xaxes(type="log", title_text="phase-information budget", row=2)
    figure.update_yaxes(title_text="predicted BPB", row=2)
    figure.update_layout(
        title="Delphi 3e18 in-swarm hybrid phase-ordering panel",
        template="plotly_white",
        width=1500,
        height=1000,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
        margin={"l": 80, "r": 30, "t": 100, "b": 100},
    )
    figure.write_html(output_dir / "panel_diagnostics.html", include_plotlyjs=True, config=PLOT_CONFIG)


def upload_artifact(local_path: Path, remote_path: str) -> None:
    with local_path.open("rb") as source, fsspec.open(remote_path, "wb") as destination:
        destination.write(source.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument(
        "--aggregate-kl-coefficients",
        default=",".join(str(value) for value in AGGREGATE_KL_COEFFICIENTS),
    )
    parser.add_argument(
        "--phase-information-budgets",
        default=",".join(str(value) for value in PHASE_INFORMATION_BUDGETS),
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    aggregate_kl_coefficients = parse_float_tuple(args.aggregate_kl_coefficients)
    phase_information_budgets = parse_float_tuple(args.phase_information_budgets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mixture_dir = args.output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)

    sources = hpr_panel.composition.load_sources()
    datasets = {
        target: hpr_panel.delphi_3e18_policy_datasets(sources, target)[observatory.TWO_PHASE] for target in TARGETS
    }
    reference = datasets["uncheatable"]
    if any(dataset.n != FIT_ROWS for dataset in datasets.values()):
        raise ValueError("Every target must use exactly 280 two-phase fit rows")
    if not np.allclose(datasets["uncheatable"].weights, datasets["table9"].weights):
        raise ValueError("Target fit panels do not share policy coordinates")
    alpha0, alpha1 = observatory.phase_fractions(reference)
    natural = observatory.natural_weights(reference, alpha0)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)

    records: list[dict[str, Any]] = []
    fit_metadata: dict[str, Any] = {
        "fit_rows": FIT_ROWS,
        "fit_coordinate_sha256": hashlib.sha256(np.round(reference.weights, COORDINATE_DECIMALS).tobytes()).hexdigest(),
        "phase_fractions": [alpha0, alpha1],
        "targets": {},
    }
    for target_index, target in enumerate(TARGETS):
        dataset = datasets[target]
        print(f"Selecting 3e18 separate-heads aggregate fit for {target}", flush=True)
        aggregate_predictor, separate_sweep = fit_separate_anchor(dataset, target)
        print(f"Fitting four 3e18 phase-ordering surrogates for {target}", flush=True)
        phase_predictors = fit_phase_predictors(dataset)
        fit_metadata["targets"][target] = {
            "aggregate_model": aggregate_predictor.selection,
            "aggregate_cv": separate_sweep.to_dict(orient="records"),
            "phase_models": {name: predictor.selection for name, predictor in phase_predictors.items()},
        }
        (args.output_dir / "fitted_models.partial.json").write_text(
            json.dumps(json_clean(fit_metadata), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

        anchors: dict[float, SolveResult] = {}
        for aggregate_kl in aggregate_kl_coefficients:
            result = optimize_policy(
                aggregate_predictor,
                dataset,
                natural,
                tied=True,
                aggregate_kl_coefficient=aggregate_kl,
                seed=20260720 + target_index,
            )
            anchors[aggregate_kl] = result
            candidate_id = f"hyb3_{TARGET_TAGS[target]}_tied_akl{float_tag(aggregate_kl)}"
            records.append(
                candidate_record(
                    candidate_id=candidate_id,
                    target=target,
                    policy_class=observatory.SINGLE_PHASE,
                    candidate_kind="tied_separate_heads_anchor",
                    model_id="separate_heads",
                    weights=result.weights,
                    selected_prediction=result.prediction,
                    aggregate_predictor=aggregate_predictor,
                    phase_predictor=None,
                    dataset=dataset,
                    natural=natural,
                    aggregate_kl_coefficient=aggregate_kl,
                    phase_information_budget=0.0,
                    successful_starts=result.successful_starts,
                )
            )

        for model_index, (model_id, predictor) in enumerate(phase_predictors.items()):
            raw = optimize_policy(
                predictor,
                dataset,
                natural,
                tied=False,
                aggregate_kl_coefficient=0.0,
                seed=20260730 + 10 * target_index + model_index,
            )
            records.append(
                candidate_record(
                    candidate_id=f"hyb3_{TARGET_TAGS[target]}_{MODEL_TAGS[model_id]}_raw",
                    target=target,
                    policy_class=observatory.TWO_PHASE,
                    candidate_kind=f"raw_optimum_{model_id}",
                    model_id=model_id,
                    weights=raw.weights,
                    selected_prediction=raw.prediction,
                    aggregate_predictor=aggregate_predictor,
                    phase_predictor=predictor,
                    dataset=dataset,
                    natural=natural,
                    aggregate_kl_coefficient=None,
                    phase_information_budget=None,
                    successful_starts=raw.successful_starts,
                )
            )
            for aggregate_kl, anchor in anchors.items():
                aggregate = anchor.weights[0]
                for budget_index, budget in enumerate(phase_information_budgets):
                    result = optimize_fixed_aggregate(
                        predictor,
                        aggregate,
                        budget,
                        alpha0,
                        alpha1,
                        seed=20260800 + 100 * target_index + 10 * model_index + budget_index,
                    )
                    candidate_id = (
                        f"hyb3_{TARGET_TAGS[target]}_{MODEL_TAGS[model_id]}_"
                        f"akl{float_tag(aggregate_kl)}_eps{float_tag(budget)}"
                    )
                    records.append(
                        candidate_record(
                            candidate_id=candidate_id,
                            target=target,
                            policy_class=observatory.TWO_PHASE,
                            candidate_kind=f"fixed_aggregate_{model_id}",
                            model_id=model_id,
                            weights=result.weights,
                            selected_prediction=result.prediction,
                            aggregate_predictor=aggregate_predictor,
                            phase_predictor=predictor,
                            dataset=dataset,
                            natural=natural,
                            aggregate_kl_coefficient=aggregate_kl,
                            phase_information_budget=budget,
                            successful_starts=result.successful_starts,
                        )
                    )

    first_by_hash: dict[str, str] = {}
    for record in records:
        weights = np.asarray(record.pop("weights"), dtype=float)
        if weights.shape != (2, reference.m) or np.any(weights < -1e-8):
            raise ValueError(f"Invalid weights for {record['candidate_id']}")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError(f"Unnormalized weights for {record['candidate_id']}")
        coordinate_hash = str(record["coordinate_hash"])
        record["duplicate_coordinate"] = coordinate_hash in first_by_hash
        record["coordinate_primary_candidate"] = first_by_hash.setdefault(
            coordinate_hash,
            str(record["candidate_id"]),
        )
        coordinate_kind, existing_run = existing_coordinate(
            weights,
            reference.weights,
            heldout_weights,
            heldout_frame,
            alpha0,
            alpha1,
        )
        record["existing_coordinate"] = coordinate_kind
        record["existing_run_name"] = existing_run
        mixture_path = mixture_dir / f"{record['candidate_id']}.csv"
        hpr_panel.mixture_frame(reference, natural, weights).to_csv(mixture_path, index=False)
        record["mixture_path"] = str(mixture_path.relative_to(args.output_dir))
        for phase in (0, 1):
            for domain, weight in zip(reference.domain_names, weights[phase], strict=True):
                record[f"phase_{phase}_{domain}"] = float(weight)

    manifest = pd.DataFrame(records).sort_values(
        ["target", "policy_class", "model", "aggregate_kl_coefficient", "phase_information_budget"],
        na_position="first",
    )
    launch = manifest.loc[manifest["existing_coordinate"].eq("new") & ~manifest["duplicate_coordinate"]].reset_index(
        drop=True
    )
    if launch.empty:
        raise RuntimeError("No new policies remain after deduplication")
    phase_columns = [
        column for column in manifest.columns if column.startswith("phase_0_") or column.startswith("phase_1_")
    ]
    launcher_columns = [
        "candidate_id",
        "target",
        "policy_class",
        "candidate_kind",
        "fit_source",
        "aggregate_kl_coefficient",
        "phase_information_budget",
        "selected_model_prediction",
        "aggregate_kl_to_proportional",
        "phase_information_kl",
        "max_simulated_epoch",
        *phase_columns,
    ]
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    launch[launcher_columns].to_csv(args.output_dir / "launcher_source_panel.csv", index=False)
    (args.output_dir / "fitted_models.json").write_text(
        json.dumps(json_clean(fit_metadata), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    render_diagnostics(manifest, args.output_dir)

    report_lines = [
        "# Delphi 3e18 in-swarm hybrid phase-ordering panel",
        "",
        "- Every fit uses the same 280-row two-phase Delphi 3e18 panel; no independent one-phase panel is used.",
        "- A three-seed CV-selected separate-heads fit chooses tied aggregate anchors.",
        "- Four phase models choose only phase order at exactly fixed aggregate exposure.",
        f"- Aggregate KL sweep: `{list(aggregate_kl_coefficients)}`.",
        f"- Phase-information sweep: `{list(phase_information_budgets)}`.",
        f"- Proposal arms before deduplication: `{len(manifest)}`; new unique launch rows: `{len(launch)}`.",
        "- Raw optima are retained as explicit extrapolation diagnostics.",
        "",
        "## Launch counts",
        "",
        launch.groupby(["target", "policy_class", "candidate_kind"])
        .size()
        .rename("rows")
        .reset_index()
        .to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report_lines))

    source_sha = hashlib.sha256((args.output_dir / "launcher_source_panel.csv").read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((args.output_dir / "candidate_manifest.csv").read_bytes()).hexdigest()
    gcs_source = f"{args.gcs_output_dir.rstrip('/')}/source/launcher_source_panel-{source_sha[:16]}.csv"
    gcs_manifest = f"{args.gcs_output_dir.rstrip('/')}/source/candidate_manifest-{manifest_sha[:16]}.csv"
    summary = {
        "proposal_arms": len(manifest),
        "launch_ready_unique_new_coordinates": len(launch),
        "fit_source": "delphi_3e18",
        "fit_rows": FIT_ROWS,
        "aggregate_kl_coefficients": list(aggregate_kl_coefficients),
        "phase_information_budgets": list(phase_information_budgets),
        "models": list(MODELS),
        "source_phase_fractions": [alpha0, alpha1],
        "launcher_source_panel_sha256": source_sha,
        "candidate_manifest_sha256": manifest_sha,
        "gcs_launcher_source_panel": gcs_source,
        "gcs_candidate_manifest": gcs_manifest,
        "uploaded": bool(args.upload),
        "jobs_submitted": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.upload:
        upload_artifact(args.output_dir / "launcher_source_panel.csv", gcs_source)
        upload_artifact(args.output_dir / "candidate_manifest.csv", gcs_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
