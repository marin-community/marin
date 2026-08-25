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
"""Materialize a matched one-/two-phase HPR optimum validation panel at 3e18.

Fit hyperparameters are selected only by fit-panel cross-validation. Deployment
regularization is kept separate: the one-phase fit chooses an aggregate under a
KL penalty to proportional, while the two-phase fit chooses phase order at that
exact aggregate under a phase-information budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.special import softmax

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_decoupled_phase_information_constraints_300m as phase_information,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_3e18_fixed_budget_frontier_composition as composition,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIT_SOURCES = ("300m", "delphi_3e18")
DEFAULT_OUTPUT_DIRS = {
    "300m": SCRIPT_DIR / "reference_outputs/hpr_300m_to_3e18_optimum_validation_panel_20260720",
    "delphi_3e18": SCRIPT_DIR / "reference_outputs/hpr_3e18_to_3e18_optimum_validation_panel_20260720",
}
DEFAULT_GCS_OUTPUT_DIRS = {
    "300m": ("gs://marin-us-east5/pinlin_calvin_xu/data_mixture/hpr_300m_to_3e18_optimum_validation_panel_20260720"),
    "delphi_3e18": (
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/hpr_3e18_to_3e18_optimum_validation_panel_20260720"
    ),
}
TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
TARGET_TAGS = {"uncheatable": "unch", "table9": "t9"}
AGGREGATE_KL_COEFFICIENTS = (0.0, 0.025, 0.05, 0.1, 0.2)
PHASE_INFORMATION_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.025)
POLICY_CLASSES = (observatory.SINGLE_PHASE, observatory.TWO_PHASE)
FIT_ROWS = 280
ONE_PHASE_CONTROL_ROWS = 42
ONE_PHASE_NEW_ROWS = 238
OPTIMIZER_STARTS = 24
SENSITIVITY_CONFIGS = 3
COORDINATE_DECIMALS = 12
EXACT_POLICY_TV = 1e-9
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class OptimizationResult:
    """One finite multistart policy optimum."""

    weights: np.ndarray
    predicted_bpb: float
    regularized_objective: float
    successful_starts: int
    finite_starts: int


@dataclass(frozen=True)
class FittedPolicy:
    """CV-selected HPR fit for one target and one policy class."""

    target: str
    policy_class: str
    dataset: pooled.Dataset
    config: hierarchical.Config
    model: hierarchical.Model
    sweep: pd.DataFrame


@dataclass(frozen=True)
class FixedAggregateResult:
    """A phase-order optimum at one fixed aggregate distribution."""

    weights: np.ndarray
    prediction: float
    successful_starts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-source", choices=FIT_SOURCES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gcs-output-dir")
    parser.add_argument("--exclude-source-panel", type=Path)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--aggregate-kl-coefficients",
        default=",".join(str(value) for value in AGGREGATE_KL_COEFFICIENTS),
    )
    parser.add_argument(
        "--phase-information-budgets",
        default=",".join(str(value) for value in PHASE_INFORMATION_BUDGETS),
    )
    parser.add_argument("--optimizer-starts", type=int, default=OPTIMIZER_STARTS)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIRS[args.fit_source]
    if args.gcs_output_dir is None:
        args.gcs_output_dir = DEFAULT_GCS_OUTPUT_DIRS[args.fit_source]
    return args


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


def weighted_policy_tv(left: np.ndarray, right: np.ndarray, alpha0: float, alpha1: float) -> np.ndarray:
    phase0 = 0.5 * np.abs(left[..., 0, :] - right[..., 0, :]).sum(axis=-1)
    phase1 = 0.5 * np.abs(left[..., 1, :] - right[..., 1, :]).sum(axis=-1)
    return alpha0 * phase0 + alpha1 * phase1


def categorical_kl(left: np.ndarray, right: np.ndarray) -> float:
    safe_left = np.clip(np.asarray(left, dtype=float), 1e-12, 1.0)
    safe_right = np.clip(np.asarray(right, dtype=float), 1e-12, 1.0)
    return float(np.sum(safe_left * (np.log(safe_left) - np.log(safe_right))))


def make_raw_dataset(
    reference: pooled.Dataset,
    frame: pd.DataFrame,
    weights: np.ndarray,
    target: str,
    name: str,
) -> pooled.Dataset:
    return pooled.Dataset(
        name=name,
        frame=frame.reset_index(drop=True),
        y=frame[TARGET_COLUMNS[target]].to_numpy(dtype=float),
        weights=np.asarray(weights, dtype=float),
        c0=np.asarray(reference.c0, dtype=float),
        c1=np.asarray(reference.c1, dtype=float),
        domain_names=list(reference.domain_names),
    )


def delphi_3e18_policy_datasets(sources: composition.Sources, target: str) -> dict[str, pooled.Dataset]:
    tied = np.max(np.abs(sources.broad.weights[:, 0] - sources.broad.weights[:, 1]), axis=1) < 1e-10
    if int(tied.sum()) != ONE_PHASE_CONTROL_ROWS:
        raise ValueError(f"Expected {ONE_PHASE_CONTROL_ROWS} tied fit controls, found {int(tied.sum())}")
    if len(sources.single.frame) != ONE_PHASE_NEW_ROWS:
        raise ValueError(f"Expected {ONE_PHASE_NEW_ROWS} new one-phase rows, found {len(sources.single.frame)}")

    one_phase_frame = pd.concat(
        [sources.broad.frame.loc[tied], sources.single.frame],
        ignore_index=True,
        sort=False,
    )
    one_phase_frame["panel_source"] = one_phase_frame["source_pool"].fillna("one_phase")
    one_phase_weights = np.concatenate([sources.broad.weights[tied], sources.single.weights], axis=0)
    if len(one_phase_frame) != FIT_ROWS or not np.allclose(one_phase_weights[:, 0], one_phase_weights[:, 1]):
        raise ValueError("The one-phase fit must contain 280 phase-tied policies")

    two_phase_frame = sources.broad.frame.copy().reset_index(drop=True)
    if len(two_phase_frame) != FIT_ROWS:
        raise ValueError(f"Expected {FIT_ROWS} two-phase fit rows, found {len(two_phase_frame)}")
    return {
        observatory.SINGLE_PHASE: make_raw_dataset(
            sources.reference,
            one_phase_frame,
            one_phase_weights,
            target,
            f"delphi_3e18_single_{target}",
        ),
        observatory.TWO_PHASE: make_raw_dataset(
            sources.reference,
            two_phase_frame,
            sources.broad.weights,
            target,
            f"delphi_3e18_{target}",
        ),
    }


def policy_datasets(
    fit_source: str,
    target: str,
    sources: composition.Sources | None,
) -> dict[str, pooled.Dataset]:
    if fit_source == "delphi_3e18":
        if sources is None:
            raise ValueError("The Delphi 3e18 source requires loaded composition sources")
        return delphi_3e18_policy_datasets(sources, target)
    if fit_source != "300m":
        raise ValueError(f"Unknown fit source {fit_source!r}")

    two_phase = pooled.load_300m_dataset(target)
    one_phase = observatory.load_300m_single_phase_dataset(target, two_phase)
    if two_phase.n != FIT_ROWS or one_phase.n != FIT_ROWS:
        raise ValueError(
            f"Expected matched 280-row 300M panels, found {two_phase.n} two-phase and {one_phase.n} one-phase"
        )
    if not np.allclose(one_phase.weights[:, 0], one_phase.weights[:, 1], atol=1e-10):
        raise ValueError("The 300M one-phase fit contains a phase-untied policy")
    return {
        observatory.SINGLE_PHASE: one_phase,
        observatory.TWO_PHASE: two_phase,
    }


def fit_policy(dataset: pooled.Dataset, target: str, policy_class: str) -> FittedPolicy:
    config, selection = observatory.select_hierarchical_phase_replay_config(dataset, policy_class)
    if policy_class == observatory.SINGLE_PHASE:
        if config.shape.late_multiplier != 1.0 or config.shape.forgetting_rate != 0.0:
            raise ValueError("The one-phase HPR restriction retained phase-order parameters")
    sweep = pd.DataFrame(selection["candidateSweep"]).sort_values(["rmse", "spearman"], ascending=[True, False])
    model = observatory.hierarchical_phase_replay_fit(dataset, np.arange(dataset.n), config)
    return FittedPolicy(target, policy_class, dataset, config, model, sweep.reset_index(drop=True))


def config_from_row(row: pd.Series) -> hierarchical.Config:
    shape = family_grp.Shape(
        exponent=float(row["exponent"]),
        late_multiplier=float(row["late_multiplier"]),
        forgetting_rate=float(row["forgetting_rate"]),
        penalty_threshold=float(row["penalty_threshold"]),
        quality_discount=float(row.get("quality_discount", 1.0)),
    )
    return hierarchical.Config(
        variant=hierarchical.Variant(str(row["variant"])),
        shape_index=int(row["shape_index"]),
        shape=shape,
        l2=float(row["l2"]),
        residual_shrink=float(row["residual_shrink"]),
        undercoverage_fraction=float(row["undercoverage_fraction"]),
        coverage_gate_ratio=float(row["coverage_gate_ratio"]),
    )


def scalar_prediction(model: hierarchical.Model, weights: np.ndarray) -> float:
    return float(model.predict(np.asarray(weights, dtype=float)[None, :, :])[0])


def optimization_starts(
    dataset: pooled.Dataset,
    natural: np.ndarray,
    policy_class: str,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)

    def logits(weights: np.ndarray) -> np.ndarray:
        values = weights[0] if policy_class == observatory.SINGLE_PHASE else weights
        logged = np.log(np.clip(values, 1e-12, 1.0))
        return np.clip(logged[..., :-1] - logged[..., [-1]], -10.0, 10.0).reshape(-1)

    starts = [logits(np.stack([natural, natural]))]
    starts.extend(logits(dataset.weights[index]) for index in np.argsort(dataset.y)[: min(8, dataset.n)])
    concentrations = (0.25, 1.0, 4.0)
    while len(starts) < count:
        concentration = concentrations[(len(starts) - 1) % len(concentrations)]
        if policy_class == observatory.SINGLE_PHASE:
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
    fitted: FittedPolicy,
    natural: np.ndarray,
    aggregate_kl_coefficient: float,
    starts: int,
    seed: int,
) -> OptimizationResult:
    dataset = fitted.dataset
    one_phase = fitted.policy_class == observatory.SINGLE_PHASE

    def weights_from_logits(logits: np.ndarray) -> np.ndarray:
        phase_count = 1 if one_phase else 2
        reduced = np.asarray(logits, dtype=float).reshape(phase_count, dataset.m - 1)
        values = softmax(np.column_stack([reduced, np.zeros(phase_count, dtype=float)]), axis=1)
        if one_phase:
            return np.stack([values[0], values[0]])
        return values

    alpha0, alpha1 = observatory.phase_fractions(dataset)

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits)
        aggregate = alpha0 * weights[0] + alpha1 * weights[1]
        return scalar_prediction(fitted.model, weights) + aggregate_kl_coefficient * categorical_kl(
            aggregate,
            natural,
        )

    best: tuple[float, np.ndarray] | None = None
    successful = 0
    finite = 0
    for start in optimization_starts(dataset, natural, fitted.policy_class, starts, seed):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-10.0, 10.0)] * len(start),
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
        finite += 1
        weights = weights_from_logits(np.asarray(result.x, dtype=float))
        candidate = (float(result.fun), weights)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError(
            f"No finite optimum for {fitted.target}/{fitted.policy_class}, KL={aggregate_kl_coefficient:g}"
        )
    return OptimizationResult(
        weights=best[1],
        predicted_bpb=scalar_prediction(fitted.model, best[1]),
        regularized_objective=best[0],
        successful_starts=successful,
        finite_starts=finite,
    )


def feasible_phase_start(
    delta: np.ndarray,
    aggregate: np.ndarray,
    active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    phase_information_budget: float,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    scale = 1.0
    for _attempt in range(60):
        candidate = scale * delta
        full_delta = np.zeros_like(aggregate)
        full_delta[active] = candidate
        weights = phase_information.fixed_aggregate.weights_from_delta(aggregate, full_delta, alpha0, alpha1)
        information = phase_information.fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)
        if np.all(candidate >= lower) and np.all(candidate <= upper) and information <= 0.8 * phase_information_budget:
            return candidate
        scale *= 0.5
    return np.zeros_like(delta)


def optimize_fixed_aggregate(
    fitted: FittedPolicy,
    aggregate: np.ndarray,
    phase_information_budget: float,
    alpha0: float,
    alpha1: float,
) -> FixedAggregateResult:
    """Optimize phase order using the realized rather than nominal phase split."""
    active = np.flatnonzero(aggregate > 1e-12)
    if len(active) < 2:
        tied = np.stack([aggregate, aggregate])
        return FixedAggregateResult(tied, scalar_prediction(fitted.model, tied), 1)
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

    def information(delta: np.ndarray) -> float:
        weights = weights_from_delta(delta)
        return phase_information.fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)

    rng = np.random.default_rng(20260720)
    starts = [np.zeros(len(active), dtype=float)]
    starts.extend(
        feasible_phase_start(
            phase_information.fixed_aggregate.random_start(
                aggregate,
                -aggregate / alpha1,
                aggregate / alpha0,
                rng,
            )[active],
            aggregate,
            active,
            lower,
            upper,
            phase_information_budget,
            alpha0,
            alpha1,
        )
        for _index in range(12)
    )
    constraints = [
        {"type": "eq", "fun": lambda delta: float(np.sum(delta))},
        {"type": "ineq", "fun": lambda delta: phase_information_budget - information(delta)},
    ]
    bounds = list(zip(lower, upper, strict=True))
    tied = np.stack([aggregate, aggregate])
    best: tuple[float, np.ndarray] | None = (scalar_prediction(fitted.model, tied), tied)
    successful = 0
    for start in starts:
        result = minimize(
            lambda delta: scalar_prediction(fitted.model, weights_from_delta(np.asarray(delta, dtype=float))),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-11},
        )
        if result.success:
            successful += 1
        weights = weights_from_delta(np.asarray(result.x, dtype=float))
        prediction = scalar_prediction(fitted.model, weights)
        if (
            np.isfinite(prediction)
            and information(np.asarray(result.x, dtype=float)) <= phase_information_budget + 1e-7
            and float(weights.min()) >= -1e-7
        ):
            candidate = (prediction, weights)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise RuntimeError(f"No fixed-aggregate solution for epsilon={phase_information_budget:g}")
    return FixedAggregateResult(best[1], best[0], successful)


def nearest_policy_tv(weights: np.ndarray, references: np.ndarray, alpha0: float, alpha1: float) -> float:
    return float(weighted_policy_tv(weights[None, :, :], references, alpha0, alpha1).min())


def candidate_geometry(
    weights: np.ndarray,
    dataset: pooled.Dataset,
    natural: np.ndarray,
    heldout_weights: np.ndarray,
) -> dict[str, float | int | str]:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    phase_kl = phase_information.fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)
    epochs = weights[0] * dataset.c0 + weights[1] * dataset.c1
    return {
        "coordinate_hash": policy_hash(weights),
        "aggregate_kl_to_proportional": categorical_kl(aggregate, natural),
        "phase_information_kl": float(phase_kl),
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "max_bucket_weight": float(weights.max()),
        "max_simulated_epoch": float(epochs.max()),
        "mean_simulated_epoch": float(epochs.mean()),
        "min_fit_policy_tv": nearest_policy_tv(weights, dataset.weights, alpha0, alpha1),
        "min_existing_heldout_policy_tv": nearest_policy_tv(weights, heldout_weights, alpha0, alpha1),
    }


def record_candidate(
    *,
    candidate_id: str,
    target: str,
    policy_class: str,
    candidate_kind: str,
    weights: np.ndarray,
    one_phase: FittedPolicy,
    two_phase: FittedPolicy,
    natural: np.ndarray,
    deployment_dataset: pooled.Dataset,
    deployment_natural: np.ndarray,
    heldout_weights: np.ndarray,
    aggregate_kl_coefficient: float | None,
    phase_information_budget: float | None,
    regularized_objective: float,
    successful_starts: int,
    finite_starts: int,
) -> dict[str, Any]:
    selected = one_phase if policy_class == observatory.SINGLE_PHASE else two_phase
    source_geometry = candidate_geometry(weights, selected.dataset, natural, heldout_weights)
    deployment_geometry = candidate_geometry(
        weights,
        deployment_dataset,
        deployment_natural,
        heldout_weights,
    )
    return {
        "candidate_id": candidate_id,
        "target": target,
        "model": hierarchical.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY.value,
        "policy_class": policy_class,
        "candidate_kind": candidate_kind,
        "aggregate_kl_coefficient": aggregate_kl_coefficient,
        "phase_information_budget": phase_information_budget,
        "selected_model_prediction": scalar_prediction(selected.model, weights),
        "one_phase_model_prediction": scalar_prediction(one_phase.model, weights),
        "two_phase_model_prediction": scalar_prediction(two_phase.model, weights),
        "regularized_objective": regularized_objective,
        "successful_starts": successful_starts,
        "finite_starts": finite_starts,
        **deployment_geometry,
        **{f"source_{key}": value for key, value in source_geometry.items() if key != "coordinate_hash"},
        "weights": weights,
    }


def mixture_frame(dataset: pooled.Dataset, natural: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    budget = observatory.target_budget(dataset, alpha0)
    token_counts = alpha0 * budget * natural / np.maximum(dataset.c0, 1e-12)
    return pd.DataFrame(
        {
            "domain": dataset.domain_names,
            "proportional": natural,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "aggregate_weight": aggregate,
            "available_tokens": token_counts,
            "simulated_epochs": weights[0] * dataset.c0 + weights[1] * dataset.c1,
            "phase_0_epoch_multiplier": weights[0] / np.maximum(natural, 1e-12),
            "phase_1_epoch_multiplier": weights[1] / np.maximum(natural, 1e-12),
            "phase_0_delta": weights[0] - natural,
            "phase_1_delta": weights[1] - natural,
        }
    )


def selected_config_row(fitted: FittedPolicy, fit_source: str) -> dict[str, Any]:
    winner = fitted.sweep.iloc[0]
    runner_up = fitted.sweep.iloc[1]
    config = fitted.config
    return {
        "fit_source": fit_source,
        "target": fitted.target,
        "policy_class": fitted.policy_class,
        "fit_rows": fitted.dataset.n,
        "variant": config.variant.value,
        "shape_index": config.shape_index,
        **asdict(config.shape),
        "l2": config.l2,
        "residual_shrink": config.residual_shrink,
        "oof_rmse": float(winner["rmse"]),
        "oof_spearman": float(winner["spearman"]),
        "runner_up_oof_rmse": float(runner_up["rmse"]),
        "runner_up_relative_rmse_gap": float(runner_up["rmse"] / winner["rmse"] - 1.0),
    }


def cv_optimum_sensitivity(
    fitted: FittedPolicy,
    natural: np.ndarray,
    starts: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference: np.ndarray | None = None
    for rank, candidate in fitted.sweep.head(SENSITIVITY_CONFIGS).iterrows():
        config = config_from_row(candidate)
        model = observatory.hierarchical_phase_replay_fit(fitted.dataset, np.arange(fitted.dataset.n), config)
        local = FittedPolicy(fitted.target, fitted.policy_class, fitted.dataset, config, model, fitted.sweep)
        result = optimize_policy(local, natural, 0.0, max(12, starts // 2), 20260720 + int(rank))
        if reference is None:
            reference = result.weights
        alpha0, alpha1 = observatory.phase_fractions(fitted.dataset)
        rows.append(
            {
                "target": fitted.target,
                "policy_class": fitted.policy_class,
                "cv_rank": int(rank),
                "oof_rmse": float(candidate["rmse"]),
                "oof_spearman": float(candidate["spearman"]),
                "shape_index": config.shape_index,
                **asdict(config.shape),
                "l2": config.l2,
                "residual_shrink": config.residual_shrink,
                "raw_predicted_bpb": result.predicted_bpb,
                "policy_tv_from_cv_winner": float(
                    weighted_policy_tv(result.weights[None], reference[None], alpha0, alpha1).item()
                ),
                **candidate_geometry(result.weights, fitted.dataset, natural, fitted.dataset.weights),
            }
        )
    return rows


def existing_coordinate(
    weights: np.ndarray,
    fit_weights: np.ndarray,
    heldout_weights: np.ndarray,
    heldout_frame: pd.DataFrame,
    alpha0: float,
    alpha1: float,
) -> tuple[str, str | None]:
    fit_distance = weighted_policy_tv(weights[None, None], fit_weights[None], alpha0, alpha1).reshape(-1)
    if float(fit_distance.min()) <= EXACT_POLICY_TV:
        return "fit", None
    heldout_distance = weighted_policy_tv(
        weights[None, None],
        heldout_weights[None],
        alpha0,
        alpha1,
    ).reshape(-1)
    index = int(np.argmin(heldout_distance))
    if float(heldout_distance[index]) <= EXACT_POLICY_TV:
        return "heldout", str(heldout_frame.iloc[index]["wandb_run_name"])
    return "new", None


def render_diagnostics(manifest: pd.DataFrame, output_dir: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable predicted path",
            "Table-9 predicted path",
            "Uncheatable policy geometry",
            "Table-9 policy geometry",
        ),
        vertical_spacing=0.14,
    )
    colors = {observatory.SINGLE_PHASE: "#e76f51", observatory.TWO_PHASE: "#264653"}
    for column, target in enumerate(TARGETS, start=1):
        selected = manifest.loc[manifest["target"].eq(target)]
        for policy_class in POLICY_CLASSES:
            local = selected.loc[selected["policy_class"].eq(policy_class)].copy()
            x = local["aggregate_kl_coefficient"].fillna(-0.025)
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=local["selected_model_prediction"],
                    mode="markers",
                    marker={
                        "color": colors[policy_class],
                        "size": 8,
                        "opacity": 0.72,
                    },
                    name=policy_class,
                    legendgroup=policy_class,
                    showlegend=column == 1,
                    customdata=np.column_stack(
                        [
                            local["candidate_id"],
                            local["phase_information_budget"].fillna(-1.0),
                            local["max_simulated_epoch"],
                            local["existing_coordinate"],
                        ]
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>aggregate KL coefficient=%{x:.4f}<br>"
                        "phase budget=%{customdata[1]:.4f}<br>pred=%{y:.5f}<br>"
                        "max epoch=%{customdata[2]:.2f}<br>coordinate=%{customdata[3]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=local["phase_information_kl"],
                    y=local["max_simulated_epoch"],
                    mode="markers",
                    marker={
                        "color": local["selected_model_prediction"],
                        "colorscale": "RdYlGn_r",
                        "size": 8,
                        "showscale": False,
                    },
                    name=policy_class,
                    legendgroup=policy_class,
                    showlegend=False,
                    text=local["candidate_id"],
                    hovertemplate="%{text}<br>phase information=%{x:.5f}<br>max epoch=%{y:.2f}<extra></extra>",
                ),
                row=2,
                col=column,
            )
    figure.update_xaxes(title_text="aggregate KL penalty coefficient", row=1)
    figure.update_xaxes(title_text="realized phase-information KL", row=2)
    figure.update_yaxes(title_text="predicted BPB", row=1)
    figure.update_yaxes(title_text="maximum simulated epoch", row=2)
    figure.update_layout(
        title="HPR matched one-/two-phase validation panel: local policy audit",
        template="plotly_white",
        width=1500,
        height=950,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_dir / "panel_diagnostics.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    manifest: pd.DataFrame,
    launch: pd.DataFrame,
    selected_configs: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_dir: Path,
    fit_source: str,
) -> None:
    selected_columns = [
        "target",
        "policy_class",
        "oof_rmse",
        "oof_spearman",
        "l2",
        "residual_shrink",
        "exponent",
        "late_multiplier",
        "forgetting_rate",
        "penalty_threshold",
        "runner_up_relative_rmse_gap",
    ]
    sensitivity_summary = sensitivity.groupby(["target", "policy_class"], as_index=False).agg(
        max_top3_policy_tv=("policy_tv_from_cv_winner", "max"),
        max_top3_epoch=("max_simulated_epoch", "max"),
        min_top3_predicted_bpb=("raw_predicted_bpb", "min"),
    )
    lines = [
        "# Hierarchical phase replay optimum validation panel",
        "",
        f"Source fit scale: `{fit_source}`. Deployment and evaluation scale: Delphi `3e18`.",
        "",
        "## Decision boundary",
        "",
        (
            "The surrogate is frozen to Hierarchical phase replay. The one-phase restriction is fitted independently "
            "on the matched 280-row one-phase panel; the two-phase model is fitted on the matched 280-row two-phase "
            f"panel at `{fit_source}`. No Delphi 3e18 heldout outcomes enter model fitting, hyperparameter selection, "
            "or candidate optimization."
        ),
        "",
        (
            "Fit hyperparameters have a nominal five-fold fit-panel CV winner and are not training-swept. Their very "
            "small CV margins and raw-optimum sensitivity remain a model-identification warning, not a deployment "
            "tuning axis. This matches the current Observatory fit-panel selector; it is not the older "
            "panel-source-stratified CV protocol."
        ),
        "",
        (
            "The recent paired-random-effects GLS variant is not used to generate candidates. Its audit classified "
            "it as an observation-model improvement rather than a new response surface, and its raw optima remained "
            "unsupported and unstable."
        ),
        "",
        (
            "Deployment regularization remains unresolved after CV. The panel therefore sweeps the aggregate KL "
            "coefficient and, for the two-phase model, a phase-information budget while holding the aggregate exactly "
            "equal to the independently fitted one-phase optimum."
        ),
        "",
        "## Selected fit configurations",
        "",
        selected_configs[selected_columns].to_markdown(index=False, floatfmt=".6g"),
        "",
        "## CV-near-tie raw-optimum sensitivity",
        "",
        sensitivity_summary.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Candidate counts",
        "",
        f"- Proposal arms before coordinate deduplication: {len(manifest)}.",
        f"- New unique launch-ready coordinates: {len(launch)}.",
        f"- Existing fit coordinates: {int(manifest['existing_coordinate'].eq('fit').sum())}.",
        f"- Existing heldout coordinates: {int(manifest['existing_coordinate'].eq('heldout').sum())}.",
        f"- Duplicate proposal aliases: {int(manifest['duplicate_coordinate'].sum())}.",
        "",
        "## Panel construction",
        "",
        (
            "For each target, optimize five one-phase aggregates with KL coefficients "
            f"{list(AGGREGATE_KL_COEFFICIENTS)}. Include the unconstrained two-phase raw optimum. For each one-phase "
            "aggregate, optimize phase order under phase-information budgets "
            f"{list(PHASE_INFORMATION_BUDGETS)}. The phase-ordered policies preserve the one-phase aggregate exactly."
        ),
        "",
        "The phase-tied one-phase candidate at each aggregate is the exact control for every phase-ordered candidate "
        "on that aggregate; no duplicate epsilon-zero checkpoint is needed.",
        "",
        "Uploading is optional and content-addressed. This materializer never submits training jobs.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def upload_artifact(local_path: Path, remote_path: str) -> None:
    with local_path.open("rb") as source, fsspec.open(remote_path, "wb") as destination:
        destination.write(source.read())


def main() -> None:
    args = parse_args()
    aggregate_kl_coefficients = parse_float_tuple(args.aggregate_kl_coefficients)
    phase_information_budgets = parse_float_tuple(args.phase_information_budgets)
    if 0.0 not in aggregate_kl_coefficients:
        raise ValueError("The aggregate path must include the raw one-phase optimum at coefficient zero")
    if any(value <= 0.0 for value in phase_information_budgets):
        raise ValueError("Positive phase-information budgets are required; tied controls are the one-phase candidates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mixtures_dir = args.output_dir / "mixtures"
    mixtures_dir.mkdir(parents=True, exist_ok=True)

    sources = composition.load_sources() if args.fit_source == "delphi_3e18" else None
    datasets_by_target = {target: policy_datasets(args.fit_source, target, sources) for target in TARGETS}
    reference = datasets_by_target[TARGETS[0]][observatory.TWO_PHASE]
    alpha0, alpha1 = observatory.phase_fractions(reference)
    if not np.allclose((alpha0, alpha1), (0.8, 0.2), atol=0.005):
        raise ValueError(f"Expected an approximately 80/20 phase split, found {alpha0:.6f}/{alpha1:.6f}")
    natural = observatory.natural_weights(reference, alpha0)
    deployment_reference = observatory.load_delphi_3e18_fit_dataset("uncheatable")
    deployment_alpha0, deployment_alpha1 = observatory.phase_fractions(deployment_reference)
    deployment_natural = observatory.natural_weights(deployment_reference, deployment_alpha0)
    if list(reference.domain_names) != list(deployment_reference.domain_names):
        raise ValueError("Source and deployment domain orders differ")
    if not np.allclose(natural, deployment_natural, atol=1e-10):
        raise ValueError("Source and deployment proportional anchors differ")
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(deployment_reference)
    deployment_fit_weights = deployment_reference.weights

    fitted: dict[tuple[str, str], FittedPolicy] = {}
    config_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        datasets = datasets_by_target[target]
        for policy_class in POLICY_CLASSES:
            print(f"Selecting HPR config for {target}/{policy_class}", flush=True)
            policy_fit = fit_policy(datasets[policy_class], target, policy_class)
            fitted[(target, policy_class)] = policy_fit
            config_rows.append(selected_config_row(policy_fit, args.fit_source))
            sensitivity_rows.extend(cv_optimum_sensitivity(policy_fit, natural, args.optimizer_starts))

    records: list[dict[str, Any]] = []
    candidate_prefix = "hpr300m" if args.fit_source == "300m" else "hpr3e18"
    for target in TARGETS:
        one_phase = fitted[(target, observatory.SINGLE_PHASE)]
        two_phase = fitted[(target, observatory.TWO_PHASE)]
        aggregates: dict[float, OptimizationResult] = {}
        for aggregate_kl in aggregate_kl_coefficients:
            print(f"Optimizing {target}/one-phase at KL coefficient {aggregate_kl:g}", flush=True)
            result = optimize_policy(one_phase, natural, aggregate_kl, args.optimizer_starts, 20260720)
            aggregates[aggregate_kl] = result
            records.append(
                record_candidate(
                    candidate_id=f"{candidate_prefix}_{TARGET_TAGS[target]}_1p_aklc{float_tag(aggregate_kl)}",
                    target=target,
                    policy_class=observatory.SINGLE_PHASE,
                    candidate_kind="one_phase_aggregate_path",
                    weights=result.weights,
                    one_phase=one_phase,
                    two_phase=two_phase,
                    natural=natural,
                    deployment_dataset=deployment_reference,
                    deployment_natural=deployment_natural,
                    heldout_weights=heldout_weights,
                    aggregate_kl_coefficient=aggregate_kl,
                    phase_information_budget=0.0,
                    regularized_objective=result.regularized_objective,
                    successful_starts=result.successful_starts,
                    finite_starts=result.finite_starts,
                )
            )

        print(f"Optimizing {target}/two-phase raw optimum", flush=True)
        raw = optimize_policy(two_phase, natural, 0.0, args.optimizer_starts, 20260721)
        records.append(
            record_candidate(
                candidate_id=f"{candidate_prefix}_{TARGET_TAGS[target]}_2p_raw",
                target=target,
                policy_class=observatory.TWO_PHASE,
                candidate_kind="two_phase_raw_optimum",
                weights=raw.weights,
                one_phase=one_phase,
                two_phase=two_phase,
                natural=natural,
                deployment_dataset=deployment_reference,
                deployment_natural=deployment_natural,
                heldout_weights=heldout_weights,
                aggregate_kl_coefficient=None,
                phase_information_budget=None,
                regularized_objective=raw.regularized_objective,
                successful_starts=raw.successful_starts,
                finite_starts=raw.finite_starts,
            )
        )

        for aggregate_kl, aggregate_result in aggregates.items():
            aggregate = aggregate_result.weights[0]
            for phase_budget in phase_information_budgets:
                print(
                    f"Optimizing {target}/two-phase at KL coefficient {aggregate_kl:g}, epsilon {phase_budget:g}",
                    flush=True,
                )
                result = optimize_fixed_aggregate(
                    two_phase,
                    aggregate,
                    phase_budget,
                    alpha0,
                    alpha1,
                )
                candidate_id = (
                    f"{candidate_prefix}_{TARGET_TAGS[target]}_2p_aklc{float_tag(aggregate_kl)}_"
                    f"eps{float_tag(phase_budget)}"
                )
                records.append(
                    record_candidate(
                        candidate_id=candidate_id,
                        target=target,
                        policy_class=observatory.TWO_PHASE,
                        candidate_kind="fixed_aggregate_phase_path",
                        weights=result.weights,
                        one_phase=one_phase,
                        two_phase=two_phase,
                        natural=natural,
                        deployment_dataset=deployment_reference,
                        deployment_natural=deployment_natural,
                        heldout_weights=heldout_weights,
                        aggregate_kl_coefficient=aggregate_kl,
                        phase_information_budget=phase_budget,
                        regularized_objective=result.prediction,
                        successful_starts=result.successful_starts,
                        finite_starts=result.successful_starts,
                    )
                )

    for record in records:
        weights = np.asarray(record.pop("weights"), dtype=float)
        if weights.shape != (2, reference.m) or np.any(weights < -1e-9):
            raise ValueError(f"Invalid policy weights for {record['candidate_id']}")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError(f"Unnormalized policy weights for {record['candidate_id']}")
        if record["candidate_kind"] == "fixed_aggregate_phase_path":
            matching = next(
                candidate
                for candidate in records
                if candidate["target"] == record["target"]
                and candidate["candidate_kind"] == "one_phase_aggregate_path"
                and candidate["aggregate_kl_coefficient"] == record["aggregate_kl_coefficient"]
            )
            tied_weights = np.asarray(matching["stored_weights"], dtype=float) if "stored_weights" in matching else None
            if tied_weights is not None:
                expected = tied_weights[0]
                aggregate = alpha0 * weights[0] + alpha1 * weights[1]
                if not np.allclose(aggregate, expected, atol=1e-7):
                    raise ValueError(f"Fixed-aggregate solve drifted for {record['candidate_id']}")
        record["stored_weights"] = weights
        coordinate_kind, existing_run = existing_coordinate(
            weights,
            deployment_fit_weights,
            heldout_weights,
            heldout_frame,
            deployment_alpha0,
            deployment_alpha1,
        )
        record["existing_coordinate"] = coordinate_kind
        record["existing_run_name"] = existing_run

    # Recheck fixed aggregates after every tied policy has its in-memory weights.
    tied_aggregates = {
        (record["target"], record["aggregate_kl_coefficient"]): np.asarray(record["stored_weights"])[0]
        for record in records
        if record["candidate_kind"] == "one_phase_aggregate_path"
    }
    for record in records:
        if record["candidate_kind"] != "fixed_aggregate_phase_path":
            continue
        weights = np.asarray(record["stored_weights"])
        aggregate = alpha0 * weights[0] + alpha1 * weights[1]
        expected = tied_aggregates[(record["target"], record["aggregate_kl_coefficient"])]
        if not np.allclose(aggregate, expected, atol=1e-7):
            raise ValueError(f"Fixed-aggregate solve drifted for {record['candidate_id']}")
        if float(record["source_phase_information_kl"]) > float(record["phase_information_budget"]) + 1e-7:
            raise ValueError(f"Phase-information constraint failed for {record['candidate_id']}")

    first_by_hash: dict[str, str] = {}
    excluded_hashes: dict[str, str] = {}
    if args.exclude_source_panel is not None:
        excluded = pd.read_csv(args.exclude_source_panel)
        phase0_columns = [f"phase_0_{domain}" for domain in deployment_reference.domain_names]
        phase1_columns = [f"phase_1_{domain}" for domain in deployment_reference.domain_names]
        missing = [column for column in [*phase0_columns, *phase1_columns] if column not in excluded.columns]
        if missing:
            raise ValueError(f"Excluded source panel is missing phase columns: {missing[:5]}")
        for _index, row in excluded.iterrows():
            weights = np.stack(
                [
                    row[phase0_columns].to_numpy(dtype=float),
                    row[phase1_columns].to_numpy(dtype=float),
                ]
            )
            excluded_hashes[policy_hash(weights)] = str(row["candidate_id"])
    for record in records:
        coordinate_hash = str(record["coordinate_hash"])
        record["duplicate_coordinate"] = coordinate_hash in first_by_hash
        record["coordinate_primary_candidate"] = first_by_hash.setdefault(coordinate_hash, str(record["candidate_id"]))
        record["cross_panel_alias"] = excluded_hashes.get(coordinate_hash)
        weights = np.asarray(record.pop("stored_weights"), dtype=float)
        mixture_path = mixtures_dir / f"{record['candidate_id']}.csv"
        mixture_frame(deployment_reference, deployment_natural, weights).to_csv(mixture_path, index=False)
        record["mixture_path"] = str(mixture_path.relative_to(args.output_dir))
        for phase in (0, 1):
            for domain, weight in zip(reference.domain_names, weights[phase], strict=True):
                record[f"phase_{phase}_{domain}"] = float(weight)

    manifest = pd.DataFrame(records).sort_values(
        ["target", "policy_class", "candidate_kind", "aggregate_kl_coefficient", "phase_information_budget"],
        na_position="first",
    )
    launch = manifest.loc[
        manifest["existing_coordinate"].eq("new")
        & ~manifest["duplicate_coordinate"]
        & manifest["cross_panel_alias"].isna()
    ].reset_index(drop=True)
    if launch.empty:
        raise RuntimeError("No new coordinates remain after deduplication")

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
    manifest["fit_source"] = args.fit_source
    launch["fit_source"] = args.fit_source
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    launch[launcher_columns].to_csv(args.output_dir / "launcher_source_panel.csv", index=False)
    pd.DataFrame(config_rows).to_csv(args.output_dir / "selected_configs.csv", index=False)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(args.output_dir / "cv_optimum_sensitivity.csv", index=False)
    aliases = manifest.loc[
        manifest["duplicate_coordinate"]
        | ~manifest["existing_coordinate"].eq("new")
        | manifest["cross_panel_alias"].notna(),
        [
            "candidate_id",
            "coordinate_primary_candidate",
            "existing_coordinate",
            "existing_run_name",
            "cross_panel_alias",
            "coordinate_hash",
        ],
    ]
    aliases.to_csv(args.output_dir / "candidate_aliases.csv", index=False)
    summary = {
        "proposal_arms": len(manifest),
        "launch_ready_unique_new_coordinates": len(launch),
        "fit_source": args.fit_source,
        "deployment_scale": "delphi_3e18",
        "targets": list(TARGETS),
        "fit_rows": {policy: FIT_ROWS for policy in POLICY_CLASSES},
        "aggregate_kl_coefficients": list(aggregate_kl_coefficients),
        "phase_information_budgets": list(phase_information_budgets),
        "source_phase_fractions": [alpha0, alpha1],
        "deployment_phase_fractions": [deployment_alpha0, deployment_alpha1],
        "fit_hyperparameters_swept_in_training": False,
        "deployment_hyperparameters_require_validation": True,
        "jobs_submitted": False,
        "cross_panel_aliases": int(manifest["cross_panel_alias"].notna().sum()),
        "candidate_manifest_sha256": hashlib.sha256(
            (args.output_dir / "candidate_manifest.csv").read_bytes()
        ).hexdigest(),
        "launcher_source_panel_sha256": hashlib.sha256(
            (args.output_dir / "launcher_source_panel.csv").read_bytes()
        ).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    selected_configs = pd.DataFrame(config_rows)
    render_diagnostics(manifest, args.output_dir)
    write_report(manifest, launch, selected_configs, sensitivity, args.output_dir, args.fit_source)
    source_panel_sha256 = str(summary["launcher_source_panel_sha256"])
    manifest_sha256 = str(summary["candidate_manifest_sha256"])
    gcs_source_panel = f"{args.gcs_output_dir.rstrip('/')}/source/launcher_source_panel-{source_panel_sha256[:16]}.csv"
    gcs_candidate_manifest = f"{args.gcs_output_dir.rstrip('/')}/source/candidate_manifest-{manifest_sha256[:16]}.csv"
    summary.update(
        {
            "gcs_launcher_source_panel": gcs_source_panel,
            "gcs_candidate_manifest": gcs_candidate_manifest,
            "uploaded": bool(args.upload),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.upload:
        upload_artifact(args.output_dir / "launcher_source_panel.csv", gcs_source_panel)
        upload_artifact(args.output_dir / "candidate_manifest.csv", gcs_candidate_manifest)
        for name in (
            "candidate_aliases.csv",
            "selected_configs.csv",
            "cv_optimum_sensitivity.csv",
            "summary.json",
            "report.md",
        ):
            upload_artifact(args.output_dir / name, f"{args.gcs_output_dir.rstrip('/')}/source/{name}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
