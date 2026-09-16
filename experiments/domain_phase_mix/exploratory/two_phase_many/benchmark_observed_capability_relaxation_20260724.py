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
#   "statsmodels",
#   "tabulate",
# ]
# ///
"""Benchmark observed-capability relaxation under an exact 280-checkpoint budget.

The aggregate model is fit to macro BPB on charged phase-tied rows. Component
terminal tied-policy responses share its physical acquisition geometry and are
fit to independently measured benchmark-component BPBs. Controlled
fixed-aggregate phase pairs identify a small number of effective recency rates.

The targeted pairwise phase-order panel is sealed and is never read here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_aggregate_comparators_20260724 as comparators,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_marginal_acquisition_joint_20260724 as marginal_joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_pooled_acquisition_protocol_20260724 as strict_protocol,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_frontier_control_aggregate_identification_20260724 as aggregate_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_marginal_acquisition_phase_potential_20260724 as phase_potential,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_delphi_3e18_augmented_swarm as fit_export,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "observed_capability_relaxation_20260724"
COMPONENT_PANEL = REFERENCE_OUTPUTS / "delphi_3e18_observed_components_20260724" / "observed_component_panel.csv"
DEFAULT_SEEDS = strict_protocol.DEFAULT_SEEDS
ACTIVE_ARMS = ("all_tied", "frontier_controls_only", "phase_probe_32", "phase_probe_112")
GAMMA_GRID = np.asarray(
    (1.0, 1.5, 2.0, 3.0, 4.0, 5.25, 6.5, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0),
    dtype=float,
)
HUBER_DELTA = 1.345
COMPONENT_NOISE_FLOOR = 2e-4
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

UNCHEATABLE_COMPONENT_WEIGHTS = {
    "ao3_english": 0.172503330543,
    "arxiv_computer_science": 0.145914717247,
    "arxiv_physics": 0.167314897215,
    "bbc_news": 0.120622732206,
    "github_cpp": 0.154344986556,
    "github_python": 0.147859851769,
    "wikipedia_english": 0.091439484465,
}


@dataclass(frozen=True)
class TargetSpec:
    """Observed components and their fixed macro aggregation."""

    target: str
    columns: tuple[str, ...]
    names: tuple[str, ...]
    weights: np.ndarray
    groups: tuple[str, ...]


@dataclass(frozen=True)
class ComponentModels:
    """Terminal tied-policy response heads sharing one feature geometry."""

    models: tuple[orthogonal.AggregateModel, ...]
    names: tuple[str, ...]

    def predict_tied(self, policies: np.ndarray) -> np.ndarray:
        policies = np.asarray(policies, dtype=float)
        weights = np.stack([policies, policies], axis=1)
        return np.column_stack([model.predict(weights) for model in self.models])

    def path_terms(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = np.asarray(weights, dtype=float)
        alpha0 = self.models[0].phase_fraction
        aggregate = orthogonal.aggregate_weights(weights, alpha0)
        return (
            self.predict_tied(weights[:, 0, :]),
            self.predict_tied(weights[:, 1, :]),
            self.predict_tied(aggregate),
        )

    @property
    def active_parameter_count(self) -> int:
        return sum(
            1
            + int(np.sum(model.bucket_coef > 1e-12))
            + int(np.sum(model.family_coef > 1e-12))
            + int(model.replay_coef > 1e-12)
            for model in self.models
        )


class ComponentSurface(Protocol):
    """Prediction interface for one fit or a cross-fitted ensemble."""

    names: tuple[str, ...]

    def predict_tied(self, policies: np.ndarray) -> np.ndarray: ...

    def path_terms(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    @property
    def active_parameter_count(self) -> int: ...


@dataclass(frozen=True)
class CrossFittedComponentModels:
    """Average of fold-out component response fits."""

    folds: tuple[ComponentModels, ...]
    names: tuple[str, ...]

    def predict_tied(self, policies: np.ndarray) -> np.ndarray:
        return np.mean(
            np.stack([fold.predict_tied(policies) for fold in self.folds]),
            axis=0,
        )

    def path_terms(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        terms = [fold.path_terms(weights) for fold in self.folds]
        return tuple(np.mean(np.stack([term[index] for term in terms]), axis=0) for index in range(3))

    @property
    def active_parameter_count(self) -> int:
        return round(np.mean([fold.active_parameter_count for fold in self.folds]))


@dataclass(frozen=True)
class RateFit:
    """Frozen capability relaxation rates."""

    variant: str
    component_rates: np.ndarray
    group_rates: dict[str, float]
    group_gammas: dict[str, float]
    objective: float


@dataclass(frozen=True)
class PairCapabilityTerms:
    """Precomputed terminal tied-policy responses for antithetic pairs."""

    plus_phase0: np.ndarray
    plus_phase1: np.ndarray
    plus_aggregate: np.ndarray
    minus_phase0: np.ndarray
    minus_phase1: np.ndarray
    minus_aggregate: np.ndarray

    def corrections(self, retained: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        plus = retained[None, :] * self.plus_phase0 + (1.0 - retained[None, :]) * self.plus_phase1 - self.plus_aggregate
        minus = (
            retained[None, :] * self.minus_phase0 + (1.0 - retained[None, :]) * self.minus_phase1 - self.minus_aggregate
        )
        return plus, minus

    def odd_prediction(self, retained: np.ndarray) -> np.ndarray:
        plus, minus = self.corrections(retained)
        return 0.5 * (plus - minus)

    def even_prediction(self, retained: np.ndarray) -> np.ndarray:
        plus, minus = self.corrections(retained)
        return 0.5 * (plus + minus)


@dataclass(frozen=True)
class JointModel:
    """Macro aggregate predictor plus finite-time capability correction."""

    aggregate: orthogonal.AggregateModel
    components: ComponentSurface
    spec: TargetSpec
    rates: RateFit

    def predict(self, weights: np.ndarray) -> np.ndarray:
        phase0, phase1, aggregate_component = self.components.path_terms(weights)
        retained = retained_phase0_fraction(
            self.rates.component_rates,
            self.aggregate.phase_fraction,
        )
        correction = retained[None, :] * phase0 + (1.0 - retained[None, :]) * phase1 - aggregate_component
        return self.aggregate.predict(weights) + correction @ self.spec.weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    return parser.parse_args()


def table9_group(name: str) -> str:
    math_tasks = ("minerva_math_", "basic_skills_arithmetic")
    code_tasks = (
        "codex_humaneval",
        "mbpp",
        "mt_mbpp_",
        "basic_skills_coding",
        "basic_skills_string_operations",
    )
    reading_tasks = (
        "hellaswag",
        "winogrande",
        "socialiqa",
        "piqa",
        "coqa",
        "drop",
        "squad",
        "lambada",
    )
    if name.startswith(math_tasks):
        return "mathematics"
    if name.startswith(code_tasks):
        return "code"
    if name.startswith(reading_tasks):
        return "reading_qa"
    return "knowledge_reasoning"


def target_spec(target: str, frame: pd.DataFrame) -> TargetSpec:
    if target == "uncheatable":
        columns = tuple(f"eval/uncheatable_eval/{name}/bpb" for name in UNCHEATABLE_COMPONENT_WEIGHTS)
        names = tuple(UNCHEATABLE_COMPONENT_WEIGHTS)
        raw_weights = np.asarray(
            [UNCHEATABLE_COMPONENT_WEIGHTS[name] for name in names],
            dtype=float,
        )
        weights = raw_weights / raw_weights.sum()
        groups = tuple(
            (
                "narrative_reference"
                if name in {"ao3_english", "bbc_news", "wikipedia_english"}
                else "science" if name in {"arxiv_computer_science", "arxiv_physics"} else "code"
            )
            for name in names
        )
    elif target == "table9":
        columns = tuple(fit_export.table9_component_fields(frame.columns))
        names = tuple(column.removeprefix("olmo_base_eval/easy_bpb/").removesuffix("/bpb") for column in columns)
        weights = np.full(len(columns), 1.0 / len(columns), dtype=float)
        groups = tuple(table9_group(name) for name in names)
    else:
        raise ValueError(f"Unknown target {target!r}")
    if not columns or any(column not in frame for column in columns):
        raise ValueError(f"Component panel is incomplete for {target}")
    return TargetSpec(target, columns, names, weights, groups)


def retained_phase0_fraction(rate: np.ndarray, alpha0: float) -> np.ndarray:
    rate = np.asarray(rate, dtype=float)
    alpha1 = 1.0 - alpha0
    result = np.full(rate.shape, alpha0, dtype=float)
    positive = rate > 1e-10
    value = rate[positive]
    result[positive] = np.exp(-value * alpha1) * (-np.expm1(-value * alpha0)) / (-np.expm1(-value))
    return result


def retained_phase0_fraction_from_gamma(gamma: np.ndarray, alpha0: float) -> np.ndarray:
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma < 1.0):
        raise ValueError("Effective recency gamma must be at least one")
    alpha1 = 1.0 - alpha0
    return alpha0 / (alpha0 + gamma * alpha1)


def gamma_to_rate(gamma: float, alpha0: float) -> float:
    """Invert the monotone reduced-form recency map."""
    if gamma < 1.0:
        raise ValueError("Effective recency gamma must be at least one")
    if gamma == 1.0:
        return 0.0
    target = float(retained_phase0_fraction_from_gamma(np.asarray([gamma]), alpha0)[0])
    lower = 0.0
    upper = 1.0
    while float(retained_phase0_fraction(np.asarray([upper]), alpha0)[0]) > target:
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        retained = float(retained_phase0_fraction(np.asarray([midpoint]), alpha0)[0])
        if retained > target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def panel_weights(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    phase0 = np.column_stack([frame[f"phase_0_{domain}"].to_numpy(dtype=float) for domain in domains])
    phase1 = np.column_stack([frame[f"phase_1_{domain}"].to_numpy(dtype=float) for domain in domains])
    weights = np.stack([phase0, phase1], axis=1)
    if not np.allclose(weights.sum(axis=2), 1.0, atol=1e-10):
        raise ValueError("Component-panel policies are not normalized")
    return weights


def weights_for_row_names(
    frame: pd.DataFrame,
    names: list[str],
    domains: list[str],
) -> np.ndarray:
    if frame["row_name"].duplicated().any():
        raise ValueError("Component panel contains duplicate row names")
    position = {str(name): index for index, name in enumerate(frame["row_name"])}
    missing = sorted(set(names) - set(position))
    if missing:
        raise ValueError(f"Missing component-panel weights for {missing[:5]}")
    weights = panel_weights(frame, domains)
    return weights[np.asarray([position[name] for name in names], dtype=int)]


def row_names(frame: pd.DataFrame) -> list[str]:
    result = []
    for _, row in frame.iterrows():
        candidate = row.get("candidate_id")
        run = row.get("run_name")
        if pd.notna(candidate):
            result.append(str(candidate))
        elif pd.notna(run):
            result.append(str(run))
        else:
            raise ValueError("Training row lacks candidate_id and run_name")
    return result


def component_training_dataset(
    training: pooled.Dataset,
    component_panel: pd.DataFrame,
    component_column: str,
) -> pooled.Dataset:
    lookup = component_panel.set_index("row_name").copy()[component_column]
    names = row_names(training.frame)
    missing = sorted(set(names) - set(lookup.index))
    if missing:
        raise ValueError(f"Missing component outcomes for {missing[:5]}")
    return pooled.Dataset(
        name=f"{training.name}_{component_column}",
        frame=training.frame.copy(),
        y=lookup.loc[names].to_numpy(dtype=float),
        weights=np.asarray(training.weights, dtype=float),
        c0=np.asarray(training.c0, dtype=float),
        c1=np.asarray(training.c1, dtype=float),
        domain_names=list(training.domain_names),
    )


def fit_component_models_on_indices(
    training: pooled.Dataset,
    component_panel: pd.DataFrame,
    spec: TargetSpec,
    indices: np.ndarray,
    l2: float,
) -> ComponentModels:
    families = orthogonal.family_partition(training.domain_names)
    models = []
    for column in spec.columns:
        component = component_training_dataset(training, component_panel, column)
        models.append(
            orthogonal.fit_aggregate(
                component,
                indices,
                aggregate_audit.FROZEN_POOLED_CONFIG,
                aggregate_audit.FROZEN_POOLED_SHAPE,
                l2,
                families,
            )
        )
    return ComponentModels(tuple(models), spec.names)


def fit_cross_fitted_component_models(
    training: pooled.Dataset,
    component_panel: pd.DataFrame,
    spec: TargetSpec,
    fold: np.ndarray,
    l2: float,
) -> tuple[CrossFittedComponentModels, np.ndarray]:
    all_indices = np.arange(training.n)
    component_observed = np.column_stack(
        [component_training_dataset(training, component_panel, column).y for column in spec.columns]
    )
    oof_prediction = np.full(component_observed.shape, np.nan, dtype=float)
    fold_models = []
    for fold_index in sorted(np.unique(fold)):
        test = np.flatnonzero(fold == fold_index)
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        models = fit_component_models_on_indices(
            training,
            component_panel,
            spec,
            train,
            l2,
        )
        fold_models.append(models)
        oof_prediction[test] = np.column_stack([model.predict(training.weights[test]) for model in models.models])
    if not np.isfinite(oof_prediction).all():
        raise ValueError("Component cross-fitting produced non-finite predictions")
    return (
        CrossFittedComponentModels(tuple(fold_models), spec.names),
        oof_prediction,
    )


def subset_pair_dataset(
    dataset: phase_potential.PairDataset,
    mask: np.ndarray,
) -> phase_potential.PairDataset:
    indices = np.flatnonzero(mask)
    return phase_potential.PairDataset(
        target=dataset.target,
        frame=dataset.frame.iloc[indices].reset_index(drop=True),
        aggregate=dataset.aggregate[indices],
        contrast=dataset.contrast[indices],
        odd=dataset.odd[indices],
        even=dataset.even[indices],
        noise=dataset.noise[indices],
        domain_names=dataset.domain_names,
    )


def component_pair_observations(
    pairs: phase_potential.PairDataset,
    component_panel: pd.DataFrame,
    spec: TargetSpec,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = component_panel.set_index("row_name").copy()
    plus = lookup.loc[pairs.frame["plus_candidate_id"], list(spec.columns)].to_numpy(dtype=float)
    minus = lookup.loc[pairs.frame["minus_candidate_id"], list(spec.columns)].to_numpy(dtype=float)
    controls = component_panel[
        component_panel["panel"].eq("frontier_phase_fiber") & component_panel["contrast_family"].eq("center_control")
    ].copy()
    controls["seed_block"] = controls["seed_block"].astype(int)
    control_lookup = controls.set_index(["anchor_id", "seed_block"])
    control_keys = list(
        zip(
            pairs.frame["anchor_id"].astype(str),
            pairs.frame["seed_block"].astype(int),
            strict=True,
        )
    )
    center = control_lookup.loc[control_keys, list(spec.columns)].to_numpy(dtype=float)
    return 0.5 * (plus - minus), 0.5 * (plus + minus) - center


def component_noise(
    component_panel: pd.DataFrame,
    spec: TargetSpec,
) -> np.ndarray:
    controls = component_panel[
        component_panel["panel"].eq("frontier_phase_fiber") & component_panel["contrast_family"].eq("center_control")
    ]
    per_anchor = controls.groupby("anchor_id", sort=True)[list(spec.columns)].std(ddof=1)
    noise = np.nanmedian(per_anchor.to_numpy(dtype=float), axis=0) / np.sqrt(2.0)
    return np.maximum(noise, COMPONENT_NOISE_FLOOR)


def huber_loss(residual: np.ndarray) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(
        absolute <= HUBER_DELTA,
        0.5 * residual**2,
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    )


def pair_capability_terms(
    pairs: phase_potential.PairDataset,
    components: ComponentSurface,
    component_panel: pd.DataFrame,
) -> PairCapabilityTerms:
    plus_ids = pairs.frame["plus_candidate_id"].astype(str).tolist()
    minus_ids = pairs.frame["minus_candidate_id"].astype(str).tolist()
    domains = list(pairs.domain_names)
    plus = weights_for_row_names(component_panel, plus_ids, domains)
    minus = weights_for_row_names(component_panel, minus_ids, domains)
    plus_phase0, plus_phase1, plus_aggregate = components.path_terms(plus)
    minus_phase0, minus_phase1, minus_aggregate = components.path_terms(minus)
    return PairCapabilityTerms(
        plus_phase0,
        plus_phase1,
        plus_aggregate,
        minus_phase0,
        minus_phase1,
        minus_aggregate,
    )


def rate_labels(spec: TargetSpec, variant: str) -> tuple[str, ...]:
    if variant == "shared_rate":
        return tuple("shared" for _ in spec.columns)
    if variant == "grouped_rate":
        return spec.groups
    if variant == "per_component_rate":
        return spec.names
    if variant == "zero_rate":
        return tuple("zero" for _ in spec.columns)
    raise ValueError(f"Unknown rate variant {variant!r}")


def fit_rates(
    selected: np.ndarray,
    terms: PairCapabilityTerms,
    component_observed_odd: np.ndarray,
    noise: np.ndarray,
    spec: TargetSpec,
    variant: str,
    alpha0: float,
) -> RateFit:
    labels = rate_labels(spec, variant)
    if variant == "zero_rate":
        return RateFit(
            variant,
            np.zeros(len(spec.columns)),
            {"zero": 0.0},
            {"zero": 1.0},
            float("nan"),
        )
    group_gammas: dict[str, float] = {}
    for label in dict.fromkeys(labels):
        component_indices = np.flatnonzero(np.asarray(labels, dtype=object) == label)
        component_weight = spec.weights[component_indices]
        component_weight = component_weight / component_weight.sum()
        best_gamma = None
        best_loss = math.inf
        for gamma in GAMMA_GRID:
            gammas = np.ones(len(spec.columns), dtype=float)
            gammas[component_indices] = gamma
            retained = retained_phase0_fraction_from_gamma(
                gammas,
                alpha0,
            )
            predicted = terms.odd_prediction(retained)
            standardized = (
                predicted[selected][:, component_indices] - component_observed_odd[selected][:, component_indices]
            ) / noise[component_indices][None, :]
            loss = float(np.mean(huber_loss(standardized) @ component_weight))
            if loss < best_loss:
                best_loss = loss
                best_gamma = float(gamma)
        if best_gamma is None:
            raise AssertionError("Recency grid is empty")
        group_gammas[str(label)] = best_gamma
    component_gammas = np.asarray([group_gammas[label] for label in labels], dtype=float)
    retained = retained_phase0_fraction_from_gamma(component_gammas, alpha0)
    standardized = (terms.odd_prediction(retained)[selected] - component_observed_odd[selected]) / noise[None, :]
    objective = float(np.mean(huber_loss(standardized) @ spec.weights))
    group_rates = {label: gamma_to_rate(gamma, alpha0) for label, gamma in group_gammas.items()}
    component_rates = np.asarray([group_rates[label] for label in labels], dtype=float)
    return RateFit(
        variant,
        component_rates,
        group_rates,
        group_gammas,
        objective,
    )


def cross_fitted_rate_predictions(
    pairs: phase_potential.PairDataset,
    selected: np.ndarray,
    terms: PairCapabilityTerms,
    component_observed_odd: np.ndarray,
    noise: np.ndarray,
    spec: TargetSpec,
    variant: str,
    alpha0: float,
) -> tuple[np.ndarray, list[RateFit]]:
    selected_groups = pairs.frame.iloc[selected]["direction_group"].to_numpy(dtype=object)
    splits = phase_potential.local_grouped_splits(selected_groups)
    prediction = np.full(
        (len(selected), len(spec.columns)),
        np.nan,
        dtype=float,
    )
    fold_fits = []
    for train, test in splits:
        fitted = fit_rates(
            selected[train],
            terms,
            component_observed_odd,
            noise,
            spec,
            variant,
            alpha0,
        )
        fold_fits.append(fitted)
        retained = retained_phase0_fraction(fitted.component_rates, alpha0)
        prediction[test] = terms.odd_prediction(retained)[selected[test]]
    if not np.isfinite(prediction).all():
        raise ValueError("Rate cross-fitting produced non-finite predictions")
    return prediction, fold_fits


def macro_collapsed_components(
    aggregate: orthogonal.AggregateModel,
    spec: TargetSpec,
) -> ComponentModels:
    return ComponentModels(
        tuple(aggregate for _ in spec.columns),
        spec.names,
    )


def metric_rows(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            **metadata,
            "scope": scope,
            **orthogonal.regression_metrics(observed[mask], predicted[mask]),
        }
        for scope, mask in aggregate_audit.scope_masks(frame).items()
    ]


def component_pair_metric_row(
    observed: np.ndarray,
    predicted: np.ndarray,
    noise: np.ndarray,
    spec: TargetSpec,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    residual = predicted - observed
    macro_observed = observed @ spec.weights
    macro_predicted = predicted @ spec.weights
    macro_metrics = orthogonal.regression_metrics(macro_observed, macro_predicted)
    resolved = np.abs(observed) >= noise[None, :]
    return {
        **metadata,
        "component_weighted_rmse": float(np.sqrt(np.mean((residual**2) @ spec.weights))),
        "component_standardized_huber": float(np.mean(huber_loss(residual / noise[None, :]) @ spec.weights)),
        "component_resolved_sign_accuracy": (
            float(np.mean(np.sign(predicted[resolved]) == np.sign(observed[resolved])))
            if np.any(resolved)
            else float("nan")
        ),
        **{f"macro_{key}": value for key, value in macro_metrics.items()},
    }


def baseline_predictions(target: str, positions: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for model_id in marginal_joint.OBSERVATORY_BASELINES:
        path = orthogonal.OBSERVATORY_CACHE / target / "two_phase" / f"{model_id}.json"
        if not path.exists():
            continue
        result[f"observatory_{model_id}"] = np.asarray(
            json.loads(path.read_text())["prediction"],
            dtype=float,
        )[positions]
    return result


def run_target_seed(
    target: str,
    arm: strict_protocol.BudgetArm,
    seed: int,
    component_panel: pd.DataFrame,
    pair_dataset: phase_potential.PairDataset,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[pd.DataFrame],
]:
    (
        _reference,
        _heldout_frame,
        _heldout_weights,
        single,
        controls,
        evaluation_frame,
        evaluation_weights,
        observed,
        clusters,
    ) = comparators.target_data(target)
    spec = target_spec(target, component_panel)
    training = strict_protocol.aggregate_training_dataset(
        target,
        single,
        controls,
        arm,
        seed,
    )
    fold = strict_protocol.grouped_stratified_folds(training, seed)
    aggregate_fit = aggregate_audit.frozen_pooled_fit(training, fold)
    aggregate = aggregate_fit.model
    components, component_oof_prediction = fit_cross_fitted_component_models(
        training,
        component_panel,
        spec,
        fold,
        aggregate_audit.FROZEN_POOLED_L2,
    )
    component_observed_training = np.column_stack(
        [component_training_dataset(training, component_panel, column).y for column in spec.columns]
    )
    component_oof_rmse = float(
        np.sqrt(np.mean((component_oof_prediction - component_observed_training) ** 2 @ spec.weights))
    )
    terms = pair_capability_terms(pair_dataset, components, component_panel)
    selected = marginal_joint.selected_pairs(
        pair_dataset,
        aggregate,
        arm.treatment_count,
        seed,
    )
    observed_component_odd, observed_component_even = component_pair_observations(
        pair_dataset,
        component_panel,
        spec,
    )
    noise = component_noise(component_panel, spec)

    rate_fits = {
        variant: fit_rates(
            selected,
            terms,
            observed_component_odd,
            noise,
            spec,
            variant,
            aggregate.phase_fraction,
        )
        for variant in (
            "zero_rate",
            "shared_rate",
            "grouped_rate",
            "per_component_rate",
        )
        if arm.treatment_count > 0 or variant == "zero_rate"
    }
    if arm.treatment_count > 0:
        collapsed = macro_collapsed_components(aggregate, spec)
        collapsed_terms = pair_capability_terms(
            pair_dataset,
            collapsed,
            component_panel,
        )
        rate_fits["macro_collapsed_grouped_rate"] = fit_rates(
            selected,
            collapsed_terms,
            observed_component_odd,
            noise,
            spec,
            "grouped_rate",
            aggregate.phase_fraction,
        )

    positions = evaluation_frame["position"].to_numpy(dtype=int)
    base_frame = evaluation_frame.copy()
    base_frame["target"] = target
    base_frame["cluster"] = clusters
    base_frame["observed"] = observed
    base_frame["evaluation_row"] = np.arange(len(base_frame))
    metrics = []
    pair_metrics = []
    component_pair_metrics = []
    rate_rows = []
    predictions = []
    metadata_base = {
        "target": target,
        "arm": arm.name,
        "seed": seed,
        **asdict(arm),
        "aggregate_fit_rows": training.n,
        "aggregate_oof_rmse": orthogonal.regression_metrics(
            training.y,
            aggregate_fit.oof_prediction,
        )["rmse"],
        "component_oof_rmse": component_oof_rmse,
        "component_active_parameter_count": components.active_parameter_count,
    }

    models: dict[str, Any] = {"physical_pooled_acquisition": aggregate}
    for variant, rates in rate_fits.items():
        model_components = (
            macro_collapsed_components(aggregate, spec) if variant == "macro_collapsed_grouped_rate" else components
        )
        variant_terms = collapsed_terms if variant == "macro_collapsed_grouped_rate" else terms
        models[f"observed_capability_relaxation_{variant}"] = JointModel(
            aggregate,
            model_components,
            spec,
            rates,
        )
        retained = retained_phase0_fraction(
            rates.component_rates,
            aggregate.phase_fraction,
        )
        predicted_component_odd = variant_terms.odd_prediction(retained)
        predicted_component_even = variant_terms.even_prediction(retained)
        selected_mask = np.isin(np.arange(pair_dataset.n), selected)
        for scope, mask in {
            "selected_fit": selected_mask,
            "unselected_off_budget_oracle": ~selected_mask,
        }.items():
            if not np.any(mask):
                continue
            for parity, component_observed, component_predicted in (
                ("odd", observed_component_odd, predicted_component_odd),
                ("even", observed_component_even, predicted_component_even),
            ):
                component_pair_metrics.append(
                    component_pair_metric_row(
                        component_observed[mask],
                        component_predicted[mask],
                        noise,
                        spec,
                        {
                            **metadata_base,
                            "model": variant,
                            "scope": scope,
                            "parity": parity,
                        },
                    )
                )
        if arm.treatment_count > 0:
            rate_variant = "grouped_rate" if variant == "macro_collapsed_grouped_rate" else variant
            rate_oof_prediction, _fold_rates = cross_fitted_rate_predictions(
                pair_dataset,
                selected,
                variant_terms,
                observed_component_odd,
                noise,
                spec,
                rate_variant,
                aggregate.phase_fraction,
            )
            component_pair_metrics.append(
                component_pair_metric_row(
                    observed_component_odd[selected],
                    rate_oof_prediction,
                    noise,
                    spec,
                    {
                        **metadata_base,
                        "model": variant,
                        "scope": "selected_rate_oof",
                        "parity": "odd",
                    },
                )
            )
        for label, rate in rates.group_rates.items():
            rate_rows.append(
                {
                    **metadata_base,
                    "model": variant,
                    "rate_group": label,
                    "rate": rate,
                    "gamma": rates.group_gammas[label],
                    "retained_phase0_fraction": float(
                        retained_phase0_fraction(
                            np.asarray([rate]),
                            aggregate.phase_fraction,
                        )[0]
                    ),
                    "fit_objective": rates.objective,
                    "selected_pair_count": len(selected),
                }
            )

    for model_name, model in models.items():
        prediction = model.predict(evaluation_weights)
        metadata = {**metadata_base, "model": model_name}
        metrics.extend(metric_rows(base_frame, observed, prediction, metadata))
        local = base_frame.copy()
        for name, value in metadata.items():
            local[name] = value
        local["predicted"] = prediction
        local["residual"] = prediction - observed
        predictions.append(local)

        plus_prediction = model.predict(
            weights_for_row_names(
                component_panel,
                pair_dataset.frame["plus_candidate_id"].astype(str).tolist(),
                single.domain_names,
            )
        )
        minus_prediction = model.predict(
            weights_for_row_names(
                component_panel,
                pair_dataset.frame["minus_candidate_id"].astype(str).tolist(),
                single.domain_names,
            )
        )
        predicted_odd = 0.5 * (plus_prediction - minus_prediction)
        for scope, mask in {
            "selected_pairs": np.isin(np.arange(pair_dataset.n), selected),
            "unselected_pairs": ~np.isin(np.arange(pair_dataset.n), selected),
        }.items():
            if not np.any(mask):
                continue
            pair_metrics.append(
                {
                    **metadata,
                    "scope": scope,
                    **orthogonal.regression_metrics(
                        pair_dataset.odd[mask],
                        predicted_odd[mask],
                    ),
                }
            )

    if arm.name == "all_tied":
        for model_name, prediction in baseline_predictions(target, positions).items():
            metadata = {
                **metadata_base,
                "arm": "observatory",
                "seed": -1,
                "model": model_name,
            }
            metrics.extend(metric_rows(base_frame, observed, prediction, metadata))
    return metrics, pair_metrics, component_pair_metrics, rate_rows, predictions


def objective_reconstruction_rows(
    component_panel: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for target, macro_column in (
        ("uncheatable", "uncheatable_bpb"),
        ("table9", "table9_macro_bpb"),
    ):
        spec = target_spec(target, component_panel)
        reconstructed = component_panel[list(spec.columns)].to_numpy(dtype=float) @ spec.weights
        observed = component_panel[macro_column].to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "n": len(observed),
                "rmse": float(np.sqrt(np.mean((reconstructed - observed) ** 2))),
                "max_abs_error": float(np.max(np.abs(reconstructed - observed))),
                "intercept": float(np.mean(observed - reconstructed)),
            }
        )
    return rows


def write_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    selected = metrics[
        metrics["scope"].eq("append_only_without_compact_optimum")
        & metrics["arm"].isin(("observatory", "phase_probe_32", "phase_probe_112"))
    ].copy()
    figure = px.scatter(
        selected,
        x="rmse",
        y="regret_at_1",
        color="model",
        symbol="arm",
        facet_col="target",
        hover_data=["seed", "calibration_slope", "optimism_gt_0p05", "worst_optimism"],
        title="Observed-capability relaxation: heldout RMSE versus Regret@1",
    )
    figure.write_html(
        output_dir / "heldout_metric_frontier.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    calibration = predictions[
        predictions["arm"].eq("phase_probe_112")
        & predictions["model"].isin(
            (
                "physical_pooled_acquisition",
                "observed_capability_relaxation_shared_rate",
                "observed_capability_relaxation_grouped_rate",
            )
        )
    ].copy()
    figure = px.scatter(
        calibration,
        x="predicted",
        y="observed",
        color="model",
        symbol="seed",
        facet_col="target",
        trendline="ols",
        hover_data=["candidate_id", "source"],
        title="Observed-capability relaxation: historical 3e18 calibration",
    )
    figure.write_html(
        output_dir / "heldout_calibration.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_report(
    reconstruction: pd.DataFrame,
    metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    component_pair_metrics: pd.DataFrame,
    rates: pd.DataFrame,
    output_dir: Path,
) -> None:
    scope = "append_only_without_compact_optimum"
    selected = metrics[
        metrics["scope"].eq(scope) & metrics["arm"].isin(("observatory", "phase_probe_32", "phase_probe_112"))
    ].copy()
    columns = [
        "target",
        "arm",
        "seed",
        "model",
        "n",
        "rmse",
        "spearman",
        "regret_at_1",
        "calibration_slope",
        "bias",
        "optimism_gt_0p05",
        "worst_optimism",
    ]
    pair_selected = pair_metrics[pair_metrics["scope"].eq("unselected_pairs")].copy()
    component_selected = component_pair_metrics[component_pair_metrics["arm"].eq("phase_probe_112")].copy()
    rate_selected = rates[rates["arm"].isin(("phase_probe_32", "phase_probe_112"))].copy()
    pair_table = (
        pair_selected[
            [
                "target",
                "arm",
                "seed",
                "model",
                "n",
                "rmse",
                "spearman",
                "regret_at_1",
                "calibration_slope",
            ]
        ]
        .sort_values(["target", "rmse"])
        .to_markdown(index=False)
    )
    component_table = (
        component_selected[
            [
                "target",
                "seed",
                "model",
                "scope",
                "parity",
                "component_weighted_rmse",
                "component_standardized_huber",
                "macro_rmse",
                "macro_bias",
            ]
        ]
        .sort_values(["target", "scope", "parity", "component_weighted_rmse"])
        .to_markdown(index=False)
    )
    rate_table = (
        rate_selected[
            [
                "target",
                "arm",
                "seed",
                "model",
                "rate_group",
                "gamma",
                "rate",
                "retained_phase0_fraction",
                "selected_pair_count",
            ]
        ]
        .sort_values(["target", "arm", "seed", "model", "rate_group"])
        .to_markdown(index=False)
    )
    report = f"""# Observed-capability relaxation benchmark

This benchmark was frozen before historical-heldout evaluation. It does not
read the sealed targeted pairwise phase-order panel.

## Objective reconstruction

{reconstruction.to_markdown(index=False)}

## Historical 3e18 heldouts

Scope: `{scope}`.

{selected[columns].sort_values(["target", "rmse"]).to_markdown(index=False)}

## Unselected controlled phase pairs

{pair_table}

## Component odd and even response

Unselected rows are off-budget oracle diagnostics and do not gate promotion.
The even response was not used to select recency and is an orthogonal
falsification target.

{component_table}

## Relaxation rates

{rate_table}

## Interpretation rule

The grouped-rate candidate is promoted only if it improves phase-order
prediction over the shared-rate ablation, improves a primary historical
heldout diagnostic without violating the frozen gates, and has non-boundary,
stable rates. A lower component-fit loss alone is not sufficient.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds:
        raise ValueError("At least one seed is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    component_panel = pd.read_csv(COMPONENT_PANEL)
    expected_counts = {
        "two_phase_fit": 280,
        "one_phase_fit": 280,
        "frontier_phase_fiber": 200,
    }
    if component_panel["panel"].value_counts().to_dict() != expected_counts:
        raise ValueError("Observed component panel has unexpected composition")

    reconstruction = pd.DataFrame(objective_reconstruction_rows(component_panel))
    if float(reconstruction["max_abs_error"].max()) > 1e-6:
        raise ValueError("Fixed component aggregation does not reconstruct a headline target")

    pair_datasets = {}
    for target, dataset in phase_potential.pair_datasets().items():
        pair_datasets[target] = subset_pair_dataset(
            dataset,
            dataset.frame["panel"].eq("frontier_phase_fiber").to_numpy(),
        )
        if pair_datasets[target].n != 96:
            raise ValueError(f"Expected 96 fiber pairs for {target}")

    metrics = []
    pair_metrics = []
    component_pair_metrics = []
    rates = []
    predictions = []
    arms = tuple(arm for arm in strict_protocol.ARMS if arm.name in ACTIVE_ARMS)
    for target in orthogonal.TARGETS:
        for arm in arms:
            arm_seeds = (seeds[0],) if arm.name == "all_tied" else seeds
            for seed in arm_seeds:
                (
                    local_metrics,
                    local_pairs,
                    local_component_pairs,
                    local_rates,
                    local_predictions,
                ) = run_target_seed(
                    target,
                    arm,
                    seed,
                    component_panel,
                    pair_datasets[target],
                )
                metrics.extend(local_metrics)
                pair_metrics.extend(local_pairs)
                component_pair_metrics.extend(local_component_pairs)
                rates.extend(local_rates)
                predictions.extend(local_predictions)

    metric_frame = pd.DataFrame(metrics)
    pair_frame = pd.DataFrame(pair_metrics)
    component_pair_frame = pd.DataFrame(component_pair_metrics)
    rate_frame = pd.DataFrame(rates)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    reconstruction.to_csv(args.output_dir / "objective_reconstruction.csv", index=False)
    metric_frame.to_csv(args.output_dir / "joint_metrics.csv", index=False)
    pair_frame.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    component_pair_frame.to_csv(
        args.output_dir / "component_pair_metrics.csv",
        index=False,
    )
    rate_frame.to_csv(args.output_dir / "rate_estimates.csv", index=False)
    prediction_frame.to_parquet(args.output_dir / "predictions.parquet", index=False)
    write_plots(metric_frame, prediction_frame, args.output_dir)
    write_report(
        reconstruction,
        metric_frame,
        pair_frame,
        component_pair_frame,
        rate_frame,
        args.output_dir,
    )
    provenance = {
        "component_panel": str(COMPONENT_PANEL),
        "gamma_grid": GAMMA_GRID.tolist(),
        "active_arms": [asdict(arm) for arm in arms],
        "seeds": seeds,
        "sealed_panel_read": False,
    }
    (args.output_dir / "benchmark_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
