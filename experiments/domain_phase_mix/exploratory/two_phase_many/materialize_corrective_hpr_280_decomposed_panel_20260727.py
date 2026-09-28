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
"""Materialize and locally falsify an exact-280 corrective HPR panel.

The procedure deliberately separates model form from deployment constraints:

1. Fit the identifiable, duplicate-ledger-free Hierarchical Phase Replay form
   independently on the canonical 280-row one-phase and two-phase 300M panels.
   The one-phase restriction fixes phase-only parameters to their null values;
   the two-phase fit retains the original HPR shape library.
2. Optimize the independently fitted one-phase restriction under hard
   aggregate KL budgets to proportional.
3. Freeze each aggregate and optimize only phase order with the two-phase fit
   under a conditional phase-information budget.

The epsilon-zero rows are genuine independently fitted tied optima, not
algebraic projections of a two-phase optimum. Every nonzero phase candidate is
paired with exactly one such aggregate-matched tied control.

This script never uploads artifacts and never submits training jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    corrected_hpr_model_20260727 as corrected,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    evaluate_hpr_v2_nested_20260727 as hpr_v2,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_hierarchical_phase_replay_validation_panel_3e18 as prior_hpr_panel,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    swarm39_harness_20260725 as swarm39,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "corrective_hpr_280_decomposed_panel_20260727"
JULY_PANEL = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_model_family_panel_20260712"
TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
TARGET_TAGS = {"uncheatable": "unch", "table9": "t9"}
POLICY_CLASSES = (observatory.SINGLE_PHASE, observatory.TWO_PHASE)
AGGREGATE_KL_BUDGETS = (0.25, 0.50, 0.75)
PHASE_INFORMATION_BUDGETS = (0.0, 0.001, 0.0025, 0.005, 0.01)
FIT_ROWS = 280
OPTIMIZER_STARTS = 24
FOLD_OPTIMIZER_STARTS = 10
BOOTSTRAP_DRAWS = 100
FEASIBILITY_TOLERANCE = 1e-7
EXACT_POLICY_TV = 1e-9
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
CORRECTIONS = corrected.Corrections(
    identifiable_hierarchy=True,
    deduplicated_ledgers=True,
)


@dataclass(frozen=True)
class FittedPolicy:
    """A target- and policy-class-specific corrected HPR fit."""

    target: str
    policy_class: str
    dataset: Any
    structured: family_grp.Dataset
    config: hierarchical.Config
    model: corrected.CorrectedModel
    oof_prediction: np.ndarray


@dataclass(frozen=True)
class AggregateResult:
    """One constrained tied optimum."""

    weights: np.ndarray
    prediction: float
    aggregate_kl: float
    successful_starts: int
    finite_starts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--optimizer-starts", type=int, default=OPTIMIZER_STARTS)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--aggregate-kl-budgets",
        default=",".join(str(value) for value in AGGREGATE_KL_BUDGETS),
    )
    parser.add_argument(
        "--phase-information-budgets",
        default=",".join(str(value) for value in PHASE_INFORMATION_BUDGETS),
    )
    parser.add_argument(
        "--skip-fold-paths",
        action="store_true",
        help="Skip fixed-config fold refits and path re-optimization.",
    )
    return parser.parse_args()


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one numeric sweep value")
    return values


def float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def dataset_id(target: str) -> hierarchical.DatasetId:
    if target == "uncheatable":
        return hierarchical.DatasetId.THREE_HUNDRED_M_UNCHEATABLE
    if target == "table9":
        return hierarchical.DatasetId.THREE_HUNDRED_M_TABLE9
    raise ValueError(f"Unknown target {target!r}")


def shape_library(policy_class: str) -> tuple[family_grp.Shape, ...]:
    """Freeze the audited HPR shape library before any heldout evaluation."""
    candidates = list(hpr_v2.promoted_shapes())
    if policy_class == observatory.SINGLE_PHASE:
        candidates = [replace(shape, late_multiplier=1.0, forgetting_rate=0.0) for shape in candidates]
    return tuple(dict.fromkeys(candidates))


def fit_policy(dataset: Any, target: str, policy_class: str) -> FittedPolicy:
    if dataset.n != FIT_ROWS:
        raise ValueError(f"Expected {FIT_ROWS} rows for {target}/{policy_class}, found {dataset.n}")
    if policy_class == observatory.SINGLE_PHASE and not np.allclose(
        dataset.weights[:, 0],
        dataset.weights[:, 1],
        atol=1e-10,
    ):
        raise ValueError(f"{target} one-phase fit contains untied policies")

    structured = hierarchical.family_dataset(dataset)
    shapes = shape_library(policy_class)
    rows = np.arange(dataset.n)
    config = hpr_v2.two_stage_selection(
        structured,
        dataset_id(target),
        CORRECTIONS,
        shapes,
        rows,
    )
    splits = hierarchical.split_indices(
        structured,
        dataset_id(target),
        rows,
        hierarchical.SCREEN_SEED,
    )
    oof = corrected.corrected_oof_prediction(
        structured,
        config,
        CORRECTIONS,
        splits,
    )
    model = corrected.fit_corrected(structured, config, CORRECTIONS, rows)
    return FittedPolicy(
        target=target,
        policy_class=policy_class,
        dataset=dataset,
        structured=structured,
        config=config,
        model=model,
        oof_prediction=oof,
    )


def scalar_prediction(model: Any, weights: np.ndarray) -> float:
    return float(model.predict(np.asarray(weights, dtype=float)[None, :, :])[0])


def categorical_kl(left: np.ndarray, right: np.ndarray) -> float:
    left_safe = np.clip(np.asarray(left, dtype=float), 1e-12, 1.0)
    right_safe = np.clip(np.asarray(right, dtype=float), 1e-12, 1.0)
    return float(np.sum(left_safe * (np.log(left_safe) - np.log(right_safe))))


def reduced_logits(weights: np.ndarray) -> np.ndarray:
    logged = np.log(np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0))
    return np.clip(logged[:-1] - logged[-1], -12.0, 12.0)


def weights_from_reduced_logits(logits: np.ndarray) -> np.ndarray:
    return softmax(np.concatenate([np.asarray(logits, dtype=float), [0.0]]))


def shrink_to_kl_budget(
    weights: np.ndarray,
    natural: np.ndarray,
    budget: float,
) -> np.ndarray:
    if categorical_kl(weights, natural) <= 0.8 * budget:
        return weights
    scale = 1.0
    for _attempt in range(80):
        candidate = natural + scale * (weights - natural)
        candidate /= candidate.sum()
        if categorical_kl(candidate, natural) <= 0.8 * budget:
            return candidate
        scale *= 0.5
    return natural.copy()


def aggregate_starts(
    fitted: FittedPolicy,
    natural: np.ndarray,
    budget: float,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    starts = [natural.copy()]
    for index in np.argsort(fitted.dataset.y)[: min(10, fitted.dataset.n)]:
        starts.append(
            shrink_to_kl_budget(
                fitted.dataset.weights[index, 0],
                natural,
                budget,
            )
        )
    scales = (0.25, 0.50, 1.0, 1.5)
    while len(starts) < count:
        scale = scales[(len(starts) - 1) % len(scales)]
        sample = softmax(np.log(np.maximum(natural, 1e-12)) + scale * generator.normal(size=len(natural)))
        starts.append(shrink_to_kl_budget(sample, natural, budget))
    return [reduced_logits(weights) for weights in starts[:count]]


def optimize_tied_aggregate(
    fitted: FittedPolicy,
    natural: np.ndarray,
    aggregate_kl_budget: float,
    starts: int,
    seed: int,
) -> AggregateResult:
    """Optimize the independently fitted one-phase model under a hard KL ball."""
    if fitted.policy_class != observatory.SINGLE_PHASE:
        raise ValueError("Aggregate optimization requires the independently fitted one-phase model")
    if aggregate_kl_budget <= 0:
        raise ValueError("Aggregate KL budget must be positive")

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_reduced_logits(logits)
        tied = np.stack([weights, weights])
        return scalar_prediction(fitted.model, tied)

    def constraint(logits: np.ndarray) -> float:
        weights = weights_from_reduced_logits(logits)
        return aggregate_kl_budget - categorical_kl(weights, natural)

    best: tuple[float, np.ndarray] | None = None
    successful = 0
    finite = 0
    for start in aggregate_starts(
        fitted,
        natural,
        aggregate_kl_budget,
        starts,
        seed,
    ):
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(-12.0, 12.0)] * len(start),
            constraints=[{"type": "ineq", "fun": constraint}],
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if result.success:
            successful += 1
        if not np.isfinite(result.fun):
            continue
        finite += 1
        aggregate = weights_from_reduced_logits(np.asarray(result.x, dtype=float))
        realized_kl = categorical_kl(aggregate, natural)
        if realized_kl > aggregate_kl_budget + FEASIBILITY_TOLERANCE:
            continue
        prediction = objective(np.asarray(result.x, dtype=float))
        if best is None or prediction < best[0]:
            best = (prediction, aggregate)
    if best is None:
        raise RuntimeError(f"No feasible tied optimum for {fitted.target} at aggregate KL {aggregate_kl_budget:g}")
    aggregate = best[1]
    return AggregateResult(
        weights=np.stack([aggregate, aggregate]),
        prediction=best[0],
        aggregate_kl=categorical_kl(aggregate, natural),
        successful_starts=successful,
        finite_starts=finite,
    )


def phase_information(
    weights: np.ndarray,
    aggregate: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> float:
    return float(alpha0 * categorical_kl(weights[0], aggregate) + alpha1 * categorical_kl(weights[1], aggregate))


def nearest_aggregate_tv(aggregate: np.ndarray, references: np.ndarray) -> float:
    reference_aggregate = 0.8 * references[:, 0] + 0.2 * references[:, 1]
    return float(0.5 * np.abs(reference_aggregate - aggregate).sum(axis=1).min())


def nearest_policy_tv(weights: np.ndarray, references: np.ndarray) -> float:
    return float(
        prior_hpr_panel.weighted_policy_tv(
            weights[None, :, :],
            references,
            0.8,
            0.2,
        ).min()
    )


def candidate_id(target: str, aggregate_budget: float, phase_budget: float) -> str:
    return f"hprc280_{TARGET_TAGS[target]}_aklb{float_tag(aggregate_budget)}_" f"eps{float_tag(phase_budget)}"


def record_candidate(
    *,
    target: str,
    aggregate_budget: float,
    phase_budget: float,
    weights: np.ndarray,
    tied_weights: np.ndarray,
    one_phase: FittedPolicy,
    two_phase: FittedPolicy,
    natural: np.ndarray,
    heldout_weights: np.ndarray,
    aggregate_result: AggregateResult,
    successful_phase_starts: int,
) -> dict[str, Any]:
    alpha0, alpha1 = observatory.phase_fractions(one_phase.dataset)
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    tied_aggregate = tied_weights[0]
    if not np.allclose(aggregate, tied_aggregate, atol=1e-7):
        raise ValueError("Phase optimization changed the tied aggregate")
    information = phase_information(weights, aggregate, alpha0, alpha1)
    if information > phase_budget + FEASIBILITY_TOLERANCE:
        raise ValueError(f"Phase information {information:.8f} exceeds budget {phase_budget:.8f}")
    one_tied_prediction = scalar_prediction(one_phase.model, tied_weights)
    two_tied_prediction = scalar_prediction(two_phase.model, tied_weights)
    two_candidate_prediction = scalar_prediction(two_phase.model, weights)
    epochs = np.array(
        [
            TARGET_BUDGET_DOLMA3_COMMON_CRAWL * value / TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain]
            for domain, value in zip(one_phase.dataset.domain_names, aggregate, strict=True)
        ]
    )
    return {
        "candidate_id": candidate_id(target, aggregate_budget, phase_budget),
        "target": target,
        "policy_class": observatory.SINGLE_PHASE if phase_budget == 0.0 else observatory.TWO_PHASE,
        "candidate_kind": (
            "independently_fitted_tied_control" if phase_budget == 0.0 else "aggregate_matched_phase_order"
        ),
        "aggregate_control_id": candidate_id(target, aggregate_budget, 0.0),
        "aggregate_kl_budget": aggregate_budget,
        "aggregate_kl_to_proportional": categorical_kl(aggregate, natural),
        "phase_information_budget": phase_budget,
        "phase_information_kl": information,
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "one_phase_tied_prediction": one_tied_prediction,
        "two_phase_tied_prediction": two_tied_prediction,
        "two_phase_candidate_prediction": two_candidate_prediction,
        "within_two_phase_predicted_gain": two_tied_prediction - two_candidate_prediction,
        "cross_fit_predicted_difference": one_tied_prediction - two_candidate_prediction,
        "aggregate_optimizer_prediction": aggregate_result.prediction,
        "aggregate_successful_starts": aggregate_result.successful_starts,
        "aggregate_finite_starts": aggregate_result.finite_starts,
        "phase_successful_starts": successful_phase_starts,
        "max_bucket_weight": float(weights.max()),
        "max_simulated_epoch": float(epochs.max()),
        "mean_simulated_epoch": float(epochs.mean()),
        "min_one_phase_fit_aggregate_tv": nearest_aggregate_tv(
            aggregate,
            one_phase.dataset.weights,
        ),
        "min_two_phase_fit_aggregate_tv": nearest_aggregate_tv(
            aggregate,
            two_phase.dataset.weights,
        ),
        "min_two_phase_fit_policy_tv": nearest_policy_tv(
            weights,
            two_phase.dataset.weights,
        ),
        "min_3e18_heldout_policy_tv": nearest_policy_tv(weights, heldout_weights),
        "coordinate_hash": prior_hpr_panel.policy_hash(weights),
        "weights": weights,
    }


def canonical_weights(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    phase0 = frame[[f"phase_0_weight::{domain}" for domain in domains]].to_numpy(float)
    phase1 = frame[[f"phase_1_weight::{domain}" for domain in domains]].to_numpy(float)
    return np.stack([phase0, phase1], axis=1)


def load_july_policies(domains: list[str]) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    weights: list[np.ndarray] = []
    for path in sorted((JULY_PANEL / "mixtures").glob("*.csv")):
        frame = pd.read_csv(path).set_index("domain").reindex(domains)
        if frame[["phase_0_weight", "phase_1_weight"]].isna().any().any():
            raise ValueError(f"July mixture {path} is missing canonical domains")
        labels.append(f"july:{path.stem}")
        weights.append(
            np.stack(
                [
                    frame["phase_0_weight"].to_numpy(float),
                    frame["phase_1_weight"].to_numpy(float),
                ]
            )
        )
    if not weights:
        return [], np.zeros((0, 2, len(domains)), dtype=float)
    return labels, np.stack(weights)


def duplicate_references(
    one_phase: Any,
    two_phase: Any,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    labels = [
        *[f"300m_one_fit:{value}" for value in one_phase.frame["run_name"].astype(str)],
        *[f"300m_two_fit:{value}" for value in two_phase.frame["run_name"].astype(str)],
        *[f"3e18_heldout:{value}" for value in heldout_frame["heldout_id"].astype(str)],
    ]
    arrays = [one_phase.weights, two_phase.weights, heldout_weights]
    july_labels, july_weights = load_july_policies(list(two_phase.domain_names))
    labels.extend(july_labels)
    arrays.append(july_weights)
    return labels, np.concatenate(arrays, axis=0)


def duplicate_audit(
    manifest: pd.DataFrame,
    stored_weights: dict[str, np.ndarray],
    reference_labels: list[str],
    reference_weights: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prior_candidates: list[np.ndarray] = []
    prior_labels: list[str] = []
    for candidate in manifest["candidate_id"]:
        weights = stored_weights[candidate]
        distances = prior_hpr_panel.weighted_policy_tv(
            weights[None, :, :],
            reference_weights,
            0.8,
            0.2,
        )
        index = int(np.argmin(distances))
        nearest_candidate_tv = float("inf")
        nearest_candidate = None
        if prior_candidates:
            candidate_distances = prior_hpr_panel.weighted_policy_tv(
                weights[None, :, :],
                np.stack(prior_candidates),
                0.8,
                0.2,
            )
            candidate_index = int(np.argmin(candidate_distances))
            nearest_candidate_tv = float(candidate_distances[candidate_index])
            nearest_candidate = prior_labels[candidate_index]
        rows.append(
            {
                "candidate_id": candidate,
                "nearest_existing_reference": reference_labels[index],
                "nearest_existing_policy_tv": float(distances[index]),
                "existing_exact_duplicate": float(distances[index]) <= EXACT_POLICY_TV,
                "nearest_prior_candidate": nearest_candidate,
                "nearest_prior_candidate_tv": nearest_candidate_tv,
                "candidate_exact_duplicate": nearest_candidate_tv <= EXACT_POLICY_TV,
            }
        )
        prior_candidates.append(weights)
        prior_labels.append(candidate)
    return pd.DataFrame(rows)


def original_hpr_fit(
    dataset: Any,
    target: str,
    policy_class: str,
) -> tuple[Any, np.ndarray, hierarchical.Config]:
    config, _selection = observatory.select_hierarchical_phase_replay_config(
        dataset,
        policy_class,
    )
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in observatory.folds(dataset, hierarchical.SCREEN_SEED):
        model = observatory.hierarchical_phase_replay_fit(dataset, train, config)
        prediction[test] = model.predict(dataset.weights[test])
    model = observatory.hierarchical_phase_replay_fit(
        dataset,
        np.arange(dataset.n),
        config,
    )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete original HPR OOF for {target}/{policy_class}")
    return model, prediction, config


def rank_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    order = np.argsort(predicted)
    best = float(np.min(observed))
    result: dict[str, float | int] = {
        "n": len(observed),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "prediction_range": float(np.ptp(predicted)),
        "observed_range": float(np.ptp(observed)),
    }
    for count in (1, 3, 5):
        selected = order[: min(count, len(order))]
        result[f"regret_at_{count}"] = float(np.min(observed[selected]) - best)
    return result


def local_evaluation(
    fitted: dict[tuple[str, str], FittedPolicy],
    originals: dict[tuple[str, str], tuple[Any, np.ndarray, hierarchical.Config]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cross_scale_rows: list[dict[str, Any]] = []
    two_reference = fitted[("uncheatable", observatory.TWO_PHASE)].dataset
    canonical_heldout = pd.read_csv(swarm39.CANONICAL / "300m_heldouts.csv")
    heldout_weights = canonical_weights(canonical_heldout, list(two_reference.domain_names))
    delphi_reference = observatory.load_delphi_3e18_fit_dataset("uncheatable")
    delphi_frame, delphi_weights = observatory.load_delphi_3e18_heldouts(delphi_reference)

    for target in TARGETS:
        target_column = TARGET_COLUMNS[target]
        for policy_class in POLICY_CLASSES:
            fit = fitted[(target, policy_class)]
            original_model, original_oof, _config = originals[(target, policy_class)]
            for model_name, prediction in (
                ("corrected_hpr", fit.oof_prediction),
                ("original_hpr", original_oof),
            ):
                rows.append(
                    {
                        "target": target,
                        "policy_class": policy_class,
                        "evaluation": "fit_oof",
                        "model": model_name,
                        **swarm39.metric_row(fit.dataset.y, prediction),
                    }
                )

            if policy_class == observatory.TWO_PHASE:
                mask = (
                    canonical_heldout["policy_class"].eq("two_phase") & canonical_heldout[target_column].notna()
                ).to_numpy()
                observed = canonical_heldout.loc[mask, target_column].to_numpy(float)
                for model_name, model in (
                    ("corrected_hpr", fit.model),
                    ("original_hpr", original_model),
                ):
                    predicted = model.predict(heldout_weights[mask])
                    rows.append(
                        {
                            "target": target,
                            "policy_class": policy_class,
                            "evaluation": "300m_coordinate_disjoint",
                            "model": model_name,
                            **swarm39.metric_row(observed, predicted),
                        }
                    )

            tied = (
                np.max(
                    np.abs(delphi_weights[:, 0] - delphi_weights[:, 1]),
                    axis=1,
                )
                < 1e-10
            )
            policy_mask = tied if policy_class == observatory.SINGLE_PHASE else ~tied
            observed = delphi_frame.loc[policy_mask, target_column].to_numpy(float)
            for model_name, model in (
                ("corrected_hpr", fit.model),
                ("original_hpr", original_model),
            ):
                predicted = model.predict(delphi_weights[policy_mask])
                cross_scale_rows.append(
                    {
                        "target": target,
                        "policy_class": policy_class,
                        "evaluation": "3e18_rank_only",
                        "model": model_name,
                        **rank_metrics(observed, predicted),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(cross_scale_rows)


def selected_config_row(fitted: FittedPolicy) -> dict[str, Any]:
    diagnostics = corrected.design_diagnostics(
        fitted.structured,
        fitted.config,
        CORRECTIONS,
    )
    metrics = swarm39.metric_row(fitted.dataset.y, fitted.oof_prediction)
    return {
        "target": fitted.target,
        "policy_class": fitted.policy_class,
        "fit_rows": fitted.dataset.n,
        "variant": fitted.config.variant.value,
        "shape_index": fitted.config.shape_index,
        **asdict(fitted.config.shape),
        "l2": fitted.config.l2,
        "residual_shrink": fitted.config.residual_shrink,
        **{f"oof_{key}": value for key, value in metrics.items()},
        **diagnostics,
    }


def bootstrap_fixed_candidates(
    fitted: dict[tuple[str, str], FittedPolicy],
    manifest: pd.DataFrame,
    stored_weights: dict[str, np.ndarray],
    draws: int,
) -> pd.DataFrame:
    """Bootstrap fitted heads while holding selected configs and policies fixed."""
    generator = np.random.default_rng(20260727)
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        local = manifest.loc[manifest["target"].eq(target)]
        for draw in range(draws):
            models: dict[str, corrected.CorrectedModel] = {}
            for policy_class in POLICY_CLASSES:
                fit = fitted[(target, policy_class)]
                sample = generator.integers(0, fit.structured.n, fit.structured.n)
                models[policy_class] = corrected.fit_corrected(
                    fit.structured,
                    fit.config,
                    CORRECTIONS,
                    sample,
                )
            for record in local.to_dict("records"):
                weights = stored_weights[str(record["candidate_id"])]
                control = stored_weights[str(record["aggregate_control_id"])]
                one_prediction = scalar_prediction(
                    models[observatory.SINGLE_PHASE],
                    control,
                )
                tied_prediction = scalar_prediction(
                    models[observatory.TWO_PHASE],
                    control,
                )
                candidate_prediction = scalar_prediction(
                    models[observatory.TWO_PHASE],
                    weights,
                )
                rows.append(
                    {
                        "candidate_id": record["candidate_id"],
                        "target": target,
                        "draw": draw,
                        "one_phase_tied_prediction": one_prediction,
                        "two_phase_tied_prediction": tied_prediction,
                        "two_phase_candidate_prediction": candidate_prediction,
                        "within_two_phase_predicted_gain": tied_prediction - candidate_prediction,
                    }
                )
    return pd.DataFrame(rows)


def summarize_bootstrap(draws: pd.DataFrame) -> pd.DataFrame:
    return draws.groupby(["candidate_id", "target"], as_index=False).agg(
        prediction_mean=("two_phase_candidate_prediction", "mean"),
        prediction_sd=("two_phase_candidate_prediction", "std"),
        gain_mean=("within_two_phase_predicted_gain", "mean"),
        gain_p05=("within_two_phase_predicted_gain", lambda values: np.quantile(values, 0.05)),
        gain_p50=("within_two_phase_predicted_gain", "median"),
        gain_p95=("within_two_phase_predicted_gain", lambda values: np.quantile(values, 0.95)),
        gain_positive_share=("within_two_phase_predicted_gain", lambda values: np.mean(values > 0)),
    )


def fold_path_stability(
    fitted: dict[tuple[str, str], FittedPolicy],
    full_manifest: pd.DataFrame,
    full_weights: dict[str, np.ndarray],
    natural: np.ndarray,
    aggregate_budgets: tuple[float, ...],
    phase_budgets: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        one = fitted[(target, observatory.SINGLE_PHASE)]
        two = fitted[(target, observatory.TWO_PHASE)]
        one_splits = hierarchical.split_indices(
            one.structured,
            dataset_id(target),
            np.arange(one.structured.n),
            hierarchical.SCREEN_SEED,
        )
        two_splits = hierarchical.split_indices(
            two.structured,
            dataset_id(target),
            np.arange(two.structured.n),
            hierarchical.SCREEN_SEED,
        )
        alpha0, alpha1 = observatory.phase_fractions(one.dataset)
        for fold, ((one_train, _one_test), (two_train, _two_test)) in enumerate(
            zip(one_splits, two_splits, strict=True)
        ):
            fold_one = replace(
                one,
                model=corrected.fit_corrected(
                    one.structured,
                    one.config,
                    CORRECTIONS,
                    one_train,
                ),
            )
            fold_two = replace(
                two,
                model=corrected.fit_corrected(
                    two.structured,
                    two.config,
                    CORRECTIONS,
                    two_train,
                ),
            )
            for aggregate_budget in aggregate_budgets:
                aggregate_result = optimize_tied_aggregate(
                    fold_one,
                    natural,
                    aggregate_budget,
                    FOLD_OPTIMIZER_STARTS,
                    20260727 + fold,
                )
                aggregate = aggregate_result.weights[0]
                for phase_budget in phase_budgets:
                    if phase_budget == 0.0:
                        weights = aggregate_result.weights
                        successful = aggregate_result.successful_starts
                    else:
                        phase_result = prior_hpr_panel.optimize_fixed_aggregate(
                            fold_two,
                            aggregate,
                            phase_budget,
                            alpha0,
                            alpha1,
                        )
                        weights = phase_result.weights
                        successful = phase_result.successful_starts
                    identifier = candidate_id(target, aggregate_budget, phase_budget)
                    full = full_weights[identifier]
                    control = np.stack([aggregate, aggregate])
                    rows.append(
                        {
                            "candidate_id": identifier,
                            "target": target,
                            "fold": fold,
                            "policy_tv_to_full_path": nearest_policy_tv(
                                weights,
                                full[None, :, :],
                            ),
                            "within_two_phase_predicted_gain": (
                                scalar_prediction(
                                    fold_two.model,
                                    control,
                                )
                                - scalar_prediction(fold_two.model, weights)
                            ),
                            "successful_starts": successful,
                        }
                    )
    return pd.DataFrame(rows)


def render_path_plot(manifest: pd.DataFrame, output_dir: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable predicted phase gain",
            "Table-9 predicted phase gain",
            "Uncheatable policy geometry",
            "Table-9 policy geometry",
        ),
        vertical_spacing=0.14,
    )
    colors = ("#1a9850", "#fee08b", "#d73027")
    for column, target in enumerate(TARGETS, start=1):
        local_target = manifest.loc[manifest["target"].eq(target)]
        for color, aggregate_budget in zip(colors, sorted(local_target["aggregate_kl_budget"].unique()), strict=True):
            local = local_target.loc[local_target["aggregate_kl_budget"].eq(aggregate_budget)].sort_values(
                "phase_information_budget"
            )
            figure.add_trace(
                go.Scatter(
                    x=local["phase_information_budget"],
                    y=local["within_two_phase_predicted_gain"],
                    mode="lines+markers",
                    name=f"aggregate KL ≤ {aggregate_budget:g}",
                    legendgroup=f"akl-{aggregate_budget:g}",
                    showlegend=column == 1,
                    line={"color": color},
                    customdata=np.column_stack(
                        [
                            local["candidate_id"],
                            local["phase_total_variation"],
                            local["max_simulated_epoch"],
                            local["min_two_phase_fit_policy_tv"],
                        ]
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>phase budget=%{x:.4g}<br>"
                        "predicted gain=%{y:.6f}<br>phase TV=%{customdata[1]:.4f}<br>"
                        "max epochs=%{customdata[2]:.2f}<br>"
                        "nearest fit TV=%{customdata[3]:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=local["phase_total_variation"],
                    y=local["max_simulated_epoch"],
                    mode="markers",
                    marker={
                        "color": local["within_two_phase_predicted_gain"],
                        "colorscale": "RdYlGn_r",
                        "size": 10,
                    },
                    name=f"aggregate KL ≤ {aggregate_budget:g}",
                    legendgroup=f"akl-{aggregate_budget:g}",
                    showlegend=False,
                    text=local["candidate_id"],
                    hovertemplate=("%{text}<br>phase TV=%{x:.4f}<br>" "max epochs=%{y:.2f}<extra></extra>"),
                ),
                row=2,
                col=column,
            )
    figure.update_xaxes(title_text="phase-information budget", row=1)
    figure.update_xaxes(title_text="phase total variation", row=2)
    figure.update_yaxes(title_text="predicted BPB gain over tied", row=1)
    figure.update_yaxes(title_text="maximum simulated epochs", row=2)
    figure.update_layout(
        title="Exact-280 corrective HPR: tied controls and aggregate-matched phase paths",
        template="plotly_white",
        width=1500,
        height=950,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(
        output_dir / "predicted_paths.html",
        include_plotlyjs=True,
        config=PLOT_CONFIG,
    )


def render_mixture_explorer(
    manifest: pd.DataFrame,
    stored_weights: dict[str, np.ndarray],
    natural: np.ndarray,
    domains: list[str],
    output_dir: Path,
) -> None:
    figure = go.Figure()
    buttons: list[dict[str, Any]] = []
    for candidate_index, row in manifest.reset_index(drop=True).iterrows():
        identifier = str(row["candidate_id"])
        weights = stored_weights[identifier]
        aggregate = 0.8 * weights[0] + 0.2 * weights[1]
        order = np.argsort(-np.maximum.reduce([natural, weights[0], weights[1], aggregate]))
        y = [domains[index] for index in order]
        values = (
            ("proportional", natural[order], "#8da0ae"),
            ("phase 0", weights[0, order], "#e76f51"),
            ("phase 1", weights[1, order], "#2a9d8f"),
            ("aggregate", aggregate[order], "#264653"),
        )
        visibility = [False] * (4 * len(manifest))
        for trace_offset, (name, x, color) in enumerate(values):
            figure.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    orientation="h",
                    name=name,
                    marker_color=color,
                    visible=candidate_index == 0,
                    hovertemplate=f"{name}<br>%{{y}}<br>weight=%{{x:.6f}}<extra></extra>",
                )
            )
            visibility[4 * candidate_index + trace_offset] = True
        buttons.append(
            {
                "label": identifier,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "title": (
                            f"{identifier}<br><sup>{row['target']} · "
                            f"aggregate KL {row['aggregate_kl_to_proportional']:.4f} · "
                            f"phase information {row['phase_information_kl']:.4f}</sup>"
                        )
                    },
                ],
            }
        )
    first = manifest.iloc[0]
    figure.update_layout(
        title=(
            f"{first['candidate_id']}<br><sup>{first['target']} · "
            f"aggregate KL {first['aggregate_kl_to_proportional']:.4f} · "
            f"phase information {first['phase_information_kl']:.4f}</sup>"
        ),
        template="plotly_white",
        barmode="group",
        width=1500,
        height=1300,
        margin={"l": 300, "t": 150},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        xaxis_title="mixture weight",
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "y": 1.03},
    )
    figure.write_html(
        output_dir / "candidate_mixtures.html",
        include_plotlyjs=True,
        config=PLOT_CONFIG,
    )


def write_report(
    output_dir: Path,
    manifest: pd.DataFrame,
    configs: pd.DataFrame,
    metrics: pd.DataFrame,
    cross_scale: pd.DataFrame,
    bootstrap: pd.DataFrame,
    folds: pd.DataFrame,
    duplicates: pd.DataFrame,
    gate: dict[str, bool],
) -> None:
    config_columns = [
        "target",
        "policy_class",
        "fit_rows",
        "oof_rmse",
        "oof_spearman",
        "exponent",
        "late_multiplier",
        "forgetting_rate",
        "penalty_threshold",
        "l2",
        "residual_shrink",
        "columns",
        "active_columns",
        "effective_dof",
    ]
    best_paths = (
        manifest.loc[manifest["phase_information_budget"].gt(0)]
        .sort_values("within_two_phase_predicted_gain", ascending=False)
        .groupby("target", as_index=False)
        .head(3)
    )
    bootstrap_best = bootstrap.loc[bootstrap["candidate_id"].isin(best_paths["candidate_id"])]
    fold_summary = (
        folds.groupby(["candidate_id", "target"], as_index=False).agg(
            median_policy_tv=("policy_tv_to_full_path", "median"),
            max_policy_tv=("policy_tv_to_full_path", "max"),
            positive_gain_fold_share=("within_two_phase_predicted_gain", lambda values: np.mean(values > 0)),
        )
        if not folds.empty
        else pd.DataFrame()
    )
    lines = [
        "# Corrective exact-280 HPR decomposed panel",
        "",
        "## Frozen procedure",
        "",
        "- Source data: canonical 300M one-phase 280-row and two-phase 280-row panels.",
        "- Targets are fit and optimized separately: Uncheatable BPB and Table-9 macro BPB.",
        "- Model: HPR with a full-rank hierarchical penalty and duplicate ledgers removed.",
        "- The one-phase restriction fixes late multiplier to one and forgetting to zero.",
        (
            "- The two-phase fit retains the original HPR shape library; no normalized ledger, recency kernel, "
            "bounded output link, or model ensemble is used."
        ),
        "- One-phase and two-phase restrictions are independently selected and fitted.",
        f"- Aggregate KL budgets: `{sorted(manifest['aggregate_kl_budget'].unique().tolist())}`.",
        f"- Conditional phase-information budgets: `{sorted(manifest['phase_information_budget'].unique().tolist())}`.",
        "- The six epsilon-zero policies are independently optimized tied controls.",
        "- All nonzero candidates preserve their matched tied aggregate to numerical tolerance.",
        "",
        "## Selected fits",
        "",
        configs[config_columns].to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Local and development evaluation",
        "",
        metrics.to_markdown(index=False, floatfmt=".6g"),
        "",
        "The 3e18 table below is rank-only cross-scale development evidence. Absolute RMSE and calibration are not "
        "reported because the 300M model's output level is not a 3e18 calibration model.",
        "",
        cross_scale.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Strongest predicted phase paths",
        "",
        best_paths[
            [
                "candidate_id",
                "target",
                "aggregate_kl_to_proportional",
                "phase_information_kl",
                "phase_total_variation",
                "within_two_phase_predicted_gain",
                "max_simulated_epoch",
                "min_two_phase_fit_policy_tv",
            ]
        ].to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Fixed-candidate bootstrap",
        "",
        "This resamples fitted heads with the selected configurations held fixed. It is not a full hyperparameter "
        "or raw-optimum bootstrap.",
        "",
        bootstrap_best.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Fixed-config fold path stability",
        "",
        (fold_summary.to_markdown(index=False, floatfmt=".6g") if not fold_summary.empty else "Skipped by request."),
        "",
        "## Duplicate audit",
        "",
        f"- Existing-coordinate duplicates: `{int(duplicates['existing_exact_duplicate'].sum())}`.",
        f"- Within-panel coordinate aliases: `{int(duplicates['candidate_exact_duplicate'].sum())}`.",
        "",
        "## Preregistered local gate",
        "",
        *[f"- {'PASS' if value else 'FAIL'}: `{name}`." for name, value in gate.items()],
        "",
        "Passing this gate only makes the panel launchable. It does not establish that HPR is the paper model or "
        "that any candidate will create a 3e18 frontier.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    aggregate_budgets = parse_float_tuple(args.aggregate_kl_budgets)
    phase_budgets = parse_float_tuple(args.phase_information_budgets)
    if any(value <= 0 for value in aggregate_budgets):
        raise ValueError("Aggregate KL budgets must be positive")
    if 0.0 not in phase_budgets or any(value < 0 for value in phase_budgets):
        raise ValueError("Phase-information path must contain epsilon zero and no negative values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mixtures_dir = args.output_dir / "mixtures"
    mixtures_dir.mkdir(parents=True, exist_ok=True)

    datasets = {target: prior_hpr_panel.policy_datasets("300m", target, None) for target in TARGETS}
    reference = datasets["uncheatable"][observatory.TWO_PHASE]
    alpha0, alpha1 = observatory.phase_fractions(reference)
    if not np.allclose((alpha0, alpha1), (0.8, 0.2), atol=1e-10):
        raise ValueError(f"Expected 80/20 source split, found {alpha0:.8f}/{alpha1:.8f}")
    natural = observatory.natural_weights(reference, alpha0)
    delphi_reference = observatory.load_delphi_3e18_fit_dataset("uncheatable")
    delphi_frame, delphi_weights = observatory.load_delphi_3e18_heldouts(delphi_reference)
    if list(reference.domain_names) != list(delphi_reference.domain_names):
        raise ValueError("300M and 3e18 bucket orders differ")

    fitted: dict[tuple[str, str], FittedPolicy] = {}
    originals: dict[
        tuple[str, str],
        tuple[Any, np.ndarray, hierarchical.Config],
    ] = {}
    config_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for policy_class in POLICY_CLASSES:
            print(f"Fitting corrected HPR: {target}/{policy_class}", flush=True)
            dataset = datasets[target][policy_class]
            fit = fit_policy(dataset, target, policy_class)
            fitted[(target, policy_class)] = fit
            config_rows.append(selected_config_row(fit))
            print(f"Fitting original HPR baseline: {target}/{policy_class}", flush=True)
            originals[(target, policy_class)] = original_hpr_fit(
                dataset,
                target,
                policy_class,
            )

    records: list[dict[str, Any]] = []
    stored_weights: dict[str, np.ndarray] = {}
    for target in TARGETS:
        one = fitted[(target, observatory.SINGLE_PHASE)]
        two = fitted[(target, observatory.TWO_PHASE)]
        for aggregate_budget in aggregate_budgets:
            print(
                f"Optimizing {target} tied aggregate at KL budget {aggregate_budget:g}",
                flush=True,
            )
            aggregate_result = optimize_tied_aggregate(
                one,
                natural,
                aggregate_budget,
                args.optimizer_starts,
                20260727,
            )
            tied = aggregate_result.weights
            aggregate = tied[0]
            for phase_budget in phase_budgets:
                if phase_budget == 0.0:
                    weights = tied
                    successful_phase_starts = aggregate_result.successful_starts
                else:
                    print(
                        f"Optimizing {target} phase order at aggregate KL {aggregate_budget:g}, "
                        f"phase budget {phase_budget:g}",
                        flush=True,
                    )
                    phase_result = prior_hpr_panel.optimize_fixed_aggregate(
                        two,
                        aggregate,
                        phase_budget,
                        alpha0,
                        alpha1,
                    )
                    weights = phase_result.weights
                    successful_phase_starts = phase_result.successful_starts
                record = record_candidate(
                    target=target,
                    aggregate_budget=aggregate_budget,
                    phase_budget=phase_budget,
                    weights=weights,
                    tied_weights=tied,
                    one_phase=one,
                    two_phase=two,
                    natural=natural,
                    heldout_weights=delphi_weights,
                    aggregate_result=aggregate_result,
                    successful_phase_starts=successful_phase_starts,
                )
                identifier = str(record["candidate_id"])
                stored_weights[identifier] = np.asarray(record.pop("weights"), dtype=float)
                records.append(record)

    manifest = (
        pd.DataFrame(records)
        .sort_values(["target", "aggregate_kl_budget", "phase_information_budget"])
        .reset_index(drop=True)
    )
    reference_labels, reference_weights = duplicate_references(
        datasets["uncheatable"][observatory.SINGLE_PHASE],
        datasets["uncheatable"][observatory.TWO_PHASE],
        delphi_frame,
        delphi_weights,
    )
    duplicates = duplicate_audit(
        manifest,
        stored_weights,
        reference_labels,
        reference_weights,
    )
    manifest = manifest.merge(duplicates, on="candidate_id", validate="one_to_one")
    manifest["launch_primary"] = ~manifest["existing_exact_duplicate"] & ~manifest["candidate_exact_duplicate"]

    for row in manifest.to_dict("records"):
        identifier = str(row["candidate_id"])
        weights = stored_weights[identifier]
        frame = prior_hpr_panel.mixture_frame(
            delphi_reference,
            natural,
            weights,
        )
        aggregate = alpha0 * weights[0] + alpha1 * weights[1]
        frame["simulated_epochs"] = [
            TARGET_BUDGET_DOLMA3_COMMON_CRAWL * value / TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain]
            for domain, value in zip(reference.domain_names, aggregate, strict=True)
        ]
        frame.to_csv(mixtures_dir / f"{identifier}.csv", index=False)
        for phase in (0, 1):
            for domain, value in zip(
                reference.domain_names,
                weights[phase],
                strict=True,
            ):
                manifest.loc[
                    manifest["candidate_id"].eq(identifier),
                    f"phase_{phase}_{domain}",
                ] = float(value)

    metrics, cross_scale = local_evaluation(fitted, originals)
    bootstrap_draws = bootstrap_fixed_candidates(
        fitted,
        manifest,
        stored_weights,
        args.bootstrap_draws,
    )
    bootstrap = summarize_bootstrap(bootstrap_draws)
    folds = (
        pd.DataFrame()
        if args.skip_fold_paths
        else fold_path_stability(
            fitted,
            manifest,
            stored_weights,
            natural,
            aggregate_budgets,
            phase_budgets,
        )
    )

    configs = pd.DataFrame(config_rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    manifest.loc[manifest["launch_primary"]].to_csv(
        args.output_dir / "launcher_source_panel.csv",
        index=False,
    )
    configs.to_csv(args.output_dir / "selected_configs.csv", index=False)
    metrics.to_csv(args.output_dir / "local_metrics.csv", index=False)
    cross_scale.to_csv(args.output_dir / "cross_scale_rank_metrics.csv", index=False)
    bootstrap_draws.to_csv(args.output_dir / "bootstrap_draws.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_summary.csv", index=False)
    folds.to_csv(args.output_dir / "fold_path_stability.csv", index=False)
    duplicates.to_csv(args.output_dir / "duplicate_audit.csv", index=False)

    corrected_oof = metrics.loc[metrics["evaluation"].eq("fit_oof") & metrics["model"].eq("corrected_hpr")].set_index(
        ["target", "policy_class"]
    )
    original_oof = metrics.loc[metrics["evaluation"].eq("fit_oof") & metrics["model"].eq("original_hpr")].set_index(
        ["target", "policy_class"]
    )
    oof_ratios = corrected_oof["rmse"] / original_oof["rmse"]
    nonzero = manifest.loc[manifest["phase_information_budget"].gt(0)]
    bootstrap_nonzero = bootstrap.loc[bootstrap["candidate_id"].isin(nonzero["candidate_id"])]
    gate = {
        "exact_280_rows_for_all_four_fits": all(fit.dataset.n == FIT_ROWS for fit in fitted.values()),
        "corrected_oof_within_5_percent_of_original_hpr": bool((oof_ratios <= 1.05).all()),
        "all_aggregate_kl_constraints_satisfied": bool(
            (manifest["aggregate_kl_to_proportional"] <= manifest["aggregate_kl_budget"] + FEASIBILITY_TOLERANCE).all()
        ),
        "all_phase_information_constraints_satisfied": bool(
            (manifest["phase_information_kl"] <= manifest["phase_information_budget"] + FEASIBILITY_TOLERANCE).all()
        ),
        "all_phase_paths_preserve_their_tied_aggregate": True,
        "no_existing_or_internal_coordinate_duplicates": bool(
            not manifest["existing_exact_duplicate"].any() and not manifest["candidate_exact_duplicate"].any()
        ),
        "at_least_one_bootstrap_robust_phase_gain_per_target": all(
            bool(
                (
                    bootstrap_nonzero.loc[
                        bootstrap_nonzero["target"].eq(target),
                        "gain_p05",
                    ]
                    > 0
                ).any()
            )
            for target in TARGETS
        ),
        "plausible_weight_and_epoch_caps": bool(
            (manifest["max_bucket_weight"] <= 0.50).all() and (manifest["max_simulated_epoch"] <= 25.0).all()
        ),
    }
    if not folds.empty:
        gate["median_fold_path_tv_below_0p20"] = bool(
            folds.groupby("candidate_id")["policy_tv_to_full_path"].median().max() <= 0.20
        )

    render_path_plot(manifest, args.output_dir)
    render_mixture_explorer(
        manifest,
        stored_weights,
        natural,
        list(reference.domain_names),
        args.output_dir,
    )
    write_report(
        args.output_dir,
        manifest,
        configs,
        metrics,
        cross_scale,
        bootstrap,
        folds,
        duplicates,
        gate,
    )
    summary = {
        "fit_source": "300m_exact_280",
        "targets": list(TARGETS),
        "fit_rows": {policy_class: FIT_ROWS for policy_class in POLICY_CLASSES},
        "aggregate_kl_budgets": list(aggregate_budgets),
        "phase_information_budgets": list(phase_budgets),
        "proposal_rows": len(manifest),
        "tied_controls": int(manifest["phase_information_budget"].eq(0).sum()),
        "phase_order_candidates": int(manifest["phase_information_budget"].gt(0).sum()),
        "launch_primary_rows": int(manifest["launch_primary"].sum()),
        "corrections": asdict(CORRECTIONS),
        "one_phase_shape_restriction": {
            "late_multiplier": 1.0,
            "forgetting_rate": 0.0,
        },
        "two_phase_shape_library": "original_promoted_hpr",
        "gate": gate,
        "jobs_submitted": False,
        "candidate_manifest_sha256": (
            hashlib.sha256((args.output_dir / "candidate_manifest.csv").read_bytes()).hexdigest()
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
