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
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Measure how additional phase-varying evidence changes two-phase sample efficiency.

The independently fitted one-phase restriction is held fixed at its 280 logical
policies. Two-phase models always receive the original 280-row fit swarm and
optionally receive the 238 independently trained tied policies plus an
increasing number of phase-varying development policies. Model structure and
hyperparameters are frozen from the original Observatory fits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical_grp,
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_phase_policy_sample_efficiency_20260721"
CACHE_ROOT = SCRIPT_DIR / "reference_outputs/mixture_fit_observatory_cache_20260713/delphi_3e18"
TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
MODEL_IDS = (
    "effective_exposure",
    "separate_heads",
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_bucket_replay",
)
OPTIONAL_MODEL_IDS = ("grp",)
PLOT_MODEL_IDS = (*MODEL_IDS, *OPTIONAL_MODEL_IDS)
MODEL_LABELS = {model_id: observatory.MODEL_LABELS[model_id] for model_id in PLOT_MODEL_IDS}
EXTENSION_SERIES = (
    "delphi_3e18_frontier_phase_fiber_20260719",
    "delphi_3e18_frontier_random_phase_population_20260720",
)
EXTRA_TWO_PHASE_BUDGETS = (0, 60, 120, 240, 280, 360, 480)
DESIGNS = ("two_phase_only", "tied_spine_plus_two_phase")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_OPTIMIZER_STARTS = 12
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5


@dataclass(frozen=True)
class FrozenSpec:
    """A model's original-swarm tuning, held fixed along the learning curve."""

    model_id: str
    policy_class: str
    tuning: dict[str, Any]


@dataclass(frozen=True)
class CoordinatePool:
    """Coordinate-distinct policies and their outcomes."""

    frame: pd.DataFrame
    weights: np.ndarray


@dataclass(frozen=True)
class RawOptimum:
    """One multistart optimum of the unregularized fitted response surface."""

    weights: np.ndarray
    predicted_bpb: float
    optimizer_converged: bool
    successful_starts: int
    finite_starts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--models", default=",".join(MODEL_IDS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--budgets", default=",".join(str(value) for value in EXTRA_TWO_PHASE_BUDGETS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-only", action="store_true")
    parser.add_argument("--optima-only", action="store_true")
    parser.add_argument("--skip-optima", action="store_true")
    parser.add_argument("--optimizer-starts", type=int, default=DEFAULT_OPTIMIZER_STARTS)
    return parser.parse_args()


def parse_list(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("At least one value is required")
    return values


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in parse_list(raw))


def frozen_spec(target: str, policy_class: str, model_id: str) -> FrozenSpec:
    path = CACHE_ROOT / target / policy_class / f"{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Observatory fit {path}")
    payload = json.loads(path.read_text())
    tuning = payload["fitDetail"]["tuning"]
    return FrozenSpec(model_id=model_id, policy_class=policy_class, tuning=tuning)


def parameter_count_table(targets: tuple[str, ...], models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for target in targets:
        for model_id in models:
            row: dict[str, Any] = {"target": target, "model": model_id, "model_label": MODEL_LABELS[model_id]}
            for policy_class in (observatory.SINGLE_PHASE, observatory.TWO_PHASE):
                path = CACHE_ROOT / target / policy_class / f"{model_id}.json"
                payload = json.loads(path.read_text())
                row[f"{policy_class}_parameter_count"] = int(payload["fitDetail"]["parameterCount"])
            rows.append(row)
    return pd.DataFrame(rows)


def shape_from_tuning(tuning: dict[str, Any]) -> family_grp.Shape:
    values = tuning["shapeParameters"]
    return family_grp.Shape(
        exponent=float(values["exponent"]),
        late_multiplier=float(values["lateMultiplier"]),
        forgetting_rate=float(values["forgettingRate"]),
        penalty_threshold=float(values["penaltyThreshold"]),
        quality_discount=float(values.get("qualityDiscount", 1.0)),
    )


def fit_frozen_model(dataset: pooled.Dataset, spec: FrozenSpec) -> Any:
    indices = np.arange(dataset.n)
    if spec.model_id == "effective_exposure":
        return observatory.dsp_fit(dataset, indices, spec.model_id, spec.policy_class)
    if spec.model_id == "separate_heads":
        return observatory.separate_fit(dataset, indices, float(spec.tuning["l2"]), spec.policy_class)
    if spec.model_id == "compact_retained_state":
        return observatory.compact_fit(dataset, indices, float(spec.tuning["l2"]), spec.policy_class)
    if spec.model_id == "grp":
        return observatory.grp_300m_fit(dataset, indices, float(spec.tuning["l2"]), spec.policy_class)
    if spec.model_id == "bucket_family_grp":
        return observatory.bucket_fit(dataset, indices, shape_from_tuning(spec.tuning), float(spec.tuning["l2"]))
    if spec.model_id == "hierarchical_phase_bucket_replay":
        shape = shape_from_tuning(spec.tuning)
        config = hierarchical_grp.Config(
            variant=hierarchical_grp.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            shape_index=-1,
            shape=shape,
            l2=float(spec.tuning["l2"]),
            residual_shrink=float(spec.tuning["residualShrink"]),
            undercoverage_fraction=0.0,
            coverage_gate_ratio=0.0,
        )
        return observatory.hierarchical_phase_replay_fit(dataset, indices, config)
    raise ValueError(f"Unsupported model {spec.model_id!r}")


def predict_frozen_model(model: Any, dataset: pooled.Dataset, spec: FrozenSpec, weights: np.ndarray) -> np.ndarray:
    if spec.model_id == "effective_exposure":
        return observatory.dsp_predict(model, dataset, weights)
    if spec.model_id == "separate_heads":
        return observatory.separate_predict(model, dataset, weights, spec.policy_class)
    return np.asarray(model.predict(weights), dtype=float)


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    logits = np.log(values)
    return (logits - logits.mean(axis=1, keepdims=True)).ravel()


def logits_to_weights(logits: np.ndarray, domains: int) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=float).reshape(2, domains), axis=1)


def weighted_policy_tv(left: np.ndarray, right: np.ndarray, alpha0: float, alpha1: float) -> float:
    phase0 = 0.5 * np.abs(left[0] - right[0]).sum()
    phase1 = 0.5 * np.abs(left[1] - right[1]).sum()
    return float(alpha0 * phase0 + alpha1 * phase1)


def standardized_support_distance(dataset: pooled.Dataset, weights: np.ndarray) -> float:
    fit = dataset.weights.reshape(dataset.n, -1)
    scale = np.maximum(np.std(fit, axis=0), 1e-3)
    distances = np.linalg.norm((fit - weights.reshape(1, -1)) / scale, axis=1)
    return float(np.min(distances))


def optimum_starts(
    dataset: pooled.Dataset,
    model: Any,
    spec: FrozenSpec,
    seed: int,
    count: int,
    previous: np.ndarray | None,
) -> list[np.ndarray]:
    if count < 4:
        raise ValueError("Raw optimum audit requires at least four starts")
    alpha0, _alpha1 = observatory.phase_fractions(dataset)
    proportional = observatory.natural_weights(dataset, alpha0)
    proportional_policy = np.stack([proportional, proportional], axis=0)
    fitted_prediction = predict_frozen_model(model, dataset, spec, dataset.weights)
    starts = [weights_to_logits(proportional_policy)]
    if previous is not None:
        starts.append(weights_to_logits(previous))
    starts.extend(
        [
            weights_to_logits(dataset.weights[int(np.argmin(dataset.y))]),
            weights_to_logits(dataset.weights[int(np.argmin(fitted_prediction))]),
        ]
    )
    rng = np.random.default_rng(seed)
    concentrations = (0.25, 1.0, 4.0)
    while len(starts) < count:
        concentration = concentrations[(len(starts) - 1) % len(concentrations)]
        weights = np.stack(
            [
                rng.dirichlet(np.full(dataset.m, concentration)),
                rng.dirichlet(np.full(dataset.m, concentration)),
            ],
            axis=0,
        )
        starts.append(weights_to_logits(weights))
    return starts[:count]


def optimize_raw_model(
    dataset: pooled.Dataset,
    model: Any,
    spec: FrozenSpec,
    seed: int,
    count: int,
    previous: np.ndarray | None,
) -> RawOptimum:
    def objective(logits: np.ndarray) -> float:
        weights = logits_to_weights(logits, dataset.m)
        prediction = predict_frozen_model(model, dataset, spec, weights[None, :, :])
        return float(prediction[0])

    candidates: list[tuple[float, np.ndarray, bool]] = []
    for start in optimum_starts(dataset, model, spec, seed, count, previous):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            candidates.append((float(result.fun), np.asarray(result.x, dtype=float), bool(result.success)))
    if not candidates:
        raise RuntimeError(f"No finite optimum for {dataset.name}/{spec.model_id}")
    best = min(candidates, key=lambda candidate: candidate[0])
    return RawOptimum(
        weights=logits_to_weights(best[1], dataset.m),
        predicted_bpb=best[0],
        optimizer_converged=best[2],
        successful_starts=sum(candidate[2] for candidate in candidates),
        finite_starts=len(candidates),
    )


def coordinate_pool(frame: pd.DataFrame, weights: np.ndarray) -> CoordinatePool:
    """Collapse repeat observations without changing policy-coordinate weight."""
    rows: list[dict[str, Any]] = []
    coordinates: list[np.ndarray] = []
    for coordinate, raw_indices in frame.groupby("mixture_sha256", sort=True).indices.items():
        indices = np.asarray(raw_indices, dtype=int)
        local_weights = weights[indices]
        if not np.allclose(local_weights, local_weights[0], atol=1e-12):
            raise ValueError(f"Coordinate {coordinate} has inconsistent weights")
        local = frame.iloc[indices]
        row = local.iloc[0].to_dict()
        row["mixture_sha256"] = str(coordinate)
        row["repeat_count"] = len(local)
        row["training_series"] = ";".join(sorted(set(local["training_series"].astype(str))))
        for column in TARGET_COLUMNS.values():
            row[column] = float(local[column].mean())
        rows.append(row)
        coordinates.append(local_weights[0])
    return CoordinatePool(pd.DataFrame(rows).reset_index(drop=True), np.asarray(coordinates, dtype=float))


def target_dataset(
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


def combined_dataset(
    reference: pooled.Dataset,
    pools: tuple[tuple[pd.DataFrame, np.ndarray], ...],
    target: str,
    name: str,
) -> pooled.Dataset:
    frame = pd.concat([pool[0] for pool in pools], ignore_index=True, sort=False)
    weights = np.concatenate([pool[1] for pool in pools], axis=0)
    return target_dataset(reference, frame, weights, target, name)


def collapsed_weights(weights: np.ndarray, alpha0: float) -> np.ndarray:
    aggregate = alpha0 * weights[:, 0, :] + (1.0 - alpha0) * weights[:, 1, :]
    return np.stack([aggregate, aggregate], axis=1)


def heldout_pools(reference: pooled.Dataset) -> tuple[CoordinatePool, CoordinatePool]:
    frame, weights = observatory.load_delphi_3e18_heldouts(reference)
    mask = (
        frame["policy_class"].eq("two_phase")
        & frame["fit_panel_overlap"].eq("coordinate_disjoint")
        & frame["training_state"].eq("finished")
        & frame["checkpoint_declared_complete"].eq(1)
    )
    phase_frame = frame.loc[mask].reset_index(drop=True)
    phase_weights = weights[mask.to_numpy()]
    extension_mask = phase_frame["training_series"].isin(EXTENSION_SERIES).to_numpy()
    extension = coordinate_pool(phase_frame.loc[extension_mask].reset_index(drop=True), phase_weights[extension_mask])
    evaluation = coordinate_pool(
        phase_frame.loc[~extension_mask].reset_index(drop=True),
        phase_weights[~extension_mask],
    )
    if len(extension.frame) != 480:
        raise ValueError(f"Expected 480 extension coordinates, found {len(extension.frame)}")
    if len(evaluation.frame) < 400:
        raise ValueError(f"Expected at least 400 evaluation coordinates, found {len(evaluation.frame)}")
    return extension, evaluation


def tied_independent_pool(single: pooled.Dataset) -> tuple[pd.DataFrame, np.ndarray]:
    mask = single.frame["disposition"].eq("scheduled_new_training").to_numpy()
    if int(mask.sum()) != 238:
        raise ValueError(f"Expected 238 independent tied rows, found {int(mask.sum())}")
    return single.frame.loc[mask].reset_index(drop=True), single.weights[mask]


def largest_remainder_quotas(counts: pd.Series, total: int) -> dict[str, int]:
    ideal = total * counts / counts.sum()
    quotas = np.floor(ideal).astype(int)
    remainder = total - int(quotas.sum())
    for key in (ideal - quotas).sort_values(ascending=False).index[:remainder]:
        quotas.loc[key] += 1
    return {str(key): int(value) for key, value in quotas.items()}


def extension_order(pool: CoordinatePool, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for series, indices in pool.frame.groupby("training_series", sort=True).indices.items():
        result[str(series)] = rng.permutation(np.asarray(indices, dtype=int))
    return result


def extension_indices(pool: CoordinatePool, budget: int, seed: int) -> np.ndarray:
    if budget < 0 or budget > len(pool.frame):
        raise ValueError(f"Invalid extension budget {budget}")
    if budget == 0:
        return np.asarray([], dtype=int)
    counts = pool.frame["training_series"].value_counts(sort=False)
    quotas = largest_remainder_quotas(counts, budget)
    order = extension_order(pool, seed)
    selected = np.concatenate([order[series][:quota] for series, quota in quotas.items()])
    if len(selected) != budget:
        raise AssertionError(f"Selected {len(selected)} extension rows for budget {budget}")
    return np.sort(selected)


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.std(observed) < 1e-12 or np.std(predicted) < 1e-12:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[: min(k, len(predicted))]
    return float(np.min(observed[selected]) - np.min(observed))


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = predicted - observed
    optimism = observed - predicted
    centered_prediction = predicted - predicted.mean()
    denominator = float(centered_prediction @ centered_prediction)
    slope = (
        float(centered_prediction @ (observed - observed.mean()) / denominator) if denominator > 1e-15 else float("nan")
    )
    tail_count = min(len(observed), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))))
    tail = np.argsort(predicted)[:tail_count]
    selected = int(np.argmin(predicted))
    return {
        "n_eval": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "normalized_rmse": float(np.sqrt(np.mean(residual**2)) / np.std(observed, ddof=1)),
        "spearman": safe_spearman(observed, predicted),
        "bias": float(np.mean(residual)),
        "calibration_slope": slope,
        "calibration_error": abs(slope - 1.0),
        "regret_at_1": regret_at_k(observed, predicted, 1),
        "regret_at_3": regret_at_k(observed, predicted, 3),
        "regret_at_5": regret_at_k(observed, predicted, 5),
        "lower_tail_optimism": float(np.mean(np.maximum(optimism[tail], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        "optimism_gt_0p05": int(np.sum(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
        "selected_optimism": float(optimism[selected]),
        "selected_observed": float(observed[selected]),
        "selected_predicted": float(predicted[selected]),
        "frontier_observed": float(np.min(observed)),
    }


def protocol_payload(
    targets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "version": 1,
        "targets": list(targets),
        "models": list(models),
        "seeds": list(seeds),
        "extra_two_phase_budgets": list(budgets),
        "one_phase_logical_rows": 280,
        "two_phase_base_rows": 280,
        "independent_tied_rows_added_to_two_phase_fit": 238,
        "shared_tied_aliases_deduplicated": 42,
        "extension_series": list(EXTENSION_SERIES),
        "evaluation_rule": "all other completed coordinate-disjoint two-phase development policies",
        "hyperparameter_rule": "freeze Observatory tuning selected on the original 280-row policy-matched fit",
        "primary_comparison": ["fixed_one_phase_aggregate", *DESIGNS],
        "data_status": "exposed development diagnostic; not confirmatory",
    }


def load_existing(output_dir: Path, protocol: dict[str, Any], overwrite: bool) -> pd.DataFrame:
    protocol_path = output_dir / "protocol.json"
    metrics_path = output_dir / "learning_curve_runs.csv"
    if overwrite:
        return pd.DataFrame()
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing protocol differs; use a new output directory or --overwrite")
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["target"], row["model"], row["design"], int(row["extra_two_phase_rows"]), int(row["seed"])


def persist_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).sort_values(["target", "model", "design", "extra_two_phase_rows", "seed"]).to_csv(
        output_dir / "learning_curve_runs.csv", index=False
    )


def append_metrics(
    rows: list[dict[str, Any]],
    completed: set[tuple[Any, ...]],
    base: dict[str, Any],
    observed: np.ndarray,
    predicted: np.ndarray,
    output_dir: Path,
) -> None:
    key = row_key(base)
    if key in completed:
        return
    rows.append({**base, **metrics(observed, predicted)})
    completed.add(key)
    persist_rows(output_dir, rows)


def run_target(
    target: str,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    extension: CoordinatePool,
    evaluation: CoordinatePool,
    rows: list[dict[str, Any]],
    completed: set[tuple[Any, ...]],
    output_dir: Path,
) -> None:
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, _single_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    target_column = TARGET_COLUMNS[target]
    base_frame = reference.frame.copy()
    base_frame[target_column] = reference.y
    single.frame[target_column] = single.y
    tied_frame, tied_weights = tied_independent_pool(single)
    evaluation_dataset = target_dataset(
        reference,
        evaluation.frame,
        evaluation.weights,
        target,
        f"delphi_3e18_learning_eval_{target}",
    )
    alpha0, _alpha1 = observatory.phase_fractions(reference)
    aggregate_eval_weights = collapsed_weights(evaluation.weights, alpha0)

    for model_id in models:
        one_spec = frozen_spec(target, observatory.SINGLE_PHASE, model_id)
        one_bases = [
            {
                "target": target,
                "model": model_id,
                "model_label": MODEL_LABELS[model_id],
                "design": "fixed_one_phase_aggregate",
                "extra_two_phase_rows": budget,
                "two_phase_rows": 0,
                "tied_rows": 280,
                "total_unique_training_rows": 280,
                "seed": seed,
            }
            for seed in seeds
            for budget in budgets
        ]
        missing_one_bases = [base for base in one_bases if row_key(base) not in completed]
        if missing_one_bases:
            one_model = fit_frozen_model(single, one_spec)
            one_prediction = predict_frozen_model(one_model, single, one_spec, aggregate_eval_weights)
        for base in missing_one_bases:
            append_metrics(
                rows,
                completed,
                base,
                evaluation_dataset.y,
                one_prediction,
                output_dir,
            )

        two_spec = frozen_spec(target, observatory.TWO_PHASE, model_id)
        for seed in seeds:
            for budget in budgets:
                indices = extension_indices(extension, budget, seed)
                extra_frame = extension.frame.iloc[indices].reset_index(drop=True)
                extra_weights = extension.weights[indices]
                for design in DESIGNS:
                    tied_count = 0
                    if design == "tied_spine_plus_two_phase":
                        tied_count = len(tied_frame)
                    base = {
                        "target": target,
                        "model": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "design": design,
                        "extra_two_phase_rows": budget,
                        "two_phase_rows": 280 + budget,
                        "tied_rows": tied_count,
                        "total_unique_training_rows": 280 + tied_count + budget,
                        "seed": seed,
                    }
                    if row_key(base) in completed:
                        continue
                    pools: list[tuple[pd.DataFrame, np.ndarray]] = [(base_frame, reference.weights)]
                    if design == "tied_spine_plus_two_phase":
                        pools.append((tied_frame, tied_weights))
                    if budget:
                        pools.append((extra_frame, extra_weights))
                    train = combined_dataset(
                        reference,
                        tuple(pools),
                        target,
                        f"delphi_3e18_learning_{target}_{design}_{280 + budget}",
                    )
                    model = fit_frozen_model(train, two_spec)
                    prediction = predict_frozen_model(model, train, two_spec, evaluation.weights)
                    append_metrics(
                        rows,
                        completed,
                        base,
                        evaluation_dataset.y,
                        prediction,
                        output_dir,
                    )


def optimum_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["target"],
        row["model"],
        row["design"],
        int(row["extra_two_phase_rows"]),
        int(row["seed"]),
    )


def persist_optimum_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).sort_values(["target", "model", "design", "seed", "extra_two_phase_rows"]).to_csv(
        output_dir / "raw_optimum_runs.csv", index=False
    )


def minimum_policy_tv(
    policies: np.ndarray,
    weights: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> tuple[int, float]:
    phase0 = 0.5 * np.abs(policies[:, 0, :] - weights[None, 0, :]).sum(axis=1)
    phase1 = 0.5 * np.abs(policies[:, 1, :] - weights[None, 1, :]).sum(axis=1)
    distances = alpha0 * phase0 + alpha1 * phase1
    index = int(np.argmin(distances))
    return index, float(distances[index])


def categorical_kl(left: np.ndarray, right: np.ndarray) -> float:
    safe_left = np.clip(np.asarray(left, dtype=float), 1e-12, 1.0)
    safe_right = np.clip(np.asarray(right, dtype=float), 1e-12, 1.0)
    return float(np.sum(safe_left * (np.log(safe_left) - np.log(safe_right))))


def raw_optimum_record(
    optimum: RawOptimum,
    train: pooled.Dataset,
    evaluation: CoordinatePool,
    target: str,
    model_id: str,
    design: str,
    budget: int,
    tied_count: int,
    seed: int,
    previous: np.ndarray | None,
) -> dict[str, Any]:
    alpha0, alpha1 = observatory.phase_fractions(train)
    proportional = observatory.natural_weights(train, alpha0)
    aggregate = alpha0 * optimum.weights[0] + alpha1 * optimum.weights[1]
    exposure = optimum.weights[0] * train.c0 + optimum.weights[1] * train.c1
    fit_exposure = train.weights[:, 0, :] * train.c0[None, :] + train.weights[:, 1, :] * train.c1[None, :]
    fit_index, fit_tv = minimum_policy_tv(train.weights, optimum.weights, alpha0, alpha1)
    eval_index, eval_tv = minimum_policy_tv(evaluation.weights, optimum.weights, alpha0, alpha1)
    phase_information = alpha0 * categorical_kl(optimum.weights[0], aggregate)
    phase_information += alpha1 * categorical_kl(optimum.weights[1], aggregate)
    return {
        "target": target,
        "model": model_id,
        "model_label": MODEL_LABELS[model_id],
        "design": design,
        "extra_two_phase_rows": budget,
        "two_phase_rows": 280 + budget,
        "tied_rows": tied_count,
        "total_unique_training_rows": 280 + tied_count + budget,
        "seed": seed,
        "predicted_bpb": optimum.predicted_bpb,
        "optimizer_converged": optimum.optimizer_converged,
        "successful_starts": optimum.successful_starts,
        "finite_starts": optimum.finite_starts,
        "predicted_gain_vs_best_fit_observed": float(np.min(train.y) - optimum.predicted_bpb),
        "max_bucket_weight": float(optimum.weights.max()),
        "max_simulated_epochs": float(exposure.max()),
        "max_epoch_ratio_to_fit": float(exposure.max() / np.maximum(fit_exposure.max(), 1e-12)),
        "phase_total_variation": float(0.5 * np.abs(optimum.weights[0] - optimum.weights[1]).sum()),
        "phase_information_kl": phase_information,
        "aggregate_hhi": float(np.square(aggregate).sum()),
        "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - proportional).sum()),
        "standardized_fit_support_distance": standardized_support_distance(train, optimum.weights),
        "nearest_fit_policy_tv": fit_tv,
        "nearest_fit_observed": float(train.y[fit_index]),
        "nearest_fit_row": str(train.frame.iloc[fit_index].get("run_name", fit_index)),
        "nearest_evaluation_policy_tv": eval_tv,
        "nearest_evaluation_observed": float(evaluation.frame.iloc[eval_index][TARGET_COLUMNS[target]]),
        "nearest_evaluation_row": str(evaluation.frame.iloc[eval_index].get("run_name", eval_index)),
        "successive_optimum_tv": (
            weighted_policy_tv(previous, optimum.weights, alpha0, alpha1) if previous is not None else float("nan")
        ),
        "phase_0_weights_json": json.dumps(optimum.weights[0].tolist(), separators=(",", ":")),
        "phase_1_weights_json": json.dumps(optimum.weights[1].tolist(), separators=(",", ":")),
    }


def run_raw_optimum_target(
    target: str,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    extension: CoordinatePool,
    evaluation: CoordinatePool,
    rows: list[dict[str, Any]],
    completed: set[tuple[Any, ...]],
    output_dir: Path,
    optimizer_starts: int,
) -> None:
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, _single_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    target_column = TARGET_COLUMNS[target]
    base_frame = reference.frame.copy()
    base_frame[target_column] = reference.y
    single.frame[target_column] = single.y
    tied_frame, tied_weights = tied_independent_pool(single)

    for model_id in models:
        spec = frozen_spec(target, observatory.TWO_PHASE, model_id)
        for design in DESIGNS:
            tied_count = len(tied_frame) if design == "tied_spine_plus_two_phase" else 0
            for seed in seeds:
                previous: np.ndarray | None = None
                for budget in sorted(budgets):
                    key = (target, model_id, design, budget, seed)
                    existing = next((row for row in rows if optimum_row_key(row) == key), None)
                    if existing is not None:
                        previous = np.stack(
                            [
                                np.asarray(json.loads(existing["phase_0_weights_json"]), dtype=float),
                                np.asarray(json.loads(existing["phase_1_weights_json"]), dtype=float),
                            ],
                            axis=0,
                        )
                        continue
                    indices = extension_indices(extension, budget, seed)
                    pools: list[tuple[pd.DataFrame, np.ndarray]] = [(base_frame, reference.weights)]
                    if design == "tied_spine_plus_two_phase":
                        pools.append((tied_frame, tied_weights))
                    if budget:
                        pools.append(
                            (
                                extension.frame.iloc[indices].reset_index(drop=True),
                                extension.weights[indices],
                            )
                        )
                    train = combined_dataset(
                        reference,
                        tuple(pools),
                        target,
                        f"delphi_3e18_optimum_{target}_{design}_{280 + budget}",
                    )
                    model = fit_frozen_model(train, spec)
                    optimum = optimize_raw_model(
                        train,
                        model,
                        spec,
                        seed=seed * 10_000 + budget,
                        count=optimizer_starts,
                        previous=previous,
                    )
                    row = raw_optimum_record(
                        optimum,
                        train,
                        evaluation,
                        target,
                        model_id,
                        design,
                        budget,
                        tied_count,
                        seed,
                        previous,
                    )
                    rows.append(row)
                    completed.add(key)
                    previous = optimum.weights
                    persist_optimum_rows(output_dir, rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics_columns = [
        "rmse",
        "normalized_rmse",
        "spearman",
        "bias",
        "calibration_slope",
        "calibration_error",
        "regret_at_1",
        "regret_at_3",
        "regret_at_5",
        "lower_tail_optimism",
        "low_tail_rmse",
        "optimism_gt_0p05",
        "worst_optimism",
        "selected_optimism",
        "selected_observed",
        "selected_predicted",
        "frontier_observed",
    ]
    grouping = [
        "target",
        "model",
        "model_label",
        "design",
        "extra_two_phase_rows",
        "two_phase_rows",
        "tied_rows",
        "total_unique_training_rows",
        "n_eval",
    ]
    summary = runs.groupby(grouping, as_index=False)[metrics_columns].agg(["mean", "std"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns.to_flat_index()]
    return summary


def plot_metric(summary: pd.DataFrame, metric: str, label: str, output_dir: Path) -> None:
    targets = list(dict.fromkeys(summary["target"].astype(str)))
    figure = make_subplots(rows=1, cols=len(targets), subplot_titles=[target.title() for target in targets])
    colors = dict(
        zip(
            PLOT_MODEL_IDS,
            [
                "#a50026",
                "#d73027",
                "#f46d43",
                "#fdae61",
                "#66bd63",
                "#006837",
            ],
            strict=True,
        )
    )
    dashes = {"two_phase_only": "solid", "tied_spine_plus_two_phase": "dash", "fixed_one_phase_aggregate": "dot"}
    for column, target in enumerate(targets, start=1):
        target_frame = summary.loc[summary["target"].eq(target)]
        for model_id in PLOT_MODEL_IDS:
            model_frame = target_frame.loc[target_frame["model"].eq(model_id)]
            for design in ("fixed_one_phase_aggregate", *DESIGNS):
                local = model_frame.loc[model_frame["design"].eq(design)].sort_values("two_phase_rows")
                if local.empty:
                    continue
                x = local["two_phase_rows"].to_numpy(dtype=float)
                if design == "fixed_one_phase_aggregate":
                    x = 280 + local["extra_two_phase_rows"].to_numpy(dtype=float)
                mean = local[f"{metric}_mean"].to_numpy(dtype=float)
                std = local[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
                name = f"{MODEL_LABELS[model_id]} · {design.replace('_', ' ')}"
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean,
                        mode="lines+markers",
                        name=name,
                        legendgroup=name,
                        showlegend=column == 1,
                        line={"color": colors[model_id], "dash": dashes[design]},
                        error_y={"type": "data", "array": std, "visible": design != "fixed_one_phase_aggregate"},
                        customdata=np.column_stack(
                            [local["total_unique_training_rows"], local["tied_rows"], local["extra_two_phase_rows"]]
                        ),
                        hovertemplate=(
                            "%{fullData.name}<br>base/extension rows=%{x:.0f}<br>"
                            + f"{label}=%{{y:.5f}}<br>"
                            + "total unique fit rows=%{customdata[0]:.0f}<br>"
                            + "additional independent tied rows=%{customdata[1]:.0f}<br>"
                            + "added phase-varying rows=%{customdata[2]:.0f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=column,
                )
        figure.update_xaxes(title_text="Original two-phase fit rows + added phase-varying rows", row=1, col=column)
        figure.update_yaxes(title_text=label if column == 1 else None, row=1, col=column)
    figure.update_layout(
        template="plotly_white",
        title=f"Fixed 280-row aggregate fit versus expanding two-phase evidence: {label}",
        width=1500,
        height=650,
        legend={"orientation": "h", "y": -0.24},
        margin={"b": 190},
    )
    figure.write_html(output_dir / f"learning_curve_{metric}.html", include_plotlyjs=True, config=PLOT_CONFIG)


def add_optimum_endpoint_distances(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["tv_to_endpoint_optimum"] = np.nan
    for (_target, _model, _design, _seed), indices in result.groupby(
        ["target", "model", "design", "seed"], sort=False
    ).indices.items():
        local_indices = np.asarray(indices, dtype=int)
        local = result.iloc[local_indices]
        endpoint = local.loc[local["extra_two_phase_rows"].eq(local["extra_two_phase_rows"].max())].iloc[0]
        endpoint_weights = np.stack(
            [
                np.asarray(json.loads(endpoint["phase_0_weights_json"]), dtype=float),
                np.asarray(json.loads(endpoint["phase_1_weights_json"]), dtype=float),
            ],
            axis=0,
        )
        reference = observatory.load_delphi_3e18_fit_dataset(str(endpoint["target"]))
        alpha0, alpha1 = observatory.phase_fractions(reference)
        for index in local_indices:
            row = result.iloc[index]
            weights = np.stack(
                [
                    np.asarray(json.loads(row["phase_0_weights_json"]), dtype=float),
                    np.asarray(json.loads(row["phase_1_weights_json"]), dtype=float),
                ],
                axis=0,
            )
            result.loc[result.index[index], "tv_to_endpoint_optimum"] = weighted_policy_tv(
                weights, endpoint_weights, alpha0, alpha1
            )
    return result


def summarize_optima(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "predicted_bpb",
        "predicted_gain_vs_best_fit_observed",
        "max_bucket_weight",
        "max_simulated_epochs",
        "max_epoch_ratio_to_fit",
        "phase_total_variation",
        "phase_information_kl",
        "aggregate_hhi",
        "aggregate_tv_to_proportional",
        "standardized_fit_support_distance",
        "nearest_fit_policy_tv",
        "nearest_fit_observed",
        "nearest_evaluation_policy_tv",
        "nearest_evaluation_observed",
        "successive_optimum_tv",
        "tv_to_endpoint_optimum",
        "successful_starts",
        "finite_starts",
    ]
    grouping = [
        "target",
        "model",
        "model_label",
        "design",
        "extra_two_phase_rows",
        "two_phase_rows",
        "tied_rows",
        "total_unique_training_rows",
    ]
    summary = frame.groupby(grouping, as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns.to_flat_index()]
    return summary


def plot_optimum_metric(summary: pd.DataFrame, metric: str, label: str, output_dir: Path) -> None:
    targets = list(dict.fromkeys(summary["target"].astype(str)))
    figure = make_subplots(rows=1, cols=len(targets), subplot_titles=[target.title() for target in targets])
    colors = dict(
        zip(
            PLOT_MODEL_IDS,
            ["#a50026", "#d73027", "#f46d43", "#fdae61", "#66bd63", "#006837"],
            strict=True,
        )
    )
    dashes = {"two_phase_only": "solid", "tied_spine_plus_two_phase": "dash"}
    for column, target in enumerate(targets, start=1):
        target_frame = summary.loc[summary["target"].eq(target)]
        for model_id in PLOT_MODEL_IDS:
            for design in DESIGNS:
                local = target_frame.loc[
                    target_frame["model"].eq(model_id) & target_frame["design"].eq(design)
                ].sort_values("two_phase_rows")
                if local.empty:
                    continue
                name = f"{MODEL_LABELS[model_id]} · {design.replace('_', ' ')}"
                figure.add_trace(
                    go.Scatter(
                        x=local["two_phase_rows"],
                        y=local[f"{metric}_mean"],
                        mode="lines+markers",
                        name=name,
                        legendgroup=name,
                        showlegend=column == 1,
                        line={"color": colors[model_id], "dash": dashes[design]},
                        error_y={
                            "type": "data",
                            "array": local[f"{metric}_std"].fillna(0.0),
                            "visible": True,
                        },
                        customdata=np.column_stack([local["total_unique_training_rows"], local["tied_rows"]]),
                        hovertemplate=(
                            "%{fullData.name}<br>base/extension rows=%{x:.0f}<br>"
                            + f"{label}=%{{y:.5f}}<br>"
                            + "total unique fit rows=%{customdata[0]:.0f}<br>"
                            + "additional tied rows=%{customdata[1]:.0f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=column,
                )
        figure.update_xaxes(title_text="Original two-phase fit rows + added phase-varying rows", row=1, col=column)
        figure.update_yaxes(title_text=label if column == 1 else None, row=1, col=column)
    figure.update_layout(
        template="plotly_white",
        title=f"Unregularized two-phase optimum path: {label}",
        width=1500,
        height=650,
        legend={"orientation": "h", "y": -0.24},
        margin={"b": 190},
    )
    figure.write_html(output_dir / f"raw_optimum_{metric}.html", include_plotlyjs=True, config=PLOT_CONFIG)


def optimum_endpoint_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    max_budget = int(frame["extra_two_phase_rows"].max())
    for (target, model, design), local in frame.groupby(["target", "model", "design"], sort=True):
        endpoint = local.loc[local["extra_two_phase_rows"].eq(max_budget)]
        prior = local.loc[local["extra_two_phase_rows"].eq(360)]
        if prior.empty:
            prior = local.loc[local["extra_two_phase_rows"].lt(max_budget)]
            prior = prior.loc[prior["extra_two_phase_rows"].eq(prior["extra_two_phase_rows"].max())]
        endpoint_vectors = [
            np.stack(
                [
                    np.asarray(json.loads(row.phase_0_weights_json), dtype=float),
                    np.asarray(json.loads(row.phase_1_weights_json), dtype=float),
                ],
                axis=0,
            )
            for row in endpoint.itertuples(index=False)
        ]
        reference = observatory.load_delphi_3e18_fit_dataset(str(target))
        alpha0, alpha1 = observatory.phase_fractions(reference)
        pairwise = [
            weighted_policy_tv(endpoint_vectors[left], endpoint_vectors[right], alpha0, alpha1)
            for left in range(len(endpoint_vectors))
            for right in range(left + 1, len(endpoint_vectors))
        ]
        rows.append(
            {
                "target": target,
                "model": model,
                "model_label": MODEL_LABELS[str(model)],
                "design": design,
                "endpoint_predicted_bpb": float(endpoint["predicted_bpb"].mean()),
                "endpoint_predicted_gain_vs_fit_frontier": float(endpoint["predicted_gain_vs_best_fit_observed"].mean()),
                "endpoint_pairwise_seed_tv": float(np.mean(pairwise)) if pairwise else 0.0,
                "prior_to_endpoint_tv": float(prior["tv_to_endpoint_optimum"].mean()),
                "endpoint_max_weight": float(endpoint["max_bucket_weight"].mean()),
                "endpoint_max_epochs": float(endpoint["max_simulated_epochs"].mean()),
                "endpoint_phase_tv": float(endpoint["phase_total_variation"].mean()),
                "endpoint_fit_support_distance": float(endpoint["standardized_fit_support_distance"].mean()),
                "endpoint_nearest_fit_tv": float(endpoint["nearest_fit_policy_tv"].mean()),
                "endpoint_nearest_eval_tv": float(endpoint["nearest_evaluation_policy_tv"].mean()),
                "endpoint_nearest_eval_observed": float(endpoint["nearest_evaluation_observed"].mean()),
                "all_optimizer_runs_converged": bool(local["optimizer_converged"].all()),
            }
        )
    return pd.DataFrame(rows)


def write_optimum_report(endpoints: pd.DataFrame, output_dir: Path) -> None:
    prediction_winners = (
        endpoints.sort_values(["target", "endpoint_nearest_eval_observed", "endpoint_nearest_eval_tv"])
        .groupby("target", as_index=False)
        .first()
    )
    most_stable = (
        endpoints.sort_values(["target", "prior_to_endpoint_tv", "endpoint_pairwise_seed_tv"])
        .groupby("target", as_index=False)
        .first()
    )
    report = "\n".join(
        [
            "# Unregularized optimum convergence audit",
            "",
            "## Interpretation",
            "",
            (
                "Each frozen model form and hyperparameter setting is refit at every evidence budget and optimized "
                "without a deployment KL or trust region. The audit uses common multistarts plus a warm start from "
                "the preceding budget. Stability therefore concerns the learned raw response surface rather than a "
                "regularizer forcing nearby policies."
            ),
            "",
            (
                "`prior_to_endpoint_tv` is the weighted policy TV between the 640-row and 760-row optima. "
                "`endpoint_pairwise_seed_tv` measures optimizer/data-order stability at the complete 760-row endpoint. "
                "A stable optimum can still be implausible; concentration, repetition, support distance, and the "
                "nearest observed evaluation policy must be inspected jointly."
            ),
            "",
            "## Endpoint diagnostics",
            "",
            endpoints.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Best nearest observed endpoint policy",
            "",
            prediction_winners[
                [
                    "target",
                    "model_label",
                    "design",
                    "endpoint_nearest_eval_observed",
                    "endpoint_nearest_eval_tv",
                    "endpoint_predicted_bpb",
                ]
            ].to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Most stable endpoint path",
            "",
            most_stable[
                [
                    "target",
                    "model_label",
                    "design",
                    "prior_to_endpoint_tv",
                    "endpoint_pairwise_seed_tv",
                    "endpoint_fit_support_distance",
                ]
            ].to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Guardrail",
            "",
            (
                "Nearest-policy outcomes are diagnostics, not validation of the continuous optimum. No raw optimum "
                "is called healthy solely because its path stabilizes or its nearest archived policy is strong."
            ),
        ]
    )
    (output_dir / "raw_optimum_report.md").write_text(report + "\n")


def selected_archive_policy_paths(
    runs: pd.DataFrame,
    extension: CoordinatePool,
    evaluation: CoordinatePool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_runs = runs.loc[runs["target"].eq(target)]
        if target_runs.empty:
            continue
        reference = observatory.load_delphi_3e18_fit_dataset(target)
        heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
        single, _single_indices = observatory.load_delphi_3e18_single_phase_dataset(
            target,
            reference,
            heldout_frame,
            heldout_weights,
        )
        target_column = TARGET_COLUMNS[target]
        base_frame = reference.frame.copy()
        base_frame[target_column] = reference.y
        single.frame[target_column] = single.y
        _tied_frame, tied_weights = tied_independent_pool(single)
        observed = evaluation.frame[target_column].to_numpy(dtype=float)
        alpha0, alpha1 = observatory.phase_fractions(reference)
        proportional = observatory.natural_weights(reference, alpha0)
        observed_to_index = {float(value): index for index, value in enumerate(observed)}
        if len(observed_to_index) != len(observed):
            raise ValueError(f"Selected-policy reconstruction requires unique {target_column} values")
        for run in target_runs.itertuples(index=False):
            selected_index = observed_to_index[float(run.selected_observed)]
            selected_weights = evaluation.weights[selected_index]
            if run.design == "fixed_one_phase_aggregate":
                aggregate = alpha0 * selected_weights[0] + alpha1 * selected_weights[1]
                policy_weights = np.stack([aggregate, aggregate], axis=0)
                train_weights = single.weights
            else:
                policy_weights = selected_weights
                indices = extension_indices(extension, int(run.extra_two_phase_rows), int(run.seed))
                train_policies = [reference.weights]
                if run.design == "tied_spine_plus_two_phase":
                    train_policies.append(tied_weights)
                if len(indices):
                    train_policies.append(extension.weights[indices])
                train_weights = np.concatenate(train_policies, axis=0)
            aggregate = alpha0 * policy_weights[0] + alpha1 * policy_weights[1]
            exposure = policy_weights[0] * reference.c0 + policy_weights[1] * reference.c1
            fit_dataset = pooled.Dataset(
                name=f"selected_path_{target}_{run.design}",
                frame=pd.DataFrame(index=np.arange(len(train_weights))),
                y=np.zeros(len(train_weights), dtype=float),
                weights=train_weights,
                c0=reference.c0,
                c1=reference.c1,
                domain_names=reference.domain_names,
            )
            fit_index, fit_tv = minimum_policy_tv(train_weights, policy_weights, alpha0, alpha1)
            selected_row = evaluation.frame.iloc[selected_index]
            rows.append(
                {
                    "target": target,
                    "model": run.model,
                    "model_label": run.model_label,
                    "design": run.design,
                    "extra_two_phase_rows": int(run.extra_two_phase_rows),
                    "two_phase_rows": int(run.two_phase_rows),
                    "tied_rows": int(run.tied_rows),
                    "total_unique_training_rows": int(run.total_unique_training_rows),
                    "seed": int(run.seed),
                    "selected_coordinate": str(selected_row["mixture_sha256"]),
                    "selected_row": str(selected_row.get("run_name", selected_index)),
                    "selected_series": str(selected_row["training_series"]),
                    "selected_observed": float(run.selected_observed),
                    "selected_predicted": float(run.selected_predicted),
                    "regret_at_1": float(run.regret_at_1),
                    "max_bucket_weight": float(policy_weights.max()),
                    "max_simulated_epochs": float(exposure.max()),
                    "phase_total_variation": float(0.5 * np.abs(policy_weights[0] - policy_weights[1]).sum()),
                    "phase_information_kl": (
                        alpha0 * categorical_kl(policy_weights[0], aggregate)
                        + alpha1 * categorical_kl(policy_weights[1], aggregate)
                    ),
                    "aggregate_hhi": float(np.square(aggregate).sum()),
                    "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - proportional).sum()),
                    "standardized_fit_support_distance": standardized_support_distance(fit_dataset, policy_weights),
                    "nearest_fit_policy_tv": fit_tv,
                    "nearest_fit_index": fit_index,
                    "phase_0_weights_json": json.dumps(policy_weights[0].tolist(), separators=(",", ":")),
                    "phase_1_weights_json": json.dumps(policy_weights[1].tolist(), separators=(",", ":")),
                }
            )
    result = pd.DataFrame(rows)
    result["successive_selected_policy_tv"] = np.nan
    result["tv_to_endpoint_selected_policy"] = np.nan
    for (_target, _model, _design, _seed), indices in result.groupby(
        ["target", "model", "design", "seed"], sort=False
    ).indices.items():
        ordered_indices = sorted(indices, key=lambda index: int(result.iloc[index]["extra_two_phase_rows"]))
        endpoint_row = result.iloc[ordered_indices[-1]]
        endpoint_weights = np.stack(
            [
                np.asarray(json.loads(endpoint_row["phase_0_weights_json"]), dtype=float),
                np.asarray(json.loads(endpoint_row["phase_1_weights_json"]), dtype=float),
            ],
            axis=0,
        )
        reference = observatory.load_delphi_3e18_fit_dataset(str(endpoint_row["target"]))
        alpha0, alpha1 = observatory.phase_fractions(reference)
        previous: np.ndarray | None = None
        for index in ordered_indices:
            row = result.iloc[index]
            weights = np.stack(
                [
                    np.asarray(json.loads(row["phase_0_weights_json"]), dtype=float),
                    np.asarray(json.loads(row["phase_1_weights_json"]), dtype=float),
                ],
                axis=0,
            )
            if previous is not None:
                result.loc[result.index[index], "successive_selected_policy_tv"] = weighted_policy_tv(
                    previous, weights, alpha0, alpha1
                )
            result.loc[result.index[index], "tv_to_endpoint_selected_policy"] = weighted_policy_tv(
                weights, endpoint_weights, alpha0, alpha1
            )
            previous = weights
    return result


def selected_policy_endpoint_table(paths: pd.DataFrame) -> pd.DataFrame:
    max_budget = int(paths["extra_two_phase_rows"].max())
    rows: list[dict[str, Any]] = []
    for (target, model, design), local in paths.groupby(["target", "model", "design"], sort=True):
        endpoint = local.loc[local["extra_two_phase_rows"].eq(max_budget)]
        late = local.loc[local["extra_two_phase_rows"].ge(280)]
        rows.append(
            {
                "target": target,
                "model": model,
                "model_label": MODEL_LABELS[str(model)],
                "design": design,
                "endpoint_selected_observed": float(endpoint["selected_observed"].mean()),
                "endpoint_selected_predicted": float(endpoint["selected_predicted"].mean()),
                "endpoint_regret_at_1": float(endpoint["regret_at_1"].mean()),
                "endpoint_max_weight": float(endpoint["max_bucket_weight"].mean()),
                "endpoint_max_epochs": float(endpoint["max_simulated_epochs"].mean()),
                "endpoint_phase_tv": float(endpoint["phase_total_variation"].mean()),
                "endpoint_aggregate_tv_to_proportional": float(endpoint["aggregate_tv_to_proportional"].mean()),
                "endpoint_fit_support_distance": float(endpoint["standardized_fit_support_distance"].mean()),
                "endpoint_nearest_fit_tv": float(endpoint["nearest_fit_policy_tv"].mean()),
                "late_distinct_selected_policies": int(late["selected_coordinate"].nunique()),
                "late_max_tv_to_endpoint": float(late["tv_to_endpoint_selected_policy"].max()),
                "late_mean_tv_to_endpoint": float(late["tv_to_endpoint_selected_policy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_selected_policy_metric(paths: pd.DataFrame, metric: str, label: str, output_dir: Path) -> None:
    grouping = ["target", "model", "model_label", "design", "extra_two_phase_rows", "two_phase_rows"]
    summary = paths.groupby(grouping, as_index=False).agg(
        metric_mean=(metric, "mean"),
        metric_std=(metric, "std"),
    )
    summary = summary.rename(columns={"metric_mean": f"{metric}_mean", "metric_std": f"{metric}_std"})
    targets = list(dict.fromkeys(summary["target"].astype(str)))
    figure = make_subplots(rows=1, cols=len(targets), subplot_titles=[target.title() for target in targets])
    colors = dict(
        zip(
            PLOT_MODEL_IDS,
            ["#a50026", "#d73027", "#f46d43", "#fdae61", "#66bd63", "#006837"],
            strict=True,
        )
    )
    dashes = {
        "fixed_one_phase_aggregate": "dot",
        "two_phase_only": "solid",
        "tied_spine_plus_two_phase": "dash",
    }
    for column, target in enumerate(targets, start=1):
        for model_id in PLOT_MODEL_IDS:
            for design in ("fixed_one_phase_aggregate", *DESIGNS):
                local = summary.loc[
                    summary["target"].eq(target) & summary["model"].eq(model_id) & summary["design"].eq(design)
                ].sort_values("two_phase_rows")
                if local.empty:
                    continue
                name = f"{MODEL_LABELS[model_id]} · {design.replace('_', ' ')}"
                figure.add_trace(
                    go.Scatter(
                        x=local["two_phase_rows"],
                        y=local[f"{metric}_mean"],
                        mode="lines+markers",
                        name=name,
                        legendgroup=name,
                        showlegend=column == 1,
                        line={"color": colors[model_id], "dash": dashes[design]},
                        error_y={
                            "type": "data",
                            "array": local[f"{metric}_std"].fillna(0.0),
                            "visible": design != "fixed_one_phase_aggregate",
                        },
                        hovertemplate="%{fullData.name}<br>fit rows=%{x:.0f}<br>" + f"{label}=%{{y:.5f}}<extra></extra>",
                    ),
                    row=1,
                    col=column,
                )
        figure.update_xaxes(title_text="Original two-phase fit rows + added phase-varying rows", row=1, col=column)
        figure.update_yaxes(title_text=label if column == 1 else None, row=1, col=column)
    figure.update_layout(
        template="plotly_white",
        title=f"Predicted-best archived policy path: {label}",
        width=1500,
        height=650,
        legend={"orientation": "h", "y": -0.24},
        margin={"b": 190},
    )
    figure.write_html(output_dir / f"selected_policy_{metric}.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_selected_policy_report(
    learning_summary: pd.DataFrame,
    endpoints: pd.DataFrame,
    output_dir: Path,
) -> None:
    max_budget = int(learning_summary["extra_two_phase_rows"].max())
    endpoint_prediction = learning_summary.loc[
        learning_summary["design"].isin(DESIGNS) & learning_summary["extra_two_phase_rows"].eq(max_budget)
    ]
    prediction_winners = (
        endpoint_prediction.sort_values(["target", "rmse_mean", "regret_at_1_mean"])
        .groupby("target", as_index=False)
        .first()
    )
    endpoint_selection_winners = (
        endpoints.loc[endpoints["design"].isin(DESIGNS)]
        .sort_values(["target", "endpoint_regret_at_1", "endpoint_selected_observed"])
        .groupby("target", as_index=False)
        .first()
    )
    any_budget_winners = (
        learning_summary.loc[learning_summary["design"].isin(DESIGNS)]
        .sort_values(["target", "regret_at_1_mean", "rmse_mean"])
        .groupby("target", as_index=False)
        .first()
    )
    two_phase_endpoints = endpoints.loc[endpoints["design"].isin(DESIGNS)]
    stable_count = int(two_phase_endpoints["late_distinct_selected_policies"].eq(1).sum())
    report = "\n".join(
        [
            "# Which models are best, and do their selected policies converge?",
            "",
            "## Answer",
            "",
            (
                "No single model dominates prediction and policy selection. At the 760-row endpoint, Compact "
                "retained state has the best heldout RMSE on both targets. Separate heads has the best Uncheatable "
                "selection, while Compact retained state has the best Table-9 selection. HPR is a useful calibrated "
                "shared-state comparator but is not the endpoint winner on either target."
            ),
            "",
            (
                f"The predicted-best archived policy stabilizes after 280 added phase-varying rows in "
                f"{stable_count}/{len(two_phase_endpoints)} model/target/design paths. This is decision convergence, "
                "not convergence to the true optimum: endpoint Regret@1 remains material, and the best Uncheatable "
                "selection across all budgets occurs at the original 280-row Separate-heads fit rather than the "
                "760-row endpoint."
            ),
            "",
            "## Endpoint prediction winners",
            "",
            prediction_winners[
                [
                    "target",
                    "model_label",
                    "design",
                    "rmse_mean",
                    "spearman_mean",
                    "calibration_slope_mean",
                    "regret_at_1_mean",
                ]
            ].to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Endpoint selection winners",
            "",
            endpoint_selection_winners[
                [
                    "target",
                    "model_label",
                    "design",
                    "endpoint_selected_observed",
                    "endpoint_selected_predicted",
                    "endpoint_regret_at_1",
                    "endpoint_max_weight",
                    "endpoint_max_epochs",
                    "endpoint_phase_tv",
                    "endpoint_nearest_fit_tv",
                ]
            ].to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Best selection at any evidence budget",
            "",
            any_budget_winners[
                [
                    "target",
                    "model_label",
                    "design",
                    "extra_two_phase_rows",
                    "rmse_mean",
                    "regret_at_1_mean",
                    "selected_observed_mean",
                ]
            ].to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Health assessment",
            "",
            (
                "The endpoint-selected policies are not degenerate corners: the two winners have maximum bucket "
                "weights below 0.18 and maximum simulated exposure below 7.8 epochs. The Uncheatable Separate-heads "
                "winner is nearly phase tied and its selected-policy optimism is noise-scale. The Table-9 Compact "
                "winner has moderate phase asymmetry but remains strongly optimistic, so it is not a trustworthy "
                "solved optimum despite its plausible weights. Both are still outside the empirical fit support."
            ),
            "",
            (
                "This audit concerns the best policy among 461 frozen observed candidates. It does not establish that "
                "the continuous unregularized surrogate optimum is healthy; prior raw-optimum audits remain evidence "
                "that continuous extrapolation is unsafe."
            ),
        ]
    )
    (output_dir / "selected_policy_report.md").write_text(report + "\n")


def endpoint_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in summary.groupby(["target", "model"]):
        baseline = group.loc[group["design"].eq("fixed_one_phase_aggregate")].iloc[0]
        for design in DESIGNS:
            local = group.loc[group["design"].eq(design)].sort_values("two_phase_rows")
            first = local.iloc[0]
            last = local.iloc[-1]
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "design": design,
                    "start_two_phase_rows": int(first["two_phase_rows"]),
                    "end_two_phase_rows": int(last["two_phase_rows"]),
                    "rmse_start": float(first["rmse_mean"]),
                    "rmse_end": float(last["rmse_mean"]),
                    "rmse_change": float(last["rmse_mean"] - first["rmse_mean"]),
                    "regret1_start": float(first["regret_at_1_mean"]),
                    "regret1_end": float(last["regret_at_1_mean"]),
                    "regret1_change": float(last["regret_at_1_mean"] - first["regret_at_1_mean"]),
                    "one_phase_rmse": float(baseline["rmse_mean"]),
                    "one_phase_regret1": float(baseline["regret_at_1_mean"]),
                    "end_beats_one_phase_rmse": bool(last["rmse_mean"] < baseline["rmse_mean"]),
                    "end_beats_one_phase_regret1": bool(last["regret_at_1_mean"] < baseline["regret_at_1_mean"]),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    endpoints: pd.DataFrame,
    parameter_counts: pd.DataFrame,
    evaluation_count: int,
    output_dir: Path,
) -> None:
    best = (
        summary.loc[summary["design"].isin(DESIGNS)]
        .sort_values(["target", "regret_at_1_mean", "rmse_mean"])
        .groupby("target", as_index=False)
        .first()
    )
    improved_rmse = endpoints.groupby("design")["rmse_change"].agg(["mean", "min", "max"])
    phase_only = endpoints.loc[endpoints["design"].eq("two_phase_only")].set_index(["target", "model"])
    tied_spine = endpoints.loc[endpoints["design"].eq("tied_spine_plus_two_phase")].set_index(["target", "model"])
    rmse_improvement_count = int((endpoints["rmse_change"] < -1e-9).sum())
    regret_improvement_count = int((endpoints["regret1_change"] < -1e-9).sum())
    regret_regression_count = int((endpoints["regret1_change"] > 1e-9).sum())
    tied_rmse_win_count = int((tied_spine["rmse_end"] < phase_only["rmse_end"] - 1e-9).sum())
    tied_regret_win_count = int((tied_spine["regret1_end"] < phase_only["regret1_end"] - 1e-9).sum())
    report = "\n".join(
        [
            "# Fixed one-phase benchmark versus expanding two-phase evidence",
            "",
            "## Protocol",
            "",
            (
                "The one-phase restriction is independently fit to all 280 logical tied policies. The original "
                "two-phase fit panel has 280 observations: 238 phase-varying policies and 42 tied policies that are "
                "reused by the logical one-phase panel. Two-phase forms are fit either to that original panel alone "
                "or to it plus the 238 newly trained, coordinate-disjoint tied checkpoints. Both designs are then "
                "expanded with up to 480 systematic phase-varying frontier-fiber/random-population coordinates. "
                "Hyperparameters remain frozen at the values selected by the original 280-row Observatory fits."
            ),
            "",
            (
                f"All curves are evaluated on the same {evaluation_count} coordinate-distinct two-phase development "
                "policies from every other proposal series. The one-phase baseline predicts each policy after "
                "collapsing it to its aggregate mixture, so it is an aggregate-only predictor on exactly the same "
                "evaluation outcomes."
            ),
            "",
            (
                "This is exposed development evidence, not a confirmatory test. The extension pool is deliberately "
                "local and structured rather than IID."
            ),
            "",
            "## Dimensionality",
            "",
            (
                "For 39 buckets, the one-phase policy simplex has 38 dimensions and the unconstrained two-phase "
                "policy has 76. That does not imply that every surrogate doubles its fitted degrees of freedom: "
                "shared-state models can retain nearly the same nominal parameter count. Ridge penalties and nonlinear "
                "identifiability make the effective degrees of freedom smaller than these nominal counts. Matching "
                "the one-phase ratio of 280 rows to 38 policy dimensions would require 560 two-phase rows; the curves "
                "reach 760 base/extension rows."
            ),
            "",
            parameter_counts.to_markdown(index=False),
            "",
            "## Conclusion",
            "",
            (
                f"Adding phase-varying evidence lowers endpoint RMSE in {rmse_improvement_count}/{len(endpoints)} "
                f"model/target/design curves, but improves Regret@1 in only {regret_improvement_count}/{len(endpoints)} "
                f"and worsens it in {regret_regression_count}/{len(endpoints)}. The tied-spine design beats the pure "
                f"two-phase design at the endpoint in {tied_rmse_win_count}/10 RMSE comparisons and "
                f"{tied_regret_win_count}/10 Regret@1 comparisons."
            ),
            "",
            (
                "These descriptive counts are evidence that the two-phase fits are partly sample-limited for broad "
                "response prediction and calibration. They are not evidence that row count alone solves optimum "
                "selection: most gains arrive in the first 60--120 added rows, while selected-policy regret usually "
                "plateaus or changes discontinuously. "
                "Tied examples are useful aggregate-response evidence, but naive joint fitting does not reliably "
                "convert them into better phase-contrast decisions."
            ),
            "",
            "## Statistical caveats",
            "",
            (
                "The endpoint fits have no resampling uncertainty: budget zero adds no rows and budget 480 includes "
                "the complete extension pool for every seed. The 18/20 tally compares competing model hypotheses; it "
                "does not represent 20 exchangeable replicates."
            ),
            "",
            (
                "Regret@1 is a step function over a fixed, noisy 461-policy archive. Flat or discontinuous regret does "
                "not by itself establish sample-size saturation, and archive selection does not test whether the raw "
                "continuous surrogate optimum would validate. Regret@3 and Regret@5 are retained in the run table and "
                "show the same qualitative lack of reliable improvement."
            ),
            "",
            (
                "The tied-spine and pure two-phase curves are not equal-cost designs: tied spine spends 238 additional "
                "checkpoints at every x-coordinate. The companion fixed-budget composition audit remains the relevant "
                "test of replacing broad phase-varying coverage with tied rows, and it did not improve selection."
            ),
            "",
            (
                "Finally, the one-phase benchmark is deliberately fixed at 280 logical policies rather than traced as "
                "its own learning curve. This audit asks how much added two-phase evidence is needed to beat that fixed "
                "benchmark; it does not estimate relative one-phase and two-phase learning exponents."
            ),
            "",
            "## Endpoint changes",
            "",
            endpoints.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Mean RMSE change from 280 to 760 base/extension rows",
            "",
            improved_rmse.to_markdown(floatfmt=".5f"),
            "",
            "## Best discrete selection on the fixed evaluation archive",
            "",
            best[
                [
                    "target",
                    "model",
                    "design",
                    "two_phase_rows",
                    "tied_rows",
                    "rmse_mean",
                    "regret_at_1_mean",
                    "selected_observed_mean",
                    "calibration_slope_mean",
                ]
            ].to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Interpretation rule",
            "",
            (
                "A falling two-phase curve supports the claim that phase-sensitive fits are sample-limited at 280 "
                "rows. A tied-spine advantage at the same phase-varying count supports the decomposition that tied "
                "policies identify the aggregate response while phase-varying policies identify curriculum lift. "
                "Improvement in RMSE without improvement in Regret@1 does not establish a better solved optimum."
            ),
        ]
    )
    (output_dir / "report.md").write_text(report + "\n")


def main() -> None:
    args = parse_args()
    targets = parse_list(args.targets)
    models = parse_list(args.models)
    seeds = parse_int_list(args.seeds)
    budgets = parse_int_list(args.budgets)
    if not set(targets).issubset(TARGETS):
        raise ValueError(f"Unknown targets {set(targets) - set(TARGETS)}")
    if not set(models).issubset(PLOT_MODEL_IDS):
        raise ValueError(f"Unknown models {set(models) - set(PLOT_MODEL_IDS)}")
    if not budgets or min(budgets) < 0 or max(budgets) > 480:
        raise ValueError(f"Budgets must be in [0, 480], got {budgets}")
    if args.optima_only and args.skip_optima:
        raise ValueError("--optima-only and --skip-optima are mutually exclusive")
    if args.optimizer_starts < 4:
        raise ValueError("--optimizer-starts must be at least four")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(targets, models, seeds, budgets)
    existing = load_existing(args.output_dir, protocol, args.overwrite)
    rows = existing.to_dict("records")
    completed = {row_key(row) for row in rows}
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    reference = observatory.load_delphi_3e18_fit_dataset(targets[0])
    extension, evaluation = heldout_pools(reference)
    inventory = pd.DataFrame(
        [
            {"role": "phase_extension", "series": series, "coordinates": int(count)}
            for series, count in extension.frame["training_series"].value_counts().items()
        ]
        + [
            {"role": "frozen_evaluation", "series": series, "coordinates": int(count)}
            for series, count in evaluation.frame["training_series"].value_counts().items()
        ]
    )
    inventory.to_csv(args.output_dir / "data_inventory.csv", index=False)

    if args.rebuild_only and not rows:
        raise ValueError("--rebuild-only requires an existing learning_curve_runs.csv")
    if not args.rebuild_only and not args.optima_only:
        for target in targets:
            run_target(
                target,
                models,
                seeds,
                budgets,
                extension,
                evaluation,
                rows,
                completed,
                args.output_dir,
            )

    if rows:
        runs = pd.DataFrame(rows)
        summary = summarize(runs)
        endpoints = endpoint_table(summary)
        parameter_counts = parameter_count_table(targets, models)
        summary.to_csv(args.output_dir / "learning_curve_summary.csv", index=False)
        endpoints.to_csv(args.output_dir / "endpoint_comparison.csv", index=False)
        parameter_counts.to_csv(args.output_dir / "model_parameter_counts.csv", index=False)
        for metric, label in (
            ("rmse", "Heldout RMSE"),
            ("regret_at_1", "Heldout Regret@1"),
            ("regret_at_3", "Heldout Regret@3"),
            ("regret_at_5", "Heldout Regret@5"),
            ("calibration_error", "Absolute calibration-slope error"),
            ("selected_observed", "Observed BPB of selected heldout policy"),
        ):
            plot_metric(summary, metric, label, args.output_dir)
        write_report(summary, endpoints, parameter_counts, len(evaluation.frame), args.output_dir)
        selected_paths = selected_archive_policy_paths(runs, extension, evaluation)
        selected_endpoints = selected_policy_endpoint_table(selected_paths)
        selected_paths.to_csv(args.output_dir / "selected_policy_paths.csv", index=False)
        selected_endpoints.to_csv(args.output_dir / "selected_policy_endpoint_diagnostics.csv", index=False)
        for metric, label in (
            ("selected_observed", "Observed BPB"),
            ("tv_to_endpoint_selected_policy", "Policy TV to endpoint selection"),
            ("max_simulated_epochs", "Maximum simulated epochs"),
            ("phase_total_variation", "Phase total variation"),
            ("standardized_fit_support_distance", "Standardized fit-support distance"),
        ):
            plot_selected_policy_metric(selected_paths, metric, label, args.output_dir)
        write_selected_policy_report(summary, selected_endpoints, args.output_dir)
    elif not args.optima_only:
        raise ValueError("No learning-curve rows are available")

    optimum_path = args.output_dir / "raw_optimum_runs.csv"
    if optimum_path.exists() and not args.overwrite:
        optimum_rows = pd.read_csv(optimum_path).to_dict("records")
    else:
        optimum_rows = []
    optimum_completed = {optimum_row_key(row) for row in optimum_rows}
    optimum_protocol = {
        "targets": list(targets),
        "models": list(models),
        "seeds": list(seeds),
        "extra_two_phase_budgets": list(budgets),
        "optimizer_starts": args.optimizer_starts,
        "objective": "unregularized fitted two-phase surrogate BPB",
        "warm_start": "preceding evidence budget within target/model/design/seed",
    }
    optimum_protocol_path = args.output_dir / "raw_optimum_protocol.json"
    if optimum_protocol_path.exists() and not args.overwrite:
        existing_optimum_protocol = json.loads(optimum_protocol_path.read_text())
        if existing_optimum_protocol != optimum_protocol:
            raise ValueError("Existing raw optimum protocol differs; use a new output directory or --overwrite")
    optimum_protocol_path.write_text(json.dumps(optimum_protocol, indent=2, sort_keys=True) + "\n")
    if not args.rebuild_only and args.optima_only:
        for target in targets:
            run_raw_optimum_target(
                target,
                models,
                seeds,
                budgets,
                extension,
                evaluation,
                optimum_rows,
                optimum_completed,
                args.output_dir,
                args.optimizer_starts,
            )
    if optimum_rows:
        optimum_runs = add_optimum_endpoint_distances(pd.DataFrame(optimum_rows))
        optimum_runs.to_csv(optimum_path, index=False)
        optimum_summary = summarize_optima(optimum_runs)
        optimum_endpoints = optimum_endpoint_table(optimum_runs)
        optimum_summary.to_csv(args.output_dir / "raw_optimum_summary.csv", index=False)
        optimum_endpoints.to_csv(args.output_dir / "raw_optimum_endpoint_diagnostics.csv", index=False)
        for metric, label in (
            ("predicted_bpb", "Predicted BPB"),
            ("tv_to_endpoint_optimum", "Policy TV to 760-row endpoint optimum"),
            ("successive_optimum_tv", "Policy TV from preceding evidence budget"),
            ("max_simulated_epochs", "Maximum simulated epochs"),
            ("phase_total_variation", "Phase total variation"),
            ("standardized_fit_support_distance", "Standardized fit-support distance"),
            ("nearest_evaluation_observed", "Observed BPB of nearest evaluation policy"),
        ):
            plot_optimum_metric(optimum_summary, metric, label, args.output_dir)
        write_optimum_report(optimum_endpoints, args.output_dir)


if __name__ == "__main__":
    main()
