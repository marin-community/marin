# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Fit family-aware GRP quality variants on the production Grug-MoE swarm.

This benchmark deliberately leaves the Mixture Fit Observatory unchanged. It
compares two production-specific extensions of the regularized GRP surrogate:

* ``canonical_family_quality`` compresses each C-family's Q tiers through one
  monotone, globally shared quality discount.
* ``bucket_resolved_quality`` gives every bucket its own response amplitude,
  while retaining nonlinear C-family coverage and overexposure channels.

Hyperparameters are selected with repeated cross-validation. Reported primary
OOF metrics come from an outer five-fold nested CV so shape and ridge selection
do not see the held-out fold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls
from scipy.stats import qmc, spearmanr
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PRODUCTION_DATA = REFERENCE_OUTPUTS / "grug_moe_production_swarm_results_20260704/production_swarm_840_wide.csv"
PRODUCTION_MODEL = REFERENCE_OUTPUTS / "grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/model.json"
OBSERVATORY_CACHE = REFERENCE_OUTPUTS / "mixture_fit_observatory_cache_20260713/production/uncheatable/two_phase"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "production_grp_quality_variants_20260713"
TARGET_COLUMN = "eval/uncheatable_eval/bpb"
DOMAIN_PATTERN = re.compile(r"c(?P<family>\d+)q(?P<quality>\d+)$")
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
FULL_CV_SEEDS = (0, 17, 41)
OUTER_CV_SEED = 713
N_SPLITS = 5
INNER_SPLITS = 3
L2_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0, 10.0)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Variant(StrEnum):
    CANONICAL = "canonical_family_quality"
    BUCKET_RESOLVED = "bucket_resolved_quality"


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    target: np.ndarray
    weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    domains: tuple[str, ...]
    family_names: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    quality: np.ndarray

    @property
    def n(self) -> int:
        return len(self.target)

    @property
    def m(self) -> int:
        return len(self.domains)


@dataclass(frozen=True)
class Shape:
    exponent: float
    late_multiplier: float
    forgetting_rate: float
    penalty_threshold: float
    quality_discount: float = 1.0


@dataclass(frozen=True)
class FittedHead:
    intercept: float
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    l2: float

    def predict_design(self, design: np.ndarray) -> np.ndarray:
        return np.asarray(self.intercept + design @ self.coefficients, dtype=float)


@dataclass(frozen=True)
class SelectedFit:
    variant: Variant
    shape: Shape
    l2: float
    head: FittedHead
    repeated_oof_prediction: np.ndarray
    nested_oof_prediction: np.ndarray
    nested_selections: tuple[dict[str, float | int], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PRODUCTION_DATA)
    parser.add_argument("--model", type=Path, default=PRODUCTION_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=96)
    return parser.parse_args()


def load_dataset(data_path: Path, model_path: Path) -> Dataset:
    frame = pd.read_csv(data_path)
    model = json.loads(model_path.read_text())
    domains = tuple(model["domain_names"])
    w0 = frame[[f"phase_0/{domain}" for domain in domains]].to_numpy(dtype=float)
    w1 = frame[[f"phase_1/{domain}" for domain in domains]].to_numpy(dtype=float)
    w0 /= w0.sum(axis=1, keepdims=True)
    w1 /= w1.sum(axis=1, keepdims=True)

    family_to_indices: dict[str, list[int]] = {}
    quality = np.full(len(domains), -1, dtype=int)
    for index, domain in enumerate(domains):
        match = DOMAIN_PATTERN.fullmatch(domain)
        if match is None:
            family = domain
        else:
            family = f"c{match.group('family')}"
            quality[index] = int(match.group("quality"))
        family_to_indices.setdefault(family, []).append(index)

    family_names = tuple(family_to_indices)
    family_members = tuple(np.asarray(family_to_indices[name], dtype=int) for name in family_names)
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError(f"{TARGET_COLUMN} has {np.count_nonzero(~np.isfinite(target))} non-finite rows")
    return Dataset(
        frame=frame,
        target=target,
        weights=np.stack([w0, w1], axis=1),
        c0=np.asarray(model["c0"], dtype=float),
        c1=np.asarray(model["c1"], dtype=float),
        domains=domains,
        family_names=family_names,
        family_members=family_members,
        quality=quality,
    )


def retained_exposure(dataset: Dataset, shape: Shape) -> np.ndarray:
    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    e0 = phase0_weight * dataset.c0[None, :]
    e1 = phase1_weight * dataset.c1[None, :]
    retained_phase0 = np.exp(-shape.forgetting_rate * (1.0 - phase1_weight)) * e0
    return np.maximum(retained_phase0 + shape.late_multiplier * e1, 0.0)


def response(exposure: np.ndarray, exponent: float) -> np.ndarray:
    return np.maximum(exposure, 1e-12) ** exponent


def penalty(exposure: np.ndarray, threshold: float) -> np.ndarray:
    return np.logaddexp(0.0, np.log1p(np.maximum(exposure, 0.0)) - threshold) ** 2


def quality_weights(dataset: Dataset, shape: Shape, members: np.ndarray) -> np.ndarray:
    member_quality = dataset.quality[members]
    if len(members) == 1 or np.max(member_quality) <= 0:
        return np.ones(len(members), dtype=float)
    maximum = float(np.max(member_quality))
    normalized_gap = 1.0 - member_quality.astype(float) / maximum
    return shape.quality_discount**normalized_gap


def build_design(dataset: Dataset, variant: Variant, shape: Shape) -> tuple[np.ndarray, tuple[str, ...]]:
    retained = retained_exposure(dataset, shape)
    family_totals = np.column_stack([retained[:, members].sum(axis=1) for members in dataset.family_members])

    if variant is Variant.CANONICAL:
        family_signal = np.column_stack(
            [
                response(retained[:, members] @ quality_weights(dataset, shape, members), shape.exponent)
                for members in dataset.family_members
            ]
        )
        family_penalty = penalty(family_totals, shape.penalty_threshold)
        feature_names = tuple(f"family_signal:{name}" for name in dataset.family_names) + tuple(
            f"family_penalty:{name}" for name in dataset.family_names
        )
        return np.hstack([-family_signal, family_penalty]), feature_names

    if variant is Variant.BUCKET_RESOLVED:
        bucket_signal = response(retained, shape.exponent)
        nonsingleton = [index for index, members in enumerate(dataset.family_members) if len(members) > 1]
        family_signal = response(family_totals[:, nonsingleton], shape.exponent)
        family_penalty = penalty(family_totals, shape.penalty_threshold)
        feature_names = (
            tuple(f"bucket_signal:{domain}" for domain in dataset.domains)
            + tuple(f"family_signal:{dataset.family_names[index]}" for index in nonsingleton)
            + tuple(f"family_penalty:{name}" for name in dataset.family_names)
        )
        return np.hstack([-bucket_signal, -family_signal, family_penalty]), feature_names

    raise ValueError(f"Unsupported variant {variant}")


def fit_head(
    design: np.ndarray, target: np.ndarray, indices: np.ndarray, l2: float, names: tuple[str, ...]
) -> FittedHead:
    train_design = design[indices]
    train_target = target[indices]
    design_mean = train_design.mean(axis=0, keepdims=True)
    target_mean = float(train_target.mean())
    centered_design = train_design - design_mean
    centered_target = train_target - target_mean
    if l2 > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(l2) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1], dtype=float)])
    coefficients, _residual = nnls(centered_design, centered_target, maxiter=30 * design.shape[1])
    intercept = target_mean - float((design_mean @ coefficients).item())
    return FittedHead(intercept, coefficients, names, l2)


def kfold_indices(indices: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(indices[train], indices[test]) for train, test in splitter.split(indices)]


def oof_prediction(
    design: np.ndarray,
    target: np.ndarray,
    splits_by_seed: tuple[list[tuple[np.ndarray, np.ndarray]], ...],
    l2: float,
    names: tuple[str, ...],
) -> np.ndarray:
    predictions = np.full((len(splits_by_seed), len(target)), np.nan, dtype=float)
    covered_by_seed: list[np.ndarray] = []
    for seed_index, splits in enumerate(splits_by_seed):
        covered = np.unique(np.concatenate([test for _train, test in splits]))
        covered_by_seed.append(covered)
        for train, test in splits:
            head = fit_head(design, target, train, l2, names)
            predictions[seed_index, test] = head.predict_design(design[test])
        if not np.isfinite(predictions[seed_index, covered]).all():
            raise RuntimeError("OOF prediction matrix is incomplete on covered rows")
    common_covered = covered_by_seed[0]
    if any(not np.array_equal(common_covered, covered) for covered in covered_by_seed[1:]):
        raise ValueError("Repeated CV seeds must cover the same row set")
    prediction = np.full(len(target), np.nan, dtype=float)
    prediction[common_covered] = predictions[:, common_covered].mean(axis=0)
    return prediction


def rmse(observed: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - observed) ** 2)))


def shape_candidates(variant: Variant, count: int) -> tuple[Shape, ...]:
    dimension = 5 if variant is Variant.CANONICAL else 4
    sample_count = 1 << math.ceil(math.log2(max(count, 2)))
    unit = qmc.Sobol(d=dimension, scramble=True, seed=117 if variant is Variant.CANONICAL else 211).random_base2(
        int(math.log2(sample_count))
    )[:count]

    candidates: list[Shape] = []
    for row in unit:
        exponent = float(np.exp(np.log(0.08) + row[0] * (np.log(1.2) - np.log(0.08))))
        late_multiplier = float(np.exp(np.log(0.75) + row[1] * (np.log(12.0) - np.log(0.75))))
        forgetting_rate = float(np.exp(np.log(1e-5) + row[2] * (np.log(4.0) - np.log(1e-5))))
        penalty_threshold = float(row[3] * 7.0)
        quality_discount = float(0.05 + 0.95 * row[4]) if variant is Variant.CANONICAL else 1.0
        candidates.append(Shape(exponent, late_multiplier, forgetting_rate, penalty_threshold, quality_discount))

    transferred = Shape(
        exponent=0.33989885260566105,
        late_multiplier=6.627794351309641,
        forgetting_rate=6.14421235332821e-06,
        penalty_threshold=5.136810831800622,
        quality_discount=0.2629059619755788 if variant is Variant.CANONICAL else 1.0,
    )
    no_phase_premium = Shape(0.33989885260566105, 1.0, 0.0, 5.136810831800622, transferred.quality_discount)
    return tuple([transferred, no_phase_premium, *candidates])


def candidate_record(variant: Variant, shape_index: int, shape: Shape, l2: float, score: float) -> dict[str, Any]:
    return {
        "variant": variant.value,
        "shape_index": shape_index,
        **asdict(shape),
        "l2": l2,
        "repeated_cv_rmse": score,
    }


def select_hyperparameters(
    dataset: Dataset,
    variant: Variant,
    designs: tuple[tuple[np.ndarray, tuple[str, ...]], ...],
    shapes: tuple[Shape, ...],
    splits_by_seed: tuple[list[tuple[np.ndarray, np.ndarray]], ...],
) -> tuple[int, float, np.ndarray, list[dict[str, Any]]]:
    covered = np.unique(np.concatenate([test for splits in splits_by_seed for _train, test in splits]))
    rows: list[dict[str, Any]] = []
    best: tuple[float, int, float, np.ndarray] | None = None
    for shape_index, (shape, (design, names)) in enumerate(zip(shapes, designs, strict=True)):
        for l2 in L2_GRID:
            prediction = oof_prediction(design, dataset.target, splits_by_seed, l2, names)
            score = rmse(dataset.target[covered], prediction[covered])
            rows.append(candidate_record(variant, shape_index, shape, l2, score))
            candidate = (score, shape_index, l2, prediction)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise RuntimeError(f"No hyperparameter candidates for {variant}")
    return best[1], best[2], best[3], rows


def nested_oof(
    dataset: Dataset,
    variant: Variant,
    designs: tuple[tuple[np.ndarray, tuple[str, ...]], ...],
    shapes: tuple[Shape, ...],
) -> tuple[np.ndarray, tuple[dict[str, float | int], ...], list[np.ndarray]]:
    outer_splits = kfold_indices(np.arange(dataset.n), N_SPLITS, OUTER_CV_SEED)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selections: list[dict[str, float | int]] = []
    test_indices: list[np.ndarray] = []
    for fold, (outer_train, outer_test) in enumerate(outer_splits):
        inner_splits = (kfold_indices(outer_train, INNER_SPLITS, OUTER_CV_SEED + fold + 1),)
        shape_index, l2, _inner_prediction, _rows = select_hyperparameters(
            dataset, variant, designs, shapes, inner_splits
        )
        design, names = designs[shape_index]
        head = fit_head(design, dataset.target, outer_train, l2, names)
        prediction[outer_test] = head.predict_design(design[outer_test])
        selections.append(
            {
                "outer_fold": fold,
                "shape_index": shape_index,
                **asdict(shapes[shape_index]),
                "l2": l2,
            }
        )
        test_indices.append(outer_test)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Nested OOF prediction incomplete for {variant}")
    return prediction, tuple(selections), test_indices


def metric_summary(
    observed: np.ndarray,
    prediction: np.ndarray,
    fold_test_indices: list[np.ndarray] | None = None,
) -> dict[str, float | int | None]:
    residual = prediction - observed
    lower_tail_count = min(len(observed), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))))
    lower_tail = np.argsort(prediction)[:lower_tail_count]
    lower_tail_error = observed[lower_tail] - prediction[lower_tail]
    fold_regrets: list[float] = []
    if fold_test_indices is not None:
        for test in fold_test_indices:
            selected = int(test[int(np.argmin(prediction[test]))])
            fold_regrets.append(float(observed[selected] - np.min(observed[test])))
    return {
        "n": len(observed),
        "rmse": rmse(observed, prediction),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "regret_at_1": float(observed[int(np.argmin(prediction))] - np.min(observed)),
        "fold_mean_regret_at_1": float(np.mean(fold_regrets)) if fold_regrets else None,
        "lower_tail_optimism": float(np.mean(np.maximum(lower_tail_error, 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(lower_tail_error**2))),
        "lower_tail_count": lower_tail_count,
    }


def fit_variant(dataset: Dataset, variant: Variant, num_shapes: int) -> tuple[SelectedFit, list[dict[str, Any]]]:
    shapes = shape_candidates(variant, num_shapes)
    designs = tuple(build_design(dataset, variant, shape) for shape in shapes)
    repeated_splits = tuple(kfold_indices(np.arange(dataset.n), N_SPLITS, seed) for seed in FULL_CV_SEEDS)
    shape_index, l2, repeated_prediction, sweep = select_hyperparameters(
        dataset, variant, designs, shapes, repeated_splits
    )
    nested_prediction, nested_selections, _nested_tests = nested_oof(dataset, variant, designs, shapes)
    design, names = designs[shape_index]
    head = fit_head(design, dataset.target, np.arange(dataset.n), l2, names)
    return (
        SelectedFit(
            variant=variant,
            shape=shapes[shape_index],
            l2=l2,
            head=head,
            repeated_oof_prediction=repeated_prediction,
            nested_oof_prediction=nested_prediction,
            nested_selections=nested_selections,
        ),
        sweep,
    )


def coefficient_frame(dataset: Dataset, fit: SelectedFit) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": fit.variant.value,
            "feature": fit.head.feature_names,
            "coefficient": fit.head.coefficients,
        }
    )


def quality_frame(dataset: Dataset, fit: SelectedFit) -> pd.DataFrame:
    proportional_weight0 = (1.0 / dataset.c0) / np.sum(1.0 / dataset.c0)
    proportional_weight1 = (1.0 / dataset.c1) / np.sum(1.0 / dataset.c1)
    phase0_exposure = proportional_weight0 * dataset.c0
    phase1_exposure = proportional_weight1 * dataset.c1
    proportional_retained = (
        np.exp(-fit.shape.forgetting_rate * (1.0 - proportional_weight1)) * phase0_exposure
        + fit.shape.late_multiplier * phase1_exposure
    )
    coefficient = dict(zip(fit.head.feature_names, fit.head.coefficients, strict=True))

    def penalty_derivative(total: float) -> float:
        shifted = np.log1p(total) - fit.shape.penalty_threshold
        softplus_value = float(np.logaddexp(0.0, shifted))
        sigmoid_value = float(1.0 / (1.0 + np.exp(-shifted)))
        return 2.0 * softplus_value * sigmoid_value / (1.0 + total)

    if fit.variant is Variant.CANONICAL:
        rows = []
        for family, members in zip(dataset.family_names, dataset.family_members, strict=True):
            weights = quality_weights(dataset, fit.shape, members)
            weighted_total = float(proportional_retained[members] @ weights)
            raw_total = float(proportional_retained[members].sum())
            response_derivative = fit.shape.exponent * max(weighted_total, 1e-12) ** (fit.shape.exponent - 1.0)
            signal_coefficient = coefficient[f"family_signal:{family}"]
            penalty_coefficient = coefficient[f"family_penalty:{family}"]
            for member, weight in zip(members, weights, strict=True):
                rows.append(
                    {
                        "variant": fit.variant.value,
                        "family": family,
                        "domain": dataset.domains[member],
                        "quality": int(dataset.quality[member]),
                        "raw_quality_value": float(weight),
                        "local_marginal_utility": float(
                            signal_coefficient * response_derivative * weight
                            - penalty_coefficient * penalty_derivative(raw_total)
                        ),
                    }
                )
        return pd.DataFrame(rows)

    rows = []
    for family, members in zip(dataset.family_names, dataset.family_members, strict=True):
        raw_total = float(proportional_retained[members].sum())
        family_signal_coefficient = coefficient.get(f"family_signal:{family}", 0.0)
        family_penalty_coefficient = coefficient[f"family_penalty:{family}"]
        family_response_derivative = fit.shape.exponent * max(raw_total, 1e-12) ** (fit.shape.exponent - 1.0)
        for member in members:
            domain = dataset.domains[member]
            bucket_coefficient = coefficient[f"bucket_signal:{domain}"]
            bucket_response_derivative = fit.shape.exponent * max(float(proportional_retained[member]), 1e-12) ** (
                fit.shape.exponent - 1.0
            )
            rows.append(
                {
                    "variant": fit.variant.value,
                    "family": family,
                    "domain": domain,
                    "quality": int(dataset.quality[member]),
                    "raw_quality_value": float(bucket_coefficient),
                    "local_marginal_utility": float(
                        bucket_coefficient * bucket_response_derivative
                        + family_signal_coefficient * family_response_derivative
                        - family_penalty_coefficient * penalty_derivative(raw_total)
                    ),
                }
            )
    return pd.DataFrame(rows)


def learned_quality_diagnostics(quality: pd.DataFrame) -> dict[str, float | int]:
    usable = quality.loc[quality["quality"].ge(0)].copy()
    diagnostics: dict[str, float | int] = {}
    for value_column, prefix in (
        ("raw_quality_value", "raw_amplitude"),
        ("local_marginal_utility", "proportional_marginal_utility"),
    ):
        correlations = []
        monotone_edges = 0
        total_edges = 0
        monotone_families = 0
        family_count = 0
        for _family, rows in usable.groupby("family"):
            rows = rows.sort_values("quality")
            if len(rows) < 2:
                continue
            values = rows[value_column].to_numpy(dtype=float)
            qualities = rows["quality"].to_numpy(dtype=float)
            correlation = spearmanr(qualities, values).statistic
            correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
            differences = np.diff(values)
            monotone_edges += int(np.count_nonzero(differences >= -1e-12))
            total_edges += len(differences)
            monotone_families += int(np.all(differences >= -1e-12))
            family_count += 1
        diagnostics["family_count"] = family_count
        diagnostics[f"{prefix}_monotone_family_count"] = monotone_families
        diagnostics[f"{prefix}_monotone_edge_fraction"] = monotone_edges / total_edges
        diagnostics[f"{prefix}_median_within_family_quality_spearman"] = float(np.median(correlations))
    return diagnostics


def observatory_baselines() -> pd.DataFrame:
    rows = []
    for path in sorted(OBSERVATORY_CACHE.glob("*.json")):
        detail = json.loads(path.read_text())["fitDetail"]
        diagnostics = detail["diagnostics"]["oof"]
        rows.append(
            {
                "model": detail["modelLabel"],
                "evaluation": "observatory_oof",
                "parameter_count": detail["parameterCount"],
                "rmse": diagnostics["rmse"],
                "mae": diagnostics["mae"],
                "spearman": diagnostics["spearman"],
                "regret_at_1": diagnostics["regretAt1"],
                "fold_mean_regret_at_1": diagnostics["foldMeanRegretAt1"],
                "lower_tail_optimism": diagnostics["lowerTailOptimism"],
                "low_tail_rmse": diagnostics["lowTailRmse"],
            }
        )
    return pd.DataFrame(rows)


def summary_frame(dataset: Dataset, fits: tuple[SelectedFit, ...]) -> pd.DataFrame:
    rows = []
    nested_tests = [test for _train, test in kfold_indices(np.arange(dataset.n), N_SPLITS, OUTER_CV_SEED)]
    for fit in fits:
        full_design, _names = build_design(dataset, fit.variant, fit.shape)
        evaluations = {
            "nested_oof": (fit.nested_oof_prediction, nested_tests),
            "selected_shape_repeated_oof": (fit.repeated_oof_prediction, None),
            "train": (fit.head.predict_design(full_design), None),
        }
        for evaluation, (prediction, tests) in evaluations.items():
            metrics = metric_summary(dataset.target, prediction, tests)
            rows.append(
                {
                    "model": fit.variant.value,
                    "evaluation": evaluation,
                    "parameter_count": len(fit.head.coefficients) + 1 + (5 if fit.variant is Variant.CANONICAL else 4),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def prediction_frame(dataset: Dataset, fits: tuple[SelectedFit, ...]) -> pd.DataFrame:
    rows = []
    for fit in fits:
        for index, prediction in enumerate(fit.nested_oof_prediction):
            rows.append(
                {
                    "variant": fit.variant.value,
                    "row_index": index,
                    "candidate_name": dataset.frame.iloc[index]["candidate_name"],
                    "observed_bpb": dataset.target[index],
                    "nested_oof_prediction": prediction,
                    "residual": prediction - dataset.target[index],
                }
            )
    return pd.DataFrame(rows)


def write_plot(dataset: Dataset, fits: tuple[SelectedFit, ...], summary: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Nested OOF predictions", "Decision diagnostics", "Learned Q-tier utilities"),
        horizontal_spacing=0.1,
    )
    colors = {Variant.CANONICAL: "#d73027", Variant.BUCKET_RESOLVED: "#1a9850"}
    for fit in fits:
        color = colors[fit.variant]
        figure.add_trace(
            go.Scatter(
                x=dataset.target,
                y=fit.nested_oof_prediction,
                mode="markers",
                marker={"color": color, "size": 6, "opacity": 0.58},
                name=fit.variant.value,
                legendgroup=fit.variant.value,
                customdata=np.column_stack([dataset.frame["candidate_name"]]),
                hovertemplate="%{customdata[0]}<br>observed=%{x:.5f}<br>nested OOF=%{y:.5f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    low = min(float(dataset.target.min()), *(float(fit.nested_oof_prediction.min()) for fit in fits))
    high = max(float(dataset.target.max()), *(float(fit.nested_oof_prediction.max()) for fit in fits))
    figure.add_trace(
        go.Scatter(
            x=[low, high], y=[low, high], mode="lines", line={"color": "#667085", "dash": "dash"}, showlegend=False
        ),
        row=1,
        col=1,
    )

    nested = summary.loc[summary["evaluation"].eq("nested_oof")]
    for metric, dash in (("fold_mean_regret_at_1", "solid"), ("lower_tail_optimism", "dot")):
        figure.add_trace(
            go.Bar(
                x=nested["model"],
                y=nested[metric],
                name=metric,
                marker_color=[colors[Variant(model)] for model in nested["model"]],
                opacity=1.0 if dash == "solid" else 0.55,
                legendgroup=metric,
            ),
            row=1,
            col=2,
        )

    bucket_fit = next(fit for fit in fits if fit.variant is Variant.BUCKET_RESOLVED)
    quality = quality_frame(dataset, bucket_fit)
    quality = quality.loc[quality["quality"].ge(0)]
    for family, rows in quality.groupby("family"):
        figure.add_trace(
            go.Scatter(
                x=rows["quality"],
                y=rows["local_marginal_utility"],
                mode="lines+markers",
                line={"color": "rgba(26,152,80,0.22)", "width": 1},
                marker={"color": "#1a9850", "size": 4},
                name=family,
                showlegend=False,
                hovertemplate=f"{family}<br>Q=%{{x}}<br>marginal utility=%{{y:.5g}}<extra></extra>",
            ),
            row=1,
            col=3,
        )

    figure.update_xaxes(title_text="Observed Uncheatable BPB", row=1, col=1)
    figure.update_yaxes(title_text="Predicted BPB", row=1, col=1)
    figure.update_xaxes(title_text="Variant", tickangle=-18, row=1, col=2)
    figure.update_yaxes(title_text="BPB", row=1, col=2)
    figure.update_xaxes(title_text="Quality tier Q", dtick=1, row=1, col=3)
    figure.update_yaxes(title_text="Marginal BPB improvement / retained epoch", row=1, col=3)
    figure.update_layout(
        title="Production Grug-MoE: family-aware GRP quality variants",
        template="plotly_white",
        height=660,
        width=1740,
        barmode="group",
        legend={"orientation": "h", "y": -0.22},
        margin={"l": 70, "r": 40, "t": 90, "b": 155},
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)


def write_report(
    dataset: Dataset,
    fits: tuple[SelectedFit, ...],
    summary: pd.DataFrame,
    quality_diagnostics: dict[str, float | int],
    output_path: Path,
) -> None:
    nested = summary.loc[summary["evaluation"].eq("nested_oof")].copy()
    baseline = observatory_baselines().sort_values("rmse")
    lines = [
        "# Production GRP quality-structure benchmark",
        "",
        "## Protocol",
        "",
        f"The dataset has {dataset.n} rows, {dataset.m} buckets, and {len(dataset.family_names)} C/tail families. "
        "The primary metrics are from five-fold nested CV: each outer training fold independently selects the "
        "nonlinear shape and ridge using three-fold inner CV. The full-data shape uses repeated five-fold CV over "
        f"seeds {FULL_CV_SEEDS}. No Observatory code or cache was changed.",
        "",
        "## Nested OOF results",
        "",
        nested.to_markdown(index=False),
        "",
        "## Existing Observatory baselines",
        "",
        "These use the Observatory's repeated OOF protocol rather than nested shape selection, so they are a useful "
        "reference but not a perfectly matched test.",
        "",
        baseline.to_markdown(index=False),
        "",
        "## Selected full-data hyperparameters",
        "",
    ]
    for fit in fits:
        lines.extend(
            [
                f"### {fit.variant.value}",
                "",
                "```json",
                json.dumps({**asdict(fit.shape), "l2": fit.l2}, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Learned quality diagnostic",
            "",
            "The bucket-resolved model does not impose the declared Q ordering. Agreement with that ordering is "
            "therefore an out-of-model diagnostic, not a fitted constraint.",
            "",
            "```json",
            json.dumps(quality_diagnostics, indent=2),
            "```",
            "",
            "## Interpretation guardrail",
            "",
            "The production panel is one 840-row D-optimal design. Nested OOF tests interpolation and selection "
            "inside that design; the resumed domain-ablation interventions remain the decisive out-of-design test.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data, args.model)

    fits: list[SelectedFit] = []
    sweeps: list[pd.DataFrame] = []
    for variant in Variant:
        fit, sweep = fit_variant(dataset, variant, args.num_shapes)
        fits.append(fit)
        sweeps.append(pd.DataFrame(sweep))

    fit_tuple = tuple(fits)
    summary = summary_frame(dataset, fit_tuple)
    predictions = prediction_frame(dataset, fit_tuple)
    coefficients = pd.concat([coefficient_frame(dataset, fit) for fit in fit_tuple], ignore_index=True)
    quality = pd.concat([quality_frame(dataset, fit) for fit in fit_tuple], ignore_index=True)
    nested_selections = pd.DataFrame(
        [{"variant": fit.variant.value, **selection} for fit in fit_tuple for selection in fit.nested_selections]
    )
    quality_diagnostics = learned_quality_diagnostics(quality.loc[quality["variant"].eq(Variant.BUCKET_RESOLVED.value)])

    summary.to_csv(args.output_dir / "fit_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    pd.concat(sweeps, ignore_index=True).to_csv(args.output_dir / "hyperparameter_sweep.csv", index=False)
    nested_selections.to_csv(args.output_dir / "nested_cv_selections.csv", index=False)
    coefficients.to_csv(args.output_dir / "fitted_coefficients.csv", index=False)
    quality.to_csv(args.output_dir / "quality_parameters.csv", index=False)
    (args.output_dir / "quality_diagnostics.json").write_text(json.dumps(quality_diagnostics, indent=2) + "\n")
    for fit in fit_tuple:
        (args.output_dir / f"{fit.variant.value}_model.json").write_text(
            json.dumps(
                {
                    "variant": fit.variant.value,
                    "shape": asdict(fit.shape),
                    "l2": fit.l2,
                    "intercept": fit.head.intercept,
                    "feature_names": fit.head.feature_names,
                    "coefficients": fit.head.coefficients.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
    write_plot(dataset, fit_tuple, summary, args.output_dir / "fit_diagnostics.html")
    write_report(dataset, fit_tuple, summary, quality_diagnostics, args.output_dir / "report.md")
    print(summary.to_string(index=False))
    print(json.dumps(quality_diagnostics, indent=2))


if __name__ == "__main__":
    main()
