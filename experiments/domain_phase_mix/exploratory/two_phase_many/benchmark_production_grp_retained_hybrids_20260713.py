# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Benchmark family replay structure and compact retained-state GRP hybrids.

The production Grug-MoE and 300M swarms have known bucket-family partitions.
This script asks two narrow questions without changing the Mixture Fit
Observatory:

1. Does bucket-resolved family GRP improve when replay-harm onset is learned
   per family rather than shared globally?
2. Which compact retained-state mechanisms transfer into family-aware GRP:
   a shared Weibull learning curve, literal replay beyond one epoch, nonlinear
   family coverage, or family-specific replay strength?

Every primary metric comes from five-fold outer CV. Shared shapes, ridge, and
the hierarchical shrinkage for family-specific onsets are selected using only
the corresponding outer training fold. Outer-fold results are checkpointed so
an interrupted run resumes without recomputing completed folds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    generic_family_followup as generic_family,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRS = {
    "production": SCRIPT_DIR / "reference_outputs/production_grp_retained_hybrids_20260713",
    "300m_uncheatable": SCRIPT_DIR / "reference_outputs/300m_grp_retained_hybrids_20260714",
}
OUTER_CV_SEED = 1701
INNER_CV_SEED = 1702
OUTER_SPLITS = 5
INNER_SPLITS = 3
L2_GRID = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
TAU_SHRINK_GRID = (0.0, 1e-7, 1e-6, 1e-5, 1e-4)
TAU_BOUNDS = (0.0, 7.0)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
REFERENCE_MODEL = "power_global_tau"
CANDIDATE_MODEL = "weibull_family_coverage_family_replay"
BOOTSTRAP_SEED = 1703
BOOTSTRAP_DRAWS = 20_000


class DatasetId(StrEnum):
    PRODUCTION = "production"
    THREE_HUNDRED_M_UNCHEATABLE = "300m_uncheatable"


class ResponseKind(StrEnum):
    POWER = "power"
    WEIBULL = "weibull"


class ReplayKind(StrEnum):
    FAMILY_AGGREGATE_GLOBAL_TAU = "family_aggregate_global_tau"
    FAMILY_AGGREGATE_FAMILY_TAU = "family_aggregate_family_tau"
    SHARED_LITERAL = "shared_literal"
    FAMILY_LITERAL = "family_literal"


@dataclass(frozen=True)
class Variant:
    name: str
    response: ResponseKind
    replay: ReplayKind
    family_signal: bool


@dataclass(frozen=True)
class Shape:
    rate: float
    exponent: float
    late_multiplier: float
    forgetting_rate: float
    penalty_threshold: float


@dataclass(frozen=True)
class SelectedHyperparameters:
    shape_index: int
    shape: Shape
    l2: float
    cv_rmse: float


@dataclass(frozen=True)
class FittedModel:
    variant: Variant
    shape: Shape
    l2: float
    head: family_grp.FittedHead
    family_tau: np.ndarray | None
    tau_shrink: float | None


VARIANTS = (
    Variant(
        "power_global_tau",
        ResponseKind.POWER,
        ReplayKind.FAMILY_AGGREGATE_GLOBAL_TAU,
        True,
    ),
    Variant(
        "power_family_tau",
        ResponseKind.POWER,
        ReplayKind.FAMILY_AGGREGATE_FAMILY_TAU,
        True,
    ),
    Variant(
        "power_family_literal_replay",
        ResponseKind.POWER,
        ReplayKind.FAMILY_LITERAL,
        True,
    ),
    Variant(
        "weibull_global_tau",
        ResponseKind.WEIBULL,
        ReplayKind.FAMILY_AGGREGATE_GLOBAL_TAU,
        True,
    ),
    Variant(
        "compact_weibull_shared_replay",
        ResponseKind.WEIBULL,
        ReplayKind.SHARED_LITERAL,
        False,
    ),
    Variant(
        "weibull_family_coverage_shared_replay",
        ResponseKind.WEIBULL,
        ReplayKind.SHARED_LITERAL,
        True,
    ),
    Variant(
        "weibull_bucket_family_replay",
        ResponseKind.WEIBULL,
        ReplayKind.FAMILY_LITERAL,
        False,
    ),
    Variant(
        "weibull_family_coverage_family_replay",
        ResponseKind.WEIBULL,
        ReplayKind.FAMILY_LITERAL,
        True,
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=DatasetId, choices=tuple(DatasetId), default=DatasetId.PRODUCTION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-shapes", type=int, default=32)
    parser.add_argument("--variants", default=",".join(variant.name for variant in VARIANTS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_benchmark_dataset(dataset_id: DatasetId) -> family_grp.Dataset:
    if dataset_id is DatasetId.PRODUCTION:
        return family_grp.load_dataset(family_grp.PRODUCTION_DATA, family_grp.PRODUCTION_MODEL)

    raw = pooled.load_300m_dataset("uncheatable")
    reference = generic_family.load_generic_family_packet()
    if reference.base.domain_names != raw.domain_names:
        raise ValueError("300M family partition and fit panel use different domain orderings")
    family_names = tuple(generic_family.GENERIC_FAMILY_NAMES)
    family_members = tuple(np.asarray(reference.family_map[name], dtype=int) for name in family_names)
    covered = np.concatenate(family_members)
    if sorted(covered.tolist()) != list(range(raw.m)):
        raise ValueError("300M semantic families do not partition the 39 buckets")
    return family_grp.Dataset(
        frame=raw.frame,
        target=np.asarray(raw.y, dtype=float),
        weights=np.asarray(raw.weights, dtype=float),
        c0=np.asarray(raw.c0, dtype=float),
        c1=np.asarray(raw.c1, dtype=float),
        domains=tuple(raw.domain_names),
        family_names=family_names,
        family_members=family_members,
        quality=np.full(raw.m, -1, dtype=int),
    )


def dataset_title(dataset_id: DatasetId) -> str:
    if dataset_id is DatasetId.PRODUCTION:
        return "Production GRP replay hierarchy and compact retained-state grafts"
    return "300M GRP replay hierarchy and compact retained-state grafts"


def split_indices(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    indices: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if dataset_id is DatasetId.PRODUCTION:
        return family_grp.kfold_indices(indices, n_splits, seed)
    local_panel = dataset.frame.iloc[indices].reset_index(drop=True)
    local_splits = component_dsp.panel_stratified_folds(local_panel, n_splits=n_splits, seed=seed)
    return [(indices[train], indices[test]) for train, test in local_splits]


def retained_exposure(dataset: family_grp.Dataset, shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    e0 = phase0_weight * dataset.c0[None, :]
    e1 = phase1_weight * dataset.c1[None, :]
    retained_phase0 = np.exp(-shape.forgetting_rate * (1.0 - phase1_weight)) * e0
    return np.maximum(retained_phase0 + shape.late_multiplier * e1, 0.0), e0 + e1


def response(exposure: np.ndarray, shape: Shape, kind: ResponseKind) -> np.ndarray:
    exposure = np.maximum(exposure, 0.0)
    if kind is ResponseKind.POWER:
        return np.maximum(exposure, 1e-12) ** shape.exponent
    if kind is ResponseKind.WEIBULL:
        return -np.expm1(-((shape.rate * exposure) ** shape.exponent))
    raise ValueError(f"Unsupported response {kind}")


def softplus_penalty(exposure: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    delta = np.log1p(np.maximum(exposure, 0.0)) - threshold
    return np.logaddexp(0.0, delta) ** 2


def nonsingleton_families(dataset: family_grp.Dataset) -> tuple[int, ...]:
    return tuple(index for index, members in enumerate(dataset.family_members) if len(members) > 1)


def build_design(
    dataset: family_grp.Dataset,
    variant: Variant,
    shape: Shape,
    family_tau: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    retained, total_exposure = retained_exposure(dataset, shape)
    family_total = np.column_stack([retained[:, members].sum(axis=1) for members in dataset.family_members])
    pieces = [-response(retained, shape, variant.response)]
    names: list[str] = [f"bucket_signal:{domain}" for domain in dataset.domains]

    if variant.family_signal:
        family_indices = nonsingleton_families(dataset)
        if family_indices:
            if variant.response is ResponseKind.WEIBULL:
                family_exposure = np.column_stack(
                    [retained[:, dataset.family_members[index]].mean(axis=1) for index in family_indices]
                )
            else:
                family_exposure = family_total[:, family_indices]
            pieces.append(-response(family_exposure, shape, variant.response))
            names.extend(f"family_signal:{dataset.family_names[index]}" for index in family_indices)

    if variant.replay is ReplayKind.FAMILY_AGGREGATE_GLOBAL_TAU:
        pieces.append(softplus_penalty(family_total, shape.penalty_threshold))
        names.extend(f"family_penalty:{name}" for name in dataset.family_names)
    elif variant.replay is ReplayKind.FAMILY_AGGREGATE_FAMILY_TAU:
        if family_tau is None or family_tau.shape != (len(dataset.family_names),):
            raise ValueError("Family-specific replay requires one threshold per family")
        pieces.append(softplus_penalty(family_total, family_tau[None, :]))
        names.extend(f"family_penalty:{name}" for name in dataset.family_names)
    else:
        literal_replay = np.maximum(total_exposure - 1.0, 0.0) ** 2
        if variant.replay is ReplayKind.SHARED_LITERAL:
            pieces.append(literal_replay.sum(axis=1, keepdims=True))
            names.append("shared_literal_replay")
        elif variant.replay is ReplayKind.FAMILY_LITERAL:
            pieces.append(
                np.column_stack([literal_replay[:, members].sum(axis=1) for members in dataset.family_members])
            )
            names.extend(f"family_literal_replay:{name}" for name in dataset.family_names)
        else:
            raise ValueError(f"Unsupported replay {variant.replay}")
    return np.hstack(pieces), tuple(names)


def shared_shape_candidates(variant: Variant, count: int) -> tuple[Shape, ...]:
    dimension = 5 if variant.replay is ReplayKind.FAMILY_AGGREGATE_GLOBAL_TAU else 4
    sample_count = 1 << math.ceil(math.log2(max(count, 2)))
    seed = sum(ord(character) for character in variant.name)
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(int(math.log2(sample_count)))[:count]
    candidates: list[Shape] = []
    for row in unit:
        if variant.response is ResponseKind.POWER:
            rate = 1.0
            exponent = float(np.exp(np.log(0.08) + row[0] * (np.log(1.2) - np.log(0.08))))
        else:
            rate = float(np.exp(np.log(0.05) + row[0] * (np.log(20.0) - np.log(0.05))))
            exponent = float(0.2 + 0.8 * row[1])
        offset = 1 if variant.response is ResponseKind.POWER else 2
        late_multiplier = float(np.exp(np.log(0.75) + row[offset] * (np.log(12.0) - np.log(0.75))))
        forgetting_rate = float(np.exp(np.log(1e-5) + row[offset + 1] * (np.log(4.0) - np.log(1e-5))))
        penalty_threshold = (
            float(row[offset + 2] * 7.0) if variant.replay is ReplayKind.FAMILY_AGGREGATE_GLOBAL_TAU else 0.0
        )
        candidates.append(Shape(rate, exponent, late_multiplier, forgetting_rate, penalty_threshold))

    if variant.response is ResponseKind.POWER:
        references = (
            Shape(1.0, 0.33989885260566105, 6.627794351309641, 6.14421235332821e-06, 5.136810831800622),
            Shape(1.0, 0.36662675542192796, 5.422536475313316, 0.031812506876921436, 6.508345540612936),
        )
    else:
        references = (
            Shape(2.79573, 0.661316, 4.73257, 0.0779311, 5.136810831800622),
            Shape(0.25, 0.67, 4.0, 0.1, 4.0),
        )
    return tuple([*references, *candidates])


def select_shared_hyperparameters(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    variant: Variant,
    shapes: tuple[Shape, ...],
    indices: np.ndarray,
    seed: int,
) -> SelectedHyperparameters:
    splits = split_indices(dataset, dataset_id, indices, INNER_SPLITS, seed)
    best: tuple[float, int, float] | None = None
    for shape_index, shape in enumerate(shapes):
        design, names = build_design(dataset, variant, shape)
        for l2 in L2_GRID:
            errors = []
            for train, test in splits:
                head = family_grp.fit_head(design, dataset.target, train, l2, names)
                errors.append(head.predict_design(design[test]) - dataset.target[test])
            score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
            candidate = (score, shape_index, l2)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError(f"No shared-shape candidates for {variant.name}")
    score, shape_index, l2 = best
    return SelectedHyperparameters(shape_index, shapes[shape_index], l2, score)


def fit_shared_model(
    dataset: family_grp.Dataset,
    variant: Variant,
    selected: SelectedHyperparameters,
    indices: np.ndarray,
) -> FittedModel:
    design, names = build_design(dataset, variant, selected.shape)
    head = family_grp.fit_head(design, dataset.target, indices, selected.l2, names)
    return FittedModel(variant, selected.shape, selected.l2, head, None, None)


def replay_coefficient_slice(dataset: family_grp.Dataset, variant: Variant) -> slice:
    width = dataset.m
    if variant.family_signal:
        width += len(nonsingleton_families(dataset))
    return slice(width, width + len(dataset.family_names))


def fit_family_tau_model(
    dataset: family_grp.Dataset,
    variant: Variant,
    shape: Shape,
    l2: float,
    indices: np.ndarray,
    tau_shrink: float,
) -> FittedModel:
    tau_anchor = shape.penalty_threshold
    replay_slice = replay_coefficient_slice(dataset, variant)
    retained, _total = retained_exposure(dataset, shape)
    family_total = np.column_stack([retained[:, members].sum(axis=1) for members in dataset.family_members])
    train_family_total = family_total[indices]
    logged_family_total = np.log1p(train_family_total)
    initial_tau_values = (
        np.full(len(dataset.family_names), tau_anchor, dtype=float),
        np.quantile(logged_family_total, 0.5, axis=0),
        np.quantile(logged_family_total, 0.75, axis=0),
        np.quantile(logged_family_total, 0.9, axis=0),
    )

    def objective_and_gradient(tau: np.ndarray) -> tuple[float, np.ndarray]:
        design, names = build_design(dataset, variant, shape, tau)
        head = family_grp.fit_head(design, dataset.target, indices, l2, names)
        residual = head.predict_design(design[indices]) - dataset.target[indices]
        coefficients = head.coefficients[replay_slice]
        delta = np.log1p(train_family_total) - tau[None, :]
        softplus = np.logaddexp(0.0, delta)
        derivative = -2.0 * softplus / (1.0 + np.exp(-delta))
        data_gradient = 2.0 * np.mean(
            residual[:, None] * coefficients[None, :] * derivative,
            axis=0,
        )
        displacement = tau - tau_anchor
        shrink_loss = tau_shrink * float(np.mean(displacement**2))
        shrink_gradient = 2.0 * tau_shrink * displacement / len(displacement)
        ridge_loss = l2 * float(np.sum(head.coefficients**2)) / len(indices)
        loss = float(np.mean(residual**2)) + ridge_loss + shrink_loss
        return loss, data_gradient + shrink_gradient

    results = [
        minimize(
            objective_and_gradient,
            np.clip(initial_tau, *TAU_BOUNDS),
            method="L-BFGS-B",
            jac=True,
            bounds=[TAU_BOUNDS] * len(initial_tau),
            options={"maxiter": 80, "ftol": 1e-12, "maxls": 30},
        )
        for initial_tau in initial_tau_values
    ]
    finite_results = [result for result in results if np.isfinite(result.fun)]
    if not finite_results:
        raise RuntimeError("Family-threshold optimization returned a non-finite objective")
    result = min(finite_results, key=lambda candidate: float(candidate.fun))
    family_tau = np.asarray(result.x, dtype=float)
    design, names = build_design(dataset, variant, shape, family_tau)
    head = family_grp.fit_head(design, dataset.target, indices, l2, names)
    return FittedModel(variant, shape, l2, head, family_tau, tau_shrink)


def select_tau_shrink(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    variant: Variant,
    selected: SelectedHyperparameters,
    indices: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    splits = split_indices(dataset, dataset_id, indices, INNER_SPLITS, seed)
    best: tuple[float, float] | None = None
    for tau_shrink in TAU_SHRINK_GRID:
        errors = []
        for train, test in splits:
            model = fit_family_tau_model(
                dataset,
                variant,
                selected.shape,
                selected.l2,
                train,
                tau_shrink,
            )
            design, _names = build_design(dataset, variant, selected.shape, model.family_tau)
            errors.append(model.head.predict_design(design[test]) - dataset.target[test])
        score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        candidate = (score, tau_shrink)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("No family-threshold shrinkage candidate")
    return best[1], best[0]


def model_prediction(dataset: family_grp.Dataset, model: FittedModel, indices: np.ndarray) -> np.ndarray:
    design, _names = build_design(dataset, model.variant, model.shape, model.family_tau)
    return model.head.predict_design(design[indices])


def checkpoint_paths(output_dir: Path, variant: Variant, fold: int) -> tuple[Path, Path]:
    stem = output_dir / "checkpoints" / f"{variant.name}__outer_{fold}"
    return stem.with_suffix(".json"), stem.with_suffix(".npy")


def load_fold_checkpoint(
    output_dir: Path,
    variant: Variant,
    fold: int,
    test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    metadata_path, prediction_path = checkpoint_paths(output_dir, variant, fold)
    if not metadata_path.exists() or not prediction_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata["test_indices"] != test.tolist():
        raise ValueError(f"Stale checkpoint indices in {metadata_path}")
    prediction = np.load(prediction_path)
    if prediction.shape != test.shape:
        raise ValueError(f"Stale checkpoint prediction shape in {prediction_path}")
    return prediction, metadata


def save_fold_checkpoint(
    output_dir: Path,
    variant: Variant,
    fold: int,
    test: np.ndarray,
    prediction: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    metadata_path, prediction_path = checkpoint_paths(output_dir, variant, fold)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, prediction)
    metadata_path.write_text(json.dumps({"test_indices": test.tolist(), **metadata}, indent=2, allow_nan=False) + "\n")


def nested_oof(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    variant: Variant,
    shapes: tuple[Shape, ...],
    output_dir: Path,
    *,
    force: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    outer = split_indices(dataset, dataset_id, np.arange(dataset.n), OUTER_SPLITS, OUTER_CV_SEED)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    base_variant = VARIANT_BY_NAME["power_global_tau"]
    base_shapes = shared_shape_candidates(base_variant, len(shapes) - 2)
    for fold, (train, test) in enumerate(outer):
        if not force and (cached := load_fold_checkpoint(output_dir, variant, fold, test)) is not None:
            prediction[test], metadata = cached
            selections.append(metadata)
            continue
        print(f"{variant.name}: outer fold {fold + 1}/{len(outer)}", flush=True)
        if variant.replay is ReplayKind.FAMILY_AGGREGATE_FAMILY_TAU:
            selected = select_shared_hyperparameters(
                dataset,
                dataset_id,
                base_variant,
                base_shapes,
                train,
                INNER_CV_SEED + fold,
            )
            tau_shrink, tau_cv_rmse = select_tau_shrink(
                dataset,
                dataset_id,
                variant,
                selected,
                train,
                INNER_CV_SEED + 100 + fold,
            )
            model = fit_family_tau_model(
                dataset,
                variant,
                selected.shape,
                selected.l2,
                train,
                tau_shrink,
            )
            metadata = {
                "outer_fold": fold,
                "shape_index": selected.shape_index,
                "shape": asdict(selected.shape),
                "l2": selected.l2,
                "base_cv_rmse": selected.cv_rmse,
                "tau_shrink": tau_shrink,
                "tau_cv_rmse": tau_cv_rmse,
                "family_tau": model.family_tau.tolist(),
            }
        else:
            selected = select_shared_hyperparameters(
                dataset,
                dataset_id,
                variant,
                shapes,
                train,
                INNER_CV_SEED + fold,
            )
            model = fit_shared_model(dataset, variant, selected, train)
            metadata = {
                "outer_fold": fold,
                "shape_index": selected.shape_index,
                "shape": asdict(selected.shape),
                "l2": selected.l2,
                "cv_rmse": selected.cv_rmse,
            }
        fold_prediction = model_prediction(dataset, model, test)
        prediction[test] = fold_prediction
        save_fold_checkpoint(output_dir, variant, fold, test, fold_prediction, metadata)
        selections.append({"test_indices": test.tolist(), **metadata})
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested OOF prediction for {variant.name}")
    return prediction, [test for _train, test in outer], selections


def parameter_count(dataset: family_grp.Dataset, variant: Variant) -> int:
    linear = 1 + dataset.m
    if variant.family_signal:
        linear += len(nonsingleton_families(dataset))
    if variant.replay is ReplayKind.SHARED_LITERAL:
        linear += 1
    else:
        linear += len(dataset.family_names)
    nonlinear = 3 if variant.response is ResponseKind.POWER else 4
    if variant.replay is ReplayKind.FAMILY_AGGREGATE_GLOBAL_TAU:
        nonlinear += 1
    elif variant.replay is ReplayKind.FAMILY_AGGREGATE_FAMILY_TAU:
        nonlinear += len(dataset.family_names)
    return linear + nonlinear


def fit_full_model(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    variant: Variant,
    shapes: tuple[Shape, ...],
) -> tuple[FittedModel, dict[str, Any]]:
    indices = np.arange(dataset.n)
    base_variant = VARIANT_BY_NAME["power_global_tau"]
    if variant.replay is ReplayKind.FAMILY_AGGREGATE_FAMILY_TAU:
        base_shapes = shared_shape_candidates(base_variant, len(shapes) - 2)
        selected = select_shared_hyperparameters(
            dataset,
            dataset_id,
            base_variant,
            base_shapes,
            indices,
            INNER_CV_SEED + 999,
        )
        tau_shrink, tau_cv_rmse = select_tau_shrink(
            dataset,
            dataset_id,
            variant,
            selected,
            indices,
            INNER_CV_SEED + 1999,
        )
        model = fit_family_tau_model(
            dataset,
            variant,
            selected.shape,
            selected.l2,
            indices,
            tau_shrink,
        )
        metadata = {
            "shape": asdict(selected.shape),
            "l2": selected.l2,
            "base_cv_rmse": selected.cv_rmse,
            "tau_shrink": tau_shrink,
            "tau_cv_rmse": tau_cv_rmse,
            "family_tau": dict(zip(dataset.family_names, model.family_tau.tolist(), strict=True)),
        }
        return model, metadata
    selected = select_shared_hyperparameters(dataset, dataset_id, variant, shapes, indices, INNER_CV_SEED + 999)
    model = fit_shared_model(dataset, variant, selected, indices)
    return model, {"shape": asdict(selected.shape), "l2": selected.l2, "cv_rmse": selected.cv_rmse}


def plot_diagnostics(
    dataset: family_grp.Dataset,
    title: str,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    ordered = metrics.sort_values("rmse")
    palette = sample_colorscale("RdYlGn_r", np.linspace(0.05, 0.95, len(ordered)))
    colors = dict(zip(ordered["model"], palette[: len(ordered)], strict=True))
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Nested OOF RMSE", "Nested OOF Spearman", "Observed vs nested OOF"),
        column_widths=(0.22, 0.22, 0.56),
    )
    figure.add_bar(
        x=ordered["model"],
        y=ordered["rmse"],
        marker_color=[colors[name] for name in ordered["model"]],
        showlegend=False,
        row=1,
        col=1,
    )
    figure.add_bar(
        x=ordered["model"],
        y=ordered["spearman"],
        marker_color=[colors[name] for name in ordered["model"]],
        showlegend=False,
        row=1,
        col=2,
    )
    name_column = next(
        (name for name in ("candidate_name", "run_name", "candidate_run_name") if name in dataset.frame),
        None,
    )
    if name_column is None:
        raise ValueError("Dataset has no candidate-name column for plot tooltips")
    for model, frame in predictions.groupby("model", sort=False):
        figure.add_scatter(
            x=frame["observed"],
            y=frame["prediction"],
            mode="markers",
            marker={"size": 5, "opacity": 0.45, "color": colors[model]},
            name=model,
            customdata=np.column_stack([dataset.frame.iloc[frame["row_index"]][name_column]]),
            hovertemplate="%{customdata[0]}<br>observed=%{x:.6f}<br>predicted=%{y:.6f}<extra></extra>",
            row=1,
            col=3,
        )
    observed_min = float(predictions["observed"].min())
    observed_max = float(predictions["observed"].max())
    figure.add_shape(
        type="line",
        x0=observed_min,
        y0=observed_min,
        x1=observed_max,
        y1=observed_max,
        line={"color": "#777", "dash": "dot"},
        row=1,
        col=3,
    )
    figure.update_xaxes(tickangle=-35, row=1, col=1)
    figure.update_xaxes(tickangle=-35, row=1, col=2)
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=760,
        width=1900,
        margin={"b": 240},
    )
    figure.write_html(output_dir / "fit_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def paired_comparison(predictions: pd.DataFrame) -> dict[str, Any] | None:
    prediction_by_model = predictions.pivot(index="row_index", columns="model", values="prediction")
    if REFERENCE_MODEL not in prediction_by_model or CANDIDATE_MODEL not in prediction_by_model:
        return None
    observed = predictions.drop_duplicates("row_index").set_index("row_index")["observed"]
    reference_error = prediction_by_model[REFERENCE_MODEL] - observed
    candidate_error = prediction_by_model[CANDIDATE_MODEL] - observed
    squared_error_delta = candidate_error.to_numpy() ** 2 - reference_error.to_numpy() ** 2

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_means = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    batch_size = 1_000
    for start in range(0, BOOTSTRAP_DRAWS, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, len(squared_error_delta), size=(stop - start, len(squared_error_delta)))
        bootstrap_means[start:stop] = squared_error_delta[indices].mean(axis=1)

    return {
        "reference_model": REFERENCE_MODEL,
        "candidate_model": CANDIDATE_MODEL,
        "mean_squared_error_delta": float(squared_error_delta.mean()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_95_percent_interval": np.quantile(bootstrap_means, [0.025, 0.975]).tolist(),
        "bootstrap_probability_candidate_better": float(np.mean(bootstrap_means < 0.0)),
    }


def write_report(
    dataset_id: DatasetId,
    title: str,
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    comparison: dict[str, Any] | None,
    full_models: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    best = metrics.sort_values(["rmse", "lower_tail_optimism"]).iloc[0]
    matched_lines: list[str] = []
    if comparison is not None:
        fold_rmse = fold_metrics.pivot(index="outer_fold", columns="model", values="rmse")
        reference_model = str(comparison["reference_model"])
        candidate_model = str(comparison["candidate_model"])
        fold_wins = int((fold_rmse[candidate_model] < fold_rmse[reference_model]).sum())
        fold_count = len(fold_rmse)
        rmse_by_model = metrics.set_index("model")["rmse"]
        candidate_rmse_delta = float(rmse_by_model[candidate_model] / rmse_by_model[reference_model] - 1.0)
        compact_rmse = float(rmse_by_model["compact_weibull_shared_replay"])
        coverage_delta = float(rmse_by_model["weibull_family_coverage_shared_replay"] / compact_rmse - 1.0)
        family_replay_delta = float(rmse_by_model["weibull_bucket_family_replay"] / compact_rmse - 1.0)
        combined_delta = float(rmse_by_model["weibull_family_coverage_family_replay"] / compact_rmse - 1.0)
        matched_lines = [
            "## Matched comparison",
            "",
            fold_rmse.to_markdown(),
            "",
            f"Relative to `{reference_model}`, `{candidate_model}` changes nested-OOF RMSE by "
            f"{candidate_rmse_delta:+.2%} and wins {fold_wins}/{fold_count} outer folds. Its mean squared-error "
            f"difference is {comparison['mean_squared_error_delta']:.3e}; a {comparison['bootstrap_draws']:,}-draw "
            "row bootstrap gives a 95% interval of "
            f"[{comparison['bootstrap_95_percent_interval'][0]:.3e}, "
            f"{comparison['bootstrap_95_percent_interval'][1]:.3e}] and places "
            f"{comparison['bootstrap_probability_candidate_better']:.2%} of draws below zero. "
            "This is a paired diagnostic over sampled designs, not a formal independence-robust test.",
            "",
            "Relative to compact retained state, family coverage alone changes RMSE by "
            f"{coverage_delta:+.2%}, family-specific replay alone by {family_replay_delta:+.2%}, and their "
            f"combination by {combined_delta:+.2%}. The full-data family-onset fit "
            "collapses to the shared onset when all learned family onsets equal the selected shared value.",
            "",
        ]
    lines = [
        f"# {title}",
        "",
        "## Shared retained state",
        "",
        "$$z_i=\\exp[-\\lambda(1-w_i^{(1)})]e_i^{(0)}+\\eta e_i^{(1)}," "\\qquad q_i=e_i^{(0)}+e_i^{(1)}.$$",
        "",
        "Power variants use $S(x)=x^a$; compact variants use "
        "$S(x)=1-\\exp[-(\\rho x)^p]$. Bucket-specific nonnegative amplitudes multiply $S(z_i)$. "
        "Family-aware variants add one nonnegative coverage amplitude per nonsingleton bucket family.",
        "",
        "The original GRP replay term is $B_C\\operatorname{softplus}(\\log(1+Z_C)-\\tau)^2$ "
        "for $Z_C=\\sum_{i\\in C}z_i$. The family-onset extension replaces the shared $\\tau$ by "
        "$\\tau_C$, hierarchically shrunk toward the selected shared onset. Literal-replay variants use "
        "$R_C=\\sum_{i\\in C}[q_i-1]_+^2$, preserving compact retained state's separation between useful "
        "retained learning and actual repeated data.",
        "",
        "## Nested OOF metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        *matched_lines,
        "## Selected full-data fits",
        "",
        "~~~json",
        json.dumps(full_models, indent=2, allow_nan=False),
        "~~~",
        "",
        "## Result",
        "",
        f"The lowest nested-OOF RMSE is **{best['model']}** at {best['rmse']:.6f}. "
        "Selection and lower-tail diagnostics must agree before promoting a model to mixture validation.",
        "",
        "## Reproduce",
        "",
        "~~~bash",
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        f"benchmark_production_grp_retained_hybrids_20260713.py --dataset {dataset_id}",
        "~~~",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[str(args.dataset)]
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    unknown = set(requested) - set(VARIANT_BY_NAME)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    variants = [VARIANT_BY_NAME[name] for name in requested]
    dataset = load_benchmark_dataset(args.dataset)
    title = dataset_title(args.dataset)

    metric_rows = []
    fold_metric_rows = []
    prediction_rows = []
    selection_rows = []
    full_models: dict[str, dict[str, Any]] = {}
    for variant in variants:
        shapes = shared_shape_candidates(variant, args.num_shapes)
        prediction, fold_indices, selections = nested_oof(
            dataset,
            args.dataset,
            variant,
            shapes,
            output_dir,
            force=args.force,
        )
        summary = family_grp.metric_summary(dataset.target, prediction, fold_indices)
        metric_rows.append(
            {
                "model": variant.name,
                "parameter_count": parameter_count(dataset, variant),
                **summary,
            }
        )
        for outer_fold, test in enumerate(fold_indices):
            residual = prediction[test] - dataset.target[test]
            fold_metric_rows.append(
                {
                    "model": variant.name,
                    "outer_fold": outer_fold,
                    "n": len(test),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mae": float(np.mean(np.abs(residual))),
                }
            )
        for row_index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True)):
            prediction_rows.append(
                {
                    "model": variant.name,
                    "row_index": row_index,
                    "observed": observed,
                    "prediction": predicted,
                    "residual": predicted - observed,
                }
            )
        selection_rows.extend({"model": variant.name, **selection} for selection in selections)
        full_model, full_metadata = fit_full_model(dataset, args.dataset, variant, shapes)
        full_metadata["parameter_count"] = parameter_count(dataset, variant)
        full_metadata["active_coefficient_count"] = int(np.sum(full_model.head.coefficients > 1e-10))
        full_models[variant.name] = full_metadata

    metrics = pd.DataFrame(metric_rows).sort_values(["rmse", "lower_tail_optimism"])
    fold_metrics = pd.DataFrame(fold_metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    comparison = paired_comparison(predictions)
    metrics.to_csv(output_dir / "nested_oof_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "nested_oof_predictions.csv", index=False)
    selections.to_json(output_dir / "nested_cv_selections.json", orient="records", indent=2)
    if comparison is not None:
        (output_dir / "paired_comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    (output_dir / "full_models.json").write_text(json.dumps(full_models, indent=2, allow_nan=False) + "\n")
    plot_diagnostics(dataset, title, metrics, predictions, output_dir)
    write_report(args.dataset, title, metrics, fold_metrics, comparison, full_models, output_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
