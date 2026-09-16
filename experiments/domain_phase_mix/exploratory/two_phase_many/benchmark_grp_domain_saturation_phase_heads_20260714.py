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
"""Benchmark learned domain saturation and independent phase heads in family GRP.

This is a controlled 2x2 comparison inside Bucket-resolved family GRP:

1. shared power response versus shrinkage-pooled, target-learned per-domain
   Weibull saturation;
2. retained exposure with one scalar late multiplier versus independent early
   and late response heads.

The family response and family replay-harm channels are preserved in every
corner. Shape, ridge, and domain-rate shrinkage are selected inside each outer
fold. Fold checkpoints make the complete analysis safely resumable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_grp_saturation_hierarchy_20260714 as hierarchy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_production_grp_retained_hybrids_20260713 as retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/grp_domain_saturation_phase_heads_20260714"
OUTER_CV_SEED = 3141
INNER_CV_SEED = 3142
DEFAULT_OUTER_SPLITS = 5
DEFAULT_INNER_SPLITS = 3
L2_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0)
RATE_SHRINK_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
RATE_BOUNDS = (1e-4, 100.0)
CHECKPOINT_SCHEMA = 1
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class DatasetId(StrEnum):
    PRODUCTION_UNCHEATABLE = "production_uncheatable"
    THREE_HUNDRED_M_UNCHEATABLE = "300m_uncheatable"
    THREE_HUNDRED_M_TABLE9 = "300m_table9"


class ResponseKind(StrEnum):
    POWER = "power"
    DOMAIN_WEIBULL = "domain_weibull"


class PhaseKind(StrEnum):
    ETA = "eta"
    SEPARATE_HEADS = "separate_heads"


@dataclass(frozen=True)
class Variant:
    name: str
    response: ResponseKind
    phase: PhaseKind


@dataclass(frozen=True)
class DesignLayout:
    bucket_slices: tuple[slice, ...]
    family_slices: tuple[slice, ...]
    nonsingleton_families: tuple[int, ...]


@dataclass(frozen=True)
class SharedSelection:
    shape: retained.Shape
    l2: float
    inner_rmse: float


@dataclass(frozen=True)
class RateModel:
    log_rates: np.ndarray
    head: family_grp.FittedHead
    objective: float
    iterations: int
    converged: bool


VARIANTS = (
    Variant("power_eta", ResponseKind.POWER, PhaseKind.ETA),
    Variant("power_separate_heads", ResponseKind.POWER, PhaseKind.SEPARATE_HEADS),
    Variant("domain_weibull_eta", ResponseKind.DOMAIN_WEIBULL, PhaseKind.ETA),
    Variant("domain_weibull_separate_heads", ResponseKind.DOMAIN_WEIBULL, PhaseKind.SEPARATE_HEADS),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
REFERENCE_VARIANT = VARIANT_BY_NAME["power_eta"]
COMPARISON_PAIRS = (
    ("power_eta", "power_separate_heads", "replace eta with phase heads under power response"),
    ("power_eta", "domain_weibull_eta", "learn domain saturation under eta"),
    (
        "power_separate_heads",
        "domain_weibull_separate_heads",
        "learn domain saturation under phase heads",
    ),
    (
        "domain_weibull_eta",
        "domain_weibull_separate_heads",
        "replace eta with phase heads under learned saturation",
    ),
    ("power_eta", "domain_weibull_separate_heads", "joint saturation and phase-head intervention"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(dataset.value for dataset in DatasetId),
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=16)
    parser.add_argument("--outer-splits", type=int, default=DEFAULT_OUTER_SPLITS)
    parser.add_argument("--inner-splits", type=int, default=DEFAULT_INNER_SPLITS)
    parser.add_argument("--rate-maxiter", type=int, default=40)
    parser.add_argument("--check-gradient", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_dataset(dataset_id: DatasetId) -> family_grp.Dataset:
    return hierarchy.load_dataset(hierarchy.DatasetId(dataset_id.value))


def split_indices(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    indices: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return hierarchy.split_indices(
        dataset,
        hierarchy.DatasetId(dataset_id.value),
        indices,
        n_splits,
        seed,
    )


def candidate_shapes(variant: Variant, count: int) -> tuple[retained.Shape, ...]:
    base_name = "power_global_tau" if variant.response is ResponseKind.POWER else "weibull_global_tau"
    base_variant = retained.VARIANT_BY_NAME[base_name]
    shapes = retained.shared_shape_candidates(base_variant, count)
    if variant.phase is PhaseKind.SEPARATE_HEADS:
        shapes = tuple(replace(shape, late_multiplier=1.0) for shape in shapes)
    unique: list[retained.Shape] = []
    seen: set[tuple[float, ...]] = set()
    for shape in shapes:
        key = tuple(round(value, 12) for value in asdict(shape).values())
        if key in seen:
            continue
        seen.add(key)
        unique.append(shape)
    return tuple(unique)


def phase_exposures(
    dataset: family_grp.Dataset,
    shape: retained.Shape,
    phase: PhaseKind,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    early_raw = phase0_weight * dataset.c0[None, :]
    late = phase1_weight * dataset.c1[None, :]
    retained_early = np.exp(-shape.forgetting_rate * (1.0 - phase1_weight)) * early_raw
    if phase is PhaseKind.ETA:
        combined = np.maximum(retained_early + shape.late_multiplier * late, 0.0)
        return (combined,), combined
    if phase is PhaseKind.SEPARATE_HEADS:
        early = np.maximum(retained_early, 0.0)
        late = np.maximum(late, 0.0)
        return (early, late), early + late
    raise ValueError(f"Unsupported phase kind {phase}")


def response(
    exposure: np.ndarray,
    shape: retained.Shape,
    kind: ResponseKind,
    rates: np.ndarray | None,
) -> np.ndarray:
    exposure = np.maximum(exposure, 0.0)
    if kind is ResponseKind.POWER:
        return np.maximum(exposure, 1e-12) ** shape.exponent
    if rates is None:
        raise ValueError("Domain Weibull response requires rates")
    scaled = exposure * rates[None, :]
    return -np.expm1(-(scaled**shape.exponent))


def response_log_rate_derivative(
    exposure: np.ndarray,
    shape: retained.Shape,
    rates: np.ndarray,
) -> np.ndarray:
    scaled_power = (np.maximum(exposure, 0.0) * rates[None, :]) ** shape.exponent
    return shape.exponent * scaled_power * np.exp(-scaled_power)


def family_rates(dataset: family_grp.Dataset, log_rates: np.ndarray) -> np.ndarray:
    return np.exp(
        np.asarray(
            [float(np.mean(log_rates[members])) for members in dataset.family_members],
            dtype=float,
        )
    )


def build_design(
    dataset: family_grp.Dataset,
    variant: Variant,
    shape: retained.Shape,
    log_rates: np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...], DesignLayout]:
    heads, replay_exposure = phase_exposures(dataset, shape, variant.phase)
    nonsingleton = retained.nonsingleton_families(dataset)
    bucket_rates = None if log_rates is None else np.exp(log_rates)
    induced_family_rates = None if log_rates is None else family_rates(dataset, log_rates)
    pieces: list[np.ndarray] = []
    names: list[str] = []
    bucket_slices: list[slice] = []
    family_slices: list[slice] = []
    offset = 0

    for head_index, exposure in enumerate(heads):
        bucket_signal = response(exposure, shape, variant.response, bucket_rates)
        pieces.append(-bucket_signal)
        bucket_slices.append(slice(offset, offset + dataset.m))
        names.extend(f"phase{head_index}:bucket_signal:{domain}" for domain in dataset.domains)
        offset += dataset.m

        if nonsingleton:
            if variant.response is ResponseKind.POWER:
                family_exposure = np.column_stack(
                    [exposure[:, dataset.family_members[index]].sum(axis=1) for index in nonsingleton]
                )
                response_rates = None
            else:
                family_exposure = np.column_stack(
                    [exposure[:, dataset.family_members[index]].mean(axis=1) for index in nonsingleton]
                )
                if induced_family_rates is None:
                    raise ValueError("Domain Weibull family response requires induced family rates")
                response_rates = induced_family_rates[np.asarray(nonsingleton, dtype=int)]
            family_signal = response(family_exposure, shape, variant.response, response_rates)
            pieces.append(-family_signal)
            family_slices.append(slice(offset, offset + len(nonsingleton)))
            names.extend(f"phase{head_index}:family_signal:{dataset.family_names[index]}" for index in nonsingleton)
            offset += len(nonsingleton)

    family_total = np.column_stack([replay_exposure[:, members].sum(axis=1) for members in dataset.family_members])
    pieces.append(retained.softplus_penalty(family_total, shape.penalty_threshold))
    names.extend(f"family_penalty:{name}" for name in dataset.family_names)
    layout = DesignLayout(tuple(bucket_slices), tuple(family_slices), nonsingleton)
    return np.hstack(pieces), tuple(names), layout


def prediction_log_rate_derivative(
    dataset: family_grp.Dataset,
    variant: Variant,
    shape: retained.Shape,
    log_rates: np.ndarray,
    head: family_grp.FittedHead,
    layout: DesignLayout,
) -> np.ndarray:
    heads, _replay_exposure = phase_exposures(dataset, shape, variant.phase)
    rates = np.exp(log_rates)
    induced_family_rates = family_rates(dataset, log_rates)
    derivative = np.zeros((dataset.n, dataset.m), dtype=float)
    for head_index, exposure in enumerate(heads):
        bucket_derivative = response_log_rate_derivative(exposure, shape, rates)
        bucket_coefficients = head.coefficients[layout.bucket_slices[head_index]]
        derivative -= bucket_derivative * bucket_coefficients[None, :]

        if not layout.nonsingleton_families:
            continue
        family_exposure = np.column_stack(
            [exposure[:, dataset.family_members[index]].mean(axis=1) for index in layout.nonsingleton_families]
        )
        selected_rates = induced_family_rates[np.asarray(layout.nonsingleton_families, dtype=int)]
        family_derivative = response_log_rate_derivative(family_exposure, shape, selected_rates)
        family_coefficients = head.coefficients[layout.family_slices[head_index]]
        for local_index, family_index in enumerate(layout.nonsingleton_families):
            members = dataset.family_members[family_index]
            contribution = -family_coefficients[local_index] * family_derivative[:, local_index] / len(members)
            derivative[:, members] += contribution[:, None]
    return derivative


def select_shared_hyperparameters(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    variant: Variant,
    shapes: tuple[retained.Shape, ...],
    indices: np.ndarray,
    seed: int,
    inner_splits: int,
) -> SharedSelection:
    splits = split_indices(dataset, dataset_id, indices, inner_splits, seed)
    best: tuple[float, int, float] | None = None
    for shape_index, shape in enumerate(shapes):
        anchor_log_rates = (
            None if variant.response is ResponseKind.POWER else np.full(dataset.m, math.log(shape.rate), dtype=float)
        )
        design, names, _layout = build_design(dataset, variant, shape, anchor_log_rates)
        for l2 in L2_GRID:
            errors: list[np.ndarray] = []
            for train, test in splits:
                head = family_grp.fit_head(design, dataset.target, train, l2, names)
                errors.append(head.predict_design(design[test]) - dataset.target[test])
            score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
            candidate = (score, shape_index, l2)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError(f"No shared hyperparameter candidate for {variant.name}")
    return SharedSelection(shapes[best[1]], best[2], best[0])


def rate_objective_and_gradient(
    log_rates: np.ndarray,
    dataset: family_grp.Dataset,
    variant: Variant,
    selection: SharedSelection,
    indices: np.ndarray,
    rate_shrink: float,
) -> tuple[float, np.ndarray]:
    design, names, layout = build_design(dataset, variant, selection.shape, log_rates)
    head = family_grp.fit_head(design, dataset.target, indices, selection.l2, names)
    residual = head.predict_design(design[indices]) - dataset.target[indices]
    prediction_derivative = prediction_log_rate_derivative(
        dataset,
        variant,
        selection.shape,
        log_rates,
        head,
        layout,
    )[indices]
    data_gradient = 2.0 * np.mean(residual[:, None] * prediction_derivative, axis=0)
    anchor = math.log(selection.shape.rate)
    displacement = log_rates - anchor
    shrink_loss = rate_shrink * float(np.mean(displacement**2))
    shrink_gradient = 2.0 * rate_shrink * displacement / len(displacement)
    ridge_loss = selection.l2 * float(np.sum(head.coefficients**2)) / len(indices)
    loss = float(np.mean(residual**2)) + ridge_loss + shrink_loss
    return loss, data_gradient + shrink_gradient


def fit_domain_rate_model(
    dataset: family_grp.Dataset,
    variant: Variant,
    selection: SharedSelection,
    indices: np.ndarray,
    rate_shrink: float,
    maxiter: int,
    multistart: bool,
) -> RateModel:
    anchor = math.log(selection.shape.rate)
    starts = [np.full(dataset.m, anchor, dtype=float)]
    if multistart:
        heads, _replay = phase_exposures(dataset, selection.shape, variant.phase)
        exposure = np.sum(np.stack(heads, axis=0), axis=0)
        prior = hierarchy.inverse_median_prior(exposure, indices)
        starts.append(np.clip(anchor + 0.5 * prior, math.log(RATE_BOUNDS[0]), math.log(RATE_BOUNDS[1])))
    bounds = [(math.log(RATE_BOUNDS[0]), math.log(RATE_BOUNDS[1]))] * dataset.m
    results = [
        minimize(
            rate_objective_and_gradient,
            start,
            args=(dataset, variant, selection, indices, rate_shrink),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-11, "maxls": 30},
        )
        for start in starts
    ]
    finite = [result for result in results if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError(f"Domain-rate optimization failed for {variant.name}")
    result = min(finite, key=lambda candidate: float(candidate.fun))
    log_rates = np.asarray(result.x, dtype=float)
    design, names, _layout = build_design(dataset, variant, selection.shape, log_rates)
    head = family_grp.fit_head(design, dataset.target, indices, selection.l2, names)
    return RateModel(
        log_rates=log_rates,
        head=head,
        objective=float(result.fun),
        iterations=int(result.nit),
        converged=bool(result.success),
    )


def select_rate_shrink(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    variant: Variant,
    selection: SharedSelection,
    indices: np.ndarray,
    seed: int,
    inner_splits: int,
    rate_maxiter: int,
) -> tuple[float, float]:
    splits = split_indices(dataset, dataset_id, indices, inner_splits, seed)
    best: tuple[float, float] | None = None
    for rate_shrink in RATE_SHRINK_GRID:
        errors: list[np.ndarray] = []
        for train, test in splits:
            model = fit_domain_rate_model(
                dataset,
                variant,
                selection,
                train,
                rate_shrink,
                max(12, rate_maxiter // 2),
                False,
            )
            design, _names, _layout = build_design(dataset, variant, selection.shape, model.log_rates)
            errors.append(model.head.predict_design(design[test]) - dataset.target[test])
        score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        candidate = (score, rate_shrink)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError(f"No rate-shrink candidate for {variant.name}")
    return best[1], best[0]


def model_prediction(
    dataset: family_grp.Dataset,
    variant: Variant,
    selection: SharedSelection,
    head: family_grp.FittedHead,
    log_rates: np.ndarray | None,
    indices: np.ndarray,
) -> np.ndarray:
    design, _names, _layout = build_design(dataset, variant, selection.shape, log_rates)
    return head.predict_design(design[indices])


def checkpoint_paths(
    output_dir: Path,
    dataset_id: DatasetId,
    variant: Variant,
    fold: int,
) -> tuple[Path, Path]:
    stem = output_dir / "checkpoints" / f"{dataset_id.value}__{variant.name}__outer_{fold}"
    return stem.with_suffix(".json"), stem.with_suffix(".npy")


def load_checkpoint(
    output_dir: Path,
    dataset_id: DatasetId,
    variant: Variant,
    fold: int,
    test: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    metadata_path, prediction_path = checkpoint_paths(output_dir, dataset_id, variant, fold)
    if not metadata_path.exists() or not prediction_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    expected = {"schema": CHECKPOINT_SCHEMA, "test_indices": test.tolist(), **config}
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    prediction = np.load(prediction_path)
    if prediction.shape != test.shape:
        return None
    return np.asarray(prediction, dtype=float), metadata["selection"]


def save_checkpoint(
    output_dir: Path,
    dataset_id: DatasetId,
    variant: Variant,
    fold: int,
    test: np.ndarray,
    prediction: np.ndarray,
    selection: dict[str, Any],
    config: dict[str, Any],
) -> None:
    metadata_path, prediction_path = checkpoint_paths(output_dir, dataset_id, variant, fold)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, prediction)
    metadata_path.write_text(
        json.dumps(
            {
                "schema": CHECKPOINT_SCHEMA,
                "test_indices": test.tolist(),
                **config,
                "selection": selection,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def nested_oof(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    variant: Variant,
    shapes: tuple[retained.Shape, ...],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    outer = split_indices(
        dataset,
        dataset_id,
        np.arange(dataset.n),
        args.outer_splits,
        OUTER_CV_SEED,
    )
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    config = {
        "num_shapes": args.num_shapes,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "l2_grid": list(L2_GRID),
        "rate_shrink_grid": list(RATE_SHRINK_GRID),
        "rate_maxiter": args.rate_maxiter,
    }
    for fold, (train, test) in enumerate(outer):
        cached = (
            None
            if args.force
            else load_checkpoint(
                output_dir,
                dataset_id,
                variant,
                fold,
                test,
                config,
            )
        )
        if cached is not None:
            fold_prediction, selection_row = cached
            prediction[test] = fold_prediction
            selections.append(selection_row)
            continue

        print(
            f"{dataset_id.value} {variant.name}: outer fold {fold + 1}/{len(outer)}",
            flush=True,
        )
        shared = select_shared_hyperparameters(
            dataset,
            dataset_id,
            variant,
            shapes,
            train,
            INNER_CV_SEED + fold,
            args.inner_splits,
        )
        rate_shrink: float | None = None
        rate_inner_rmse: float | None = None
        log_rates: np.ndarray | None = None
        rate_model: RateModel | None = None
        if variant.response is ResponseKind.DOMAIN_WEIBULL:
            rate_shrink, rate_inner_rmse = select_rate_shrink(
                dataset,
                dataset_id,
                variant,
                shared,
                train,
                INNER_CV_SEED + 100 + fold,
                args.inner_splits,
                args.rate_maxiter,
            )
            rate_model = fit_domain_rate_model(
                dataset,
                variant,
                shared,
                train,
                rate_shrink,
                args.rate_maxiter,
                True,
            )
            head = rate_model.head
            log_rates = rate_model.log_rates
        else:
            design, names, _layout = build_design(dataset, variant, shared.shape, None)
            head = family_grp.fit_head(design, dataset.target, train, shared.l2, names)

        fold_prediction = model_prediction(dataset, variant, shared, head, log_rates, test)
        prediction[test] = fold_prediction
        selection_row = {
            "dataset": dataset_id.value,
            "variant": variant.name,
            "outer_fold": fold,
            "shape": asdict(shared.shape),
            "l2": shared.l2,
            "shared_inner_rmse": shared.inner_rmse,
            "rate_shrink": rate_shrink,
            "rate_inner_rmse": rate_inner_rmse,
            "rate_min": None if log_rates is None else float(np.exp(log_rates).min()),
            "rate_median": None if log_rates is None else float(np.median(np.exp(log_rates))),
            "rate_max": None if log_rates is None else float(np.exp(log_rates).max()),
            "rate_log_sd": None if log_rates is None else float(np.std(log_rates)),
            "rate_iterations": None if rate_model is None else rate_model.iterations,
            "rate_converged": None if rate_model is None else rate_model.converged,
            "active_coefficient_count": int(np.count_nonzero(head.coefficients > 1e-10)),
        }
        save_checkpoint(
            output_dir,
            dataset_id,
            variant,
            fold,
            test,
            fold_prediction,
            selection_row,
            config,
        )
        selections.append(selection_row)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {dataset_id.value} {variant.name}")
    return prediction, [test for _train, test in outer], selections


def parameter_count(dataset: family_grp.Dataset, variant: Variant) -> int:
    heads = 1 if variant.phase is PhaseKind.ETA else 2
    linear = 1 + heads * (dataset.m + len(retained.nonsingleton_families(dataset)))
    linear += len(dataset.family_names)
    nonlinear = 4 if variant.phase is PhaseKind.ETA else 3
    if variant.response is ResponseKind.DOMAIN_WEIBULL:
        nonlinear += dataset.m + 1
    return linear + nonlinear


def paired_bootstrap(
    observed: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    draws: int = 20_000,
) -> dict[str, Any]:
    delta = (candidate - observed) ** 2 - (reference - observed) ** 2
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "mean_mse_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_better": float(np.mean(means < 0.0)),
    }


def check_rate_gradient() -> None:
    dataset = load_dataset(DatasetId.THREE_HUNDRED_M_UNCHEATABLE)
    power_variant = VARIANT_BY_NAME["power_eta"]
    power_shape = candidate_shapes(power_variant, 2)[0]
    design, _names, _layout = build_design(dataset, power_variant, power_shape, None)
    reference_design, _reference_names = retained.build_design(
        dataset,
        retained.VARIANT_BY_NAME["power_global_tau"],
        power_shape,
    )
    parity_error = float(np.max(np.abs(design - reference_design)))
    if parity_error > 1e-12:
        raise ValueError(f"Power + eta control does not reproduce Bucket-resolved GRP: {parity_error:.3e}")

    variant = VARIANT_BY_NAME["domain_weibull_separate_heads"]
    shape = candidate_shapes(variant, 2)[0]
    selection = SharedSelection(shape, 0.1, 0.0)
    indices = np.arange(200)
    rng = np.random.default_rng(7)
    log_rates = np.full(dataset.m, math.log(shape.rate)) + rng.normal(0.0, 0.15, dataset.m)
    _loss, analytic = rate_objective_and_gradient(
        log_rates,
        dataset,
        variant,
        selection,
        indices,
        1e-4,
    )
    errors = []
    for index in (0, 7, 19, dataset.m - 1):
        step = 1e-5
        plus = log_rates.copy()
        minus = log_rates.copy()
        plus[index] += step
        minus[index] -= step
        plus_loss = rate_objective_and_gradient(plus, dataset, variant, selection, indices, 1e-4)[0]
        minus_loss = rate_objective_and_gradient(minus, dataset, variant, selection, indices, 1e-4)[0]
        numeric = (plus_loss - minus_loss) / (2.0 * step)
        errors.append(abs(float(analytic[index]) - numeric))
    max_error = max(errors)
    if max_error > 1e-6:
        raise ValueError(f"Profiled domain-rate gradient check failed: max abs error {max_error:.3e}")
    print(f"Power + eta design parity max abs error: {parity_error:.3e}")
    print(f"Profiled domain-rate gradient max abs error: {max_error:.3e}")


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(dict.fromkeys(metrics["dataset"].tolist()))
    colors = dict(
        zip(
            [variant.name for variant in VARIANTS],
            sample_colorscale("RdYlGn_r", np.linspace(0.1, 0.9, len(VARIANTS))),
            strict=True,
        )
    )
    figure = make_subplots(
        rows=len(datasets),
        cols=5,
        subplot_titles=tuple(
            f"{dataset}: {title}"
            for dataset in datasets
            for title in ("RMSE", "Spearman", "Regret@1", "Tail optimism", "Low-tail RMSE")
        ),
    )
    metric_names = (
        "rmse",
        "spearman",
        "fold_mean_regret_at_1",
        "lower_tail_optimism",
        "low_tail_rmse",
    )
    for row, dataset in enumerate(datasets, start=1):
        frame = metrics.loc[metrics["dataset"].eq(dataset)]
        for column, metric in enumerate(metric_names, start=1):
            figure.add_bar(
                x=frame["variant"],
                y=frame[metric],
                marker_color=[colors[name] for name in frame["variant"]],
                text=[f"{value:.5f}" for value in frame[metric]],
                textposition="outside",
                showlegend=False,
                row=row,
                col=column,
            )
    figure.update_layout(
        title="Bucket-resolved family GRP: domain saturation x phase heads",
        template="plotly_white",
        width=2400,
        height=440 * len(datasets),
        margin={"b": 170},
    )
    figure.write_html(
        output_dir / "domain_saturation_phase_heads_metrics.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_report(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    selections: pd.DataFrame,
    output_dir: Path,
) -> None:
    indexed = metrics.set_index(["dataset", "variant"])
    interpretation: list[str] = []
    for dataset in metrics["dataset"].drop_duplicates():
        power_eta = indexed.loc[(dataset, "power_eta")]
        power_heads = indexed.loc[(dataset, "power_separate_heads")]
        weibull_eta = indexed.loc[(dataset, "domain_weibull_eta")]
        weibull_heads = indexed.loc[(dataset, "domain_weibull_separate_heads")]
        interpretation.append(
            f"- **{dataset}:** learned per-domain saturation changes the eta-model RMSE by "
            f"{weibull_eta['rmse'] / power_eta['rmse'] - 1:+.2%}. Independent heads change the power-model "
            f"RMSE by {power_heads['rmse'] / power_eta['rmse'] - 1:+.2%}, and change the learned-saturation "
            f"RMSE by {weibull_heads['rmse'] / weibull_eta['rmse'] - 1:+.2%}."
        )
    lines = [
        "# Learned domain saturation and independent phase heads in Bucket-resolved family GRP",
        "",
        "## Controlled model matrix",
        "",
        "All four variants retain bucket signals, nonlinear family coverage, and one replay-harm coefficient "
        "per semantic family. The exact current control is `power_eta`:",
        "",
        "$$z_i=r_i e_i^{(0)}+\\eta e_i^{(1)},\\qquad S_i(z_i)=z_i^a.$$",
        "",
        "The saturation intervention replaces the shared power response with",
        "",
        "$$S_i(z)=1-\\exp[-(\\rho_i z)^p],$$",
        "",
        "where each $\\log\\rho_i$ is learned from target labels and partially pooled toward the shared "
        "rate selected inside the fold. The phase intervention removes $\\eta$ and uses independent "
        "nonnegative response coefficients for $r_i e_i^{(0)}$ and $e_i^{(1)}$. The Weibull phase-head model "
        "shares each domain's saturation clock across its two heads, avoiding a doubled nonlinear parameter set.",
        "",
        "## Nested OOF results",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Paired squared-error comparisons against power + eta",
        "",
        comparisons.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "The learned domain clocks are deliberately evaluated with nested shrinkage selection. A lower training "
        "error alone is not evidence for saturation because rate and response amplitude are weakly identified in "
        "the low-exposure regime. The independent-head comparison likewise pays for its extra linear coefficients "
        "in held-out folds rather than receiving an in-sample advantage.",
        "",
        "## Fold selections",
        "",
        selections.to_markdown(index=False),
        "",
        "## Reproduce",
        "",
        "~~~bash",
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        "benchmark_grp_domain_saturation_phase_heads_20260714.py",
        "~~~",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.check_gradient:
        check_rate_gradient()
        return
    requested = tuple(DatasetId(value.strip()) for value in args.datasets.split(",") if value.strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for dataset_id in requested:
        dataset = load_dataset(dataset_id)
        dataset_predictions: dict[str, np.ndarray] = {}
        fold_indices: list[np.ndarray] | None = None
        for variant in VARIANTS:
            shapes = candidate_shapes(variant, args.num_shapes)
            prediction, current_folds, selections = nested_oof(
                dataset,
                dataset_id,
                variant,
                shapes,
                args.output_dir,
                args,
            )
            if fold_indices is None:
                fold_indices = current_folds
            elif any(not np.array_equal(left, right) for left, right in zip(fold_indices, current_folds, strict=True)):
                raise ValueError(f"Variant outer folds differ for {dataset_id.value}")
            dataset_predictions[variant.name] = prediction
            summary = family_grp.metric_summary(dataset.target, prediction, current_folds)
            metrics_rows.append(
                {
                    "dataset": dataset_id.value,
                    "variant": variant.name,
                    "parameter_count": parameter_count(dataset, variant),
                    **summary,
                }
            )
            for fold, test in enumerate(current_folds):
                fold_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.name,
                        "outer_fold": fold,
                        "rmse": float(np.sqrt(np.mean((prediction[test] - dataset.target[test]) ** 2))),
                    }
                )
            for row_index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True)):
                prediction_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.name,
                        "row_index": row_index,
                        "observed": observed,
                        "prediction": predicted,
                        "residual": predicted - observed,
                    }
                )
            selection_rows.extend(selections)

        for comparison_index, (reference_name, candidate_name, intervention) in enumerate(
            COMPARISON_PAIRS,
            start=1,
        ):
            reference = dataset_predictions[reference_name]
            candidate = dataset_predictions[candidate_name]
            fold_improvements = []
            if fold_indices is None:
                raise RuntimeError("No outer folds were generated")
            for test in fold_indices:
                reference_mse = float(np.mean((reference[test] - dataset.target[test]) ** 2))
                candidate_mse = float(np.mean((candidate[test] - dataset.target[test]) ** 2))
                fold_improvements.append(candidate_mse < reference_mse)
            comparison_rows.append(
                {
                    "dataset": dataset_id.value,
                    "intervention": intervention,
                    "reference": reference_name,
                    "candidate": candidate_name,
                    "folds_improved": int(sum(fold_improvements)),
                    "folds_total": len(fold_improvements),
                    **paired_bootstrap(
                        dataset.target,
                        reference,
                        candidate,
                        OUTER_CV_SEED + 1000 * comparison_index + list(DatasetId).index(dataset_id),
                    ),
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.json_normalize(selection_rows, sep=".")
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    selections.to_csv(args.output_dir / "selections.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    plot_metrics(metrics, args.output_dir)
    write_report(metrics, comparisons, selections, args.output_dir)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
