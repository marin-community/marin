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
"""Test sample-efficient bucket and family saturation clocks in retained-state GRP.

The current retained-Weibull GRP uses one saturation rate for every bucket and
family. This benchmark nests that model and adds at most one nonlinear degree
of freedom: the amount of dispersion retained from an unlabeled, exposure-only
saturation-timescale prior.

For bucket clocks,

    log rho_i = log rho + delta log(r_i / geometric_mean(r)),

where r_i is the inverse median retained exposure of bucket i. Family clocks
use the same construction on family-mean retained exposure. ``delta=0`` is the
shared-rate baseline. Shape, dispersion, and ridge selection occur inside each
outer fold.
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
    benchmark_production_grp_retained_hybrids_20260713 as retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    generic_family_followup as generic_family,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/grp_saturation_hierarchy_20260714"
OUTER_CV_SEED = 2714
INNER_CV_SEED = 2715
OUTER_SPLITS = 5
INNER_SPLITS = 3
L2_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0)
SPREAD_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
RATE_BOUNDS = (1e-4, 100.0)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
CHECKPOINT_SCHEMA = 1
POWER_MODEL = "power_bucket_resolved"


class DatasetId(StrEnum):
    PRODUCTION_UNCHEATABLE = "production_uncheatable"
    THREE_HUNDRED_M_UNCHEATABLE = "300m_uncheatable"
    THREE_HUNDRED_M_TABLE9 = "300m_table9"


class SaturationScope(StrEnum):
    SHARED = "shared"
    FAMILY = "family"
    BUCKET = "bucket"


@dataclass(frozen=True)
class Selection:
    shape: retained.Shape
    spread: float
    l2: float
    inner_rmse: float


@dataclass(frozen=True)
class Rates:
    bucket: np.ndarray
    family: np.ndarray
    bucket_prior: np.ndarray
    family_prior: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(dataset.value for dataset in DatasetId),
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def family_partition(raw: pooled.Dataset) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    reference = generic_family.load_generic_family_packet()
    if reference.base.domain_names != raw.domain_names:
        raise ValueError("300M family partition and fit panel use different domain orderings")
    names = tuple(generic_family.GENERIC_FAMILY_NAMES)
    members = tuple(np.asarray(reference.family_map[name], dtype=int) for name in names)
    covered = np.concatenate(members)
    if sorted(covered.tolist()) != list(range(raw.m)):
        raise ValueError("300M semantic families do not partition the 39 buckets")
    return names, members


def load_dataset(dataset_id: DatasetId) -> family_grp.Dataset:
    if dataset_id is DatasetId.PRODUCTION_UNCHEATABLE:
        return family_grp.load_dataset(family_grp.PRODUCTION_DATA, family_grp.PRODUCTION_MODEL)

    objective = "uncheatable" if dataset_id is DatasetId.THREE_HUNDRED_M_UNCHEATABLE else "table9"
    raw = pooled.load_300m_dataset(objective)
    family_names, family_members = family_partition(raw)
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


def split_indices(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    indices: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if dataset_id is DatasetId.PRODUCTION_UNCHEATABLE:
        return family_grp.kfold_indices(indices, n_splits, seed)
    local_panel = dataset.frame.iloc[indices].reset_index(drop=True)
    local_splits = component_dsp.panel_stratified_folds(local_panel, n_splits=n_splits, seed=seed)
    return [(indices[train], indices[test]) for train, test in local_splits]


def retained_exposure(dataset: family_grp.Dataset, shape: retained.Shape) -> np.ndarray:
    return retained.retained_exposure(dataset, shape)[0]


def inverse_median_prior(values: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    train = values[train_indices]
    positive = np.where(train > 1e-10, train, np.nan)
    medians = np.nanmedian(positive, axis=0)
    finite = np.isfinite(medians) & (medians > 0.0)
    if not np.any(finite):
        raise ValueError("No positive retained exposure available for saturation prior")
    fallback = float(np.median(medians[finite]))
    medians = np.where(finite, medians, fallback)
    inverse = 1.0 / np.maximum(medians, 1e-8)
    log_inverse = np.log(inverse)
    return log_inverse - float(np.mean(log_inverse))


def rates(
    dataset: family_grp.Dataset,
    shape: retained.Shape,
    scope: SaturationScope,
    spread: float,
    train_indices: np.ndarray,
) -> Rates:
    exposure = retained_exposure(dataset, shape)
    family_mean = np.column_stack([exposure[:, members].mean(axis=1) for members in dataset.family_members])
    bucket_prior = inverse_median_prior(exposure, train_indices)
    family_prior = inverse_median_prior(family_mean, train_indices)
    if scope is SaturationScope.SHARED:
        bucket_log_offset = np.zeros(dataset.m, dtype=float)
        family_log_offset = np.zeros(len(dataset.family_names), dtype=float)
    elif scope is SaturationScope.FAMILY:
        family_log_offset = spread * family_prior
        bucket_log_offset = np.zeros(dataset.m, dtype=float)
        for family_index, members in enumerate(dataset.family_members):
            bucket_log_offset[members] = family_log_offset[family_index]
    elif scope is SaturationScope.BUCKET:
        bucket_log_offset = spread * bucket_prior
        family_log_offset = np.asarray(
            [float(np.mean(bucket_log_offset[members])) for members in dataset.family_members],
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported saturation scope {scope}")
    base_log_rate = math.log(shape.rate)
    bucket_rate = np.clip(np.exp(base_log_rate + bucket_log_offset), *RATE_BOUNDS)
    family_rate = np.clip(np.exp(base_log_rate + family_log_offset), *RATE_BOUNDS)
    return Rates(bucket_rate, family_rate, bucket_prior, family_prior)


def weibull_response(exposure: np.ndarray, rate: np.ndarray, exponent: float) -> np.ndarray:
    return -np.expm1(-((np.maximum(exposure, 0.0) * rate[None, :]) ** exponent))


def build_design(
    dataset: family_grp.Dataset,
    shape: retained.Shape,
    scope: SaturationScope,
    spread: float,
    prior_indices: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], Rates]:
    exposure = retained_exposure(dataset, shape)
    learned_rates = rates(dataset, shape, scope, spread, prior_indices)
    bucket_signal = weibull_response(exposure, learned_rates.bucket, shape.exponent)
    family_total = np.column_stack([exposure[:, members].sum(axis=1) for members in dataset.family_members])
    nonsingleton = retained.nonsingleton_families(dataset)
    pieces = [-bucket_signal]
    names: list[str] = [f"bucket_signal:{domain}" for domain in dataset.domains]
    if nonsingleton:
        family_mean = np.column_stack(
            [exposure[:, dataset.family_members[index]].mean(axis=1) for index in nonsingleton]
        )
        family_signal = weibull_response(
            family_mean,
            learned_rates.family[np.asarray(nonsingleton, dtype=int)],
            shape.exponent,
        )
        pieces.append(-family_signal)
        names.extend(f"family_signal:{dataset.family_names[index]}" for index in nonsingleton)
    pieces.append(retained.softplus_penalty(family_total, shape.penalty_threshold))
    names.extend(f"family_penalty:{name}" for name in dataset.family_names)
    return np.hstack(pieces), tuple(names), learned_rates


def candidate_spreads(scope: SaturationScope) -> tuple[float, ...]:
    return (0.0,) if scope is SaturationScope.SHARED else SPREAD_GRID


def select_dispersion(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    shape: retained.Shape,
    scope: SaturationScope,
    indices: np.ndarray,
    seed: int,
) -> Selection:
    splits = split_indices(dataset, dataset_id, indices, INNER_SPLITS, seed)
    best: tuple[float, float, float] | None = None
    for spread in candidate_spreads(scope):
        for l2 in L2_GRID:
            errors = []
            for train, test in splits:
                design, names, _rates = build_design(dataset, shape, scope, spread, train)
                head = family_grp.fit_head(design, dataset.target, train, l2, names)
                errors.append(head.predict_design(design[test]) - dataset.target[test])
            score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
            candidate = (score, spread, l2)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError(f"No dispersion candidate for {scope}")
    return Selection(shape=shape, spread=best[1], l2=best[2], inner_rmse=best[0])


def baseline_dataset_id(dataset_id: DatasetId) -> retained.DatasetId:
    if dataset_id is DatasetId.PRODUCTION_UNCHEATABLE:
        return retained.DatasetId.PRODUCTION
    return retained.DatasetId.THREE_HUNDRED_M_UNCHEATABLE


def selected_shared_shape(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    indices: np.ndarray,
    shapes: tuple[retained.Shape, ...],
    seed: int,
) -> retained.SelectedHyperparameters:
    variant = retained.VARIANT_BY_NAME["weibull_global_tau"]
    return retained.select_shared_hyperparameters(
        dataset,
        baseline_dataset_id(dataset_id),
        variant,
        shapes,
        indices,
        seed,
    )


def fit_and_predict(
    dataset: family_grp.Dataset,
    selection: Selection,
    scope: SaturationScope,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, Rates, family_grp.FittedHead]:
    design, names, learned_rates = build_design(
        dataset,
        selection.shape,
        scope,
        selection.spread,
        train,
    )
    head = family_grp.fit_head(design, dataset.target, train, selection.l2, names)
    return head.predict_design(design[test]), learned_rates, head


def checkpoint_paths(output_dir: Path, dataset_id: DatasetId, fold: int) -> tuple[Path, Path]:
    stem = output_dir / "checkpoints" / f"{dataset_id.value}__outer_{fold}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def load_checkpoint(
    output_dir: Path,
    dataset_id: DatasetId,
    fold: int,
    test: np.ndarray,
    num_shapes: int,
) -> tuple[dict[SaturationScope, np.ndarray], list[dict[str, Any]]] | None:
    metadata_path, prediction_path = checkpoint_paths(output_dir, dataset_id, fold)
    if not metadata_path.exists() or not prediction_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "test_indices": test.tolist(),
        "num_shapes": num_shapes,
        "spread_grid": list(SPREAD_GRID),
        "l2_grid": list(L2_GRID),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    saved = np.load(prediction_path)
    predictions = {scope: np.asarray(saved[scope.value], dtype=float) for scope in SaturationScope}
    if any(value.shape != test.shape for value in predictions.values()):
        return None
    return predictions, list(metadata["selections"])


def save_checkpoint(
    output_dir: Path,
    dataset_id: DatasetId,
    fold: int,
    test: np.ndarray,
    num_shapes: int,
    predictions: dict[SaturationScope, np.ndarray],
    selections: list[dict[str, Any]],
) -> None:
    metadata_path, prediction_path = checkpoint_paths(output_dir, dataset_id, fold)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(prediction_path, **{scope.value: value for scope, value in predictions.items()})
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "test_indices": test.tolist(),
        "num_shapes": num_shapes,
        "spread_grid": list(SPREAD_GRID),
        "l2_grid": list(L2_GRID),
        "selections": selections,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def nested_oof(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    shapes: tuple[retained.Shape, ...],
    output_dir: Path,
    num_shapes: int,
    force: bool,
) -> tuple[dict[SaturationScope, np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    outer = split_indices(dataset, dataset_id, np.arange(dataset.n), OUTER_SPLITS, OUTER_CV_SEED)
    predictions = {scope: np.full(dataset.n, np.nan, dtype=float) for scope in SaturationScope}
    selection_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(outer):
        cached = None if force else load_checkpoint(output_dir, dataset_id, fold, test, num_shapes)
        if cached is not None:
            cached_predictions, cached_selections = cached
            for scope in SaturationScope:
                predictions[scope][test] = cached_predictions[scope]
            selection_rows.extend(cached_selections)
            continue

        print(f"{dataset_id.value}: outer fold {fold + 1}/{len(outer)}", flush=True)
        shared = selected_shared_shape(dataset, dataset_id, train, shapes, INNER_CV_SEED + fold)
        fold_predictions: dict[SaturationScope, np.ndarray] = {}
        fold_selections: list[dict[str, Any]] = []
        for scope in SaturationScope:
            selection = select_dispersion(
                dataset,
                dataset_id,
                shared.shape,
                scope,
                train,
                INNER_CV_SEED + 100 + fold,
            )
            prediction, learned_rates, head = fit_and_predict(dataset, selection, scope, train, test)
            predictions[scope][test] = prediction
            fold_predictions[scope] = prediction
            fold_selections.append(
                {
                    "dataset": dataset_id.value,
                    "outer_fold": fold,
                    "scope": scope.value,
                    "shape": asdict(selection.shape),
                    "spread": selection.spread,
                    "l2": selection.l2,
                    "inner_rmse": selection.inner_rmse,
                    "bucket_rate_min": float(learned_rates.bucket.min()),
                    "bucket_rate_max": float(learned_rates.bucket.max()),
                    "family_rate_min": float(learned_rates.family.min()),
                    "family_rate_max": float(learned_rates.family.max()),
                    "active_coefficient_count": int(np.count_nonzero(head.coefficients > 1e-10)),
                }
            )
        save_checkpoint(
            output_dir,
            dataset_id,
            fold,
            test,
            num_shapes,
            fold_predictions,
            fold_selections,
        )
        selection_rows.extend(fold_selections)
    if any(not np.isfinite(prediction).all() for prediction in predictions.values()):
        raise RuntimeError(f"Incomplete OOF predictions for {dataset_id}")
    return predictions, [test for _train, test in outer], selection_rows


def power_checkpoint_paths(output_dir: Path, dataset_id: DatasetId, fold: int) -> tuple[Path, Path]:
    stem = output_dir / "checkpoints" / f"{dataset_id.value}__{POWER_MODEL}__outer_{fold}"
    return stem.with_suffix(".json"), stem.with_suffix(".npy")


def nested_power_oof(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    shapes: tuple[retained.Shape, ...],
    output_dir: Path,
    num_shapes: int,
    force: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    outer = split_indices(dataset, dataset_id, np.arange(dataset.n), OUTER_SPLITS, OUTER_CV_SEED)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selection_rows: list[dict[str, Any]] = []
    variant = retained.VARIANT_BY_NAME["power_global_tau"]
    for fold, (train, test) in enumerate(outer):
        metadata_path, prediction_path = power_checkpoint_paths(output_dir, dataset_id, fold)
        if not force and metadata_path.exists() and prediction_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata.get("schema") == CHECKPOINT_SCHEMA
                and metadata.get("test_indices") == test.tolist()
                and metadata.get("num_shapes") == num_shapes
            ):
                fold_prediction = np.load(prediction_path)
                if fold_prediction.shape == test.shape:
                    prediction[test] = fold_prediction
                    selection_rows.append(dict(metadata["selection"]))
                    continue

        selected = retained.select_shared_hyperparameters(
            dataset,
            baseline_dataset_id(dataset_id),
            variant,
            shapes,
            train,
            INNER_CV_SEED + 300 + fold,
        )
        model = retained.fit_shared_model(dataset, variant, selected, train)
        fold_prediction = retained.model_prediction(dataset, model, test)
        prediction[test] = fold_prediction
        selection = {
            "dataset": dataset_id.value,
            "outer_fold": fold,
            "scope": POWER_MODEL,
            "shape": asdict(selected.shape),
            "spread": None,
            "l2": selected.l2,
            "inner_rmse": selected.cv_rmse,
            "bucket_rate_min": None,
            "bucket_rate_max": None,
            "family_rate_min": None,
            "family_rate_max": None,
            "active_coefficient_count": int(np.count_nonzero(model.head.coefficients > 1e-10)),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prediction_path, fold_prediction)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "test_indices": test.tolist(),
                    "num_shapes": num_shapes,
                    "selection": selection,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        selection_rows.append(selection)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF predictions for {dataset_id} {POWER_MODEL}")
    return prediction, [test for _train, test in outer], selection_rows


def fit_full_models(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    shapes: tuple[retained.Shape, ...],
) -> dict[str, dict[str, Any]]:
    indices = np.arange(dataset.n)
    shared = selected_shared_shape(dataset, dataset_id, indices, shapes, INNER_CV_SEED + 999)
    models: dict[str, dict[str, Any]] = {}
    for scope in SaturationScope:
        selection = select_dispersion(
            dataset,
            dataset_id,
            shared.shape,
            scope,
            indices,
            INNER_CV_SEED + 1999,
        )
        _prediction, learned_rates, head = fit_and_predict(dataset, selection, scope, indices, indices)
        models[scope.value] = {
            "shape": asdict(selection.shape),
            "spread": selection.spread,
            "l2": selection.l2,
            "inner_rmse": selection.inner_rmse,
            "bucket_rates": dict(zip(dataset.domains, learned_rates.bucket.tolist(), strict=True)),
            "family_rates": dict(zip(dataset.family_names, learned_rates.family.tolist(), strict=True)),
            "active_coefficient_count": int(np.count_nonzero(head.coefficients > 1e-10)),
            "parameter_count": parameter_count(dataset, scope),
        }
    return models


def fit_full_power_model(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    shapes: tuple[retained.Shape, ...],
) -> dict[str, Any]:
    indices = np.arange(dataset.n)
    variant = retained.VARIANT_BY_NAME["power_global_tau"]
    selected = retained.select_shared_hyperparameters(
        dataset,
        baseline_dataset_id(dataset_id),
        variant,
        shapes,
        indices,
        INNER_CV_SEED + 2999,
    )
    model = retained.fit_shared_model(dataset, variant, selected, indices)
    return {
        "shape": asdict(selected.shape),
        "l2": selected.l2,
        "inner_rmse": selected.cv_rmse,
        "active_coefficient_count": int(np.count_nonzero(model.head.coefficients > 1e-10)),
        "parameter_count": retained.parameter_count(dataset, variant),
    }


def parameter_count(dataset: family_grp.Dataset, scope: SaturationScope) -> int:
    linear = 1 + dataset.m + len(retained.nonsingleton_families(dataset)) + len(dataset.family_names)
    nonlinear = 5 + int(scope is not SaturationScope.SHARED)
    return linear + nonlinear


def paired_bootstrap(
    observed: np.ndarray,
    shared: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    draws: int = 20_000,
) -> dict[str, Any]:
    delta = (candidate - observed) ** 2 - (shared - observed) ** 2
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "mean_mse_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_better": float(np.mean(means < 0.0)),
    }


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(dict.fromkeys(metrics["dataset"].tolist()))
    model_names = [POWER_MODEL, *[scope.value for scope in SaturationScope]]
    colors = dict(
        zip(
            model_names,
            sample_colorscale("RdYlGn_r", np.linspace(0.1, 0.9, len(model_names))),
            strict=True,
        )
    )
    figure = make_subplots(
        rows=len(datasets),
        cols=3,
        subplot_titles=tuple(
            title
            for dataset in datasets
            for title in (f"{dataset}: RMSE", f"{dataset}: Spearman", f"{dataset}: Regret@1")
        ),
    )
    for row, dataset in enumerate(datasets, start=1):
        frame = metrics.loc[metrics["dataset"].eq(dataset)]
        for col, metric in enumerate(("rmse", "spearman", "fold_mean_regret_at_1"), start=1):
            figure.add_bar(
                x=frame["scope"],
                y=frame[metric],
                marker_color=[colors[scope] for scope in frame["scope"]],
                text=[f"{value:.6f}" for value in frame[metric]],
                textposition="outside",
                showlegend=False,
                row=row,
                col=col,
            )
    figure.update_layout(
        title="Retained-Weibull GRP: shared vs hierarchical saturation clocks",
        template="plotly_white",
        width=1800,
        height=430 * len(datasets),
        margin={"b": 120},
    )
    figure.write_html(output_dir / "saturation_hierarchy_metrics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    full_models: dict[str, Any],
    output_dir: Path,
) -> None:
    indexed = metrics.set_index(["dataset", "scope"])
    interpretation: list[str] = []
    for dataset in metrics["dataset"].drop_duplicates():
        power = indexed.loc[(dataset, POWER_MODEL)]
        shared = indexed.loc[(dataset, SaturationScope.SHARED.value)]
        family = indexed.loc[(dataset, SaturationScope.FAMILY.value)]
        bucket = indexed.loc[(dataset, SaturationScope.BUCKET.value)]
        interpretation.append(
            f"- **{dataset}:** family clocks change RMSE by {family['rmse'] / shared['rmse'] - 1:+.2%} "
            f"versus shared saturation. Bucket clocks change RMSE by "
            f"{bucket['rmse'] / shared['rmse'] - 1:+.2%}, Spearman from {shared['spearman']:.3f} to "
            f"{bucket['spearman']:.3f}, and low-tail RMSE by "
            f"{bucket['low_tail_rmse'] / shared['low_tail_rmse'] - 1:+.2%}. Against the power-law "
            f"Bucket-resolved control, bucket-clock RMSE changes by {bucket['rmse'] / power['rmse'] - 1:+.2%}."
        )
    lines = [
        "# Hierarchical saturation clocks in retained-state GRP",
        "",
        "## Model",
        "",
        "The shared baseline uses $S(z)=1-\\exp[-(\\rho z)^p]$ for every bucket and family. "
        "The family-clock extension uses one exposure-derived clock per family; the bucket-clock extension uses "
        "one per bucket and induces each family clock as the geometric mean of its members. In either case,",
        "",
        "$$\\log \\rho_j=\\log \\rho+\\delta(\\log r_j-\\overline{\\log r}),$$",
        "",
        "where $r_j$ is inverse median retained exposure computed without target labels. The learned "
        "$\\delta\\in\\{0,0.25,0.5,1,1.5,2\\}$ controls only the amount of timescale dispersion; "
        "$\\delta=0$ exactly recovers shared saturation. The Weibull exponent $p$, retained-state parameters, "
        "replay onset, dispersion, and ridge penalty are selected inside each outer fold.",
        "The `power_bucket_resolved` control is the current Bucket-resolved family GRP: it replaces finite "
        "Weibull saturation by a shared diminishing-return power $z^a$ while keeping the same bucket, family, "
        "retained-state, and replay channels.",
        "",
        "## Nested OOF results",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Paired comparisons against shared saturation",
        "",
        comparisons.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "Bucket-specific clocks are a useful partial-pooling diagnostic on the 300M panels, but they do not "
        "produce a universal winner: the production fit prefers the original power response, while family "
        "clocks are neutral or harmful. The Table-9 full-data bucket spread reaches the top of the tested grid; "
        "outer folds select 1.0--1.5, so the OOF gain is not solely a boundary artifact, but the resulting rate "
        "range is broad and should not yet be treated as a deployment-ready saturation law.",
        "",
        "## Full-data selections",
        "",
        "~~~json",
        json.dumps(full_models, indent=2, allow_nan=False),
        "~~~",
        "",
        "## Reproduce",
        "",
        "~~~bash",
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        "benchmark_grp_saturation_hierarchy_20260714.py",
        "~~~",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    requested = tuple(DatasetId(value.strip()) for value in args.datasets.split(",") if value.strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    full_models: dict[str, Any] = {}
    baseline_variant = retained.VARIANT_BY_NAME["weibull_global_tau"]
    shapes = retained.shared_shape_candidates(baseline_variant, args.num_shapes)
    power_variant = retained.VARIANT_BY_NAME["power_global_tau"]
    power_shapes = retained.shared_shape_candidates(power_variant, args.num_shapes)

    for dataset_id in requested:
        dataset = load_dataset(dataset_id)
        power_prediction, power_fold_indices, power_selections = nested_power_oof(
            dataset,
            dataset_id,
            power_shapes,
            args.output_dir,
            args.num_shapes,
            args.force,
        )
        predictions, fold_indices, selections = nested_oof(
            dataset,
            dataset_id,
            shapes,
            args.output_dir,
            args.num_shapes,
            args.force,
        )
        if any(not np.array_equal(left, right) for left, right in zip(power_fold_indices, fold_indices, strict=True)):
            raise ValueError(f"Power and saturation controls use different outer folds for {dataset_id}")
        power_summary = family_grp.metric_summary(dataset.target, power_prediction, power_fold_indices)
        metrics_rows.append(
            {
                "dataset": dataset_id.value,
                "scope": POWER_MODEL,
                "parameter_count": retained.parameter_count(dataset, power_variant),
                **power_summary,
            }
        )
        for fold, test in enumerate(power_fold_indices):
            residual = power_prediction[test] - dataset.target[test]
            fold_rows.append(
                {
                    "dataset": dataset_id.value,
                    "scope": POWER_MODEL,
                    "outer_fold": fold,
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                }
            )
        for row_index, (observed, predicted) in enumerate(zip(dataset.target, power_prediction, strict=True)):
            prediction_rows.append(
                {
                    "dataset": dataset_id.value,
                    "scope": POWER_MODEL,
                    "row_index": row_index,
                    "observed": observed,
                    "prediction": predicted,
                    "residual": predicted - observed,
                }
            )
        selection_rows.extend(power_selections)
        selection_rows.extend(selections)
        for scope, prediction in predictions.items():
            metrics_rows.append(
                {
                    "dataset": dataset_id.value,
                    "scope": scope.value,
                    "parameter_count": parameter_count(dataset, scope),
                    **family_grp.metric_summary(dataset.target, prediction, fold_indices),
                }
            )
            for fold, test in enumerate(fold_indices):
                residual = prediction[test] - dataset.target[test]
                fold_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "scope": scope.value,
                        "outer_fold": fold,
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                    }
                )
            for row_index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True)):
                prediction_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "scope": scope.value,
                        "row_index": row_index,
                        "observed": observed,
                        "prediction": predicted,
                        "residual": predicted - observed,
                    }
                )
            if scope is not SaturationScope.SHARED:
                comparison_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "scope": scope.value,
                        **paired_bootstrap(
                            dataset.target,
                            predictions[SaturationScope.SHARED],
                            prediction,
                            seed=3100 + len(comparison_rows),
                        ),
                    }
                )
        full_models[dataset_id.value] = {
            POWER_MODEL: fit_full_power_model(dataset, dataset_id, power_shapes),
            **fit_full_models(dataset, dataset_id, shapes),
        }

    metrics = pd.DataFrame(metrics_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    selections.to_csv(args.output_dir / "selections.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    (args.output_dir / "full_models.json").write_text(json.dumps(full_models, indent=2, allow_nan=False) + "\n")
    plot_metrics(metrics, args.output_dir)
    write_report(metrics, comparisons, full_models, args.output_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
