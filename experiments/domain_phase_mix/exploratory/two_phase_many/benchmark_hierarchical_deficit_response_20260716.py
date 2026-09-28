# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""Test scaling-law deficit responses for optimistic poor-mixture predictions.

The promoted GRP model represents learning as a local benefit ``-a e**alpha``.
That approximation remains finite when exposure to an essential bucket goes to
zero, so it can be systematically optimistic on concentrated policies outside
the fit panel. This benchmark replaces only that response with a normalized
scaling-law deficit,

``a [(e / e_prop + delta)**(-alpha) - (1 + delta)**(-alpha)]``,

or its logarithmic limit. Both are zero at the proportional reference, rise
under missing exposure, and asymptote under additional exposure. Hierarchical
family pooling, retained exposure, replay harm, and the phase-shift term remain
unchanged. Nonlinear settings are selected on fit-panel CV; the historical
3e18 validation archive is scored only after selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_calibration_forms_20260715 as calibration,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_deficit_response_20260716"
DEFICIT_FLOOR_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
SURPLUS_CREDIT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FOUNDATION_FAMILY = "broad_text"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Variant(StrEnum):
    CURRENT = "current_power_benefit"
    LOG_DEFICIT = "log_scaling_deficit"
    POWER_DEFICIT = "inverse_power_scaling_deficit"
    LOG_DEFICIT_LITERAL_REPLAY = "log_scaling_deficit_literal_replay"
    POWER_DEFICIT_LITERAL_REPLAY = "inverse_power_scaling_deficit_literal_replay"
    LOG_DEFICIT_HYBRID_REPLAY = "log_scaling_deficit_hybrid_replay"
    POWER_DEFICIT_HYBRID_REPLAY = "inverse_power_scaling_deficit_hybrid_replay"
    POWER_DEFICIT_HYBRID_SPLIT_RESPONSE = "inverse_power_deficit_split_shortage_surplus"
    POWER_DEFICIT_HYBRID_ASYMMETRIC = "inverse_power_deficit_asymmetric_surplus"
    POWER_DEFICIT_HYBRID_BOTTLENECK = "inverse_power_deficit_family_bottleneck"
    POWER_DEFICIT_HYBRID_CONCENTRATION = "inverse_power_deficit_aggregate_concentration"
    POWER_DEFICIT_HYBRID_FAMILY_IMBALANCE = "inverse_power_deficit_family_imbalance"
    POWER_DEFICIT_HYBRID_JOINT = "inverse_power_deficit_hybrid_joint_family"
    POWER_DEFICIT_HYBRID_CONDITIONED_REPLAY = "inverse_power_deficit_conditioned_replay"
    POWER_DEFICIT_HYBRID_EARLY_FAMILY = "inverse_power_deficit_hybrid_early_family"
    POWER_DEFICIT_HYBRID_EARLY_FAMILY_SPLIT_RESPONSE = "inverse_power_deficit_early_family_split_shortage_surplus"
    POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC = "inverse_power_deficit_early_family_asymmetric_surplus"
    POWER_DEFICIT_HYBRID_EARLY_FOUNDATION_ASYMMETRIC = "inverse_power_deficit_early_foundation_asymmetric_surplus"
    POWER_DEFICIT_HYBRID_EARLY_FAMILY_JOINT = "inverse_power_deficit_hybrid_early_family_joint"
    POWER_DEFICIT_HYBRID_BOTH_FAMILIES = "inverse_power_deficit_hybrid_both_phase_families"
    POWER_DEFICIT_HYBRID_BOTH_FAMILIES_ASYMMETRIC = "inverse_power_deficit_both_phase_families_asymmetric_surplus"

    @property
    def log_deficit(self) -> bool:
        return self in {
            Variant.LOG_DEFICIT,
            Variant.LOG_DEFICIT_LITERAL_REPLAY,
            Variant.LOG_DEFICIT_HYBRID_REPLAY,
        }

    @property
    def literal_replay(self) -> bool:
        return self in {Variant.LOG_DEFICIT_LITERAL_REPLAY, Variant.POWER_DEFICIT_LITERAL_REPLAY}

    @property
    def add_literal_replay(self) -> bool:
        return self in {
            Variant.LOG_DEFICIT_LITERAL_REPLAY,
            Variant.POWER_DEFICIT_LITERAL_REPLAY,
            Variant.LOG_DEFICIT_HYBRID_REPLAY,
            Variant.POWER_DEFICIT_HYBRID_REPLAY,
            Variant.POWER_DEFICIT_HYBRID_SPLIT_RESPONSE,
            Variant.POWER_DEFICIT_HYBRID_ASYMMETRIC,
            Variant.POWER_DEFICIT_HYBRID_BOTTLENECK,
            Variant.POWER_DEFICIT_HYBRID_CONCENTRATION,
            Variant.POWER_DEFICIT_HYBRID_FAMILY_IMBALANCE,
            Variant.POWER_DEFICIT_HYBRID_JOINT,
            Variant.POWER_DEFICIT_HYBRID_CONDITIONED_REPLAY,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_SPLIT_RESPONSE,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FOUNDATION_ASYMMETRIC,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_JOINT,
            Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES,
            Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES_ASYMMETRIC,
        }

    @property
    def phase_family_deficit(self) -> bool:
        return self in {
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_SPLIT_RESPONSE,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FOUNDATION_ASYMMETRIC,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_JOINT,
            Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES,
            Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES_ASYMMETRIC,
        }

    @property
    def both_phase_family_deficit(self) -> bool:
        return self in {
            Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES,
            Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES_ASYMMETRIC,
        }

    @property
    def joint_family_deficit(self) -> bool:
        return self in {
            Variant.POWER_DEFICIT_HYBRID_JOINT,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_JOINT,
        }

    @property
    def conditioned_replay(self) -> bool:
        return self is Variant.POWER_DEFICIT_HYBRID_CONDITIONED_REPLAY

    @property
    def asymmetric_surplus(self) -> bool:
        return self in {
            Variant.POWER_DEFICIT_HYBRID_ASYMMETRIC,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FOUNDATION_ASYMMETRIC,
            Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES_ASYMMETRIC,
        }

    @property
    def split_response(self) -> bool:
        return self in {
            Variant.POWER_DEFICIT_HYBRID_SPLIT_RESPONSE,
            Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_SPLIT_RESPONSE,
        }

    @property
    def family_bottleneck(self) -> bool:
        return self is Variant.POWER_DEFICIT_HYBRID_BOTTLENECK

    @property
    def aggregate_concentration(self) -> bool:
        return self is Variant.POWER_DEFICIT_HYBRID_CONCENTRATION

    @property
    def family_imbalance(self) -> bool:
        return self is Variant.POWER_DEFICIT_HYBRID_FAMILY_IMBALANCE

    @property
    def foundation_family_only(self) -> bool:
        return self is Variant.POWER_DEFICIT_HYBRID_EARLY_FOUNDATION_ASYMMETRIC


@dataclass(frozen=True)
class Config:
    variant: Variant
    base: base.Config
    deficit_floor: float
    surplus_credit: float = 1.0


@dataclass(frozen=True)
class Design:
    values: np.ndarray
    names: tuple[str, ...]
    ridge_multipliers: np.ndarray


@dataclass(frozen=True)
class Model:
    dataset: family_grp.Dataset
    config: Config
    intercept: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design = build_design(candidate, self.config)
        return np.asarray(self.intercept + design.values @ self.coefficients, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="delphi_3e18_uncheatable,delphi_3e18_table9",
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument("--num-shapes", type=int, default=12)
    parser.add_argument("--top-shape-floor-pairs", type=int, default=3)
    parser.add_argument(
        "--variants",
        default=",".join(variant.value for variant in Variant),
        help="Comma-separated structural variants to benchmark.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalized_deficit(ratio: np.ndarray, config: Config) -> np.ndarray:
    floor = config.deficit_floor
    safe = np.maximum(ratio, 0.0) + floor
    reference = 1.0 + floor
    if config.variant.log_deficit:
        return np.log(reference) - np.log(safe)
    if config.variant in {
        Variant.POWER_DEFICIT,
        Variant.POWER_DEFICIT_LITERAL_REPLAY,
        Variant.POWER_DEFICIT_HYBRID_REPLAY,
        Variant.POWER_DEFICIT_HYBRID_SPLIT_RESPONSE,
        Variant.POWER_DEFICIT_HYBRID_ASYMMETRIC,
        Variant.POWER_DEFICIT_HYBRID_BOTTLENECK,
        Variant.POWER_DEFICIT_HYBRID_CONCENTRATION,
        Variant.POWER_DEFICIT_HYBRID_FAMILY_IMBALANCE,
        Variant.POWER_DEFICIT_HYBRID_JOINT,
        Variant.POWER_DEFICIT_HYBRID_CONDITIONED_REPLAY,
        Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY,
        Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_SPLIT_RESPONSE,
        Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC,
        Variant.POWER_DEFICIT_HYBRID_EARLY_FOUNDATION_ASYMMETRIC,
        Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_JOINT,
        Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES,
        Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES_ASYMMETRIC,
    }:
        exponent = config.base.shape.exponent
        response = safe ** (-exponent) - reference ** (-exponent)
        if config.variant.asymmetric_surplus:
            return np.maximum(response, 0.0) - config.surplus_credit * np.maximum(-response, 0.0)
        return response
    raise ValueError(f"Unsupported deficit response: {config.variant}")


def build_deficit_design(dataset: family_grp.Dataset, config: Config) -> Design:
    exposure = base.retained_exposure(dataset, config.base.shape)
    reference = base.proportional_bucket_exposure(dataset, config.base.shape)
    bucket_ratio = exposure / np.maximum(reference[None, :], 1e-12)
    bucket_deficit = normalized_deficit(bucket_ratio, config)
    family_total = np.column_stack([exposure[:, members].sum(axis=1) for members in dataset.family_members])
    family_reference = np.asarray([reference[members].sum() for members in dataset.family_members], dtype=float)
    family_ratio = family_total / np.maximum(family_reference[None, :], 1e-12)
    family_deficit = normalized_deficit(family_ratio, config)

    pieces: list[np.ndarray] = []
    names: list[str] = []
    ridge: list[float] = []
    nonsingleton = [
        (name, members)
        for name, members in zip(dataset.family_names, dataset.family_members, strict=True)
        if len(members) > 1
    ]
    response_channels = (("net", bucket_deficit, family_deficit),)
    if config.variant.split_response:
        response_channels = (
            ("shortage", np.maximum(bucket_deficit, 0.0), np.maximum(family_deficit, 0.0)),
            ("surplus", -np.maximum(-bucket_deficit, 0.0), -np.maximum(-family_deficit, 0.0)),
        )
    for channel_name, channel_bucket, channel_family in response_channels:
        singleton = [members[0] for members in dataset.family_members if len(members) == 1]
        if singleton:
            pieces.append(channel_bucket[:, singleton])
            names.extend(f"{channel_name}_singleton:{dataset.domains[index]}" for index in singleton)
            ridge.extend([1.0] * len(singleton))

        for family_name, members in nonsingleton:
            pieces.append(channel_bucket[:, members].sum(axis=1, keepdims=True))
            names.append(f"{channel_name}_pooled_family:{family_name}")
            ridge.append(1.0)
        if nonsingleton:
            residual_members = np.concatenate([members for _name, members in nonsingleton])
            pieces.append(channel_bucket[:, residual_members])
            names.extend(f"{channel_name}_bucket_excess:{dataset.domains[index]}" for index in residual_members)
            ridge.extend([config.base.residual_shrink] * len(residual_members))
            family_indices = [index for index, members in enumerate(dataset.family_members) if len(members) > 1]
            pieces.append(channel_family[:, family_indices])
            names.extend(f"{channel_name}_family_coverage:{dataset.family_names[index]}" for index in family_indices)
            ridge.extend([1.0] * len(family_indices))

    family_member_replay = np.column_stack(
        [
            base.overexposure_harm(exposure, config.base.shape.penalty_threshold)[:, members].mean(axis=1)
            for members in dataset.family_members
        ]
    )
    if not config.variant.literal_replay:
        pieces.append(base.overexposure_harm(family_total, config.base.shape.penalty_threshold))
        names.extend(f"family_total_replay:{name}" for name in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))

        pieces.append(family_member_replay)
        names.extend(f"family_member_replay:{name}" for name in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))

    if config.variant.add_literal_replay:
        actual_epochs = dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :]
        literal_replay = np.maximum(actual_epochs - 1.0, 0.0) ** 2
        pieces.append(literal_replay.sum(axis=1, keepdims=True))
        names.append("shared_literal_replay")
        ridge.append(1.0)

    if config.variant.family_bottleneck:
        pieces.append(np.max(np.maximum(family_deficit, 0.0), axis=1, keepdims=True))
        names.append("weakest_family_deficit")
        ridge.append(1.0)

    if config.variant.aggregate_concentration:
        phase_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
        aggregate = phase_fraction * dataset.weights[:, 0, :] + (1.0 - phase_fraction) * dataset.weights[:, 1, :]
        proportional = base.proportional_weights(dataset)
        reference_hhi = float(np.sum(proportional**2))
        excess_hhi = np.maximum(np.sum(aggregate**2, axis=1) / reference_hhi - 1.0, 0.0)
        pieces.append(excess_hhi[:, None])
        names.append("aggregate_excess_concentration")
        ridge.append(1.0)

    if config.variant.family_imbalance:
        centered_log_ratio = np.log(family_ratio + config.deficit_floor) - np.log(1.0 + config.deficit_floor)
        centered_log_ratio -= centered_log_ratio.mean(axis=1, keepdims=True)
        pieces.append(np.mean(centered_log_ratio**2, axis=1, keepdims=True))
        names.append("semantic_family_log_exposure_variance")
        ridge.append(1.0)

    if config.variant.joint_family_deficit:
        positive_deficit = np.maximum(family_deficit, 0.0)
        pair_count = max(positive_deficit.shape[1] * (positive_deficit.shape[1] - 1) / 2.0, 1.0)
        pairwise = (positive_deficit.sum(axis=1) ** 2 - np.sum(positive_deficit**2, axis=1)) / (2.0 * pair_count)
        pieces.append(pairwise[:, None])
        names.append("joint_family_deficit")
        ridge.append(1.0)

    if config.variant.conditioned_replay:
        weakest_family_shortage = np.max(np.maximum(family_deficit, 0.0), axis=1)
        mean_member_replay = family_member_replay.mean(axis=1)
        pieces.append((weakest_family_shortage * mean_member_replay)[:, None])
        names.append("weakest_family_shortage_x_member_replay")
        ridge.append(1.0)

    if config.variant.phase_family_deficit:
        proportional = base.proportional_weights(dataset)
        selected_family_indices = list(range(len(dataset.family_names)))
        if config.variant.foundation_family_only:
            if FOUNDATION_FAMILY not in dataset.family_names:
                raise ValueError(f"Dataset has no {FOUNDATION_FAMILY!r} semantic family")
            selected_family_indices = [dataset.family_names.index(FOUNDATION_FAMILY)]
        phases = (0, 1) if config.variant.both_phase_family_deficit else (0,)
        for phase in phases:
            phase_family_ratio = np.column_stack(
                [
                    dataset.weights[:, phase, members].sum(axis=1) / max(proportional[members].sum(), 1e-12)
                    for members in dataset.family_members
                ]
            )
            phase_deficit = normalized_deficit(phase_family_ratio, config)
            phase_channels = (("net", phase_deficit),)
            if config.variant.split_response:
                phase_channels = (
                    ("shortage", np.maximum(phase_deficit, 0.0)),
                    ("surplus", -np.maximum(-phase_deficit, 0.0)),
                )
            for channel_name, channel in phase_channels:
                pieces.append(channel[:, selected_family_indices])
                names.extend(
                    f"phase{phase}_{channel_name}_family:{dataset.family_names[index]}"
                    for index in selected_family_indices
                )
                ridge.extend([1.0] * len(selected_family_indices))

    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    pieces.append(0.5 * np.abs(phase0_weight - phase1_weight).sum(axis=1, keepdims=True))
    names.append("phase_shift_tv")
    ridge.append(1.0)
    return Design(np.hstack(pieces), tuple(names), np.asarray(ridge, dtype=float))


def build_design(dataset: family_grp.Dataset, config: Config) -> Design:
    if config.variant is Variant.CURRENT:
        current = base.build_design(dataset, config.base)
        return Design(current.values, current.names, current.ridge_multipliers)
    return build_deficit_design(dataset, config)


def fit_model(dataset: family_grp.Dataset, config: Config, indices: np.ndarray) -> Model:
    design = build_design(dataset, config)
    x = design.values[indices]
    y = dataset.target[indices]
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    if config.base.l2 > 0.0:
        ridge = np.sqrt(config.base.l2 * design.ridge_multipliers)
        centered_x = np.vstack([centered_x, np.diag(ridge)])
        centered_y = np.concatenate([centered_y, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_x, centered_y, maxiter=40 * centered_x.shape[1])
    intercept = y_mean - float((x_mean @ coefficients).item())
    return Model(dataset, config, intercept, coefficients)


def oof_prediction(
    dataset: family_grp.Dataset,
    config: Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = fit_model(dataset, config, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {config.variant}")
    return prediction


def config_record(config: Config, metrics: dict[str, float | int]) -> dict[str, Any]:
    return {
        "variant": config.variant.value,
        "shape_index": config.base.shape_index,
        **asdict(config.base.shape),
        "l2": config.base.l2,
        "residual_shrink": config.base.residual_shrink,
        "deficit_floor": config.deficit_floor,
        "surplus_credit": config.surplus_credit,
        **metrics,
    }


def score_configs(
    dataset: family_grp.Dataset,
    configs: list[Config],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Config, np.ndarray, list[dict[str, Any]]]:
    best: tuple[float, float, Config, np.ndarray] | None = None
    rows: list[dict[str, Any]] = []
    for config in configs:
        prediction = oof_prediction(dataset, config, splits)
        metrics = calibration.calibration_summary(dataset.target, prediction)
        rows.append(config_record(config, metrics))
        candidate = (float(metrics["rmse"]), -float(metrics["spearman"]), config, prediction)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No deficit-response configuration was scored")
    return best[2], best[3], rows


def deficit_configs(
    variant: Variant,
    shapes: tuple[family_grp.Shape, ...],
    top_pairs: int,
    dataset: family_grp.Dataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[Config], list[dict[str, Any]]]:
    screen = [
        Config(
            variant,
            base.Config(
                base.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
                shape_index,
                shape,
                l2,
                10.0,
                0.0,
                0.0,
            ),
            floor,
            surplus_credit,
        )
        for shape_index, shape in enumerate(shapes)
        for l2 in base.L2_GRID
        for floor in DEFICIT_FLOOR_GRID
        for surplus_credit in (SURPLUS_CREDIT_GRID if variant.asymmetric_surplus else (1.0,))
    ]
    _config, _prediction, screen_rows = score_configs(dataset, screen, splits)
    best_by_pair: dict[tuple[int, float, float], float] = {}
    for row in screen_rows:
        key = (int(row["shape_index"]), float(row["deficit_floor"]), float(row["surplus_credit"]))
        best_by_pair[key] = min(best_by_pair.get(key, float("inf")), float(row["rmse"]))
    selected_pairs = [key for key, _score in sorted(best_by_pair.items(), key=lambda item: item[1])[:top_pairs]]
    configs = [
        Config(
            variant,
            base.Config(
                base.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
                shape_index,
                shapes[shape_index],
                l2,
                residual_shrink,
                0.0,
                0.0,
            ),
            floor,
            surplus_credit,
        )
        for shape_index, floor, surplus_credit in selected_pairs
        for l2 in base.L2_GRID
        for residual_shrink in base.RESIDUAL_SHRINK_GRID
    ]
    return configs, screen_rows


def benchmark_dataset(
    dataset_id: base.DatasetId,
    num_shapes: int,
    top_pairs: int,
    variants: tuple[Variant, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, num_shapes)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)

    selected: dict[Variant, tuple[Config, np.ndarray]] = {}
    screen_rows: list[dict[str, Any]] = []
    if Variant.CURRENT in variants:
        current_configs = [
            Config(Variant.CURRENT, item, 0.0)
            for item in base.structural_configs(
                base.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
                shapes,
                list(range(len(shapes))),
            )
        ]
        current, prediction, rows = score_configs(dataset, current_configs, splits)
        selected[Variant.CURRENT] = (current, prediction)
        screen_rows.extend({"dataset": dataset_id.value, **row} for row in rows)

    for variant in (variant for variant in variants if variant is not Variant.CURRENT):
        print(f"{dataset_id.value}: screening {variant.value}", flush=True)
        configs, first_stage = deficit_configs(variant, shapes, top_pairs, dataset, splits)
        config, prediction, second_stage = score_configs(dataset, configs, splits)
        selected[variant] = (config, prediction)
        screen_rows.extend({"dataset": dataset_id.value, "stage": "shape_floor", **row} for row in first_stage)
        screen_rows.extend({"dataset": dataset_id.value, "stage": "full", **row} for row in second_stage)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    heldout = base.heldout_data(dataset_id, dataset)
    for variant, (config, oof) in selected.items():
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "split": "fit_oof",
                **config_record(config, calibration.calibration_summary(dataset.target, oof)),
            }
        )
        for index, (observed, predicted) in enumerate(zip(dataset.target, oof, strict=True)):
            prediction_rows.append(
                {
                    "dataset": dataset_id.value,
                    "variant": variant.value,
                    "split": "fit_oof",
                    "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                    "group": str(dataset.frame.iloc[index].get("panel_source", "fit")),
                    "observed": observed,
                    "predicted": predicted,
                }
            )

        if heldout is None:
            continue
        heldout_frame, heldout_weights, heldout_target = heldout
        model = fit_model(dataset, config, np.arange(dataset.n))
        heldout_prediction = model.predict(heldout_weights)
        selected_index = int(np.argmin(heldout_prediction))
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "split": "heldout",
                **config_record(config, calibration.calibration_summary(heldout_target, heldout_prediction)),
                **base.grouped_heldout_summary(heldout_frame, heldout_target, heldout_prediction),
                "selected_run": str(heldout_frame.iloc[selected_index]["wandb_run_name"]),
                "selected_observed": float(heldout_target[selected_index]),
                "selected_predicted": float(heldout_prediction[selected_index]),
                "selected_optimism": float(heldout_target[selected_index] - heldout_prediction[selected_index]),
            }
        )
        for index, (observed, predicted) in enumerate(zip(heldout_target, heldout_prediction, strict=True)):
            prediction_rows.append(
                {
                    "dataset": dataset_id.value,
                    "variant": variant.value,
                    "split": "heldout",
                    "row_id": str(heldout_frame.iloc[index]["wandb_run_name"]),
                    "group": str(heldout_frame.iloc[index]["training_series"]),
                    "observed": observed,
                    "predicted": predicted,
                }
            )
    return metric_rows, screen_rows, prediction_rows


def render(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        Variant.CURRENT.value: "#d73027",
        Variant.LOG_DEFICIT.value: "#fdae61",
        Variant.POWER_DEFICIT.value: "#1a9850",
        Variant.LOG_DEFICIT_LITERAL_REPLAY.value: "#fee08b",
        Variant.POWER_DEFICIT_LITERAL_REPLAY.value: "#006837",
        Variant.LOG_DEFICIT_HYBRID_REPLAY.value: "#91cf60",
        Variant.POWER_DEFICIT_HYBRID_REPLAY.value: "#004529",
        Variant.POWER_DEFICIT_HYBRID_SPLIT_RESPONSE.value: "#018571",
        Variant.POWER_DEFICIT_HYBRID_ASYMMETRIC.value: "#2c7bb6",
        Variant.POWER_DEFICIT_HYBRID_BOTTLENECK.value: "#f46d43",
        Variant.POWER_DEFICIT_HYBRID_CONCENTRATION.value: "#313695",
        Variant.POWER_DEFICIT_HYBRID_FAMILY_IMBALANCE.value: "#542788",
        Variant.POWER_DEFICIT_HYBRID_JOINT.value: "#3288bd",
        Variant.POWER_DEFICIT_HYBRID_CONDITIONED_REPLAY.value: "#d73027",
        Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY.value: "#66bd63",
        Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_SPLIT_RESPONSE.value: "#238443",
        Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC.value: "#762a83",
        Variant.POWER_DEFICIT_HYBRID_EARLY_FOUNDATION_ASYMMETRIC.value: "#9970ab",
        Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_JOINT.value: "#5e4fa2",
        Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES.value: "#238443",
        Variant.POWER_DEFICIT_HYBRID_BOTH_FAMILIES_ASYMMETRIC.value: "#005a32",
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: frozen-heldout residuals",
            "Table-9: frozen-heldout residuals",
            "Fit-panel OOF RMSE",
            "Frozen-heldout RMSE",
        ),
    )
    datasets = (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9)
    for column, dataset_id in enumerate(datasets, start=1):
        selected = predictions.loc[predictions["dataset"].eq(dataset_id.value) & predictions["split"].eq("heldout")]
        for variant in Variant:
            local = selected.loc[selected["variant"].eq(variant.value)]
            figure.add_trace(
                go.Scatter(
                    x=local["observed"],
                    y=local["predicted"] - local["observed"],
                    mode="markers",
                    marker={"color": colors[variant.value], "size": 5, "opacity": 0.45},
                    name=variant.value,
                    legendgroup=variant.value,
                    showlegend=column == 1,
                    customdata=np.column_stack([local["row_id"], local["group"]]),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>observed=%{x:.5f}"
                        "<br>predicted-observed=%{y:.5f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=0.0, line={"color": "#64748b", "dash": "dash"}, row=1, col=column)

    for column, split in enumerate(("fit_oof", "heldout"), start=1):
        selected = metrics.loc[metrics["split"].eq(split)]
        for variant in Variant:
            local = selected.loc[selected["variant"].eq(variant.value)]
            figure.add_trace(
                go.Bar(
                    x=local["dataset"],
                    y=local["rmse"],
                    marker_color=colors[variant.value],
                    name=variant.value,
                    legendgroup=variant.value,
                    showlegend=False,
                ),
                row=2,
                col=column,
            )
    figure.update_layout(
        title="Scaling-law deficit response at 3e18",
        template="plotly_white",
        barmode="group",
        width=1600,
        height=1000,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.update_xaxes(title_text="Observed BPB", row=1)
    figure.update_yaxes(title_text="Prediction residual (predicted - observed)", row=1)
    figure.update_yaxes(title_text="RMSE", row=2)
    figure.write_html(output_dir / "deficit_response.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    dataset_ids = tuple(base.DatasetId(value) for value in args.datasets.split(",") if value)
    variants = tuple(Variant(value) for value in args.variants.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        print(f"Benchmarking {dataset_id.value}", flush=True)
        metrics, screens, predictions = benchmark_dataset(
            dataset_id,
            args.num_shapes,
            args.top_shape_floor_pairs,
            variants,
        )
        metric_rows.extend(metrics)
        screen_rows.extend(screens)
        prediction_rows.extend(predictions)

    metrics = pd.DataFrame(metric_rows)
    screens = pd.DataFrame(screen_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    screens.to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    render(metrics, predictions, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "datasets": [dataset.value for dataset in dataset_ids],
                "variants": [variant.value for variant in variants],
                "deficit_floor_grid": list(DEFICIT_FLOOR_GRID),
                "selection": "fit-panel five-fold CV; heldout archive frozen until final scoring",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    columns = [
        "dataset",
        "variant",
        "split",
        "rmse",
        "spearman",
        "regret_at_1",
        "lower_tail_optimism",
        "optimism_gt_0p05_count",
        "worst_optimism",
        "selected_optimism",
    ]
    report_columns = [column for column in columns if column in metrics.columns]
    (args.output_dir / "report.md").write_text(
        "# Scaling-law deficit response\n\n"
        "The deficit response is selected only on the fit panel; historical 3e18 validations remain frozen.\n\n"
        + metrics[report_columns].to_markdown(index=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
