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
# ]
# ///
"""Benchmark compact structural extensions to Bucket-family GRP.

The current Bucket-family GRP has one nonnegative benefit coefficient per
bucket, nonlinear family-coverage benefits, and family replay-harm channels.
Its 3e18 fit-panel OOF score is strong, but its transfer to previously trained
optimizer proposals is worse than the coarser original GRP.

This benchmark tests nested, mechanistic changes. The promoted candidate uses:

* hierarchical pooling represents each non-singleton family's bucket utility
  as a shared family coefficient plus nonnegative bucket-specific excess;
* family-level saturating coverage and overexposure channels;
* family-averaged member replay harm to capture concentration within a family;
* one global phase-transition cost rather than per-bucket phase interactions.

The file also retains rejected structural variants so the negative modeling
tests are reproducible rather than silently discarded.

All nonlinear shapes and shrinkage strengths are selected using fit-panel CV.
The historical 3e18 validation archive is used only as a development transfer
set and is never included in the fit or hyperparameter-selection loss.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_grp_saturation_hierarchy_20260714 as hierarchy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_coverage_grp_20260715"
OUTER_SEED = 7151
SCREEN_SEED = 7152
N_SPLITS = 5
L2_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0)
RESIDUAL_SHRINK_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
UNDERCOVERAGE_FRACTION_GRID = (0.25, 0.5, 0.75, 1.0)
COVERAGE_GATE_RATIO_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class DatasetId(StrEnum):
    PRODUCTION_UNCHEATABLE = "production_uncheatable"
    THREE_HUNDRED_M_UNCHEATABLE = "300m_uncheatable"
    THREE_HUNDRED_M_TABLE9 = "300m_table9"
    DELPHI_3E18_UNCHEATABLE = "delphi_3e18_uncheatable"
    DELPHI_3E18_TABLE9 = "delphi_3e18_table9"


class Variant(StrEnum):
    BUCKET_RESOLVED = "bucket_resolved"
    HIERARCHICAL = "hierarchical"
    BUCKET_RESOLVED_UNDERCOVERAGE = "bucket_resolved_undercoverage"
    HIERARCHICAL_UNDERCOVERAGE = "hierarchical_undercoverage"
    HIERARCHICAL_COVERAGE_GATE = "hierarchical_coverage_gate"
    SOURCE_GROUP_HIERARCHICAL = "source_group_hierarchical"
    HIERARCHICAL_MEMBER_DEFICIT = "hierarchical_member_deficit"
    HIERARCHICAL_PHASE_SHIFT = "hierarchical_phase_shift"
    HIERARCHICAL_GEOMETRY = "hierarchical_geometry"
    HIERARCHICAL_INFORMATION = "hierarchical_information"
    HIERARCHICAL_BUCKET_REPLAY = "hierarchical_bucket_replay"
    HIERARCHICAL_PHASE_BUCKET_REPLAY = "hierarchical_phase_bucket_replay"
    HIERARCHICAL_FAMILY_CONCENTRATION = "hierarchical_family_concentration"
    HIERARCHICAL_PHASE_REPLAY_CONCENTRATION = "hierarchical_phase_replay_concentration"
    FAMILY_COMPOSITION = "family_composition"
    FAMILY_COMPOSITION_PHASE_REPLAY = "family_composition_phase_replay"
    HIERARCHICAL_PHASE_EXCESS_REPLAY = "hierarchical_phase_excess_replay"
    HIERARCHICAL_PHASE_REPLAY_COMPLEMENTARITY = "hierarchical_phase_replay_complementarity"

    @property
    def hierarchical(self) -> bool:
        return self in {
            Variant.HIERARCHICAL,
            Variant.HIERARCHICAL_UNDERCOVERAGE,
            Variant.HIERARCHICAL_COVERAGE_GATE,
            Variant.SOURCE_GROUP_HIERARCHICAL,
            Variant.HIERARCHICAL_MEMBER_DEFICIT,
            Variant.HIERARCHICAL_PHASE_SHIFT,
            Variant.HIERARCHICAL_GEOMETRY,
            Variant.HIERARCHICAL_INFORMATION,
            Variant.HIERARCHICAL_BUCKET_REPLAY,
            Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            Variant.HIERARCHICAL_FAMILY_CONCENTRATION,
            Variant.HIERARCHICAL_PHASE_REPLAY_CONCENTRATION,
            Variant.HIERARCHICAL_PHASE_EXCESS_REPLAY,
            Variant.HIERARCHICAL_PHASE_REPLAY_COMPLEMENTARITY,
        }

    @property
    def undercoverage(self) -> bool:
        return self in {
            Variant.BUCKET_RESOLVED_UNDERCOVERAGE,
            Variant.HIERARCHICAL_UNDERCOVERAGE,
        }

    @property
    def coverage_gate(self) -> bool:
        return self is Variant.HIERARCHICAL_COVERAGE_GATE

    @property
    def source_group_pooling(self) -> bool:
        return self is Variant.SOURCE_GROUP_HIERARCHICAL

    @property
    def member_deficit(self) -> bool:
        return self is Variant.HIERARCHICAL_MEMBER_DEFICIT

    @property
    def phase_shift(self) -> bool:
        return self in {
            Variant.HIERARCHICAL_PHASE_SHIFT,
            Variant.HIERARCHICAL_GEOMETRY,
            Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            Variant.HIERARCHICAL_PHASE_REPLAY_CONCENTRATION,
            Variant.FAMILY_COMPOSITION_PHASE_REPLAY,
            Variant.HIERARCHICAL_PHASE_EXCESS_REPLAY,
            Variant.HIERARCHICAL_PHASE_REPLAY_COMPLEMENTARITY,
        }

    @property
    def aggregate_geometry(self) -> bool:
        return self is Variant.HIERARCHICAL_GEOMETRY

    @property
    def information_geometry(self) -> bool:
        return self is Variant.HIERARCHICAL_INFORMATION

    @property
    def bucket_replay(self) -> bool:
        return self in {
            Variant.HIERARCHICAL_BUCKET_REPLAY,
            Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            Variant.HIERARCHICAL_PHASE_REPLAY_CONCENTRATION,
            Variant.FAMILY_COMPOSITION_PHASE_REPLAY,
            Variant.HIERARCHICAL_PHASE_REPLAY_COMPLEMENTARITY,
        }

    @property
    def family_concentration(self) -> bool:
        return self in {
            Variant.HIERARCHICAL_FAMILY_CONCENTRATION,
            Variant.HIERARCHICAL_PHASE_REPLAY_CONCENTRATION,
        }

    @property
    def family_composition(self) -> bool:
        return self in {
            Variant.FAMILY_COMPOSITION,
            Variant.FAMILY_COMPOSITION_PHASE_REPLAY,
        }

    @property
    def excess_replay(self) -> bool:
        return self is Variant.HIERARCHICAL_PHASE_EXCESS_REPLAY

    @property
    def family_complementarity(self) -> bool:
        return self is Variant.HIERARCHICAL_PHASE_REPLAY_COMPLEMENTARITY


@dataclass(frozen=True)
class Config:
    variant: Variant
    shape_index: int
    shape: family_grp.Shape
    l2: float
    residual_shrink: float
    undercoverage_fraction: float
    coverage_gate_ratio: float


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
        default=",".join(dataset.value for dataset in DatasetId),
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(variant.value for variant in Variant),
        help="Comma-separated model variants; bucket_resolved is always included as the reference.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=12)
    parser.add_argument("--top-shapes", type=int, default=3)
    return parser.parse_args()


def family_dataset(raw: pooled.Dataset) -> family_grp.Dataset:
    names, members = hierarchy.family_partition(raw)
    return family_grp.Dataset(
        frame=raw.frame,
        target=np.asarray(raw.y, dtype=float),
        weights=np.asarray(raw.weights, dtype=float),
        c0=np.asarray(raw.c0, dtype=float),
        c1=np.asarray(raw.c1, dtype=float),
        domains=tuple(raw.domain_names),
        family_names=names,
        family_members=members,
        quality=np.full(raw.m, -1, dtype=int),
    )


def load_dataset(dataset_id: DatasetId) -> family_grp.Dataset:
    if dataset_id is DatasetId.PRODUCTION_UNCHEATABLE:
        return hierarchy.load_dataset(hierarchy.DatasetId.PRODUCTION_UNCHEATABLE)
    if dataset_id is DatasetId.THREE_HUNDRED_M_UNCHEATABLE:
        return hierarchy.load_dataset(hierarchy.DatasetId.THREE_HUNDRED_M_UNCHEATABLE)
    if dataset_id is DatasetId.THREE_HUNDRED_M_TABLE9:
        return hierarchy.load_dataset(hierarchy.DatasetId.THREE_HUNDRED_M_TABLE9)
    # Local import keeps the reusable model definitions independent of the
    # Observatory, which imports this module to expose the promoted form.
    from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: PLC0415
        export_mixture_fit_observatory as observatory,
    )

    target = "uncheatable" if dataset_id is DatasetId.DELPHI_3E18_UNCHEATABLE else "table9"
    return family_dataset(observatory.load_delphi_3e18_fit_dataset(target))


def split_indices(
    dataset: family_grp.Dataset,
    dataset_id: DatasetId,
    indices: np.ndarray,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if dataset_id is DatasetId.PRODUCTION_UNCHEATABLE:
        return family_grp.kfold_indices(indices, N_SPLITS, seed)
    local = dataset.frame.iloc[indices].reset_index(drop=True)
    local_splits = component_dsp.panel_stratified_folds(local, n_splits=N_SPLITS, seed=seed)
    return [(indices[train], indices[test]) for train, test in local_splits]


def retained_exposure(dataset: family_grp.Dataset, shape: family_grp.Shape) -> np.ndarray:
    phase0_weight = dataset.weights[:, 0, :]
    phase1_weight = dataset.weights[:, 1, :]
    early = phase0_weight * dataset.c0[None, :]
    late = phase1_weight * dataset.c1[None, :]
    retained_early = np.exp(-shape.forgetting_rate * (1.0 - phase1_weight)) * early
    return np.maximum(retained_early + shape.late_multiplier * late, 0.0)


def proportional_weights(dataset: family_grp.Dataset) -> np.ndarray:
    phase_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    phase0_tokens = phase_fraction / np.maximum(dataset.c0, 1e-12)
    phase1_tokens = (1.0 - phase_fraction) / np.maximum(dataset.c1, 1e-12)
    token_proxy = 0.5 * (phase0_tokens + phase1_tokens)
    return token_proxy / token_proxy.sum()


def proportional_family_exposure(
    dataset: family_grp.Dataset,
    shape: family_grp.Shape,
) -> np.ndarray:
    weights = proportional_weights(dataset)
    reference = replace(
        dataset,
        weights=np.stack([weights, weights], axis=0)[None, :, :],
        target=np.zeros(1, dtype=float),
    )
    exposure = retained_exposure(reference, shape)[0]
    return np.asarray([exposure[members].sum() for members in dataset.family_members], dtype=float)


def proportional_bucket_exposure(
    dataset: family_grp.Dataset,
    shape: family_grp.Shape,
) -> np.ndarray:
    weights = proportional_weights(dataset)
    reference = replace(
        dataset,
        weights=np.stack([weights, weights], axis=0)[None, :, :],
        target=np.zeros(1, dtype=float),
    )
    return retained_exposure(reference, shape)[0]


def power_response(exposure: np.ndarray, exponent: float) -> np.ndarray:
    return np.maximum(exposure, 1e-12) ** exponent


def overexposure_harm(exposure: np.ndarray, threshold: float) -> np.ndarray:
    delta = np.log1p(np.maximum(exposure, 0.0)) - threshold
    return np.logaddexp(0.0, delta) ** 2


def excess_replay_harm(exposure: np.ndarray, log_threshold: float) -> np.ndarray:
    """Cumulative harm after the existing replay-onset exposure."""
    onset = max(float(np.expm1(log_threshold)), 1e-8)
    return np.maximum(exposure / onset - 1.0, 0.0) ** 2


def undercoverage_harm(exposure: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    delta = np.log1p(np.maximum(threshold, 0.0))[None, :] - np.log1p(np.maximum(exposure, 0.0))
    return np.logaddexp(0.0, delta) ** 2


def member_coverage_deficit(
    exposure: np.ndarray,
    reference: np.ndarray,
    fraction: float,
    members: tuple[np.ndarray, ...],
) -> np.ndarray:
    threshold = fraction * reference
    ratio = np.divide(
        exposure,
        threshold[None, :],
        out=np.ones_like(exposure),
        where=threshold[None, :] > 1e-12,
    )
    deficit = np.maximum(1.0 - ratio, 0.0) ** 2
    return np.column_stack([deficit[:, indices].mean(axis=1) for indices in members])


def row_kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    safe_left = np.maximum(left, 1e-12)
    safe_right = np.maximum(right, 1e-12)
    return np.sum(left * np.log(safe_left / safe_right), axis=1)


def source_groups(dataset: family_grp.Dataset) -> tuple[tuple[str, np.ndarray], ...]:
    grouped: dict[str, list[int]] = {}
    for index, domain in enumerate(dataset.domains):
        production_match = re.fullmatch(r"c(?P<family>\d+)q\d+", domain)
        if production_match is not None:
            group = f"c{production_match.group('family')}"
        elif domain.startswith("dolma3_cc/") and domain.endswith(("_high", "_low")):
            group = re.sub(r"_(?:high|low)$", "", domain)
        else:
            group = domain
        grouped.setdefault(group, []).append(index)
    return tuple((name, np.asarray(members, dtype=int)) for name, members in grouped.items())


def pooling_groups(dataset: family_grp.Dataset, variant: Variant) -> tuple[tuple[str, np.ndarray], ...]:
    if variant.source_group_pooling:
        return source_groups(dataset)
    return tuple(zip(dataset.family_names, dataset.family_members, strict=True))


def build_design(dataset: family_grp.Dataset, config: Config) -> Design:
    exposure = retained_exposure(dataset, config.shape)
    bucket_signal = power_response(exposure, config.shape.exponent)
    family_total = np.column_stack([exposure[:, members].sum(axis=1) for members in dataset.family_members])
    nonsingleton = tuple(index for index, members in enumerate(dataset.family_members) if len(members) > 1)
    if config.variant.coverage_gate:
        reference = proportional_family_exposure(dataset, config.shape)
        for family_index in nonsingleton:
            members = dataset.family_members[family_index]
            ratio = config.coverage_gate_ratio
            gate = family_total[:, family_index] / (
                family_total[:, family_index] + ratio * reference[family_index] + 1e-12
            )
            proportional_gate = 1.0 / (1.0 + ratio)
            bucket_signal[:, members] *= (gate / proportional_gate)[:, None]
    pieces: list[np.ndarray] = []
    names: list[str] = []
    ridge: list[float] = []

    if config.variant.family_composition:
        # A family's exposure controls saturation; its composition controls
        # average quality. Related buckets therefore cannot earn independent
        # full diminishing-return benefits merely by being co-selected.
        for family_index, (family_name, members) in enumerate(
            zip(dataset.family_names, dataset.family_members, strict=True)
        ):
            total = np.maximum(family_total[:, [family_index]], 1e-12)
            family_signal = power_response(total, config.shape.exponent)
            shares = exposure[:, members] / total
            pieces.append(-family_signal * shares)
            names.extend(f"family_quality:{family_name}:{dataset.domains[index]}" for index in members)
            ridge.extend([1.0] * len(members))
    elif config.variant.hierarchical:
        groups = pooling_groups(dataset, config.variant)
        singleton = [members[0] for _name, members in groups if len(members) == 1]
        if singleton:
            pieces.append(-bucket_signal[:, singleton])
            names.extend(f"singleton_signal:{dataset.domains[index]}" for index in singleton)
            ridge.extend([1.0] * len(singleton))
        nonsingleton_groups = [(name, members) for name, members in groups if len(members) > 1]
        for group_name, members in nonsingleton_groups:
            pieces.append(-bucket_signal[:, members].sum(axis=1, keepdims=True))
            names.append(f"pooled_base_signal:{group_name}")
            ridge.append(1.0)
        if nonsingleton_groups:
            residual_members = np.concatenate([members for _name, members in nonsingleton_groups])
            pieces.append(-bucket_signal[:, residual_members])
            names.extend(f"bucket_excess_signal:{dataset.domains[index]}" for index in residual_members)
            ridge.extend([config.residual_shrink] * len(residual_members))
    else:
        pieces.append(-bucket_signal)
        names.extend(f"bucket_signal:{domain}" for domain in dataset.domains)
        ridge.extend([1.0] * dataset.m)

    if nonsingleton and not config.variant.family_composition:
        family_signal = power_response(family_total[:, nonsingleton], config.shape.exponent)
        pieces.append(-family_signal)
        names.extend(f"family_coverage_signal:{dataset.family_names[index]}" for index in nonsingleton)
        ridge.extend([1.0] * len(nonsingleton))

    pieces.append(overexposure_harm(family_total, config.shape.penalty_threshold))
    names.extend(f"family_overexposure:{name}" for name in dataset.family_names)
    ridge.extend([1.0] * len(dataset.family_names))

    if config.variant.family_complementarity:
        reference = proportional_family_exposure(dataset, config.shape)
        normalized = family_total / np.maximum(reference[None, :], 1e-12)
        joint_coverage = np.expm1(np.mean(np.log1p(normalized), axis=1))
        pieces.append(-joint_coverage[:, None])
        names.append("joint_family_coverage")
        ridge.append(1.0)

    if config.variant.bucket_replay:
        bucket_harm = overexposure_harm(exposure, config.shape.penalty_threshold)
        pieces.append(np.column_stack([bucket_harm[:, members].mean(axis=1) for members in dataset.family_members]))
        names.extend(f"family_member_replay:{name}" for name in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))

    if config.variant.excess_replay:
        bucket_harm = excess_replay_harm(exposure, config.shape.penalty_threshold)
        pieces.append(np.column_stack([bucket_harm[:, members].mean(axis=1) for members in dataset.family_members]))
        names.extend(f"family_excess_replay:{name}" for name in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))

    if config.variant.family_concentration:
        concentration_pieces = []
        concentration_names = []
        for family_index, (family_name, members) in enumerate(
            zip(dataset.family_names, dataset.family_members, strict=True)
        ):
            if len(members) == 1:
                continue
            shares = exposure[:, members] / np.maximum(family_total[:, [family_index]], 1e-12)
            hhi = np.sum(shares**2, axis=1)
            normalized = (hhi - 1.0 / len(members)) / (1.0 - 1.0 / len(members))
            concentration_pieces.append(normalized[:, None])
            concentration_names.append(f"family_concentration:{family_name}")
        if concentration_pieces:
            pieces.extend(concentration_pieces)
            names.extend(concentration_names)
            ridge.extend([1.0] * len(concentration_pieces))

    if config.variant.undercoverage:
        reference = proportional_family_exposure(dataset, config.shape)
        threshold = config.undercoverage_fraction * reference
        pieces.append(undercoverage_harm(family_total, threshold))
        names.extend(f"family_undercoverage:{name}" for name in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))

    if config.variant.member_deficit:
        reference = proportional_bucket_exposure(dataset, config.shape)
        pieces.append(
            member_coverage_deficit(
                exposure,
                reference,
                config.undercoverage_fraction,
                dataset.family_members,
            )
        )
        names.extend(f"family_member_deficit:{name}" for name in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))

    if config.variant.phase_shift:
        phase0_weight = dataset.weights[:, 0, :]
        phase1_weight = dataset.weights[:, 1, :]
        pieces.append(0.5 * np.abs(phase0_weight - phase1_weight).sum(axis=1, keepdims=True))
        names.append("phase_shift_tv")
        ridge.append(1.0)
        if config.variant.aggregate_geometry:
            phase_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
            aggregate = phase_fraction * phase0_weight + (1.0 - phase_fraction) * phase1_weight
            pieces.append(np.column_stack([np.sum(aggregate**2, axis=1), np.sum(phase1_weight**2, axis=1)]))
            names.extend(["aggregate_concentration", "late_phase_concentration"])
            ridge.extend([1.0, 1.0])

    if config.variant.information_geometry:
        phase0_weight = dataset.weights[:, 0, :]
        phase1_weight = dataset.weights[:, 1, :]
        phase_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
        aggregate = phase_fraction * phase0_weight + (1.0 - phase_fraction) * phase1_weight
        natural = proportional_weights(dataset)
        aggregate_shift = row_kl(aggregate, np.broadcast_to(natural, aggregate.shape))
        phase_information = phase_fraction * row_kl(phase0_weight, aggregate) + (1.0 - phase_fraction) * row_kl(
            phase1_weight,
            aggregate,
        )
        pieces.append(np.column_stack([aggregate_shift, phase_information]))
        names.extend(["aggregate_kl_to_proportional", "phase_information"])
        ridge.extend([1.0, 1.0])

    return Design(np.hstack(pieces), tuple(names), np.asarray(ridge, dtype=float))


def fit_model(dataset: family_grp.Dataset, config: Config, indices: np.ndarray) -> Model:
    design = build_design(dataset, config)
    train_design = design.values[indices]
    train_target = dataset.target[indices]
    design_mean = train_design.mean(axis=0, keepdims=True)
    target_mean = float(train_target.mean())
    centered_design = train_design - design_mean
    centered_target = train_target - target_mean
    if config.l2 > 0.0:
        ridge = np.sqrt(config.l2 * design.ridge_multipliers)
        centered_design = np.vstack([centered_design, np.diag(ridge)])
        centered_target = np.concatenate([centered_target, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_design, centered_target, maxiter=40 * centered_design.shape[1])
    intercept = target_mean - float((design_mean @ coefficients).item())
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


def metric_summary(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = prediction - observed
    lower_tail_count = min(
        len(observed),
        max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))),
    )
    lower_tail = np.argsort(prediction)[:lower_tail_count]
    lower_tail_error = observed[lower_tail] - prediction[lower_tail]
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "regret_at_1": float(observed[int(np.argmin(prediction))] - np.min(observed)),
        "lower_tail_optimism": float(np.mean(np.maximum(lower_tail_error, 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(lower_tail_error**2))),
        "lower_tail_count": lower_tail_count,
    }


def config_record(config: Config, metrics: dict[str, float | int]) -> dict[str, Any]:
    return {
        "variant": config.variant.value,
        "shape_index": config.shape_index,
        **asdict(config.shape),
        "l2": config.l2,
        "residual_shrink": config.residual_shrink,
        "undercoverage_fraction": config.undercoverage_fraction,
        "coverage_gate_ratio": config.coverage_gate_ratio,
        **metrics,
    }


def baseline_configs(shapes: tuple[family_grp.Shape, ...]) -> list[Config]:
    return [
        Config(Variant.BUCKET_RESOLVED, shape_index, shape, l2, 1.0, 0.0, 0.0)
        for shape_index, shape in enumerate(shapes)
        for l2 in L2_GRID
    ]


def structural_configs(
    variant: Variant,
    shapes: tuple[family_grp.Shape, ...],
    shape_indices: list[int],
) -> list[Config]:
    residual_grid = RESIDUAL_SHRINK_GRID if variant.hierarchical else (1.0,)
    undercoverage_grid = UNDERCOVERAGE_FRACTION_GRID if variant.undercoverage or variant.member_deficit else (0.0,)
    coverage_gate_grid = COVERAGE_GATE_RATIO_GRID if variant.coverage_gate else (0.0,)
    return [
        Config(
            variant,
            shape_index,
            shapes[shape_index],
            l2,
            residual_shrink,
            undercoverage_fraction,
            coverage_gate_ratio,
        )
        for shape_index in shape_indices
        for l2 in L2_GRID
        for residual_shrink in residual_grid
        for undercoverage_fraction in undercoverage_grid
        for coverage_gate_ratio in coverage_gate_grid
    ]


def score_configs(
    dataset: family_grp.Dataset,
    configs: list[Config],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Config, np.ndarray, list[dict[str, Any]]]:
    best: tuple[float, float, Config, np.ndarray] | None = None
    rows: list[dict[str, Any]] = []
    for config in configs:
        prediction = oof_prediction(dataset, config, splits)
        metrics = metric_summary(dataset.target, prediction)
        rows.append(config_record(config, metrics))
        candidate = (float(metrics["rmse"]), -float(metrics["spearman"]), config, prediction)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No configurations were scored")
    return best[2], best[3], rows


def heldout_data(
    dataset_id: DatasetId,
    fit_dataset: family_grp.Dataset,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    if dataset_id not in {DatasetId.DELPHI_3E18_UNCHEATABLE, DatasetId.DELPHI_3E18_TABLE9}:
        return None
    from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: PLC0415
        export_mixture_fit_observatory as observatory,
    )

    raw_reference = observatory.load_delphi_3e18_fit_dataset(
        "uncheatable" if dataset_id is DatasetId.DELPHI_3E18_UNCHEATABLE else "table9"
    )
    frame, weights = observatory.load_delphi_3e18_heldouts(raw_reference)
    keep = frame["fit_panel_overlap"].eq("coordinate_disjoint").to_numpy()
    target_column = "uncheatable_bpb" if dataset_id is DatasetId.DELPHI_3E18_UNCHEATABLE else "table9_macro_bpb"
    return frame.loc[keep].reset_index(drop=True), weights[keep], frame.loc[keep, target_column].to_numpy(float)


def grouped_heldout_summary(
    frame: pd.DataFrame,
    observed: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    rows = []
    for _series, indices in frame.groupby("training_series", sort=False).indices.items():
        local = np.asarray(indices, dtype=int)
        rows.append(metric_summary(observed[local], prediction[local]))
    return {
        "series_count": len(rows),
        "series_macro_rmse": float(np.mean([row["rmse"] for row in rows])),
        "series_macro_regret_at_1": float(np.mean([row["regret_at_1"] for row in rows])),
        "series_macro_lower_tail_optimism": float(np.mean([row["lower_tail_optimism"] for row in rows])),
    }


def benchmark_dataset(
    dataset_id: DatasetId,
    variants: tuple[Variant, ...],
    num_shapes: int,
    top_shapes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = load_dataset(dataset_id)
    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, num_shapes)
    splits = split_indices(dataset, dataset_id, np.arange(dataset.n), SCREEN_SEED)
    baseline_config, baseline_prediction, baseline_rows = score_configs(dataset, baseline_configs(shapes), splits)
    best_by_shape: dict[int, float] = {}
    for row in baseline_rows:
        index = int(row["shape_index"])
        best_by_shape[index] = min(best_by_shape.get(index, float("inf")), float(row["rmse"]))
    shape_indices = [index for index, _score in sorted(best_by_shape.items(), key=lambda item: item[1])[:top_shapes]]

    selected: dict[Variant, tuple[Config, np.ndarray]] = {
        Variant.BUCKET_RESOLVED: (baseline_config, baseline_prediction)
    }
    screen_rows = [{"dataset": dataset_id.value, **row} for row in baseline_rows]
    for variant in variants:
        if variant is Variant.BUCKET_RESOLVED:
            continue
        config, prediction, rows = score_configs(
            dataset,
            structural_configs(variant, shapes, shape_indices),
            splits,
        )
        selected[variant] = (config, prediction)
        screen_rows.extend({"dataset": dataset_id.value, **row} for row in rows)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    heldout = heldout_data(dataset_id, dataset)
    for variant, (config, oof) in selected.items():
        fit_metrics = metric_summary(dataset.target, oof)
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "split": "fit_oof",
                **config_record(config, fit_metrics),
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
                    "policy_class": "two_phase_fit",
                    "observed": observed,
                    "predicted": predicted,
                }
            )
        if heldout is None:
            continue
        heldout_frame, heldout_weights, heldout_target = heldout
        full_model = fit_model(dataset, config, np.arange(dataset.n))
        heldout_prediction = full_model.predict(heldout_weights)
        phase_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
        heldout_aggregate = phase_fraction * heldout_weights[:, 0, :] + (1.0 - phase_fraction) * heldout_weights[:, 1, :]
        natural = proportional_weights(dataset)
        aggregate_kl = row_kl(heldout_aggregate, np.broadcast_to(natural, heldout_aggregate.shape))
        phase_tv = 0.5 * np.abs(heldout_weights[:, 0, :] - heldout_weights[:, 1, :]).sum(axis=1)
        candidate_dataset = replace(dataset, weights=heldout_weights, target=np.zeros(len(heldout_weights)))
        exposure = retained_exposure(candidate_dataset, config.shape)
        reference_exposure = proportional_bucket_exposure(dataset, config.shape)
        exposure_ratio = exposure / np.maximum(reference_exposure[None, :], 1e-12)
        heldout_metrics = metric_summary(heldout_target, heldout_prediction)
        group_metrics = grouped_heldout_summary(heldout_frame, heldout_target, heldout_prediction)
        selected_index = int(np.argmin(heldout_prediction))
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "split": "heldout",
                **config_record(config, heldout_metrics),
                **group_metrics,
                "selected_observed": float(heldout_target[selected_index]),
                "selected_predicted": float(heldout_prediction[selected_index]),
                "selected_optimism": float(heldout_target[selected_index] - heldout_prediction[selected_index]),
                "selected_run": str(heldout_frame.iloc[selected_index]["wandb_run_name"]),
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
                    "policy_class": str(heldout_frame.iloc[index]["policy_class"]),
                    "observed": observed,
                    "predicted": predicted,
                    "phase_tv": phase_tv[index],
                    "aggregate_kl": aggregate_kl[index],
                    "max_weight": float(heldout_weights[index].max()),
                    "max_exposure_ratio": float(exposure_ratio[index].max()),
                    "under_quarter_exposure_count": int(np.sum(exposure_ratio[index] < 0.25)),
                }
            )
    return metric_rows, screen_rows, prediction_rows


def render(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    delphi = metrics.loc[metrics["dataset"].str.startswith("delphi_3e18")]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Fit-panel OOF RMSE",
            "Existing 3e18 heldout RMSE",
            "3e18 Uncheatable heldout calibration",
            "3e18 Table-9 heldout calibration",
        ),
    )
    colors = {
        Variant.BUCKET_RESOLVED.value: "#d73027",
        Variant.HIERARCHICAL.value: "#fc8d59",
        Variant.BUCKET_RESOLVED_UNDERCOVERAGE.value: "#91cf60",
        Variant.HIERARCHICAL_UNDERCOVERAGE.value: "#1a9850",
        Variant.HIERARCHICAL_COVERAGE_GATE.value: "#006837",
        Variant.SOURCE_GROUP_HIERARCHICAL.value: "#2166ac",
        Variant.HIERARCHICAL_MEMBER_DEFICIT.value: "#762a83",
        Variant.HIERARCHICAL_PHASE_SHIFT.value: "#1b7837",
        Variant.HIERARCHICAL_GEOMETRY.value: "#00441b",
        Variant.HIERARCHICAL_INFORMATION.value: "#5e3c99",
        Variant.HIERARCHICAL_BUCKET_REPLAY.value: "#e66101",
        Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY.value: "#b35806",
        Variant.HIERARCHICAL_FAMILY_CONCENTRATION.value: "#80cdc1",
        Variant.HIERARCHICAL_PHASE_REPLAY_CONCENTRATION.value: "#01665e",
        Variant.FAMILY_COMPOSITION.value: "#7f3b08",
        Variant.FAMILY_COMPOSITION_PHASE_REPLAY.value: "#542788",
        Variant.HIERARCHICAL_PHASE_EXCESS_REPLAY.value: "#2d004b",
        Variant.HIERARCHICAL_PHASE_REPLAY_COMPLEMENTARITY.value: "#008837",
    }
    fit = delphi.loc[delphi["split"].eq("fit_oof")]
    heldout = delphi.loc[delphi["split"].eq("heldout")]
    present_variants = tuple(Variant(value) for value in metrics["variant"].drop_duplicates())
    for variant in present_variants:
        selected = fit.loc[fit["variant"].eq(variant.value)]
        figure.add_trace(
            go.Bar(
                x=selected["dataset"],
                y=selected["rmse"],
                name=variant.value,
                marker_color=colors[variant.value],
                legendgroup=variant.value,
            ),
            row=1,
            col=1,
        )
        selected = heldout.loc[heldout["variant"].eq(variant.value)]
        figure.add_trace(
            go.Bar(
                x=selected["dataset"],
                y=selected["rmse"],
                name=variant.value,
                marker_color=colors[variant.value],
                legendgroup=variant.value,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    for column, dataset_id in enumerate(
        (DatasetId.DELPHI_3E18_UNCHEATABLE, DatasetId.DELPHI_3E18_TABLE9),
        start=1,
    ):
        selected = predictions.loc[predictions["dataset"].eq(dataset_id.value) & predictions["split"].eq("heldout")]
        bounds = [
            float(selected[["observed", "predicted"]].min().min()),
            float(selected[["observed", "predicted"]].max().max()),
        ]
        figure.add_trace(
            go.Scatter(x=bounds, y=bounds, mode="lines", line={"color": "#777", "dash": "dash"}, showlegend=False),
            row=2,
            col=column,
        )
        for variant in present_variants:
            local = selected.loc[selected["variant"].eq(variant.value)]
            figure.add_trace(
                go.Scatter(
                    x=local["predicted"],
                    y=local["observed"],
                    mode="markers",
                    marker={"color": colors[variant.value], "size": 5, "opacity": 0.45},
                    name=variant.value,
                    legendgroup=variant.value,
                    showlegend=False,
                    customdata=np.column_stack([local["row_id"], local["group"]]),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>pred=%{x:.5f}<br>obs=%{y:.5f}<extra></extra>",
                ),
                row=2,
                col=column,
            )
    figure.update_layout(
        title="Hierarchical pooling and family undercoverage in Bucket-family GRP",
        template="plotly_white",
        barmode="group",
        width=1500,
        height=1000,
        legend={"orientation": "h", "y": 1.08},
    )
    figure.update_xaxes(title_text="Predicted BPB", row=2)
    figure.update_yaxes(title_text="Observed BPB", row=2)
    figure.write_html(output_dir / "benchmark.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    output = [f"| {' | '.join(headers)} |", f"| {' | '.join('---' for _ in headers)} |"]
    output.extend(f"| {' | '.join(row)} |" for row in rows)
    return "\n".join(output)


def write_report(metrics: pd.DataFrame, policy_metrics: pd.DataFrame, output_dir: Path) -> None:
    rows: list[list[str]] = []
    dataset_ids = [DatasetId(value) for value in metrics["dataset"].drop_duplicates()]
    for dataset in dataset_ids:
        for variant in tuple(Variant(value) for value in metrics["variant"].drop_duplicates()):
            fit = metrics.loc[
                metrics["dataset"].eq(dataset.value)
                & metrics["variant"].eq(variant.value)
                & metrics["split"].eq("fit_oof")
            ].iloc[0]
            heldout = metrics.loc[
                metrics["dataset"].eq(dataset.value)
                & metrics["variant"].eq(variant.value)
                & metrics["split"].eq("heldout")
            ]
            rows.append(
                [
                    dataset.value,
                    variant.value,
                    f"{fit['rmse']:.5f}",
                    f"{fit['spearman']:.3f}",
                    "-" if heldout.empty else f"{heldout.iloc[0]['rmse']:.5f}",
                    "-" if heldout.empty else f"{heldout.iloc[0]['spearman']:.3f}",
                    "-" if heldout.empty else f"{heldout.iloc[0]['regret_at_1']:.5f}",
                    "-" if heldout.empty else f"{heldout.iloc[0]['selected_optimism']:.5f}",
                ]
            )
    policy_rows: list[list[str]] = []
    for row in policy_metrics.to_dict(orient="records"):
        policy_rows.append(
            [
                str(row["dataset"]),
                str(row["variant"]),
                str(row["policy_class"]),
                f"{float(row['rmse']):.5f}",
                f"{float(row['spearman']):.3f}",
                f"{float(row['regret_at_1']):.5f}",
                f"{float(row['selected_optimism']):.5f}",
                str(row["selected_run"]),
            ]
        )
    report = [
        "# Hierarchical coverage GRP benchmark",
        "",
        (
            "Hyperparameters are selected only by fit-panel CV. Historical 3e18 validations are a heterogeneous "
            "development transfer set, not IID test data."
        ),
        "",
        markdown_table(
            rows,
            [
                "Dataset",
                "Variant",
                "OOF RMSE",
                "OOF rho",
                "Heldout RMSE",
                "Heldout rho",
                "Heldout regret@1",
                "Selected optimism",
            ],
        ),
        "",
        "## Policy-matched 3e18 heldouts",
        "",
        (
            "The two-phase fit is intended to choose two-phase policies. Phase-tied rows remain useful off-policy "
            "stress tests, but they must not determine the headline two-phase regret."
        ),
        "",
        markdown_table(
            policy_rows,
            [
                "Dataset",
                "Variant",
                "Policy",
                "RMSE",
                "rho",
                "Regret@1",
                "Selected optimism",
                "Selected run",
            ],
        ),
        "",
        "## Promoted form",
        "",
        (
            "`hierarchical_phase_bucket_replay` keeps one retained-exposure state per bucket, pools utility through "
            "a family base plus strongly shrunk bucket excesses, adds family coverage and family-total replay harm, "
            "then adds mean member-level replay harm per family and one global phase-transition TV cost. All linear "
            "coefficients are nonnegative. Nonlinear shape, L2, and residual shrinkage are selected only by "
            "fit-panel CV."
        ),
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")


def heldout_policy_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    heldout = predictions.loc[predictions["split"].eq("heldout")]
    for (dataset, variant, policy_class), group in heldout.groupby(["dataset", "variant", "policy_class"], sort=False):
        observed = group["observed"].to_numpy(float)
        predicted = group["predicted"].to_numpy(float)
        selected_index = int(np.argmin(predicted))
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "policy_class": policy_class,
                **metric_summary(observed, predicted),
                "selected_run": str(group.iloc[selected_index]["row_id"]),
                "selected_observed": observed[selected_index],
                "selected_predicted": predicted[selected_index],
                "selected_optimism": observed[selected_index] - predicted[selected_index],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    datasets = tuple(DatasetId(value.strip()) for value in args.datasets.split(",") if value.strip())
    requested_variants = tuple(Variant(value.strip()) for value in args.variants.split(",") if value.strip())
    variants = tuple(dict.fromkeys((Variant.BUCKET_RESOLVED, *requested_variants)))
    metric_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        print(f"Benchmarking {dataset_id.value}", flush=True)
        metrics, screens, predictions = benchmark_dataset(dataset_id, variants, args.num_shapes, args.top_shapes)
        metric_rows.extend(metrics)
        screen_rows.extend(screens)
        prediction_rows.extend(predictions)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    screens = pd.DataFrame(screen_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    screens.to_csv(output_dir / "hyperparameter_screen.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    policy_metrics = heldout_policy_metrics(predictions)
    policy_metrics.to_csv(output_dir / "heldout_policy_metrics.csv", index=False)
    worst = predictions.loc[predictions["split"].eq("heldout")].copy()
    worst["optimism"] = worst["observed"] - worst["predicted"]
    worst = (
        worst.sort_values("optimism", ascending=False)
        .groupby(["dataset", "variant", "policy_class"], sort=False)
        .head(10)
    )
    worst.to_csv(output_dir / "worst_heldout_optimism.csv", index=False)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": [dataset.value for dataset in datasets],
        "variants": [variant.value for variant in variants],
        "fit_protocol": "five-fold panel-stratified CV; full fit projected onto historical 3e18 validations",
        "heldout_role": "development transfer set; never included in fit or hyperparameter selection",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if {DatasetId.DELPHI_3E18_UNCHEATABLE, DatasetId.DELPHI_3E18_TABLE9}.issubset(datasets):
        render(metrics, predictions, output_dir)
    write_report(metrics, policy_metrics, output_dir)


if __name__ == "__main__":
    main()
