# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Identify aggregate exposure and phase ordering in separate experiments.

The aggregate spine is fit only to phase-tied policies. The phase model is fit
only to fixed-aggregate policy contrasts. This prevents aggregate quality and
phase ordering from serving as interchangeable explanations for the same
observations.

The phase model simulates a bounded acquired-capability state. Its simplest
form retains phase-0 capability through the phase boundary. A nested transfer
variant lets an early web/reference foundation state increase phase-1
specialist acquisition. Every phase prediction is differenced against the
model's simulation of the tied policy at the exact same aggregate, so its
single-phase restriction is identically zero.
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
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import least_squares, nnls
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "orthogonal_aggregate_phase_identification_20260724"
HELDOUT_CURRENT = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"
FIBER_PANEL = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_20260719"
FIBER_RESULTS = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_results_20260719/observed_results.csv"
RANDOM_PANEL = REFERENCE_OUTPUTS / "delphi_3e18_frontier_random_phase_population_20260720"
AGGRESSIVE_PANEL = REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_20260722"
AGGRESSIVE_RESULTS = (
    REFERENCE_OUTPUTS
    / "delphi_3e18_aggressive_phase_asymmetry_results_20260723/observed_results_with_control_deltas.csv"
)
HYBRID_PANEL = REFERENCE_OUTPUTS / "delphi_3e18_hybrid_phase_ordering_panel_20260720"
OBSERVATORY_CACHE = REFERENCE_OUTPUTS / "mixture_fit_observatory_cache_20260713/delphi_3e18"

TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
RHO_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
POWER_GRID = (0.35, 0.5, 0.67, 1.0)
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
LOSS_KINDS = ("gaussian", "huber")
N_FOLDS = 5
CV_SEED = 20260724
HUBER_MULTIPLIER = 1.5
IRLS_STEPS = 6
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
PARETO_BASELINE_MODELS = (
    "canonical",
    "effective_exposure",
    "effective_exposure_geometry",
    "separate_heads",
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_bucket_replay",
)


class ReplayKind(StrEnum):
    NONE = "none"
    LINEAR = "linear"
    SQUARED = "squared"
    EPOCH_SQUARED = "epoch_squared"


class PhaseShiftKind(StrEnum):
    NONE = "none"
    HELLINGER = "hellinger"
    EPOCH_CONTRAST = "epoch_contrast"


class PhaseKind(StrEnum):
    NULL = "phase_null"
    GLOBAL_RETENTION = "global_retention"
    TWO_GROUP_RETENTION = "two_group_retention"
    FOUNDATION_TRANSFER = "foundation_transfer"


@dataclass(frozen=True)
class FamilyPartition:
    names: tuple[str, ...]
    members: tuple[np.ndarray, ...]
    bucket_group: np.ndarray


@dataclass(frozen=True)
class AggregateConfig:
    name: str
    include_families: bool
    replay: ReplayKind
    loss: str


@dataclass(frozen=True)
class AggregateShape:
    rho: float
    power: float


@dataclass(frozen=True)
class AggregateModel:
    config: AggregateConfig
    shape: AggregateShape
    l2: float
    intercept: float
    bucket_coef: np.ndarray
    family_coef: np.ndarray
    replay_coef: float
    c_total: np.ndarray
    phase_fraction: float
    families: FamilyPartition

    def predict(self, weights: np.ndarray) -> np.ndarray:
        aggregate = aggregate_weights(weights, phase_fraction=self.phase_fraction)
        design, bucket_width, family_width = aggregate_design(
            aggregate,
            self.c_total,
            self.families,
            self.config,
            self.shape,
        )
        coef = np.concatenate(
            [
                self.bucket_coef,
                self.family_coef,
                np.asarray([self.replay_coef]) if self.config.replay is not ReplayKind.NONE else np.asarray([]),
            ]
        )
        if bucket_width != len(self.bucket_coef) or family_width != len(self.family_coef):
            raise AssertionError("Aggregate coefficient width mismatch")
        return self.intercept + design @ coef


@dataclass(frozen=True)
class PhaseConfig:
    kind: PhaseKind
    shift: PhaseShiftKind
    huber_scale: float

    @property
    def name(self) -> str:
        shift_suffix = "" if self.shift is PhaseShiftKind.NONE else f"_{self.shift.value}"
        scale_suffix = (
            "" if self.kind is PhaseKind.NULL and self.shift is PhaseShiftKind.NONE else f"_h{self.huber_scale:g}"
        )
        return f"{self.kind.value}{shift_suffix}{scale_suffix}"


@dataclass(frozen=True)
class PhaseModel:
    config: PhaseConfig
    params: np.ndarray
    aggregate_model: AggregateModel

    def predict_delta(self, weights: np.ndarray) -> np.ndarray:
        return phase_delta(weights, self.aggregate_model, self.config, self.params)


@dataclass(frozen=True)
class PhaseRows:
    frame: pd.DataFrame
    weights: np.ndarray
    target_delta: np.ndarray
    base_weight: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--max-aggregate-candidates", type=int, default=0)
    return parser.parse_args()


def json_clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    return value


def family_partition(domain_names: list[str]) -> FamilyPartition:
    web_reference = [
        index
        for index, domain in enumerate(domain_names)
        if domain.startswith("dolma3_cc/") or domain in {"dolma3_wikipedia", "dolmino_common_crawl_hq"}
    ]
    technical = [
        domain_names.index(domain)
        for domain in (
            "dolma3_arxiv",
            "dolma3_finemath_3plus",
            "dolma3_stack_edu",
            "dolmino_olmocr_pdfs_hq",
            "dolmino_stack_edu_fim",
            "dolmino_stem_heavy_crawl",
            "dolmino_synth_code",
            "dolmino_synth_math",
            "dolmino_synth_thinking",
        )
    ]
    instruction = [
        domain_names.index("dolmino_synth_instruction"),
        domain_names.index("dolmino_synth_qa"),
    ]
    members = tuple(np.asarray(indices, dtype=int) for indices in (web_reference, technical, instruction))
    flattened = np.concatenate(members)
    if sorted(flattened.tolist()) != list(range(len(domain_names))):
        raise ValueError("The predeclared family partition must cover every bucket exactly once")
    bucket_group = np.empty(len(domain_names), dtype=int)
    for group, indices in enumerate(members):
        bucket_group[indices] = group
    return FamilyPartition(
        names=("web_reference", "technical", "instruction_qa"),
        members=members,
        bucket_group=bucket_group,
    )


def aggregate_weights(weights: np.ndarray, phase_fraction: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    return phase_fraction * weights[:, 0, :] + (1.0 - phase_fraction) * weights[:, 1, :]


def acquisition_response(exposure: np.ndarray, shape: AggregateShape) -> np.ndarray:
    exposure = np.maximum(np.asarray(exposure, dtype=float), 0.0)
    return -np.expm1(-((shape.rho * exposure) ** shape.power))


def family_epochs(
    aggregate: np.ndarray,
    c_total: np.ndarray,
    families: FamilyPartition,
) -> np.ndarray:
    columns = []
    for members in families.members:
        family_mass = aggregate[:, members].sum(axis=1)
        family_token_fraction = np.sum(1.0 / c_total[members])
        columns.append(family_mass / family_token_fraction)
    return np.column_stack(columns)


def replay_feature(aggregate: np.ndarray, c_total: np.ndarray, replay: ReplayKind) -> np.ndarray:
    repeated_mass = np.maximum(aggregate - 1.0 / c_total[None, :], 0.0)
    if replay is ReplayKind.LINEAR:
        return repeated_mass.sum(axis=1, keepdims=True)
    if replay is ReplayKind.SQUARED:
        return (repeated_mass**2).sum(axis=1, keepdims=True)
    if replay is ReplayKind.EPOCH_SQUARED:
        repeated_epochs = np.maximum(aggregate * c_total[None, :] - 1.0, 0.0)
        return (repeated_epochs**2).sum(axis=1, keepdims=True)
    return np.empty((len(aggregate), 0), dtype=float)


def aggregate_design(
    aggregate: np.ndarray,
    c_total: np.ndarray,
    families: FamilyPartition,
    config: AggregateConfig,
    shape: AggregateShape,
) -> tuple[np.ndarray, int, int]:
    bucket_signal = acquisition_response(aggregate * c_total[None, :], shape)
    pieces = [-bucket_signal]
    family_width = 0
    if config.include_families:
        family_signal = acquisition_response(family_epochs(aggregate, c_total, families), shape)
        pieces.append(-family_signal)
        family_width = family_signal.shape[1]
    replay = replay_feature(aggregate, c_total, config.replay)
    if replay.shape[1]:
        pieces.append(replay)
    return np.hstack(pieces), bucket_signal.shape[1], family_width


def weighted_nonnegative_head(
    design: np.ndarray,
    target: np.ndarray,
    l2: float,
    base_weight: np.ndarray,
    loss: str,
) -> tuple[float, np.ndarray]:
    robust_weight = np.ones(len(target), dtype=float)
    intercept = float(np.average(target, weights=base_weight))
    coef = np.zeros(design.shape[1], dtype=float)
    iterations = 1 if loss == "gaussian" else IRLS_STEPS
    for _ in range(iterations):
        weight = base_weight * robust_weight
        weight /= np.mean(weight)
        feature_mean = np.average(design, axis=0, weights=weight)
        target_mean = float(np.average(target, weights=weight))
        centered_design = design - feature_mean[None, :]
        centered_target = target - target_mean
        feature_scale = np.sqrt(np.average(centered_design**2, axis=0, weights=weight))
        feature_scale = np.maximum(feature_scale, 1e-8)
        scaled_design = centered_design / feature_scale[None, :]
        sqrt_weight = np.sqrt(weight)
        lhs = scaled_design * sqrt_weight[:, None]
        rhs = centered_target * sqrt_weight
        if l2 > 0.0:
            lhs = np.vstack([lhs, np.sqrt(l2) * np.eye(design.shape[1])])
            rhs = np.concatenate([rhs, np.zeros(design.shape[1])])
        scaled_coef, _ = nnls(lhs, rhs, maxiter=100 * max(design.shape[1], 1))
        coef = scaled_coef / feature_scale
        intercept = target_mean - float(feature_mean @ coef)
        if loss == "gaussian":
            break
        residual = intercept + design @ coef - target
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median)))
        scale = max(scale, 1e-6)
        threshold = HUBER_MULTIPLIER * scale
        robust_weight = np.minimum(1.0, threshold / np.maximum(np.abs(residual - median), 1e-12))
    return intercept, coef


def fit_aggregate(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: AggregateConfig,
    shape: AggregateShape,
    l2: float,
    families: FamilyPartition,
) -> AggregateModel:
    c_total = np.asarray(dataset.c0 + dataset.c1, dtype=float)
    phase_fraction_by_bucket = dataset.c0 / c_total
    if np.ptp(phase_fraction_by_bucket) > 1e-12:
        raise ValueError("Aggregate model requires one shared phase fraction")
    phase_fraction = float(np.mean(phase_fraction_by_bucket))
    aggregate = aggregate_weights(
        dataset.weights[indices],
        phase_fraction=phase_fraction,
    )
    design, bucket_width, family_width = aggregate_design(aggregate, c_total, families, config, shape)
    intercept, coef = weighted_nonnegative_head(
        design,
        dataset.y[indices],
        l2,
        np.ones(len(indices), dtype=float),
        config.loss,
    )
    cursor = bucket_width
    bucket_coef = coef[:cursor]
    family_coef = coef[cursor : cursor + family_width]
    cursor += family_width
    replay_coef = float(coef[cursor]) if config.replay is not ReplayKind.NONE else 0.0
    return AggregateModel(
        config=config,
        shape=shape,
        l2=l2,
        intercept=intercept,
        bucket_coef=bucket_coef,
        family_coef=family_coef,
        replay_coef=replay_coef,
        c_total=c_total,
        phase_fraction=phase_fraction,
        families=families,
    )


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.std(observed) < 1e-12 or np.std(predicted) < 1e-12:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[: min(k, len(predicted))]
    return float(np.min(observed[selected]) - np.min(observed))


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = predicted - observed
    optimism = observed - predicted
    centered = predicted - np.mean(predicted)
    denominator = float(centered @ centered)
    slope = float(centered @ (observed - np.mean(observed)) / denominator) if denominator > 1e-15 else float("nan")
    tail_count = min(len(observed), max(10, math.ceil(0.1 * len(observed))))
    tail = np.argsort(predicted)[:tail_count]
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": safe_spearman(observed, predicted),
        "bias": float(np.mean(residual)),
        "calibration_slope": slope,
        "regret_at_1": regret_at_k(observed, predicted, 1),
        "regret_at_3": regret_at_k(observed, predicted, 3),
        "regret_at_5": regret_at_k(observed, predicted, 5),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        "lower_tail_optimism": float(np.mean(np.maximum(optimism[tail], 0.0))),
        "optimism_gt_0p05": int(np.sum(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
    }


def aggregate_configs() -> tuple[AggregateConfig, ...]:
    return tuple(
        AggregateConfig(
            name=f"{'family' if include_families else 'bucket'}_{replay.value}_{loss}",
            include_families=include_families,
            replay=replay,
            loss=loss,
        )
        for include_families in (False, True)
        for replay in ReplayKind
        for loss in LOSS_KINDS
    )


def aggregate_oof_sweep(
    dataset: pooled.Dataset,
    families: FamilyPartition,
    max_candidates: int,
    evaluation_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    evaluation_indices = np.flatnonzero(evaluation_mask)
    if len(evaluation_indices) != 238:
        raise ValueError(f"Expected 238 non-alias aggregate evaluation rows, found {len(evaluation_indices)}")
    local_folds = component_dsp.panel_stratified_folds(
        dataset.frame.iloc[evaluation_indices].reset_index(drop=True),
        n_splits=N_FOLDS,
        seed=CV_SEED,
    )
    all_indices = np.arange(dataset.n)
    folds = [
        (
            np.setdiff1d(all_indices, evaluation_indices[local_test], assume_unique=True),
            evaluation_indices[local_test],
        )
        for _local_train, local_test in local_folds
    ]
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    candidates = [
        (config, AggregateShape(rho, power), l2)
        for config in aggregate_configs()
        for rho in RHO_GRID
        for power in POWER_GRID
        for l2 in L2_GRID
    ]
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    for candidate_index, (config, shape, l2) in enumerate(candidates):
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds:
            model = fit_aggregate(dataset, train, config, shape, l2, families)
            prediction[test] = model.predict(dataset.weights[test])
        if not np.isfinite(prediction[evaluation_mask]).all():
            raise ValueError(f"Non-finite OOF prediction for {config.name}")
        key = f"{config.name}_rho{shape.rho:g}_p{shape.power:g}_l2{l2:g}"
        metrics = regression_metrics(dataset.y[evaluation_mask], prediction[evaluation_mask])
        rows.append(
            {
                "candidate_index": candidate_index,
                "key": key,
                **asdict(config),
                **asdict(shape),
                "l2": l2,
                "parameter_count": (
                    dataset.m
                    + (len(families.names) if config.include_families else 0)
                    + int(config.replay is not ReplayKind.NONE)
                    + 3
                ),
                "fit_rows": dataset.n,
                "evaluation_rows": len(evaluation_indices),
                **metrics,
            }
        )
        predictions[key] = prediction
    frame = pd.DataFrame(rows).sort_values(
        ["rmse", "regret_at_1", "parameter_count", "key"],
        ignore_index=True,
    )
    return frame, predictions


def coordinate_key(weights: np.ndarray) -> bytes:
    return np.round(np.asarray(weights, dtype=np.float64), 12).tobytes()


def tied_heldout_dataset(
    target: str,
    reference: pooled.Dataset,
    single: pooled.Dataset,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
) -> pooled.Dataset:
    training_keys = {coordinate_key(weights) for weights in single.weights}
    tied = heldout_frame["policy_class"].eq("single_phase_tied").to_numpy()
    disjoint = heldout_frame["fit_panel_overlap"].eq("coordinate_disjoint").to_numpy()
    complete = np.isfinite(heldout_frame[TARGET_COLUMNS[target]].to_numpy(dtype=float))
    candidates = np.flatnonzero(tied & disjoint & complete)
    candidates = np.asarray(
        [index for index in candidates if coordinate_key(heldout_weights[index]) not in training_keys],
        dtype=int,
    )
    frame = heldout_frame.iloc[candidates].copy().reset_index(drop=True)
    weights = heldout_weights[candidates]
    frame["_coordinate"] = [coordinate_key(weight).hex() for weight in weights]
    target_column = TARGET_COLUMNS[target]
    coordinate_rows = []
    coordinate_weights = []
    for _coordinate, indices in frame.groupby("_coordinate", sort=True).indices.items():
        group = frame.iloc[np.asarray(indices, dtype=int)]
        record = group.iloc[0].to_dict()
        record[target_column] = float(group[target_column].mean())
        record["repeat_count"] = len(group)
        coordinate_rows.append(record)
        coordinate_weights.append(weights[np.asarray(indices, dtype=int)[0]])
    result_frame = pd.DataFrame(coordinate_rows)
    result_weights = np.asarray(coordinate_weights, dtype=float)
    return pooled.Dataset(
        name=f"delphi_3e18_tied_heldout_{target}",
        frame=result_frame,
        y=result_frame[target_column].to_numpy(dtype=float),
        weights=result_weights,
        c0=np.asarray(reference.c0, dtype=float),
        c1=np.asarray(reference.c1, dtype=float),
        domain_names=list(reference.domain_names),
    )


def observatory_baseline_metrics(
    target: str,
    reference: pooled.Dataset,
    single: pooled.Dataset,
    single_evaluation_indices: np.ndarray,
    single_evaluation_mask: np.ndarray,
    heldout_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evaluation_length = reference.n + len(heldout_frame)
    heldout_positions = reference.n + np.arange(len(heldout_frame))
    coordinate_disjoint = heldout_frame["fit_panel_overlap"].eq("coordinate_disjoint").to_numpy()
    for policy_class, fit_dataset, fit_indices, heldout_policy in (
        (
            "two_phase",
            reference,
            np.arange(reference.n),
            heldout_frame["policy_class"].eq("two_phase").to_numpy(),
        ),
        (
            "single_phase",
            single,
            single_evaluation_indices[single_evaluation_mask],
            heldout_frame["policy_class"].eq("single_phase_tied").to_numpy(),
        ),
    ):
        fit_index_set = set(np.asarray(fit_indices, dtype=int).tolist())
        evaluation_mask = np.asarray(
            [
                bool(disjoint and policy_match and position not in fit_index_set)
                for position, disjoint, policy_match in zip(
                    heldout_positions,
                    coordinate_disjoint,
                    heldout_policy,
                    strict=True,
                )
            ],
            dtype=bool,
        )
        complete = np.isfinite(heldout_frame[TARGET_COLUMNS[target]].to_numpy(dtype=float))
        evaluation_mask &= complete
        observed_heldout = heldout_frame.loc[evaluation_mask, TARGET_COLUMNS[target]].to_numpy(dtype=float)
        for model_id in PARETO_BASELINE_MODELS:
            cache_path = OBSERVATORY_CACHE / target / policy_class / f"{model_id}.json"
            if not cache_path.exists():
                continue
            payload = json.loads(cache_path.read_text())
            prediction = np.asarray(payload["prediction"], dtype=float)
            if len(prediction) != evaluation_length:
                raise ValueError(
                    f"Stale Observatory cache {cache_path}: expected {evaluation_length} predictions, "
                    f"found {len(prediction)}"
                )
            fit_prediction = prediction[np.asarray(fit_indices, dtype=int)]
            heldout_prediction = prediction[heldout_positions[evaluation_mask]]
            fit_observed = fit_dataset.y[single_evaluation_mask] if policy_class == "single_phase" else fit_dataset.y
            rows.append(
                {
                    "target": target,
                    "policy_class": policy_class,
                    "model": model_id,
                    "split": "fit_oof",
                    **regression_metrics(fit_observed, fit_prediction),
                }
            )
            rows.append(
                {
                    "target": target,
                    "policy_class": policy_class,
                    "model": model_id,
                    "split": "policy_matched_heldout",
                    **regression_metrics(observed_heldout, heldout_prediction),
                }
            )
    return pd.DataFrame(rows)


def policy_matched_heldout_mask(
    target: str,
    heldout_frame: pd.DataFrame,
    policy_class: str,
) -> np.ndarray:
    expected_policy = {
        "single_phase": "single_phase_tied",
        "two_phase": "two_phase",
    }[policy_class]
    return (
        heldout_frame["fit_panel_overlap"].eq("coordinate_disjoint").to_numpy()
        & heldout_frame["policy_class"].eq(expected_policy).to_numpy()
        & np.isfinite(heldout_frame[TARGET_COLUMNS[target]].to_numpy(dtype=float))
    )


def weights_from_long(path: Path, candidate_ids: list[str], domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    lookup = frame.set_index(["candidate_id", "phase", "domain"])["weight"]
    return np.asarray(
        [
            [[lookup.loc[(candidate_id, phase, domain)] for domain in domains] for phase in (0, 1)]
            for candidate_id in candidate_ids
        ],
        dtype=float,
    )


def weights_from_wide(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    return np.stack(
        [
            frame[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float),
            frame[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float),
        ],
        axis=1,
    )


def balanced_panel_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["panel"].value_counts()
    weight = frame["panel"].map({panel: 1.0 / count for panel, count in counts.items()}).to_numpy(dtype=float)
    return weight / np.mean(weight)


def load_fiber_phase_rows(target: str, domains: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    results = pd.read_csv(FIBER_RESULTS)
    if float(results["aggregate_max_abs_error"].max()) > 1e-12:
        raise ValueError("Frontier-fiber policies are not fixed-aggregate")
    results = results[~results["contrast_family"].eq("center_control")].copy()
    delta_column = f"{target}_delta_vs_same_seed_center"
    frame = results[
        [
            "candidate_id",
            "anchor_id",
            "contrast_family",
            "direction_id",
            "sign",
            "seed_block",
            "phase_tv",
            delta_column,
        ]
    ].rename(columns={delta_column: "target_delta"})
    frame["panel"] = "frontier_fiber"
    frame["anchor_key"] = "fiber:" + frame["anchor_id"].astype(str)
    weights = weights_from_long(FIBER_PANEL / "phase_weights.csv", frame["candidate_id"].tolist(), domains)
    return frame, weights


def load_aggressive_phase_rows(target: str, domains: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    results = pd.read_csv(AGGRESSIVE_RESULTS)
    if float(results["aggregate_max_abs_error"].max()) > 1e-12:
        raise ValueError("Aggressive-asymmetry policies are not fixed-aggregate")
    results = results[~results["contrast_family"].eq("center_control")].copy()
    delta_column = f"{target}_delta_vs_control"
    frame = results[
        [
            "candidate_id",
            "anchor_id",
            "contrast_family",
            "direction_id",
            "sign",
            "seed_block",
            "phase_tv",
            delta_column,
            *[f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains],
        ]
    ].rename(columns={delta_column: "target_delta"})
    frame["panel"] = "aggressive_asymmetry"
    frame["anchor_key"] = "aggressive:" + frame["anchor_id"].astype(str)
    weights = weights_from_wide(frame, domains)
    return frame.drop(columns=[f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains]), weights


def load_registry_panel(
    panel_dir: Path,
    training_series: str,
    target: str,
    domains: list[str],
    panel_name: str,
    phase_fraction: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(panel_dir / "candidate_manifest.csv")
    heldout = pd.read_csv(HELDOUT_CURRENT)
    heldout = heldout[heldout["training_series"].eq(training_series)].copy()
    columns = ["candidate_id", TARGET_COLUMNS[target]]
    merged = manifest.merge(heldout[columns], on="candidate_id", how="left", validate="one_to_one")
    if merged[TARGET_COLUMNS[target]].isna().any():
        raise ValueError(f"Incomplete {target} metrics in {panel_name}")
    if panel_name == "random_population":
        if float(merged["aggregate_max_abs_error"].max()) > 1e-12:
            raise ValueError("Random-population policies are not fixed-aggregate")
        controls = merged[merged["contrast_family"].eq("center_control")].copy()
        control_keys = ["anchor_id", "seed_block"]
        if controls.duplicated(control_keys).any():
            raise ValueError("Random-population control keys are not unique")
        control = controls.set_index(control_keys)[TARGET_COLUMNS[target]].to_dict()
        treatment = merged[~merged["contrast_family"].eq("center_control")].copy()
        treatment["target_delta"] = [
            value - control[(anchor, seed)]
            for value, anchor, seed in zip(
                treatment[TARGET_COLUMNS[target]],
                treatment["anchor_id"],
                treatment["seed_block"],
                strict=True,
            )
        ]
        treatment["anchor_key"] = "random:" + treatment["anchor_id"].astype(str)
        frame = treatment[
            [
                "candidate_id",
                "anchor_id",
                "contrast_family",
                "direction_id",
                "sign",
                "seed_block",
                "phase_tv",
                "target_delta",
                "anchor_key",
            ]
        ].copy()
        weights = weights_from_long(panel_dir / "phase_weights.csv", frame["candidate_id"].tolist(), domains)
    elif panel_name == "hybrid_ordering":
        fixed = merged[
            merged["candidate_kind"].astype(str).str.startswith("fixed_aggregate_")
            | merged["candidate_kind"].eq("tied_separate_heads_anchor")
        ].copy()
        controls = fixed[fixed["candidate_kind"].eq("tied_separate_heads_anchor")].copy()
        control_keys = ["target", "aggregate_kl_coefficient"]
        if controls.duplicated(control_keys).any():
            raise ValueError("Hybrid-ordering control keys are not unique")
        control = controls.set_index(control_keys)[TARGET_COLUMNS[target]].to_dict()
        treatment = fixed[fixed["policy_class"].eq("two_phase")].copy()
        treatment["target_delta"] = [
            value - control[(proposal_target, aggregate_kl)]
            for value, proposal_target, aggregate_kl in zip(
                treatment[TARGET_COLUMNS[target]],
                treatment["target"],
                treatment["aggregate_kl_coefficient"],
                strict=True,
            )
        ]
        treatment["anchor_id"] = (
            treatment["target"].astype(str) + ":" + treatment["aggregate_kl_coefficient"].map(lambda value: f"{value:g}")
        )
        treatment["anchor_key"] = "hybrid:" + treatment["anchor_id"]
        treatment["contrast_family"] = treatment["model"].astype(str)
        treatment["direction_id"] = treatment["candidate_id"].astype(str)
        treatment["sign"] = "proposed"
        treatment["seed_block"] = 0
        treatment["phase_tv"] = treatment["phase_total_variation"]
        frame = treatment[
            [
                "candidate_id",
                "anchor_id",
                "contrast_family",
                "direction_id",
                "sign",
                "seed_block",
                "phase_tv",
                "target_delta",
                "anchor_key",
            ]
        ].copy()
        weights = weights_from_wide(treatment, domains)
        control_weights = weights_from_wide(controls, domains)
        control_aggregate = {
            key: aggregate_weights(control_weights[[index]], phase_fraction)[0]
            for index, key in enumerate(controls[control_keys].itertuples(index=False, name=None))
        }
        treatment_aggregate = aggregate_weights(weights, phase_fraction)
        aggregate_error = np.asarray(
            [
                np.max(np.abs(value - control_aggregate[(proposal_target, aggregate_kl)]))
                for value, proposal_target, aggregate_kl in zip(
                    treatment_aggregate,
                    treatment["target"],
                    treatment["aggregate_kl_coefficient"],
                    strict=True,
                )
            ],
            dtype=float,
        )
        if float(np.max(aggregate_error)) > 1e-10:
            raise ValueError(
                "Hybrid-ordering treatments do not match their tied aggregate: "
                f"max error {float(np.max(aggregate_error)):.3e}"
            )
    else:
        raise ValueError(f"Unknown registry panel {panel_name}")
    frame["panel"] = panel_name
    return frame, weights


def load_phase_rows(
    target: str,
    domains: list[str],
    phase_fraction: float,
) -> PhaseRows:
    panels = [
        load_fiber_phase_rows(target, domains),
        load_registry_panel(
            RANDOM_PANEL,
            "delphi_3e18_frontier_random_phase_population_20260720",
            target,
            domains,
            "random_population",
            phase_fraction,
        ),
        load_aggressive_phase_rows(target, domains),
        load_registry_panel(
            HYBRID_PANEL,
            "delphi_3e18_hybrid_phase_ordering_validation_20260720",
            target,
            domains,
            "hybrid_ordering",
            phase_fraction,
        ),
    ]
    frame = pd.concat([panel[0] for panel in panels], ignore_index=True, sort=False)
    weights = np.concatenate([panel[1] for panel in panels], axis=0)
    target_delta = frame["target_delta"].to_numpy(dtype=float)
    if len(frame) != len(weights) or not np.isfinite(target_delta).all():
        raise ValueError("Fixed-aggregate phase rows are incomplete")
    aggregate = aggregate_weights(weights, phase_fraction)
    if not np.allclose(aggregate.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Fixed-aggregate phase rows contain unnormalized policies")
    frame["source_anchor_key"] = frame["anchor_key"].astype(str)
    frame["anchor_key"] = [coordinate_key(value).hex() for value in aggregate]
    anchor_counts = frame.groupby("source_anchor_key", sort=False)["anchor_key"].nunique()
    if int(anchor_counts.max()) != 1:
        raise ValueError("A source anchor maps to more than one aggregate coordinate")
    return PhaseRows(
        frame=frame,
        weights=weights,
        target_delta=target_delta,
        base_weight=balanced_panel_weights(frame),
    )


def phase_parameter_bounds(config: PhaseConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.kind is PhaseKind.NULL:
        lower = np.asarray([], dtype=float)
        upper = np.asarray([], dtype=float)
    elif config.kind is PhaseKind.GLOBAL_RETENTION:
        lower = np.asarray([0.0], dtype=float)
        upper = np.asarray([1.0], dtype=float)
    elif config.kind is PhaseKind.TWO_GROUP_RETENTION:
        lower = np.asarray([0.0, 0.0], dtype=float)
        upper = np.asarray([1.0, 1.0], dtype=float)
    elif config.kind is PhaseKind.FOUNDATION_TRANSFER:
        lower = np.asarray([0.0, 0.0, 0.0], dtype=float)
        upper = np.asarray([1.0, 1.0, 10.0], dtype=float)
    else:
        raise ValueError(f"Unknown phase kind {config.kind}")
    if config.shift is PhaseShiftKind.HELLINGER:
        lower = np.append(lower, 0.0)
        upper = np.append(upper, 0.1)
    elif config.shift is PhaseShiftKind.EPOCH_CONTRAST:
        lower = np.append(lower, 0.0)
        upper = np.append(upper, 1.0)
    return lower, upper


def decode_phase_params(config: PhaseConfig, params: np.ndarray) -> tuple[np.ndarray, float, float]:
    cursor = 0
    if config.kind is PhaseKind.NULL:
        retention = np.ones(3, dtype=float)
        transfer = 0.0
    elif config.kind is PhaseKind.GLOBAL_RETENTION:
        retention = np.full(3, params[cursor], dtype=float)
        cursor += 1
        transfer = 0.0
    elif config.kind is PhaseKind.TWO_GROUP_RETENTION:
        retention = np.asarray([params[cursor], params[cursor + 1], params[cursor + 1]], dtype=float)
        cursor += 2
        transfer = 0.0
    elif config.kind is PhaseKind.FOUNDATION_TRANSFER:
        retention = np.asarray([params[cursor], params[cursor + 1], params[cursor + 1]], dtype=float)
        transfer = float(params[cursor + 2])
        cursor += 3
    else:
        raise ValueError(f"Unknown phase kind {config.kind}")
    shift = float(params[cursor]) if config.shift is not PhaseShiftKind.NONE else 0.0
    return retention, transfer, shift


def phase_shift_feature(
    weights: np.ndarray,
    model: AggregateModel,
    shift: PhaseShiftKind,
) -> np.ndarray:
    if shift is PhaseShiftKind.NONE:
        return np.zeros(len(weights), dtype=float)
    if shift is PhaseShiftKind.HELLINGER:
        return np.sum(
            (np.sqrt(np.maximum(weights[:, 0, :], 0.0)) - np.sqrt(np.maximum(weights[:, 1, :], 0.0))) ** 2,
            axis=1,
        )
    if shift is PhaseShiftKind.EPOCH_CONTRAST:
        aggregate = aggregate_weights(weights, model.phase_fraction)
        contrast = model.phase_fraction * (weights[:, 0, :] - aggregate)
        shifted_epochs = contrast * model.c_total[None, :]
        return np.mean(shifted_epochs**2, axis=1)
    raise ValueError(f"Unknown phase shift kind {shift}")


def phase_state(
    q0: np.ndarray,
    q1: np.ndarray,
    group: np.ndarray,
    foundation_state: np.ndarray,
    shape: AggregateShape,
    retention: np.ndarray,
    transfer: float,
) -> np.ndarray:
    h0 = (shape.rho * np.maximum(q0, 0.0)) ** shape.power
    h_total = (shape.rho * np.maximum(q0 + q1, 0.0)) ** shape.power
    late_hazard = np.maximum(h_total - h0, 0.0)
    specialist = (group[None, :] != 0).astype(float)
    late_hazard *= 1.0 + transfer * foundation_state[:, None] * specialist
    state0 = -np.expm1(-h0)
    retained = retention[group][None, :] * state0
    return retained + (1.0 - retained) * (-np.expm1(-late_hazard))


def simulated_phase_loss(
    weights: np.ndarray,
    model: AggregateModel,
    config: PhaseConfig,
    params: np.ndarray,
) -> np.ndarray:
    alpha0 = model.phase_fraction
    alpha1 = 1.0 - alpha0
    q0 = weights[:, 0, :] * (alpha0 * model.c_total)[None, :]
    q1 = weights[:, 1, :] * (alpha1 * model.c_total)[None, :]
    family_q0 = family_epochs(alpha0 * weights[:, 0, :], model.c_total, model.families)
    family_q1 = family_epochs(alpha1 * weights[:, 1, :], model.c_total, model.families)
    retention, transfer, shift = decode_phase_params(config, params)
    foundation_state = acquisition_response(family_q0[:, [0]], model.shape)[:, 0]
    bucket_state = phase_state(
        q0,
        q1,
        model.families.bucket_group,
        foundation_state,
        model.shape,
        retention,
        transfer,
    )
    family_state = phase_state(
        family_q0,
        family_q1,
        np.arange(len(model.families.names), dtype=int),
        foundation_state,
        model.shape,
        retention,
        transfer,
    )
    value = -bucket_state @ model.bucket_coef
    if len(model.family_coef):
        value = value - family_state @ model.family_coef
    return value + shift * phase_shift_feature(weights, model, config.shift)


def phase_delta(
    weights: np.ndarray,
    model: AggregateModel,
    config: PhaseConfig,
    params: np.ndarray,
) -> np.ndarray:
    if config.kind is PhaseKind.NULL and config.shift is PhaseShiftKind.NONE:
        return np.zeros(len(weights), dtype=float)
    alpha0 = model.phase_fraction
    aggregate = aggregate_weights(weights, alpha0)
    tied = np.stack([aggregate, aggregate], axis=1)
    return simulated_phase_loss(weights, model, config, params) - simulated_phase_loss(tied, model, config, params)


def phase_configs(target: str) -> tuple[PhaseConfig, ...]:
    huber_scales = {
        "uncheatable": (0.001, 0.002, 0.004),
        "table9": (0.002, 0.004, 0.008),
    }[target]
    configs = [PhaseConfig(PhaseKind.NULL, PhaseShiftKind.NONE, huber_scales[0])]
    for huber_scale in huber_scales:
        configs.extend(
            PhaseConfig(kind, shift, huber_scale)
            for kind in (PhaseKind.NULL, PhaseKind.GLOBAL_RETENTION, PhaseKind.TWO_GROUP_RETENTION)
            for shift in PhaseShiftKind
            if not (kind is PhaseKind.NULL and shift is PhaseShiftKind.NONE)
        )
    return tuple(configs)


def phase_starts(config: PhaseConfig) -> tuple[np.ndarray, ...]:
    lower, upper = phase_parameter_bounds(config)
    if len(lower) == 0:
        return (np.asarray([], dtype=float),)
    if config.kind is not PhaseKind.FOUNDATION_TRANSFER:
        return (0.5 * (lower + upper),)
    rng = np.random.default_rng(CV_SEED)
    starts = [0.5 * (lower + upper), lower.copy()]
    null_like = lower.copy()
    retention_width = {
        PhaseKind.NULL: 0,
        PhaseKind.GLOBAL_RETENTION: 1,
        PhaseKind.TWO_GROUP_RETENTION: 2,
        PhaseKind.FOUNDATION_TRANSFER: 2,
    }[config.kind]
    null_like[:retention_width] = 1.0
    starts.append(np.minimum(np.maximum(null_like, lower), upper))
    starts.extend(rng.uniform(lower, upper) for _ in range(6))
    return tuple(np.asarray(start, dtype=float) for start in starts)


def fit_phase(
    rows: PhaseRows,
    indices: np.ndarray,
    aggregate_model: AggregateModel,
    config: PhaseConfig,
) -> PhaseModel:
    lower, upper = phase_parameter_bounds(config)
    if len(lower) == 0:
        return PhaseModel(config, np.asarray([], dtype=float), aggregate_model)

    def residual(params: np.ndarray) -> np.ndarray:
        prediction = phase_delta(rows.weights[indices], aggregate_model, config, params)
        return np.sqrt(rows.base_weight[indices]) * (prediction - rows.target_delta[indices])

    best_params = phase_starts(config)[0]
    best_cost = float("inf")
    for start in phase_starts(config):
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            loss="huber",
            f_scale=config.huber_scale,
            x_scale="jac",
            max_nfev=1000,
        )
        if np.isfinite(result.cost) and float(result.cost) < best_cost:
            best_cost = float(result.cost)
            best_params = np.asarray(result.x, dtype=float)
    return PhaseModel(config, best_params, aggregate_model)


def phase_group_regret(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    k: int,
) -> float:
    regrets = []
    for _anchor, indices in frame.groupby("anchor_key", sort=True).indices.items():
        group = np.asarray(indices, dtype=int)
        if len(group) < 2:
            continue
        selected = group[np.argsort(predicted[group])[: min(k, len(group))]]
        regrets.append(float(np.min(observed[selected]) - np.min(observed[group])))
    return float(np.mean(regrets)) if regrets else float("nan")


def phase_metrics(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    result = regression_metrics(observed, predicted)
    result["regret_at_1"] = phase_group_regret(frame, observed, predicted, 1)
    result["regret_at_3"] = phase_group_regret(frame, observed, predicted, 3)
    result["regret_at_5"] = phase_group_regret(frame, observed, predicted, 5)
    return result


def phase_panel_metrics(
    rows: PhaseRows,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes = [("all", np.arange(len(rows.frame)))]
    scopes.extend(
        (str(panel), np.asarray(indices, dtype=int))
        for panel, indices in rows.frame.groupby("panel", sort=True).indices.items()
    )
    for model, prediction in predictions.items():
        for scope, indices in scopes:
            records.append(
                {
                    "model": model,
                    "scope": scope,
                    **phase_metrics(
                        rows.frame.iloc[indices].reset_index(drop=True),
                        rows.target_delta[indices],
                        prediction[indices],
                    ),
                }
            )
    return pd.DataFrame(records).sort_values(["scope", "rmse", "model"], ignore_index=True)


def antithetic_pair_rows(
    rows: PhaseRows,
    prediction: np.ndarray,
) -> pd.DataFrame:
    frame = rows.frame.copy()
    frame["row_index"] = np.arange(len(frame))
    frame["phase_tv_key"] = frame["phase_tv"].round(12)
    paired = frame[frame["sign"].isin(("plus", "minus"))].copy()
    key_columns = ["panel", "anchor_key", "direction_id", "seed_block", "phase_tv_key"]
    records: list[dict[str, Any]] = []
    for key, group in paired.groupby(key_columns, sort=True, dropna=False):
        signs = set(group["sign"])
        if signs != {"plus", "minus"} or len(group) != 2:
            continue
        plus_index = int(group.loc[group["sign"].eq("plus"), "row_index"].iloc[0])
        minus_index = int(group.loc[group["sign"].eq("minus"), "row_index"].iloc[0])
        observed_plus = float(rows.target_delta[plus_index])
        observed_minus = float(rows.target_delta[minus_index])
        predicted_plus = float(prediction[plus_index])
        predicted_minus = float(prediction[minus_index])
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "plus_candidate_id": rows.frame.iloc[plus_index]["candidate_id"],
                "minus_candidate_id": rows.frame.iloc[minus_index]["candidate_id"],
                "observed_plus": observed_plus,
                "observed_minus": observed_minus,
                "predicted_plus": predicted_plus,
                "predicted_minus": predicted_minus,
                "observed_odd": 0.5 * (observed_plus - observed_minus),
                "predicted_odd": 0.5 * (predicted_plus - predicted_minus),
                "observed_even": 0.5 * (observed_plus + observed_minus),
                "predicted_even": 0.5 * (predicted_plus + predicted_minus),
            }
        )
    result = pd.DataFrame(records)
    expected_pairs = 192
    if len(result) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} controlled antithetic pairs, found {len(result)}")
    return result


def antithetic_component_metrics(
    rows: PhaseRows,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    for model, prediction in predictions.items():
        pairs = antithetic_pair_rows(rows, prediction)
        pairs.insert(0, "model", model)
        pair_frames.append(pairs)
        scopes = [("all", np.arange(len(pairs)))]
        scopes.extend(
            (str(panel), np.asarray(indices, dtype=int))
            for panel, indices in pairs.groupby("panel", sort=True).indices.items()
        )
        for scope, indices in scopes:
            for component in ("odd", "even"):
                observed = pairs[f"observed_{component}"].to_numpy(dtype=float)[indices]
                predicted = pairs[f"predicted_{component}"].to_numpy(dtype=float)[indices]
                metrics = regression_metrics(observed, predicted)
                nonzero = np.abs(observed) > 1e-12
                sign_accuracy = (
                    float(np.mean(np.sign(observed[nonzero]) == np.sign(predicted[nonzero])))
                    if np.any(nonzero)
                    else float("nan")
                )
                zero_rmse = float(np.sqrt(np.mean(np.square(observed))))
                metric_records.append(
                    {
                        "model": model,
                        "scope": scope,
                        "component": component,
                        "observed_sd": float(np.std(observed, ddof=1)),
                        "predicted_sd": float(np.std(predicted, ddof=1)),
                        "zero_baseline_rmse": zero_rmse,
                        "rmse_ratio_to_zero": float(metrics["rmse"]) / zero_rmse if zero_rmse > 0 else float("nan"),
                        "sign_accuracy": sign_accuracy,
                        **metrics,
                    }
                )
    return (
        pd.DataFrame(metric_records).sort_values(["scope", "component", "rmse", "model"], ignore_index=True),
        pd.concat(pair_frames, ignore_index=True),
    )


def phase_mean_panel_rmse(
    rows: PhaseRows,
    prediction: np.ndarray,
) -> float:
    values = []
    for _panel, indices in rows.frame.groupby("panel", sort=True).indices.items():
        group = np.asarray(indices, dtype=int)
        residual = prediction[group] - rows.target_delta[group]
        values.append(float(np.sqrt(np.mean(residual**2))))
    return float(np.mean(values))


def phase_cross_validation(
    rows: PhaseRows,
    aggregate_model: AggregateModel,
    target: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    panels = sorted(rows.frame["panel"].unique())
    anchors = sorted(rows.frame["anchor_key"].unique())
    summaries: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for config in phase_configs(target):
        panel_prediction = np.full(len(rows.frame), np.nan, dtype=float)
        panel_fold_params = []
        for panel in panels:
            test = np.flatnonzero(rows.frame["panel"].eq(panel).to_numpy())
            train = np.flatnonzero(~rows.frame["panel"].eq(panel).to_numpy())
            model = fit_phase(rows, train, aggregate_model, config)
            panel_prediction[test] = model.predict_delta(rows.weights[test])
            panel_fold_params.append({"panel": panel, "params": model.params.tolist()})
        anchor_prediction = np.full(len(rows.frame), np.nan, dtype=float)
        anchor_fold_params = []
        for anchor in anchors:
            test = np.flatnonzero(rows.frame["anchor_key"].eq(anchor).to_numpy())
            train = np.flatnonzero(~rows.frame["anchor_key"].eq(anchor).to_numpy())
            model = fit_phase(rows, train, aggregate_model, config)
            anchor_prediction[test] = model.predict_delta(rows.weights[test])
            anchor_fold_params.append({"anchor": anchor, "params": model.params.tolist()})
        if not np.isfinite(panel_prediction).all() or not np.isfinite(anchor_prediction).all():
            raise ValueError(f"Non-finite phase prediction for {config.name}")
        panel_metrics = phase_metrics(rows.frame, rows.target_delta, panel_prediction)
        anchor_metrics = phase_metrics(rows.frame, rows.target_delta, anchor_prediction)
        summaries.append(
            {
                "model": config.name,
                "parameter_count": len(phase_parameter_bounds(config)[0]),
                "huber_scale": config.huber_scale,
                "panel_mean_rmse": phase_mean_panel_rmse(rows, panel_prediction),
                "panel_fold_params_json": json.dumps(panel_fold_params),
                "anchor_fold_params_json": json.dumps(anchor_fold_params),
                **panel_metrics,
                **{f"lao_{key}": value for key, value in anchor_metrics.items()},
            }
        )
        predictions[config.name] = panel_prediction
    return (
        pd.DataFrame(summaries).sort_values(
            ["panel_mean_rmse", "lao_regret_at_1", "parameter_count", "model"],
            ignore_index=True,
        ),
        predictions,
    )


def plot_aggregate_calibration(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    figure = px.scatter(
        frame,
        x="observed",
        y="predicted",
        color="split",
        hover_data=["run_name"],
        color_discrete_map={"OOF": "#e77836", "tied heldout": "#1f736d"},
        title=title,
    )
    minimum = float(min(frame["observed"].min(), frame["predicted"].min()))
    maximum = float(max(frame["observed"].max(), frame["predicted"].max()))
    figure.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            line={"color": "#183447", "dash": "dash"},
            name="identity",
        )
    )
    figure.update_layout(template="plotly_white", width=1000, height=760)
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def plot_phase_calibration(
    rows: PhaseRows,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    records = []
    for model, prediction in predictions.items():
        for index, value in enumerate(prediction):
            records.append(
                {
                    "model": model,
                    "observed_delta": rows.target_delta[index],
                    "predicted_delta": value,
                    "panel": rows.frame.iloc[index]["panel"],
                    "candidate_id": rows.frame.iloc[index]["candidate_id"],
                }
            )
    frame = pd.DataFrame(records)
    figure = px.scatter(
        frame,
        x="observed_delta",
        y="predicted_delta",
        color="panel",
        facet_col="model",
        facet_col_wrap=2,
        hover_data=["candidate_id"],
        title=title,
    )
    minimum = float(min(frame["observed_delta"].min(), frame["predicted_delta"].min()))
    maximum = float(max(frame["observed_delta"].max(), frame["predicted_delta"].max()))
    figure.add_shape(
        type="line",
        x0=minimum,
        x1=maximum,
        y0=minimum,
        y1=maximum,
        line={"color": "#183447", "dash": "dash"},
    )
    figure.update_layout(template="plotly_white", width=1300, height=1600)
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def selected_aggregate_model(
    dataset: pooled.Dataset,
    families: FamilyPartition,
    sweep: pd.DataFrame,
) -> AggregateModel:
    best_rmse = float(sweep["rmse"].min())
    numerically_tied = sweep[sweep["rmse"] <= best_rmse + 1e-8]
    row = numerically_tied.sort_values(
        ["regret_at_1", "parameter_count", "key"],
    ).iloc[0]
    config = AggregateConfig(
        name=str(row["name"]),
        include_families=bool(row["include_families"]),
        replay=ReplayKind(str(row["replay"])),
        loss=str(row["loss"]),
    )
    shape = AggregateShape(float(row["rho"]), float(row["power"]))
    return fit_aggregate(dataset, np.arange(dataset.n), config, shape, float(row["l2"]), families)


def run_target(target: str, output_dir: Path, max_candidates: int) -> dict[str, Any]:
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, single_evaluation_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    if len({coordinate_key(weights) for weights in single.weights}) != single.n:
        raise ValueError("The 280-row tied panel contains duplicate aggregate coordinates")
    single_evaluation_mask = single.frame["disposition"].ne("reused_exact_phase_tied_alias").to_numpy()
    if int(single_evaluation_mask.sum()) != 238:
        raise ValueError("The tied aggregate evaluation mask must contain 238 newly trained policies")
    baseline_metrics = observatory_baseline_metrics(
        target,
        reference,
        single,
        single_evaluation_indices,
        single_evaluation_mask,
        heldout_frame,
    )
    families = family_partition(single.domain_names)
    sweep, oof_predictions = aggregate_oof_sweep(
        single,
        families,
        max_candidates,
        single_evaluation_mask,
    )
    sweep.to_csv(output_dir / f"{target}_aggregate_cv_sweep.csv", index=False)
    aggregate_model = selected_aggregate_model(single, families, sweep)
    selected_row = sweep[
        sweep["key"].eq(
            f"{aggregate_model.config.name}_rho{aggregate_model.shape.rho:g}"
            f"_p{aggregate_model.shape.power:g}_l2{aggregate_model.l2:g}"
        )
    ]
    if len(selected_row) != 1:
        raise ValueError("Could not recover the selected aggregate sweep row")
    best_key = str(selected_row.iloc[0]["key"])
    tied_heldout = tied_heldout_dataset(target, reference, single, heldout_frame, heldout_weights)
    heldout_prediction = aggregate_model.predict(tied_heldout.weights)
    policy_heldout_mask = policy_matched_heldout_mask(target, heldout_frame, "single_phase")
    single_fit_heldout_indices = single_evaluation_indices[single_evaluation_indices >= reference.n] - reference.n
    policy_heldout_mask[single_fit_heldout_indices] = False
    policy_heldout_observed = heldout_frame.loc[
        policy_heldout_mask,
        TARGET_COLUMNS[target],
    ].to_numpy(dtype=float)
    policy_heldout_prediction = aggregate_model.predict(heldout_weights[policy_heldout_mask])
    candidate_baseline = pd.DataFrame(
        [
            {
                "target": target,
                "policy_class": "single_phase",
                "model": "orthogonal_physical_aggregate",
                "split": "fit_oof",
                **regression_metrics(
                    single.y[single_evaluation_mask],
                    oof_predictions[best_key][single_evaluation_mask],
                ),
            },
            {
                "target": target,
                "policy_class": "single_phase",
                "model": "orthogonal_physical_aggregate",
                "split": "policy_matched_heldout",
                **regression_metrics(policy_heldout_observed, policy_heldout_prediction),
            },
        ]
    )
    baseline_metrics = pd.concat([baseline_metrics, candidate_baseline], ignore_index=True)
    baseline_metrics.to_csv(output_dir / f"{target}_observatory_pareto_baseline.csv", index=False)
    aggregate_metrics = pd.DataFrame(
        [
            {
                "target": target,
                "split": "OOF",
                **regression_metrics(
                    single.y[single_evaluation_mask],
                    oof_predictions[best_key][single_evaluation_mask],
                ),
            },
            {
                "target": target,
                "split": "tied_heldout",
                **regression_metrics(tied_heldout.y, heldout_prediction),
            },
        ]
    )
    aggregate_metrics.to_csv(output_dir / f"{target}_aggregate_metrics.csv", index=False)
    calibration = pd.concat(
        [
            pd.DataFrame(
                {
                    "observed": single.y[single_evaluation_mask],
                    "predicted": oof_predictions[best_key][single_evaluation_mask],
                    "split": "OOF",
                    "run_name": single.frame.loc[single_evaluation_mask, "run_name"].astype(str),
                }
            ),
            pd.DataFrame(
                {
                    "observed": tied_heldout.y,
                    "predicted": heldout_prediction,
                    "split": "tied heldout",
                    "run_name": tied_heldout.frame["wandb_run_base"].astype(str),
                }
            ),
        ],
        ignore_index=True,
    )
    calibration.to_csv(output_dir / f"{target}_aggregate_predictions.csv", index=False)
    plot_aggregate_calibration(
        calibration,
        output_dir / f"{target}_aggregate_calibration.html",
        f"{target}: physical aggregate acquisition",
    )

    phase_rows = load_phase_rows(target, single.domain_names, aggregate_model.phase_fraction)
    phase_summary, phase_predictions = phase_cross_validation(phase_rows, aggregate_model, target)
    phase_summary.insert(0, "target", target)
    phase_summary.to_csv(output_dir / f"{target}_phase_lopo_metrics.csv", index=False)
    panel_metrics = phase_panel_metrics(phase_rows, phase_predictions)
    panel_metrics.insert(0, "target", target)
    panel_metrics.to_csv(output_dir / f"{target}_phase_lopo_panel_metrics.csv", index=False)
    antithetic_metrics, antithetic_predictions = antithetic_component_metrics(phase_rows, phase_predictions)
    antithetic_metrics.insert(0, "target", target)
    antithetic_metrics.to_csv(output_dir / f"{target}_phase_antithetic_metrics.csv", index=False)
    antithetic_predictions.insert(0, "target", target)
    antithetic_predictions.to_csv(output_dir / f"{target}_phase_antithetic_predictions.csv", index=False)
    prediction_frame = phase_rows.frame.copy()
    prediction_frame["observed_delta"] = phase_rows.target_delta
    for model, prediction in phase_predictions.items():
        prediction_frame[f"predicted_{model}"] = prediction
    prediction_frame.to_csv(output_dir / f"{target}_phase_lopo_predictions.csv", index=False)
    plot_phase_calibration(
        phase_rows,
        phase_predictions,
        output_dir / f"{target}_phase_lopo_calibration.html",
        f"{target}: leave-panel-out fixed-aggregate phase deltas",
    )

    best_phase_config = next(config for config in phase_configs(target) if config.name == phase_summary.iloc[0]["model"])
    phase_model = fit_phase(
        phase_rows,
        np.arange(len(phase_rows.frame)),
        aggregate_model,
        best_phase_config,
    )
    model_payload = {
        "target": target,
        "aggregate": {
            "config": asdict(aggregate_model.config),
            "shape": asdict(aggregate_model.shape),
            "l2": aggregate_model.l2,
            "intercept": aggregate_model.intercept,
            "bucket_coef": aggregate_model.bucket_coef,
            "family_coef": aggregate_model.family_coef,
            "replay_coef": aggregate_model.replay_coef,
            "family_names": aggregate_model.families.names,
        },
        "phase": {
            "config": asdict(phase_model.config),
            "params": phase_model.params,
        },
    }
    (output_dir / f"{target}_selected_models.json").write_text(
        json.dumps(json_clean(model_payload), indent=2, sort_keys=True) + "\n"
    )
    return {
        "target": target,
        "selected_aggregate": best_key,
        "selected_phase": best_phase_config.name,
        "aggregate_metrics": aggregate_metrics.to_dict(orient="records"),
        "phase_metrics": phase_summary.iloc[0].to_dict(),
        "phase_rows": len(phase_rows.frame),
        "tied_heldout_coordinates": tied_heldout.n,
    }


def write_report(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Orthogonal aggregate and phase-order identification",
        "",
        "This is a development-screen result. The targeted pairwise causal panel remained sealed.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['target']}",
                "",
                f"- Selected aggregate: `{summary['selected_aggregate']}`.",
                f"- Selected phase model: `{summary['selected_phase']}`.",
                f"- Fixed-aggregate development rows: {summary['phase_rows']}.",
                f"- Coordinate-disjoint tied heldouts: {summary['tied_heldout_coordinates']}.",
                "",
                "Aggregate metrics:",
                "",
                pd.DataFrame(summary["aggregate_metrics"]).to_markdown(index=False),
                "",
                "Best leave-panel-out phase metrics:",
                "",
                pd.DataFrame([summary["phase_metrics"]]).to_markdown(index=False),
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    targets = tuple(part.strip() for part in args.targets.split(",") if part.strip())
    unknown = sorted(set(targets).difference(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_target(target, args.output_dir, args.max_aggregate_candidates) for target in targets]
    write_report(args.output_dir, summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(json_clean(summaries), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
