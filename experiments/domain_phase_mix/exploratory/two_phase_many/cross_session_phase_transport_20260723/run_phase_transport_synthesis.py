# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Evaluate aggregate-spine plus finite-potential phase-transport surrogates."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import lsq_linear, nnls
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
TWO_PHASE_MANY = HERE.parent
PACKET = TWO_PHASE_MANY / "chatgpt_pro_two_phase_surrogate_packet_20260721"
HELDOUT_PATH = TWO_PHASE_MANY / "reference_outputs" / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
OUTPUT = TWO_PHASE_MANY / "reference_outputs" / "cross_session_phase_transport_20260723"

TARGETS = ("uncheatable_bpb", "table9_macro_bpb")
MODEL_IDS = (
    "zero_transition",
    "han39",
    "fpt_total_global",
    "fpt_total_family",
    "fpt_shortage_family",
    "fpt_decomposed_family",
)
RIDGE_GRID = (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
RANDOM_SEEDS = (20260723, 20260724, 20260725, 20260726, 20260727)
EPS = 1e-12


@dataclass(frozen=True)
class Panel:
    domains: tuple[str, ...]
    families: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    c0: np.ndarray
    c1: np.ndarray
    proportional: np.ndarray
    alpha0: float
    alpha1: float
    one_weights: np.ndarray
    two_weights: np.ndarray
    one_targets: dict[str, np.ndarray]
    two_targets: dict[str, np.ndarray]
    group_ids: np.ndarray


@dataclass(frozen=True)
class SFOSFit:
    intercept: float
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    feature_families: tuple[int, ...]
    effective_df: float


@dataclass(frozen=True)
class PhaseFit:
    model_id: str
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    ridge: float
    effective_df: float
    rank: int
    condition_number: float


def _weights(frame: pd.DataFrame, domains: tuple[str, ...]) -> np.ndarray:
    phase0 = frame[[f"phase_0_weight::{domain}" for domain in domains]].to_numpy(float)
    phase1 = frame[[f"phase_1_weight::{domain}" for domain in domains]].to_numpy(float)
    weights = np.stack([phase0, phase1], axis=1)
    if not np.allclose(weights.sum(axis=2), 1.0, atol=2e-7):
        raise ValueError("Mixture weights are not normalized")
    return weights


def load_panel() -> Panel:
    catalog = json.loads((PACKET / "data" / "catalog.json").read_text())
    spec = catalog["datasets"]["delphi_3e18_two_phase_fit"]
    domains = tuple(spec["domains"])
    family_names = tuple(spec["families"])
    domain_index = {domain: index for index, domain in enumerate(domains)}
    family_members = tuple(
        np.asarray([domain_index[domain] for domain in spec["families"][family]], dtype=int) for family in family_names
    )
    one = pd.read_csv(PACKET / "data/canonical/delphi_3e18_one_phase_fit.csv")
    two = pd.read_csv(PACKET / "data/canonical/delphi_3e18_two_phase_fit.csv")
    if not np.array_equal(one["group_id"].astype(str), two["group_id"].astype(str)):
        raise ValueError("One-phase and two-phase rows are not aligned by group_id")

    c0 = np.asarray(spec["c0"], dtype=float)
    c1 = np.asarray(spec["c1"], dtype=float)
    phase_fraction = c0 / (c0 + c1)
    alpha0 = float(np.median(phase_fraction))
    if np.max(np.abs(phase_fraction - alpha0)) > 1e-8:
        raise ValueError("Phase fraction is not constant across buckets")
    alpha1 = 1.0 - alpha0
    proportional = 1.0 / (c0 + c1)
    proportional /= proportional.sum()

    one_weights = _weights(one, domains)
    two_weights = _weights(two, domains)
    aggregate = alpha0 * two_weights[:, 0] + alpha1 * two_weights[:, 1]
    if not np.allclose(one_weights[:, 0], one_weights[:, 1], atol=5e-10):
        raise ValueError("One-phase fit rows are not tied")
    if not np.allclose(aggregate, one_weights[:, 0], atol=5e-10):
        raise ValueError("The paired policies are not aggregate matched")

    return Panel(
        domains=domains,
        families=family_names,
        family_members=family_members,
        c0=c0,
        c1=c1,
        proportional=proportional,
        alpha0=alpha0,
        alpha1=alpha1,
        one_weights=one_weights,
        two_weights=two_weights,
        one_targets={target: one[target].to_numpy(float) for target in TARGETS},
        two_targets={target: two[target].to_numpy(float) for target in TARGETS},
        group_ids=one["group_id"].astype(str).to_numpy(),
    )


def smooth_positive(value: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return 0.5 * (value + np.sqrt(value * value + epsilon * epsilon))


def hierarchy_matrix(size: int, groups: Sequence[Sequence[int]]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for group in groups:
        members = np.asarray(group, dtype=int)
        if len(members) <= 1:
            continue
        for member in members:
            row = np.zeros(size)
            row[member] = 1.0
            row[members] -= 1.0 / len(members)
            rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, size))


def sfos_design(
    panel: Panel,
    aggregate_weights: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    weights = np.asarray(aggregate_weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    relative = weights / np.maximum(panel.proportional[None, :], EPS)
    shortage = (relative + 1.0) ** -0.35
    log_excess = np.log1p(relative) - 2.0
    replay = smooth_positive(log_excess) ** 2

    columns: list[np.ndarray] = []
    names: list[str] = []
    kinds: list[str] = []
    families: list[int] = []
    bucket_family = np.empty(len(panel.domains), dtype=int)
    for family_index, members in enumerate(panel.family_members):
        bucket_family[members] = family_index
    for bucket, domain in enumerate(panel.domains):
        columns.append(shortage[:, bucket])
        names.append(f"shortage::{domain}")
        kinds.append("shortage")
        families.append(int(bucket_family[bucket]))

    family_means: list[np.ndarray] = []
    for family_index, (family, members) in enumerate(zip(panel.families, panel.family_members, strict=True)):
        family_weights = panel.proportional[members]
        family_weights /= family_weights.sum()
        mean = replay[:, members] @ family_weights
        family_means.append(mean)
        columns.append(mean)
        names.append(f"family_replay::{family}")
        kinds.append("replay")
        families.append(family_index)

    for family_index, (family, members) in enumerate(zip(panel.families, panel.family_members, strict=True)):
        family_weights = panel.proportional[members]
        family_weights /= family_weights.sum()
        values = replay[:, members]
        maximum = np.max(values, axis=1, keepdims=True)
        soft_extreme = maximum[:, 0] + np.log(np.sum(family_weights[None, :] * np.exp(values - maximum), axis=1))
        columns.append(smooth_positive(soft_extreme - family_means[family_index]))
        names.append(f"family_extreme::{family}")
        kinds.append("replay")
        families.append(family_index)

    design = np.column_stack(columns)
    return design, tuple(names), tuple(kinds), tuple(families)


def fit_sfos(
    panel: Panel,
    aggregate_weights: np.ndarray,
    target: np.ndarray,
) -> SFOSFit:
    design, names, kinds, families = sfos_design(panel, aggregate_weights)
    mean = design.mean(axis=0)
    centered = design - mean[None, :]
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    scale = np.where(scale < 1e-10, 1.0, scale)
    standardized = centered / scale[None, :]
    target_mean = float(np.mean(target))
    target_centered = target - target_mean

    shortage_groups = [[int(index) for index in members] for members in panel.family_members]
    hierarchy = hierarchy_matrix(design.shape[1], shortage_groups)
    penalty = np.vstack(
        [
            math.sqrt(10.0) * np.eye(design.shape[1]),
            math.sqrt(10.0) * hierarchy,
        ]
    )
    augmented = np.vstack([standardized, penalty])
    rhs = np.concatenate([target_centered, np.zeros(penalty.shape[0])])
    standardized_coefficients, _ = nnls(
        augmented,
        rhs,
        maxiter=max(1000, 5 * design.shape[1]),
    )
    coefficients = standardized_coefficients / scale
    intercept = target_mean - float(mean @ coefficients)

    active = np.flatnonzero(standardized_coefficients > 1e-9)
    if len(active):
        active_design = standardized[:, active]
        active_penalty = 10.0 * np.eye(len(active))
        if hierarchy.size:
            active_hierarchy = hierarchy[:, active]
            active_penalty += 10.0 * active_hierarchy.T @ active_hierarchy
        gram = active_design.T @ active_design
        effective_df = 1.0 + float(np.trace(np.linalg.pinv(gram + active_penalty) @ gram))
    else:
        effective_df = 1.0
    return SFOSFit(
        intercept=intercept,
        coefficients=coefficients,
        feature_names=names,
        feature_kinds=kinds,
        feature_families=families,
        effective_df=effective_df,
    )


def predict_sfos(
    panel: Panel,
    fit: SFOSFit,
    aggregate_weights: np.ndarray,
) -> np.ndarray:
    design, _, _, _ = sfos_design(panel, aggregate_weights)
    return fit.intercept + design @ fit.coefficients


def sfos_family_potentials(
    panel: Panel,
    fit: SFOSFit,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design, _, _, _ = sfos_design(panel, weights)
    contribution = design * fit.coefficients[None, :]
    shortage = np.zeros((len(design), len(panel.families)))
    replay = np.zeros_like(shortage)
    for column, (kind, family) in enumerate(zip(fit.feature_kinds, fit.feature_families, strict=True)):
        destination = shortage if kind == "shortage" else replay
        destination[:, family] += contribution[:, column]
    return shortage, replay


def potential_transport_design(
    panel: Panel,
    aggregate_fit: SFOSFit,
    weights: np.ndarray,
    model_id: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    phase0 = weights[:, 0]
    phase1 = weights[:, 1]
    aggregate = panel.alpha0 * phase0 + panel.alpha1 * phase1
    shortage0, replay0 = sfos_family_potentials(panel, aggregate_fit, phase0)
    shortage1, replay1 = sfos_family_potentials(panel, aggregate_fit, phase1)
    shortage_aggregate, replay_aggregate = sfos_family_potentials(panel, aggregate_fit, aggregate)

    odd_shortage = panel.alpha0 * panel.alpha1 * (shortage0 - shortage1)
    odd_replay = panel.alpha0 * panel.alpha1 * (replay0 - replay1)
    jensen_shortage = panel.alpha0 * shortage0 + panel.alpha1 * shortage1 - shortage_aggregate
    jensen_replay = panel.alpha0 * replay0 + panel.alpha1 * replay1 - replay_aggregate

    if model_id == "fpt_total_global":
        odd = np.sum(odd_shortage + odd_replay, axis=1, keepdims=True)
        jensen = np.sum(jensen_shortage + jensen_replay, axis=1, keepdims=True)
        return (
            np.hstack([odd, jensen]),
            ("odd::total", "jensen::total"),
            np.asarray([False, True]),
        )

    if model_id == "fpt_total_family":
        odd = odd_shortage + odd_replay
        jensen = jensen_shortage + jensen_replay
        names = tuple(
            [f"odd::total::{family}" for family in panel.families]
            + [f"jensen::total::{family}" for family in panel.families]
        )
        return (
            np.hstack([odd, jensen]),
            names,
            np.asarray([False] * len(panel.families) + [True] * len(panel.families)),
        )

    if model_id == "fpt_shortage_family":
        names = tuple(
            [f"odd::shortage::{family}" for family in panel.families]
            + [f"jensen::shortage::{family}" for family in panel.families]
        )
        return (
            np.hstack([odd_shortage, jensen_shortage]),
            names,
            np.asarray([False] * len(panel.families) + [True] * len(panel.families)),
        )

    if model_id == "fpt_decomposed_family":
        names = tuple(
            [f"odd::shortage::{family}" for family in panel.families]
            + [f"jensen::shortage::{family}" for family in panel.families]
            + [f"odd::replay::{family}" for family in panel.families]
            + [f"jensen::replay::{family}" for family in panel.families]
        )
        design = np.hstack([odd_shortage, jensen_shortage, odd_replay, jensen_replay])
        constrained = np.asarray(
            [False] * len(panel.families)
            + [True] * len(panel.families)
            + [False] * len(panel.families)
            + [True] * len(panel.families)
        )
        return design, names, constrained
    raise ValueError(f"Unsupported potential-transport model {model_id}")


def han_design(
    panel: Panel,
    weights: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    phase0 = weights[:, 0]
    phase1 = weights[:, 1]
    aggregate = panel.alpha0 * phase0 + panel.alpha1 * phase1
    proportional = np.maximum(panel.proportional, EPS)
    kappa = 0.5
    x0 = panel.alpha0 * phase0 / proportional[None, :]
    x1 = panel.alpha1 * phase1 / proportional[None, :]
    tied0 = panel.alpha0 * aggregate / proportional[None, :]
    tied1 = panel.alpha1 * aggregate / proportional[None, :]
    h0 = 1.0 - np.exp(-kappa * np.clip(x0, 0.0, 1e6))
    h1 = 1.0 - np.exp(-kappa * np.clip(x1, 0.0, 1e6))
    ht0 = 1.0 - np.exp(-kappa * np.clip(tied0, 0.0, 1e6))
    ht1 = 1.0 - np.exp(-kappa * np.clip(tied1, 0.0, 1e6))
    abandonment = h0 * (1.0 - h1) - ht0 * (1.0 - ht1)
    novelty = -((1.0 - h0) * h1 - (1.0 - ht0) * ht1)
    names = tuple(
        [f"abandonment::{domain}" for domain in panel.domains] + [f"late_novelty::{domain}" for domain in panel.domains]
    )
    return (
        np.hstack([abandonment, novelty]),
        names,
        np.ones(2 * len(panel.domains), dtype=bool),
    )


def phase_design(
    panel: Panel,
    aggregate_fit: SFOSFit,
    weights: np.ndarray,
    model_id: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    if model_id == "zero_transition":
        return np.zeros((len(weights), 0)), (), np.zeros(0, dtype=bool)
    if model_id == "han39":
        return han_design(panel, weights)
    return potential_transport_design(panel, aggregate_fit, weights, model_id)


def _phase_hierarchy(panel: Panel, model_id: str, feature_count: int) -> np.ndarray:
    if model_id != "han39":
        return np.zeros((0, feature_count))
    groups: list[list[int]] = []
    bucket_count = len(panel.domains)
    for channel_offset in (0, bucket_count):
        for members in panel.family_members:
            groups.append([channel_offset + int(index) for index in members])
    return hierarchy_matrix(feature_count, groups)


def fit_phase(
    panel: Panel,
    model_id: str,
    design: np.ndarray,
    target_delta: np.ndarray,
    constrained: np.ndarray,
    ridge: float,
    names: tuple[str, ...],
) -> PhaseFit:
    if design.shape[1] == 0:
        return PhaseFit(model_id, np.zeros(0), (), ridge, 0.0, 0, 1.0)
    scale = np.sqrt(np.mean(design * design, axis=0))
    scale = np.where(scale < 1e-10, 1.0, scale)
    standardized = design / scale[None, :]
    hierarchy = _phase_hierarchy(panel, model_id, design.shape[1])
    hierarchy_penalty = 100.0 if model_id == "han39" else 0.0
    penalty_parts: list[np.ndarray] = []
    if ridge > 0:
        penalty_parts.append(math.sqrt(ridge) * np.eye(design.shape[1]))
    if hierarchy_penalty > 0 and hierarchy.size:
        penalty_parts.append(math.sqrt(hierarchy_penalty) * hierarchy)
    augmented = np.vstack([standardized, *penalty_parts]) if penalty_parts else standardized
    rhs = np.concatenate([target_delta, np.zeros(augmented.shape[0] - len(target_delta))])
    lower = np.where(constrained, 0.0, -np.inf)
    result = lsq_linear(
        augmented,
        rhs,
        bounds=(lower, np.full(design.shape[1], np.inf)),
        method="trf",
        tol=1e-12,
        lsmr_tol=1e-12,
        max_iter=2000,
    )
    if not result.success:
        raise RuntimeError(f"{model_id} phase fit failed: {result.message}")
    coefficients = result.x / scale
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    positive = singular_values[singular_values > 1e-10]
    condition_number = float(positive.max() / positive.min()) if len(positive) else float("inf")
    active = np.flatnonzero(np.logical_or(~constrained, result.x > 1e-9))
    active_design = standardized[:, active]
    if len(active):
        gram = active_design.T @ active_design
        active_penalty = ridge * np.eye(len(active))
        if hierarchy_penalty > 0 and hierarchy.size:
            active_hierarchy = hierarchy[:, active]
            active_penalty += hierarchy_penalty * active_hierarchy.T @ active_hierarchy
        effective_df = float(np.trace(np.linalg.pinv(gram + active_penalty) @ gram))
    else:
        effective_df = 0.0
    return PhaseFit(
        model_id=model_id,
        coefficients=coefficients,
        feature_names=names,
        ridge=ridge,
        effective_df=effective_df,
        rank=int(np.linalg.matrix_rank(standardized)),
        condition_number=condition_number,
    )


def predict_phase(fit: PhaseFit, design: np.ndarray) -> np.ndarray:
    if design.shape[1] == 0:
        return np.zeros(len(design))
    return design @ fit.coefficients


def metric_dict(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    order = np.argsort(predicted)
    best = float(np.min(observed))
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.ptp(predicted) > 1e-12 else float("nan")
    result: dict[str, float | int] = {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "bias": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, predicted).statistic) if np.ptp(predicted) > 1e-12 else float("nan"),
        "observed_on_predicted_slope": slope,
        "optimism_over_0p05_count": int(np.count_nonzero(observed - predicted > 0.05)),
        "worst_optimism": float(np.max(observed - predicted)),
    }
    for count in (1, 3, 5):
        result[f"regret_at_{count}"] = float(np.min(observed[order[: min(count, len(order))]]) - best)
    tail_count = max(5, math.ceil(0.15 * len(observed)))
    tail = order[:tail_count]
    result["low_tail_rmse"] = float(np.sqrt(np.mean(residual[tail] * residual[tail])))
    result["low_tail_optimism"] = float(np.mean(np.maximum(observed[tail] - predicted[tail], 0.0)))
    return result


def random_splits(indices: np.ndarray, seed: int, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(indices[train], indices[test]) for train, test in splitter.split(indices)]


def direction_splits(panel: Panel, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    contrast = (panel.two_weights[:, 1] - panel.two_weights[:, 0]) / np.maximum(panel.proportional[None, :], EPS)
    norm = np.linalg.norm(contrast, axis=1)
    normalized = np.divide(
        contrast,
        norm[:, None],
        out=np.zeros_like(contrast),
        where=norm[:, None] > 1e-12,
    )
    _, _, right = np.linalg.svd(normalized, full_matrices=False)
    coordinates = normalized @ right[:5].T
    coordinates = np.column_stack([coordinates, np.log1p(norm)])
    labels = KMeans(
        n_clusters=folds,
        random_state=20260728,
        n_init=50,
    ).fit_predict(coordinates)
    all_indices = np.arange(len(contrast))
    return [(all_indices[labels != fold], all_indices[labels == fold]) for fold in range(folds)]


def _select_ridge(
    panel: Panel,
    model_id: str,
    target: str,
    outer_train: np.ndarray,
    seed: int,
) -> float:
    if model_id == "zero_transition":
        return 0.0
    if model_id == "han39":
        return 10.0
    errors: dict[float, list[np.ndarray]] = {ridge: [] for ridge in RIDGE_GRID}
    for inner_train, inner_test in random_splits(outer_train, seed, folds=4):
        aggregate_fit = fit_sfos(
            panel,
            panel.one_weights[inner_train, 0],
            panel.one_targets[target][inner_train],
        )
        train_design, names, constrained = phase_design(
            panel,
            aggregate_fit,
            panel.two_weights[inner_train],
            model_id,
        )
        test_design, _, _ = phase_design(
            panel,
            aggregate_fit,
            panel.two_weights[inner_test],
            model_id,
        )
        train_delta = panel.two_targets[target][inner_train] - panel.one_targets[target][inner_train]
        test_delta = panel.two_targets[target][inner_test] - panel.one_targets[target][inner_test]
        for ridge in RIDGE_GRID:
            fit = fit_phase(
                panel,
                model_id,
                train_design,
                train_delta,
                constrained,
                ridge,
                names,
            )
            errors[ridge].append(predict_phase(fit, test_design) - test_delta)
    return min(
        RIDGE_GRID,
        key=lambda ridge: (
            float(np.sqrt(np.mean(np.concatenate(errors[ridge]) ** 2))),
            ridge,
        ),
    )


def run_paired_cv(panel: Panel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    ridge_rows: list[dict[str, object]] = []
    schemes = [
        *[(f"random_seed_{seed}", seed, random_splits(np.arange(280), seed)) for seed in RANDOM_SEEDS],
        ("direction", 20260728, direction_splits(panel)),
    ]
    for target in TARGETS:
        for model_id in MODEL_IDS:
            for scheme, seed, splits in schemes:
                for fold, (train, test) in enumerate(splits):
                    ridge = _select_ridge(
                        panel,
                        model_id,
                        target,
                        train,
                        seed + fold + 101,
                    )
                    aggregate_fit = fit_sfos(
                        panel,
                        panel.one_weights[train, 0],
                        panel.one_targets[target][train],
                    )
                    train_design, names, constrained = phase_design(
                        panel,
                        aggregate_fit,
                        panel.two_weights[train],
                        model_id,
                    )
                    test_design, _, _ = phase_design(
                        panel,
                        aggregate_fit,
                        panel.two_weights[test],
                        model_id,
                    )
                    train_delta = panel.two_targets[target][train] - panel.one_targets[target][train]
                    phase_fit = fit_phase(
                        panel,
                        model_id,
                        train_design,
                        train_delta,
                        constrained,
                        ridge,
                        names,
                    )
                    predicted_one = predict_sfos(
                        panel,
                        aggregate_fit,
                        panel.one_weights[test, 0],
                    )
                    predicted_delta = predict_phase(phase_fit, test_design)
                    predicted_two = predicted_one + predicted_delta
                    observed_delta = panel.two_targets[target][test] - panel.one_targets[target][test]
                    for local, row_index in enumerate(test):
                        prediction_rows.append(
                            {
                                "target": target,
                                "model_id": model_id,
                                "scheme": scheme,
                                "fold": fold,
                                "row_index": int(row_index),
                                "group_id": panel.group_ids[row_index],
                                "observed_one": panel.one_targets[target][row_index],
                                "predicted_one": predicted_one[local],
                                "observed_delta": observed_delta[local],
                                "predicted_delta": predicted_delta[local],
                                "observed_two": panel.two_targets[target][row_index],
                                "predicted_two": predicted_two[local],
                            }
                        )
                    ridge_rows.append(
                        {
                            "target": target,
                            "model_id": model_id,
                            "scheme": scheme,
                            "fold": fold,
                            "selected_ridge": ridge,
                            "aggregate_effective_df": aggregate_fit.effective_df,
                            "phase_effective_df": phase_fit.effective_df,
                            "phase_rank": phase_fit.rank,
                            "phase_condition_number": phase_fit.condition_number,
                        }
                    )
                    for name, coefficient in zip(
                        phase_fit.feature_names,
                        phase_fit.coefficients,
                        strict=True,
                    ):
                        coefficient_rows.append(
                            {
                                "target": target,
                                "model_id": model_id,
                                "scheme": scheme,
                                "fold": fold,
                                "feature": name,
                                "coefficient": coefficient,
                            }
                        )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(ridge_rows),
        pd.DataFrame(coefficient_rows),
    )


def summarize_cv(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model_id, scheme), group in predictions.groupby(
        ["target", "model_id", "scheme"],
        sort=False,
    ):
        for response in ("one", "delta", "two"):
            rows.append(
                {
                    "target": target,
                    "model_id": model_id,
                    "scheme": scheme,
                    "response": response,
                    **metric_dict(
                        group[f"observed_{response}"].to_numpy(float),
                        group[f"predicted_{response}"].to_numpy(float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _json_weights(value: str, domains: tuple[str, ...]) -> np.ndarray:
    mapping = json.loads(value)
    return np.asarray([float(mapping[domain]) for domain in domains], dtype=float)


def load_heldouts(panel: Panel) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(HELDOUT_PATH, low_memory=False)
    phase0 = np.stack([_json_weights(value, panel.domains) for value in frame["phase_0_weights_json"]])
    phase1 = np.stack([_json_weights(value, panel.domains) for value in frame["phase_1_weights_json"]])
    weights = np.stack([phase0, phase1], axis=1)
    if not np.allclose(weights.sum(axis=2), 1.0, atol=2e-7):
        raise ValueError("Heldout policies are not normalized")
    return frame, weights


def _full_fit_ridge(
    panel: Panel,
    model_id: str,
    target: str,
) -> float:
    return _select_ridge(
        panel,
        model_id,
        target,
        np.arange(len(panel.group_ids)),
        20260729,
    )


def run_heldout_predictions(
    panel: Panel,
    heldouts: pd.DataFrame,
    heldout_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    aggregate_weights = panel.alpha0 * heldout_weights[:, 0] + panel.alpha1 * heldout_weights[:, 1]
    for target in TARGETS:
        aggregate_fit = fit_sfos(
            panel,
            panel.one_weights[:, 0],
            panel.one_targets[target],
        )
        predicted_aggregate = predict_sfos(
            panel,
            aggregate_fit,
            aggregate_weights,
        )
        for model_id in MODEL_IDS:
            ridge = _full_fit_ridge(panel, model_id, target)
            train_design, names, constrained = phase_design(
                panel,
                aggregate_fit,
                panel.two_weights,
                model_id,
            )
            heldout_design, _, _ = phase_design(
                panel,
                aggregate_fit,
                heldout_weights,
                model_id,
            )
            target_delta = panel.two_targets[target] - panel.one_targets[target]
            phase_fit = fit_phase(
                panel,
                model_id,
                train_design,
                target_delta,
                constrained,
                ridge,
                names,
            )
            predicted_phase = predict_phase(phase_fit, heldout_design)
            result = heldouts.copy()
            result["fit_target"] = target
            result["model_id"] = model_id
            result["predicted_aggregate"] = predicted_aggregate
            result["predicted_phase_delta"] = predicted_phase
            result["predicted_target"] = predicted_aggregate + predicted_phase
            result["observed_target"] = heldouts[target].to_numpy(float)
            result["residual"] = result["predicted_target"] - result["observed_target"]
            result["optimism"] = result["observed_target"] - result["predicted_target"]
            result["phase_tv"] = 0.5 * np.sum(
                np.abs(heldout_weights[:, 1] - heldout_weights[:, 0]),
                axis=1,
            )
            result["aggregate_hash"] = [
                sha256(np.round(weights, 10).astype(np.float64).tobytes()).hexdigest() for weights in aggregate_weights
            ]
            prediction_frames.append(result)
            parameter_rows.append(
                {
                    "target": target,
                    "model_id": model_id,
                    "ridge": ridge,
                    "aggregate_effective_df": aggregate_fit.effective_df,
                    "phase_effective_df": phase_fit.effective_df,
                    "phase_rank": phase_fit.rank,
                    "phase_condition_number": phase_fit.condition_number,
                    "phase_parameter_count": len(phase_fit.coefficients),
                    "phase_coefficients_json": json.dumps(
                        dict(
                            zip(
                                phase_fit.feature_names,
                                phase_fit.coefficients.tolist(),
                                strict=True,
                            )
                        ),
                        sort_keys=True,
                    ),
                }
            )
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(parameter_rows)


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def summarize_heldouts(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    coordinate_disjoint = ~predictions["fit_panel_overlap"].map(_is_true)
    masks: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=predictions.index)),
        ("coordinate_disjoint", coordinate_disjoint),
        (
            "target_matched",
            predictions["proposal_target"].fillna("").map(lambda value: str(value))
            == predictions["fit_target"].map({"uncheatable_bpb": "uncheatable", "table9_macro_bpb": "table9"}),
        ),
        (
            "coordinate_disjoint_target_matched",
            coordinate_disjoint
            & (
                predictions["proposal_target"].fillna("").map(str)
                == predictions["fit_target"].map(
                    {
                        "uncheatable_bpb": "uncheatable",
                        "table9_macro_bpb": "table9",
                    }
                )
            ),
        ),
    ]
    for policy_class in ("single_phase_tied", "two_phase"):
        masks.append(
            (
                f"coordinate_disjoint_policy_{policy_class}",
                coordinate_disjoint & (predictions["policy_class"].astype(str) == policy_class),
            )
        )
    for (target, model_id), group in predictions.groupby(
        ["fit_target", "model_id"],
        sort=False,
    ):
        for slice_name, mask in masks:
            selected = group.loc[mask.loc[group.index]]
            if len(selected) < 5:
                continue
            rows.append(
                {
                    "target": target,
                    "model_id": model_id,
                    "slice": slice_name,
                    **metric_dict(
                        selected["observed_target"].to_numpy(float),
                        selected["predicted_target"].to_numpy(float),
                    ),
                }
            )
        for (panel_tag, candidate_kind), selected in group.loc[coordinate_disjoint.loc[group.index]].groupby(
            [
                group["panel_tag"].fillna("historical"),
                group["candidate_kind"].fillna("historical"),
            ],
            dropna=False,
        ):
            if len(selected) < 5:
                continue
            rows.append(
                {
                    "target": target,
                    "model_id": model_id,
                    "slice": f"series::{panel_tag}::{candidate_kind}",
                    **metric_dict(
                        selected["observed_target"].to_numpy(float),
                        selected["predicted_target"].to_numpy(float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def exact_fiber_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in predictions.groupby(
        ["fit_target", "model_id"],
        sort=False,
    ):
        tied = group.loc[group["phase_tv"] < 1e-9]
        controls = (
            tied.groupby("aggregate_hash")["observed_target"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "observed_tied", "count": "tied_count"})
        )
        candidates = group.loc[group["phase_tv"] >= 1e-9].join(
            controls,
            on="aggregate_hash",
            how="inner",
        )
        if candidates.empty:
            continue
        candidates = candidates.copy()
        candidates["observed_fiber_delta"] = candidates["observed_target"] - candidates["observed_tied"]
        candidates["predicted_fiber_delta"] = candidates["predicted_phase_delta"]
        candidates["fiber_residual"] = candidates["predicted_fiber_delta"] - candidates["observed_fiber_delta"]
        rows.append(candidates)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_fibers(fibers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model_id), group in fibers.groupby(
        ["fit_target", "model_id"],
        sort=False,
    ):
        rows.append(
            {
                "target": target,
                "model_id": model_id,
                "slice": "all_exact_aggregate_fibers",
                **metric_dict(
                    group["observed_fiber_delta"].to_numpy(float),
                    group["predicted_fiber_delta"].to_numpy(float),
                ),
            }
        )
        for (panel_tag, candidate_kind), selected in group.groupby(
            [
                group["panel_tag"].fillna("historical"),
                group["candidate_kind"].fillna("historical"),
            ],
            dropna=False,
        ):
            if len(selected) < 5:
                continue
            rows.append(
                {
                    "target": target,
                    "model_id": model_id,
                    "slice": f"series::{panel_tag}::{candidate_kind}",
                    **metric_dict(
                        selected["observed_fiber_delta"].to_numpy(float),
                        selected["predicted_fiber_delta"].to_numpy(float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_scatter(
    frame: pd.DataFrame,
    observed_column: str,
    predicted_column: str,
    title: str,
    output_name: str,
) -> None:
    targets = list(TARGETS)
    colors = {
        "zero_transition": "#7f8c8d",
        "han39": "#d95f02",
        "fpt_total_global": "#7570b3",
        "fpt_total_family": "#1b9e77",
        "fpt_shortage_family": "#e6ab02",
        "fpt_decomposed_family": "#a6761d",
        "compact_zero": "#7f8c8d",
        "compact_fpt_global": "#1b9e77",
    }
    model_ids = tuple(frame["model_id"].drop_duplicates().astype(str))
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable", "Table-9 macro"),
        horizontal_spacing=0.1,
    )
    for column, target in enumerate(targets, start=1):
        selected_target = frame.loc[(frame["target"] if "target" in frame else frame["fit_target"]) == target]
        for model_id in model_ids:
            selected = selected_target.loc[selected_target["model_id"] == model_id]
            if selected.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=selected[observed_column],
                    y=selected[predicted_column],
                    mode="markers",
                    name=model_id,
                    legendgroup=model_id,
                    showlegend=column == 1,
                    marker={
                        "size": 6,
                        "opacity": 0.55,
                        "color": colors.get(model_id, "#4c78a8"),
                    },
                    customdata=np.column_stack(
                        [
                            selected.get(
                                "heldout_id",
                                selected.get("group_id", pd.Series("", index=selected.index)),
                            ),
                            selected.get(
                                "candidate_kind",
                                pd.Series("", index=selected.index),
                            ).fillna(""),
                        ]
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>observed=%{x:.6f}<br>predicted=%{y:.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        values = np.concatenate(
            [
                selected_target[observed_column].to_numpy(float),
                selected_target[predicted_column].to_numpy(float),
            ]
        )
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        figure.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                line={"color": "#506070", "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        width=1500,
        height=700,
        legend={"orientation": "h", "y": -0.15},
    )
    figure.update_xaxes(title_text="Observed BPB")
    figure.update_yaxes(title_text="Predicted BPB")
    figure.write_html(
        OUTPUT / output_name,
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )


def write_report(
    cv_metrics: pd.DataFrame,
    heldout_metrics: pd.DataFrame,
    fiber_metrics: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    random_cv = cv_metrics.loc[cv_metrics["scheme"].str.startswith("random_seed") & (cv_metrics["response"] == "delta")]
    random_summary = (
        random_cv.groupby(["target", "model_id"], as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            regret_at_1=("regret_at_1", "mean"),
            calibration_slope=("observed_on_predicted_slope", "mean"),
        )
        .sort_values(["target", "rmse"])
    )
    heldout_primary = heldout_metrics.loc[heldout_metrics["slice"] == "coordinate_disjoint_target_matched"].sort_values(
        ["target", "rmse"]
    )
    fiber_primary = fiber_metrics.loc[fiber_metrics["slice"] == "all_exact_aggregate_fibers"].sort_values(
        ["target", "rmse"]
    )
    lines = [
        "# Cross-session finite-potential transport synthesis",
        "",
        "This is exposed local development evidence, not confirmation.",
        "",
        "## Mechanism",
        "",
        (
            "The independently fitted SFOS39 one-phase spine is decomposed into "
            "family shortage and replay potentials. For each potential, the "
            "two-phase correction uses an odd signed secant and an even Jensen "
            "batching gap. Both are exactly zero for tied policies. The fitted "
            "chronological head therefore cannot repair aggregate misspecification "
            "by changing the single-phase restriction."
        ),
        "",
        "## Paired phase-delta CV",
        "",
        random_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Coordinate-disjoint target-matched heldouts",
        "",
        heldout_primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Exact aggregate fibers",
        "",
        fiber_primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Full-fit complexity",
        "",
        parameters[
            [
                "target",
                "model_id",
                "ridge",
                "aggregate_effective_df",
                "phase_effective_df",
                "phase_rank",
                "phase_condition_number",
                "phase_parameter_count",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Status",
        "",
        (
            "Candidate status is assigned after comparing these artifacts with "
            "the frozen gate in `PROTOCOL.json` and completing the raw optimization "
            "audit."
        ),
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


def write_registry() -> None:
    rows = [
        {
            "family_name": "zero_transition",
            "relationship": "Aggregate-only control",
            "new_mechanism": "None",
            "additional_degrees_of_freedom": 0,
            "status": "active_control",
        },
        {
            "family_name": "han39",
            "relationship": "Session-2 occupancy baseline",
            "new_mechanism": "Bounded abandonment and late novelty",
            "additional_degrees_of_freedom": 78,
            "status": "active_baseline",
        },
        {
            "family_name": "fpt_total_global",
            "relationship": "Combines exact reversal, secant-Jensen transport, and finite potential transport",
            "new_mechanism": "Global chronological transport of the fitted aggregate potential",
            "additional_degrees_of_freedom": 2,
            "status": "active",
        },
        {
            "family_name": "fpt_total_family",
            "relationship": "Family-pooled extension of fpt_total_global",
            "new_mechanism": "Family-specific chronological transport",
            "additional_degrees_of_freedom": 6,
            "status": "active",
        },
        {
            "family_name": "fpt_shortage_family",
            "relationship": "Nested shortage-only restriction",
            "new_mechanism": "Transport of unresolved diminishing-return potential only",
            "additional_degrees_of_freedom": 6,
            "status": "active",
        },
        {
            "family_name": "fpt_decomposed_family",
            "relationship": "Nested extension of fpt_total_family",
            "new_mechanism": "Separate shortage and finite-corpus replay transport",
            "additional_degrees_of_freedom": 12,
            "status": "active",
        },
    ]
    pd.DataFrame(rows).to_csv(OUTPUT / "approach_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "evaluation_round": "cross_session_phase_transport_batch_1",
                "candidate_ids": json.dumps(list(MODEL_IDS)),
                "fit_data": "Delphi 3e18 280 exact aggregate-matched one/two-phase pairs",
                "heldout_data": str(HELDOUT_PATH),
                "heldout_status": "exposed development",
                "outcomes_inspected_before_candidate_freeze": (
                    "Historical archive outcomes and five ChatGPT Pro session reports; "
                    "no result from this new candidate batch."
                ),
                "freeze_artifact": str(HERE / "PROTOCOL.json"),
            }
        ]
    ).to_csv(OUTPUT / "data_use_ledger.csv", index=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    predictions, ridge_audit, coefficients = run_paired_cv(panel)
    cv_metrics = summarize_cv(predictions)
    heldouts, heldout_weights = load_heldouts(panel)
    heldout_predictions, parameters = run_heldout_predictions(
        panel,
        heldouts,
        heldout_weights,
    )
    heldout_metrics = summarize_heldouts(heldout_predictions)
    fibers = exact_fiber_predictions(heldout_predictions)
    fiber_metrics = summarize_fibers(fibers)

    predictions.to_csv(OUTPUT / "paired_cv_predictions.csv", index=False)
    cv_metrics.to_csv(OUTPUT / "paired_cv_metrics.csv", index=False)
    ridge_audit.to_csv(OUTPUT / "paired_cv_fit_audit.csv", index=False)
    coefficients.to_csv(OUTPUT / "paired_cv_coefficients.csv", index=False)
    heldout_predictions.to_csv(OUTPUT / "heldout_predictions.csv", index=False)
    heldout_metrics.to_csv(OUTPUT / "heldout_metrics.csv", index=False)
    parameters.to_csv(OUTPUT / "full_fit_parameters.csv", index=False)
    fibers.to_csv(OUTPUT / "exact_fiber_predictions.csv", index=False)
    fiber_metrics.to_csv(OUTPUT / "exact_fiber_metrics.csv", index=False)
    write_registry()
    write_report(cv_metrics, heldout_metrics, fiber_metrics, parameters)

    random_predictions = predictions.loc[predictions["scheme"].str.startswith("random_seed")]
    plot_scatter(
        random_predictions,
        "observed_delta",
        "predicted_delta",
        "Paired chronological-response OOF",
        "paired_delta_oof_scatter.html",
    )
    coordinate_disjoint = heldout_predictions.loc[~heldout_predictions["fit_panel_overlap"].map(_is_true)]
    plot_scatter(
        coordinate_disjoint,
        "observed_target",
        "predicted_target",
        "Coordinate-disjoint 3e18 development heldouts",
        "heldout_calibration_scatter.html",
    )
    if not fibers.empty:
        plot_scatter(
            fibers.rename(columns={"fit_target": "target"}),
            "observed_fiber_delta",
            "predicted_fiber_delta",
            "Exact aggregate-fiber chronological response",
            "exact_fiber_delta_scatter.html",
        )

    print(OUTPUT / "report.md")


if __name__ == "__main__":
    main()
