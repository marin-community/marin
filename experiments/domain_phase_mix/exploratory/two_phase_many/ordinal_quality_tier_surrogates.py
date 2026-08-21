# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pandas>=2.2"]
# ///
"""Evaluate the shared ordinal-ladder TSJL surrogate on candidate means.

This script implements the shared ordinal-ladder block proposed in the external
TSJL packet and evaluates it on the fixed-subset 240-candidate mean table.

Important caveat:
The packet omitted the exact ``source15`` hard-partition feature builder used
for the base TSJL block. To make the shared model runnable locally, this script
uses the closest packet-only reconstruction we found from the stated TSJL
primitive family. That reconstruction matches the reported in-sample shared-row
metrics almost exactly, but the CV row may still differ.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    DEFAULT_OBJECTIVE_METRIC,
    load_two_phase_many_candidate_summary_spec,
)

ROOT = Path(__file__).resolve().parents[4]
EXPLORATORY_DIR = ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many"
PACKET_REFERENCE_PATH = EXPLORATORY_DIR / "chatgpt_5_4_packet/code/reference_surrogates.py"
REPO_LITERATURE_PATH = EXPLORATORY_DIR / "literature_motivated_surrogates.py"
DOWNLOAD_LITERATURE_PATH = Path("/Users/calvinxu/Downloads/TSJL/literature_motivated_surrogates.py")
DEFAULT_CANDIDATE_SUMMARY = EXPLORATORY_DIR / "qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_candidate_summary.csv"
DEFAULT_OUTPUT_CSV = EXPLORATORY_DIR / "ordinal_quality_tier_shared_local.csv"
DEFAULT_DIFF_CSV = EXPLORATORY_DIR / "ordinal_quality_tier_shared_diff_vs_reference.csv"
DEFAULT_MAPPING_CSV = EXPLORATORY_DIR / "ordinal_quality_tier_mapping_local.csv"
DEFAULT_EPOCH_AWARE_OUTPUT_CSV = EXPLORATORY_DIR / "ordinal_quality_tier_shared_epoch_aware_local.csv"

MODEL_NAME = "Ordinal-Ladder Quality TSJL (shared)"
EPOCH_MODE_UNIT = "unit"
EPOCH_MODE_REAL = "real"
SOURCE15_KAPPA = 0.1
SOURCE15_POWER = 1.5
SOURCE15_TAU_TOTAL = 0.2
SOURCE15_SIGMA_TOTAL = 0.1
SOURCE15_TAU_SHIFT = 0.05
SOURCE15_SIGMA_SHIFT = 0.05
SHARED_ALPHA = 0.75
SHARED_RESIDUAL_BETA = 1.0
SHARED_EXPOSURE_KAPPA = 0.50
SHARED_EXPOSURE_POWER = 2.5
SHARED_TAU_TOTAL = 0.30
SHARED_SIGMA_TOTAL = 0.10
SHARED_TAU_SHIFT = 0.05
SHARED_SIGMA_SHIFT = 0.10
TAIL_BASIS_DEGREE = 0
FIT_LAMBDAS = (1e-6, 1e-6)
CV_LAMBDA_GRIDS = ((1e-2, 1e-1, 1.0, 10.0), (1e-4, 1e-3, 1e-2, 1e-1, 1.0))
REFERENCE_ROW_NAME = "Ordinal-Ladder Quality TSJL (shared)"

# This 8-feature base block is the closest packet-only reconstruction we found
# for the missing source15 TSJL builder. It reproduces the shared model's
# reported in-sample row almost exactly on the packet dataset.
SOURCE15_FEATURE_NAMES = (
    "sat",
    "th2",
    "shift",
    "abs_shift",
    "th_abs",
    "bal",
    "sat_x_sth",
    "th_x_bal",
)

EPS = 1e-8
HUBER_DELTA = 0.02


@dataclass(frozen=True)
class LadderTopic:
    """Ordered quality ladder for one retained topic."""

    name: str
    domain_indices: tuple[int, ...]
    tier_labels: tuple[str, ...]


@dataclass(frozen=True)
class LadderStructure:
    """Full set of ordered ladder topics."""

    topics: tuple[LadderTopic, ...]


@dataclass(frozen=True)
class SourceGroup:
    """One hard-partition source15 group."""

    name: str
    domain_indices: tuple[int, ...]


def load_module(module_path: Path, module_name: str):
    """Import a Python file by path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def infer_binary_ladder_mapping(domain_names: list[str]) -> pd.DataFrame:
    """Infer current low/high quality ladders from domain names."""
    rows: list[dict[str, object]] = []
    for domain_name in domain_names:
        ladder_topic: str | None = None
        tier_rank: int | None = None
        tier_label: str | None = None
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_low"):
            ladder_topic = domain_name.split("/", 1)[1].removesuffix("_low")
            tier_rank = 0
            tier_label = "low"
        elif domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high"):
            ladder_topic = domain_name.split("/", 1)[1].removesuffix("_high")
            tier_rank = 1
            tier_label = "high"
        rows.append(
            {
                "domain_name": domain_name,
                "ladder_topic": ladder_topic,
                "tier_rank": tier_rank,
                "tier_label": tier_label,
            }
        )
    return pd.DataFrame.from_records(rows)


def ladder_structure_from_mapping(mapping: pd.DataFrame, domain_names: list[str]) -> LadderStructure:
    """Build the ordered ladder structure from a mapping table."""
    domain_to_idx = {name: idx for idx, name in enumerate(domain_names)}
    topics: list[LadderTopic] = []
    for topic_name, topic_frame in mapping.dropna(subset=["ladder_topic"]).groupby("ladder_topic"):
        ordered = topic_frame.sort_values("tier_rank")
        domain_indices = tuple(domain_to_idx[name] for name in ordered["domain_name"])
        tier_labels = tuple(
            str(label) if pd.notna(label) else f"tier_{rank}"
            for label, rank in zip(ordered["tier_label"], ordered["tier_rank"], strict=True)
        )
        topics.append(LadderTopic(str(topic_name), domain_indices, tier_labels))
    topics.sort(key=lambda topic: topic.name)
    return LadderStructure(tuple(topics))


def build_source15_groups(domain_names: list[str]) -> tuple[SourceGroup, ...]:
    """Build the 15-group hard partition used by the TSJL base block."""
    cc_high: list[int] = []
    cc_low: list[int] = []
    groups: list[SourceGroup] = []
    for domain_idx, domain_name in enumerate(domain_names):
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high"):
            cc_high.append(domain_idx)
            continue
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_low"):
            cc_low.append(domain_idx)
            continue
        groups.append(SourceGroup(domain_name, (domain_idx,)))

    return (
        SourceGroup("cc_high", tuple(cc_high)),
        SourceGroup("cc_low", tuple(cc_low)),
        *groups,
    )


def softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def hill(x: np.ndarray, kappa: float, p: float) -> np.ndarray:
    """Hill-style saturation."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    kp = float(max(kappa, 1e-12)) ** float(p)
    xp = x ** float(p)
    return xp / (xp + kp + 1e-12)


def entropy(probs: np.ndarray) -> np.ndarray:
    """Categorical entropy."""
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def simpson_diversity(probs: np.ndarray) -> np.ndarray:
    """Simpson diversity."""
    probs = np.asarray(probs, dtype=float)
    return 1.0 - np.sum(probs * probs, axis=-1)


def threshold_basis(n_thresholds: int, degree: int) -> np.ndarray:
    """Polynomial basis over upper-tail thresholds."""
    z = np.linspace(1.0 / (n_thresholds + 1.0), n_thresholds / (n_thresholds + 1.0), n_thresholds)
    centered = z - np.mean(z)
    columns = [np.ones_like(centered)]
    if degree >= 1:
        columns.append(centered)
    if degree >= 2:
        columns.append(centered**2 - np.mean(centered**2))
    return np.column_stack(columns)


def upper_tail_shares(probs: np.ndarray) -> np.ndarray:
    """Return cumulative upper-tail mass above the lowest tier."""
    rev_cumsum = np.cumsum(probs[..., ::-1], axis=-1)[..., ::-1]
    return rev_cumsum[..., 1:]


def aggregate_upper_tail(cum_shares: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate upper-tail shares and logits with a small basis."""
    basis = threshold_basis(cum_shares.shape[-1], degree)
    tail = np.einsum("...t,tr->...r", cum_shares, basis)
    logits = np.log(np.clip(cum_shares, EPS, 1.0 - EPS) / np.clip(1.0 - cum_shares, EPS, 1.0))
    tail_logit = np.einsum("...t,tr->...r", logits, basis)
    return tail, tail_logit


def quality_scores(n_tiers: int) -> np.ndarray:
    """Evenly spaced ordinal quality scores in [-1, 1]."""
    return np.linspace(-1.0, 1.0, n_tiers, dtype=float)


def signed_threshold(x: np.ndarray, tau: float, sigma: float) -> np.ndarray:
    """Signed thresholded shift."""
    return np.sign(x) * softplus((np.abs(x) - float(tau)) / float(sigma)) * float(sigma)


def build_source15_tsjl_block(weights: np.ndarray, groups: tuple[SourceGroup, ...]) -> np.ndarray:
    """Build the hard-partition TSJL base block.

    This is a packet-only reconstruction of the missing ``source15`` base
    builder. It uses 8 features per group, which yields 120 features total and
    the expected 121-parameter model after adding a global intercept.
    """
    if weights.ndim != 3 or weights.shape[1] != 2:
        raise ValueError(f"Expected (R, 2, M), got {weights.shape}")

    columns: list[np.ndarray] = []
    for group in groups:
        w0 = weights[:, 0, list(group.domain_indices)].sum(axis=1)
        w1 = weights[:, 1, list(group.domain_indices)].sum(axis=1)
        total = w0 + w1
        shift = w1 - w0
        sat = hill(total, SOURCE15_KAPPA, SOURCE15_POWER)
        th = softplus((total - SOURCE15_TAU_TOTAL) / SOURCE15_SIGMA_TOTAL) * SOURCE15_SIGMA_TOTAL
        shift_threshold = signed_threshold(shift, SOURCE15_TAU_SHIFT, SOURCE15_SIGMA_SHIFT)
        abs_shift = np.abs(shift)
        abs_shift_threshold = softplus((abs_shift - SOURCE15_TAU_SHIFT) / SOURCE15_SIGMA_SHIFT) * SOURCE15_SIGMA_SHIFT
        balance = shift / (total + EPS)
        feature_map = {
            "sat": sat,
            "th2": th**2,
            "shift": shift,
            "abs_shift": abs_shift,
            "th_abs": abs_shift_threshold,
            "bal": balance,
            "sat_x_sth": sat * shift_threshold,
            "th_x_bal": th * balance,
        }
        columns.extend(feature_map[name] for name in SOURCE15_FEATURE_NAMES)
    return np.column_stack(columns)


def resolve_literature_module_path() -> Path | None:
    """Return the best available literature-motivated surrogate module path."""
    if REPO_LITERATURE_PATH.exists():
        return REPO_LITERATURE_PATH
    if DOWNLOAD_LITERATURE_PATH.exists():
        return DOWNLOAD_LITERATURE_PATH
    return None


def build_topic_block(
    weights: np.ndarray,
    topic: LadderTopic,
    *,
    exposure_kappa: float,
    exposure_power: float,
    tau_total: float,
    sigma_total: float,
    tau_shift: float,
    sigma_shift: float,
    tail_basis_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-topic ladder features for the shared ordinal block."""
    tier_weights = weights[:, :, list(topic.domain_indices)]
    w0 = tier_weights[:, 0, :]
    w1 = tier_weights[:, 1, :]

    u0 = np.sum(w0, axis=1)
    u1 = np.sum(w1, axis=1)
    total = u0 + u1
    shift = u1 - u0

    scores = quality_scores(tier_weights.shape[2])
    total_tier_mass = w0 + w1
    total_probs = total_tier_mass / (total[:, None] + EPS)
    diff_tier_mass = w1 - w0

    quality_total = (total_tier_mass @ scores) / (total + EPS)
    quality_shift = (diff_tier_mass @ scores) / (total + np.abs(shift) + EPS)

    total_entropy = entropy(total_probs)
    total_simpson = simpson_diversity(total_probs)

    total_cum = upper_tail_shares(total_probs)
    tail_total, tail_logit_total = aggregate_upper_tail(total_cum, tail_basis_degree)

    total_sat = hill(total, exposure_kappa, exposure_power)
    total_th = softplus((total - float(tau_total)) / float(sigma_total)) * float(sigma_total)
    shift_th = signed_threshold(shift, tau_shift, sigma_shift)

    columns: list[np.ndarray] = [
        quality_total,
        quality_shift,
        total_sat * quality_total,
        total_th * quality_total,
        shift_th * quality_shift,
        total_entropy,
        tail_logit_total[:, 0],
        total_sat * total_entropy,
        total_sat * tail_logit_total[:, 0],
        total_th * total_entropy,
        total_th * tail_logit_total[:, 0],
        total_simpson,
    ]

    for basis_idx in range(1, tail_total.shape[1]):
        columns.extend(
            [
                tail_total[:, basis_idx],
                total_sat * tail_total[:, basis_idx],
                total_th * tail_total[:, basis_idx],
                tail_logit_total[:, basis_idx],
                total_sat * tail_logit_total[:, basis_idx],
                total_th * tail_logit_total[:, basis_idx],
            ]
        )

    return np.column_stack(columns), total


def shared_weighted(topic_blocks: list[np.ndarray], topic_totals: list[np.ndarray], alpha: float) -> np.ndarray:
    """Exposure-weighted aggregation across ladder topics."""
    weights = np.column_stack(topic_totals)
    weights = np.power(np.maximum(weights, EPS), float(alpha))
    weights = weights / (weights.sum(axis=1, keepdims=True) + EPS)
    return sum(block * weights[:, idx : idx + 1] for idx, block in enumerate(topic_blocks))


def build_ordinal_ladder_shared_block(
    weights: np.ndarray, ladder_structure: LadderStructure
) -> tuple[np.ndarray, list[str]]:
    """Build the shared ordinal-ladder TSJL block."""
    topic_blocks: list[np.ndarray] = []
    topic_totals: list[np.ndarray] = []
    for topic in ladder_structure.topics:
        block, total = build_topic_block(
            weights,
            topic,
            exposure_kappa=SHARED_EXPOSURE_KAPPA,
            exposure_power=SHARED_EXPOSURE_POWER,
            tau_total=SHARED_TAU_TOTAL,
            sigma_total=SHARED_SIGMA_TOTAL,
            tau_shift=SHARED_TAU_SHIFT,
            sigma_shift=SHARED_SIGMA_SHIFT,
            tail_basis_degree=TAIL_BASIS_DEGREE,
        )
        topic_blocks.append(block)
        topic_totals.append(total)

    feature_names = [
        "quality_mean_total",
        "quality_shift_norm",
        "sat_total_x_quality_mean",
        "th_total_x_quality_mean",
        "th_shift_x_quality_shift",
        "tier_entropy_total",
        "top_tier_logit_total",
        "sat_total_x_entropy",
        "sat_total_x_top_tier_logit",
        "th_total_x_entropy",
        "th_total_x_top_tier_logit",
        "tier_gini_total",
    ]
    return shared_weighted(topic_blocks, topic_totals, alpha=SHARED_ALPHA), feature_names


def _broadcast_epoch_multipliers(epoch_multipliers: np.ndarray, n_phases: int, n_domains: int) -> np.ndarray:
    """Normalize epoch multipliers to shape (N, M)."""
    multipliers = np.asarray(epoch_multipliers, dtype=float)
    if multipliers.ndim == 1:
        if multipliers.shape != (n_domains,):
            raise ValueError(f"epoch_multipliers shape {multipliers.shape} != ({n_domains},)")
        return np.tile(multipliers[None, :], (n_phases, 1))
    if multipliers.ndim == 2:
        if multipliers.shape != (n_phases, n_domains):
            raise ValueError(f"epoch_multipliers shape {multipliers.shape} != ({n_phases}, {n_domains})")
        return multipliers
    raise ValueError(f"epoch_multipliers must be 1D or 2D, got {multipliers.ndim}D")


def epoch_exposure_tensor(spec) -> tuple[np.ndarray, np.ndarray]:
    """Return the real epoch-exposure tensor for one dataset spec."""
    multipliers = _broadcast_epoch_multipliers(spec.epoch_multipliers, spec.N, spec.M)
    return spec.weights * multipliers[None, :, :], multipliers


def load_candidate_summary_for_epoch_mode(candidate_csv: Path, *, epoch_mode: str):
    """Load the candidate summary under a chosen feature-exposure mode."""
    reference = load_module(PACKET_REFERENCE_PATH, "packet_reference_surrogates")
    if epoch_mode == EPOCH_MODE_UNIT:
        frame, spec = reference.load_candidate_summary(candidate_csv)
        multipliers = _broadcast_epoch_multipliers(spec.epoch_multipliers, spec.N, spec.M)
        return frame, spec, spec.weights, multipliers
    if epoch_mode == EPOCH_MODE_REAL:
        frame, spec, _ = load_two_phase_many_candidate_summary_spec(
            candidate_csv,
            objective_metric=DEFAULT_OBJECTIVE_METRIC,
            name="ordinal_quality_tier_epoch_aware",
        )
        feature_weights, multipliers = epoch_exposure_tensor(spec)
        return frame, spec, feature_weights, multipliers
    raise ValueError(f"Unknown epoch_mode {epoch_mode!r}")


def fit_ridge_blocks(feature_blocks: list[np.ndarray], y: np.ndarray, lambdas: tuple[float, ...]) -> np.ndarray:
    """Fit a blockwise ridge model with one global intercept."""
    design = np.column_stack(feature_blocks)
    matrix = np.column_stack([np.ones(design.shape[0], dtype=float), design])
    penalty = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)

    start = 1
    for block, lam in zip(feature_blocks, lambdas, strict=True):
        width = block.shape[1]
        penalty[start : start + width, start : start + width] = float(lam) * np.eye(width)
        start += width
    return np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ y)


def predict_ridge_blocks(feature_blocks: list[np.ndarray], beta: np.ndarray) -> np.ndarray:
    """Predict from a blockwise ridge model."""
    design = np.column_stack(feature_blocks)
    matrix = np.column_stack([np.ones(design.shape[0], dtype=float), design])
    return matrix @ beta


def huber_loss(residuals: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    """Elementwise Huber loss."""
    abs_residuals = np.abs(np.asarray(residuals, dtype=float))
    return np.where(abs_residuals <= delta, 0.5 * residuals * residuals, delta * (abs_residuals - 0.5 * delta))


def compute_simple_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    """Compute CV metrics without candidate metadata."""
    from scipy.stats import spearmanr

    residuals = np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float)
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    chosen_idx = int(np.argmax(y_hat))
    best_idx = int(np.argmax(y))
    return {
        "r2": float(1.0 - sse / sst),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "spearman": float(spearmanr(y, y_hat).statistic),
        "huber": float(np.mean(huber_loss(residuals))),
        "regret_at_1": float(y[best_idx] - y[chosen_idx]),
    }


def kfold_indices(n_rows: int, *, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic K-fold split helper."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows, dtype=int)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    return [(np.setdiff1d(indices, test_idx, assume_unique=True), test_idx) for test_idx in folds]


def cv_search(feature_blocks: list[np.ndarray], y: np.ndarray) -> tuple[dict[str, float], tuple[float, ...]]:
    """Grid-search blockwise ridge penalties under 5-fold CV."""
    design = np.column_stack(feature_blocks)
    matrix = np.column_stack([np.ones(design.shape[0], dtype=float), design])
    block_slices: list[slice] = []
    start = 1
    for block in feature_blocks:
        stop = start + block.shape[1]
        block_slices.append(slice(start, stop))
        start = stop

    fold_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in kfold_indices(len(y), seed=0):
        x_train = matrix[train_idx]
        y_train = y[train_idx]
        x_test = matrix[test_idx]
        fold_cache.append((test_idx, x_train.T @ x_train, x_train.T @ y_train, x_test))

    best_metrics: dict[str, float] | None = None
    best_lambdas: tuple[float, ...] | None = None
    for lam0 in CV_LAMBDA_GRIDS[0]:
        for lam1 in CV_LAMBDA_GRIDS[1]:
            penalty = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
            for lam, block_slice in zip((lam0, lam1), block_slices, strict=True):
                penalty[block_slice, block_slice] = float(lam) * np.eye(block_slice.stop - block_slice.start)
            y_hat = np.zeros_like(y, dtype=float)
            for test_idx, xtx, xty, x_test in fold_cache:
                beta = np.linalg.solve(xtx + penalty, xty)
                y_hat[test_idx] = x_test @ beta
            metrics = compute_simple_metrics(y, y_hat)
            if best_metrics is None or metrics["r2"] > best_metrics["r2"]:
                best_metrics = metrics
                best_lambdas = (float(lam0), float(lam1))

    if best_metrics is None or best_lambdas is None:
        raise RuntimeError("CV search failed to evaluate any lambda pair")
    return best_metrics, best_lambdas


def evaluate_shared_model(
    candidate_csv: Path, *, epoch_mode: str = EPOCH_MODE_UNIT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the shared ordinal-ladder model and return metrics + mapping."""
    reference = load_module(PACKET_REFERENCE_PATH, "packet_reference_surrogates")
    frame, spec, feature_weights, multipliers = load_candidate_summary_for_epoch_mode(
        candidate_csv,
        epoch_mode=epoch_mode,
    )

    source_groups = build_source15_groups(spec.domain_names)
    literature_path = resolve_literature_module_path()
    if literature_path is not None:
        literature = load_module(literature_path, "literature_motivated_surrogates_local")
        base_block, base_info = literature.build_tsjl_features(
            replace(spec, weights=feature_weights),
            grouping="source15",
            kappa=SOURCE15_KAPPA,
            p=SOURCE15_POWER,
            tau_t=SOURCE15_TAU_TOTAL,
            sigma_t=SOURCE15_SIGMA_TOTAL,
            tau_m=SOURCE15_TAU_SHIFT,
            sigma_m=SOURCE15_SIGMA_SHIFT,
        )
        base_source = str(literature_path)
        base_n_params = int(base_info["n_params"])
    else:
        base_block = build_source15_tsjl_block(feature_weights, source_groups)
        base_source = "reconstructed_fallback"
        base_n_params = int(1 + base_block.shape[1])

    mapping = infer_binary_ladder_mapping(spec.domain_names)
    ladder_structure = ladder_structure_from_mapping(mapping, spec.domain_names)
    shared_block, shared_feature_names = build_ordinal_ladder_shared_block(feature_weights, ladder_structure)

    beta = fit_ridge_blocks([base_block, shared_block], spec.y, FIT_LAMBDAS)
    y_hat = predict_ridge_blocks([base_block, shared_block], beta)
    metrics = reference.compute_metrics(frame, spec.y, y_hat)
    cv_metrics, cv_best_lambdas = cv_search([base_block, shared_block], spec.y)

    record = {
        "model": MODEL_NAME,
        "n_params": int(base_n_params + shared_block.shape[1]),
        "source15_feature_names": json.dumps(SOURCE15_FEATURE_NAMES),
        "source15_feature_source": base_source,
        "fit_lambdas": json.dumps(FIT_LAMBDAS),
        "cv_best_lambdas": json.dumps(cv_best_lambdas),
        "cv_r2": float(cv_metrics["r2"]),
        "cv_rmse": float(cv_metrics["rmse"]),
        "cv_spearman": float(cv_metrics["spearman"]),
        "cv_huber": float(cv_metrics["huber"]),
        "cv_regret_at_1": float(cv_metrics["regret_at_1"]),
        "shared_feature_names": json.dumps(shared_feature_names),
        "epoch_mode": epoch_mode,
        "epoch_multiplier_min": float(np.min(multipliers)),
        "epoch_multiplier_max": float(np.max(multipliers)),
        "n_small_domains": len(getattr(spec, "small_domains", []) or []),
    }
    record.update(metrics)

    return pd.DataFrame([record]), mapping


def compare_to_reference(local_row: pd.DataFrame, reference_csv: Path) -> pd.DataFrame:
    """Compare the local row to the ChatGPT-provided reference row."""
    reference_frame = pd.read_csv(reference_csv)
    ref_row = reference_frame[reference_frame["model"] == REFERENCE_ROW_NAME]
    if len(ref_row) != 1:
        raise ValueError(f"Expected exactly one {REFERENCE_ROW_NAME!r} row in {reference_csv}")
    ref_record = ref_row.iloc[0]
    local_record = local_row.iloc[0]

    comparison_columns = [
        "n_params",
        "r2",
        "rmse",
        "spearman",
        "huber_loss_mean",
        "regret_at_1",
        "cv_r2",
        "cv_rmse",
        "cv_spearman",
        "cv_huber",
        "cv_regret_at_1",
    ]
    diffs = {"model": MODEL_NAME}
    for column in comparison_columns:
        diffs[f"local_{column}"] = local_record[column]
        diffs[f"reference_{column}"] = ref_record[column]
        if pd.notna(local_record[column]) and pd.notna(ref_record[column]):
            diffs[f"delta_{column}"] = float(local_record[column] - ref_record[column])
    return pd.DataFrame([diffs])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, default=DEFAULT_CANDIDATE_SUMMARY)
    parser.add_argument("--epoch-mode", choices=(EPOCH_MODE_UNIT, EPOCH_MODE_REAL), default=EPOCH_MODE_UNIT)
    parser.add_argument("--reference-comparison-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--diff-csv", type=Path, default=DEFAULT_DIFF_CSV)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    return parser.parse_args()


def main() -> None:
    """Run the local evaluation and write outputs."""
    args = parse_args()
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = DEFAULT_OUTPUT_CSV if args.epoch_mode == EPOCH_MODE_UNIT else DEFAULT_EPOCH_AWARE_OUTPUT_CSV
    local_row, mapping = evaluate_shared_model(args.candidate_summary, epoch_mode=args.epoch_mode)
    mapping.to_csv(args.mapping_csv, index=False)
    local_row.to_csv(output_csv, index=False)
    print(local_row.to_string(index=False))
    print(f"\nWrote {output_csv}")
    print(f"Wrote {args.mapping_csv}")

    if args.reference_comparison_csv is not None:
        diff = compare_to_reference(local_row, args.reference_comparison_csv)
        diff.to_csv(args.diff_csv, index=False)
        print()
        print(diff.to_string(index=False))
        print(f"\nWrote {args.diff_csv}")


if __name__ == "__main__":
    main()
