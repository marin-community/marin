# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Design diagnostics for the 167-partition production swarm.

This script keeps the collaborator handoff schema simple: one row per candidate,
with `candidate_name`, `candidate_type`, and per-phase bucket weights. The
diagnostics compare feature maps for BayesFeatureLinear-style subset selection
from either a uniform-simplex Sobol proposal pool or a proportional-centered
logit proposal pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.linalg import qr
from scipy.spatial import cKDTree
from scipy.stats import norm, qmc
from tqdm.auto import tqdm

DEFAULT_SOURCE_URL = (
    "https://gist.githubusercontent.com/Helw150/8d466c2be46062a21dc0980fdfef4e38/raw/" "datakit_moe_mix_buckets.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "reference_outputs"
    / "production_swarm_mixture_design_167p_20260523_cap100_pool262144"
)
PHASE_NAMES = ("phase_0", "phase_1")
PHASE_FRACTIONS = np.asarray([0.8, 0.2], dtype=float)
TARGET_BUDGET = 10_000_000_000_000
DEFAULT_POOL_SIZE = 262_144
DEFAULT_N_SELECT = 1200
DEFAULT_SEED = 42
DEFAULT_MIN_WEIGHT = 1.5259e-5
DEFAULT_MIN_CONFIG_DISTANCE = 0.001
DEFAULT_MAX_EPOCH_CAP = 100.0
MIXTURE_QUANTUM_DENOMINATOR = 65_536
INFO_RIDGE = 1e-3
FEATURE_EPS = 1e-8
DEFAULT_SELECTION_STRATEGIES = ("qr",)
PROPOSAL_TYPES = ("uniform-simplex", "proportional-logit")
DEFAULT_PROPOSAL = "uniform-simplex"
DEFAULT_PROPOSAL_SIGMAS = "0.05,0.10,0.20,0.35,0.60,1.00,1.50"
BANDED_BALANCED_MAX_EPOCH_EDGES = np.asarray([0.0, 25.0, 50.0, 75.0, 90.0, 100.0], dtype=float)
BANDED_BALANCED_QUOTA_FRACTIONS = np.asarray([0.10, 0.30, 0.35, 0.20, 0.05], dtype=float)


def progress(message: str) -> None:
    """Print a timestamped progress message for long local design runs."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def start_stage(stage_bar: tqdm, message: str) -> None:
    """Print a progress message and update the visible coarse-stage progress bar."""
    progress(message)
    stage_bar.set_postfix_str(message[:96])


@dataclass(frozen=True)
class BucketTable:
    """Production bucket metadata."""

    buckets: tuple[str, ...]
    tokens: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class FeatureBundle:
    """Feature matrix and names for one selection variant."""

    name: str
    matrix: np.ndarray
    feature_names: tuple[str, ...]


def load_bucket_table(source_url: str) -> BucketTable:
    """Load the collaborator bucket table from a CSV URL."""
    payload = urllib.request.urlopen(source_url).read()
    source_sha256 = hashlib.sha256(payload).hexdigest()
    rows = list(csv.DictReader(payload.decode("utf-8-sig").splitlines()))
    if not rows:
        raise ValueError(f"No rows loaded from {source_url}")

    required = {"bucket", "tokens", "epochs_for_8T", "epochs_for_2T"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Bucket table missing columns: {sorted(missing)}")

    buckets = tuple(str(row["bucket"]) for row in rows)
    if len(set(buckets)) != len(buckets):
        raise ValueError("Bucket names must be unique")

    tokens = np.asarray([int(row["tokens"]) for row in rows], dtype=float)
    if np.any(tokens <= 0):
        raise ValueError("All bucket token counts must be positive")

    max_epoch8_error = max(abs(float(row["epochs_for_8T"]) - 8_000_000_000_000 / int(row["tokens"])) for row in rows)
    max_epoch2_error = max(abs(float(row["epochs_for_2T"]) - 2_000_000_000_000 / int(row["tokens"])) for row in rows)
    if max(max_epoch8_error, max_epoch2_error) > 1e-3:
        raise ValueError(
            "Epoch columns are not consistent with 8T/tokens and 2T/tokens: "
            f"max_epoch8_error={max_epoch8_error}, max_epoch2_error={max_epoch2_error}"
        )

    return BucketTable(buckets=buckets, tokens=tokens, source_sha256=source_sha256)


def sobol_simplex_pool(*, n_points: int, n_phases: int, n_domains: int, seed: int, min_weight: float) -> np.ndarray:
    """Generate uniform-simplex Sobol schedules with the existing min-weight normalization."""
    if n_points <= 0 or n_points & (n_points - 1):
        raise ValueError(f"n_points must be a positive power of two for Sobol.random_base2, got {n_points}")
    if min_weight < 0:
        raise ValueError(f"min_weight must be nonnegative, got {min_weight}")

    sobol = qmc.Sobol(d=n_phases * n_domains, scramble=True, seed=seed)
    raw = np.clip(sobol.random_base2(int(math.log2(n_points))), 1e-9, 1.0 - 1e-9)
    weights = np.empty((n_points, n_phases, n_domains), dtype=float)
    for row_idx, row in enumerate(raw.reshape(n_points, n_phases, n_domains)):
        phase_raw = -np.log(row)
        phase_simplex = phase_raw / phase_raw.sum(axis=1, keepdims=True)
        if min_weight > 0:
            phase_simplex = np.where(phase_simplex >= min_weight, phase_simplex, 0.0)
            phase_simplex = phase_simplex / phase_simplex.sum(axis=1, keepdims=True)
        weights[row_idx] = phase_simplex
    return weights


def parse_float_schedule(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float schedule."""
    entries = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not entries:
        raise ValueError("Float schedule must contain at least one value")
    return entries


def proportional_logit_pool(
    *,
    n_points: int,
    n_phases: int,
    proportional: np.ndarray,
    seed: int,
    sigma_values: tuple[float, ...],
) -> np.ndarray:
    """Generate Sobol-normal logit tilts around the proportional mixture."""
    if n_points <= 0 or n_points & (n_points - 1):
        raise ValueError(f"n_points must be a positive power of two for Sobol.random_base2, got {n_points}")
    if n_phases <= 0:
        raise ValueError(f"n_phases must be positive, got {n_phases}")
    prop = np.asarray(proportional, dtype=float)
    if prop.ndim != 1:
        raise ValueError(f"proportional must be one-dimensional, got shape {prop.shape}")
    if np.any(prop <= 0):
        raise ValueError("proportional weights must be strictly positive")
    prop = prop / prop.sum()
    sigmas = np.asarray(sigma_values, dtype=float)
    if sigmas.ndim != 1 or len(sigmas) == 0:
        raise ValueError("sigma_values must be a non-empty one-dimensional sequence")
    if np.any(sigmas < 0):
        raise ValueError("sigma_values must be nonnegative")

    sobol = qmc.Sobol(d=n_phases * len(prop), scramble=True, seed=seed)
    raw = np.clip(sobol.random_base2(int(math.log2(n_points))), 1e-9, 1.0 - 1e-9)
    z = norm.ppf(raw).reshape(n_points, n_phases, len(prop))
    z = z - (z * prop[None, None, :]).sum(axis=2, keepdims=True)
    row_sigmas = np.resize(sigmas, n_points)
    logits = np.log(prop)[None, None, :] + row_sigmas[:, None, None] * z
    logits = logits - logits.max(axis=2, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=2, keepdims=True)


def build_proposal_pool(
    *,
    proposal: str,
    n_points: int,
    n_phases: int,
    n_domains: int,
    seed: int,
    min_weight: float,
    proportional: np.ndarray,
    sigma_values: tuple[float, ...],
) -> np.ndarray:
    """Generate the requested schedule proposal pool."""
    if proposal == "uniform-simplex":
        return sobol_simplex_pool(
            n_points=n_points,
            n_phases=n_phases,
            n_domains=n_domains,
            seed=seed,
            min_weight=min_weight,
        )
    if proposal == "proportional-logit":
        del min_weight, n_domains
        return proportional_logit_pool(
            n_points=n_points,
            n_phases=n_phases,
            proportional=proportional,
            seed=seed,
            sigma_values=sigma_values,
        )
    raise ValueError(f"Unknown proposal: {proposal}")


def integer_simplex_counts(probabilities: np.ndarray, denominator: int = MIXTURE_QUANTUM_DENOMINATOR) -> np.ndarray:
    """Project a probability vector to integer simplex counts by largest remainder."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1:
        raise ValueError(f"Expected a one-dimensional probability vector, got shape {probs.shape}")
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    if np.any(probs < 0):
        raise ValueError("Probabilities must be nonnegative")
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probability vector must have positive mass")

    normalized = probs / total
    raw_counts = normalized * denominator
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(denominator - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts), kind="mergesort")
        counts[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(raw_counts - counts, kind="mergesort")
        for idx in order:
            if remainder == 0:
                break
            if counts[idx] > 0:
                counts[idx] -= 1
                remainder += 1
    if counts.sum() != denominator:
        raise ValueError(f"Integer simplex projection failed: counts sum to {counts.sum()}, expected {denominator}")
    return counts


def quantize_simplex(probabilities: np.ndarray, denominator: int = MIXTURE_QUANTUM_DENOMINATOR) -> np.ndarray:
    """Project a probability vector to the executable mixture lattice."""
    return integer_simplex_counts(probabilities, denominator).astype(float) / float(denominator)


def average_phase_tv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return average phase-TV distance from `left` to each row of `right`."""
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    return np.abs(right_arr - left_arr[None, :, :]).sum(axis=(1, 2)) / (2.0 * left_arr.shape[0])


def filter_min_average_phase_tv(weights: np.ndarray, min_distance: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep ordered representatives separated by average phase-TV distance."""
    if min_distance < 0:
        raise ValueError(f"min_distance must be nonnegative, got {min_distance}")
    if min_distance == 0 or len(weights) <= 1:
        return weights, np.arange(len(weights), dtype=np.int64)

    flat = weights.reshape(weights.shape[0], -1)
    l1_threshold = 2.0 * weights.shape[1] * min_distance
    tree = cKDTree(flat)
    keep = np.ones(weights.shape[0], dtype=bool)
    kept_indices: list[int] = []

    row_iterable = range(weights.shape[0])
    if weights.shape[0] >= 4096:
        row_iterable = tqdm(row_iterable, desc="min-distance rejection", unit="row", leave=False)
    for idx in row_iterable:
        neighbor_indices = [
            neighbor
            for neighbor in tree.query_ball_point(flat[idx], r=l1_threshold, p=np.inf)
            if neighbor < idx and keep[neighbor]
        ]
        if neighbor_indices:
            distances = average_phase_tv(weights[idx], weights[np.asarray(neighbor_indices, dtype=np.int64)])
            if np.any(distances < min_distance):
                keep[idx] = False
                continue
        kept_indices.append(idx)

    source_indices = np.asarray(kept_indices, dtype=np.int64)
    return weights[source_indices], source_indices


def filter_max_epoch_cap(
    weights: np.ndarray,
    epochs: np.ndarray,
    source_indices: np.ndarray,
    *,
    max_epoch_cap: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep candidates whose maximum phase-domain epoch count is below a cap."""
    if max_epoch_cap is None:
        return weights, epochs, source_indices
    if max_epoch_cap <= 0:
        raise ValueError(f"max_epoch_cap must be positive when provided, got {max_epoch_cap}")
    feasible = epochs.max(axis=(1, 2)) <= max_epoch_cap
    return weights[feasible], epochs[feasible], source_indices[feasible]


def normalized_entropy(values: np.ndarray, axis: int) -> np.ndarray:
    """Return entropy normalized to [0, 1] along `axis`."""
    total = values.sum(axis=axis, keepdims=True)
    probs = np.divide(values, np.maximum(total, FEATURE_EPS), out=np.zeros_like(values), where=total > FEATURE_EPS)
    safe_probs = np.where(probs > FEATURE_EPS, probs, 1.0)
    entropy = -np.sum(np.where(probs > FEATURE_EPS, probs * np.log(safe_probs), 0.0), axis=axis)
    choices = values.shape[axis]
    if choices <= 1:
        return np.zeros_like(entropy)
    return entropy / math.log(choices)


def standardize_feature_parts(parts: list[tuple[str, np.ndarray]]) -> FeatureBundle:
    """Flatten feature parts, standardize columns, and return named feature matrix."""
    matrices: list[np.ndarray] = []
    names: list[str] = []
    for prefix, values in parts:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        matrices.append(arr.reshape(arr.shape[0], -1))
        width = matrices[-1].shape[1]
        names.extend(f"{prefix}_{idx}" for idx in range(width))

    raw = np.column_stack(matrices)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    safe_std = np.where(std > FEATURE_EPS, std, 1.0)
    standardized = (raw - mean) / safe_std
    return FeatureBundle(name="", matrix=standardized, feature_names=tuple(names))


def standardize_weighted_feature_parts(parts: list[tuple[str, np.ndarray, np.ndarray | float]]) -> FeatureBundle:
    """Flatten, standardize, and post-weight feature columns.

    The post-standardization weights are intentional: applying mass weights
    before standardization would be cancelled by the column-wise z-score.
    """
    matrices: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    names: list[str] = []
    for prefix, values, column_scale in parts:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        flat = arr.reshape(arr.shape[0], -1)
        scale_arr = np.asarray(column_scale, dtype=float)
        if scale_arr.ndim == 0:
            flat_scale = np.full(flat.shape[1], float(scale_arr), dtype=float)
        else:
            flat_scale = np.broadcast_to(scale_arr, arr.shape[1:]).reshape(-1).astype(float)
        if flat_scale.shape != (flat.shape[1],):
            raise ValueError(f"Scale for {prefix} has shape {flat_scale.shape}, expected {(flat.shape[1],)}")
        matrices.append(flat)
        scales.append(flat_scale)
        names.extend(f"{prefix}_{idx}" for idx in range(flat.shape[1]))

    raw = np.column_stack(matrices)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    safe_std = np.where(std > FEATURE_EPS, std, 1.0)
    standardized = (raw - mean) / safe_std
    weighted = standardized * np.concatenate(scales)[None, :]
    return FeatureBundle(name="", matrix=weighted, feature_names=tuple(names))


def phase_fractions_for(n_phases: int) -> np.ndarray:
    """Return phase fractions for feature scaling."""
    if n_phases == len(PHASE_FRACTIONS):
        return PHASE_FRACTIONS
    return np.full(n_phases, 1.0 / n_phases, dtype=float)


def fisher_feature_scales(proportional: np.ndarray, n_phases: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RMS-normalized Fisher/mass feature scales.

    Phase-domain scales weight each standardized column by
    sqrt(n_phases * n_domains * phase_fraction_h * proportional_i), so the
    average squared scale over phase-domain columns is one.
    """
    prop = np.asarray(proportional, dtype=float)
    if np.any(prop <= 0):
        raise ValueError("Proportional weights must be strictly positive for Fisher feature scaling")
    phase_fractions = phase_fractions_for(n_phases)
    n_domains = prop.shape[0]
    phase_domain = np.sqrt(n_phases * n_domains * phase_fractions[:, None] * prop[None, :])
    domain = np.sqrt(n_domains * prop)
    phase = np.sqrt(n_phases * phase_fractions)
    return phase_domain, domain, phase


def dsp_aligned_parts(
    weights: np.ndarray,
    epochs: np.ndarray,
    proportional: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the unstandardized DSP-aligned feature parts."""
    eps = DEFAULT_MIN_WEIGHT
    prop = proportional[None, None, :]
    epoch_multipliers = np.nanmedian(
        np.divide(epochs, weights, out=np.full_like(epochs, np.nan), where=weights > FEATURE_EPS),
        axis=0,
        keepdims=True,
    )
    prop_epochs = prop * epoch_multipliers
    log_tilt = np.log((weights + eps) / (prop + eps))
    log_epochs = np.log1p(epochs)
    centered_log_epochs = log_epochs - np.log1p(prop_epochs)
    total_epochs = epochs.sum(axis=1)
    prop_total_epochs = prop_epochs.sum(axis=1)
    phase_tv_to_prop = np.abs(weights - prop).sum(axis=2) / 2.0
    phase_tv = np.abs(weights[:, 0, :] - weights[:, 1, :]).sum(axis=1) / 2.0
    return {
        "log_tilt": log_tilt,
        "centered_log_epoch": centered_log_epochs,
        "centered_log_total_epoch": np.log1p(total_epochs) - np.log1p(prop_total_epochs[0]),
        "delta_log_tilt": np.diff(log_tilt, axis=1),
        "delta_centered_log_epoch": np.diff(centered_log_epochs, axis=1),
        "phase_entropy_by_domain": normalized_entropy(weights.transpose(0, 2, 1), axis=2),
        "domain_entropy_by_phase": normalized_entropy(weights, axis=2),
        "phase_tv_to_prop": phase_tv_to_prop,
        "phase_tv": phase_tv,
        "max_centered_log_epoch": centered_log_epochs.max(axis=2),
    }


def raw_epoch_features(weights: np.ndarray, epochs: np.ndarray, proportional: np.ndarray) -> FeatureBundle:
    """Existing feature family with raw epoch exposure."""
    del proportional
    bundle = standardize_feature_parts(
        [
            ("weight", weights),
            ("epoch", epochs),
            ("total_epoch", epochs.sum(axis=1)),
            ("delta_weight", np.diff(weights, axis=1)),
            ("delta_epoch", np.diff(epochs, axis=1)),
            ("phase_entropy_by_domain", normalized_entropy(weights.transpose(0, 2, 1), axis=2)),
            ("domain_entropy_by_phase", normalized_entropy(weights, axis=2)),
            ("phase_range", weights.max(axis=1) - weights.min(axis=1)),
            ("boundary_fraction", np.mean((weights <= 0.05) | (weights >= 0.95), axis=(1, 2))),
        ]
    )
    return FeatureBundle("raw_epoch", bundle.matrix, bundle.feature_names)


def log_epoch_features(weights: np.ndarray, epochs: np.ndarray, proportional: np.ndarray) -> FeatureBundle:
    """Existing feature family with log1p-transformed epoch exposure."""
    del proportional
    log_epochs = np.log1p(epochs)
    bundle = standardize_feature_parts(
        [
            ("weight", weights),
            ("log_epoch", log_epochs),
            ("log_total_epoch", np.log1p(epochs.sum(axis=1))),
            ("delta_weight", np.diff(weights, axis=1)),
            ("delta_log_epoch", np.diff(log_epochs, axis=1)),
            ("phase_entropy_by_domain", normalized_entropy(weights.transpose(0, 2, 1), axis=2)),
            ("domain_entropy_by_phase", normalized_entropy(weights, axis=2)),
            ("phase_range", weights.max(axis=1) - weights.min(axis=1)),
            ("boundary_fraction", np.mean((weights <= 0.05) | (weights >= 0.95), axis=(1, 2))),
        ]
    )
    return FeatureBundle("log_epoch", bundle.matrix, bundle.feature_names)


def dsp_aligned_features(weights: np.ndarray, epochs: np.ndarray, proportional: np.ndarray) -> FeatureBundle:
    """DSP-aligned feature family with relative tilts and saturated exposure."""
    parts = dsp_aligned_parts(weights, epochs, proportional)
    bundle = standardize_feature_parts(
        [
            ("log_tilt", parts["log_tilt"]),
            ("centered_log_epoch", parts["centered_log_epoch"]),
            ("centered_log_total_epoch", parts["centered_log_total_epoch"]),
            ("delta_log_tilt", parts["delta_log_tilt"]),
            ("delta_centered_log_epoch", parts["delta_centered_log_epoch"]),
            ("phase_entropy_by_domain", parts["phase_entropy_by_domain"]),
            ("domain_entropy_by_phase", parts["domain_entropy_by_phase"]),
            ("phase_tv_to_prop", parts["phase_tv_to_prop"]),
            ("phase_tv", parts["phase_tv"]),
            ("max_centered_log_epoch", parts["max_centered_log_epoch"]),
        ]
    )
    return FeatureBundle("dsp_aligned", bundle.matrix, bundle.feature_names)


def fisher_dsp_aligned_features(weights: np.ndarray, epochs: np.ndarray, proportional: np.ndarray) -> FeatureBundle:
    """DSP-aligned features with post-standardization Fisher/mass column weights."""
    n_phases = weights.shape[1]
    phase_domain_scale, domain_scale, phase_scale = fisher_feature_scales(proportional, n_phases)
    parts = dsp_aligned_parts(weights, epochs, proportional)
    domain_delta_scale = (
        domain_scale if n_phases == 2 else np.broadcast_to(domain_scale, parts["delta_log_tilt"].shape[1:])
    )
    bundle = standardize_weighted_feature_parts(
        [
            ("log_tilt", parts["log_tilt"], phase_domain_scale),
            ("centered_log_epoch", parts["centered_log_epoch"], phase_domain_scale),
            ("centered_log_total_epoch", parts["centered_log_total_epoch"], domain_scale),
            ("delta_log_tilt", parts["delta_log_tilt"], domain_delta_scale),
            ("delta_centered_log_epoch", parts["delta_centered_log_epoch"], domain_delta_scale),
            ("phase_entropy_by_domain", parts["phase_entropy_by_domain"], domain_scale),
            ("domain_entropy_by_phase", parts["domain_entropy_by_phase"], phase_scale),
            ("phase_tv_to_prop", parts["phase_tv_to_prop"], phase_scale),
            ("phase_tv", parts["phase_tv"], 1.0),
            ("max_centered_log_epoch", parts["max_centered_log_epoch"], phase_scale),
        ]
    )
    return FeatureBundle("fisher_dsp_aligned", bundle.matrix, bundle.feature_names)


def fisher_dsp_aligned_with_risk_features(
    weights: np.ndarray,
    epochs: np.ndarray,
    proportional: np.ndarray,
) -> FeatureBundle:
    """Fisher-weighted DSP features plus global repetition-risk summaries."""
    base = fisher_dsp_aligned_features(weights, epochs, proportional)
    small_bucket_mask = proportional <= np.quantile(proportional, 0.10)
    risk = standardize_feature_parts(
        [
            ("risk_log_max_epoch", np.log1p(epochs.max(axis=2))),
            ("risk_log_q95_epoch", np.log1p(np.quantile(epochs, 0.95, axis=2))),
            ("risk_log_total_epoch_max", np.log1p(epochs.sum(axis=1).max(axis=1))),
            (
                "risk_log_phase_epoch_imbalance",
                np.log1p(np.abs(epochs[:, 0, :].sum(axis=1) - epochs[:, 1, :].sum(axis=1))),
            ),
            ("risk_small_bucket_mass", weights[:, :, small_bucket_mask].sum(axis=(1, 2))),
        ]
    )
    matrix = np.column_stack([base.matrix, risk.matrix])
    return FeatureBundle(
        "fisher_dsp_aligned_with_risk",
        matrix,
        (*base.feature_names, *risk.feature_names),
    )


def dsp_aligned_with_weights_features(
    weights: np.ndarray,
    epochs: np.ndarray,
    proportional: np.ndarray,
) -> FeatureBundle:
    """DSP-aligned feature family with raw weights retained for absolute simplex coverage."""
    base = dsp_aligned_features(weights, epochs, proportional)
    extra = standardize_feature_parts(
        [
            ("weight", weights),
            ("delta_weight", np.diff(weights, axis=1)),
            ("phase_range", weights.max(axis=1) - weights.min(axis=1)),
        ]
    )
    matrix = np.column_stack([base.matrix, extra.matrix])
    return FeatureBundle(
        "dsp_aligned_with_weights",
        matrix,
        (*base.feature_names, *extra.feature_names),
    )


FEATURE_BUILDERS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], FeatureBundle]] = {
    "raw_epoch": raw_epoch_features,
    "log_epoch": log_epoch_features,
    "dsp_aligned": dsp_aligned_features,
    "fisher_dsp_aligned": fisher_dsp_aligned_features,
    "fisher_dsp_aligned_with_risk": fisher_dsp_aligned_with_risk_features,
    "dsp_aligned_with_weights": dsp_aligned_with_weights_features,
}

D_OPTIMAL_CANDIDATE_TYPES = frozenset(FEATURE_BUILDERS)


def select_by_qr_pivot(features: FeatureBundle, n_select: int, row_scale: np.ndarray | None = None) -> np.ndarray:
    """Select rows by QR pivoting on the transposed Bayesian linear design matrix."""
    if n_select > features.matrix.shape[0]:
        raise ValueError(f"Cannot select {n_select} rows from {features.matrix.shape[0]} candidates")
    design = np.column_stack([np.ones(features.matrix.shape[0]), features.matrix])
    if row_scale is not None:
        scale = np.asarray(row_scale, dtype=float)
        if scale.shape != (features.matrix.shape[0],):
            raise ValueError(f"row_scale has shape {scale.shape}, expected {(features.matrix.shape[0],)}")
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("row_scale must contain positive finite values")
        design = design * scale[:, None]
    _, _, pivots = qr(design.T, pivoting=True, mode="economic", check_finite=False)
    return np.asarray(pivots[:n_select], dtype=np.int64)


def epoch_band_quotas(*, n_select: int, quota_fractions: np.ndarray) -> np.ndarray:
    """Return integer quotas whose sum is exactly `n_select`."""
    if n_select <= 0:
        raise ValueError(f"n_select must be positive, got {n_select}")
    fractions = np.asarray(quota_fractions, dtype=float)
    if fractions.ndim != 1 or len(fractions) == 0:
        raise ValueError("quota_fractions must be a non-empty one-dimensional array")
    if np.any(fractions < 0) or fractions.sum() <= 0:
        raise ValueError("quota_fractions must be nonnegative with positive total mass")
    raw = fractions / fractions.sum() * n_select
    quotas = np.floor(raw).astype(np.int64)
    remainder = int(n_select - quotas.sum())
    if remainder > 0:
        order = np.argsort(-(raw - quotas), kind="mergesort")
        quotas[order[:remainder]] += 1
    return quotas


def epoch_band_indices(max_epoch: np.ndarray, band_edges: np.ndarray) -> list[np.ndarray]:
    """Return disjoint row indices for half-open max-epoch bands, with the last band closed."""
    values = np.asarray(max_epoch, dtype=float)
    edges = np.asarray(band_edges, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"max_epoch must be one-dimensional, got shape {values.shape}")
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("band_edges must contain at least two values")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("band_edges must be strictly increasing")
    bands: list[np.ndarray] = []
    for band_idx, (lower, upper) in enumerate(pairwise(edges)):
        if band_idx == len(edges) - 2:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values >= lower) & (values < upper)
        bands.append(np.flatnonzero(mask))
    return bands


def select_banded_by_qr(
    features: FeatureBundle,
    *,
    n_select: int,
    max_epoch: np.ndarray,
    band_edges: np.ndarray,
    quota_fractions: np.ndarray,
) -> np.ndarray:
    """Select rows by running QR pivoting inside explicit max-epoch bands."""
    bands = epoch_band_indices(max_epoch, band_edges)
    if len(bands) != len(quota_fractions):
        raise ValueError(f"Expected {len(bands)} quota fractions, got {len(quota_fractions)}")
    quotas = epoch_band_quotas(n_select=n_select, quota_fractions=quota_fractions)
    capacities = np.asarray([len(indices) for indices in bands], dtype=np.int64)
    if capacities.sum() < n_select:
        raise ValueError(f"Only {capacities.sum()} candidates are available across bands; need {n_select}")

    capped_quotas = np.minimum(quotas, capacities)
    deficit = int(n_select - capped_quotas.sum())
    if deficit > 0:
        remaining_capacity = capacities - capped_quotas
        for band_idx in np.argsort(band_edges[:-1], kind="mergesort"):
            if deficit == 0:
                break
            add = min(deficit, int(remaining_capacity[band_idx]))
            capped_quotas[band_idx] += add
            deficit -= add
    if deficit:
        raise ValueError(f"Could not allocate {n_select} selections across epoch bands")

    selected_parts: list[np.ndarray] = []
    for band_indices, quota in zip(bands, capped_quotas, strict=True):
        if quota == 0:
            continue
        band_features = FeatureBundle(
            name=features.name,
            matrix=features.matrix[band_indices],
            feature_names=features.feature_names,
        )
        selected_parts.append(band_indices[select_by_qr_pivot(band_features, int(quota))])
    return np.concatenate(selected_parts).astype(np.int64)


def epoch_penalty_row_scale(
    epochs: np.ndarray,
    *,
    max_epoch_cap: float,
    penalty_lambda: float,
    power: float = 2.0,
) -> np.ndarray:
    """Return positive row scales that reduce QR leverage for high-repetition schedules."""
    if max_epoch_cap <= 0:
        raise ValueError(f"max_epoch_cap must be positive, got {max_epoch_cap}")
    if penalty_lambda < 0:
        raise ValueError(f"penalty_lambda must be nonnegative, got {penalty_lambda}")
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")
    max_epoch = np.asarray(epochs, dtype=float).max(axis=(1, 2))
    normalized = np.maximum(max_epoch / max_epoch_cap, 0.0)
    return np.exp(-penalty_lambda * normalized**power)


def epoch_scale_penalty_row_scale(
    epochs: np.ndarray,
    *,
    epoch_tau: float,
    penalty_lambda: float,
    power: float = 1.0,
    penalty_kind: str = "power",
) -> np.ndarray:
    """Return row scales from an epoch-scale penalty independent of any hard cap."""
    if epoch_tau <= 0:
        raise ValueError(f"epoch_tau must be positive, got {epoch_tau}")
    if penalty_lambda < 0:
        raise ValueError(f"penalty_lambda must be nonnegative, got {penalty_lambda}")
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")
    max_epoch = np.asarray(epochs, dtype=float).max(axis=(1, 2))
    normalized = np.maximum(max_epoch / epoch_tau, 0.0)
    if penalty_kind == "power":
        return (1.0 + normalized**power) ** (-penalty_lambda)
    if penalty_kind == "exp":
        return np.exp(-penalty_lambda * normalized**power)
    raise ValueError(f"Unknown penalty_kind: {penalty_kind}")


def _float_token(value: str) -> float:
    """Parse a float from a strategy token where `p` may stand for a decimal point."""
    return float(value.replace("p", "."))


def parse_scale_penalty_strategy(strategy: str) -> tuple[str, float, float] | None:
    """Parse strategies such as `power_tau20_lam2` or `exp_tau10_lam0p5`."""
    if strategy.startswith("power_tau"):
        kind = "power"
        rest = strategy.removeprefix("power_tau")
    elif strategy.startswith("exp_tau"):
        kind = "exp"
        rest = strategy.removeprefix("exp_tau")
    else:
        return None
    tau_token, separator, lambda_token = rest.partition("_lam")
    if separator != "_lam":
        raise ValueError(f"Scale-penalty strategy must include _lam: {strategy}")
    return kind, _float_token(tau_token), _float_token(lambda_token)


def selection_output_name(variant: str, strategy: str) -> str:
    """Return the candidate type/output directory for one feature/selection pair."""
    if strategy == "qr":
        return variant
    return f"{variant}_{strategy}"


def select_candidates(
    features: FeatureBundle,
    *,
    n_select: int,
    epochs: np.ndarray,
    strategy: str,
    max_epoch_cap: float | None,
) -> np.ndarray:
    """Select candidates with the requested D-optimal strategy."""
    if strategy == "qr":
        return select_by_qr_pivot(features, n_select)
    if strategy == "banded_balanced":
        return select_banded_by_qr(
            features,
            n_select=n_select,
            max_epoch=epochs.max(axis=(1, 2)),
            band_edges=BANDED_BALANCED_MAX_EPOCH_EDGES,
            quota_fractions=BANDED_BALANCED_QUOTA_FRACTIONS,
        )
    if strategy.startswith("penalty_lam"):
        if max_epoch_cap is None:
            raise ValueError(f"{strategy} requires max_epoch_cap; use power_tau*_lam* or exp_tau*_lam* without a cap")
        penalty_lambda = float(strategy.removeprefix("penalty_lam"))
        return select_by_qr_pivot(
            features,
            n_select,
            row_scale=epoch_penalty_row_scale(
                epochs,
                max_epoch_cap=max_epoch_cap,
                penalty_lambda=penalty_lambda,
            ),
        )
    parsed_scale_penalty = parse_scale_penalty_strategy(strategy)
    if parsed_scale_penalty is not None:
        penalty_kind, epoch_tau, penalty_lambda = parsed_scale_penalty
        return select_by_qr_pivot(
            features,
            n_select,
            row_scale=epoch_scale_penalty_row_scale(
                epochs,
                epoch_tau=epoch_tau,
                penalty_lambda=penalty_lambda,
                penalty_kind=penalty_kind,
            ),
        )
    raise ValueError(f"Unknown selection strategy: {strategy}")


def dual_logdet(features: FeatureBundle, selected: np.ndarray) -> float:
    """Compute logdet(lambda I + X'X) through the dual Gram matrix."""
    design = np.column_stack([np.ones(features.matrix.shape[0]), features.matrix])
    chosen = design[selected]
    n_rows, n_params = chosen.shape
    gram = chosen @ chosen.T
    gram[np.diag_indices_from(gram)] += INFO_RIDGE
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        return float("-inf")
    return float((n_params - n_rows) * math.log(INFO_RIDGE) + logdet)


def schedule_summary(
    *,
    name: str,
    weights: np.ndarray,
    selected: np.ndarray,
    epochs: np.ndarray,
    proportional: np.ndarray,
) -> dict[str, float | int | str]:
    """Summarize selected schedules."""
    selected_weights = weights[selected]
    selected_epochs = epochs[selected]
    max_epoch = selected_epochs.max(axis=(1, 2))
    tv_to_prop = np.abs(selected_weights - proportional[None, None, :]).sum(axis=2) / 2.0
    phase_tv = np.abs(selected_weights[:, 0, :] - selected_weights[:, 1, :]).sum(axis=1) / 2.0
    support = (selected_weights > 0).sum(axis=2)
    return {
        "variant": name,
        "n": len(selected),
        "mean_tv_to_proportional": float(tv_to_prop.mean()),
        "q95_tv_to_proportional": float(np.quantile(tv_to_prop, 0.95)),
        "mean_phase_tv": float(phase_tv.mean()),
        "q95_phase_tv": float(np.quantile(phase_tv, 0.95)),
        "mean_support_per_phase": float(support.mean()),
        "q05_support_per_phase": float(np.quantile(support, 0.05)),
        "max_weight": float(selected_weights.max()),
        "zero_fraction": float(np.mean(selected_weights == 0)),
        "median_max_epoch": float(np.quantile(max_epoch, 0.50)),
        "q90_max_epoch": float(np.quantile(max_epoch, 0.90)),
        "q95_max_epoch": float(np.quantile(max_epoch, 0.95)),
        "q99_max_epoch": float(np.quantile(max_epoch, 0.99)),
        "max_epoch": float(max_epoch.max()),
    }


def candidate_columns(buckets: tuple[str, ...]) -> list[str]:
    """Return the collaborator candidate CSV columns."""
    return ["candidate_name", "candidate_type"] + [f"{phase}/{bucket}" for phase in PHASE_NAMES for bucket in buckets]


def weights_from_candidate_frame(frame: pd.DataFrame, buckets: tuple[str, ...]) -> np.ndarray:
    """Return phase weights with shape rows x phases x buckets from a candidate CSV frame."""
    expected_columns = candidate_columns(buckets)
    if list(frame.columns) != expected_columns:
        raise ValueError("Candidate frame does not match collaborator schema")
    phase_weights = []
    for phase in PHASE_NAMES:
        columns = [f"{phase}/{bucket}" for bucket in buckets]
        phase_weights.append(frame[columns].to_numpy(dtype=float))
    return np.stack(phase_weights, axis=1)


def candidate_diagnostics_frame(
    frame: pd.DataFrame,
    *,
    buckets: tuple[str, ...],
    tokens: np.ndarray,
    proportional: np.ndarray,
    denominator: int,
    target_budget: float = TARGET_BUDGET,
) -> pd.DataFrame:
    """Compute row-level sanity diagnostics for any collaborator-shaped candidate frame."""
    weights = weights_from_candidate_frame(frame, buckets)
    token_counts = np.asarray(tokens, dtype=float)
    prop = np.asarray(proportional, dtype=float)
    epochs = weights * (PHASE_FRACTIONS[:, None] * target_budget / token_counts[None, :])[None, :, :]
    tv_to_prop = np.abs(weights - prop[None, None, :]).sum(axis=2) / 2.0
    phase_tv = np.abs(weights[:, 0, :] - weights[:, 1, :]).sum(axis=1) / 2.0
    support = (weights > 0).sum(axis=2)
    max_epoch_flat = epochs.reshape(len(frame), -1).argmax(axis=1)
    max_epoch_phase_idx = max_epoch_flat // len(buckets)
    max_epoch_bucket_idx = max_epoch_flat % len(buckets)
    max_weight_flat = weights.reshape(len(frame), -1).argmax(axis=1)
    max_weight_phase_idx = max_weight_flat // len(buckets)
    max_weight_bucket_idx = max_weight_flat % len(buckets)
    lattice_error = np.abs(weights * denominator - np.round(weights * denominator)).max(axis=(1, 2))
    phase_sum_error = np.abs(weights.sum(axis=2) - 1.0).max(axis=1)
    return pd.DataFrame(
        {
            "candidate_name": frame["candidate_name"].astype(str),
            "candidate_type": frame["candidate_type"].astype(str),
            "row_index": np.arange(len(frame), dtype=np.int64),
            "scope_is_d_optimal": frame["candidate_type"].astype(str).isin(D_OPTIMAL_CANDIDATE_TYPES).to_numpy(),
            "tv_to_proportional_phase0": tv_to_prop[:, 0],
            "tv_to_proportional_phase1": tv_to_prop[:, 1],
            "mean_tv_to_proportional": tv_to_prop.mean(axis=1),
            "phase_tv": phase_tv,
            "support_phase0": support[:, 0],
            "support_phase1": support[:, 1],
            "support_min": support.min(axis=1),
            "zero_fraction": np.mean(weights == 0, axis=(1, 2)),
            "max_weight": weights.max(axis=(1, 2)),
            "max_weight_phase": [PHASE_NAMES[idx] for idx in max_weight_phase_idx],
            "max_weight_bucket": [buckets[idx] for idx in max_weight_bucket_idx],
            "max_epoch": epochs.max(axis=(1, 2)),
            "phase0_max_epoch": epochs[:, 0, :].max(axis=1),
            "phase1_max_epoch": epochs[:, 1, :].max(axis=1),
            "max_epoch_phase": [PHASE_NAMES[idx] for idx in max_epoch_phase_idx],
            "max_epoch_bucket": [buckets[idx] for idx in max_epoch_bucket_idx],
            "lattice_max_abs_error": lattice_error,
            "phase_sum_max_abs_error": phase_sum_error,
        }
    )


def nearest_neighbor_average_phase_tv(weights: np.ndarray) -> np.ndarray:
    """Return each row's nearest-neighbor average phase-TV distance."""
    if len(weights) <= 1:
        return np.full(len(weights), np.nan)
    flat = weights.reshape(weights.shape[0], -1)
    distances, _ = cKDTree(flat).query(flat, k=2, p=1)
    return distances[:, 1] / (2.0 * weights.shape[1])


def _histogram_figure(diagnostics: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for column, label in (
        ("max_epoch", "max epoch"),
        ("mean_tv_to_proportional", "mean TV to proportional"),
        ("phase_tv", "phase TV"),
        ("max_weight", "max weight"),
        ("support_min", "minimum phase support"),
    ):
        fig.add_trace(go.Histogram(x=diagnostics[column], name=label, opacity=0.72, nbinsx=48))
    fig.update_layout(
        title=f"{title}: core distributions",
        barmode="overlay",
        height=520,
        xaxis_title="value",
        yaxis_title="count",
    )
    return fig


def _type_counts_figure(diagnostics: pd.DataFrame, title: str) -> go.Figure:
    counts = diagnostics["candidate_type"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.to_numpy()))
    fig.update_layout(
        title=f"{title}: Candidate type counts",
        height=420,
        xaxis_title="candidate_type",
        yaxis_title="rows",
    )
    return fig


def _row_order_figure(diagnostics: pd.DataFrame, title: str) -> go.Figure:
    type_codes = {
        candidate_type: idx for idx, candidate_type in enumerate(sorted(diagnostics["candidate_type"].unique()))
    }
    colors = diagnostics["candidate_type"].map(type_codes).to_numpy()
    fig = go.Figure(
        go.Scattergl(
            x=diagnostics["row_index"],
            y=np.zeros(len(diagnostics)),
            mode="markers",
            marker={"color": colors, "colorscale": "RdYlGn_r", "size": 7, "showscale": False},
            text=diagnostics["candidate_name"],
            customdata=diagnostics[["candidate_type", "max_epoch", "mean_tv_to_proportional"]].to_numpy(),
            hovertemplate=(
                "row=%{x}<br>name=%{text}<br>type=%{customdata[0]}"
                "<br>max_epoch=%{customdata[1]:.3f}<br>mean_tv=%{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{title}: row-order audit strip",
        height=240,
        yaxis={"visible": False},
        xaxis_title="CSV row",
    )
    return fig


def _tv_epoch_scatter_figure(diagnostics: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for candidate_type, group in diagnostics.groupby("candidate_type", sort=True):
        fig.add_trace(
            go.Scattergl(
                x=group["mean_tv_to_proportional"],
                y=group["max_epoch"],
                mode="markers",
                name=candidate_type,
                text=group["candidate_name"],
                hovertemplate="name=%{text}<br>mean_tv=%{x:.3f}<br>max_epoch=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"{title}: repetition vs distance from proportional",
        height=520,
        xaxis_title="mean phase TV to proportional",
        yaxis_title="max epoch",
    )
    return fig


def _bucket_epoch_figure(diagnostics: pd.DataFrame, title: str) -> go.Figure:
    bucket_max = (
        diagnostics.groupby("max_epoch_bucket", sort=False)["max_epoch"].max().sort_values(ascending=False).head(30)
    )
    fig = go.Figure(go.Bar(x=bucket_max.index.tolist(), y=bucket_max.to_numpy()))
    fig.update_layout(
        title=f"{title}: worst buckets by realized max epoch",
        height=480,
        xaxis_title="bucket",
        yaxis_title="max epoch",
    )
    return fig


def _heatmap_figure(frame: pd.DataFrame, buckets: tuple[str, ...], title: str) -> go.Figure:
    weights = weights_from_candidate_frame(frame, buckets)
    mean_phase_weights = weights.mean(axis=1)
    order = np.argsort(frame["candidate_type"].astype(str).to_numpy(), kind="mergesort")
    fig = go.Figure(
        go.Heatmap(
            z=np.log10(np.maximum(mean_phase_weights[order], FEATURE_EPS)),
            x=buckets,
            y=frame["candidate_name"].astype(str).to_numpy()[order],
            colorscale="RdYlGn_r",
            colorbar={"title": "log10(mean phase weight)"},
        )
    )
    fig.update_layout(
        title=f"{title}: mean-phase mixture heatmap",
        height=620,
        xaxis_title="bucket",
        yaxis_title="candidate",
    )
    return fig


def _baseline_table_figure(diagnostics: pd.DataFrame, title: str) -> go.Figure:
    baseline = diagnostics[diagnostics["candidate_type"].str.startswith("baseline")].copy()
    if baseline.empty:
        baseline = diagnostics.head(0).copy()
    columns = [
        "candidate_name",
        "candidate_type",
        "max_epoch",
        "mean_tv_to_proportional",
        "phase_tv",
        "support_min",
        "max_weight",
    ]
    fig = go.Figure(
        go.Table(
            header={"values": columns},
            cells={"values": [baseline[column].tolist() for column in columns]},
        )
    )
    fig.update_layout(title=f"{title}: baseline summary", height=max(260, 90 + 32 * max(len(baseline), 1)))
    return fig


def _figure_html(fig: go.Figure, *, include_plotlyjs: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def write_interactive_sanity_dashboard(
    output_path: Path,
    *,
    scopes: dict[str, pd.DataFrame],
    buckets: tuple[str, ...],
    tokens: np.ndarray,
    proportional: np.ndarray,
    denominator: int,
    title: str,
    metadata: dict[str, object],
    target_budget: float = TARGET_BUDGET,
) -> None:
    """Write an interactive Plotly HTML dashboard for candidate sanity checks."""
    if not scopes:
        raise ValueError("At least one dashboard scope is required")
    sections: list[str] = []
    include_plotlyjs = True
    metadata_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(metadata.items())
    )
    for scope_idx, (scope_name, frame) in enumerate(scopes.items()):
        diagnostics = candidate_diagnostics_frame(
            frame,
            buckets=buckets,
            tokens=tokens,
            proportional=proportional,
            denominator=denominator,
            target_budget=target_budget,
        )
        weights = weights_from_candidate_frame(frame, buckets)
        diagnostics = diagnostics.copy()
        diagnostics["nearest_neighbor_average_phase_tv"] = nearest_neighbor_average_phase_tv(weights)
        figures = [
            _type_counts_figure(diagnostics, scope_name),
            _row_order_figure(diagnostics, scope_name),
            _histogram_figure(diagnostics, scope_name),
            _tv_epoch_scatter_figure(diagnostics, scope_name),
            _bucket_epoch_figure(diagnostics, scope_name),
            _baseline_table_figure(diagnostics, scope_name),
            _heatmap_figure(frame, buckets, scope_name),
        ]
        figure_html = []
        for fig in figures:
            figure_html.append(_figure_html(fig, include_plotlyjs=include_plotlyjs))
            include_plotlyjs = False
        display = "block" if scope_idx == 0 else "none"
        sections.append(
            f'<section class="scope-section" data-scope="{html.escape(scope_name)}" style="display:{display}">'
            f"<h2>{html.escape(scope_name)}</h2>" + "\n".join(figure_html) + "</section>"
        )
    buttons = "\n".join(
        f'<button type="button" onclick="showScope({json.dumps(scope_name)})">{html.escape(scope_name)}</button>'
        for scope_name in scopes
    )
    output_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17233c; }}
    h1, h2 {{ color: #13294b; }}
    button {{
      margin-right: 8px;
      padding: 8px 12px;
      border: 1px solid #b7c4d6;
      border-radius: 6px;
      background: #f7fafc;
      cursor: pointer;
    }}
    table.metadata {{ border-collapse: collapse; margin: 12px 0 24px; }}
    table.metadata th, table.metadata td {{ border: 1px solid #d7dee8; padding: 6px 10px; text-align: left; }}
  </style>
  <script>
    function showScope(scopeName) {{
      for (const section of document.querySelectorAll('.scope-section')) {{
        section.style.display = section.dataset.scope === scopeName ? 'block' : 'none';
      }}
    }}
  </script>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Use the buttons to toggle diagnostics between D-optimal-only and full-swarm views when both are available.</p>
  <div>{buttons}</div>
  <table class="metadata"><tbody>{metadata_rows}</tbody></table>
  {''.join(sections)}
</body>
</html>
""",
        encoding="utf-8",
    )


def write_candidate_csv(
    *,
    output_path: Path,
    variant: str,
    weights: np.ndarray,
    selected: np.ndarray,
    buckets: tuple[str, ...],
) -> None:
    """Write collaborator-shaped candidate CSV for one variant."""
    rows: list[dict[str, float | str]] = []
    for candidate_idx, pool_idx in enumerate(selected):
        row: dict[str, float | str] = {
            "candidate_name": f"{variant}_{candidate_idx:06d}",
            "candidate_type": variant,
        }
        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            for bucket_idx, bucket in enumerate(buckets):
                row[f"{phase_name}/{bucket}"] = float(weights[pool_idx, phase_idx, bucket_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_manifest_csv(
    *,
    output_path: Path,
    variant: str,
    weights: np.ndarray,
    selected: np.ndarray,
    epochs: np.ndarray,
    proportional: np.ndarray,
    source_indices: np.ndarray,
) -> None:
    """Write per-candidate diagnostics for one variant."""
    rows: list[dict[str, float | int | str]] = []
    for candidate_idx, pool_idx in enumerate(selected):
        weight = weights[pool_idx]
        epoch = epochs[pool_idx]
        rows.append(
            {
                "candidate_name": f"{variant}_{candidate_idx:06d}",
                "candidate_type": variant,
                "source_pool_index": int(source_indices[pool_idx]),
                "filtered_pool_index": int(pool_idx),
                "tv_to_proportional_phase0": float(np.abs(weight[0] - proportional).sum() / 2.0),
                "tv_to_proportional_phase1": float(np.abs(weight[1] - proportional).sum() / 2.0),
                "phase_tv": float(np.abs(weight[0] - weight[1]).sum() / 2.0),
                "support_phase0": int(np.sum(weight[0] > 0)),
                "support_phase1": int(np.sum(weight[1] > 0)),
                "max_weight": float(weight.max()),
                "max_epoch": float(epoch.max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def validate_candidate_csv(path: Path, n_select: int, buckets: tuple[str, ...]) -> None:
    """Validate collaborator-shaped candidate CSV invariants."""
    frame = pd.read_csv(path)
    expected_columns = ["candidate_name", "candidate_type"] + [
        f"{phase}/{bucket}" for phase in PHASE_NAMES for bucket in buckets
    ]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Unexpected candidate CSV columns for {path}")
    if len(frame) != n_select:
        raise ValueError(f"Expected {n_select} rows in {path}, got {len(frame)}")
    if frame["candidate_name"].duplicated().any():
        raise ValueError(f"Duplicate candidate names in {path}")
    for phase in PHASE_NAMES:
        columns = [f"{phase}/{bucket}" for bucket in buckets]
        values = frame[columns].to_numpy(dtype=float)
        if np.any(values < 0):
            raise ValueError(f"Negative weights in {path} for {phase}")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError(f"Phase weights do not sum to 1 in {path} for {phase}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--n-select", type=int, default=DEFAULT_N_SELECT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--proposal", choices=PROPOSAL_TYPES, default=DEFAULT_PROPOSAL)
    parser.add_argument("--proposal-sigmas", default=DEFAULT_PROPOSAL_SIGMAS)
    parser.add_argument("--min-weight", type=float, default=DEFAULT_MIN_WEIGHT)
    parser.add_argument("--min-config-distance", type=float, default=DEFAULT_MIN_CONFIG_DISTANCE)
    parser.add_argument("--max-epoch-cap", type=float, default=DEFAULT_MAX_EPOCH_CAP)
    parser.add_argument("--no-max-epoch-cap", action="store_true")
    parser.add_argument("--variant", action="append", choices=tuple(FEATURE_BUILDERS), help="Feature variant to run")
    parser.add_argument(
        "--selection-strategy",
        action="append",
        help="Candidate selection strategy to run for each feature variant",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_arg_parser().parse_args()


def main() -> None:
    """Run feature-variant design diagnostics."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = tuple(args.variant) if args.variant else tuple(FEATURE_BUILDERS)
    selection_strategies = tuple(args.selection_strategy) if args.selection_strategy else DEFAULT_SELECTION_STRATEGIES
    max_epoch_cap = None if args.no_max_epoch_cap else args.max_epoch_cap
    proposal_sigmas = parse_float_schedule(args.proposal_sigmas)
    summaries: list[dict[str, float | int | str]] = []
    random_rng = np.random.default_rng(args.seed + 1)
    stage_total = 5 + len(variants) * (1 + 6 * len(selection_strategies))
    with tqdm(total=stage_total, desc="production-swarm-design", unit="stage") as stage_bar:
        start_stage(stage_bar, f"Loading bucket table from {args.source_url}")
        bucket_table = load_bucket_table(args.source_url)
        n_domains = len(bucket_table.buckets)
        proportional = bucket_table.tokens / bucket_table.tokens.sum()
        stage_bar.update(1)

        proposal_detail = (
            f"min_weight={args.min_weight}"
            if args.proposal == "uniform-simplex"
            else f"proposal_sigmas={proposal_sigmas}"
        )
        start_stage(
            stage_bar,
            f"Generating {args.proposal} pool: pool_size={args.pool_size}, n_phases={len(PHASE_NAMES)}, "
            f"n_partitions={n_domains}, {proposal_detail}",
        )
        weights = build_proposal_pool(
            proposal=args.proposal,
            n_points=args.pool_size,
            n_phases=len(PHASE_NAMES),
            n_domains=n_domains,
            seed=args.seed,
            min_weight=args.min_weight,
            proportional=proportional,
            sigma_values=proposal_sigmas,
        )
        stage_bar.update(1)

        start_stage(
            stage_bar,
            f"Applying min average phase-TV rejection: min_config_distance={args.min_config_distance}",
        )
        weights, source_indices = filter_min_average_phase_tv(weights, args.min_config_distance)
        multipliers = PHASE_FRACTIONS[:, None] * TARGET_BUDGET / bucket_table.tokens[None, :]
        epochs = weights * multipliers[None, :, :]
        pool_size_after_distance_rejection = len(weights)
        stage_bar.update(1)

        start_stage(
            stage_bar,
            f"Distance rejection kept {pool_size_after_distance_rejection}/{args.pool_size}; "
            f"applying max_epoch_cap={max_epoch_cap}",
        )
        weights, epochs, source_indices = filter_max_epoch_cap(
            weights,
            epochs,
            source_indices,
            max_epoch_cap=max_epoch_cap,
        )
        progress(f"Feasibility filtering kept {len(weights)}/{pool_size_after_distance_rejection}")
        if len(weights) < args.n_select:
            raise ValueError(
                f"Only {len(weights)} candidates remain after filters; need at least {args.n_select} "
                f"(min_config_distance={args.min_config_distance}, max_epoch_cap={args.max_epoch_cap})"
            )
        stage_bar.update(1)

        for variant in variants:
            start_stage(stage_bar, f"Building features for {variant}")
            features = FEATURE_BUILDERS[variant](weights, epochs, proportional)
            stage_bar.update(1)

            for selection_strategy in selection_strategies:
                output_name = selection_output_name(variant, selection_strategy)
                variant_dir = output_dir / output_name
                variant_dir.mkdir(exist_ok=True)

                start_stage(
                    stage_bar,
                    f"Selecting {args.n_select} candidates for {output_name} from {len(weights)} survivors",
                )
                selected = select_candidates(
                    features,
                    n_select=args.n_select,
                    epochs=epochs,
                    strategy=selection_strategy,
                    max_epoch_cap=max_epoch_cap,
                )
                stage_bar.update(1)

                start_stage(stage_bar, f"Computing selected logdet for {output_name}")
                selected_logdet = dual_logdet(features, selected)
                stage_bar.update(1)

                start_stage(stage_bar, f"Computing 20 random logdet baselines for {output_name}")
                random_logdets = [
                    dual_logdet(features, random_rng.choice(len(weights), size=args.n_select, replace=False))
                    for _ in tqdm(range(20), desc=f"{output_name} random logdet", unit="draw", leave=False)
                ]
                stage_bar.update(1)

                start_stage(stage_bar, f"Writing candidate and manifest CSVs for {output_name}")
                summary = schedule_summary(
                    name=output_name,
                    weights=weights,
                    selected=selected,
                    epochs=epochs,
                    proportional=proportional,
                )
                selected_max_epoch = epochs[selected].max(axis=(1, 2))
                band_counts = [
                    int(np.sum((selected_max_epoch >= lower) & (selected_max_epoch < upper)))
                    for lower, upper in pairwise(BANDED_BALANCED_MAX_EPOCH_EDGES[:-1])
                ]
                band_counts.append(
                    int(
                        np.sum(
                            (selected_max_epoch >= BANDED_BALANCED_MAX_EPOCH_EDGES[-2])
                            & (selected_max_epoch <= BANDED_BALANCED_MAX_EPOCH_EDGES[-1])
                        )
                    )
                )
                summary.update(
                    {
                        "feature_variant": variant,
                        "selection_strategy": selection_strategy,
                        "feature_dim": int(features.matrix.shape[1]),
                        "pool_size_before_distance_rejection": int(args.pool_size),
                        "pool_size_after_distance_rejection": int(pool_size_after_distance_rejection),
                        "distance_rejected_count": int(args.pool_size - pool_size_after_distance_rejection),
                        "pool_size_after_max_epoch_cap": len(weights),
                        "max_epoch_cap_rejected_count": int(pool_size_after_distance_rejection - len(weights)),
                        "min_config_distance": float(args.min_config_distance),
                        "max_epoch_cap": float(max_epoch_cap) if max_epoch_cap is not None else None,
                        "proposal": args.proposal,
                        "proposal_sigmas": ",".join(str(value) for value in proposal_sigmas),
                        "selected_logdet": float(selected_logdet),
                        "random_logdet_mean_20": float(np.mean(random_logdets)),
                        "random_logdet_std_20": float(np.std(random_logdets)),
                        "selected_minus_random_mean_logdet": float(selected_logdet - np.mean(random_logdets)),
                        "max_epoch_band_0_25_count": band_counts[0],
                        "max_epoch_band_25_50_count": band_counts[1],
                        "max_epoch_band_50_75_count": band_counts[2],
                        "max_epoch_band_75_90_count": band_counts[3],
                        "max_epoch_band_90_100_count": band_counts[4],
                    }
                )
                summaries.append(summary)
                write_candidate_csv(
                    output_path=variant_dir / "candidate_mixtures.csv",
                    variant=output_name,
                    weights=weights,
                    selected=selected,
                    buckets=bucket_table.buckets,
                )
                write_manifest_csv(
                    output_path=variant_dir / "candidate_manifest.csv",
                    variant=output_name,
                    weights=weights,
                    selected=selected,
                    epochs=epochs,
                    proportional=proportional,
                    source_indices=source_indices,
                )
                stage_bar.update(1)

                start_stage(stage_bar, f"Validating candidate CSV for {output_name}")
                candidate_csv = variant_dir / "candidate_mixtures.csv"
                validate_candidate_csv(candidate_csv, args.n_select, bucket_table.buckets)
                candidate_frame = pd.read_csv(candidate_csv)
                stage_bar.update(1)

                start_stage(stage_bar, f"Writing interactive sanity dashboard for {output_name}")
                write_interactive_sanity_dashboard(
                    variant_dir / "sanity_dashboard.html",
                    scopes={"D-optimal only": candidate_frame},
                    buckets=bucket_table.buckets,
                    tokens=bucket_table.tokens,
                    proportional=proportional,
                    denominator=MIXTURE_QUANTUM_DENOMINATOR,
                    title=f"Production Swarm 167p Sanity Dashboard: {output_name}",
                    metadata={
                        "variant": variant,
                        "selection_strategy": selection_strategy,
                        "pool_size": args.pool_size,
                        "pool_size_after_max_epoch_cap": len(weights),
                        "n_select": args.n_select,
                        "proposal": args.proposal,
                        "proposal_sigmas": ",".join(str(value) for value in proposal_sigmas),
                        "max_epoch_cap": max_epoch_cap,
                        "selected_logdet": f"{selected_logdet:.6f}",
                        "random_logdet_mean_20": f"{float(np.mean(random_logdets)):.6f}",
                    },
                )
                stage_bar.update(1)

        start_stage(stage_bar, "Writing feature summary")
        summary_frame = pd.DataFrame(summaries).sort_values(["q95_max_epoch", "selected_minus_random_mean_logdet"])
        summary_frame.to_csv(output_dir / "feature_variant_summary.csv", index=False)
        payload = {
            "source_url": args.source_url,
            "source_sha256": bucket_table.source_sha256,
            "n_partitions": n_domains,
            "total_tokens": int(bucket_table.tokens.sum()),
            "target_budget": TARGET_BUDGET,
            "phase_fractions": PHASE_FRACTIONS.tolist(),
            "pool_size": args.pool_size,
            "proposal": args.proposal,
            "proposal_sigmas": proposal_sigmas,
            "pool_size_after_distance_rejection": int(pool_size_after_distance_rejection),
            "distance_rejected_count": int(args.pool_size - pool_size_after_distance_rejection),
            "pool_size_after_max_epoch_cap": len(weights),
            "max_epoch_cap_rejected_count": int(pool_size_after_distance_rejection - len(weights)),
            "n_select": args.n_select,
            "seed": args.seed,
            "min_weight": args.min_weight,
            "min_config_distance": args.min_config_distance,
            "max_epoch_cap": max_epoch_cap,
            "variants": summaries,
        }
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        stage_bar.update(1)
    print(summary_frame.to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
