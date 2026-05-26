# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Build an offline candidate library for the factor-DSP constraint dashboard.

This script intentionally precomputes the expensive part of the dashboard: many
candidate mixtures, canonical DSP predictions for the current factor aggregate,
and trust-region diagnostics. The notebook/UI should filter this cache rather
than sampling or scoring hundreds of thousands of candidates interactively.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.neighbors import NearestNeighbors

PROPORTIONAL_RUN_NAME = "baseline_proportional"
DEFAULT_NUM_SOBOL_CANDIDATES = 524_288
DEFAULT_MAX_ALPHA = 0.45
DEFAULT_SEED = 5416
DEFAULT_TOP_CANDIDATE_COUNT = 2_000
DEFAULT_TOP_DOMAIN_COUNT = 8
DEFAULT_LCB_Z = 1.0
DEFAULT_ENDPOINT_KL_PENALTIES = "0,0.02,0.05,0.1,0.2,0.5,1,2,5,10"
DEFAULT_ENDPOINT_RANDOM_START_COUNT = 16
DEFAULT_ENDPOINT_RANDOM_START_ALPHA = 2.0
DEFAULT_ENDPOINT_MAXITER = 1000
DEFAULT_ENDPOINT_MAX_WEIGHT_PENALTY = 2.0
DEFAULT_ENDPOINT_MAX_WEIGHT_TARGET = 0.50
DEFAULT_ENDPOINT_PATH_T_VALUES = "0,0.05,0.1,0.2,0.25,0.35,0.5,0.65,0.75,0.9,1"
EPS = 1e-12
PHASES = ("phase_0", "phase_1")
REPRO_ROOT = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "reference_outputs/collaborator_grug_v4_aggregate_repro_20260525"
)
DEFAULT_MODEL_JSON = REPRO_ROOT / "canonical_dsp_sent_zip/model.json"
DEFAULT_MODEL_SUMMARY_JSON = REPRO_ROOT / "canonical_dsp_sent_zip/summary.json"
DEFAULT_MIXTURE_WEIGHTS_CSV = REPRO_ROOT / "canonical_dsp_sent_zip/mixture_weights.csv"
DEFAULT_SIGNAL_CSV = REPRO_ROOT / "sent_zip_input/raw_metric_matrix_300m/raw_metric_matrix_300m.csv"
DEFAULT_OUTPUT_DIR = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "reference_outputs/factor_dsp_candidate_library_y_factor_20260526"
)


@dataclass(frozen=True)
class EffectiveExposureDSPModel:
    """Canonical effective-exposure DSP model for the factor aggregate."""

    domain_names: tuple[str, ...]
    rho: np.ndarray
    tau: np.ndarray
    gamma: np.ndarray
    intercept: float
    benefit_coef: np.ndarray
    penalty_coef: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--model-summary-json", type=Path, default=DEFAULT_MODEL_SUMMARY_JSON)
    parser.add_argument("--mixture-weights-csv", type=Path, default=DEFAULT_MIXTURE_WEIGHTS_CSV)
    parser.add_argument("--signal-csv", type=Path, default=DEFAULT_SIGNAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-sobol-candidates", type=int, default=DEFAULT_NUM_SOBOL_CANDIDATES)
    parser.add_argument("--max-alpha", type=float, default=DEFAULT_MAX_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lcb-z", type=float, default=DEFAULT_LCB_Z)
    parser.add_argument("--top-candidate-count", type=int, default=DEFAULT_TOP_CANDIDATE_COUNT)
    parser.add_argument("--top-domain-count", type=int, default=DEFAULT_TOP_DOMAIN_COUNT)
    parser.add_argument("--skip-full-weights", action="store_true")
    parser.add_argument("--endpoint-discovery", action="store_true")
    parser.add_argument("--endpoint-only", action="store_true")
    parser.add_argument("--endpoint-output-dir", type=Path, default=None)
    parser.add_argument("--endpoint-kl-penalties", default=DEFAULT_ENDPOINT_KL_PENALTIES)
    parser.add_argument("--endpoint-random-start-count", type=int, default=DEFAULT_ENDPOINT_RANDOM_START_COUNT)
    parser.add_argument("--endpoint-random-start-alpha", type=float, default=DEFAULT_ENDPOINT_RANDOM_START_ALPHA)
    parser.add_argument("--endpoint-observed-start-count", type=int, default=16)
    parser.add_argument("--endpoint-maxiter", type=int, default=DEFAULT_ENDPOINT_MAXITER)
    parser.add_argument("--endpoint-max-weight-penalty", type=float, default=DEFAULT_ENDPOINT_MAX_WEIGHT_PENALTY)
    parser.add_argument("--endpoint-max-weight-target", type=float, default=DEFAULT_ENDPOINT_MAX_WEIGHT_TARGET)
    parser.add_argument("--endpoint-path-t-values", default=DEFAULT_ENDPOINT_PATH_T_VALUES)
    return parser.parse_args()


def load_model(path: Path) -> EffectiveExposureDSPModel:
    data = json.loads(path.read_text())
    if data.get("variant") != "dsp_effective_exposure_penalty_nnls":
        raise ValueError(f"expected canonical effective-exposure DSP model, got {data.get('variant')}")
    params = data["params"]
    domain_names = tuple(str(domain) for domain in data["domain_names"])
    return EffectiveExposureDSPModel(
        domain_names=domain_names,
        rho=np.asarray(params["rho"], dtype=np.float64),
        tau=np.asarray(params["tau"], dtype=np.float64),
        gamma=np.asarray(params["gamma"], dtype=np.float64),
        intercept=float(data["intercept"]),
        benefit_coef=np.asarray(data["benefit_coef"], dtype=np.float64),
        penalty_coef=np.asarray(data["penalty_coef"], dtype=np.float64),
        c0=np.asarray(data["c0"], dtype=np.float64),
        c1=np.asarray(data["c1"], dtype=np.float64),
        metrics=dict(data.get("metrics", {})),
    )


def softplus(values: np.ndarray) -> np.ndarray:
    """Stable softplus for vectorized penalty features."""
    return np.logaddexp(0.0, values)


def predict_y_factor(model: EffectiveExposureDSPModel, weights: np.ndarray, *, chunk_size: int = 65_536) -> np.ndarray:
    """Predict higher-is-better factor utility from phase weights."""
    if weights.ndim != 3 or weights.shape[1:] != (2, len(model.domain_names)):
        raise ValueError(f"weights must have shape (n, 2, {len(model.domain_names)}), got {weights.shape}")
    predicted = np.empty(weights.shape[0], dtype=np.float64)
    for start in range(0, weights.shape[0], chunk_size):
        end = min(start + chunk_size, weights.shape[0])
        chunk = weights[start:end].astype(np.float64, copy=False)
        phase0_exposure = chunk[:, 0, :] * model.c0
        phase1_exposure = chunk[:, 1, :] * model.c1
        effective_exposure = phase0_exposure + model.gamma * phase1_exposure
        signal = -np.expm1(-model.rho * effective_exposure)
        penalty = softplus(np.log1p(effective_exposure) - model.tau) ** 2
        loss = model.intercept - signal @ model.benefit_coef + penalty @ model.penalty_coef
        predicted[start:end] = -loss
    return predicted


def dsp_value_and_weight_grad(model: EffectiveExposureDSPModel, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Return y_factor prediction and gradient with respect to two-phase weights."""
    if weights.shape != (2, len(model.domain_names)):
        raise ValueError(f"weights must have shape (2, {len(model.domain_names)}), got {weights.shape}")
    weights64 = weights.astype(np.float64, copy=False)
    phase0_exposure = weights64[0] * model.c0
    phase1_exposure = weights64[1] * model.c1
    effective_exposure = phase0_exposure + model.gamma * phase1_exposure
    signal = -np.expm1(-model.rho * effective_exposure)
    penalty_arg = np.log1p(effective_exposure) - model.tau
    penalty_softplus = softplus(penalty_arg)
    penalty = penalty_softplus**2
    loss = model.intercept - signal @ model.benefit_coef + penalty @ model.penalty_coef
    sigmoid = 1.0 / (1.0 + np.exp(-penalty_arg))
    d_signal_d_exposure = model.rho * np.exp(-model.rho * effective_exposure)
    d_penalty_d_exposure = 2.0 * penalty_softplus * sigmoid / (1.0 + effective_exposure)
    d_y_d_exposure = model.benefit_coef * d_signal_d_exposure - model.penalty_coef * d_penalty_d_exposure
    grad = np.stack([d_y_d_exposure * model.c0, d_y_d_exposure * model.gamma * model.c1], axis=0)
    return float(-loss), grad


def softmax_phase_logits(logits: np.ndarray) -> np.ndarray:
    """Convert phase logits to simplex weights independently per phase."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (phases, domains), got {logits.shape}")
    centered = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(centered)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def logits_from_phase_weights(weights: np.ndarray) -> np.ndarray:
    """Map simplex weights to logits; softmax(logits) equals weights up to clipping."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"weights must have shape (phases, domains), got {weights.shape}")
    return np.log(np.clip(weights, EPS, None))


def kl_to_proportional(weights: np.ndarray, proportional: np.ndarray) -> float:
    """Mean phase KL divergence KL(w_phase || proportional)."""
    weights = np.asarray(weights, dtype=np.float64)
    proportional = np.asarray(proportional, dtype=np.float64)
    if weights.ndim != 2 or proportional.shape != (weights.shape[1],):
        raise ValueError("weights must be (phases, domains) and proportional must match domains")
    clipped = np.clip(weights, EPS, None)
    prop = np.clip(proportional, EPS, None)
    return float(np.mean(np.sum(clipped * (np.log(clipped) - np.log(prop)), axis=1)))


def max_weight_excess_penalty(weights: np.ndarray, target: float) -> float:
    """Squared excess above the desired max domain weight."""
    if target <= 0.0:
        raise ValueError("target must be positive")
    excess = np.maximum(weights - target, 0.0)
    return float(np.sum(excess**2))


def regularized_endpoint_value_and_grad(
    *,
    flat_logits: np.ndarray,
    model: EffectiveExposureDSPModel,
    proportional: np.ndarray,
    kl_penalty: float,
    max_weight_penalty: float,
    max_weight_target: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return negative regularized utility and gradient for L-BFGS-B."""
    logits = flat_logits.reshape(2, len(model.domain_names))
    weights = softmax_phase_logits(logits)
    predicted_y, grad_weights = dsp_value_and_weight_grad(model, weights)

    clipped = np.clip(weights, EPS, None)
    proportional_clipped = np.clip(proportional, EPS, None)
    kl_grad = (np.log(clipped) - np.log(proportional_clipped[None, :]) + 1.0) / 2.0
    excess = np.maximum(weights - max_weight_target, 0.0)
    max_penalty_grad = 2.0 * excess
    kl_value = kl_to_proportional(weights, proportional)
    max_penalty_value = max_weight_excess_penalty(weights, max_weight_target)

    utility = predicted_y - kl_penalty * kl_value - max_weight_penalty * max_penalty_value
    grad_utility_weights = grad_weights - kl_penalty * kl_grad - max_weight_penalty * max_penalty_grad
    grad_logits = weights * (grad_utility_weights - np.sum(weights * grad_utility_weights, axis=1, keepdims=True))
    components = {
        "predicted_y_factor": predicted_y,
        "kl_to_proportional": kl_value,
        "max_weight_excess_penalty": max_penalty_value,
        "regularized_utility": utility,
    }
    return -utility, -grad_logits.ravel(), components


def generate_sobol_logit_candidates(
    *,
    proportional: np.ndarray,
    num_candidates: int,
    max_alpha: float,
    seed: int,
) -> np.ndarray:
    """Generate two-phase logit-tilt candidates around proportional.

    Directions are centered and unit-normalized in local Fisher/KL coordinates:
    sum_i p_i v_i = 0 and sum_i p_i v_i^2 = 1. Each candidate samples one
    shared radius alpha in [0, max_alpha] and independent phase directions.
    """
    if num_candidates < 0:
        raise ValueError("num_candidates must be nonnegative")
    if max_alpha <= 0.0:
        raise ValueError("max_alpha must be positive")
    proportional = np.asarray(proportional, dtype=np.float64)
    if proportional.ndim != 1 or not np.isclose(proportional.sum(), 1.0, atol=1e-8):
        raise ValueError("proportional must be a simplex vector")
    if np.any(proportional <= 0.0):
        raise ValueError("proportional must have full support for logit tilts")
    if num_candidates == 0:
        return np.empty((0, 2, proportional.size), dtype=np.float32)

    dimension = 2 * proportional.size + 1
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    if num_candidates > 0 and num_candidates & (num_candidates - 1) == 0:
        samples = engine.random_base2(m=int(math.log2(num_candidates)))
    else:
        samples = engine.random(num_candidates)
    samples = np.clip(samples, 1e-7, 1.0 - 1e-7)
    radii = max_alpha * np.sqrt(samples[:, -1])
    directions = norm.ppf(samples[:, :-1]).reshape(num_candidates, 2, proportional.size)
    weighted_mean = np.sum(directions * proportional[None, None, :], axis=2, keepdims=True)
    directions = directions - weighted_mean
    weighted_norm = np.sqrt(np.sum(proportional[None, None, :] * directions**2, axis=2, keepdims=True))
    directions = directions / np.maximum(weighted_norm, EPS)

    logits = np.log(proportional)[None, None, :] + radii[:, None, None] * directions
    logits = logits - logits.max(axis=2, keepdims=True)
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=2, keepdims=True)
    return weights.astype(np.float32)


def average_phase_tv(weights: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return mean total variation across phase 0 and phase 1."""
    if weights.ndim != 3 or weights.shape[1] != 2:
        raise ValueError("weights must have shape (n, 2, domains)")
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (weights.shape[2],):
        raise ValueError("reference domain count does not match weights")
    phase_l1 = np.abs(weights.astype(np.float64, copy=False) - reference[None, None, :]).sum(axis=2)
    return 0.5 * phase_l1.mean(axis=1)


def phase_tv(weights: np.ndarray) -> np.ndarray:
    """Return TV distance between phase-0 and phase-1 weights for each candidate."""
    return 0.5 * np.abs(weights[:, 0, :] - weights[:, 1, :]).sum(axis=1)


def phase_entropy(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights.astype(np.float64, copy=False), EPS, None)
    return -(clipped * np.log(clipped)).sum(axis=2)


def nearest_observed_average_phase_tv(
    *,
    weights: np.ndarray,
    observed_ids: np.ndarray,
    observed_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact nearest observed row under average phase TV."""
    if len(observed_ids) != observed_weights.shape[0]:
        raise ValueError("observed_ids and observed_weights length mismatch")
    flat_observed = observed_weights.reshape(observed_weights.shape[0], -1).astype(np.float32, copy=False)
    flat_weights = weights.reshape(weights.shape[0], -1).astype(np.float32, copy=False)
    neighbors = NearestNeighbors(n_neighbors=1, metric="manhattan", algorithm="brute")
    neighbors.fit(flat_observed)
    distances, indices = neighbors.kneighbors(flat_weights, return_distance=True)
    return observed_ids[indices[:, 0]], distances[:, 0] / 4.0


def candidate_diagnostics(
    *,
    candidate_ids: np.ndarray,
    candidate_sources: np.ndarray,
    weights: np.ndarray,
    proportional: np.ndarray,
    predicted_y_factor: np.ndarray,
    proportional_predicted_y_factor: float,
    observed_ids: np.ndarray,
    observed_weights: np.ndarray,
    oof_rmse: float,
    lcb_z: float,
    max_nearest_observed_tv: float = 0.45,
    max_phase_weight: float = 0.50,
    min_phase_support: int = 5,
) -> pd.DataFrame:
    """Build one-row-per-candidate diagnostics for dashboard filtering."""
    if not (len(candidate_ids) == len(candidate_sources) == weights.shape[0] == len(predicted_y_factor)):
        raise ValueError("candidate metadata and prediction lengths do not match")
    nearest_ids, nearest_tvs = nearest_observed_average_phase_tv(
        weights=weights,
        observed_ids=observed_ids,
        observed_weights=observed_weights,
    )
    entropies = phase_entropy(weights)
    supports = (weights > 1e-3).sum(axis=2)
    gains = predicted_y_factor - proportional_predicted_y_factor
    max_weights = weights.max(axis=2)
    frame = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "candidate_source": candidate_sources,
            "predicted_y_factor": predicted_y_factor,
            "predicted_y_factor_gain_vs_proportional": gains,
            "predicted_y_factor_gain_lcb": gains - lcb_z * oof_rmse,
            "average_phase_tv_to_proportional": average_phase_tv(weights, proportional),
            "phase_tv": phase_tv(weights),
            "nearest_observed_run": nearest_ids,
            "nearest_observed_tv": nearest_tvs,
            "phase_0_max_weight": max_weights[:, 0],
            "phase_1_max_weight": max_weights[:, 1],
            "max_phase_weight": max_weights.max(axis=1),
            "phase_0_support_gt_1e3": supports[:, 0],
            "phase_1_support_gt_1e3": supports[:, 1],
            "min_phase_support_gt_1e3": supports.min(axis=1),
            "phase_0_entropy": entropies[:, 0],
            "phase_1_entropy": entropies[:, 1],
            "mean_phase_entropy": entropies.mean(axis=1),
            "phase_0_effective_support": np.exp(entropies[:, 0]),
            "phase_1_effective_support": np.exp(entropies[:, 1]),
            "mean_phase_effective_support": np.exp(entropies).mean(axis=1),
            "oof_rmse_used": oof_rmse,
            "lcb_z": lcb_z,
        }
    )
    frame["passes_basic_dashboard_gate"] = (
        (frame["nearest_observed_tv"] <= max_nearest_observed_tv)
        & (frame["max_phase_weight"] <= max_phase_weight)
        & (frame["min_phase_support_gt_1e3"] >= min_phase_support)
    )
    return frame


def read_signal_weights(path: Path, domain_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = pd.read_csv(path, low_memory=False)
    if "run_name" not in signal.columns:
        raise ValueError(f"signal CSV missing run_name: {path}")
    phase_columns = [[f"{phase}_{domain}" for domain in domain_names] for phase in PHASES]
    missing = [column for columns in phase_columns for column in columns if column not in signal.columns]
    if missing:
        raise ValueError(f"signal CSV missing phase columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    ids = signal["run_name"].astype(str).to_numpy()
    weights = np.stack([signal.loc[:, columns].to_numpy(dtype=np.float32) for columns in phase_columns], axis=1)
    row_ids = signal["run_id"].astype(str).to_numpy() if "run_id" in signal.columns else ids
    if not np.allclose(weights.sum(axis=2), 1.0, atol=1e-5):
        raise ValueError("observed signal weights do not sum to 1 per phase")
    return ids, row_ids, weights


def proportional_from_observed(
    *,
    observed_ids: np.ndarray,
    observed_weights: np.ndarray,
    run_name: str = PROPORTIONAL_RUN_NAME,
) -> np.ndarray:
    matches = np.flatnonzero(observed_ids == run_name)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {run_name} row, found {len(matches)}")
    weights = observed_weights[matches[0]]
    if not np.allclose(weights[0], weights[1], atol=1e-7):
        raise ValueError("proportional baseline has different phase weights; update this builder before use")
    return weights[0].astype(np.float64)


def read_named_weights(path: Path, domain_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"label", "domain", "phase_0_weight", "phase_1_weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"named mixture CSV missing columns: {sorted(missing)}")
    labels = sorted(frame["label"].astype(str).unique())
    weights = np.empty((len(labels), 2, len(domain_names)), dtype=np.float32)
    for label_index, label in enumerate(labels):
        rows = frame.loc[frame["label"].astype(str).eq(label)].set_index("domain")
        missing_domains = [domain for domain in domain_names if domain not in rows.index]
        if missing_domains:
            raise ValueError(f"named mixture {label} missing domains: {missing_domains[:5]}")
        weights[label_index, 0, :] = rows.loc[list(domain_names), "phase_0_weight"].to_numpy(dtype=np.float32)
        weights[label_index, 1, :] = rows.loc[list(domain_names), "phase_1_weight"].to_numpy(dtype=np.float32)
    if not np.allclose(weights.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("named mixture weights do not sum to 1")
    return np.asarray([f"named_{label}" for label in labels], dtype=object), weights


def named_interpolation_candidates(
    *,
    named_ids: np.ndarray,
    named_weights: np.ndarray,
    proportional: np.ndarray,
    t_values: tuple[float, ...] = (0.05, 0.10, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    proportional_weights = np.stack([proportional, proportional], axis=0).astype(np.float32)
    ids: list[str] = []
    weights: list[np.ndarray] = []
    for candidate_id, target in zip(named_ids, named_weights, strict=True):
        if str(candidate_id) == "named_proportional":
            continue
        short_id = str(candidate_id).removeprefix("named_")
        for t in t_values:
            ids.append(f"path_{short_id}_t{str(t).replace('.', 'p')}")
            weights.append(((1.0 - t) * proportional_weights + t * target).astype(np.float32))
    if not weights:
        return np.empty(0, dtype=object), np.empty((0, 2, len(proportional)), dtype=np.float32)
    return np.asarray(ids, dtype=object), np.stack(weights, axis=0)


def observed_candidates(observed_ids: np.ndarray, observed_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([f"observed_{candidate_id}" for candidate_id in observed_ids], dtype=object)
    return ids, observed_weights.astype(np.float32, copy=False)


def combine_candidates(parts: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for source, part_ids, part_weights in parts:
        if len(part_ids) != part_weights.shape[0]:
            raise ValueError(f"candidate id/weight length mismatch for {source}")
        if len(part_ids) == 0:
            continue
        ids.append(part_ids)
        sources.append(np.full(len(part_ids), source, dtype=object))
        weights.append(part_weights)
    combined_ids = np.concatenate(ids)
    if len(set(combined_ids.tolist())) != len(combined_ids):
        duplicates = pd.Series(combined_ids).loc[pd.Series(combined_ids).duplicated()].unique().tolist()
        raise ValueError(f"duplicate candidate ids: {duplicates[:10]}")
    return combined_ids, np.concatenate(sources), np.concatenate(weights, axis=0)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float schedule."""
    entries = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not entries:
        raise ValueError("expected at least one float value")
    return entries


def discover_regularized_endpoints(
    *,
    model: EffectiveExposureDSPModel,
    proportional: np.ndarray,
    start_weights: np.ndarray,
    kl_penalties: tuple[float, ...],
    max_weight_penalty: float,
    max_weight_target: float,
    random_start_count: int,
    seed: int,
    maxiter: int,
    random_start_alpha: float = DEFAULT_ENDPOINT_RANDOM_START_ALPHA,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Optimize DSP endpoints over phase logits for a KL-regularization path."""
    if start_weights.ndim != 3 or start_weights.shape[1:] != (2, len(model.domain_names)):
        raise ValueError(f"start_weights must have shape (n, 2, {len(model.domain_names)})")
    if random_start_count < 0:
        raise ValueError("random_start_count must be nonnegative")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if any(penalty < 0.0 for penalty in kl_penalties):
        raise ValueError("kl penalties must be nonnegative")
    starts = [start_weights.astype(np.float64, copy=False)]
    if random_start_count:
        starts.append(
            generate_sobol_logit_candidates(
                proportional=proportional,
                num_candidates=random_start_count,
                max_alpha=random_start_alpha,
                seed=seed + 10_003,
            ).astype(np.float64)
        )
    all_starts = np.concatenate(starts, axis=0)

    records: list[dict[str, object]] = []
    endpoint_weights: list[np.ndarray] = []
    for kl_penalty in kl_penalties:
        current_kl_penalty = float(kl_penalty)
        best_result = None
        best_components: dict[str, float] = {}
        best_weights = None
        best_start_index = -1
        for start_index, start in enumerate(all_starts):
            result = minimize(
                lambda flat_logits, kl_penalty=current_kl_penalty: regularized_endpoint_value_and_grad(
                    flat_logits=flat_logits,
                    model=model,
                    proportional=proportional,
                    kl_penalty=kl_penalty,
                    max_weight_penalty=max_weight_penalty,
                    max_weight_target=max_weight_target,
                )[:2],
                logits_from_phase_weights(start).ravel(),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
            )
            _, _, components = regularized_endpoint_value_and_grad(
                flat_logits=result.x,
                model=model,
                proportional=proportional,
                kl_penalty=current_kl_penalty,
                max_weight_penalty=max_weight_penalty,
                max_weight_target=max_weight_target,
            )
            if best_result is None or components["regularized_utility"] > best_components["regularized_utility"]:
                best_result = result
                best_components = components
                best_weights = softmax_phase_logits(result.x.reshape(2, len(model.domain_names)))
                best_start_index = start_index
        if best_result is None or best_weights is None:
            raise RuntimeError("endpoint optimization produced no result")
        endpoint_index = len(endpoint_weights)
        endpoint_id = f"endpoint_kl{str(kl_penalty).replace('.', 'p')}_mw{str(max_weight_penalty).replace('.', 'p')}"
        endpoint_weights.append(best_weights.astype(np.float32))
        records.append(
            {
                "endpoint_index": endpoint_index,
                "candidate_id": endpoint_id,
                "candidate_source": "dsp_endpoint_discovery",
                "kl_penalty": float(kl_penalty),
                "max_weight_penalty": float(max_weight_penalty),
                "max_weight_target": float(max_weight_target),
                "best_start_index": best_start_index,
                "optimization_success": bool(best_result.success),
                "optimization_message": str(best_result.message),
                "optimization_iterations": int(best_result.nit),
                "optimization_function_evals": int(best_result.nfev),
                **best_components,
                "average_phase_tv_to_proportional": float(average_phase_tv(best_weights[None, :, :], proportional)[0]),
                "phase_tv": float(phase_tv(best_weights[None, :, :])[0]),
                "max_phase_weight": float(best_weights.max()),
                "min_phase_support_gt_1e3": int((best_weights > 1e-3).sum(axis=1).min()),
            }
        )
    return pd.DataFrame.from_records(records), np.stack(endpoint_weights, axis=0)


def endpoint_path_candidates(
    *,
    endpoint_summary: pd.DataFrame,
    endpoint_weights: np.ndarray,
    proportional: np.ndarray,
    t_values: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize proportional-to-endpoint interpolation candidates."""
    proportional_weights = np.stack([proportional, proportional], axis=0).astype(np.float32)
    ids: list[str] = []
    weights: list[np.ndarray] = []
    for row in endpoint_summary.itertuples(index=False):
        endpoint_index = int(row.endpoint_index)
        for t in t_values:
            t_slug = str(t).replace(".", "p")
            ids.append(f"path_{row.candidate_id}_t{t_slug}")
            weights.append(((1.0 - t) * proportional_weights + t * endpoint_weights[endpoint_index]).astype(np.float32))
    return np.asarray(ids, dtype=object), np.stack(weights, axis=0)


def top_domain_delta_table(
    *,
    summary: pd.DataFrame,
    weights: np.ndarray,
    proportional: np.ndarray,
    domain_names: tuple[str, ...],
    top_candidate_count: int,
    top_domain_count: int,
) -> pd.DataFrame:
    """Return compact top positive/negative domain deltas for top candidates."""
    if top_candidate_count <= 0 or top_domain_count <= 0:
        return pd.DataFrame()
    if len(summary) != weights.shape[0]:
        raise ValueError("summary and weights row counts differ")
    selected = summary.nlargest(min(top_candidate_count, len(summary)), "predicted_y_factor_gain_lcb")
    records: list[dict[str, object]] = []
    for row_index, row in selected.iterrows():
        for phase_index, phase in enumerate(PHASES):
            deltas = weights[row_index, phase_index, :].astype(np.float64) - proportional
            positive_indices = np.argsort(-deltas)[:top_domain_count]
            negative_indices = np.argsort(deltas)[:top_domain_count]
            for direction, indices in (("positive", positive_indices), ("negative", negative_indices)):
                for rank, domain_index in enumerate(indices, start=1):
                    records.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "phase": phase,
                            "direction": direction,
                            "delta_rank": rank,
                            "domain": domain_names[domain_index],
                            "weight": float(weights[row_index, phase_index, domain_index]),
                            "proportional_weight": float(proportional[domain_index]),
                            "weight_delta": float(deltas[domain_index]),
                            "relative_weight_ratio": float(
                                weights[row_index, phase_index, domain_index] / proportional[domain_index]
                            ),
                        }
                    )
    return pd.DataFrame.from_records(records)


def candidate_weights_wide(
    *,
    candidate_ids: np.ndarray,
    candidate_sources: np.ndarray,
    weights: np.ndarray,
    domain_names: tuple[str, ...],
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "candidate_id": candidate_ids,
        "candidate_source": candidate_sources,
    }
    for phase_index, phase in enumerate(PHASES):
        for domain_index, domain in enumerate(domain_names):
            data[f"{phase}_{domain}"] = weights[:, phase_index, domain_index].astype(np.float32)
    return pd.DataFrame(data)


def validate_model_against_cached_summary(
    *,
    model: EffectiveExposureDSPModel,
    model_summary_path: Path,
    named_ids: np.ndarray,
    named_weights: np.ndarray,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    if not model_summary_path.exists():
        return {}
    summary = json.loads(model_summary_path.read_text())
    expected = summary.get("pred_y_by_label", {})
    if not expected:
        return {}
    predicted = predict_y_factor(model, named_weights)
    by_label = {
        str(candidate_id).removeprefix("named_"): float(value)
        for candidate_id, value in zip(named_ids, predicted, strict=True)
    }
    mismatches = {
        label: abs(by_label[label] - float(expected_value))
        for label, expected_value in expected.items()
        if label in by_label and abs(by_label[label] - float(expected_value)) > tolerance
    }
    if mismatches:
        raise ValueError(f"canonical DSP prediction mismatch against cached summary: {mismatches}")
    return by_label


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def quantiles(values: np.ndarray) -> dict[str, float]:
    probs = (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)
    labels = ("min", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "max")
    return {label: float(value) for label, value in zip(labels, np.quantile(values, probs), strict=True)}


def build_candidate_library(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading canonical DSP model: {args.model_json}", flush=True)
    model = load_model(args.model_json)
    print(f"Loading observed signal rows: {args.signal_csv}", flush=True)
    observed_ids, observed_run_ids, observed_weights = read_signal_weights(args.signal_csv, model.domain_names)
    proportional = proportional_from_observed(observed_ids=observed_ids, observed_weights=observed_weights)
    print(f"Observed rows: {len(observed_ids)}; domains: {len(model.domain_names)}", flush=True)

    print(f"Loading named canonical-DSP mixtures: {args.mixture_weights_csv}", flush=True)
    named_ids, named_weights = read_named_weights(args.mixture_weights_csv, model.domain_names)
    cached_named_predictions = validate_model_against_cached_summary(
        model=model,
        model_summary_path=args.model_summary_json,
        named_ids=named_ids,
        named_weights=named_weights,
    )
    print("Generating Sobol-logit trust-region candidates...", flush=True)
    sobol_weights = generate_sobol_logit_candidates(
        proportional=proportional,
        num_candidates=args.num_sobol_candidates,
        max_alpha=args.max_alpha,
        seed=args.seed,
    )
    sobol_ids = np.asarray([f"sobol_{index:06d}" for index in range(args.num_sobol_candidates)], dtype=object)

    interpolation_ids, interpolation_weights = named_interpolation_candidates(
        named_ids=named_ids,
        named_weights=named_weights,
        proportional=proportional,
    )
    observed_candidate_ids, observed_candidate_weights = observed_candidates(observed_ids, observed_weights)
    candidate_ids, candidate_sources, weights = combine_candidates(
        [
            ("sobol_logit_trust", sobol_ids, sobol_weights),
            ("canonical_dsp_named", named_ids, named_weights),
            ("canonical_dsp_path", interpolation_ids, interpolation_weights),
            ("observed_signal", observed_candidate_ids, observed_candidate_weights),
        ]
    )
    print(f"Scoring {len(candidate_ids)} candidates with canonical DSP...", flush=True)
    predicted = predict_y_factor(model, weights)
    proportional_predicted = float(
        predict_y_factor(model, np.stack([proportional, proportional], axis=0)[None, :, :])[0]
    )
    oof_rmse = float(model.metrics.get("oof_rmse", np.nan))

    print("Computing candidate diagnostics and nearest-observed TV...", flush=True)
    summary = candidate_diagnostics(
        candidate_ids=candidate_ids,
        candidate_sources=candidate_sources,
        weights=weights,
        proportional=proportional,
        predicted_y_factor=predicted,
        proportional_predicted_y_factor=proportional_predicted,
        observed_ids=observed_ids,
        observed_weights=observed_weights,
        oof_rmse=oof_rmse,
        lcb_z=args.lcb_z,
    )
    summary = summary.sort_values("predicted_y_factor_gain_lcb", ascending=False).reset_index(drop=True)
    row_index_by_candidate = pd.Series(np.arange(len(candidate_ids)), index=candidate_ids)
    sorted_weight_indices = row_index_by_candidate.loc[summary["candidate_id"]].to_numpy()
    sorted_weights = weights[sorted_weight_indices]

    print("Writing cache files...", flush=True)
    summary_csv = args.output_dir / "candidate_summary.csv"
    summary_parquet = args.output_dir / "candidate_summary.parquet"
    top_deltas_parquet = args.output_dir / "candidate_top_domain_deltas.parquet"
    summary.to_csv(summary_csv, index=False)
    summary.to_parquet(summary_parquet, index=False)
    top_deltas = top_domain_delta_table(
        summary=summary,
        weights=sorted_weights,
        proportional=proportional,
        domain_names=model.domain_names,
        top_candidate_count=args.top_candidate_count,
        top_domain_count=args.top_domain_count,
    )
    top_deltas.to_parquet(top_deltas_parquet, index=False)

    full_weights_parquet = None
    if not args.skip_full_weights:
        full_weights_parquet = args.output_dir / "candidate_weights_wide.parquet"
        candidate_weights_wide(
            candidate_ids=summary["candidate_id"].to_numpy(),
            candidate_sources=summary["candidate_source"].to_numpy(),
            weights=sorted_weights,
            domain_names=model.domain_names,
        ).to_parquet(full_weights_parquet, index=False)

    source_counts = summary["candidate_source"].value_counts().sort_index().to_dict()
    run_summary = {
        "output_dir": str(args.output_dir),
        "model_json": str(args.model_json),
        "model_variant": "dsp_effective_exposure_penalty_nnls",
        "signal_csv": str(args.signal_csv),
        "num_candidates": len(summary),
        "num_sobol_candidates": int(args.num_sobol_candidates),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "num_observed_rows": len(observed_ids),
        "num_domains": len(model.domain_names),
        "max_alpha": float(args.max_alpha),
        "seed": int(args.seed),
        "proportional_predicted_y_factor": proportional_predicted,
        "oof_rmse": oof_rmse,
        "oof_spearman": float(model.metrics.get("oof_spearman", np.nan)),
        "cached_named_predictions": cached_named_predictions,
        "predicted_gain_quantiles": quantiles(summary["predicted_y_factor_gain_vs_proportional"].to_numpy()),
        "predicted_lcb_gain_quantiles": quantiles(summary["predicted_y_factor_gain_lcb"].to_numpy()),
        "nearest_observed_tv_quantiles": quantiles(summary["nearest_observed_tv"].to_numpy()),
        "tv_to_proportional_quantiles": quantiles(summary["average_phase_tv_to_proportional"].to_numpy()),
        "max_phase_weight_quantiles": quantiles(summary["max_phase_weight"].to_numpy()),
        "passes_basic_dashboard_gate_count": int(summary["passes_basic_dashboard_gate"].sum()),
        "candidate_summary_csv": str(summary_csv),
        "candidate_summary_parquet": str(summary_parquet),
        "candidate_top_domain_deltas_parquet": str(top_deltas_parquet),
        "candidate_weights_wide_parquet": str(full_weights_parquet) if full_weights_parquet else None,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(args.output_dir / "summary.json", run_summary)
    pd.DataFrame({"run_name": observed_ids, "run_id": observed_run_ids}).to_csv(
        args.output_dir / "observed_runs.csv",
        index=False,
    )
    return run_summary


def build_endpoint_discovery(args: argparse.Namespace) -> dict[str, Any]:
    """Build extrapolative DSP endpoint and interpolation-path cache files."""
    started_at = time.time()
    output_dir = args.endpoint_output_dir or (args.output_dir / "endpoint_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading canonical DSP model for endpoint discovery: {args.model_json}", flush=True)
    model = load_model(args.model_json)
    observed_ids, _, observed_weights = read_signal_weights(args.signal_csv, model.domain_names)
    proportional = proportional_from_observed(observed_ids=observed_ids, observed_weights=observed_weights)
    named_ids, named_weights = read_named_weights(args.mixture_weights_csv, model.domain_names)
    validate_model_against_cached_summary(
        model=model,
        model_summary_path=args.model_summary_json,
        named_ids=named_ids,
        named_weights=named_weights,
    )

    proportional_weights = np.stack([proportional, proportional], axis=0)[None, :, :].astype(np.float32)
    observed_predicted = predict_y_factor(model, observed_weights)
    observed_start_count = min(max(args.endpoint_observed_start_count, 0), len(observed_ids))
    observed_start_indices = np.argsort(-observed_predicted)[:observed_start_count]
    start_weights = np.concatenate(
        [
            proportional_weights,
            named_weights,
            observed_weights[observed_start_indices],
        ],
        axis=0,
    )

    kl_penalties = parse_float_tuple(args.endpoint_kl_penalties)
    print(
        "Optimizing endpoint path with "
        f"{len(kl_penalties)} KL penalties, {len(start_weights)} deterministic starts, "
        f"{args.endpoint_random_start_count} random starts...",
        flush=True,
    )
    endpoint_detail, endpoint_weights = discover_regularized_endpoints(
        model=model,
        proportional=proportional,
        start_weights=start_weights,
        kl_penalties=kl_penalties,
        max_weight_penalty=float(args.endpoint_max_weight_penalty),
        max_weight_target=float(args.endpoint_max_weight_target),
        random_start_count=int(args.endpoint_random_start_count),
        seed=int(args.seed),
        maxiter=int(args.endpoint_maxiter),
        random_start_alpha=float(args.endpoint_random_start_alpha),
    )
    proportional_predicted = float(predict_y_factor(model, proportional_weights)[0])
    oof_rmse = float(model.metrics.get("oof_rmse", np.nan))
    endpoint_diag = candidate_diagnostics(
        candidate_ids=endpoint_detail["candidate_id"].to_numpy(),
        candidate_sources=endpoint_detail["candidate_source"].to_numpy(),
        weights=endpoint_weights,
        proportional=proportional,
        predicted_y_factor=endpoint_detail["predicted_y_factor"].to_numpy(dtype=float),
        proportional_predicted_y_factor=proportional_predicted,
        observed_ids=observed_ids,
        observed_weights=observed_weights,
        oof_rmse=oof_rmse,
        lcb_z=float(args.lcb_z),
    )
    keep_detail_cols = [
        "candidate_id",
        "endpoint_index",
        "kl_penalty",
        "max_weight_penalty",
        "max_weight_target",
        "best_start_index",
        "optimization_success",
        "optimization_message",
        "optimization_iterations",
        "optimization_function_evals",
        "regularized_utility",
        "kl_to_proportional",
        "max_weight_excess_penalty",
    ]
    endpoint_summary = endpoint_diag.merge(endpoint_detail.loc[:, keep_detail_cols], on="candidate_id", how="left")
    endpoint_summary = endpoint_summary.sort_values(["kl_penalty", "predicted_y_factor_gain_lcb"]).reset_index(drop=True)

    t_values = parse_float_tuple(args.endpoint_path_t_values)
    path_ids, path_weights = endpoint_path_candidates(
        endpoint_summary=endpoint_detail,
        endpoint_weights=endpoint_weights,
        proportional=proportional,
        t_values=t_values,
    )
    print(f"Scoring {len(path_ids)} endpoint interpolation-path candidates...", flush=True)
    path_predicted = predict_y_factor(model, path_weights)
    path_diag = candidate_diagnostics(
        candidate_ids=path_ids,
        candidate_sources=np.full(len(path_ids), "dsp_endpoint_path", dtype=object),
        weights=path_weights,
        proportional=proportional,
        predicted_y_factor=path_predicted,
        proportional_predicted_y_factor=proportional_predicted,
        observed_ids=observed_ids,
        observed_weights=observed_weights,
        oof_rmse=oof_rmse,
        lcb_z=float(args.lcb_z),
    )
    path_detail = []
    for row in endpoint_detail.itertuples(index=False):
        for t in t_values:
            path_detail.append(
                {
                    "candidate_id": f"path_{row.candidate_id}_t{str(t).replace('.', 'p')}",
                    "endpoint_id": row.candidate_id,
                    "endpoint_index": int(row.endpoint_index),
                    "path_t": float(t),
                    "endpoint_kl_penalty": float(row.kl_penalty),
                    "endpoint_max_weight_penalty": float(row.max_weight_penalty),
                }
            )
    path_summary = path_diag.merge(pd.DataFrame.from_records(path_detail), on="candidate_id", how="left")
    path_summary = path_summary.sort_values("predicted_y_factor_gain_lcb", ascending=False).reset_index(drop=True)

    endpoint_summary_csv = output_dir / "endpoint_summary.csv"
    endpoint_summary_parquet = output_dir / "endpoint_summary.parquet"
    endpoint_weights_parquet = output_dir / "endpoint_weights_wide.parquet"
    path_summary_csv = output_dir / "endpoint_path_summary.csv"
    path_summary_parquet = output_dir / "endpoint_path_summary.parquet"
    path_weights_parquet = output_dir / "endpoint_path_weights_wide.parquet"
    top_deltas_parquet = output_dir / "endpoint_top_domain_deltas.parquet"

    print("Writing endpoint discovery cache files...", flush=True)
    endpoint_summary.to_csv(endpoint_summary_csv, index=False)
    endpoint_summary.to_parquet(endpoint_summary_parquet, index=False)
    candidate_weights_wide(
        candidate_ids=endpoint_summary["candidate_id"].to_numpy(),
        candidate_sources=endpoint_summary["candidate_source"].to_numpy(),
        weights=endpoint_weights[endpoint_summary["endpoint_index"].to_numpy(dtype=int)],
        domain_names=model.domain_names,
    ).to_parquet(endpoint_weights_parquet, index=False)

    path_summary.to_csv(path_summary_csv, index=False)
    path_summary.to_parquet(path_summary_parquet, index=False)
    path_row_index = pd.Series(np.arange(len(path_ids)), index=path_ids)
    sorted_path_weights = path_weights[path_row_index.loc[path_summary["candidate_id"]].to_numpy()]
    candidate_weights_wide(
        candidate_ids=path_summary["candidate_id"].to_numpy(),
        candidate_sources=path_summary["candidate_source"].to_numpy(),
        weights=sorted_path_weights,
        domain_names=model.domain_names,
    ).to_parquet(path_weights_parquet, index=False)
    top_domain_delta_table(
        summary=path_summary,
        weights=sorted_path_weights,
        proportional=proportional,
        domain_names=model.domain_names,
        top_candidate_count=args.top_candidate_count,
        top_domain_count=args.top_domain_count,
    ).to_parquet(top_deltas_parquet, index=False)

    run_summary = {
        "output_dir": str(output_dir),
        "model_json": str(args.model_json),
        "model_variant": "dsp_effective_exposure_penalty_nnls",
        "num_endpoints": len(endpoint_summary),
        "num_path_candidates": len(path_summary),
        "kl_penalties": [float(value) for value in kl_penalties],
        "path_t_values": [float(value) for value in t_values],
        "deterministic_start_count": len(start_weights),
        "observed_start_count": int(observed_start_count),
        "random_start_count": int(args.endpoint_random_start_count),
        "random_start_alpha": float(args.endpoint_random_start_alpha),
        "maxiter": int(args.endpoint_maxiter),
        "max_weight_penalty": float(args.endpoint_max_weight_penalty),
        "max_weight_target": float(args.endpoint_max_weight_target),
        "proportional_predicted_y_factor": proportional_predicted,
        "oof_rmse": oof_rmse,
        "endpoint_predicted_gain_quantiles": quantiles(
            endpoint_summary["predicted_y_factor_gain_vs_proportional"].to_numpy()
        ),
        "path_predicted_gain_quantiles": quantiles(path_summary["predicted_y_factor_gain_vs_proportional"].to_numpy()),
        "path_nearest_observed_tv_quantiles": quantiles(path_summary["nearest_observed_tv"].to_numpy()),
        "endpoint_summary_csv": str(endpoint_summary_csv),
        "endpoint_summary_parquet": str(endpoint_summary_parquet),
        "endpoint_weights_wide_parquet": str(endpoint_weights_parquet),
        "endpoint_path_summary_csv": str(path_summary_csv),
        "endpoint_path_summary_parquet": str(path_summary_parquet),
        "endpoint_path_weights_wide_parquet": str(path_weights_parquet),
        "endpoint_top_domain_deltas_parquet": str(top_deltas_parquet),
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(output_dir / "summary.json", run_summary)
    return run_summary


def main() -> None:
    args = parse_args()
    if args.endpoint_only:
        summary = build_endpoint_discovery(args)
    else:
        summary = build_candidate_library(args)
        if args.endpoint_discovery:
            summary["endpoint_discovery"] = build_endpoint_discovery(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
