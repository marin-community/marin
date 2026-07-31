# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Synthetic recovery diagnostics for production-scale DSP fitting.

This benchmark treats a DSP model as the data-generating process, samples
observations from the 167-partition production-swarm candidate matrix, adds
controlled noise, and fits the same profiled nonlinear + linear-head DSP form
used by the 300M analysis scripts. The point is to separate solver and
identifiability issues from real-data model misspecification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.design_production_swarm_167p import (
    PHASE_FRACTIONS,
    PHASE_NAMES,
    TARGET_BUDGET,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    CV_SEED,
    DSPVariant,
    FittedDSPModel,
    LINEAR_REG,
    LinearMode,
    PenaltyMode,
    VARIANTS,
    _bounds,
    _features,
    _fit_linear_head,
    _pack_params,
    _predict,
    _start_bank,
    _unpack_theta,
    _uses_rho,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import PacketData

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PACKET_DIR = (
    SCRIPT_DIR
    / "reference_outputs"
    / "production_swarm_mixture_design_167p_20260523_collaborator_packet_proplogit_exp_tau20_lam0p25"
)
DEFAULT_CANDIDATE_CSV = DEFAULT_PACKET_DIR / "production_swarm_167p_candidate_mixtures.csv"
DEFAULT_BUCKET_CSV = DEFAULT_PACKET_DIR / "datakit_moe_mix_buckets.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_synthetic_recovery_167p_20260608"
DEFAULT_VARIANT = "dsp_effective_exposure_penalty_nnls"
DEFAULT_SAMPLE_SIZES = "335,503,670,838,1200,1501"
DEFAULT_NOISE_SIGMAS = "0,0.05,0.15"
TRUTH_REGIMES = ("start_bank_prior", "off_grid")
COEFFICIENT_REGIMES = ("positive", "mixed")


class LinearHeadMode(StrEnum):
    """Linear-head parameterization used after fixed DSP nonlinear features."""

    DOMAIN = "domain"
    SHARED = "shared"


@dataclass(frozen=True)
class SolverProfile:
    """Configuration for the profiled nonlinear solver."""

    name: str
    start_top_k: int
    maxiter: int
    maxls: int
    basinhopping_niter: int
    basinhopping_maxiter: int
    use_basinhopping_trigger: bool
    nnls_maxiter_multiplier: int | None
    linear_mode_override: LinearMode | None = None
    ridge_lambdas: tuple[float, ...] | None = None
    fixed_ridge_alpha: float | None = None
    head_mode: LinearHeadMode = LinearHeadMode.DOMAIN


SOLVER_PROFILES = {
    "current": SolverProfile(
        name="current",
        start_top_k=3,
        maxiter=36,
        maxls=20,
        basinhopping_niter=3,
        basinhopping_maxiter=10,
        use_basinhopping_trigger=True,
        nnls_maxiter_multiplier=None,
    ),
    "current_nnls20": SolverProfile(
        name="current_nnls20",
        start_top_k=3,
        maxiter=36,
        maxls=20,
        basinhopping_niter=3,
        basinhopping_maxiter=10,
        use_basinhopping_trigger=True,
        nnls_maxiter_multiplier=20,
    ),
    "coarse_only_nnls20": SolverProfile(
        name="coarse_only_nnls20",
        start_top_k=0,
        maxiter=0,
        maxls=0,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=20,
    ),
    "coarse_only_signed_ridge": SolverProfile(
        name="coarse_only_signed_ridge",
        start_top_k=0,
        maxiter=0,
        maxls=0,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=None,
        linear_mode_override=LinearMode.SIGNED_RIDGE,
    ),
    "coarse_only_signed_ridge_cv": SolverProfile(
        name="coarse_only_signed_ridge_cv",
        start_top_k=0,
        maxiter=0,
        maxls=0,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=None,
        linear_mode_override=LinearMode.SIGNED_RIDGE,
        ridge_lambdas=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    ),
    "coarse_only_signed_ridge_alpha1": SolverProfile(
        name="coarse_only_signed_ridge_alpha1",
        start_top_k=0,
        maxiter=0,
        maxls=0,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=None,
        linear_mode_override=LinearMode.SIGNED_RIDGE,
        fixed_ridge_alpha=1.0,
    ),
    "coarse_only_shared_nnls20": SolverProfile(
        name="coarse_only_shared_nnls20",
        start_top_k=0,
        maxiter=0,
        maxls=0,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=20,
        head_mode=LinearHeadMode.SHARED,
    ),
    "coarse_only_shared_signed_ridge": SolverProfile(
        name="coarse_only_shared_signed_ridge",
        start_top_k=0,
        maxiter=0,
        maxls=0,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=None,
        linear_mode_override=LinearMode.SIGNED_RIDGE,
        head_mode=LinearHeadMode.SHARED,
    ),
    "coarse_only_shared_signed_ridge_alpha1": SolverProfile(
        name="coarse_only_shared_signed_ridge_alpha1",
        start_top_k=0,
        maxiter=0,
        maxls=0,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=None,
        linear_mode_override=LinearMode.SIGNED_RIDGE,
        fixed_ridge_alpha=1.0,
        head_mode=LinearHeadMode.SHARED,
    ),
    "coarse_only_signed_ridge_cv_cap1": SolverProfile(
        name="coarse_only_signed_ridge_cv_cap1",
        start_top_k=0,
        maxiter=0,
        maxls=0,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=None,
        linear_mode_override=LinearMode.SIGNED_RIDGE,
        ridge_lambdas=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    ),
    "signed_ridge_current": SolverProfile(
        name="signed_ridge_current",
        start_top_k=3,
        maxiter=36,
        maxls=20,
        basinhopping_niter=3,
        basinhopping_maxiter=10,
        use_basinhopping_trigger=True,
        nnls_maxiter_multiplier=None,
        linear_mode_override=LinearMode.SIGNED_RIDGE,
    ),
    "more_starts": SolverProfile(
        name="more_starts",
        start_top_k=8,
        maxiter=72,
        maxls=30,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=20,
    ),
    "light_polish_nnls20": SolverProfile(
        name="light_polish_nnls20",
        start_top_k=2,
        maxiter=12,
        maxls=12,
        basinhopping_niter=0,
        basinhopping_maxiter=0,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=20,
    ),
    "more_starts_polish": SolverProfile(
        name="more_starts_polish",
        start_top_k=8,
        maxiter=120,
        maxls=40,
        basinhopping_niter=3,
        basinhopping_maxiter=20,
        use_basinhopping_trigger=False,
        nnls_maxiter_multiplier=20,
    ),
}


def parse_int_schedule(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer schedule."""

    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one integer")
    if any(item <= 0 for item in parsed):
        raise ValueError(f"Sample sizes must be positive: {parsed}")
    return parsed


def parse_seed_schedule(value: str) -> tuple[int, ...]:
    """Parse a comma-separated seed schedule."""

    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one seed")
    if any(item < 0 for item in parsed):
        raise ValueError(f"Seeds must be nonnegative: {parsed}")
    return parsed


def parse_float_schedule(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float schedule."""

    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one float")
    if any(item < 0.0 for item in parsed):
        raise ValueError(f"Noise sigmas must be nonnegative: {parsed}")
    return parsed


def progress(message: str) -> None:
    """Print a timestamped progress line for long local runs."""

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_tokens(bucket_csv: Path) -> tuple[list[str], np.ndarray]:
    """Load bucket names and token counts from the collaborator packet."""

    rows = list(csv.DictReader(bucket_csv.read_text().splitlines()))
    if not rows:
        raise ValueError(f"No bucket rows found in {bucket_csv}")
    required = {"bucket", "tokens"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Bucket CSV missing columns: {sorted(missing)}")
    buckets = [str(row["bucket"]) for row in rows]
    if len(set(buckets)) != len(buckets):
        raise ValueError("Bucket names must be unique")
    tokens = np.asarray([float(row["tokens"]) for row in rows], dtype=float)
    if np.any(tokens <= 0):
        raise ValueError("Token counts must be positive")
    return buckets, tokens


def load_production_packet(candidate_csv: Path, bucket_csv: Path) -> PacketData:
    """Build a DSP PacketData object from the production-swarm handoff CSV."""

    buckets, tokens = load_tokens(bucket_csv)
    frame = pd.read_csv(candidate_csv)
    expected_columns = ["candidate_name", "candidate_type"] + [
        f"{phase}/{bucket}" for phase in PHASE_NAMES for bucket in buckets
    ]
    if list(frame.columns) != expected_columns:
        raise ValueError("Candidate CSV columns do not match bucket CSV order")
    weights = np.stack(
        [
            frame[[f"{phase}/{bucket}" for bucket in buckets]].to_numpy(dtype=float)
            for phase in PHASE_NAMES
        ],
        axis=1,
    )
    if np.any(weights < 0.0):
        raise ValueError("Candidate weights must be nonnegative")
    if not np.allclose(weights.sum(axis=2), 1.0, atol=1e-8):
        raise ValueError("Each phase must sum to 1")
    epoch_multipliers = PHASE_FRACTIONS[:, None] * TARGET_BUDGET / tokens[None, :]
    return PacketData(
        frame=frame,
        name_col="candidate_name",
        y=np.zeros(len(frame), dtype=float),
        w=weights,
        m=len(buckets),
        c0=np.asarray(epoch_multipliers[0], dtype=float),
        c1=np.asarray(epoch_multipliers[1], dtype=float),
        domain_names=buckets,
    )


def variant_by_name(name: str) -> DSPVariant:
    """Return a DSP variant by name."""

    matches = [variant for variant in VARIANTS if variant.name == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous DSP variant: {name}")
    return matches[0]


def synthetic_truth_model(
    packet: PacketData,
    variant: DSPVariant,
    seed: int,
    truth_regime: str,
    coefficient_regime: str,
) -> FittedDSPModel:
    """Generate a plausible full-support DSP ground-truth model."""

    if truth_regime not in TRUTH_REGIMES:
        raise ValueError(f"Unknown truth_regime={truth_regime}")
    if coefficient_regime not in COEFFICIENT_REGIMES:
        raise ValueError(f"Unknown coefficient_regime={coefficient_regime}")
    rng = np.random.default_rng(seed)
    total_exposure = packet.w[:, 0, :] * packet.c0[None, :] + packet.w[:, 1, :] * packet.c1[None, :]
    positive = np.where(total_exposure > 1e-8, total_exposure, np.nan)
    median_exposure = np.nanmedian(positive, axis=0)
    median_exposure = np.where(np.isfinite(median_exposure), median_exposure, np.nanmedian(positive))
    params: dict[str, float | np.ndarray] = {}
    if _uses_rho(variant):
        rho = 1.0 / np.maximum(median_exposure, 1e-3)
        rho_scale = 0.35 if truth_regime == "start_bank_prior" else 1.15
        rho *= np.exp(rng.normal(scale=rho_scale, size=packet.m))
        params["rho"] = np.clip(rho, 1e-4, 2.0)
    else:
        params["rho"] = np.ones(packet.m, dtype=float)
    if variant.penalty_mode != PenaltyMode.NONE:
        tau = np.log1p(np.nanpercentile(positive, 80, axis=0))
        tau = np.where(np.isfinite(tau), tau, np.nanmedian(tau))
        tau_scale = 0.35 if truth_regime == "start_bank_prior" else 1.25
        tau_shift = 0.0 if truth_regime == "start_bank_prior" else rng.normal(loc=0.4, scale=0.6)
        params["tau"] = np.clip(tau + tau_shift + rng.normal(scale=tau_scale, size=packet.m), -2.0, 8.0)
    if variant.name == "dsp_effective_exposure_penalty_nnls":
        gamma_loc = np.log(1.8) if truth_regime == "start_bank_prior" else np.log(3.5)
        gamma_scale = 0.25 if truth_regime == "start_bank_prior" else 0.6
        params["gamma"] = float(np.exp(rng.normal(loc=gamma_loc, scale=gamma_scale)))
    else:
        for key in ("gamma", "gamma_benefit", "gamma_saturation", "gamma_penalty", "gamma_effective"):
            if key in _packable_phase_keys(variant):
                gamma_loc = np.log(1.8) if truth_regime == "start_bank_prior" else np.log(3.5)
                gamma_scale = 0.25 if truth_regime == "start_bank_prior" else 0.6
                params[key] = float(np.exp(rng.normal(loc=gamma_loc, scale=gamma_scale)))
    if coefficient_regime == "positive":
        benefit_coef = np.exp(rng.normal(loc=0.0, scale=0.45, size=packet.m))
        penalty_coef = (
            np.exp(rng.normal(loc=-0.25, scale=0.45, size=packet.m))
            if variant.penalty_mode != PenaltyMode.NONE
            else np.zeros(packet.m, dtype=float)
        )
    else:
        benefit_coef = rng.normal(loc=0.5, scale=1.0, size=packet.m)
        penalty_coef = rng.normal(loc=0.25, scale=0.8, size=packet.m) if variant.penalty_mode != PenaltyMode.NONE else np.zeros(packet.m, dtype=float)
    return FittedDSPModel(
        variant=variant,
        params=params,
        intercept=0.0,
        benefit_coef=benefit_coef,
        penalty_coef=penalty_coef,
    )


def _packable_phase_keys(variant: DSPVariant) -> set[str]:
    """Return possible phase keys by checking the packed start bank shape path."""

    # The canonical benchmark currently uses effective exposure, but keeping this
    # helper avoids silently omitting phase keys if the variant changes later.
    if variant.name in {"dsp_phase_benefit_penalty_nnls", "dsp_phase_benefit_penalty_signed"}:
        return {"gamma"}
    if variant.name == "dsp_effective_exposure_penalty_nnls":
        return {"gamma"}
    return {"gamma", "gamma_benefit", "gamma_saturation", "gamma_penalty", "gamma_effective"}


def standardized_truth(packet: PacketData, truth_model: FittedDSPModel) -> np.ndarray:
    """Return standardized lower-is-better synthetic target values."""

    raw = _predict(truth_model, packet.w, packet)
    std = float(np.std(raw))
    if std <= 1e-12:
        raise ValueError("Synthetic truth has near-zero variation")
    return (raw - float(np.mean(raw))) / std


def noisy_observations(truth: np.ndarray, sigma: float, seed: int, heteroskedastic: bool) -> np.ndarray:
    """Add controlled observation noise to standardized truth."""

    rng = np.random.default_rng(seed)
    if sigma == 0.0:
        return truth.copy()
    if not heteroskedastic:
        scale = np.full_like(truth, sigma, dtype=float)
    else:
        ranks = pd.Series(truth).rank(pct=True).to_numpy(dtype=float)
        scale = sigma * (0.5 + ranks)
    return truth + rng.normal(scale=scale, size=len(truth))


def fit_variant_with_profile(
    packet: PacketData,
    variant: DSPVariant,
    profile: SolverProfile,
    seed: int,
) -> tuple[FittedDSPModel, pd.DataFrame]:
    """Fit a DSP variant with a configurable version of the current solver."""

    variant = variant_for_profile(variant, profile)
    starts = _start_bank(packet, variant)
    bounds = _bounds(variant, packet.m)
    coarse_rows = [
        {
            "stage": "coarse",
            "start_id": start_id,
            "objective": profile_objective_with_profile(packet, variant, start, profile),
        }
        for start_id, start in enumerate(starts)
    ]
    ranked = sorted(coarse_rows, key=lambda row: float(row["objective"]))
    rows = list(coarse_rows)
    best_objective = float("inf")
    best_theta: np.ndarray | None = None

    if profile.maxiter <= 0:
        best_theta = starts[int(ranked[0]["start_id"])]
        params = _unpack_theta(best_theta, variant, packet.m)
        model = fit_linear_head_with_profile(packet.w, packet.y, packet, variant, params, profile)
        rows.append(
            {
                "stage": "coarse_selected",
                "chosen_rank": 0,
                "start_id": int(ranked[0]["start_id"]),
                "objective": float(ranked[0]["objective"]),
                "success": True,
                "message": "selected best coarse start without nonlinear refinement",
                "nit": 0,
            }
        )
        return model, pd.DataFrame.from_records(rows)

    for rank, row in enumerate(ranked[: profile.start_top_k]):
        start = starts[int(row["start_id"])]
        result = minimize(
            lambda theta: profile_objective_with_profile(packet, variant, np.asarray(theta, dtype=float), profile),
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": profile.maxiter, "ftol": 1e-7, "maxls": profile.maxls},
        )
        rows.append(
            {
                "stage": "refine",
                "chosen_rank": rank,
                "start_id": int(row["start_id"]),
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
                "nit": int(getattr(result, "nit", -1)),
            }
        )
        if float(result.fun) < best_objective:
            best_objective = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)

    if best_theta is None:
        raise RuntimeError(f"No fit result for {variant.name}")

    should_hop = profile.basinhopping_niter > 0
    if profile.use_basinhopping_trigger and len(ranked) > 1:
        should_hop = float(ranked[min(1, len(ranked) - 1)]["objective"]) > 1.1 * float(ranked[0]["objective"])
    if should_hop:
        hop_result = basinhopping(
            lambda theta: profile_objective_with_profile(packet, variant, np.asarray(theta, dtype=float), profile),
            best_theta,
            niter=profile.basinhopping_niter,
            stepsize=0.15,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": bounds,
                "options": {"maxiter": profile.basinhopping_maxiter, "ftol": 1e-7},
            },
            seed=seed,
        )
        rows.append(
            {
                "stage": "basin_hopping_diagnostic",
                "chosen_rank": -1,
                "start_id": -1,
                "objective": float(hop_result.fun),
                "success": bool(hop_result.lowest_optimization_result.success),
                "message": str(hop_result.message),
                "nit": int(getattr(hop_result.lowest_optimization_result, "nit", -1)),
            }
        )
        if float(hop_result.fun) < best_objective:
            best_theta = np.asarray(hop_result.x, dtype=float)

    params = _unpack_theta(best_theta, variant, packet.m)
    model = fit_linear_head_with_profile(packet.w, packet.y, packet, variant, params, profile)
    return model, pd.DataFrame.from_records(rows)


def variant_for_profile(variant: DSPVariant, profile: SolverProfile) -> DSPVariant:
    """Return the fitting variant implied by a solver profile."""

    if profile.linear_mode_override is None:
        return variant
    return replace(
        variant,
        name=f"{variant.name}__{profile.linear_mode_override.value}",
        linear_mode=profile.linear_mode_override,
    )


def fit_linear_head_with_profile(
    weights: np.ndarray,
    targets: np.ndarray,
    packet: PacketData,
    variant: DSPVariant,
    params: dict[str, float | np.ndarray],
    profile: SolverProfile,
) -> FittedDSPModel:
    """Fit the linear head, optionally raising SciPy NNLS max iterations."""

    if (
        profile.nnls_maxiter_multiplier is None
        and profile.ridge_lambdas is None
        and profile.fixed_ridge_alpha is None
        and profile.head_mode == LinearHeadMode.DOMAIN
    ):
        return _fit_linear_head(weights, targets, packet, variant, params)
    signal, penalty = _features(weights, packet, variant, params)
    design = design_from_features(signal, penalty, variant)
    if profile.head_mode == LinearHeadMode.SHARED:
        return fit_shared_linear_head(design, targets, variant, params, packet.m, profile)
    if variant.linear_mode == LinearMode.SIGNED_RIDGE and profile.ridge_lambdas is not None:
        alpha = choose_ridge_lambda(design, targets, profile.ridge_lambdas)
        model = fit_linear_design_head(design, targets, variant, params, packet.m, ridge_alpha=alpha)
        model.params["_linear_reg"] = alpha
        return model
    if variant.linear_mode == LinearMode.SIGNED_RIDGE and profile.fixed_ridge_alpha is not None:
        model = fit_linear_design_head(design, targets, variant, params, packet.m, ridge_alpha=profile.fixed_ridge_alpha)
        model.params["_linear_reg"] = profile.fixed_ridge_alpha
        return model
    if profile.nnls_maxiter_multiplier is None:
        return _fit_linear_head(weights, targets, packet, variant, params)
    return fit_linear_design_head(
        design,
        targets,
        variant,
        params,
        packet.m,
        ridge_alpha=LINEAR_REG,
        nnls_maxiter=profile.nnls_maxiter_multiplier * design.shape[1],
    )


def fit_shared_linear_head(
    design: np.ndarray,
    targets: np.ndarray,
    variant: DSPVariant,
    params: dict[str, float | np.ndarray],
    num_domains: int,
    profile: SolverProfile,
) -> FittedDSPModel:
    """Fit a shared benefit/penalty head and expand it to domain coefficients."""

    if variant.penalty_mode == PenaltyMode.NONE:
        shared_design = design.sum(axis=1, keepdims=True)
    else:
        shared_design = np.column_stack(
            [
                design[:, :num_domains].sum(axis=1),
                design[:, num_domains:].sum(axis=1),
            ]
        )
    ridge_alpha = LINEAR_REG
    if variant.linear_mode == LinearMode.SIGNED_RIDGE and profile.fixed_ridge_alpha is not None:
        ridge_alpha = profile.fixed_ridge_alpha
    model = fit_shared_design_head(
        shared_design,
        targets,
        variant,
        params,
        num_domains,
        ridge_alpha=ridge_alpha,
        nnls_maxiter=None
        if profile.nnls_maxiter_multiplier is None
        else profile.nnls_maxiter_multiplier * shared_design.shape[1],
    )
    model.params["_linear_reg"] = ridge_alpha
    model.params["_linear_head_mode"] = LinearHeadMode.SHARED.value
    model.params["_linear_head_param_count"] = int(shared_design.shape[1])
    return model


def design_from_features(signal: np.ndarray, penalty: np.ndarray, variant: DSPVariant) -> np.ndarray:
    """Return the profiled linear design matrix for DSP features."""

    if variant.penalty_mode == PenaltyMode.NONE:
        return -signal
    return np.hstack([-signal, penalty])


def fit_shared_design_head(
    design: np.ndarray,
    targets: np.ndarray,
    variant: DSPVariant,
    params: dict[str, float | np.ndarray],
    num_domains: int,
    *,
    ridge_alpha: float,
    nnls_maxiter: int | None,
) -> FittedDSPModel:
    """Fit a low-dimensional shared DSP head from an aggregate design matrix."""

    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(targets.mean())
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    if variant.linear_mode == LinearMode.NNLS:
        if ridge_alpha > 0.0:
            centered_design = np.vstack([centered_design, np.sqrt(ridge_alpha) * np.eye(centered_design.shape[1])])
            centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
        coef, _ = nnls(centered_design, centered_targets, maxiter=nnls_maxiter)
    else:
        lhs = centered_design.T @ centered_design + ridge_alpha * np.eye(centered_design.shape[1])
        rhs = centered_design.T @ centered_targets
        coef = np.linalg.solve(lhs, rhs)
    intercept = float(target_mean - (design_mean @ coef).item())
    benefit_scalar = float(coef[0])
    benefit_coef = np.full(num_domains, benefit_scalar, dtype=float)
    if variant.penalty_mode == PenaltyMode.NONE:
        penalty_coef = np.zeros(num_domains, dtype=float)
    else:
        penalty_scalar = float(coef[1])
        penalty_coef = np.full(num_domains, penalty_scalar, dtype=float)
    return FittedDSPModel(
        variant=variant,
        params=params,
        intercept=intercept,
        benefit_coef=benefit_coef,
        penalty_coef=penalty_coef,
    )


def fit_linear_design_head(
    design: np.ndarray,
    targets: np.ndarray,
    variant: DSPVariant,
    params: dict[str, float | np.ndarray],
    num_domains: int,
    *,
    ridge_alpha: float,
    nnls_maxiter: int | None = None,
) -> FittedDSPModel:
    """Fit the DSP linear head from a precomputed design matrix."""

    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(targets.mean())
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    if variant.linear_mode == LinearMode.NNLS:
        if ridge_alpha > 0.0:
            centered_design = np.vstack([centered_design, np.sqrt(ridge_alpha) * np.eye(centered_design.shape[1])])
            centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
        coef, _ = nnls(centered_design, centered_targets, maxiter=nnls_maxiter)
    else:
        lhs = centered_design.T @ centered_design + ridge_alpha * np.eye(centered_design.shape[1])
        rhs = centered_design.T @ centered_targets
        coef = np.linalg.solve(lhs, rhs)
    intercept = float(target_mean - (design_mean @ coef).item())
    benefit_coef = np.asarray(coef[:num_domains], dtype=float)
    if variant.penalty_mode == PenaltyMode.NONE:
        penalty_coef = np.zeros(num_domains, dtype=float)
    else:
        penalty_coef = np.asarray(coef[num_domains:], dtype=float)
    return FittedDSPModel(
        variant=variant,
        params=params,
        intercept=intercept,
        benefit_coef=benefit_coef,
        penalty_coef=penalty_coef,
    )


def choose_ridge_lambda(design: np.ndarray, targets: np.ndarray, ridge_lambdas: tuple[float, ...]) -> float:
    """Choose signed-ridge regularization by five-fold CV."""

    if len(targets) < 10:
        return float(ridge_lambdas[0])
    n_splits = min(5, len(targets))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED)
    rows: list[tuple[float, float]] = []
    for alpha in ridge_lambdas:
        residuals: list[float] = []
        for train_idx, test_idx in kf.split(design):
            train_design = design[train_idx]
            train_targets = targets[train_idx]
            design_mean = train_design.mean(axis=0, keepdims=True)
            target_mean = float(train_targets.mean())
            centered_design = train_design - design_mean
            centered_targets = train_targets - target_mean
            lhs = centered_design.T @ centered_design + float(alpha) * np.eye(centered_design.shape[1])
            rhs = centered_design.T @ centered_targets
            coef = np.linalg.solve(lhs, rhs)
            intercept = float(target_mean - (design_mean @ coef).item())
            pred = intercept + design[test_idx] @ coef
            residuals.append(float(np.mean((pred - targets[test_idx]) ** 2)))
        rows.append((float(np.mean(residuals)), float(alpha)))
    rows.sort(key=lambda item: item[0])
    return rows[0][1]


def profile_objective_with_profile(
    packet: PacketData,
    variant: DSPVariant,
    theta: np.ndarray,
    profile: SolverProfile,
) -> float:
    """Profile objective with the configured linear-head solver."""

    params = _unpack_theta(theta, variant, packet.m)
    model = fit_linear_head_with_profile(packet.w, packet.y, packet, variant, params, profile)
    pred = _predict(model, packet.w, packet)
    residual = pred - packet.y
    rmse_value = float(np.sqrt(np.mean(residual**2)))
    tail_count = max(5, int(np.ceil(0.15 * len(packet.y))))
    tail_idx = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(packet.y[tail_idx] - pred[tail_idx], 0.0)))
    return rmse_value + 0.5 * optimism


def subset_packet(packet: PacketData, indices: np.ndarray, y: np.ndarray) -> PacketData:
    """Return a PacketData view for fitting on a subset."""

    frame = packet.frame.iloc[indices].reset_index(drop=True).copy()
    return PacketData(
        frame=frame,
        name_col=packet.name_col,
        y=np.asarray(y[indices], dtype=float),
        w=np.asarray(packet.w[indices], dtype=float),
        m=packet.m,
        c0=packet.c0,
        c1=packet.c1,
        domain_names=packet.domain_names,
    )


def evaluate_fit(
    full_packet: PacketData,
    train_packet: PacketData,
    model: FittedDSPModel,
    truth: np.ndarray,
    noisy_y: np.ndarray,
) -> dict[str, Any]:
    """Evaluate prediction, rank recovery, and optimizer-relevant regret."""

    full_pred = _predict(model, full_packet.w, full_packet)
    train_pred = _predict(model, train_packet.w, train_packet)
    train_truth = truth[train_packet.frame["_source_index"].to_numpy(dtype=int)]
    train_noisy = noisy_y[train_packet.frame["_source_index"].to_numpy(dtype=int)]
    true_best_idx = int(np.argmin(truth))
    pred_best_idx = int(np.argmin(full_pred))
    actual_rank = int(np.where(np.argsort(truth) == pred_best_idx)[0][0] + 1)
    top_k = min(20, len(truth))
    true_top = set(np.argsort(truth)[:top_k].tolist())
    pred_top = set(np.argsort(full_pred)[:top_k].tolist())
    theta = _pack_params(model.params, model.variant)
    bounds = _bounds(model.variant, full_packet.m)
    near_bounds = sum(
        int(abs(float(value) - lo) < 1e-4 or abs(float(value) - hi) < 1e-4)
        for value, (lo, hi) in zip(theta, bounds, strict=True)
    )
    return {
        "train_rmse_noisy": rmse(train_noisy, train_pred),
        "train_rmse_truth": rmse(train_truth, train_pred),
        "full_rmse_truth": rmse(truth, full_pred),
        "full_spearman_truth": safe_spearman(truth, full_pred),
        "pred_best_candidate": str(full_packet.frame.iloc[pred_best_idx][full_packet.name_col]),
        "true_best_candidate": str(full_packet.frame.iloc[true_best_idx][full_packet.name_col]),
        "pred_best_true_rank": actual_rank,
        "pred_best_regret": float(truth[pred_best_idx] - truth[true_best_idx]),
        "top20_overlap": len(true_top & pred_top) / top_k,
        "nonlinear_boundary_hits": int(near_bounds),
        "linear_reg": float(model.params.get("_linear_reg", LINEAR_REG)),
        "linear_head_param_count": int(model.params.get("_linear_head_param_count", 2 * full_packet.m)),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""

    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Spearman correlation with NaN converted to zero."""

    value = float(spearmanr(y_true, y_pred).statistic)
    if math.isnan(value):
        return 0.0
    return value


def run_trial(
    full_packet: PacketData,
    variant: DSPVariant,
    profile: SolverProfile,
    sample_size: int,
    sigma: float,
    seed: int,
    heteroskedastic: bool,
    truth_regime: str,
    coefficient_regime: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one synthetic recovery trial."""

    if sample_size > len(full_packet.frame):
        raise ValueError(f"sample_size={sample_size} exceeds candidate count={len(full_packet.frame)}")
    truth_model = synthetic_truth_model(
        full_packet,
        variant,
        seed=seed,
        truth_regime=truth_regime,
        coefficient_regime=coefficient_regime,
    )
    truth = standardized_truth(full_packet, truth_model)
    noisy_y = noisy_observations(truth, sigma=sigma, seed=seed + 1009, heteroskedastic=heteroskedastic)
    rng = np.random.default_rng(seed + 2027)
    indices = np.sort(rng.choice(len(full_packet.frame), size=sample_size, replace=False))
    fit_variant = variant_for_profile(variant, profile)
    source_frame = full_packet.frame.copy()
    source_frame["_source_index"] = np.arange(len(source_frame), dtype=int)
    indexed_packet = PacketData(
        frame=source_frame,
        name_col=full_packet.name_col,
        y=full_packet.y,
        w=full_packet.w,
        m=full_packet.m,
        c0=full_packet.c0,
        c1=full_packet.c1,
        domain_names=full_packet.domain_names,
    )
    train_packet = subset_packet(indexed_packet, indices, noisy_y)
    start_time = time.time()
    fitted_model, trace = fit_variant_with_profile(train_packet, variant, profile, seed=seed)
    elapsed = time.time() - start_time
    metrics = evaluate_fit(indexed_packet, train_packet, fitted_model, truth, noisy_y)
    metrics.update(
        {
            "solver_profile": profile.name,
            "truth_variant": variant.name,
            "fit_variant": fitted_model.variant.name,
            "sample_size": int(sample_size),
            "candidate_count": int(len(full_packet.frame)),
            "partition_count": int(full_packet.m),
            "noise_sigma": float(sigma),
            "heteroskedastic": bool(heteroskedastic),
            "truth_regime": truth_regime,
            "coefficient_regime": coefficient_regime,
            "seed": int(seed),
            "elapsed_sec": float(elapsed),
            "total_param_count": int(fitted_model.total_param_count),
            "m_dependent_params_per_domain": int(fitted_model.m_dependent_params_per_domain),
            "nonlinear_param_count": int(len(_pack_params(fitted_model.params, fitted_model.variant))),
            "n_over_total_params": float(sample_size / fitted_model.total_param_count),
            "n_over_nonlinear_params": float(sample_size / len(_pack_params(fitted_model.params, fitted_model.variant))),
            "best_trace_objective": float(trace["objective"].min()),
            "refine_success_count": int(trace.loc[trace["stage"].eq("refine"), "success"].sum()),
            "refine_count": int(trace["stage"].eq("refine").sum()),
        }
    )
    trace = trace.assign(
        solver_profile=profile.name,
        truth_variant=variant.name,
        fit_variant=fit_variant.name,
        sample_size=int(sample_size),
        noise_sigma=float(sigma),
        heteroskedastic=bool(heteroskedastic),
        truth_regime=truth_regime,
        coefficient_regime=coefficient_regime,
        seed=int(seed),
    )
    return metrics, trace


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--bucket-csv", type=Path, default=DEFAULT_BUCKET_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--solver-profile", action="append", choices=tuple(SOLVER_PROFILES), default=None)
    parser.add_argument("--sample-sizes", default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--noise-sigmas", default=DEFAULT_NOISE_SIGMAS)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--truth-regime", choices=TRUTH_REGIMES, default="off_grid")
    parser.add_argument("--coefficient-regime", choices=COEFFICIENT_REGIMES, default="positive")
    parser.add_argument("--heteroskedastic", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke-sized schedule.")
    return parser


def main() -> None:
    """Run synthetic recovery diagnostics."""

    args = build_arg_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = variant_by_name(args.variant)
    sample_sizes = (335, 670) if args.quick else parse_int_schedule(args.sample_sizes)
    noise_sigmas = (0.0,) if args.quick else parse_float_schedule(args.noise_sigmas)
    seeds = (CV_SEED,) if args.quick else parse_seed_schedule(args.seeds)
    default_profiles = ["current", "current_nnls20"] if args.quick else ["current"]
    profiles = tuple(SOLVER_PROFILES[name] for name in (args.solver_profile or default_profiles))
    full_packet = load_production_packet(args.candidate_csv, args.bucket_csv)

    progress(
        "Loaded production packet: "
        f"candidates={len(full_packet.frame)} partitions={full_packet.m} variant={variant.name}"
    )
    rows: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    summary_path = output_dir / "synthetic_recovery_summary.csv"
    trace_path = output_dir / "synthetic_recovery_solver_trace.csv"
    run_spec_path = output_dir / "run_spec.json"
    for profile in profiles:
        for seed in seeds:
            for sample_size in sample_sizes:
                for sigma in noise_sigmas:
                    progress(
                        "Running trial "
                        f"profile={profile.name} seed={seed} n={sample_size} sigma={sigma} "
                        f"heteroskedastic={args.heteroskedastic}"
                    )
                    try:
                        metrics, trace = run_trial(
                            full_packet,
                            variant,
                            profile,
                            sample_size=sample_size,
                            sigma=sigma,
                            seed=seed,
                            heteroskedastic=args.heteroskedastic,
                            truth_regime=args.truth_regime,
                            coefficient_regime=args.coefficient_regime,
                        )
                    except Exception as exc:
                        metrics = {
                            "solver_profile": profile.name,
                            "truth_variant": variant.name,
                            "fit_variant": variant_for_profile(variant, profile).name,
                            "sample_size": int(sample_size),
                            "candidate_count": int(len(full_packet.frame)),
                            "partition_count": int(full_packet.m),
                            "noise_sigma": float(sigma),
                            "heteroskedastic": bool(args.heteroskedastic),
                            "truth_regime": args.truth_regime,
                            "coefficient_regime": args.coefficient_regime,
                            "seed": int(seed),
                            "fit_status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        trace = pd.DataFrame.from_records(
                            [
                                {
                                    "solver_profile": profile.name,
                                    "truth_variant": variant.name,
                                    "fit_variant": variant_for_profile(variant, profile).name,
                                    "sample_size": int(sample_size),
                                    "noise_sigma": float(sigma),
                                    "heteroskedastic": bool(args.heteroskedastic),
                                    "truth_regime": args.truth_regime,
                                    "coefficient_regime": args.coefficient_regime,
                                    "seed": int(seed),
                                    "stage": "failed",
                                    "message": f"{type(exc).__name__}: {exc}",
                                }
                            ]
                        )
                        progress(
                            "Failed trial "
                            f"profile={profile.name} n={sample_size} sigma={sigma} "
                            f"error={type(exc).__name__}: {exc}"
                        )
                    else:
                        metrics["fit_status"] = "ok"
                        progress(
                            "Finished trial "
                            f"profile={profile.name} n={sample_size} sigma={sigma} "
                            f"spearman={metrics['full_spearman_truth']:.3f} "
                            f"regret={metrics['pred_best_regret']:.4f} "
                            f"elapsed={metrics['elapsed_sec']:.1f}s"
                        )
                    rows.append(metrics)
                    traces.append(trace)
                    pd.DataFrame.from_records(rows).to_csv(summary_path, index=False)
                    pd.concat(traces, ignore_index=True).to_csv(trace_path, index=False)

    summary = pd.DataFrame.from_records(rows)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    summary.to_csv(summary_path, index=False)
    trace_frame.to_csv(trace_path, index=False)
    run_spec_path.write_text(
        json.dumps(
            {
                "candidate_csv": str(args.candidate_csv),
                "bucket_csv": str(args.bucket_csv),
                "variant": variant.name,
                "profiles": [profile.name for profile in profiles],
                "sample_sizes": list(sample_sizes),
                "noise_sigmas": list(noise_sigmas),
                "seeds": list(seeds),
                "heteroskedastic": bool(args.heteroskedastic),
                "truth_regime": args.truth_regime,
                "coefficient_regime": args.coefficient_regime,
                "candidate_count": int(len(full_packet.frame)),
                "partition_count": int(full_packet.m),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    progress(f"Wrote {summary_path}")
    progress(f"Wrote {trace_path}")
    progress(f"Wrote {run_spec_path}")


if __name__ == "__main__":
    main()
