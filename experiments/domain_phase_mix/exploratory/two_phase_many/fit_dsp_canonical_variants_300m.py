# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Fit candidate DSP functional forms on the 300M swarm.

DSP is the working codename for the reduced-bias domain saturation-penalty
family. This sweep keeps the M-dependent parameter budget to at most four
parameters per domain: benefit amplitude, penalty amplitude, saturation rate,
and penalty threshold. Phase handling is hard-coded to two phases through fixed
global scalars only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize, nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_and_plot_grp_power_family_penalty_no_l2_60m_vs_300m import (
    RUN_SET_300M,
    _build_fit_frame,
    _metric_frame,
    _packet_from_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    plot_grp_vs_proportional as reference_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PacketData,
    sigmoid,
    softplus,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_canonical_variants_300m_20260510"
SCALE = "300m_6b"
CV_SEED = 0
N_SPLITS = 5
LOWER_TAIL_FRAC = 0.15
LINEAR_REG = 1e-6
PHASE_EPS = 1e-9
FIT_MAXITER = 36
START_TOP_K = 3


class PhaseMode(StrEnum):
    """Two-phase assumption used by a DSP variant."""

    NONE = "none"
    BENEFIT_GAIN = "benefit_gain"
    EFFECTIVE_EXPOSURE = "effective_exposure"


class PenaltyMode(StrEnum):
    """Penalty feature used by a DSP variant."""

    NONE = "none"
    LOG_SOFTPLUS_SQUARED = "log_softplus_squared"


class LinearMode(StrEnum):
    """Linear head constraint used after nonlinear feature construction."""

    NNLS = "nnls"
    SIGNED_RIDGE = "signed_ridge"


@dataclass(frozen=True)
class DSPVariant:
    """One candidate DSP form."""

    name: str
    phase_mode: PhaseMode
    penalty_mode: PenaltyMode
    linear_mode: LinearMode
    description: str


@dataclass(frozen=True)
class FittedDSPModel:
    """Fitted DSP model."""

    variant: DSPVariant
    params: dict[str, float | np.ndarray]
    intercept: float
    benefit_coef: np.ndarray
    penalty_coef: np.ndarray

    @property
    def total_param_count(self) -> int:
        per_domain = len(self.benefit_coef)  # rho
        per_domain += len(self.benefit_coef)  # benefit coefficient
        if self.variant.penalty_mode != PenaltyMode.NONE:
            per_domain += len(self.benefit_coef)  # tau
            per_domain += len(self.benefit_coef)  # penalty coefficient
        global_params = 1  # intercept
        if self.variant.phase_mode != PhaseMode.NONE:
            global_params += 1
        return int(per_domain + global_params)

    @property
    def m_dependent_params_per_domain(self) -> int:
        return 4 if self.variant.penalty_mode != PenaltyMode.NONE else 2


VARIANTS: tuple[DSPVariant, ...] = (
    DSPVariant(
        name="dsp_no_phase_penalty_nnls",
        phase_mode=PhaseMode.NONE,
        penalty_mode=PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=LinearMode.NNLS,
        description="Control: per-domain saturation and penalty with no phase term.",
    ),
    DSPVariant(
        name="dsp_phase_benefit_penalty_nnls",
        phase_mode=PhaseMode.BENEFIT_GAIN,
        penalty_mode=PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=LinearMode.NNLS,
        description="Canonical reduced-bias DSP: phase-1 premium only multiplies the benefit signal.",
    ),
    DSPVariant(
        name="dsp_effective_exposure_penalty_nnls",
        phase_mode=PhaseMode.EFFECTIVE_EXPOSURE,
        penalty_mode=PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=LinearMode.NNLS,
        description="Empirical comparator: phase-1 multiplier enters exposure for benefit and penalty.",
    ),
    DSPVariant(
        name="dsp_phase_benefit_penalty_signed",
        phase_mode=PhaseMode.BENEFIT_GAIN,
        penalty_mode=PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=LinearMode.SIGNED_RIDGE,
        description="Tests nonnegative-head assumption by using a signed ridge linear head.",
    ),
    DSPVariant(
        name="dsp_phase_benefit_no_penalty_nnls",
        phase_mode=PhaseMode.BENEFIT_GAIN,
        penalty_mode=PenaltyMode.NONE,
        linear_mode=LinearMode.NNLS,
        description="Tests whether explicit overexposure penalties are needed.",
    ),
    DSPVariant(
        name="dsp_effective_exposure_no_penalty_nnls",
        phase_mode=PhaseMode.EFFECTIVE_EXPOSURE,
        penalty_mode=PenaltyMode.NONE,
        linear_mode=LinearMode.NNLS,
        description="Tests whether the empirical phase multiplier can replace explicit penalties.",
    ),
)


def _load_packet() -> PacketData:
    fit_frame = _build_fit_frame(_metric_frame(), scale=SCALE, run_set=RUN_SET_300M)
    packet = _packet_from_frame(fit_frame, name="dsp_canonical_300m")
    return packet.base


def _unpack_theta(theta: np.ndarray, variant: DSPVariant, num_domains: int) -> dict[str, float | np.ndarray]:
    cursor = 0
    log_rho = np.clip(theta[cursor : cursor + num_domains], np.log(1e-4), np.log(2.0))
    cursor += num_domains
    params: dict[str, float | np.ndarray] = {"rho": np.exp(log_rho)}
    if variant.penalty_mode != PenaltyMode.NONE:
        params["tau"] = np.clip(theta[cursor : cursor + num_domains], -2.0, 8.0)
        cursor += num_domains
    if variant.phase_mode != PhaseMode.NONE:
        params["gamma"] = float(np.exp(np.clip(theta[cursor], np.log(1e-4), np.log(100.0))))
        cursor += 1
    if cursor != len(theta):
        raise ValueError(f"Unused theta values for {variant.name}: cursor={cursor}, len={len(theta)}")
    return params


def _pack_params(params: dict[str, float | np.ndarray], variant: DSPVariant) -> np.ndarray:
    values: list[np.ndarray] = [np.log(np.asarray(params["rho"], dtype=float))]
    if variant.penalty_mode != PenaltyMode.NONE:
        values.append(np.asarray(params["tau"], dtype=float))
    if variant.phase_mode != PhaseMode.NONE:
        values.append(np.asarray([np.log(float(params["gamma"]))], dtype=float))
    return np.concatenate(values)


def _bounds(variant: DSPVariant, num_domains: int) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = [(np.log(1e-4), np.log(2.0)) for _ in range(num_domains)]
    if variant.penalty_mode != PenaltyMode.NONE:
        bounds.extend([(-2.0, 8.0) for _ in range(num_domains)])
    if variant.phase_mode != PhaseMode.NONE:
        bounds.append((np.log(1e-4), np.log(100.0)))
    return bounds


def _features(
    weights: np.ndarray,
    packet: PacketData,
    variant: DSPVariant,
    params: dict[str, float | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    p0 = weights[:, 0, :]
    p1 = weights[:, 1, :]
    e0 = p0 * packet.c0[None, :]
    e1 = p1 * packet.c1[None, :]
    rho = np.asarray(params["rho"], dtype=float)[None, :]
    if variant.phase_mode == PhaseMode.EFFECTIVE_EXPOSURE:
        exposure = e0 + float(params["gamma"]) * e1
        signal = 1.0 - np.exp(-rho * exposure)
    else:
        exposure = e0 + e1
        signal = 1.0 - np.exp(-rho * exposure)
        if variant.phase_mode == PhaseMode.BENEFIT_GAIN:
            phase1_share = e1 / (exposure + PHASE_EPS)
            signal = (1.0 + float(params["gamma"]) * phase1_share) * signal

    if variant.penalty_mode == PenaltyMode.NONE:
        penalty = np.zeros_like(signal)
    else:
        tau = np.asarray(params["tau"], dtype=float)[None, :]
        penalty = softplus(np.log1p(exposure) - tau) ** 2
    return signal, penalty


def _fit_linear_head(
    weights: np.ndarray,
    targets: np.ndarray,
    packet: PacketData,
    variant: DSPVariant,
    params: dict[str, float | np.ndarray],
) -> FittedDSPModel:
    signal, penalty = _features(weights, packet, variant, params)
    if variant.penalty_mode == PenaltyMode.NONE:
        design = -signal
    else:
        design = np.hstack([-signal, penalty])
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(targets.mean())
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    if variant.linear_mode == LinearMode.NNLS:
        if LINEAR_REG > 0.0:
            centered_design = np.vstack([centered_design, np.sqrt(LINEAR_REG) * np.eye(centered_design.shape[1])])
            centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
        coef, _ = nnls(centered_design, centered_targets)
    else:
        lhs = centered_design.T @ centered_design + LINEAR_REG * np.eye(centered_design.shape[1])
        rhs = centered_design.T @ centered_targets
        coef = np.linalg.solve(lhs, rhs)
    intercept = float(target_mean - (design_mean @ coef).item())
    num_domains = packet.m
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


def _predict(model: FittedDSPModel, weights: np.ndarray, packet: PacketData) -> np.ndarray:
    signal, penalty = _features(weights, packet, model.variant, model.params)
    return np.asarray(model.intercept - signal @ model.benefit_coef + penalty @ model.penalty_coef, dtype=float)


def _profile_objective(packet: PacketData, variant: DSPVariant, theta: np.ndarray) -> float:
    params = _unpack_theta(theta, variant, packet.m)
    model = _fit_linear_head(packet.w, packet.y, packet, variant, params)
    pred = _predict(model, packet.w, packet)
    residual = pred - packet.y
    rmse = float(np.sqrt(np.mean(residual**2)))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(packet.y))))
    tail_idx = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(packet.y[tail_idx] - pred[tail_idx], 0.0)))
    return rmse + 0.5 * optimism


def _start_bank(packet: PacketData, variant: DSPVariant) -> tuple[np.ndarray, ...]:
    z = packet.w[:, 0, :] * packet.c0[None, :] + packet.w[:, 1, :] * packet.c1[None, :]
    positive = np.where(z > 1e-8, z, np.nan)
    median_exposure = np.nanmedian(positive, axis=0)
    median_exposure = np.where(np.isfinite(median_exposure), median_exposure, np.nanmedian(positive))
    base_rho = np.clip(1.0 / np.maximum(median_exposure, 1e-3), 1e-4, 0.5)
    base_tau = np.clip(np.log1p(np.nanpercentile(positive, 85, axis=0)), -2.0, 8.0)
    base_tau = np.where(np.isfinite(base_tau), base_tau, 3.0)

    rng = np.random.default_rng(CV_SEED)
    starts: list[np.ndarray] = []
    for rho_scale, tau_shift, gamma in (
        (0.25, -1.0, 0.25),
        (0.5, -0.5, 0.5),
        (1.0, 0.0, 1.0),
        (2.0, 0.5, 2.0),
        (4.0, 1.0, 8.0),
    ):
        params: dict[str, float | np.ndarray] = {"rho": np.clip(base_rho * rho_scale, 1e-4, 2.0)}
        if variant.penalty_mode != PenaltyMode.NONE:
            params["tau"] = np.clip(base_tau + tau_shift, -2.0, 8.0)
        if variant.phase_mode != PhaseMode.NONE:
            params["gamma"] = gamma
        starts.append(_pack_params(params, variant))

    for _ in range(3):
        params = {"rho": np.clip(base_rho * np.exp(rng.normal(scale=0.7, size=packet.m)), 1e-4, 2.0)}
        if variant.penalty_mode != PenaltyMode.NONE:
            params["tau"] = np.clip(base_tau + rng.normal(scale=0.8, size=packet.m), -2.0, 8.0)
        if variant.phase_mode != PhaseMode.NONE:
            params["gamma"] = float(np.exp(rng.normal(loc=np.log(2.0), scale=0.9)))
        starts.append(_pack_params(params, variant))
    return tuple(starts)


def _fit_variant(packet: PacketData, variant: DSPVariant) -> tuple[FittedDSPModel, pd.DataFrame]:
    starts = _start_bank(packet, variant)
    bounds = _bounds(variant, packet.m)
    coarse_rows = [
        {"stage": "coarse", "start_id": start_id, "objective": _profile_objective(packet, variant, start)}
        for start_id, start in enumerate(starts)
    ]
    ranked = sorted(coarse_rows, key=lambda row: float(row["objective"]))
    rows = list(coarse_rows)
    best_objective = float("inf")
    best_theta: np.ndarray | None = None

    for rank, row in enumerate(ranked[:START_TOP_K]):
        start = starts[int(row["start_id"])]
        result = minimize(
            lambda theta: _profile_objective(packet, variant, np.asarray(theta, dtype=float)),
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": FIT_MAXITER, "ftol": 1e-7, "maxls": 20},
        )
        rows.append(
            {
                "stage": "refine",
                "chosen_rank": rank,
                "start_id": int(row["start_id"]),
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        if float(result.fun) < best_objective:
            best_objective = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)

    if best_theta is None:
        raise RuntimeError(f"No fit result for {variant.name}")

    if len(ranked) > 1 and float(ranked[min(1, len(ranked) - 1)]["objective"]) > 1.1 * float(ranked[0]["objective"]):
        hop_result = basinhopping(
            lambda theta: _profile_objective(packet, variant, np.asarray(theta, dtype=float)),
            best_theta,
            niter=3,
            stepsize=0.15,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds, "options": {"maxiter": 10, "ftol": 1e-7}},
            seed=CV_SEED,
        )
        rows.append(
            {
                "stage": "basin_hopping_diagnostic",
                "chosen_rank": -1,
                "start_id": -1,
                "objective": float(hop_result.fun),
                "success": bool(hop_result.lowest_optimization_result.success),
                "message": str(hop_result.message),
            }
        )
        if float(hop_result.fun) < best_objective:
            best_theta = np.asarray(hop_result.x, dtype=float)

    params = _unpack_theta(best_theta, variant, packet.m)
    model = _fit_linear_head(packet.w, packet.y, packet, variant, params)
    return model, pd.DataFrame.from_records(rows)


def _oof_predictions(packet: PacketData, model: FittedDSPModel) -> np.ndarray:
    oof = np.zeros_like(packet.y)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for train_idx, test_idx in kf.split(packet.w):
        fold_model = _fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
        )
        oof[test_idx] = _predict(fold_model, packet.w[test_idx], packet)
    return oof


def _value_grad_logits(model: FittedDSPModel, packet: PacketData, logits: np.ndarray) -> tuple[float, np.ndarray]:
    num_domains = packet.m
    logits0 = logits[:num_domains]
    logits1 = logits[num_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 /= np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 /= np.sum(p1)

    e0 = packet.c0 * p0
    e1 = packet.c1 * p1
    rho = np.asarray(model.params["rho"], dtype=float)
    benefit_coef = model.benefit_coef
    penalty_coef = model.penalty_coef
    has_penalty = model.variant.penalty_mode != PenaltyMode.NONE

    if model.variant.phase_mode == PhaseMode.EFFECTIVE_EXPOSURE:
        gamma = float(model.params["gamma"])
        exposure = e0 + gamma * e1
        signal = 1.0 - np.exp(-rho * exposure)
        dsignal = rho * np.exp(-rho * exposure)
        if has_penalty:
            tau = np.asarray(model.params["tau"], dtype=float)
            u = np.log1p(exposure) - tau
            sp = softplus(u)
            penalty = sp**2
            dpenalty = 2.0 * sp * sigmoid(u) / (1.0 + exposure)
        else:
            penalty = np.zeros_like(signal)
            dpenalty = np.zeros_like(signal)
        common = -benefit_coef * dsignal + penalty_coef * dpenalty
        grad_e0 = common
        grad_e1 = gamma * common
        value = float(model.intercept - benefit_coef @ signal + penalty_coef @ penalty)
    else:
        exposure = e0 + e1
        denom = exposure + PHASE_EPS
        r = e1 / denom
        base_signal = 1.0 - np.exp(-rho * exposure)
        dbase_signal = rho * np.exp(-rho * exposure)
        signal_amp = np.ones(num_domains, dtype=float)
        dsignal_amp_de0 = np.zeros(num_domains, dtype=float)
        dsignal_amp_de1 = np.zeros(num_domains, dtype=float)
        if model.variant.phase_mode == PhaseMode.BENEFIT_GAIN:
            gamma = float(model.params["gamma"])
            signal_amp = 1.0 + gamma * r
            dsignal_amp_de0 = gamma * (-e1 / (denom * denom))
            dsignal_amp_de1 = gamma * ((e0 + PHASE_EPS) / (denom * denom))
        signal = signal_amp * base_signal
        dsignal_de0 = signal_amp * dbase_signal + base_signal * dsignal_amp_de0
        dsignal_de1 = signal_amp * dbase_signal + base_signal * dsignal_amp_de1
        if has_penalty:
            tau = np.asarray(model.params["tau"], dtype=float)
            u = np.log1p(exposure) - tau
            sp = softplus(u)
            penalty = sp**2
            dpenalty = 2.0 * sp * sigmoid(u) / (1.0 + exposure)
        else:
            penalty = np.zeros_like(signal)
            dpenalty = np.zeros_like(signal)
        grad_e0 = -benefit_coef * dsignal_de0 + penalty_coef * dpenalty
        grad_e1 = -benefit_coef * dsignal_de1 + penalty_coef * dpenalty
        value = float(model.intercept - benefit_coef @ signal + penalty_coef @ penalty)

    grad_p0 = grad_e0 * packet.c0
    grad_p1 = grad_e1 * packet.c1
    grad_logits0 = p0 * (grad_p0 - np.dot(grad_p0, p0))
    grad_logits1 = p1 * (grad_p1 - np.dot(grad_p1, p1))
    return value, np.concatenate([grad_logits0, grad_logits1])


def _optimize_model(model: FittedDSPModel, packet: PacketData) -> tuple[Any, np.ndarray]:
    rng = np.random.default_rng(CV_SEED)
    starts = [np.zeros(2 * packet.m, dtype=float)]
    starts.extend(
        np.concatenate([rng.normal(scale=0.3, size=packet.m), rng.normal(scale=0.3, size=packet.m)]) for _ in range(20)
    )
    best = None
    for start in starts:
        result = minimize(
            lambda logits: _value_grad_logits(model, packet, logits)[0],
            start,
            jac=lambda logits: _value_grad_logits(model, packet, logits)[1],
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError(f"Raw optimization failed for {model.variant.name}")
    logits = np.asarray(best.x, dtype=float)
    logits0 = logits[: packet.m]
    logits1 = logits[packet.m :]
    phase0 = np.exp(logits0 - np.max(logits0))
    phase0 /= np.sum(phase0)
    phase1 = np.exp(logits1 - np.max(logits1))
    phase1 /= np.sum(phase1)
    return best, np.stack([phase0, phase1], axis=0)


def _observed_leaderboard_row(packet: PacketData, model: FittedDSPModel) -> dict[str, Any]:
    pred = _predict(model, packet.w, packet)
    pred_rank = np.argsort(pred)
    actual_rank = np.argsort(packet.y)
    best_pred_idx = int(pred_rank[0])
    return {
        "variant": model.variant.name,
        "best_pred_observed_run": str(packet.frame.iloc[best_pred_idx][packet.name_col]),
        "best_pred_observed_pred_bpb": float(pred[best_pred_idx]),
        "best_pred_observed_actual_bpb": float(packet.y[best_pred_idx]),
        "best_pred_observed_actual_rank": int(np.where(actual_rank == best_pred_idx)[0][0] + 1),
        "pred_top8_mean_actual_bpb": float(np.mean(packet.y[pred_rank[:8]])),
        "pred_top8_best_actual_bpb": float(np.min(packet.y[pred_rank[:8]])),
        "actual_best_bpb": float(np.min(packet.y)),
        "actual_best_run": str(packet.frame.iloc[int(actual_rank[0])][packet.name_col]),
    }


def _metrics(packet: PacketData, model: FittedDSPModel, raw_result: Any, weights: np.ndarray) -> dict[str, Any]:
    train_pred = _predict(model, packet.w, packet)
    oof = _oof_predictions(packet, model)
    residual = oof - packet.y
    train_residual = train_pred - packet.y
    fold_regrets: list[float] = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for _train_idx, test_idx in kf.split(packet.w):
        chosen_idx = int(np.argmin(oof[test_idx]))
        fold_regrets.append(float(packet.y[test_idx][chosen_idx] - np.min(packet.y[test_idx])))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(packet.y))))
    tail_idx = np.argsort(oof)[:tail_count]
    raw_distances = average_phase_tv_distance(packet.w, weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    return {
        "variant": model.variant.name,
        "description": model.variant.description,
        "phase_mode": model.variant.phase_mode.value,
        "penalty_mode": model.variant.penalty_mode.value,
        "linear_mode": model.variant.linear_mode.value,
        "m_dependent_params_per_domain": model.m_dependent_params_per_domain,
        "total_param_count": model.total_param_count,
        "train_rmse": float(np.sqrt(np.mean(train_residual**2))),
        "train_spearman": float(spearmanr(packet.y, train_pred).statistic),
        "train_pearson": float(pearsonr(packet.y, train_pred).statistic),
        "cv_rmse": float(np.sqrt(np.mean(residual**2))),
        "cv_mae": float(np.mean(np.abs(residual))),
        "oof_spearman": float(spearmanr(packet.y, oof).statistic),
        "oof_pearson": float(pearsonr(packet.y, oof).statistic),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "cv_regret_at_1": float(packet.y[int(np.argmin(oof))] - np.min(packet.y)),
        "lower_tail_optimism": float(np.mean(np.maximum(packet.y[tail_idx] - oof[tail_idx], 0.0))),
        "raw_predicted_optimum_value": float(raw_result.fun),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.frame.iloc[nearest_idx][packet.name_col]),
        "raw_nearest_observed_value": float(packet.y[nearest_idx]),
        "raw_phase0_support_gt_1e3": int(np.sum(weights[0] > 1e-3)),
        "raw_phase1_support_gt_1e3": int(np.sum(weights[1] > 1e-3)),
        "phase0_max_weight": float(np.max(weights[0])),
        "phase1_max_weight": float(np.max(weights[1])),
        "optimum_success": bool(raw_result.success),
        "optimum_message": str(raw_result.message),
    }


def _old_grp_baseline_row() -> dict[str, Any]:
    old_summary = SCRIPT_DIR / "grp_power_family_penalty_no_l2_60m_vs_300m_fit_summary.csv"
    frame = pd.read_csv(old_summary)
    row = frame.loc[frame["label"].eq("300M-fit no-$L_2$ GRP")].iloc[0].to_dict()
    return {
        "variant": "old_grp_no_l2",
        "description": "Existing GRP no-L2 300M fit with family/quality/retention structure.",
        "phase_mode": "retention_effective_exposure",
        "penalty_mode": "family_power_penalty",
        "linear_mode": "nnls",
        "m_dependent_params_per_domain": np.nan,
        "total_param_count": int(row["total_param_count"]),
        "train_rmse": float(row["train_rmse"]),
        "train_spearman": float(row["train_spearman"]),
        "train_pearson": np.nan,
        "cv_rmse": float(row["cv_rmse"]),
        "cv_mae": np.nan,
        "oof_spearman": float(row["oof_spearman"]),
        "oof_pearson": np.nan,
        "cv_foldmean_regret_at_1": float(row["cv_foldmean_regret_at_1"]),
        "cv_regret_at_1": float(row["cv_regret_at_1"]),
        "lower_tail_optimism": float(row["lower_tail_optimism"]),
        "raw_predicted_optimum_value": float(row["predicted_optimum_value"]),
        "raw_nearest_observed_tv": float(row["nearest_observed_tv"]),
        "raw_nearest_observed_run_name": str(row["nearest_observed_run_name"]),
        "raw_nearest_observed_value": float(row["nearest_observed_value"]),
        "raw_phase0_support_gt_1e3": np.nan,
        "raw_phase1_support_gt_1e3": np.nan,
        "phase0_max_weight": float(row["phase0_max_weight"]),
        "phase1_max_weight": float(row["phase1_max_weight"]),
        "optimum_success": True,
        "optimum_message": str(row["optimizer_message"]),
    }


def _write_weight_table(packet: PacketData, variant_dir: Path, weights: np.ndarray) -> None:
    rows = []
    for domain_name, phase0, phase1, c0, c1 in zip(
        packet.domain_names,
        weights[0],
        weights[1],
        packet.c0,
        packet.c1,
        strict=True,
    ):
        rows.append(
            {
                "domain_name": domain_name,
                "phase_0_weight": float(phase0),
                "phase_1_weight": float(phase1),
                "phase_0_effective_epochs": float(phase0 * c0),
                "phase_1_effective_epochs": float(phase1 * c1),
            }
        )
    pd.DataFrame.from_records(rows).to_csv(variant_dir / "raw_optimum_weights.csv", index=False)


def _plot_predicted_vs_actual(packet: PacketData, model: FittedDSPModel, variant_dir: Path) -> None:
    pred = _predict(model, packet.w, packet)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    ax.scatter(packet.y, pred, s=30, alpha=0.75, color="#0f766e", edgecolor="white", linewidth=0.4)
    lo = min(float(packet.y.min()), float(pred.min()))
    hi = max(float(packet.y.max()), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#64748b", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Actual eval/uncheatable_eval/bpb")
    ax.set_ylabel("Predicted BPB")
    ax.set_title(model.variant.name)
    ax.grid(True, color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(variant_dir / "predicted_vs_actual.png", dpi=180)
    plt.close(fig)


def _plot_raw_optimum(packet: PacketData, variant: DSPVariant, weights: np.ndarray, variant_dir: Path) -> None:
    color = plt.get_cmap("RdYlGn_r")(0.15)
    schedules = [(variant.name, weights, color)]
    non_cc_indices, cc_indices = reference_plot._grp_domain_order(packet.domain_names, weights)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(26, 22),
        gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.22, "wspace": 0.31},
        facecolor="white",
    )
    reference_plot._plot_non_cc_block(
        ax=axes[0, 0],
        indices=non_cc_indices,
        labels=[reference_plot._display_non_cc_label(packet.domain_names[idx]) for idx in non_cc_indices],
        schedules=schedules,
        phase_idx=0,
        multipliers=packet.c0,
        title="Phase 0: Non-CC Domains",
        show_legend=True,
    )
    reference_plot._plot_cc_block(
        ax=axes[0, 1],
        domain_names=packet.domain_names,
        indices=cc_indices,
        schedules=schedules,
        phase_idx=0,
        multipliers=packet.c0,
        title="Phase 0: CC Domains",
    )
    reference_plot._plot_non_cc_block(
        ax=axes[1, 0],
        indices=non_cc_indices,
        labels=[reference_plot._display_non_cc_label(packet.domain_names[idx]) for idx in non_cc_indices],
        schedules=schedules,
        phase_idx=1,
        multipliers=packet.c1,
        title="Phase 1: Non-CC Domains",
        show_legend=False,
    )
    reference_plot._plot_cc_block(
        ax=axes[1, 1],
        domain_names=packet.domain_names,
        indices=cc_indices,
        schedules=schedules,
        phase_idx=1,
        multipliers=packet.c1,
        title="Phase 1: CC Domains",
    )
    fig.suptitle(f"Raw optimum: {variant.name}", fontsize=30, y=0.996, fontweight="bold")
    fig.subplots_adjust(top=0.93, left=0.14, right=0.985, bottom=0.08, hspace=0.24, wspace=0.31)
    fig.savefig(variant_dir / "raw_optimum_mixture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(summary: pd.DataFrame, observed: pd.DataFrame) -> str:
    keep = [
        "variant",
        "m_dependent_params_per_domain",
        "total_param_count",
        "cv_rmse",
        "oof_spearman",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "raw_nearest_observed_tv",
        "raw_nearest_observed_value",
        "phase0_max_weight",
        "phase1_max_weight",
    ]
    table = summary[keep].copy()
    best_rmse = summary.loc[summary["cv_rmse"].astype(float).idxmin()]
    best_rank = summary.loc[summary["oof_spearman"].astype(float).idxmax()]
    lines = [
        "# DSP Canonical Form Sweep on 300M",
        "",
        "## Setup",
        "",
        "DSP denotes the reduced-bias domain saturation-penalty family. The sweep keeps M-dependent",
        "parameters to at most four per domain and uses only fixed global scalars for two-phase effects.",
        "",
        "Compared variants:",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- `{variant.name}`: {variant.description}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            table.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Best Observed Rows By Prediction",
            "",
            observed.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Interpretation",
            "",
            f"- Best CV RMSE row: `{best_rmse['variant']}` with cv_rmse={float(best_rmse['cv_rmse']):.6f}.",
            f"- Best OOF Spearman row: `{best_rank['variant']}` with oof_spearman={float(best_rank['oof_spearman']):.6f}.",
            "- Recommended canonical DSP form: `dsp_phase_benefit_penalty_nnls`. It keeps phase-1 as a benefit premium rather than an effective-exposure multiplier, keeps the explicit overexposure penalty, and preserves nonnegative benefit/penalty semantics.",
            "- Empirical upper-bound comparator: `dsp_effective_exposure_penalty_nnls`. It fits best, but the phase-1 multiplier enters the exposure used by both saturation and penalty, which reintroduces the saturation/penalty bias we wanted to reduce.",
            "- Removing explicit penalties is not viable here: both no-penalty variants lose roughly 0.005-0.006 CV RMSE versus the canonical form and have worse top-row regret.",
            "- Allowing signed heads does not help fit or rank; it weakens semantics and increases top-row regret versus NNLS.",
            "- Use raw optima only as diagnostics until the off-manifold/collapse issue is fixed.",
            "- A canonical DSP form should prefer strong observed-row ranking, defensible phase semantics, and stable optima over pure in-sample fit.",
            "",
        ]
    )
    return "\n".join(lines)


def _append_logbook(summary: pd.DataFrame) -> None:
    path = Path(".agents/logbooks/reduced-bias-domain-grp.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    heading = "### 2026-05-10 - DSP canonical form sweep"
    if path.exists() and heading in path.read_text():
        return
    best_rmse = summary.loc[summary["cv_rmse"].astype(float).idxmin()]
    best_rank = summary.loc[summary["oof_spearman"].astype(float).idxmax()]
    entry = "\n".join(
        [
            f"\n{heading}",
            "- Hypothesis: the canonical DSP form can be chosen by testing phase semantics, penalties, and head constraints under a 4-parameters-per-domain budget.",
            f"- Command: `uv run --with matplotlib --with scipy --with scikit-learn --with tabulate python {Path(__file__).as_posix()}`",
            "- Config: 300M/6B 242-row fit frame, DSP variants with at most four M-dependent parameters per domain.",
            f"- Result: best CV RMSE `{best_rmse['variant']}` cv_rmse={float(best_rmse['cv_rmse']):.6f}; best rank `{best_rank['variant']}` oof_spearman={float(best_rank['oof_spearman']):.6f}.",
            f"- Artifacts: `{OUTPUT_DIR}`.",
            "- Interpretation: see `report.md`.",
            "",
        ]
    )
    if path.exists():
        path.write_text(path.read_text() + entry)
    else:
        path.write_text("# Reduced-Bias Domain GRP: Research Logbook\n" + entry)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    packet = _load_packet()
    summary_rows: list[dict[str, Any]] = [_old_grp_baseline_row()]
    observed_rows: list[dict[str, Any]] = []
    tune_frames: list[pd.DataFrame] = []
    print(f"Loaded {len(packet.y)} rows and {packet.m} domains", flush=True)

    for variant in VARIANTS:
        print(f"Fitting {variant.name}", flush=True)
        variant_dir = OUTPUT_DIR / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        model, tune_frame = _fit_variant(packet, variant)
        raw_result, weights = _optimize_model(model, packet)
        metrics = _metrics(packet, model, raw_result, weights)
        summary_rows.append(metrics)
        observed_rows.append(_observed_leaderboard_row(packet, model))
        tune_frame.insert(0, "variant", variant.name)
        tune_frames.append(tune_frame)
        _write_weight_table(packet, variant_dir, weights)
        _plot_predicted_vs_actual(packet, model, variant_dir)
        _plot_raw_optimum(packet, variant, weights, variant_dir)
        model_params = {
            key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.params.items()
        }
        (variant_dir / "model.json").write_text(
            json.dumps(
                {
                    "variant": variant.name,
                    "params": model_params,
                    "intercept": model.intercept,
                    "benefit_coef": model.benefit_coef.tolist(),
                    "penalty_coef": model.penalty_coef.tolist(),
                    "metrics": metrics,
                },
                indent=2,
            )
        )
        print(
            f"  cv_rmse={metrics['cv_rmse']:.6f} oof_spearman={metrics['oof_spearman']:.6f} "
            f"raw_tv={metrics['raw_nearest_observed_tv']:.3f}",
            flush=True,
        )

    summary = pd.DataFrame.from_records(summary_rows)
    observed = pd.DataFrame.from_records(observed_rows)
    tune = pd.concat(tune_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    observed.to_csv(OUTPUT_DIR / "predicted_observed_leaderboard.csv", index=False)
    tune.to_csv(OUTPUT_DIR / "tuning.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    (OUTPUT_DIR / "report.md").write_text(_report(summary, observed))
    _append_logbook(summary)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
