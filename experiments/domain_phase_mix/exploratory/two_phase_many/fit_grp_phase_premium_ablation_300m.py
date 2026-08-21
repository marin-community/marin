# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B023,E501,RUF059

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Fit phase-premium ablations of 300M no-L2 GRP.

This keeps the old GRP no-L2 structural blocks (CC pairs, family totals, family
penalties, quality discount) but removes the effective-exposure form

    exp(-lam * (1 - p1)) * e0 + eta * e1

from the signal and penalty inputs. Instead, saturation and penalties use raw
total exposure, and phase-1 is modeled as a benefit amplitude premium plus an
optional penalty premium:

    phi_g = (1 + gamma * e1_g / (z_g + eps)) * f_g(z_g)
    psi_g = (1 + delta * e1_g / (z_g + eps)) * q_g(z_g)

where z_g is the raw group exposure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    CV_SEED,
    _start_bank,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_and_plot_grp_power_family_penalty_no_l2_60m_vs_300m import (
    RUN_SET_300M,
    _build_fit_frame,
    _metric_frame,
    _packet_from_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    plot_grp_vs_proportional as reference_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    GenericFamilyPacket,
    family_shares,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    CALIBRATION_CV_WEIGHT,
    CALIBRATION_DEPOPT_WEIGHT,
    CALIBRATION_FOLDMEAN_WEIGHT,
    CALIBRATION_SUPPORT_WEIGHT,
    CALIBRATION_TAIL_WEIGHT,
    LOWER_TAIL_FRAC,
    TRUSTBLEND_TOPK_ACTUAL,
    GenericFamilyPenaltyCalibrationSurrogate,
    PenaltyCalibrationVariantSpec,
    _family_tau,
    sigmoid,
    softplus,
    variant_spec,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "grp_phase_benefit_penalty_premium_ablation_300m_20260510"
VARIANT_NAME = "power_family_penalty"
ABLATION_NAME = "power_family_penalty_phase_benefit_penalty_premium_no_l2"
SCALE = "300m_6b"
PHASE_EPS = 1e-9
SPEARMAN_SPLITS = 5
START_TOP_K = 4
REFINE_METHOD = "Powell"
REFINE_MAXITER = 40


class PhasePremiumPenaltyCalibrationSurrogate(GenericFamilyPenaltyCalibrationSurrogate):
    """Old GRP blocks with raw exposure saturation and phase-1 benefit/penalty premia."""

    def __init__(
        self,
        packet: GenericFamilyPacket,
        *,
        params: dict[str, float],
        spec: PenaltyCalibrationVariantSpec,
        phase_eps: float = PHASE_EPS,
    ) -> None:
        if spec.pair_aggregator != "linear":
            raise ValueError("Phase-premium ablation currently expects the linear CC pair aggregator")
        super().__init__(
            packet,
            params=params,
            spec=spec,
            family_totals=GENERIC_FAMILY_NAMES,
            quality_discount=True,
            pair_cc_domains=True,
            include_singletons=True,
            include_pairs=True,
            include_family_totals=True,
            include_global_group_penalty=True,
            include_family_group_penalty=True,
            include_family_total_penalty=True,
        )
        self.phase_eps = float(phase_eps)

    def _raw_exposures(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.packet.base.c0[None, :]
        e1 = p1 * self.packet.base.c1[None, :]
        return e0, e1, e0 + e1

    def _phase_premium_feature(
        self,
        signal_total: np.ndarray,
        phase1_signal_total: np.ndarray,
        *,
        signal_kind: str,
        family_name: str | None = None,
        other_family_name: str | None = None,
        domain_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        gamma = float(self.params["eta"])
        base_signal = self._feature_transform(
            signal_total,
            signal_kind=signal_kind,
            family_name=family_name,
            other_family_name=other_family_name,
            domain_indices=domain_indices,
        )
        premium = 1.0 + gamma * phase1_signal_total / (signal_total + self.phase_eps)
        return premium * base_signal

    def _penalty_feature(
        self,
        total: np.ndarray,
        phase1_total: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        delta = float(self.params["delta"])
        base_penalty = softplus(np.log1p(total) - tau) ** 2
        premium = 1.0 + delta * phase1_total / (total + self.phase_eps)
        return premium * base_penalty

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        e0, e1, z = self._raw_exposures(weights)
        features: list[np.ndarray] = []
        group_totals: list[np.ndarray] = []
        group_phase1_totals: list[np.ndarray] = []
        family_group_totals: dict[str, list[np.ndarray]] = {family_name: [] for family_name in self.family_totals}
        family_group_phase1_totals: dict[str, list[np.ndarray]] = {family_name: [] for family_name in self.family_totals}
        family_totals: dict[str, np.ndarray] = {}
        family_phase1_totals: dict[str, np.ndarray] = {}

        for idx in self.packet.singletons:
            family_name = self.domain_to_family[idx]
            features.append(
                self._phase_premium_feature(
                    z[:, idx],
                    e1[:, idx],
                    signal_kind=self.spec.signal_kind,
                    family_name=family_name,
                    domain_indices=(idx,),
                )[:, None]
            )
            group_totals.append(z[:, idx])
            group_phase1_totals.append(e1[:, idx])
            family_group_totals[family_name].append(z[:, idx])
            family_group_phase1_totals[family_name].append(e1[:, idx])

        beta = float(self.params["beta"])
        for hi, lo in self.packet.pairs:
            hi_family = self.domain_to_family[hi]
            lo_family = self.domain_to_family[lo]
            signal_total = z[:, hi] + beta * z[:, lo]
            phase1_signal_total = e1[:, hi] + beta * e1[:, lo]
            features.append(
                self._phase_premium_feature(
                    signal_total,
                    phase1_signal_total,
                    signal_kind=self.spec.signal_kind,
                    family_name=hi_family,
                    other_family_name=lo_family,
                    domain_indices=(hi, lo),
                )[:, None]
            )
            raw_total = z[:, hi] + z[:, lo]
            raw_phase1_total = e1[:, hi] + e1[:, lo]
            group_totals.append(raw_total)
            group_phase1_totals.append(raw_phase1_total)
            family_group_totals[hi_family].append(raw_total)
            family_group_phase1_totals[hi_family].append(raw_phase1_total)

        for family_name in self.family_totals:
            members = self.packet.family_map[family_name]
            family_total = np.sum(z[:, members], axis=1)
            family_phase1_total = np.sum(e1[:, members], axis=1)
            family_totals[family_name] = family_total
            family_phase1_totals[family_name] = family_phase1_total
            features.append(
                self._phase_premium_feature(
                    family_total,
                    family_phase1_total,
                    signal_kind=self.spec.family_signal_kind,
                    family_name=family_name,
                    domain_indices=tuple(int(member) for member in members),
                )[:, None]
            )

        penalties: list[np.ndarray] = []
        if self.include_global_group_penalty:
            tau = float(self.params["tau"])
            penalty_inputs = np.stack(group_totals, axis=1)
            penalty_phase1_inputs = np.stack(group_phase1_totals, axis=1)
            penalties.append(
                np.sum(self._penalty_feature(penalty_inputs, penalty_phase1_inputs, tau), axis=1, keepdims=True)
            )
        if self.include_family_group_penalty:
            for family_name in self.family_totals:
                tau_f = _family_tau(self.params, family_name)
                penalty_inputs = np.stack(family_group_totals[family_name], axis=1)
                penalty_phase1_inputs = np.stack(family_group_phase1_totals[family_name], axis=1)
                penalties.append(
                    np.sum(
                        self._penalty_feature(penalty_inputs, penalty_phase1_inputs, tau_f),
                        axis=1,
                        keepdims=True,
                    )
                )
        if self.include_family_total_penalty:
            for family_name in self.family_totals:
                tau_f = _family_tau(self.params, family_name)
                penalties.append(
                    self._penalty_feature(
                        family_totals[family_name],
                        family_phase1_totals[family_name],
                        tau_f,
                    )[:, None]
                )

        design = np.hstack(features + penalties)
        design[:, : len(features)] *= -1.0
        return design

    def fit(
        self,
        weights: np.ndarray,
        targets: np.ndarray,
    ) -> PhasePremiumPenaltyCalibrationSurrogate:
        design = self.build_design(weights)
        design_mean = design.mean(axis=0, keepdims=True)
        target_mean = float(targets.mean())
        centered_design = design - design_mean
        centered_targets = targets - target_mean
        coef, _ = nnls(centered_design, centered_targets)
        self.coef_ = coef
        self.intercept_ = float(target_mean - (design_mean @ coef).item())
        return self


def _build_phase_premium_surrogate(
    packet: GenericFamilyPacket,
    params: dict[str, float],
) -> PhasePremiumPenaltyCalibrationSurrogate:
    return PhasePremiumPenaltyCalibrationSurrogate(packet, params=params, spec=variant_spec(VARIANT_NAME))


def _feature_value_and_derivatives(
    model: PhasePremiumPenaltyCalibrationSurrogate,
    signal_total: float,
    phase1_signal_total: float,
    *,
    signal_kind: str,
    family_name: str | None,
    other_family_name: str | None = None,
    domain_indices: tuple[int, ...] | None = None,
) -> tuple[float, float, float]:
    signal_arr = np.asarray([signal_total], dtype=float)
    signal_value = float(
        model._feature_transform(
            signal_arr,
            signal_kind=signal_kind,
            family_name=family_name,
            other_family_name=other_family_name,
            domain_indices=domain_indices,
        )[0]
    )
    signal_derivative = float(
        model._feature_derivative(
            signal_arr,
            signal_kind=signal_kind,
            family_name=family_name,
            other_family_name=other_family_name,
            domain_indices=domain_indices,
        )[0]
    )
    gamma = float(model.params["eta"])
    denom = signal_total + model.phase_eps
    premium = 1.0 + gamma * phase1_signal_total / denom
    feature = premium * signal_value
    d_feature_d_signal = premium * signal_derivative - signal_value * gamma * phase1_signal_total / (denom * denom)
    d_feature_d_phase1_signal = signal_value * gamma / denom
    return float(feature), float(d_feature_d_signal), float(d_feature_d_phase1_signal)


def _penalty_value_and_derivatives(
    model: PhasePremiumPenaltyCalibrationSurrogate,
    total: float,
    phase1_total: float,
    tau: float,
) -> tuple[float, float, float]:
    delta = float(model.params["delta"])
    denom = total + model.phase_eps
    inside = np.log1p(total) - tau
    sp = float(softplus(inside))
    base = sp * sp
    premium = 1.0 + delta * phase1_total / denom
    base_derivative = 2.0 * sp * float(sigmoid(inside)) / (1.0 + total)
    value = premium * base
    d_total = premium * base_derivative - base * delta * phase1_total / (denom * denom)
    d_phase1_total = base * delta / denom
    return float(value), float(d_total), float(d_phase1_total)


def _optimize_phase_premium_model(
    packet: GenericFamilyPacket,
    model: PhasePremiumPenaltyCalibrationSurrogate,
    *,
    n_random: int = 1,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit before optimization")

    parts = model.components()
    singleton_coef = parts["singleton_coef"]
    pair_coef = parts["pair_coef"]
    family_coef = parts["family_coef"]
    global_penalty_coef = float(parts["global_penalty_coef"])
    family_group_penalty_coef = {
        family_name: float(value) for family_name, value in parts["family_group_penalty_coef"].items()
    }
    family_total_penalty_coef = {
        family_name: float(value) for family_name, value in parts["family_total_penalty_coef"].items()
    }

    n_domains = packet.base.m
    c0 = packet.base.c0
    c1 = packet.base.c1
    family_indices = {
        family_name: np.asarray(packet.family_map[family_name], dtype=int) for family_name in model.family_totals
    }
    rng = np.random.default_rng(seed)

    def value_grad_logits(logits: np.ndarray) -> tuple[float, np.ndarray]:
        logits0 = logits[:n_domains]
        logits1 = logits[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 /= np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 /= np.sum(p1)

        e0 = c0 * p0
        e1 = c1 * p1
        z = e0 + e1
        value = float(model.intercept_)
        grad_e0 = np.zeros(n_domains, dtype=float)
        grad_e1 = np.zeros(n_domains, dtype=float)
        group_info: list[tuple[tuple[int, ...], float, float, str]] = []
        family_total_values: dict[str, float] = {}

        for local_idx, domain_idx in enumerate(packet.singletons):
            family_name = model.domain_to_family[domain_idx]
            feature, d_signal, d_phase1_signal = _feature_value_and_derivatives(
                model,
                float(z[domain_idx]),
                float(e1[domain_idx]),
                signal_kind=model.spec.signal_kind,
                family_name=family_name,
                domain_indices=(domain_idx,),
            )
            coef = float(singleton_coef[local_idx])
            value -= coef * feature
            grad_e0[domain_idx] -= coef * d_signal
            grad_e1[domain_idx] -= coef * (d_signal + d_phase1_signal)
            group_info.append(((domain_idx,), float(z[domain_idx]), float(e1[domain_idx]), family_name))

        beta = float(model.params["beta"])
        for local_idx, (hi, lo) in enumerate(packet.pairs):
            hi_family = model.domain_to_family[hi]
            lo_family = model.domain_to_family[lo]
            signal_total = float(z[hi] + beta * z[lo])
            phase1_signal_total = float(e1[hi] + beta * e1[lo])
            feature, d_signal, d_phase1_signal = _feature_value_and_derivatives(
                model,
                signal_total,
                phase1_signal_total,
                signal_kind=model.spec.signal_kind,
                family_name=hi_family,
                other_family_name=lo_family,
                domain_indices=(hi, lo),
            )
            coef = float(pair_coef[local_idx])
            value -= coef * feature
            grad_e0[hi] -= coef * d_signal
            grad_e1[hi] -= coef * (d_signal + d_phase1_signal)
            grad_e0[lo] -= coef * beta * d_signal
            grad_e1[lo] -= coef * beta * (d_signal + d_phase1_signal)
            raw_total = float(z[hi] + z[lo])
            raw_phase1_total = float(e1[hi] + e1[lo])
            group_info.append(((hi, lo), raw_total, raw_phase1_total, hi_family))

        for family_name in model.family_totals:
            members = family_indices[family_name]
            family_total = float(np.sum(z[members]))
            family_phase1_total = float(np.sum(e1[members]))
            family_total_values[family_name] = family_total
            coef = float(family_coef[family_name])
            feature, d_signal, d_phase1_signal = _feature_value_and_derivatives(
                model,
                family_total,
                family_phase1_total,
                signal_kind=model.spec.family_signal_kind,
                family_name=family_name,
                domain_indices=tuple(int(member) for member in members),
            )
            value -= coef * feature
            grad_e0[members] -= coef * d_signal
            grad_e1[members] -= coef * (d_signal + d_phase1_signal)

        if model.include_global_group_penalty and global_penalty_coef != 0.0:
            tau = float(model.params["tau"])
            for members, total, phase1_total, _family_name in group_info:
                penalty_value, d_total, d_phase1_total = _penalty_value_and_derivatives(
                    model,
                    total,
                    phase1_total,
                    tau,
                )
                value += global_penalty_coef * penalty_value
                for idx in members:
                    grad_e0[idx] += global_penalty_coef * d_total
                    grad_e1[idx] += global_penalty_coef * (d_total + d_phase1_total)

        if model.include_family_group_penalty:
            for members, total, phase1_total, family_name in group_info:
                coef = float(family_group_penalty_coef[family_name])
                if coef == 0.0:
                    continue
                tau_f = _family_tau(model.params, family_name)
                penalty_value, d_total, d_phase1_total = _penalty_value_and_derivatives(
                    model,
                    total,
                    phase1_total,
                    tau_f,
                )
                value += coef * penalty_value
                for idx in members:
                    grad_e0[idx] += coef * d_total
                    grad_e1[idx] += coef * (d_total + d_phase1_total)

        if model.include_family_total_penalty:
            for family_name in model.family_totals:
                coef = float(family_total_penalty_coef[family_name])
                if coef == 0.0:
                    continue
                total = family_total_values[family_name]
                phase1_total = float(np.sum(e1[family_indices[family_name]]))
                tau_f = _family_tau(model.params, family_name)
                penalty_value, d_total, d_phase1_total = _penalty_value_and_derivatives(
                    model,
                    total,
                    phase1_total,
                    tau_f,
                )
                value += coef * penalty_value
                members = family_indices[family_name]
                grad_e0[members] += coef * d_total
                grad_e1[members] += coef * (d_total + d_phase1_total)

        grad_p0 = grad_e0 * c0
        grad_p1 = grad_e1 * c1
        grad_logits0 = p0 * (grad_p0 - np.dot(grad_p0, p0))
        grad_logits1 = p1 * (grad_p1 - np.dot(grad_p1, p1))
        return value, np.concatenate([grad_logits0, grad_logits1])

    starts = [np.zeros(2 * n_domains, dtype=float)]
    starts.extend(
        np.concatenate([rng.normal(scale=0.2, size=n_domains), rng.normal(scale=0.2, size=n_domains)])
        for _ in range(n_random)
    )
    best = None
    for start in starts:
        result = minimize(
            lambda z: value_grad_logits(z)[0],
            start,
            jac=lambda z: value_grad_logits(z)[1],
            method="L-BFGS-B",
            options={"maxiter": 400},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("Phase-premium optimization failed")

    logits = np.asarray(best.x, dtype=float)
    logits0 = logits[:n_domains]
    logits1 = logits[n_domains:]
    phase0 = np.exp(logits0 - np.max(logits0))
    phase0 /= np.sum(phase0)
    phase1 = np.exp(logits1 - np.max(logits1))
    phase1 /= np.sum(phase1)
    return best, phase0, phase1


def _param_keys() -> tuple[str, ...]:
    return (
        "eta",
        "delta",
        "beta",
        "a_broad_text",
        "a_tech_code",
        "a_reasoning",
        "tau_broad_text",
        "tau_tech_code",
        "tau_reasoning",
    )


def _pack_params(params: dict[str, float]) -> np.ndarray:
    beta = float(np.clip(params["beta"], 1e-8, 1.0 - 1.0e-8))
    return np.asarray(
        [
            np.log(float(params["eta"])),
            np.log(float(params["delta"])),
            np.log(beta / (1.0 - beta)),
            np.log(float(params["a_broad_text"])),
            np.log(float(params["a_tech_code"])),
            np.log(float(params["a_reasoning"])),
            float(params["tau_broad_text"]),
            float(params["tau_tech_code"]),
            float(params["tau_reasoning"]),
        ],
        dtype=float,
    )


def _unpack_params(z: np.ndarray) -> dict[str, float]:
    values = np.asarray(z, dtype=float)
    eta = float(np.exp(np.clip(values[0], -8.0, 8.0)))
    delta = float(np.exp(np.clip(values[1], -8.0, 8.0)))
    beta = float(np.clip(1.0 / (1.0 + np.exp(-np.clip(values[2], -50.0, 50.0))), 1e-6, 1.0 - 1e-6))
    return {
        "eta": eta,
        "delta": delta,
        "lam": 0.0,
        "reg": 0.0,
        "beta": beta,
        "a_broad_text": float(np.exp(np.clip(values[3], np.log(0.02), np.log(2.0)))),
        "a_tech_code": float(np.exp(np.clip(values[4], np.log(0.02), np.log(2.0)))),
        "a_reasoning": float(np.exp(np.clip(values[5], np.log(0.02), np.log(2.0)))),
        "tau_broad_text": float(np.clip(values[6], -2.0, 8.0)),
        "tau_tech_code": float(np.clip(values[7], -2.0, 8.0)),
        "tau_reasoning": float(np.clip(values[8], -2.0, 8.0)),
    }


def _with_params(base: dict[str, float], **updates: float) -> dict[str, float]:
    params = {key: float(base[key]) for key in _param_keys() if key in base}
    params.update({key: float(value) for key, value in updates.items()})
    params["reg"] = 0.0
    params["lam"] = 0.0
    return params


def _start_params() -> tuple[dict[str, float], ...]:
    rows: list[dict[str, float]] = []
    for old in _start_bank():
        rows.append(
            _with_params(
                old,
                eta=max(float(old["eta"]) - 1.0, 1e-3),
                delta=1.0,
                beta=float(old["beta"]),
            )
        )
    seed = rows[0]
    rows.extend(
        [
            _with_params(seed, eta=0.25),
            _with_params(seed, eta=1.0),
            _with_params(seed, eta=4.0),
            _with_params(seed, eta=12.0),
            _with_params(seed, eta=1.0, delta=0.25, beta=0.5),
            _with_params(seed, eta=4.0, delta=1.0, beta=0.9),
            _with_params(seed, eta=4.0, delta=4.0),
            _with_params(seed, eta=12.0, delta=12.0),
        ]
    )
    deduped: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for row in rows:
        key = tuple(sorted((key, round(float(value), 8)) for key, value in row.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return tuple(deduped)


def _parameter_counts(packet: GenericFamilyPacket) -> dict[str, int]:
    signal_feature_count = len(packet.singletons) + len(packet.pairs) + len(GENERIC_FAMILY_NAMES)
    penalty_feature_count = len(GENERIC_FAMILY_NAMES)
    linear_coefficient_count = signal_feature_count + penalty_feature_count
    nonlinear_param_count = len(_param_keys())
    return {
        "signal_feature_count": signal_feature_count,
        "penalty_feature_count": penalty_feature_count,
        "linear_coefficient_count": linear_coefficient_count,
        "intercept_count": 1,
        "linear_head_param_count": linear_coefficient_count + 1,
        "nonlinear_param_count": nonlinear_param_count,
        "total_param_count": linear_coefficient_count + 1 + nonlinear_param_count,
    }


def _oof_metrics(packet: GenericFamilyPacket, params: dict[str, float]) -> dict[str, float]:
    y = packet.base.y
    kf = KFold(n_splits=SPEARMAN_SPLITS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    depopt_scores: list[float] = []
    rawopt_support_scores: list[float] = []

    for train_idx, test_idx in kf.split(packet.base.w):
        model = _build_phase_premium_surrogate(packet, params).fit(packet.base.w[train_idx], y[train_idx])
        pred = model.predict(packet.base.w[test_idx])
        oof[test_idx] = pred
        fold_regrets.append(float(y[test_idx][int(np.argmin(pred))] - np.min(y[test_idx])))

        raw_result, phase0, phase1 = _optimize_phase_premium_model(packet, model, n_random=1, seed=CV_SEED)
        raw_weights = np.stack([phase0, phase1], axis=0)
        distances = average_phase_tv_distance(packet.base.w[test_idx], raw_weights[None, :, :])
        nearest_count = min(TRUSTBLEND_TOPK_ACTUAL, len(test_idx))
        nearest_idx = np.argsort(distances)[:nearest_count]
        depopt_scores.append(max(float(np.min(y[test_idx][nearest_idx])) - float(raw_result.fun), 0.0))
        rawopt_support_scores.append(float(distances[nearest_idx[0]]))

    residuals = oof - y
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    tail_count = max(5, int(np.ceil(float(LOWER_TAIL_FRAC) * float(len(y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    mean_depopt = float(np.mean(depopt_scores))
    mean_support = float(np.mean(rawopt_support_scores))
    objective = (
        CALIBRATION_CV_WEIGHT * cv_rmse
        + CALIBRATION_FOLDMEAN_WEIGHT * mean_regret
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
        + CALIBRATION_DEPOPT_WEIGHT * mean_depopt
        + CALIBRATION_SUPPORT_WEIGHT * mean_support
    )
    return {
        "cv_rmse": cv_rmse,
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": mean_regret,
        "lower_tail_optimism": lower_tail_optimism,
        "cv_depopt_best8": mean_depopt,
        "cv_rawopt_nearest_tv": mean_support,
        "objective": float(objective),
        "oof_spearman": float(spearmanr(y, oof).statistic),
        "oof_pearson": float(pearsonr(y, oof).statistic),
    }


def _fast_oof_metrics(packet: GenericFamilyPacket, params: dict[str, float]) -> dict[str, float]:
    """Compute CV fit metrics without raw-optimum diagnostics for fast nonlinear tuning."""
    y = packet.base.y
    kf = KFold(n_splits=SPEARMAN_SPLITS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []

    for train_idx, test_idx in kf.split(packet.base.w):
        model = _build_phase_premium_surrogate(packet, params).fit(packet.base.w[train_idx], y[train_idx])
        pred = model.predict(packet.base.w[test_idx])
        oof[test_idx] = pred
        fold_regrets.append(float(y[test_idx][int(np.argmin(pred))] - np.min(y[test_idx])))

    residuals = oof - y
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    tail_count = max(5, int(np.ceil(float(LOWER_TAIL_FRAC) * float(len(y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    objective = (
        CALIBRATION_CV_WEIGHT * cv_rmse
        + CALIBRATION_FOLDMEAN_WEIGHT * mean_regret
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
    )
    return {
        "cv_rmse": cv_rmse,
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": mean_regret,
        "lower_tail_optimism": lower_tail_optimism,
        "cv_depopt_best8": np.nan,
        "cv_rawopt_nearest_tv": np.nan,
        "objective": float(objective),
        "oof_spearman": float(spearmanr(y, oof).statistic),
        "oof_pearson": float(pearsonr(y, oof).statistic),
    }


def _full_metrics(packet: GenericFamilyPacket, params: dict[str, float]) -> tuple[dict[str, Any], np.ndarray]:
    model = _build_phase_premium_surrogate(packet, params).fit(packet.base.w, packet.base.y)
    oof = _oof_metrics(packet, params)
    train_pred = model.predict(packet.base.w)
    raw_result, phase0, phase1 = _optimize_phase_premium_model(packet, model, n_random=20, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    raw_distances = average_phase_tv_distance(packet.base.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    metrics: dict[str, Any] = {
        **oof,
        "train_rmse": float(np.sqrt(np.mean((train_pred - packet.base.y) ** 2))),
        "train_spearman": float(spearmanr(packet.base.y, train_pred).statistic),
        "train_pearson": float(pearsonr(packet.base.y, train_pred).statistic),
        "raw_predicted_optimum_value": float(raw_result.fun),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "raw_nearest_observed_value": float(packet.base.y[nearest_idx]),
        "raw_phase0_lt_1e4": int(np.sum(phase0 < 1e-4)),
        "raw_phase1_lt_1e4": int(np.sum(phase1 < 1e-4)),
        "phase0_max_weight": float(np.max(phase0)),
        "phase1_max_weight": float(np.max(phase1)),
        "optimum_success": bool(raw_result.success),
        "optimum_message": str(raw_result.message),
    }
    metrics.update(family_shares(packet, raw_weights))
    return metrics, raw_weights


def _fit_phase_premium(packet: GenericFamilyPacket) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray]:
    starts = _start_params()
    coarse_rows = [
        {"stage": "coarse", "start_id": int(start_id), **params, **_fast_oof_metrics(packet, params)}
        for start_id, params in enumerate(starts)
    ]
    coarse_frame = pd.DataFrame.from_records(coarse_rows).sort_values(
        ["objective", "cv_rmse", "cv_depopt_best8"], ascending=[True, True, True]
    )

    refine_rows: list[dict[str, Any]] = []

    for rank, start_id in enumerate(coarse_frame["start_id"].head(START_TOP_K).tolist()):
        start = _pack_params(starts[int(start_id)])
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in cache:
                metrics = _fast_oof_metrics(packet, _unpack_params(z))
                cache[key] = float(metrics["objective"])
            return cache[key]

        result = minimize(
            objective,
            start,
            method=REFINE_METHOD,
            options={"maxiter": REFINE_MAXITER, "xtol": 1e-4, "ftol": 1e-6},
        )
        params = _unpack_params(np.asarray(result.x, dtype=float))
        metrics = _fast_oof_metrics(packet, params)
        row: dict[str, Any] = {
            "stage": "refine",
            "chosen_rank": int(rank),
            "start_id": int(start_id),
            "success": bool(result.success),
            "message": str(result.message),
            **params,
            **metrics,
        }
        refine_rows.append(row)

    if not refine_rows:
        raise RuntimeError("No phase-premium fit result")

    full_refine_rows: list[dict[str, Any]] = []
    for row in refine_rows:
        params = {key: float(row[key]) for key in _param_keys()}
        params.update({"reg": 0.0, "lam": 0.0})
        full_metrics = _oof_metrics(packet, params)
        full_refine_rows.append(
            {
                **{f"fast_{key}": value for key, value in row.items() if key in full_metrics},
                **{key: value for key, value in row.items() if key not in full_metrics},
                **full_metrics,
            }
        )
    selected_row = min(full_refine_rows, key=lambda row: float(row["objective"]))
    best_params = {key: float(selected_row[key]) for key in _param_keys()}
    best_params.update({"reg": 0.0, "lam": 0.0})

    final_metrics, weights = _full_metrics(packet, best_params)
    summary = {
        "variant": ABLATION_NAME,
        "fit_scale": SCALE,
        "fit_run_set": RUN_SET_300M,
        "fit_row_count": len(packet.base.y),
        "optimizer_success": bool(selected_row["success"]),
        "optimizer_message": str(selected_row["message"]),
        **best_params,
        **final_metrics,
        **_parameter_counts(packet),
    }
    return coarse_frame, pd.DataFrame.from_records(full_refine_rows), summary, weights


def _baseline_row(packet: GenericFamilyPacket) -> dict[str, Any]:
    old_summary = SCRIPT_DIR / "grp_power_family_penalty_no_l2_60m_vs_300m_fit_summary.csv"
    frame = pd.read_csv(old_summary)
    row = frame.loc[frame["label"].eq("300M-fit no-$L_2$ GRP")].iloc[0].to_dict()
    return {
        "variant": "power_family_penalty_effective_exposure_no_l2",
        "fit_scale": row["fit_scale"],
        "fit_run_set": row["fit_run_set"],
        "fit_row_count": int(row["fit_row_count"]),
        "optimizer_success": bool(row["optimizer_success"]),
        "optimizer_message": str(row["optimizer_message"]),
        "eta": float(row["param_eta"]),
        "lam": float(row["param_lam"]),
        "reg": float(row["param_reg"]),
        "beta": float(row["param_beta"]),
        "a_broad_text": float(row["param_a_broad_text"]),
        "a_tech_code": float(row["param_a_tech_code"]),
        "a_reasoning": float(row["param_a_reasoning"]),
        "tau_broad_text": float(row["param_tau_broad_text"]),
        "tau_tech_code": float(row["param_tau_tech_code"]),
        "tau_reasoning": float(row["param_tau_reasoning"]),
        "train_rmse": float(row["train_rmse"]),
        "train_spearman": float(row["train_spearman"]),
        "cv_rmse": float(row["cv_rmse"]),
        "cv_foldmean_regret_at_1": float(row["cv_foldmean_regret_at_1"]),
        "cv_regret_at_1": float(row["cv_regret_at_1"]),
        "lower_tail_optimism": float(row["lower_tail_optimism"]),
        "cv_depopt_best8": float(row["cv_depopt_best8"]),
        "cv_rawopt_nearest_tv": float(row["cv_rawopt_nearest_tv"]),
        "oof_spearman": float(row["oof_spearman"]),
        "raw_predicted_optimum_value": float(row["predicted_optimum_value"]),
        "raw_nearest_observed_run_name": str(row["nearest_observed_run_name"]),
        "raw_nearest_observed_value": float(row["nearest_observed_value"]),
        "raw_nearest_observed_tv": float(row["nearest_observed_tv"]),
        "phase0_max_weight": float(row["phase0_max_weight"]),
        "phase1_max_weight": float(row["phase1_max_weight"]),
        "phase0_broad_text": float(row["phase0_broad_text"]),
        "phase0_tech_code": float(row["phase0_tech_code"]),
        "phase0_reasoning": float(row["phase0_reasoning"]),
        "phase1_broad_text": float(row["phase1_broad_text"]),
        "phase1_tech_code": float(row["phase1_tech_code"]),
        "phase1_reasoning": float(row["phase1_reasoning"]),
        "total_param_count": int(row["total_param_count"]),
        "notes": "Existing 300M no-L2 GRP summary; phase-1 multiplier is inside effective exposure.",
    }


def _plot_predicted_vs_actual(packet: GenericFamilyPacket, params: dict[str, float], output_path: Path) -> None:
    model = _build_phase_premium_surrogate(packet, params).fit(packet.base.w, packet.base.y)
    pred = model.predict(packet.base.w)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    ax.scatter(packet.base.y, pred, s=30, alpha=0.75, color="#0f766e", edgecolor="white", linewidth=0.4)
    lo = min(float(packet.base.y.min()), float(pred.min()))
    hi = max(float(packet.base.y.max()), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#64748b", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Actual eval/uncheatable_eval/bpb")
    ax.set_ylabel("Predicted BPB")
    ax.set_title("Phase-premium GRP on 300M swarm")
    ax.grid(True, color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_raw_optimum(packet: GenericFamilyPacket, weights: np.ndarray, output_path: Path) -> None:
    color = plt.get_cmap("RdYlGn_r")(0.15)
    schedules = [("Phase-premium raw optimum", weights, color)]
    canonical_non_cc_indices, canonical_cc_indices = reference_plot._grp_domain_order(packet.base.domain_names, weights)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(26, 22),
        gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.22, "wspace": 0.31},
        facecolor="white",
    )
    reference_plot._plot_non_cc_block(
        ax=axes[0, 0],
        indices=canonical_non_cc_indices,
        labels=[reference_plot._display_non_cc_label(packet.base.domain_names[idx]) for idx in canonical_non_cc_indices],
        schedules=schedules,
        phase_idx=0,
        multipliers=packet.base.c0,
        title="Phase 0: Non-CC Domains",
        show_legend=True,
    )
    reference_plot._plot_cc_block(
        ax=axes[0, 1],
        domain_names=packet.base.domain_names,
        indices=canonical_cc_indices,
        schedules=schedules,
        phase_idx=0,
        multipliers=packet.base.c0,
        title="Phase 0: CC Domains",
    )
    reference_plot._plot_non_cc_block(
        ax=axes[1, 0],
        indices=canonical_non_cc_indices,
        labels=[reference_plot._display_non_cc_label(packet.base.domain_names[idx]) for idx in canonical_non_cc_indices],
        schedules=schedules,
        phase_idx=1,
        multipliers=packet.base.c1,
        title="Phase 1: Non-CC Domains",
        show_legend=False,
    )
    reference_plot._plot_cc_block(
        ax=axes[1, 1],
        domain_names=packet.base.domain_names,
        indices=canonical_cc_indices,
        schedules=schedules,
        phase_idx=1,
        multipliers=packet.base.c1,
        title="Phase 1: CC Domains",
    )
    fig.suptitle("Raw optimum for phase-premium no-L2 GRP ablation", fontsize=32, y=0.996, fontweight="bold")
    fig.subplots_adjust(top=0.93, left=0.14, right=0.985, bottom=0.08, hspace=0.24, wspace=0.31)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(summary_frame: pd.DataFrame) -> str:
    keep = [
        "variant",
        "total_param_count",
        "train_rmse",
        "oof_spearman",
        "cv_rmse",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "cv_depopt_best8",
        "cv_rawopt_nearest_tv",
        "raw_nearest_observed_tv",
        "raw_nearest_observed_value",
        "phase0_max_weight",
        "phase1_max_weight",
    ]
    table = summary_frame[keep].copy()
    old = summary_frame.loc[summary_frame["variant"].eq("power_family_penalty_effective_exposure_no_l2")].iloc[0]
    new = summary_frame.loc[summary_frame["variant"].eq(ABLATION_NAME)].iloc[0]
    conclusion = (
        "Mixed but not deployable: adding a matched phase-1 penalty premium improves rank metrics but worsens RMSE "
        "and raw-optimum sanity. "
        f"CV RMSE increases from {float(old['cv_rmse']):.6f} to {float(new['cv_rmse']):.6f}; "
        f"OOF Spearman improves from {float(old['oof_spearman']):.6f} to {float(new['oof_spearman']):.6f}; "
        f"CV foldmean regret improves from {float(old['cv_foldmean_regret_at_1']):.6f} to "
        f"{float(new['cv_foldmean_regret_at_1']):.6f}; but the raw-optimum calibration penalty rises from "
        f"{float(old['cv_depopt_best8']):.6f} to {float(new['cv_depopt_best8']):.6f}."
    )
    return "\n".join(
        [
            "# 300M GRP Phase Benefit/Penalty Premium Ablation",
            "",
            "## Question",
            "",
            "Does replacing the old GRP effective-exposure phase-1 multiplier with separate benefit and penalty premia improve the 300M fit?",
            "",
            "The ablation keeps the old GRP blocks but changes the feature for each singleton, CC pair, and family total:",
            "",
            r"$$\phi_g=(1+\gamma r_g) f_g(z_g),\qquad r_g=\frac{e_{1,g}}{z_g+\epsilon}.$$",
            "",
            r"$$\psi_g=(1+\delta r_g) q_g(z_g).$$",
            "",
            "Both benefits and penalties use raw exposure `z_g`; phase 1 no longer saturates earlier by construction, but it can carry a separate overexposure premium through `delta`.",
            "Nonlinear parameters were selected with a fast CV objective that excludes raw-optimum diagnostics; the final selected row was then evaluated with the full raw-optimum diagnostics shown below.",
            "",
            "## Results",
            "",
            table.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Interpretation",
            "",
            conclusion,
            "",
            "- The benefit/penalty premium model is useful only if it improves CV fit and reduces raw-optimum degeneracy relative to the existing effective-exposure no-L2 GRP.",
            "- `cv_rawopt_nearest_tv` and `raw_nearest_observed_tv` should be treated as optimizer-sanity metrics; lower is better.",
            "- `cv_depopt_best8` is a calibration penalty for raw optima that predict substantially better than nearby observed mixtures.",
            "",
            "## Artifacts",
            "",
            "- `summary.csv`: comparison table.",
            "- `coarse.csv` and `refine.csv`: nonlinear tuning records.",
            "- `predicted_vs_actual.png`: in-sample fit plot for the benefit/penalty premium ablation.",
            "- `raw_optimum_mixture.png`: raw optimum mixture plot.",
            "",
        ]
    )


def _append_logbook(summary_frame: pd.DataFrame) -> None:
    logbook = Path(".agents/logbooks/grp-phase-bias-ablation.md")
    logbook.parent.mkdir(parents=True, exist_ok=True)
    row = summary_frame.loc[summary_frame["variant"].eq(ABLATION_NAME)].iloc[0]
    heading = "### 2026-05-10 - 300M phase benefit+penalty premium no-L2 GRP ablation"
    if logbook.exists() and heading in logbook.read_text():
        return
    text = "\n".join(
        [
            f"\n{heading}",
            "- Hypothesis: the old GRP phase-1 multiplier inside effective exposure may incorrectly force phase-1 exposure to saturate earlier; a benefit premium plus matched penalty premium could preserve phase-1 value without the effective-exposure coupling.",
            f"- Command: `uv run python {Path(__file__).as_posix()}`",
            "- Config: 300M/6B qsplit-core + Olmix/Uniform baseline fit frame, old `power_family_penalty` GRP blocks, no L2, phase benefit premium and penalty premium outside raw exposure saturation.",
            (
                "- Result: "
                f"cv_rmse={float(row['cv_rmse']):.6f}, "
                f"oof_spearman={float(row['oof_spearman']):.6f}, "
                f"cv_foldmean_regret_at_1={float(row['cv_foldmean_regret_at_1']):.6f}, "
                f"cv_rawopt_nearest_tv={float(row['cv_rawopt_nearest_tv']):.6f}, "
                f"raw_nearest_observed_tv={float(row['raw_nearest_observed_tv']):.6f}."
            ),
            f"- Artifacts: `{OUTPUT_DIR}`.",
            "- Interpretation: see `report.md` in the artifact directory.",
            "",
        ]
    )
    if logbook.exists():
        logbook.write_text(logbook.read_text() + text)
    else:
        logbook.write_text(
            "\n".join(
                [
                    "# GRP Phase-Bias Ablation: Research Logbook",
                    "",
                    "## Scope",
                    "- Goal: test whether old GRP's phase-1 effective-exposure multiplier is a harmful inductive bias.",
                    "- Primary metric: 300M/6B `eval/uncheatable_eval/bpb` fit and raw-optimum sanity.",
                    "- Constraint: local modeling only; no training launch.",
                    text,
                ]
            )
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_frame = _build_fit_frame(_metric_frame(), scale=SCALE, run_set=RUN_SET_300M)
    packet = _packet_from_frame(fit_frame, name="grp_phase_premium_ablation_300m")

    coarse_frame, refine_frame, summary, weights = _fit_phase_premium(packet)
    baseline = _baseline_row(packet)
    summary_frame = pd.DataFrame.from_records([baseline, summary])

    coarse_frame.to_csv(OUTPUT_DIR / "coarse.csv", index=False)
    refine_frame.to_csv(OUTPUT_DIR / "refine.csv", index=False)
    summary_frame.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_frame.to_dict(orient="records"), indent=2))
    weight_rows = []
    for domain_name, phase0, phase1, c0, c1 in zip(
        packet.base.domain_names, weights[0], weights[1], packet.base.c0, packet.base.c1, strict=True
    ):
        weight_rows.append(
            {
                "domain_name": domain_name,
                "phase_0_weight": float(phase0),
                "phase_1_weight": float(phase1),
                "phase_0_effective_epochs": float(phase0 * c0),
                "phase_1_effective_epochs": float(phase1 * c1),
            }
        )
    pd.DataFrame.from_records(weight_rows).to_csv(OUTPUT_DIR / "raw_optimum_weights.csv", index=False)

    phase_params = {key: float(summary[key]) for key in _param_keys()}
    phase_params.update({"reg": 0.0, "lam": 0.0})
    _plot_predicted_vs_actual(packet, phase_params, OUTPUT_DIR / "predicted_vs_actual.png")
    _plot_raw_optimum(packet, weights, OUTPUT_DIR / "raw_optimum_mixture.png")
    (OUTPUT_DIR / "report.md").write_text(_markdown_report(summary_frame))
    _append_logbook(summary_frame)
    print(summary_frame.to_string(index=False))
    print(f"Wrote artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
