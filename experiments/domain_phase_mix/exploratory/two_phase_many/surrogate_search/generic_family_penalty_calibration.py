# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibration-oriented GRP variants with richer penalties and pair CES."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_flexible_signal import (
    domain_exponent_key,
    signal_derivative,
    signal_transform,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    GenericFamilyPacket,
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    sigmoid,
    softplus,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

CALIBRATION_CV_WEIGHT = 1.0
CALIBRATION_FOLDMEAN_WEIGHT = 0.05
CALIBRATION_TAIL_WEIGHT = 0.5
CALIBRATION_DEPOPT_WEIGHT = 0.1
CALIBRATION_SUPPORT_WEIGHT = 0.01
LOWER_TAIL_FRAC = 0.15
TRUSTBLEND_TOPK_ACTUAL = 8
TRUSTBLEND_LINE_GRID = 81


@dataclass(frozen=True)
class PenaltyCalibrationVariantSpec:
    """Structural configuration for a calibration-oriented GRP variant."""

    name: str
    signal_kind: str
    family_signal_kind: str
    family_curvature: bool
    global_group_penalty: bool
    family_group_penalty: bool
    family_total_penalty: bool
    domain_curvature: bool = False
    pair_aggregator: str = "linear"


VARIANT_SPECS = {
    "power_family": PenaltyCalibrationVariantSpec(
        name="power_family",
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=True,
        global_group_penalty=True,
        family_group_penalty=False,
        family_total_penalty=False,
    ),
    "power_boxcox_family": PenaltyCalibrationVariantSpec(
        name="power_boxcox_family",
        signal_kind="power",
        family_signal_kind="boxcox",
        family_curvature=True,
        global_group_penalty=True,
        family_group_penalty=False,
        family_total_penalty=False,
    ),
    "power_family_penalty": PenaltyCalibrationVariantSpec(
        name="power_family_penalty",
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=True,
        domain_curvature=False,
        global_group_penalty=False,
        family_group_penalty=True,
        family_total_penalty=False,
    ),
    "power_shared_penalty": PenaltyCalibrationVariantSpec(
        name="power_shared_penalty",
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=False,
        domain_curvature=False,
        global_group_penalty=False,
        family_group_penalty=True,
        family_total_penalty=False,
    ),
    "power_domain_penalty": PenaltyCalibrationVariantSpec(
        name="power_domain_penalty",
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=False,
        domain_curvature=True,
        global_group_penalty=False,
        family_group_penalty=True,
        family_total_penalty=False,
    ),
    "power_boxcox_family_penalty": PenaltyCalibrationVariantSpec(
        name="power_boxcox_family_penalty",
        signal_kind="power",
        family_signal_kind="boxcox",
        family_curvature=True,
        domain_curvature=False,
        global_group_penalty=False,
        family_group_penalty=True,
        family_total_penalty=False,
    ),
    "power_family_penalty_global_ftotal": PenaltyCalibrationVariantSpec(
        name="power_family_penalty_global_ftotal",
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=True,
        global_group_penalty=True,
        family_group_penalty=True,
        family_total_penalty=True,
    ),
    "power_shared_penalty_global_ftotal": PenaltyCalibrationVariantSpec(
        name="power_shared_penalty_global_ftotal",
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=False,
        global_group_penalty=True,
        family_group_penalty=True,
        family_total_penalty=True,
    ),
    "power_boxcox_family_penalty_global_ftotal": PenaltyCalibrationVariantSpec(
        name="power_boxcox_family_penalty_global_ftotal",
        signal_kind="power",
        family_signal_kind="boxcox",
        family_curvature=True,
        global_group_penalty=True,
        family_group_penalty=True,
        family_total_penalty=True,
    ),
    "power_family_penalty_global_ftotal_pairces": PenaltyCalibrationVariantSpec(
        name="power_family_penalty_global_ftotal_pairces",
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=True,
        global_group_penalty=True,
        family_group_penalty=True,
        family_total_penalty=True,
        pair_aggregator="ces",
    ),
    "power_shared_penalty_global_ftotal_pairces": PenaltyCalibrationVariantSpec(
        name="power_shared_penalty_global_ftotal_pairces",
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=False,
        global_group_penalty=True,
        family_group_penalty=True,
        family_total_penalty=True,
        pair_aggregator="ces",
    ),
    "power_boxcox_family_penalty_global_ftotal_pairces": PenaltyCalibrationVariantSpec(
        name="power_boxcox_family_penalty_global_ftotal_pairces",
        signal_kind="power",
        family_signal_kind="boxcox",
        family_curvature=True,
        global_group_penalty=True,
        family_group_penalty=True,
        family_total_penalty=True,
        pair_aggregator="ces",
    ),
}
PENALTY_CALIBRATION_VARIANT_NAMES = tuple(VARIANT_SPECS)


def variant_spec(variant_name: str) -> PenaltyCalibrationVariantSpec:
    """Return the structural spec for a calibration variant."""
    try:
        return VARIANT_SPECS[variant_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported penalty calibration variant {variant_name!r}") from exc


def _sigmoid_scalar_clipped(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def _family_tau(params: dict[str, float], family_name: str) -> float:
    return float(params.get(f"tau_{family_name}", params.get("tau", 3.0)))


class GenericFamilyPenaltyCalibrationSurrogate:
    """Flexible GRP surrogate with optional family curvature, richer penalties, and pair CES."""

    def __init__(
        self,
        packet: GenericFamilyPacket,
        *,
        params: dict[str, float],
        spec: PenaltyCalibrationVariantSpec,
        family_totals: tuple[str, ...] = GENERIC_FAMILY_NAMES,
        quality_discount: bool = True,
        pair_cc_domains: bool = True,
        include_singletons: bool = True,
        include_pairs: bool = True,
        include_family_totals: bool = True,
        include_global_group_penalty: bool = True,
        include_family_group_penalty: bool = True,
        include_family_total_penalty: bool = True,
    ):
        self.packet = packet
        self.params = dict(params)
        self.spec = spec
        self.family_totals = tuple(family_totals)
        self.quality_discount = bool(quality_discount)
        self.pair_cc_domains = bool(pair_cc_domains)
        self.include_singletons = bool(include_singletons)
        self.include_pairs = bool(include_pairs)
        self.include_family_totals = bool(include_family_totals)
        self.include_global_group_penalty = bool(self.spec.global_group_penalty and include_global_group_penalty)
        self.include_family_group_penalty = bool(self.spec.family_group_penalty and include_family_group_penalty)
        self.include_family_total_penalty = bool(self.spec.family_total_penalty and include_family_total_penalty)
        if not (self.include_singletons or self.include_pairs or self.include_family_totals):
            raise ValueError("At least one signal block must be enabled")
        domain_to_family: list[str] = []
        for domain_idx in range(packet.base.m):
            assigned = [family_name for family_name, members in packet.family_map.items() if domain_idx in members]
            if len(assigned) != 1:
                raise ValueError(f"Expected exactly one family for domain index {domain_idx}, got {assigned}")
            domain_to_family.append(assigned[0])
        self.domain_to_family = tuple(domain_to_family)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _retained_x(self, weights: np.ndarray) -> np.ndarray:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.packet.base.c0[None, :]
        e1 = p1 * self.packet.base.c1[None, :]
        lam = float(self.params["lam"])
        eta = float(self.params["eta"])
        return np.exp(-lam * (1.0 - p1)) * e0 + eta * e1

    def _feature_transform(
        self,
        values: np.ndarray,
        *,
        signal_kind: str,
        family_name: str | None = None,
        other_family_name: str | None = None,
        domain_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        resolved_family_name = family_name if self.spec.family_curvature else None
        resolved_other_family_name = other_family_name if self.spec.family_curvature else None
        resolved_domain_indices = domain_indices if self.spec.domain_curvature else None
        return signal_transform(
            values,
            self.params,
            signal_kind,
            resolved_family_name,
            resolved_other_family_name,
            resolved_domain_indices,
        )

    def _feature_derivative(
        self,
        values: np.ndarray,
        *,
        signal_kind: str,
        family_name: str | None = None,
        other_family_name: str | None = None,
        domain_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        resolved_family_name = family_name if self.spec.family_curvature else None
        resolved_other_family_name = other_family_name if self.spec.family_curvature else None
        resolved_domain_indices = domain_indices if self.spec.domain_curvature else None
        return signal_derivative(
            values,
            self.params,
            signal_kind,
            resolved_family_name,
            resolved_other_family_name,
            resolved_domain_indices,
        )

    def _pair_signal_total(self, x_hi: np.ndarray, x_lo: np.ndarray) -> np.ndarray:
        hi = np.asarray(x_hi, dtype=float)
        lo = np.asarray(x_lo, dtype=float)
        lo_scale = float(self.params["beta"]) if self.quality_discount else 1.0
        if self.spec.pair_aggregator == "linear":
            return hi + lo_scale * lo
        if self.spec.pair_aggregator == "ces":
            rho = float(self.params["pair_rho"])
            safe_hi = np.maximum(hi, 1e-12)
            safe_lo = np.maximum(lo_scale * lo, 1e-12)
            inner = np.power(safe_hi, rho) + np.power(safe_lo, rho)
            return np.power(np.maximum(inner, 1e-12), 1.0 / rho)
        raise ValueError(f"Unsupported pair aggregator {self.spec.pair_aggregator!r}")

    def _pair_signal_partials(self, x_hi: float, x_lo: float) -> tuple[float, float]:
        lo_scale = float(self.params["beta"]) if self.quality_discount else 1.0
        if self.spec.pair_aggregator == "linear":
            return 1.0, lo_scale
        if self.spec.pair_aggregator == "ces":
            rho = float(self.params["pair_rho"])
            safe_hi = max(float(x_hi), 1e-12)
            safe_lo = max(lo_scale * float(x_lo), 1e-12)
            inner = max(safe_hi**rho + safe_lo**rho, 1e-12)
            common = inner ** (1.0 / rho - 1.0)
            d_hi = common * safe_hi ** (rho - 1.0)
            d_lo = common * safe_lo ** (rho - 1.0) * lo_scale
            return float(d_hi), float(d_lo)
        raise ValueError(f"Unsupported pair aggregator {self.spec.pair_aggregator!r}")

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        x = self._retained_x(weights)
        features: list[np.ndarray] = []
        group_totals: list[np.ndarray] = []
        family_group_totals: dict[str, list[np.ndarray]] = {family_name: [] for family_name in self.family_totals}
        family_totals: dict[str, np.ndarray] = {}

        singleton_indices = (
            (self.packet.singletons if self.pair_cc_domains else list(range(self.packet.base.m)))
            if self.include_singletons
            else []
        )
        pair_map = self.packet.pairs if self.pair_cc_domains and self.include_pairs else []

        for idx in singleton_indices:
            family_name = self.domain_to_family[idx]
            features.append(
                self._feature_transform(
                    x[:, idx],
                    signal_kind=self.spec.signal_kind,
                    family_name=family_name,
                    domain_indices=(idx,),
                )[:, None]
            )
            group_totals.append(x[:, idx])
            family_group_totals[family_name].append(x[:, idx])

        for hi, lo in pair_map:
            hi_family = self.domain_to_family[hi]
            lo_family = self.domain_to_family[lo]
            signal_total = self._pair_signal_total(x[:, hi], x[:, lo])
            features.append(
                self._feature_transform(
                    signal_total,
                    signal_kind=self.spec.signal_kind,
                    family_name=hi_family,
                    other_family_name=lo_family,
                    domain_indices=(hi, lo),
                )[:, None]
            )
            total = x[:, hi] + x[:, lo]
            group_totals.append(total)
            family_group_totals[hi_family].append(total)

        for family_name in self.family_totals:
            members = self.packet.family_map[family_name]
            family_total = np.sum(x[:, members], axis=1)
            family_totals[family_name] = family_total
            if self.include_family_totals:
                features.append(
                    self._feature_transform(
                        family_total,
                        signal_kind=self.spec.family_signal_kind,
                        family_name=family_name,
                        domain_indices=tuple(int(member) for member in members),
                    )[:, None]
                )

        penalties: list[np.ndarray] = []
        if self.include_global_group_penalty:
            tau = float(self.params["tau"])
            penalty_inputs = np.stack(group_totals, axis=1)
            penalties.append(np.sum(softplus(np.log1p(penalty_inputs) - tau) ** 2, axis=1, keepdims=True))
        if self.include_family_group_penalty:
            for family_name in self.family_totals:
                if not family_group_totals[family_name]:
                    continue
                tau_f = _family_tau(self.params, family_name)
                penalty_inputs = np.stack(family_group_totals[family_name], axis=1)
                penalties.append(np.sum(softplus(np.log1p(penalty_inputs) - tau_f) ** 2, axis=1, keepdims=True))
        if self.include_family_total_penalty:
            for family_name in self.family_totals:
                tau_f = _family_tau(self.params, family_name)
                penalties.append(softplus(np.log1p(family_totals[family_name]) - tau_f)[:, None] ** 2)

        design = np.hstack(features + penalties)
        design[:, : len(features)] *= -1.0
        return design

    def fit(
        self,
        weights: np.ndarray,
        targets: np.ndarray,
    ) -> GenericFamilyPenaltyCalibrationSurrogate:
        design = self.build_design(weights)
        design_mean = design.mean(axis=0, keepdims=True)
        target_mean = float(targets.mean())
        centered_design = design - design_mean
        centered_targets = targets - target_mean
        reg = float(self.params["reg"])
        if reg > 0.0:
            centered_design = np.vstack([centered_design, np.sqrt(reg) * np.eye(centered_design.shape[1])])
            centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
        coef, _ = nnls(centered_design, centered_targets)
        self.coef_ = coef
        self.intercept_ = float(target_mean - (design_mean @ coef).item())
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        return np.asarray(self.intercept_ + self.build_design(weights) @ self.coef_, dtype=float)

    def components(self) -> dict[str, Any]:
        if self.coef_ is None:
            raise RuntimeError("Model must be fit")
        n_singletons = (
            (len(self.packet.singletons) if self.pair_cc_domains else self.packet.base.m)
            if self.include_singletons
            else 0
        )
        n_pairs = (len(self.packet.pairs) if self.pair_cc_domains else 0) if self.include_pairs else 0
        n_families = len(self.family_totals) if self.include_family_totals else 0
        offset = 0
        singleton_coef = np.asarray(self.coef_[offset : offset + n_singletons], dtype=float)
        offset += n_singletons
        pair_coef = np.asarray(self.coef_[offset : offset + n_pairs], dtype=float)
        offset += n_pairs
        family_coef = {
            family_name: float(coef)
            for family_name, coef in zip(self.family_totals, self.coef_[offset : offset + n_families], strict=True)
        }
        offset += n_families

        global_penalty_coef = 0.0
        if self.include_global_group_penalty:
            global_penalty_coef = float(self.coef_[offset])
            offset += 1

        family_group_penalty_coef = {family_name: 0.0 for family_name in self.family_totals}
        if self.include_family_group_penalty:
            family_group_penalty_coef = {
                family_name: float(coef)
                for family_name, coef in zip(self.family_totals, self.coef_[offset : offset + n_families], strict=True)
            }
            offset += n_families

        family_total_penalty_coef = {family_name: 0.0 for family_name in self.family_totals}
        if self.include_family_total_penalty:
            family_total_penalty_coef = {
                family_name: float(coef)
                for family_name, coef in zip(self.family_totals, self.coef_[offset : offset + n_families], strict=True)
            }
        return {
            "singleton_coef": singleton_coef,
            "pair_coef": pair_coef,
            "family_coef": family_coef,
            "global_penalty_coef": global_penalty_coef,
            "family_group_penalty_coef": family_group_penalty_coef,
            "family_total_penalty_coef": family_total_penalty_coef,
        }


def build_penalty_calibration_surrogate(
    packet: GenericFamilyPacket,
    *,
    params: dict[str, float],
    variant_name: str,
    family_totals: tuple[str, ...] = GENERIC_FAMILY_NAMES,
    quality_discount: bool = True,
    pair_cc_domains: bool = True,
    include_singletons: bool = True,
    include_pairs: bool = True,
    include_family_totals: bool = True,
    include_global_group_penalty: bool = True,
    include_family_group_penalty: bool = True,
    include_family_total_penalty: bool = True,
) -> GenericFamilyPenaltyCalibrationSurrogate:
    """Build a calibration surrogate from a named variant."""
    return GenericFamilyPenaltyCalibrationSurrogate(
        packet,
        params=params,
        spec=variant_spec(variant_name),
        family_totals=family_totals,
        quality_discount=quality_discount,
        pair_cc_domains=pair_cc_domains,
        include_singletons=include_singletons,
        include_pairs=include_pairs,
        include_family_totals=include_family_totals,
        include_global_group_penalty=include_global_group_penalty,
        include_family_group_penalty=include_family_group_penalty,
        include_family_total_penalty=include_family_total_penalty,
    )


def penalty_calibration_param_keys(variant_name: str) -> tuple[str, ...]:
    """Return nonlinear parameter keys for a calibration variant."""
    spec = variant_spec(variant_name)
    keys = ["eta", "lam", "reg", "beta"]
    if "boxcox" in {spec.signal_kind, spec.family_signal_kind}:
        keys.append("alpha")
    if spec.domain_curvature:
        keys.extend(domain_exponent_key(domain_idx) for domain_idx in range(39))
    elif spec.family_curvature:
        keys.extend(f"a_{family_name}" for family_name in GENERIC_FAMILY_NAMES)
    else:
        keys.append("a")
    if spec.global_group_penalty:
        keys.append("tau")
    if spec.family_group_penalty or spec.family_total_penalty:
        keys.extend(f"tau_{family_name}" for family_name in GENERIC_FAMILY_NAMES)
    if spec.pair_aggregator == "ces":
        keys.append("pair_rho")
    return tuple(keys)


def penalty_calibration_params_from_metrics(
    metrics: dict[str, float | bool],
    variant_name: str,
) -> dict[str, float]:
    """Extract nonlinear parameters from a benchmark metrics row."""
    return {key: float(metrics[key]) for key in penalty_calibration_param_keys(variant_name)}


def pack_penalty_calibration_params(params: dict[str, float], variant_name: str) -> np.ndarray:
    """Pack nonlinear parameters into unconstrained optimizer coordinates."""
    spec = variant_spec(variant_name)
    beta = float(np.clip(params["beta"], 1e-8, 1.0 - 1.0e-8))
    z = [
        np.log(float(params["eta"])),
        np.log(float(params["lam"])),
        np.log(float(params["reg"])),
        np.log(beta / (1.0 - beta)),
    ]
    if "boxcox" in {spec.signal_kind, spec.family_signal_kind}:
        z.append(np.log(float(params["alpha"])))
    if spec.domain_curvature:
        for domain_idx in range(39):
            z.append(np.log(float(params[domain_exponent_key(domain_idx)])))
    elif spec.family_curvature:
        for family_name in GENERIC_FAMILY_NAMES:
            z.append(np.log(float(params[f"a_{family_name}"])))
    else:
        z.append(np.log(float(params["a"])))
    if spec.global_group_penalty:
        z.append(float(params["tau"]))
    if spec.family_group_penalty or spec.family_total_penalty:
        for family_name in GENERIC_FAMILY_NAMES:
            z.append(float(params[f"tau_{family_name}"]))
    if spec.pair_aggregator == "ces":
        z.append(np.log(float(params["pair_rho"])))
    return np.asarray(z, dtype=float)


def unpack_penalty_calibration_params(z: np.ndarray, variant_name: str) -> dict[str, float]:
    """Decode unconstrained optimizer coordinates into nonlinear parameters."""
    spec = variant_spec(variant_name)
    params = {
        "eta": float(np.exp(np.clip(z[0], -8.0, 8.0))),
        "lam": float(np.exp(np.clip(z[1], -12.0, 4.0))),
        "reg": float(np.exp(np.clip(z[2], -18.0, 0.0))),
        "beta": float(np.clip(_sigmoid_scalar_clipped(float(z[3])), 1e-6, 1.0 - 1e-6)),
    }
    idx = 4
    if "boxcox" in {spec.signal_kind, spec.family_signal_kind}:
        params["alpha"] = float(np.exp(np.clip(z[idx], -8.0, 8.0)))
        idx += 1
    if spec.domain_curvature:
        for domain_idx in range(39):
            params[domain_exponent_key(domain_idx)] = float(np.exp(np.clip(z[idx], np.log(0.02), np.log(2.0))))
            idx += 1
    elif spec.family_curvature:
        for family_name in GENERIC_FAMILY_NAMES:
            params[f"a_{family_name}"] = float(np.exp(np.clip(z[idx], np.log(0.02), np.log(2.0))))
            idx += 1
    else:
        params["a"] = float(np.exp(np.clip(z[idx], np.log(0.02), np.log(2.0))))
        idx += 1
    if spec.global_group_penalty:
        params["tau"] = float(np.clip(z[idx], -2.0, 8.0))
        idx += 1
    if spec.family_group_penalty or spec.family_total_penalty:
        for family_name in GENERIC_FAMILY_NAMES:
            params[f"tau_{family_name}"] = float(np.clip(z[idx], -2.0, 8.0))
            idx += 1
    if spec.pair_aggregator == "ces":
        params["pair_rho"] = float(np.exp(np.clip(z[idx], np.log(0.2), np.log(2.0))))
    return params


def penalty_calibration_variant_parameter_counts(
    packet: GenericFamilyPacket,
    variant_name: str,
    *,
    include_singletons: bool = True,
    include_pairs: bool = True,
    include_family_totals: bool = True,
    include_global_group_penalty: bool = True,
    include_family_group_penalty: bool = True,
    include_family_total_penalty: bool = True,
) -> dict[str, int]:
    """Return nonlinear and linear parameter counts for a variant on this packet."""
    spec = variant_spec(variant_name)
    signal_feature_count = (
        int(include_singletons) * len(packet.singletons)
        + int(include_pairs) * len(packet.pairs)
        + int(include_family_totals) * len(GENERIC_FAMILY_NAMES)
    )
    penalty_feature_count = (
        int(spec.global_group_penalty and include_global_group_penalty)
        + len(GENERIC_FAMILY_NAMES) * int(spec.family_group_penalty and include_family_group_penalty)
        + len(GENERIC_FAMILY_NAMES) * int(spec.family_total_penalty and include_family_total_penalty)
    )
    linear_coefficient_count = signal_feature_count + penalty_feature_count
    intercept_count = 1
    nonlinear_param_count = len(penalty_calibration_param_keys(variant_name))
    nonlinear_param_count -= int(spec.global_group_penalty and not include_global_group_penalty)
    nonlinear_param_count -= len(GENERIC_FAMILY_NAMES) * int(
        spec.family_group_penalty and not include_family_group_penalty
    )
    nonlinear_param_count -= len(GENERIC_FAMILY_NAMES) * int(
        spec.family_total_penalty and not include_family_total_penalty
    )
    return {
        "signal_feature_count": signal_feature_count,
        "penalty_feature_count": penalty_feature_count,
        "linear_coefficient_count": linear_coefficient_count,
        "intercept_count": intercept_count,
        "linear_head_param_count": linear_coefficient_count + intercept_count,
        "nonlinear_param_count": nonlinear_param_count,
        "total_param_count": nonlinear_param_count + linear_coefficient_count + intercept_count,
    }


def optimize_penalty_calibration_model(
    packet: GenericFamilyPacket,
    model: GenericFamilyPenaltyCalibrationSurrogate,
    *,
    n_random: int = 1,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Optimize the continuous mixture implied by a fitted surrogate."""
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
    lam = float(model.params["lam"])
    eta = float(model.params["eta"])
    pair_map = packet.pairs if model.pair_cc_domains else []
    if not model.include_pairs:
        pair_map = []
    singleton_indices = (
        (packet.singletons if model.pair_cc_domains else list(range(packet.base.m))) if model.include_singletons else []
    )
    family_indices = {
        family_name: np.asarray(packet.family_map[family_name], dtype=int) for family_name in model.family_totals
    }
    rng = np.random.default_rng(seed)

    def value_grad_logits(z: np.ndarray) -> tuple[float, np.ndarray]:
        logits0 = z[:n_domains]
        logits1 = z[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 /= np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 /= np.sum(p1)

        e0 = c0 * p0
        retained = np.exp(-lam * (1.0 - p1))
        x = retained * e0 + eta * c1 * p1
        dx_dp0 = retained * c0
        dx_dp1 = lam * retained * e0 + eta * c1

        value = float(model.intercept_)
        grad_x = np.zeros(n_domains, dtype=float)
        group_info: list[tuple[tuple[int, ...], float, str]] = []
        family_total_values: dict[str, float] = {}

        for local_idx, domain_idx in enumerate(singleton_indices):
            family_name = model.domain_to_family[domain_idx]
            x_value = float(x[domain_idx])
            coef = float(singleton_coef[local_idx])
            value -= (
                coef
                * model._feature_transform(
                    np.asarray([x_value]),
                    signal_kind=model.spec.signal_kind,
                    family_name=family_name,
                    domain_indices=(domain_idx,),
                )[0]
            )
            grad_x[domain_idx] -= (
                coef
                * model._feature_derivative(
                    np.asarray([x_value]),
                    signal_kind=model.spec.signal_kind,
                    family_name=family_name,
                    domain_indices=(domain_idx,),
                )[0]
            )
            group_info.append(((domain_idx,), x_value, family_name))

        for local_idx, (hi, lo) in enumerate(pair_map):
            hi_family = model.domain_to_family[hi]
            lo_family = model.domain_to_family[lo]
            coef = float(pair_coef[local_idx])
            signal_total = float(model._pair_signal_total(np.asarray([x[hi]]), np.asarray([x[lo]]))[0])
            value -= (
                coef
                * model._feature_transform(
                    np.asarray([signal_total]),
                    signal_kind=model.spec.signal_kind,
                    family_name=hi_family,
                    other_family_name=lo_family,
                    domain_indices=(hi, lo),
                )[0]
            )
            d_hi, d_lo = model._pair_signal_partials(float(x[hi]), float(x[lo]))
            chain = (
                coef
                * model._feature_derivative(
                    np.asarray([signal_total]),
                    signal_kind=model.spec.signal_kind,
                    family_name=hi_family,
                    other_family_name=lo_family,
                    domain_indices=(hi, lo),
                )[0]
            )
            grad_x[hi] -= chain * d_hi
            grad_x[lo] -= chain * d_lo
            group_info.append(((hi, lo), float(x[hi] + x[lo]), hi_family))

        for family_name in model.family_totals:
            members = family_indices[family_name]
            family_total = float(np.sum(x[members]))
            family_total_values[family_name] = family_total
            if model.include_family_totals:
                coef = float(family_coef[family_name])
                value -= (
                    coef
                    * model._feature_transform(
                        np.asarray([family_total]),
                        signal_kind=model.spec.family_signal_kind,
                        family_name=family_name,
                        domain_indices=tuple(int(member) for member in members),
                    )[0]
                )
                grad_x[members] -= (
                    coef
                    * model._feature_derivative(
                        np.asarray([family_total]),
                        signal_kind=model.spec.family_signal_kind,
                        family_name=family_name,
                        domain_indices=tuple(int(member) for member in members),
                    )[0]
                )

        if model.include_global_group_penalty and global_penalty_coef != 0.0:
            tau = float(model.params["tau"])
            for members, total, _family_name in group_info:
                inside = np.log1p(total) - tau
                sp = float(softplus(inside))
                value += global_penalty_coef * sp * sp
                if sp == 0.0:
                    continue
                common = global_penalty_coef * 2.0 * sp * float(sigmoid(inside)) / (1.0 + total)
                for idx in members:
                    grad_x[idx] += common

        if model.include_family_group_penalty:
            for members, total, family_name in group_info:
                coef = float(family_group_penalty_coef[family_name])
                if coef == 0.0:
                    continue
                tau_f = _family_tau(model.params, family_name)
                inside = np.log1p(total) - tau_f
                sp = float(softplus(inside))
                value += coef * sp * sp
                if sp == 0.0:
                    continue
                common = coef * 2.0 * sp * float(sigmoid(inside)) / (1.0 + total)
                for idx in members:
                    grad_x[idx] += common

        if model.include_family_total_penalty:
            for family_name in model.family_totals:
                coef = float(family_total_penalty_coef[family_name])
                if coef == 0.0:
                    continue
                total = family_total_values[family_name]
                tau_f = _family_tau(model.params, family_name)
                inside = np.log1p(total) - tau_f
                sp = float(softplus(inside))
                value += coef * sp * sp
                if sp == 0.0:
                    continue
                common = coef * 2.0 * sp * float(sigmoid(inside)) / (1.0 + total)
                grad_x[family_indices[family_name]] += common

        grad_p0 = grad_x * dx_dp0
        grad_p1 = grad_x * dx_dp1
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
        raise RuntimeError("Penalty calibration optimization failed")

    z = np.asarray(best.x, dtype=float)
    logits0 = z[:n_domains]
    logits1 = z[n_domains:]
    phase0 = np.exp(logits0 - np.max(logits0))
    phase0 /= np.sum(phase0)
    phase1 = np.exp(logits1 - np.max(logits1))
    phase1 /= np.sum(phase1)
    return best, phase0, phase1


def penalty_calibration_oof_metrics(
    packet: GenericFamilyPacket,
    params: dict[str, float],
    *,
    variant_name: str,
    seed: int = 0,
    lower_tail_frac: float = LOWER_TAIL_FRAC,
    support_top_k: int = TRUSTBLEND_TOPK_ACTUAL,
) -> dict[str, float]:
    """Compute out-of-fold fit and calibration metrics for one parameter setting."""
    y = packet.base.y
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    depopt_scores: list[float] = []
    rawopt_support_scores: list[float] = []

    for tr, te in kf.split(packet.base.w):
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name=variant_name).fit(
            packet.base.w[tr],
            y[tr],
        )
        pred = model.predict(packet.base.w[te])
        oof[te] = pred
        fold_regrets.append(float(y[te][int(np.argmin(pred))] - np.min(y[te])))

        raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, n_random=1, seed=seed)
        raw_weights = np.stack([phase0, phase1], axis=0)
        distances = average_phase_tv_distance(packet.base.w[te], raw_weights[None, :, :])
        nearest_count = min(int(support_top_k), len(te))
        nearest_idx = np.argsort(distances)[:nearest_count]
        depopt_scores.append(max(float(np.min(y[te][nearest_idx])) - float(raw_result.fun), 0.0))
        rawopt_support_scores.append(float(distances[nearest_idx[0]]))

    residuals = oof - y
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    tail_count = max(5, int(np.ceil(float(lower_tail_frac) * float(len(y)))))
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
        "objective": objective,
    }


def compute_penalty_calibration_metrics(
    packet: GenericFamilyPacket,
    model: GenericFamilyPenaltyCalibrationSurrogate,
    *,
    seed: int = 0,
    valid_weights: np.ndarray | None = None,
    valid_y: np.ndarray | None = None,
) -> dict[str, float | bool]:
    """Compute training, CV, and optimum-calibration metrics for a fitted variant."""
    if model.coef_ is None or model.intercept_ is None:
        model = model.fit(packet.base.w, packet.base.y)

    metrics = penalty_calibration_oof_metrics(packet, model.params, variant_name=model.spec.name, seed=seed)
    train_pred = model.predict(packet.base.w)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=seed)
    raw_weights = np.stack([phase0, phase1], axis=0)
    raw_distances = average_phase_tv_distance(packet.base.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))

    out: dict[str, float | bool] = {
        **metrics,
        "train_rmse": float(np.sqrt(np.mean((train_pred - packet.base.y) ** 2))),
        "raw_predicted_optimum_value": float(raw_result.fun),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "raw_nearest_observed_value": float(packet.base.y[nearest_idx]),
        "raw_phase0_lt_1e4": int(np.sum(phase0 < 1e-4)),
        "raw_phase1_lt_1e4": int(np.sum(phase1 < 1e-4)),
    }
    if valid_weights is not None and valid_y is not None:
        anchor_pred = model.predict(valid_weights)
        out.update(
            {
                "anchor_mae": float(np.mean(np.abs(anchor_pred - valid_y))),
                "anchor_rmse": float(np.sqrt(np.mean((anchor_pred - valid_y) ** 2))),
                "anchor_rank_correct": bool(int(np.argmin(anchor_pred)) == int(np.argmin(valid_y))),
                "pred_validated_global": float(anchor_pred[0]),
                "pred_validated_pair": float(anchor_pred[1]),
            }
        )
    return out


def tune_penalty_calibration_params(
    packet: GenericFamilyPacket,
    *,
    variant_name: str,
    start_bank: tuple[dict[str, float], ...],
    method: str = "coarse",
    coarse_top_k: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | bool], Any]:
    """Tune nonlinear parameters for a calibration variant."""
    coarse_rows: list[dict[str, float | bool]] = []
    for start_id, params in enumerate(start_bank):
        coarse_rows.append(
            {
                "variant": variant_name,
                "stage": "coarse",
                "start_id": int(start_id),
                **params,
                **penalty_calibration_oof_metrics(packet, params, variant_name=variant_name, seed=seed),
            }
        )
    coarse_frame = pd.DataFrame.from_records(coarse_rows).sort_values(
        ["objective", "cv_rmse", "cv_depopt_best8"],
        ascending=[True, True, True],
    )
    if method == "coarse":
        best_metrics = coarse_frame.iloc[0].to_dict()
        best_metrics["success"] = True
        best_metrics["message"] = "coarse_only"
        return coarse_frame, pd.DataFrame(), best_metrics, None
    chosen_ids = coarse_frame["start_id"].head(int(coarse_top_k)).tolist()

    best_metrics: dict[str, float | bool] | None = None
    best_result: Any | None = None
    best_objective = float("inf")
    refine_rows: list[dict[str, float | bool]] = []

    for chosen_rank, start_id in enumerate(chosen_ids):
        start = pack_penalty_calibration_params(start_bank[start_id], variant_name)
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, _cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in _cache:
                metrics = penalty_calibration_oof_metrics(
                    packet,
                    unpack_penalty_calibration_params(z, variant_name),
                    variant_name=variant_name,
                    seed=seed,
                )
                _cache[key] = float(metrics["objective"])
            return _cache[key]

        options = {
            "L-BFGS-B": {"maxiter": 80, "ftol": 1e-6},
            "Nelder-Mead": {"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6},
            "Powell": {"maxiter": 30, "xtol": 1e-4, "ftol": 1e-6},
        }.get(method, {"maxiter": 120})
        result = minimize(objective, start, method=method, options=options)
        metrics = {
            **unpack_penalty_calibration_params(np.asarray(result.x, dtype=float), variant_name),
            **penalty_calibration_oof_metrics(
                packet,
                unpack_penalty_calibration_params(np.asarray(result.x, dtype=float), variant_name),
                variant_name=variant_name,
                seed=seed,
            ),
        }
        row = {
            "variant": variant_name,
            "stage": "refine",
            "chosen_rank": int(chosen_rank),
            "start_id": int(start_id),
            "success": bool(result.success),
            "message": str(result.message),
            **metrics,
        }
        refine_rows.append(row)
        if float(row["objective"]) < best_objective:
            best_objective = float(row["objective"])
            best_metrics = row
            best_result = result

    if best_metrics is None or best_result is None:
        raise RuntimeError(f"Penalty calibration tuning failed for variant {variant_name!r}")
    return coarse_frame, pd.DataFrame.from_records(refine_rows), best_metrics, best_result


def deploy_penalty_calibration_gaincapped_topkactual(
    packet: GenericFamilyPacket,
    model: GenericFamilyPenaltyCalibrationSurrogate,
    tuning_metrics: dict[str, float | bool],
    *,
    top_k: int = TRUSTBLEND_TOPK_ACTUAL,
    line_grid: int = TRUSTBLEND_LINE_GRID,
) -> dict[str, Any]:
    """Deploy a variant by trust-blending raw optimum with the observed top-k hull."""
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=0)
    raw_weights = np.stack([phase0, phase1], axis=0)
    top_indices = np.argsort(packet.base.y)[: min(int(top_k), len(packet.base.y))]
    hull_anchor_weights = packet.base.w[top_indices]
    hull_predicted_value, hull_coeffs, hull_weights = optimize_generic_family_convex_hull(
        model,
        hull_anchor_weights,
        start_indices=np.arange(min(len(top_indices), 8), dtype=int),
    )

    gain_budget = float(tuning_metrics["cv_rmse"]) + float(tuning_metrics["cv_foldmean_regret_at_1"])
    raw_predicted_value = float(raw_result.fun)
    target_gain = min(float(hull_predicted_value) - raw_predicted_value, gain_budget)
    best: tuple[tuple[int, float, float], float, float, np.ndarray, float] | None = None
    for delta in np.linspace(0.0, 1.0, int(line_grid)):
        weights = (1.0 - delta) * hull_weights + delta * raw_weights
        predicted_value = float(model.predict(weights[None, :, :])[0])
        realized_gain = float(hull_predicted_value) - predicted_value
        feasible = realized_gain <= gain_budget + 1e-12
        key = (0 if feasible else 1, predicted_value, abs(realized_gain - target_gain))
        if best is None or key < best[0]:
            best = (key, float(delta), predicted_value, weights, realized_gain)

    if best is None:
        raise RuntimeError("Penalty calibration deployment selection failed")

    _key, delta, predicted_value, weights, realized_gain = best
    return {
        "predicted_optimum_value": predicted_value,
        "weights": weights,
        "delta": delta,
        "realized_gain": realized_gain,
        "gain_budget": gain_budget,
        "raw_predicted_optimum_value": raw_predicted_value,
        "hull_predicted_optimum_value": float(hull_predicted_value),
        "hull_top_indices": top_indices.tolist(),
        "hull_top_run_names": [str(packet.base.frame.iloc[idx][packet.base.name_col]) for idx in top_indices.tolist()],
        "hull_coefficients": np.asarray(hull_coeffs, dtype=float),
        "optimizer_success": bool(raw_result.success),
        "optimizer_message": str(raw_result.message),
    }
