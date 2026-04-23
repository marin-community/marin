# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Flexible-signal GRP variants for many-domain follow-up experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    TUNED_GENERIC_FAMILY_PARAMS,
    GenericFamilyPacket,
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    sigmoid,
    softplus,
)

FLEXIBLE_SIGNAL_KINDS = ("log", "power", "boxcox")
FLEXIBLE_VARIANT_NAMES = ("log", "power", "boxcox", "power_family", "boxcox_family", "power_boxcox_family")
OBSERVED_ONLY_START_PARAM_BANK: tuple[dict[str, float], ...] = (
    {"alpha": 8.0, "eta": 8.0, "lam": 0.05, "tau": 3.0, "reg": 1e-3, "beta": 0.70},
    {"alpha": 16.0, "eta": 8.0, "lam": 0.20, "tau": 2.7, "reg": 3e-4, "beta": 0.90},
    {"alpha": 4.0, "eta": 16.0, "lam": 0.02, "tau": 3.5, "reg": 3e-3, "beta": 0.50},
)
OBSERVED_ONLY_CV_WEIGHT = 1.0
OBSERVED_ONLY_FOLDMEAN_WEIGHT = 0.05
OBSERVED_ONLY_TAIL_WEIGHT = 0.5
OBSERVED_ONLY_LOWER_TAIL_FRAC = 0.15
TRUSTBLEND_TOPK_ACTUAL = 8
TRUSTBLEND_LINE_GRID = 81
DOMAIN_EXPONENT_PREFIX = "a_domain_"


def domain_exponent_key(domain_idx: int) -> str:
    """Return the parameter key for a domain-specific exponent."""
    return f"{DOMAIN_EXPONENT_PREFIX}{domain_idx:02d}"


def _resolve_exponent_value(params: dict[str, float], key: str) -> float:
    if key in params:
        return float(params[key])
    if "a" in params:
        return float(params["a"])
    raise KeyError(f"Missing curvature parameter {key!r}")


def _resolve_curvature(
    params: dict[str, float],
    signal_kind: str,
    family_name: str | None = None,
    other_family_name: str | None = None,
    domain_indices: tuple[int, ...] | None = None,
) -> float:
    if signal_kind not in {"power", "boxcox"}:
        raise ValueError(f"Family curvature is only defined for power/boxcox, got {signal_kind!r}")
    if domain_indices:
        values = [_resolve_exponent_value(params, domain_exponent_key(domain_idx)) for domain_idx in domain_indices]
        return float(np.mean(np.asarray(values, dtype=float)))
    if family_name is None:
        return float(params["a"])
    key = f"a_{family_name}"
    first = _resolve_exponent_value(params, key)
    if other_family_name is None:
        return first
    other_key = f"a_{other_family_name}"
    second = _resolve_exponent_value(params, other_key)
    return 0.5 * (first + second)


def signal_transform(
    values: np.ndarray,
    params: dict[str, float],
    signal_kind: str,
    family_name: str | None = None,
    other_family_name: str | None = None,
    domain_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if signal_kind == "log":
        alpha = float(params["alpha"])
        return np.log1p(alpha * values)
    if signal_kind == "power":
        a = _resolve_curvature(
            params,
            signal_kind,
            family_name,
            other_family_name,
            domain_indices,
        )
        return np.power(np.maximum(values, 1e-12), a)
    if signal_kind == "boxcox":
        alpha = float(params["alpha"])
        a = _resolve_curvature(
            params,
            signal_kind,
            family_name,
            other_family_name,
            domain_indices,
        )
        u = 1.0 + alpha * values
        if abs(a) < 1e-8:
            return np.log(u)
        return (np.power(u, a) - 1.0) / a
    raise ValueError(f"Unsupported signal kind: {signal_kind}")


def signal_derivative(
    values: np.ndarray,
    params: dict[str, float],
    signal_kind: str,
    family_name: str | None = None,
    other_family_name: str | None = None,
    domain_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if signal_kind == "log":
        alpha = float(params["alpha"])
        return alpha / (1.0 + alpha * values)
    if signal_kind == "power":
        a = _resolve_curvature(
            params,
            signal_kind,
            family_name,
            other_family_name,
            domain_indices,
        )
        safe = np.maximum(values, 1e-12)
        return a * np.power(safe, a - 1.0)
    if signal_kind == "boxcox":
        alpha = float(params["alpha"])
        a = _resolve_curvature(
            params,
            signal_kind,
            family_name,
            other_family_name,
            domain_indices,
        )
        u = 1.0 + alpha * values
        if abs(a) < 1e-8:
            return alpha / u
        return alpha * np.power(u, a - 1.0)
    raise ValueError(f"Unsupported signal kind: {signal_kind}")


class GenericFamilyFlexibleSignalSurrogate:
    """GRP surrogate with pluggable monotone signal laws."""

    def __init__(
        self,
        packet: GenericFamilyPacket,
        *,
        params: dict[str, float],
        signal_kind: str,
        family_signal_kind: str | None = None,
        family_totals: tuple[str, ...] = GENERIC_FAMILY_NAMES,
        quality_discount: bool = True,
        pair_cc_domains: bool = True,
        family_curvature: bool = False,
    ):
        if signal_kind not in FLEXIBLE_SIGNAL_KINDS:
            raise ValueError(f"Unsupported signal kind: {signal_kind}")
        if family_signal_kind is not None and family_signal_kind not in FLEXIBLE_SIGNAL_KINDS:
            raise ValueError(f"Unsupported family signal kind: {family_signal_kind}")
        self.packet = packet
        self.params = dict(params)
        self.signal_kind = str(signal_kind)
        self.family_signal_kind = str(family_signal_kind or signal_kind)
        self.family_totals = tuple(family_totals)
        self.quality_discount = bool(quality_discount)
        self.pair_cc_domains = bool(pair_cc_domains)
        self.family_curvature = bool(family_curvature)
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
        signal_kind: str | None = None,
        family_name: str | None = None,
        other_family_name: str | None = None,
    ) -> np.ndarray:
        return signal_transform(
            values,
            self.params,
            self.signal_kind if signal_kind is None else signal_kind,
            family_name if self.family_curvature else None,
            other_family_name if self.family_curvature else None,
        )

    def _feature_derivative(
        self,
        values: np.ndarray,
        *,
        signal_kind: str | None = None,
        family_name: str | None = None,
        other_family_name: str | None = None,
    ) -> np.ndarray:
        return signal_derivative(
            values,
            self.params,
            self.signal_kind if signal_kind is None else signal_kind,
            family_name if self.family_curvature else None,
            other_family_name if self.family_curvature else None,
        )

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        tau = float(self.params["tau"])
        beta = float(self.params["beta"])
        x = self._retained_x(weights)

        features: list[np.ndarray] = []
        group_totals: list[np.ndarray] = []
        singleton_indices = self.packet.singletons if self.pair_cc_domains else list(range(self.packet.base.m))
        pair_map = self.packet.pairs if self.pair_cc_domains else []

        for idx in singleton_indices:
            family_name = self.domain_to_family[idx]
            features.append(self._feature_transform(x[:, idx], family_name=family_name)[:, None])
            group_totals.append(x[:, idx])

        for hi, lo in pair_map:
            pair_signal_total = x[:, hi] + (beta * x[:, lo] if self.quality_discount else x[:, lo])
            hi_family = self.domain_to_family[hi]
            lo_family = self.domain_to_family[lo]
            features.append(
                self._feature_transform(pair_signal_total, family_name=hi_family, other_family_name=lo_family)[:, None]
            )
            group_totals.append(x[:, hi] + x[:, lo])

        for family_name in self.family_totals:
            family_indices = self.packet.family_map[family_name]
            family_total = np.sum(x[:, family_indices], axis=1)
            features.append(
                self._feature_transform(
                    family_total,
                    signal_kind=self.family_signal_kind,
                    family_name=family_name,
                )[:, None]
            )

        penalty_inputs = np.stack(group_totals, axis=1)
        penalty = np.sum(softplus(np.log1p(penalty_inputs) - tau) ** 2, axis=1, keepdims=True)
        features.append(penalty)

        design = np.hstack(features)
        design[:, :-1] *= -1.0
        return design

    def fit(self, weights: np.ndarray, targets: np.ndarray) -> GenericFamilyFlexibleSignalSurrogate:
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
        design = self.build_design(weights)
        return np.asarray(self.intercept_ + design @ self.coef_, dtype=float)

    def components(self) -> dict[str, Any]:
        if self.coef_ is None:
            raise RuntimeError("Model must be fit")
        n_singletons = len(self.packet.singletons) if self.pair_cc_domains else self.packet.base.m
        n_pairs = len(self.packet.pairs) if self.pair_cc_domains else 0
        n_families = len(self.family_totals)
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
        penalty_coef = float(self.coef_[offset])
        return {
            "singleton_coef": singleton_coef,
            "pair_coef": pair_coef,
            "family_coef": family_coef,
            "penalty_coef": penalty_coef,
        }


def compute_flexible_surrogate_metrics(
    packet: GenericFamilyPacket,
    model: GenericFamilyFlexibleSignalSurrogate,
    *,
    seed: int = 0,
    valid_weights: np.ndarray | None = None,
    valid_y: np.ndarray | None = None,
) -> dict[str, float | bool]:
    y = packet.base.y
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []

    for tr, te in kf.split(packet.base.w):
        fold_model = GenericFamilyFlexibleSignalSurrogate(
            packet,
            params=model.params,
            signal_kind=model.signal_kind,
            family_signal_kind=model.family_signal_kind,
            family_totals=model.family_totals,
            quality_discount=model.quality_discount,
            pair_cc_domains=model.pair_cc_domains,
            family_curvature=model.family_curvature,
        ).fit(packet.base.w[tr], y[tr])
        pred = fold_model.predict(packet.base.w[te])
        oof[te] = pred
        fold_regrets.append(float(y[te][int(np.argmin(pred))] - np.min(y[te])))

    full_model = GenericFamilyFlexibleSignalSurrogate(
        packet,
        params=model.params,
        signal_kind=model.signal_kind,
        family_signal_kind=model.family_signal_kind,
        family_totals=model.family_totals,
        quality_discount=model.quality_discount,
        pair_cc_domains=model.pair_cc_domains,
        family_curvature=model.family_curvature,
    ).fit(packet.base.w, y)
    train_pred = full_model.predict(packet.base.w)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    lower_tail_count = max(5, int(np.ceil(0.15 * len(y))))
    tail_idx = np.argsort(oof)[:lower_tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))

    metrics: dict[str, float | bool] = {
        "train_rmse": float(np.sqrt(np.mean((train_pred - y) ** 2))),
        "train_r2": float(1.0 - float(np.sum((train_pred - y) ** 2)) / sst),
        "train_spearman": float(spearmanr(y, train_pred).statistic),
        "cv_rmse": float(np.sqrt(np.mean((oof - y) ** 2))),
        "cv_r2": float(1.0 - float(np.sum((oof - y) ** 2)) / sst),
        "cv_spearman": float(spearmanr(y, oof).statistic),
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "lower_tail_optimism": lower_tail_optimism,
    }
    if valid_weights is not None and valid_y is not None:
        anchor_pred = full_model.predict(valid_weights)
        metrics.update(
            {
                "anchor_mae": float(np.mean(np.abs(anchor_pred - valid_y))),
                "anchor_rmse": float(np.sqrt(np.mean((anchor_pred - valid_y) ** 2))),
                "anchor_rank_correct": bool(int(np.argmin(anchor_pred)) == int(np.argmin(valid_y))),
                "pred_validated_global": float(anchor_pred[0]),
                "pred_validated_pair": float(anchor_pred[1]),
            }
        )
    return metrics


def optimize_flexible_signal_model(
    packet: GenericFamilyPacket,
    model: GenericFamilyFlexibleSignalSurrogate,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit before optimization")

    parts = model.components()
    singleton_coef = parts["singleton_coef"]
    pair_coef = parts["pair_coef"]
    family_coef = parts["family_coef"]
    penalty_coef = float(parts["penalty_coef"])

    n_domains = packet.base.m
    c0 = packet.base.c0
    c1 = packet.base.c1
    eta = float(model.params["eta"])
    lam = float(model.params["lam"])
    tau = float(model.params["tau"])
    beta = float(model.params["beta"]) if model.quality_discount else 1.0
    rng = np.random.default_rng(seed)
    pair_map = packet.pairs if model.pair_cc_domains else []
    singleton_indices = packet.singletons if model.pair_cc_domains else list(range(packet.base.m))
    family_indices = {
        family_name: np.asarray(packet.family_map[family_name], dtype=int) for family_name in model.family_totals
    }

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

        for local_idx, domain_idx in enumerate(singleton_indices):
            coef = float(singleton_coef[local_idx])
            family_name = model.domain_to_family[domain_idx]
            value -= coef * model._feature_transform(np.asarray([x[domain_idx]]), family_name=family_name)[0]
            grad_x[domain_idx] -= (
                coef * model._feature_derivative(np.asarray([x[domain_idx]]), family_name=family_name)[0]
            )

        for local_idx, (hi, lo) in enumerate(pair_map):
            coef = float(pair_coef[local_idx])
            total = x[hi] + beta * x[lo]
            hi_family = model.domain_to_family[hi]
            lo_family = model.domain_to_family[lo]
            value -= (
                coef
                * model._feature_transform(np.asarray([total]), family_name=hi_family, other_family_name=lo_family)[0]
            )
            common = (
                coef
                * model._feature_derivative(
                    np.asarray([total]),
                    family_name=hi_family,
                    other_family_name=lo_family,
                )[0]
            )
            grad_x[hi] -= common
            grad_x[lo] -= common * beta

        for family_name in model.family_totals:
            coef = float(family_coef[family_name])
            members = family_indices[family_name]
            total = float(np.sum(x[members]))
            value -= (
                coef
                * model._feature_transform(
                    np.asarray([total]),
                    signal_kind=model.family_signal_kind,
                    family_name=family_name,
                )[0]
            )
            grad_x[members] -= (
                model._feature_derivative(
                    np.asarray([total]),
                    signal_kind=model.family_signal_kind,
                    family_name=family_name,
                )[0]
                * coef
            )

        if penalty_coef != 0.0:
            penalty_grad = np.zeros(n_domains, dtype=float)
            for domain_idx in singleton_indices:
                inside = np.log1p(x[domain_idx]) - tau
                sp = float(softplus(inside))
                if sp != 0.0:
                    grad_inside = 2.0 * sp * float(sigmoid(inside))
                    penalty_grad[domain_idx] += grad_inside / (1.0 + x[domain_idx])
            for hi, lo in pair_map:
                total = x[hi] + x[lo]
                inside = np.log1p(total) - tau
                sp = float(softplus(inside))
                if sp != 0.0:
                    grad_inside = 2.0 * sp * float(sigmoid(inside))
                    common = grad_inside / (1.0 + total)
                    penalty_grad[hi] += common
                    penalty_grad[lo] += common
            value += penalty_coef * np.sum(
                softplus(np.log1p(np.asarray([x[idx] for idx in singleton_indices])) - tau) ** 2
            )
            if pair_map:
                pair_totals = np.asarray([x[hi] + x[lo] for hi, lo in pair_map], dtype=float)
                value += penalty_coef * np.sum(softplus(np.log1p(pair_totals) - tau) ** 2)
            grad_x += penalty_coef * penalty_grad

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
        raise RuntimeError("Flexible-signal optimization failed")

    z = np.asarray(best.x, dtype=float)
    logits0 = z[:n_domains]
    logits1 = z[n_domains:]
    phase0 = np.exp(logits0 - np.max(logits0))
    phase0 /= np.sum(phase0)
    phase1 = np.exp(logits1 - np.max(logits1))
    phase1 /= np.sum(phase1)
    return best, phase0, phase1


def _flexible_variant_signal_kind(variant_name: str) -> str:
    if variant_name == "power_family":
        return "power"
    if variant_name == "boxcox_family":
        return "boxcox"
    if variant_name == "power_boxcox_family":
        return "power"
    return variant_name


def _flexible_variant_family_signal_kind(variant_name: str) -> str:
    if variant_name == "power_boxcox_family":
        return "boxcox"
    return _flexible_variant_signal_kind(variant_name)


def _flexible_variant_family_curvature(variant_name: str) -> bool:
    return bool(variant_name in {"power_family", "boxcox_family", "power_boxcox_family"})


def build_flexible_signal_surrogate(
    packet: GenericFamilyPacket,
    *,
    params: dict[str, float],
    variant_name: str,
    family_totals: tuple[str, ...] = GENERIC_FAMILY_NAMES,
    quality_discount: bool = True,
    pair_cc_domains: bool = True,
) -> GenericFamilyFlexibleSignalSurrogate:
    return GenericFamilyFlexibleSignalSurrogate(
        packet,
        params=params,
        signal_kind=_flexible_variant_signal_kind(variant_name),
        family_signal_kind=_flexible_variant_family_signal_kind(variant_name),
        family_totals=family_totals,
        quality_discount=quality_discount,
        pair_cc_domains=pair_cc_domains,
        family_curvature=_flexible_variant_family_curvature(variant_name),
    )


def flexible_signal_param_keys(variant_name: str) -> tuple[str, ...]:
    common = ("eta", "lam", "tau", "reg", "beta")
    if variant_name == "log":
        return ("alpha", *common)
    if variant_name == "power":
        return (*common, "a")
    if variant_name == "boxcox":
        return ("alpha", *common, "a")
    if variant_name == "power_family":
        return (*common, "a_broad_text", "a_tech_code", "a_reasoning")
    if variant_name == "boxcox_family":
        return ("alpha", *common, "a_broad_text", "a_tech_code", "a_reasoning")
    if variant_name == "power_boxcox_family":
        return ("alpha", *common, "a_broad_text", "a_tech_code", "a_reasoning")
    raise ValueError(f"Unsupported flexible variant {variant_name!r}")


def flexible_signal_params_from_metrics(metrics: dict[str, float | bool], variant_name: str) -> dict[str, float]:
    return {key: float(metrics[key]) for key in flexible_signal_param_keys(variant_name)}


def pack_flexible_signal_params_observed_only(params: dict[str, float], variant_name: str) -> np.ndarray:
    beta = float(np.clip(params["beta"], 1e-8, 1.0 - 1.0e-8))
    z = [
        np.log(float(params["eta"])),
        np.log(float(params["lam"])),
        float(params["tau"]),
        np.log(float(params["reg"])),
        np.log(beta / (1.0 - beta)),
    ]
    if variant_name == "log":
        z.append(np.log(float(params["alpha"])))
    elif variant_name == "power":
        z.append(np.log(float(params["a"])))
    elif variant_name == "boxcox":
        z.append(np.log(float(params["alpha"])))
        z.append(float(params["a"]))
    elif variant_name == "power_family":
        for family_name in GENERIC_FAMILY_NAMES:
            z.append(np.log(float(params[f"a_{family_name}"])))
    elif variant_name == "boxcox_family":
        z.append(np.log(float(params["alpha"])))
        for family_name in GENERIC_FAMILY_NAMES:
            z.append(float(params[f"a_{family_name}"]))
    elif variant_name == "power_boxcox_family":
        z.append(np.log(float(params["alpha"])))
        for family_name in GENERIC_FAMILY_NAMES:
            z.append(np.log(float(params[f"a_{family_name}"])))
    else:
        raise ValueError(f"Unsupported flexible variant {variant_name!r}")
    return np.asarray(z, dtype=float)


def _sigmoid_scalar_clipped(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def unpack_flexible_signal_params_observed_only(z: np.ndarray, variant_name: str) -> dict[str, float]:
    params = {
        "eta": float(np.exp(np.clip(z[0], -8.0, 8.0))),
        "lam": float(np.exp(np.clip(z[1], -12.0, 4.0))),
        "tau": float(np.clip(z[2], -2.0, 8.0)),
        "reg": float(np.exp(np.clip(z[3], -18.0, 0.0))),
        "beta": float(np.clip(_sigmoid_scalar_clipped(float(z[4])), 1e-6, 1.0 - 1e-6)),
    }
    idx = 5
    if variant_name == "log":
        params["alpha"] = float(np.exp(np.clip(z[idx], -8.0, 8.0)))
    elif variant_name == "power":
        params["a"] = float(np.exp(np.clip(z[idx], np.log(0.02), np.log(2.0))))
    elif variant_name == "boxcox":
        params["alpha"] = float(np.exp(np.clip(z[idx], -8.0, 8.0)))
        idx += 1
        params["a"] = float(np.clip(z[idx], -1.0, 2.0))
    elif variant_name == "power_family":
        for family_name in GENERIC_FAMILY_NAMES:
            params[f"a_{family_name}"] = float(np.exp(np.clip(z[idx], np.log(0.02), np.log(2.0))))
            idx += 1
    elif variant_name == "boxcox_family":
        params["alpha"] = float(np.exp(np.clip(z[idx], -8.0, 8.0)))
        idx += 1
        for family_name in GENERIC_FAMILY_NAMES:
            params[f"a_{family_name}"] = float(np.clip(z[idx], -1.0, 2.0))
            idx += 1
    elif variant_name == "power_boxcox_family":
        params["alpha"] = float(np.exp(np.clip(z[idx], -8.0, 8.0)))
        idx += 1
        for family_name in GENERIC_FAMILY_NAMES:
            params[f"a_{family_name}"] = float(np.exp(np.clip(z[idx], np.log(0.02), np.log(2.0))))
            idx += 1
    else:
        raise ValueError(f"Unsupported flexible variant {variant_name!r}")
    return params


def flexible_signal_start_bank_observed_only(variant_name: str) -> tuple[dict[str, float], ...]:
    bank: list[dict[str, float]] = []
    if variant_name == "log":
        bank.extend(dict(params) for params in OBSERVED_ONLY_START_PARAM_BANK)
        bank.append(dict(TUNED_GENERIC_FAMILY_PARAMS))
        for alpha in (4.0, 8.0, 16.0, 24.0):
            bank.append({"alpha": alpha, "eta": 8.0, "lam": 0.05, "tau": 3.0, "reg": 1e-3, "beta": 0.7})
    elif variant_name == "power":
        for a in (0.10, 0.18, 0.25, 0.35, 0.50, 0.80):
            for params in OBSERVED_ONLY_START_PARAM_BANK:
                row = {k: float(v) for k, v in params.items() if k != "alpha"}
                row["a"] = float(a)
                bank.append(row)
        bank.append(
            {
                "eta": float(TUNED_GENERIC_FAMILY_PARAMS["eta"]),
                "lam": float(TUNED_GENERIC_FAMILY_PARAMS["lam"]),
                "tau": float(TUNED_GENERIC_FAMILY_PARAMS["tau"]),
                "reg": float(TUNED_GENERIC_FAMILY_PARAMS["reg"]),
                "beta": float(TUNED_GENERIC_FAMILY_PARAMS["beta"]),
                "a": 0.375,
            }
        )
    elif variant_name == "boxcox":
        for alpha in (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0):
            for a in (-0.4, -0.2, -0.05, 0.0, 0.15, 0.3, 0.5, 0.8):
                for params in OBSERVED_ONLY_START_PARAM_BANK[:2]:
                    row = dict(params)
                    row["alpha"] = float(alpha)
                    row["a"] = float(a)
                    bank.append(row)
        bank.append(
            {
                "alpha": 24.0,
                "eta": float(TUNED_GENERIC_FAMILY_PARAMS["eta"]),
                "lam": float(TUNED_GENERIC_FAMILY_PARAMS["lam"]),
                "tau": float(TUNED_GENERIC_FAMILY_PARAMS["tau"]),
                "reg": float(TUNED_GENERIC_FAMILY_PARAMS["reg"]),
                "beta": float(TUNED_GENERIC_FAMILY_PARAMS["beta"]),
                "a": 0.4,
            }
        )
    elif variant_name == "power_family":
        for params in OBSERVED_ONLY_START_PARAM_BANK:
            for a_broad in (0.18, 0.28, 0.40):
                for a_tech in (0.18, 0.32, 0.56):
                    for a_reason in (0.12, 0.28, 0.56):
                        row = {k: float(v) for k, v in params.items() if k != "alpha"}
                        row["a_broad_text"] = float(a_broad)
                        row["a_tech_code"] = float(a_tech)
                        row["a_reasoning"] = float(a_reason)
                        bank.append(row)
    elif variant_name == "boxcox_family":
        for params in OBSERVED_ONLY_START_PARAM_BANK:
            for alpha in (4.0, 8.0, 16.0):
                for a_broad in (0.0, 0.10, 0.25):
                    for a_tech in (-0.10, 0.10, 0.35):
                        for a_reason in (-0.10, 0.10, 0.35):
                            row = dict(params)
                            row["alpha"] = float(alpha)
                            row["a_broad_text"] = float(a_broad)
                            row["a_tech_code"] = float(a_tech)
                            row["a_reasoning"] = float(a_reason)
                            bank.append(row)
    elif variant_name == "power_boxcox_family":
        for params in OBSERVED_ONLY_START_PARAM_BANK:
            for alpha in (4.0, 8.0, 16.0):
                for a_broad in (0.18, 0.40, 0.60):
                    for a_tech in (0.02, 0.10, 0.32):
                        for a_reason in (0.08, 0.20, 0.35):
                            row = {k: float(v) for k, v in params.items()}
                            row["alpha"] = float(alpha)
                            row["a_broad_text"] = float(a_broad)
                            row["a_tech_code"] = float(a_tech)
                            row["a_reasoning"] = float(a_reason)
                            bank.append(row)
    else:
        raise ValueError(f"Unsupported flexible variant {variant_name!r}")
    return tuple(bank)


def flexible_signal_oof_metrics_observed_only(
    packet: GenericFamilyPacket,
    params: dict[str, float],
    *,
    variant_name: str,
    seed: int = 0,
    lower_tail_frac: float = OBSERVED_ONLY_LOWER_TAIL_FRAC,
) -> dict[str, float]:
    y = packet.base.y
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    for tr, te in kf.split(packet.base.w):
        model = build_flexible_signal_surrogate(
            packet,
            params=params,
            variant_name=variant_name,
        ).fit(packet.base.w[tr], y[tr])
        pred = model.predict(packet.base.w[te])
        oof[te] = pred
        fold_regrets.append(float(y[te][int(np.argmin(pred))] - np.min(y[te])))

    residuals = oof - y
    tail_count = max(5, int(np.ceil(float(lower_tail_frac) * float(len(packet.base.y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    return {
        "cv_rmse": cv_rmse,
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "lower_tail_optimism": lower_tail_optimism,
        "objective": (
            OBSERVED_ONLY_CV_WEIGHT * cv_rmse
            + OBSERVED_ONLY_FOLDMEAN_WEIGHT * float(np.mean(fold_regrets))
            + OBSERVED_ONLY_TAIL_WEIGHT * lower_tail_optimism
        ),
    }


def evaluate_flexible_signal_params_observed_only(
    z: np.ndarray,
    packet: GenericFamilyPacket,
    *,
    variant_name: str,
    seed: int = 0,
) -> dict[str, float]:
    params = unpack_flexible_signal_params_observed_only(z, variant_name)
    return {**params, **flexible_signal_oof_metrics_observed_only(packet, params, variant_name=variant_name, seed=seed)}


def tune_flexible_signal_params_observed_only(
    packet: GenericFamilyPacket,
    *,
    variant_name: str,
    method: str = "Powell",
    coarse_top_k: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | bool], Any]:
    start_bank = flexible_signal_start_bank_observed_only(variant_name)
    coarse_rows: list[dict[str, float | bool]] = []
    for start_id, params in enumerate(start_bank):
        coarse_rows.append(
            {
                "variant": variant_name,
                "stage": "coarse",
                "start_id": int(start_id),
                **params,
                **flexible_signal_oof_metrics_observed_only(packet, params, variant_name=variant_name, seed=seed),
            }
        )
    coarse_frame = pd.DataFrame.from_records(coarse_rows).sort_values(
        ["objective", "cv_rmse", "cv_foldmean_regret_at_1"],
        ascending=[True, True, True],
    )
    chosen_ids = coarse_frame["start_id"].head(int(coarse_top_k)).tolist()

    best_metrics: dict[str, float | bool] | None = None
    best_result: Any | None = None
    best_objective = float("inf")
    refine_rows: list[dict[str, float | bool]] = []

    for chosen_rank, start_id in enumerate(chosen_ids):
        start = pack_flexible_signal_params_observed_only(start_bank[start_id], variant_name)
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, _cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in _cache:
                _cache[key] = float(
                    evaluate_flexible_signal_params_observed_only(z, packet, variant_name=variant_name, seed=seed)[
                        "objective"
                    ]
                )
            return _cache[key]

        options = {
            "L-BFGS-B": {"maxiter": 160, "ftol": 1e-6},
            "Nelder-Mead": {"maxiter": 700, "xatol": 1e-4, "fatol": 1e-6},
            "Powell": {"maxiter": 100, "xtol": 1e-4, "ftol": 1e-6},
        }.get(method, {"maxiter": 120})
        result = minimize(objective, start, method=method, options=options)
        metrics = evaluate_flexible_signal_params_observed_only(
            np.asarray(result.x, dtype=float),
            packet,
            variant_name=variant_name,
            seed=seed,
        )
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
        raise RuntimeError(f"Flexible-signal tuning failed for variant {variant_name!r}")
    return coarse_frame, pd.DataFrame.from_records(refine_rows), best_metrics, best_result


def deploy_flexible_signal_gaincapped_topkactual(
    packet: GenericFamilyPacket,
    model: GenericFamilyFlexibleSignalSurrogate,
    tuning_metrics: dict[str, float | bool],
    *,
    top_k: int = TRUSTBLEND_TOPK_ACTUAL,
    line_grid: int = TRUSTBLEND_LINE_GRID,
) -> dict[str, Any]:
    raw_result, phase0, phase1 = optimize_flexible_signal_model(packet, model, seed=0)
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
        raise RuntimeError("Flexible-signal gain-capped deployment selection failed")

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
