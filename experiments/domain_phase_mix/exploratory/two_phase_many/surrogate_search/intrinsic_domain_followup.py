# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CAMEL-style intrinsic-domain ablations for GRP."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.decomposition import NMF

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyPacket,
    GenericFamilySignalTransform,
    TUNED_GENERIC_FAMILY_PARAMS,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    softplus,
)

DEFAULT_INTRINSIC_DOMAIN_COUNT = 5
DEFAULT_NMF_MAX_ITER = 2_000


class IntrinsicFeatureMode(StrEnum):
    """How intrinsic domains enter the GRP design."""

    SOFT_FAMILY = "soft_family"
    LATENT_BOTTLENECK = "latent_bottleneck"


class IntrinsicPenaltyMode(StrEnum):
    """Where the overexposure penalty is applied."""

    GROUP = "group"
    LATENT = "latent"


@dataclass(frozen=True)
class IntrinsicGroupBasis:
    """A deterministic soft assignment from GRP groups to intrinsic domains."""

    memberships: np.ndarray
    group_names: tuple[str, ...]
    num_intrinsic_domains: int


def _singleton_indices(packet: GenericFamilyPacket, pair_cc_domains: bool) -> list[int]:
    return packet.singletons if pair_cc_domains else list(range(packet.base.m))


def _pair_map(packet: GenericFamilyPacket, pair_cc_domains: bool) -> list[tuple[int, int]]:
    return packet.pairs if pair_cc_domains else []


def intrinsic_group_names(packet: GenericFamilyPacket, *, pair_cc_domains: bool = True) -> tuple[str, ...]:
    """Return the ordered GRP group names used by intrinsic-domain ablations."""
    names = [packet.base.domain_names[idx] for idx in _singleton_indices(packet, pair_cc_domains)]
    names.extend(f"dolma3_cc/{topic}" for topic in (packet.pair_topics if pair_cc_domains else []))
    return tuple(names)


def group_signal_totals_from_retained_x(
    packet: GenericFamilyPacket,
    retained_x: np.ndarray,
    *,
    beta: float,
    pair_cc_domains: bool = True,
) -> np.ndarray:
    """Return the GRP group totals used for singleton/pair signal features."""
    totals: list[np.ndarray] = []
    for idx in _singleton_indices(packet, pair_cc_domains):
        totals.append(retained_x[:, idx])
    for hi, lo in _pair_map(packet, pair_cc_domains):
        totals.append(retained_x[:, hi] + beta * retained_x[:, lo])
    return np.stack(totals, axis=1)


def group_penalty_totals_from_retained_x(
    packet: GenericFamilyPacket,
    retained_x: np.ndarray,
    *,
    pair_cc_domains: bool = True,
) -> np.ndarray:
    """Return the GRP group totals used for the overexposure penalty."""
    totals: list[np.ndarray] = []
    for idx in _singleton_indices(packet, pair_cc_domains):
        totals.append(retained_x[:, idx])
    for hi, lo in _pair_map(packet, pair_cc_domains):
        totals.append(retained_x[:, hi] + retained_x[:, lo])
    return np.stack(totals, axis=1)


def _retained_x(
    packet: GenericFamilyPacket,
    weights: np.ndarray,
    *,
    eta: float,
    lam: float,
) -> np.ndarray:
    p0 = weights[:, 0, :]
    p1 = weights[:, 1, :]
    e0 = p0 * packet.base.c0[None, :]
    e1 = p1 * packet.base.c1[None, :]
    return np.exp(-lam * (1.0 - p1)) * e0 + eta * e1


def learn_intrinsic_group_basis(
    packet: GenericFamilyPacket | None = None,
    *,
    params: dict[str, float] | None = None,
    num_intrinsic_domains: int = DEFAULT_INTRINSIC_DOMAIN_COUNT,
    pair_cc_domains: bool = True,
    quality_discount: bool = True,
    target: str = MANY_DOMAIN_TARGET,
) -> IntrinsicGroupBasis:
    """Learn deterministic soft intrinsic-domain memberships over GRP groups."""
    packet = load_generic_family_packet(target=target) if packet is None else packet
    params = dict(TUNED_GENERIC_FAMILY_PARAMS if params is None else params)
    beta = float(params["beta"]) if quality_discount else 1.0
    retained_x = _retained_x(packet, packet.base.w, eta=float(params["eta"]), lam=float(params["lam"]))
    group_signal_totals = group_signal_totals_from_retained_x(
        packet,
        retained_x,
        beta=beta,
        pair_cc_domains=pair_cc_domains,
    )
    group_profiles = np.asarray(group_signal_totals.T, dtype=float)
    nmf = NMF(
        n_components=num_intrinsic_domains,
        init="nndsvda",
        random_state=0,
        max_iter=DEFAULT_NMF_MAX_ITER,
    )
    memberships = nmf.fit_transform(np.maximum(group_profiles, 0.0))
    row_sums = np.sum(memberships, axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    memberships = memberships / row_sums
    return IntrinsicGroupBasis(
        memberships=np.asarray(memberships, dtype=float),
        group_names=intrinsic_group_names(packet, pair_cc_domains=pair_cc_domains),
        num_intrinsic_domains=num_intrinsic_domains,
    )


class IntrinsicDomainRetainedTotalSurrogate:
    """GRP variant that replaces hard-coded families with learned intrinsic domains."""

    def __init__(
        self,
        packet: GenericFamilyPacket,
        basis: IntrinsicGroupBasis,
        *,
        params: dict[str, float] | None = None,
        feature_mode: IntrinsicFeatureMode = IntrinsicFeatureMode.SOFT_FAMILY,
        penalty_mode: IntrinsicPenaltyMode = IntrinsicPenaltyMode.GROUP,
        quality_discount: bool = True,
        pair_cc_domains: bool = True,
        include_penalty: bool = True,
        signal_transform: GenericFamilySignalTransform = GenericFamilySignalTransform.LOG_SATIETY,
    ):
        self.packet = packet
        self.basis = basis
        self.params = dict(TUNED_GENERIC_FAMILY_PARAMS if params is None else params)
        self.feature_mode = feature_mode
        self.penalty_mode = penalty_mode
        self.quality_discount = bool(quality_discount)
        self.pair_cc_domains = bool(pair_cc_domains)
        self.include_penalty = bool(include_penalty)
        self.signal_transform = signal_transform
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _retained_x(self, weights: np.ndarray) -> np.ndarray:
        return _retained_x(
            self.packet,
            weights,
            eta=float(self.params["eta"]),
            lam=float(self.params["lam"]),
        )

    def _group_signal_totals(self, retained_x: np.ndarray) -> np.ndarray:
        beta = float(self.params["beta"]) if self.quality_discount else 1.0
        return group_signal_totals_from_retained_x(
            self.packet,
            retained_x,
            beta=beta,
            pair_cc_domains=self.pair_cc_domains,
        )

    def _group_penalty_totals(self, retained_x: np.ndarray) -> np.ndarray:
        return group_penalty_totals_from_retained_x(
            self.packet,
            retained_x,
            pair_cc_domains=self.pair_cc_domains,
        )

    def latent_totals(self, weights: np.ndarray) -> np.ndarray:
        """Return the intrinsic-domain totals induced by the current schedule."""
        retained_x = self._retained_x(weights)
        group_signal_totals = self._group_signal_totals(retained_x)
        return np.asarray(group_signal_totals @ self.basis.memberships, dtype=float)

    def _signal_feature(self, signal: np.ndarray, alpha: float) -> np.ndarray:
        if self.signal_transform == GenericFamilySignalTransform.LOG_SATIETY:
            return np.log1p(alpha * signal)
        if self.signal_transform == GenericFamilySignalTransform.POWER:
            if alpha <= 0.0:
                raise ValueError(f"Power-law alpha must be positive, got {alpha}")
            return np.power(np.clip(signal, 0.0, None), alpha)
        raise ValueError(f"Unsupported signal_transform: {self.signal_transform}")

    def _design_from_group_and_latent(
        self,
        group_signal_totals: np.ndarray,
        penalty_inputs: np.ndarray | None,
    ) -> np.ndarray:
        alpha = float(self.params["alpha"])
        tau = float(self.params["tau"])
        latent_totals = np.asarray(group_signal_totals @ self.basis.memberships, dtype=float)
        features: list[np.ndarray] = []
        if self.feature_mode == IntrinsicFeatureMode.SOFT_FAMILY:
            features.append(self._signal_feature(group_signal_totals, alpha))
        features.append(self._signal_feature(latent_totals, alpha))
        if self.include_penalty:
            if penalty_inputs is None:
                raise RuntimeError("Penalty inputs must be provided when include_penalty=True")
            penalty = np.sum(softplus(np.log1p(penalty_inputs) - tau) ** 2, axis=1, keepdims=True)
            features.append(penalty)
        design = np.hstack(features)
        num_signal = design.shape[1] - int(self.include_penalty)
        design[:, :num_signal] *= -1.0
        return design

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        retained_x = self._retained_x(weights)
        group_signal_totals = self._group_signal_totals(retained_x)
        penalty_inputs = None
        if self.include_penalty:
            if self.penalty_mode == IntrinsicPenaltyMode.GROUP:
                penalty_inputs = self._group_penalty_totals(retained_x)
            else:
                penalty_inputs = np.asarray(group_signal_totals @ self.basis.memberships, dtype=float)
        return self._design_from_group_and_latent(group_signal_totals, penalty_inputs)

    def fit(self, weights: np.ndarray, targets: np.ndarray) -> IntrinsicDomainRetainedTotalSurrogate:
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
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(target_mean - (design_mean @ self.coef_).item())
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        design = self.build_design(weights)
        return np.asarray(self.intercept_ + design @ self.coef_, dtype=float)

    def predict_from_latent_totals(self, latent_totals: np.ndarray) -> np.ndarray:
        """Predict directly from latent totals for latent-only models."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        if self.feature_mode != IntrinsicFeatureMode.LATENT_BOTTLENECK:
            raise ValueError("Direct latent prediction only applies to latent-bottleneck models")
        penalty_inputs = (
            latent_totals if self.include_penalty and self.penalty_mode == IntrinsicPenaltyMode.LATENT else None
        )
        design = self._design_from_group_and_latent(
            np.zeros((latent_totals.shape[0], self.basis.memberships.shape[0]), dtype=float),
            penalty_inputs,
        )
        latent_signal = self._signal_feature(latent_totals, float(self.params["alpha"]))
        signal_width = latent_signal.shape[1]
        design[:, :signal_width] = -latent_signal
        return np.asarray(self.intercept_ + design @ self.coef_, dtype=float)


def intrinsic_param_count(
    model: IntrinsicDomainRetainedTotalSurrogate,
    *,
    count_basis: bool = True,
) -> int:
    """Return the effective parameter count for an intrinsic-domain GRP variant."""
    if model.coef_ is None:
        raise RuntimeError("Model must be fit before counting parameters")
    linear = len(model.coef_) + 1 + len(model.params)
    if not count_basis:
        return linear
    group_count, latent_count = model.basis.memberships.shape
    return linear + group_count * (latent_count - 1)


def optimize_intrinsic_domain_model(
    packet: GenericFamilyPacket,
    model: IntrinsicDomainRetainedTotalSurrogate,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Optimize a fitted intrinsic-domain surrogate over the two phase simplices."""
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit")

    n_domains = packet.base.m
    c0 = packet.base.c0
    c1 = packet.base.c1
    alpha = float(model.params["alpha"])
    eta = float(model.params["eta"])
    lam = float(model.params["lam"])
    tau = float(model.params["tau"])
    beta = float(model.params["beta"]) if model.quality_discount else 1.0
    rng = np.random.default_rng(seed)
    power_eps = 1e-12

    singleton_indices = _singleton_indices(packet, model.pair_cc_domains)
    pair_map = _pair_map(packet, model.pair_cc_domains)
    n_singletons = len(singleton_indices)
    n_pairs = len(pair_map)
    group_count = n_singletons + n_pairs
    latent_count = model.basis.num_intrinsic_domains
    memberships = np.asarray(model.basis.memberships, dtype=float)

    offset = 0
    if model.feature_mode == IntrinsicFeatureMode.SOFT_FAMILY:
        group_coef = np.asarray(model.coef_[offset : offset + group_count], dtype=float)
        offset += group_count
    else:
        group_coef = np.zeros(group_count, dtype=float)
    latent_coef = np.asarray(model.coef_[offset : offset + latent_count], dtype=float)
    offset += latent_count
    penalty_coef = float(model.coef_[offset]) if model.include_penalty else 0.0

    def signal_value_grad(total: float) -> tuple[float, float]:
        if model.signal_transform == GenericFamilySignalTransform.LOG_SATIETY:
            return np.log1p(alpha * total), alpha / (1.0 + alpha * total)
        if model.signal_transform == GenericFamilySignalTransform.POWER:
            safe_total = max(total, power_eps)
            return safe_total**alpha, alpha * safe_total ** (alpha - 1.0)
        raise ValueError(f"Unsupported signal_transform: {model.signal_transform}")

    def backprop_signal_group_grad(group_grad: np.ndarray, grad_x: np.ndarray) -> None:
        for local_idx, domain_idx in enumerate(singleton_indices):
            grad_x[domain_idx] += group_grad[local_idx]
        for local_idx, (hi, lo) in enumerate(pair_map):
            common = group_grad[n_singletons + local_idx]
            grad_x[hi] += common
            grad_x[lo] += common * beta

    def backprop_penalty_group_grad(group_grad: np.ndarray, grad_x: np.ndarray) -> None:
        for local_idx, domain_idx in enumerate(singleton_indices):
            grad_x[domain_idx] += group_grad[local_idx]
        for local_idx, (hi, lo) in enumerate(pair_map):
            common = group_grad[n_singletons + local_idx]
            grad_x[hi] += common
            grad_x[lo] += common

    def value_grad_logits(z: np.ndarray) -> tuple[float, np.ndarray]:
        logits0 = z[:n_domains]
        logits1 = z[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 /= np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 /= np.sum(p1)

        e0 = c0 * p0
        retained = np.exp(-lam * (1.0 - p1))
        retained_x = retained * e0 + eta * c1 * p1
        dx_dp0 = retained * c0
        dx_dp1 = lam * retained * e0 + eta * c1

        group_signal_totals = group_signal_totals_from_retained_x(
            packet,
            retained_x[None, :],
            beta=beta,
            pair_cc_domains=model.pair_cc_domains,
        )[0]
        latent_totals = np.asarray(group_signal_totals @ memberships, dtype=float)

        value = float(model.intercept_)
        grad_x = np.zeros(n_domains, dtype=float)
        group_signal_grad = np.zeros(group_count, dtype=float)

        if model.feature_mode == IntrinsicFeatureMode.SOFT_FAMILY:
            for group_idx in range(group_count):
                signal, signal_grad = signal_value_grad(float(group_signal_totals[group_idx]))
                value -= float(group_coef[group_idx]) * signal
                group_signal_grad[group_idx] -= float(group_coef[group_idx]) * signal_grad

        latent_grad = np.zeros(latent_count, dtype=float)
        for intrinsic_idx in range(latent_count):
            signal, signal_grad = signal_value_grad(float(latent_totals[intrinsic_idx]))
            value -= float(latent_coef[intrinsic_idx]) * signal
            latent_grad[intrinsic_idx] -= float(latent_coef[intrinsic_idx]) * signal_grad

        if np.any(latent_grad):
            group_signal_grad += memberships @ latent_grad

        if np.any(group_signal_grad):
            backprop_signal_group_grad(group_signal_grad, grad_x)

        if penalty_coef != 0.0:
            if model.penalty_mode == IntrinsicPenaltyMode.GROUP:
                penalty_inputs = group_penalty_totals_from_retained_x(
                    packet,
                    retained_x[None, :],
                    pair_cc_domains=model.pair_cc_domains,
                )[0]
                penalty_group_grad = np.zeros(group_count, dtype=float)
                for group_idx in range(group_count):
                    total = float(penalty_inputs[group_idx])
                    u = np.log1p(total) - tau
                    sp = float(softplus(np.asarray(u)))
                    value += penalty_coef * sp**2
                    sigmoid_u = float(1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0))))
                    penalty_group_grad[group_idx] += penalty_coef * (2.0 * sp * sigmoid_u / (1.0 + total))
                backprop_penalty_group_grad(penalty_group_grad, grad_x)
            else:
                latent_penalty_grad = np.zeros(latent_count, dtype=float)
                for intrinsic_idx in range(latent_count):
                    total = float(latent_totals[intrinsic_idx])
                    u = np.log1p(total) - tau
                    sp = float(softplus(np.asarray(u)))
                    value += penalty_coef * sp**2
                    sigmoid_u = float(1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0))))
                    latent_penalty_grad[intrinsic_idx] += penalty_coef * (2.0 * sp * sigmoid_u / (1.0 + total))
                if np.any(latent_penalty_grad):
                    backprop_signal_group_grad(memberships @ latent_penalty_grad, grad_x)

        grad_p0 = grad_x * dx_dp0
        grad_p1 = grad_x * dx_dp1
        grad0 = p0 * (grad_p0 - float(np.dot(grad_p0, p0)))
        grad1 = p1 * (grad_p1 - float(np.dot(grad_p1, p1)))
        return value, np.concatenate([grad0, grad1])

    starts = [
        np.zeros(2 * n_domains, dtype=float),
        *[
            np.concatenate(
                [
                    np.log(rng.dirichlet(np.ones(n_domains))),
                    np.log(rng.dirichlet(np.ones(n_domains))),
                ]
            )
            for _ in range(n_random)
        ],
    ]

    best_result = None
    best_value = float("inf")
    for start in starts:
        result = minimize(value_grad_logits, start, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Intrinsic-domain optimization failed")

    logits0 = best_result.x[:n_domains]
    logits1 = best_result.x[n_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 /= np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 /= np.sum(p1)
    return best_result, np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
