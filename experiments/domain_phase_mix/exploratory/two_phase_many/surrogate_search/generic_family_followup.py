# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic family retained-total surrogate for many-domain Uncheatable BPB."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    PacketData,
    load_two_phase_many_packet,
    softplus,
)

GENERIC_FAMILY_NAMES = ("broad_text", "tech_code", "reasoning")
TUNED_GENERIC_FAMILY_PARAMS = {
    "alpha": 12.94088092035213,
    "eta": 13.229384772843037,
    "lam": 0.035627177458741076,
    "tau": 3.2740751832677875,
    "reg": 0.0010114720923828182,
    "beta": 0.6634021668256815,
}


class GenericFamilySignalTransform(StrEnum):
    """Signal feature transform for GRP-family surrogates."""

    LOG_SATIETY = "log_satiety"
    POWER = "power"


@dataclass(frozen=True)
class GenericFamilyPacket:
    """Two-phase many-domain packet augmented with CC pairs and source families."""

    base: PacketData
    pairs: list[tuple[int, int]]
    pair_topics: list[str]
    singletons: list[int]
    family_map: dict[str, list[int]]


def load_generic_family_packet(target: str = MANY_DOMAIN_TARGET) -> GenericFamilyPacket:
    """Load the many-domain packet with CC pair structure and family assignments."""
    base = load_two_phase_many_packet(target=target)
    pairs: list[tuple[int, int]] = []
    pair_topics: list[str] = []
    paired: set[int] = set()

    for idx, domain_name in enumerate(base.domain_names):
        if idx in paired:
            continue
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high"):
            low_name = domain_name[:-5] + "_low"
            if low_name in base.domain_names:
                low_idx = base.domain_names.index(low_name)
                pairs.append((idx, low_idx))
                pair_topics.append(domain_name[len("dolma3_cc/") : -5])
                paired.add(idx)
                paired.add(low_idx)

    singletons = [idx for idx in range(base.m) if idx not in paired]
    family_map = {family_name: [] for family_name in GENERIC_FAMILY_NAMES}
    for idx, domain_name in enumerate(base.domain_names):
        is_broad = (
            domain_name.startswith("dolma3_cc/")
            or domain_name
            in {
                "dolma3_wikipedia",
                "dolmino_common_crawl_hq",
                "dolmino_olmocr_pdfs_hq",
                "dolmino_stem_heavy_crawl",
            }
            or domain_name.endswith("synth_qa")
        )
        is_tech = any(token in domain_name for token in ("stack_edu", "synth_code", "synth_math")) or (
            domain_name in {"dolma3_arxiv", "dolma3_finemath_3plus"}
        )
        is_reasoning = domain_name in {"dolmino_synth_instruction", "dolmino_synth_thinking"}

        if is_broad:
            family_map["broad_text"].append(idx)
        if is_tech:
            family_map["tech_code"].append(idx)
        if is_reasoning:
            family_map["reasoning"].append(idx)

    return GenericFamilyPacket(
        base=base,
        pairs=pairs,
        pair_topics=pair_topics,
        singletons=singletons,
        family_map=family_map,
    )


def family_shares(packet: GenericFamilyPacket, weights: np.ndarray) -> dict[str, float]:
    """Return family mass shares for both phases."""
    shares: dict[str, float] = {}
    for phase_idx in (0, 1):
        for family_name in GENERIC_FAMILY_NAMES:
            shares[f"phase{phase_idx}_{family_name}"] = float(weights[phase_idx, packet.family_map[family_name]].sum())
    return shares


def optimize_generic_family_convex_hull(
    model: GenericFamilyRetainedTotalSurrogate,
    anchors: np.ndarray,
    *,
    maxiter: int = 100,
    start_indices: np.ndarray | None = None,
    linear_penalty: np.ndarray | None = None,
    pairwise_penalty: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Optimize the model over convex combinations of anchor schedules."""
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit before convex-hull optimization")

    num_anchors = anchors.shape[0]
    if linear_penalty is not None and linear_penalty.shape != (num_anchors,):
        raise ValueError(f"linear_penalty shape {linear_penalty.shape} != ({num_anchors},)")
    if pairwise_penalty is not None and pairwise_penalty.shape != (num_anchors, num_anchors):
        raise ValueError(f"pairwise_penalty shape {pairwise_penalty.shape} != ({num_anchors}, {num_anchors})")

    def objective(z: np.ndarray) -> float:
        shifted = z - np.max(z)
        coeffs = np.exp(shifted)
        coeffs /= np.sum(coeffs)
        weights = np.tensordot(coeffs, anchors, axes=1)[None, :, :]
        value = float(model.predict(weights)[0])
        if linear_penalty is not None:
            value += float(linear_penalty @ coeffs)
        if pairwise_penalty is not None:
            value += float(coeffs @ pairwise_penalty @ coeffs)
        return value

    if start_indices is None:
        vertex_indices = range(num_anchors)
    else:
        vertex_indices = np.asarray(start_indices, dtype=int).tolist()
    starts = [np.zeros(num_anchors, dtype=float)] + [
        np.eye(num_anchors, dtype=float)[idx] * 4.0 for idx in vertex_indices
    ]
    best_result = None
    best_value = float("inf")
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": maxiter})
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Generic-family convex-hull optimization failed")

    shifted = np.asarray(best_result.x, dtype=float) - np.max(best_result.x)
    best_coeffs = np.exp(shifted)
    best_coeffs /= np.sum(best_coeffs)
    best_weights = np.tensordot(best_coeffs, anchors, axes=1)
    predicted_value = float(model.predict(best_weights[None, :, :])[0])
    return predicted_value, best_coeffs, best_weights


class GenericFamilyRetainedTotalSurrogate:
    """Generic family retained-total surrogate with paired CC buckets."""

    def __init__(
        self,
        packet: GenericFamilyPacket,
        *,
        params: dict[str, float] | None = None,
        family_totals: tuple[str, ...] = GENERIC_FAMILY_NAMES,
        quality_discount: bool = True,
        pair_cc_domains: bool = True,
        include_penalty: bool = True,
        signal_transform: GenericFamilySignalTransform = GenericFamilySignalTransform.LOG_SATIETY,
    ):
        self.packet = packet
        self.params = dict(TUNED_GENERIC_FAMILY_PARAMS if params is None else params)
        self.family_totals = tuple(family_totals)
        self.quality_discount = bool(quality_discount)
        self.pair_cc_domains = bool(pair_cc_domains)
        self.include_penalty = bool(include_penalty)
        self.signal_transform = signal_transform
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

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        alpha = float(self.params["alpha"])
        tau = float(self.params["tau"])
        beta = float(self.params["beta"])
        x = self._retained_x(weights)

        features: list[np.ndarray] = []
        group_totals: list[np.ndarray] = []

        singleton_indices = self.packet.singletons if self.pair_cc_domains else list(range(self.packet.base.m))
        pair_map = self.packet.pairs if self.pair_cc_domains else []

        for idx in singleton_indices:
            features.append(self._signal_feature(x[:, idx : idx + 1], alpha))
            group_totals.append(x[:, idx])

        for hi, lo in pair_map:
            pair_signal_total = x[:, hi] + (beta * x[:, lo] if self.quality_discount else x[:, lo])
            features.append(self._signal_feature(pair_signal_total[:, None], alpha))
            group_totals.append(x[:, hi] + x[:, lo])

        for family_name in self.family_totals:
            family_indices = self.packet.family_map[family_name]
            family_total = np.sum(x[:, family_indices], axis=1)
            features.append(self._signal_feature(family_total[:, None], alpha))

        if self.include_penalty:
            penalty_inputs = np.stack(group_totals, axis=1)
            penalty = np.sum(softplus(np.log1p(penalty_inputs) - tau) ** 2, axis=1, keepdims=True)
            features.append(penalty)

        design = np.hstack(features)
        num_signal = design.shape[1] - 1
        design[:, :num_signal] *= -1.0
        return design

    def _signal_feature(self, signal: np.ndarray, alpha: float) -> np.ndarray:
        """Map retained totals to a one-column signal feature."""
        if self.signal_transform == GenericFamilySignalTransform.LOG_SATIETY:
            return np.log1p(alpha * signal)
        if self.signal_transform == GenericFamilySignalTransform.POWER:
            if alpha <= 0.0:
                raise ValueError(f"Power-law alpha must be positive, got {alpha}")
            return np.power(np.clip(signal, 0.0, None), alpha)
        raise ValueError(f"Unsupported signal_transform: {self.signal_transform}")

    def fit(self, weights: np.ndarray, targets: np.ndarray) -> GenericFamilyRetainedTotalSurrogate:
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


def fitted_generic_family_components(
    packet: GenericFamilyPacket,
    model: GenericFamilyRetainedTotalSurrogate,
) -> dict[str, Any]:
    """Extract fitted coefficients grouped by feature type."""
    if model.coef_ is None:
        raise RuntimeError("Model must be fit")

    n_singletons = len(packet.singletons) if model.pair_cc_domains else packet.base.m
    n_pairs = len(packet.pairs) if model.pair_cc_domains else 0
    n_families = len(model.family_totals)

    offset = 0
    singleton_coef = np.asarray(model.coef_[offset : offset + n_singletons], dtype=float)
    offset += n_singletons
    pair_coef = np.asarray(model.coef_[offset : offset + n_pairs], dtype=float)
    offset += n_pairs
    family_coef = {
        family_name: float(coef)
        for family_name, coef in zip(
            model.family_totals,
            model.coef_[offset : offset + n_families],
            strict=True,
        )
    }
    offset += n_families
    penalty_coef = float(model.coef_[offset]) if model.include_penalty else 0.0
    return {
        "singleton_coef": singleton_coef,
        "pair_coef": pair_coef,
        "family_coef": family_coef,
        "penalty_coef": penalty_coef,
    }


def optimize_generic_family_model(
    packet: GenericFamilyPacket,
    model: GenericFamilyRetainedTotalSurrogate,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Optimize a fitted generic-family model over the two phase simplices."""
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit")

    parts = fitted_generic_family_components(packet, model)
    singleton_coef = parts["singleton_coef"]
    pair_coef = parts["pair_coef"]
    family_coef = parts["family_coef"]
    penalty_coef = float(parts["penalty_coef"])

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

    pair_topics = packet.pair_topics if model.pair_cc_domains else []
    pair_map = packet.pairs if model.pair_cc_domains else []
    singleton_indices = packet.singletons if model.pair_cc_domains else list(range(packet.base.m))
    family_indices = {
        family_name: np.asarray(packet.family_map[family_name], dtype=int) for family_name in model.family_totals
    }

    def signal_value_grad(total: float) -> tuple[float, float]:
        if model.signal_transform == GenericFamilySignalTransform.LOG_SATIETY:
            return np.log1p(alpha * total), alpha / (1.0 + alpha * total)
        if model.signal_transform == GenericFamilySignalTransform.POWER:
            safe_total = max(total, power_eps)
            return safe_total**alpha, alpha * safe_total ** (alpha - 1.0)
        raise ValueError(f"Unsupported signal_transform: {model.signal_transform}")

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
            signal, signal_grad = signal_value_grad(float(x[domain_idx]))
            value -= coef * signal
            grad_x[domain_idx] -= coef * signal_grad

        for local_idx, ((hi, lo), topic) in enumerate(zip(pair_map, pair_topics, strict=True)):
            del topic
            coef = float(pair_coef[local_idx])
            total = x[hi] + beta * x[lo]
            signal, signal_grad = signal_value_grad(float(total))
            value -= coef * signal
            common = coef * signal_grad
            grad_x[hi] -= common
            grad_x[lo] -= common * beta

        for family_name in model.family_totals:
            coef = float(family_coef[family_name])
            members = family_indices[family_name]
            total = float(np.sum(x[members]))
            signal, signal_grad = signal_value_grad(total)
            value -= coef * signal
            grad_x[members] -= coef * signal_grad

        if penalty_coef != 0.0:
            for domain_idx in singleton_indices:
                u = np.log1p(x[domain_idx]) - tau
                sp = float(softplus(np.asarray(u)))
                value += penalty_coef * sp**2
                sigmoid_u = float(1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0))))
                grad_x[domain_idx] += penalty_coef * (2.0 * sp * sigmoid_u / (1.0 + x[domain_idx]))

            for hi, lo in pair_map:
                total = x[hi] + x[lo]
                u = np.log1p(total) - tau
                sp = float(softplus(np.asarray(u)))
                value += penalty_coef * sp**2
                sigmoid_u = float(1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0))))
                common = penalty_coef * (2.0 * sp * sigmoid_u / (1.0 + total))
                grad_x[hi] += common
                grad_x[lo] += common

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
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Generic-family optimization failed")

    logits0 = best_result.x[:n_domains]
    logits1 = best_result.x[n_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 /= np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 /= np.sum(p1)
    return best_result, p0, p1
