# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic family retained-total surrogate for many-domain Uncheatable BPB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


class GenericFamilyRetainedTotalSurrogate:
    """Generic family retained-total surrogate with paired CC buckets."""

    def __init__(
        self,
        packet: GenericFamilyPacket,
        *,
        params: dict[str, float] | None = None,
        family_totals: tuple[str, ...] = GENERIC_FAMILY_NAMES,
        quality_discount: bool = True,
    ):
        self.packet = packet
        self.params = dict(TUNED_GENERIC_FAMILY_PARAMS if params is None else params)
        self.family_totals = tuple(family_totals)
        self.quality_discount = bool(quality_discount)
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

        for idx in self.packet.singletons:
            features.append(np.log1p(alpha * x[:, idx : idx + 1]))
            group_totals.append(x[:, idx])

        for hi, lo in self.packet.pairs:
            pair_signal_total = x[:, hi] + (beta * x[:, lo] if self.quality_discount else x[:, lo])
            features.append(np.log1p(alpha * pair_signal_total)[:, None])
            group_totals.append(x[:, hi] + x[:, lo])

        for family_name in self.family_totals:
            family_indices = self.packet.family_map[family_name]
            family_total = np.sum(x[:, family_indices], axis=1)
            features.append(np.log1p(alpha * family_total)[:, None])

        penalty_inputs = np.stack(group_totals, axis=1)
        penalty = np.sum(softplus(np.log1p(penalty_inputs) - tau) ** 2, axis=1, keepdims=True)
        features.append(penalty)

        design = np.hstack(features)
        num_signal = design.shape[1] - 1
        design[:, :num_signal] *= -1.0
        return design

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
