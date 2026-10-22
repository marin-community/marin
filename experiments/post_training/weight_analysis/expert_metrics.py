# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-expert concentration, additivity and the seed-to-seed alignment ceiling.

Definitions follow DESIGN.md, "Weight space":

- per-expert ‖Δ_e‖ with its Gini coefficient, per layer and pooled over every
  (layer, expert);
- additivity: the cosine and norm ratio between ``Δ_AB`` and ``Δ_A + Δ_B``;
- every cross-method cosine divided by the seed-to-seed cosine of the same cell
  (control C5). The ceiling is the mean of the seed-pair cosines (3 pairs for a
  3-seed Phase 1 cell), per component and per (layer, expert).

Task conflict (cosine and sign agreement between ``Δ_A`` and ``Δ_B``) is
:func:`~experiments.post_training.weight_analysis.delta_stats.compare_deltas`.
"""

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from experiments.post_training.weight_analysis.components import Block, Keyed
from experiments.post_training.weight_analysis.delta_stats import DeltaSummary, PairGeometry, aligned


def gini(values: Sequence[float]) -> float:
    """0 when every value is equal, approaching 1 when one value holds everything."""
    if len(values) < 2:
        raise ValueError(f"the Gini coefficient needs at least two values; got {len(values)}")
    total = math.fsum(values)
    if total == 0.0:
        raise ValueError("every value is zero; the Gini coefficient is undefined")
    ordered = sorted(values)
    count = len(ordered)
    weighted = math.fsum((2 * rank - count - 1) * value for rank, value in enumerate(ordered, start=1))
    return weighted / (count * total)


@dataclass(frozen=True)
class ExpertGini:
    per_layer: dict[int, float]
    pooled: float
    """Over every (layer, expert) norm at once."""


def expert_gini(summary: Keyed[DeltaSummary]) -> ExpertGini:
    """Gini of per-expert ‖Δ_e‖ from :func:`~delta_stats.summarize_deltas`."""
    norms = {key: value.delta_norm for key, value in summary.experts.items()}
    if not norms:
        raise ValueError("the summary has no routed experts")
    layers = sorted({layer for layer, _expert in norms})
    per_layer = {layer: gini([norm for (key_layer, _), norm in norms.items() if key_layer == layer]) for layer in layers}
    return ExpertGini(per_layer=per_layer, pooled=gini(list(norms.values())))


def additivity(joint: Iterable[Block], delta_a: Iterable[Block], delta_b: Iterable[Block]) -> Keyed[PairGeometry]:
    """``Δ_AB`` against ``Δ_A + Δ_B`` per key.

    Read ``.cosine`` and ``.norm_ratio`` (‖Δ_AB‖/‖Δ_A + Δ_B‖) on each value.
    """
    geometry = Keyed(overall=PairGeometry())
    for joint_block, block_a, block_b in aligned(joint, delta_a, delta_b):
        summed = block_a.array + block_b.array
        block_geometry = PairGeometry()
        block_geometry.update(joint_block.array, summed)
        for target in geometry.targets(joint_block, PairGeometry):
            target.merge(block_geometry)
    return geometry


@dataclass(frozen=True)
class SeedCeiling:
    """The seed-to-seed delta cosines of one key and their mean.

    ``pair_cosines`` holds ``None`` for a pair where a seed did not move this
    key. ``near_zero`` marks a ceiling that cannot normalize anything: undefined,
    or with |mean| below the threshold :func:`seed_ceiling` was given.
    """

    pair_cosines: tuple[float | None, ...]
    near_zero: bool

    @property
    def defined(self) -> bool:
        return all(value is not None for value in self.pair_cosines)

    @property
    def cosine(self) -> float:
        if not self.defined:
            raise ValueError(f"a seed pair has no cosine for this key: {self.pair_cosines}")
        return math.fsum(value for value in self.pair_cosines if value is not None) / len(self.pair_cosines)


def seed_ceiling(seed_deltas: Sequence[Iterable[Block]], *, near_zero: float) -> Keyed[SeedCeiling]:
    """Mean pairwise cosine of a cell's seed deltas, per key, in one pass.

    Keys whose ceiling is undefined or has |mean| < ``near_zero`` are flagged, not
    raised: the cell reports them and moves on.
    """
    if len(seed_deltas) < 2:
        raise ValueError(f"a seed ceiling needs at least two seed deltas; got {len(seed_deltas)}")
    pairs = list(itertools.combinations(range(len(seed_deltas)), 2))
    geometries: Keyed[list[PairGeometry]] = Keyed(overall=[PairGeometry() for _ in pairs])
    for row in aligned(*seed_deltas):
        block_geometries = []
        for first, second in pairs:
            pair_geometry = PairGeometry()
            pair_geometry.update(row[first].array, row[second].array)
            block_geometries.append(pair_geometry)
        for target in geometries.targets(row[0], lambda: [PairGeometry() for _ in pairs]):
            for accumulated, block_geometry in zip(target, block_geometries, strict=True):
                accumulated.merge(block_geometry)

    def ceiling(pair_geometries: list[PairGeometry]) -> SeedCeiling:
        cosines = tuple(geometry.cosine if geometry.defined else None for geometry in pair_geometries)
        defined = all(value is not None for value in cosines)
        mean = math.fsum(value for value in cosines if value is not None) / len(cosines)
        return SeedCeiling(pair_cosines=cosines, near_zero=not defined or abs(mean) < near_zero)

    return geometries.map(ceiling)


@dataclass(frozen=True)
class CeilingNormalized:
    """A cross-method cosine beside its seed ceiling and their ratio."""

    cosine: float
    seed_cosine: float | None
    ratio: float | None
    """``None`` when the ceiling is undefined or exactly zero."""
    near_zero_ceiling: bool


def normalize_by_seed_ceiling(cosines: Keyed[float], ceilings: Keyed[SeedCeiling]) -> Keyed[CeilingNormalized]:
    """Divide each cosine by the seed ceiling of the same key; keys must match."""
    if set(cosines.components) != set(ceilings.components) or set(cosines.experts) != set(ceilings.experts):
        raise ValueError("cosines and seed ceilings cover different keys")

    def normalized(cosine: float, ceiling: SeedCeiling) -> CeilingNormalized:
        if not ceiling.defined:
            return CeilingNormalized(cosine, None, None, near_zero_ceiling=True)
        seed_cosine = ceiling.cosine
        ratio = cosine / seed_cosine if seed_cosine != 0.0 else None
        return CeilingNormalized(cosine, seed_cosine, ratio, ceiling.near_zero)

    return Keyed(
        overall=normalized(cosines.overall, ceilings.overall),
        components={key: normalized(value, ceilings.components[key]) for key, value in cosines.components.items()},
        experts={key: normalized(value, ceilings.experts[key]) for key, value in cosines.experts.items()},
    )
