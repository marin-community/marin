# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static expert placements for controlled MoK throughput experiments."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Literal

MokExpertPlacement = Literal["contiguous", "r9_profile_hot_cold"]

_R9_NUM_LAYERS = 48
_R9_NUM_EXPERTS = 128

# One byte per new physical expert slot. Each value is the old logical expert
# label assigned to that slot. The source counts are the final routing
# histograms from the r9 fresh-parity MoK run. Within each layer, experts are
# stable-sorted hottest to coldest, then the hottest half and reversed coldest
# half are interleaved so every contiguous pair maps to one EP rank.
_R9_HOT_COLD_EXPERT_PERMUTATIONS_B64 = (
    "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"
)


def hot_cold_expert_permutation(loads: Sequence[int | float]) -> tuple[int, ...]:
    """Pair the hottest expert with the coldest, second hottest with second coldest, and so on."""

    if not loads or len(loads) % 2:
        raise ValueError("hot-cold placement requires a positive, even expert count")
    descending = sorted(range(len(loads)), key=lambda expert: (-loads[expert], expert))
    half = len(descending) // 2
    permutation: list[int] = []
    for hot, cold in zip(descending[:half], reversed(descending[half:]), strict=True):
        permutation.extend((hot, cold))
    return tuple(permutation)


def _decode_r9_hot_cold_permutations() -> tuple[tuple[int, ...], ...]:
    encoded = "".join(_R9_HOT_COLD_EXPERT_PERMUTATIONS_B64)
    packed = base64.b64decode(encoded, validate=True)
    expected_bytes = _R9_NUM_LAYERS * _R9_NUM_EXPERTS
    if len(packed) != expected_bytes:
        raise AssertionError(f"invalid r9 expert-placement table: {len(packed)} bytes, expected {expected_bytes}")
    rows = tuple(
        tuple(packed[offset : offset + _R9_NUM_EXPERTS]) for offset in range(0, expected_bytes, _R9_NUM_EXPERTS)
    )
    expected_labels = list(range(_R9_NUM_EXPERTS))
    if any(sorted(row) != expected_labels for row in rows):
        raise AssertionError("every r9 expert-placement row must be a permutation")
    return rows


R9_HOT_COLD_EXPERT_PERMUTATIONS = _decode_r9_hot_cold_permutations()


def resolve_mok_expert_permutations(
    placement: MokExpertPlacement,
    *,
    num_layers: int,
    num_experts: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Resolve a placement to new-slot -> old-label permutations."""

    if placement == "contiguous":
        return None
    if placement != "r9_profile_hot_cold":
        raise ValueError(f"unknown MoK expert placement: {placement}")
    if (num_layers, num_experts) != (_R9_NUM_LAYERS, _R9_NUM_EXPERTS):
        raise ValueError(
            "r9_profile_hot_cold is only defined for "
            f"{_R9_NUM_LAYERS} layers and {_R9_NUM_EXPERTS} experts; "
            f"got {num_layers} layers and {num_experts} experts"
        )
    return R9_HOT_COLD_EXPERT_PERMUTATIONS
