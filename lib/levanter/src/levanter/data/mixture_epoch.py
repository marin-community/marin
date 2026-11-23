# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Capacity-derived mixture epochs with complete per-source coverage."""

import math
from dataclasses import dataclass
from typing import Mapping

from levanter.data.mixture import allocate_mixture_counts

MIXTURE_TARGET_BLOCK_SIZE = 64_000


@dataclass(frozen=True)
class MixtureEpochPlan:
    batch_size: int
    block_size: int
    num_blocks: int
    sft_sequences: int
    run_sequences: int

    @property
    def steps(self) -> int:
        return self.run_sequences // self.batch_size

    @property
    def replay_fraction(self) -> float:
        return 1.0 - self.sft_sequences / self.run_sequences


def plan_mixture_epoch(
    sequence_counts: Mapping[str, int],
    *,
    batch_size: int,
    replay_fraction: float,
    target_block_size: int = MIXTURE_TARGET_BLOCK_SIZE,
) -> MixtureEpochPlan:
    """Keep every SFT sequence and match replay share to within one training batch."""
    if not sequence_counts or any(count <= 0 for count in sequence_counts.values()):
        raise ValueError("Every epoch source must contain at least one packed sequence")
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError(f"replay_fraction must be in (0, 1), got {replay_fraction}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    block_size = batch_size * math.ceil(target_block_size / batch_size)
    if block_size >= 2**16:
        raise ValueError("Mixture blocks must fit the native 16-bit source offset")
    sft_sequences = sum(sequence_counts.values())
    steps = math.ceil(sft_sequences / ((1.0 - replay_fraction) * batch_size))
    run_sequences = steps * batch_size
    num_blocks = math.ceil(run_sequences / block_size)
    return MixtureEpochPlan(batch_size, block_size, num_blocks, sft_sequences, run_sequences)


def epoch_weight_stages(
    sequence_counts: Mapping[str, int],
    replay_weights: Mapping[str, float],
    plan: MixtureEpochPlan,
) -> list[tuple[int, dict[str, float]]]:
    """Distribute exact SFT quotas across blocks, filling remaining slots with replay.

    A small SFT source can receive zero slots in individual blocks, but its total
    is exact. Every positive-weight replay source receives at least one slot in the epoch.
    """
    if set(sequence_counts) & set(replay_weights):
        raise ValueError("SFT and replay source names must be disjoint")
    if sum(sequence_counts.values()) != plan.sft_sequences:
        raise ValueError("Epoch plan and source capacities disagree")
    stages = []
    remaining = allocate_mixture_counts(replay_weights, plan.run_sequences - plan.sft_sequences)
    remaining_sft = dict(sequence_counts)
    for block in range(plan.num_blocks):
        start = block * plan.block_size
        stop = min(start + plan.block_size, plan.run_sequences)
        sft_draws = stop * plan.sft_sequences // plan.run_sequences - start * plan.sft_sequences // plan.run_sequences
        counts = allocate_mixture_counts(remaining_sft, sft_draws, minimum_count=0)
        remaining_sft = {name: value - counts[name] for name, value in remaining_sft.items()}
        replay = allocate_mixture_counts(remaining, stop - start - sft_draws, minimum_count=0)
        counts.update(replay)
        remaining = {name: value - replay[name] for name, value in remaining.items()}
        stages.append(
            (
                block * plan.block_size // plan.batch_size,
                {name: count / (stop - start) for name, count in counts.items()},
            )
        )
    if any(remaining.values()) or any(remaining_sft.values()):
        raise ValueError("Epoch allocation left source quotas unconsumed")
    return stages
