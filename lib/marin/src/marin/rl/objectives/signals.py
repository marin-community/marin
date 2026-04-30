# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Signal builders that turn neutral batch metadata into training signals."""

from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import haliax as hax
import haliax.haxtyping as ht
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray
from marin.rl.types import BatchInfo, SequenceBatch

from .spec import NoRewardSignalConfig, RLOOSignalConfig, SignalConfig


def compute_rloo_advantages_from_rewards(rewards: np.ndarray) -> np.ndarray:
    """Compute reward leave-one-out advantages from a reward vector."""
    rewards = np.asarray(rewards, dtype=np.float32)
    if len(rewards) <= 1:
        return np.zeros_like(rewards)

    total = rewards.sum()
    leave_one_out_baselines = (total - rewards) / (len(rewards) - 1)
    return rewards - leave_one_out_baselines


def _group_indices(group_ids: tuple[str, ...]) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        indices.setdefault(group_id, []).append(idx)
    return indices


class PreparedSignals(eqx.Module):
    """Device-ready signals derived from objective-neutral batch metadata."""

    sequence_advantages: ht.Float[NamedArray, "batch"]  # noqa: F821
    token_weights: ht.Float[NamedArray, "batch position"]
    loss_mask: ht.Float[NamedArray, "batch position"]
    group_size: ht.Int[NamedArray, "batch"]  # noqa: F821


class SignalBuilder(Protocol):
    """Build device-ready objective signals from neutral batch metadata."""

    def build(self, batch: SequenceBatch, info: BatchInfo) -> PreparedSignals:
        """Build prepared signals for one sequence batch."""
        ...


@dataclass(frozen=True)
class RLOOSignalBuilder:
    """Build per-sequence RLOO advantages and token weights."""

    config: RLOOSignalConfig

    def build(self, batch: SequenceBatch, info: BatchInfo) -> PreparedSignals:
        rewards = np.asarray(info.episode_reward.array, dtype=np.float32)
        group_indices = _group_indices(info.group_id)
        advantages = np.zeros(len(info), dtype=np.float32)
        group_sizes = np.zeros(len(info), dtype=np.int32)

        for indices in group_indices.values():
            index_array = np.asarray(indices, dtype=np.int32)
            advantages[index_array] = compute_rloo_advantages_from_rewards(rewards[index_array])
            group_sizes[index_array] = len(indices)

        sequence_advantages = jnp.asarray(advantages, dtype=jnp.float32)
        token_weights = batch.response_mask.array * sequence_advantages[:, None]
        return PreparedSignals(
            sequence_advantages=hax.named(sequence_advantages, ["batch"]),
            token_weights=hax.named(token_weights, ["batch", "position"]),
            loss_mask=batch.response_mask,
            group_size=hax.named(jnp.asarray(group_sizes, dtype=jnp.int32), ["batch"]),
        )


@dataclass(frozen=True)
class NoRewardSignalBuilder:
    """Build zero-valued sequence signals for reward-free objectives."""

    config: NoRewardSignalConfig

    def build(self, batch: SequenceBatch, info: BatchInfo) -> PreparedSignals:
        group_indices = _group_indices(info.group_id)
        group_sizes = np.zeros(len(info), dtype=np.int32)
        for indices in group_indices.values():
            index_array = np.asarray(indices, dtype=np.int32)
            group_sizes[index_array] = len(indices)

        sequence_advantages = jnp.zeros((len(info),), dtype=jnp.float32)
        token_weights = jnp.zeros_like(batch.response_mask.array)
        return PreparedSignals(
            sequence_advantages=hax.named(sequence_advantages, ["batch"]),
            token_weights=hax.named(token_weights, ["batch", "position"]),
            loss_mask=batch.response_mask,
            group_size=hax.named(jnp.asarray(group_sizes, dtype=jnp.int32), ["batch"]),
        )


def build_signal_builder(config: SignalConfig) -> SignalBuilder:
    """Instantiate a concrete signal builder from the public config."""
    if isinstance(config, RLOOSignalConfig):
        return RLOOSignalBuilder(config)
    if isinstance(config, NoRewardSignalConfig):
        return NoRewardSignalBuilder(config)

    raise TypeError(f"Unsupported signal config: {type(config)!r}")
