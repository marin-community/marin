from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import numpy as np
import ray

from .datatypes import GroupKey, RolloutGroup, RolloutRecord


@dataclass(frozen=True)
class SelectedBatch:
    """A minimal packed batch for training.

    For now, we return the selected rollouts and their token-level advantages.
    Token lengths are approximated from text (word count) in the absence of a tokenizer.
    """

    rollouts: List[RolloutRecord]
    advantages: List[np.ndarray]
    total_tokens: int


class ReplayBuffer:
    """Unified replay buffer that aggregates groups across all envs.

    Testable as a plain class. A `ReplayBufferActor` wrapper is provided below for Ray.
    """

    def __init__(self, *, prng_key: jax.Array, min_group_size: int = 2) -> None:
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least 2")
        self.prng_key = prng_key
        self.min_group_size = min_group_size
        # Map GroupKey -> list of rollouts
        self.rollout_groups: Dict[GroupKey, List[RolloutRecord]] = {}

    # -----------------------------
    # Ingestion
    # -----------------------------
    def extend_rollouts(self, rollouts: List[RolloutRecord]) -> None:
        for r in rollouts:
            key = GroupKey(r.environment, r.example_id)
            self.rollout_groups.setdefault(key, []).append(r)

    def extend_groups(self, groups: List[RolloutGroup]) -> None:
        for g in groups:
            key = GroupKey(g.environment, g.example_id)
            for r in g.rollouts:
                self.rollout_groups.setdefault(key, []).append(r)

    def purge(self) -> None:
        self.rollout_groups.clear()

    # -----------------------------
    # Sampling & packing
    # -----------------------------
    def sample_batch(
        self,
        *,
        total_token_budget: int,
        max_seq_len: int,
        step: int,
    ) -> SelectedBatch:
        """Select a batch of rollouts under a token budget across all envs.

        Strategy (v0):
        - Consider only groups with at least ``min_group_size`` rollouts.
        - Compute RLOO advantages per group.
        - Flatten rollouts with non-zero advantages and permute deterministically by step.
        - Accumulate rollouts until the token budget is exhausted (approximate tokens by word count).
        - Return the selected rollouts and their per-token advantage arrays (length clipped to max_seq_len).
        """
        usable: List[Tuple[RolloutRecord, np.ndarray, int]] = []  # (rollout, adv, approx_len)
        for group in self.rollout_groups.values():
            if len(group) < self.min_group_size:
                continue
            advs = _compute_advantages(group)
            for r, adv in zip(group, advs, strict=False):
                if not np.allclose(adv, 0.0):
                    approx_len = max(1, _approx_generated_tokens(r))
                    usable.append((r, adv, approx_len))

        if not usable:
            return SelectedBatch(rollouts=[], advantages=[], total_tokens=0)

        # Deterministic permutation by step
        key = jax.random.fold_in(self.prng_key, int(step))
        perm = list(map(int, np.array(jax.random.permutation(key, len(usable)))))
        usable = [usable[i] for i in perm]

        budget = int(total_token_budget)
        selected_rollouts: List[RolloutRecord] = []
        selected_advs: List[np.ndarray] = []
        consumed = 0
        for r, adv, approx_len in usable:
            if consumed + min(approx_len, max_seq_len) > budget:
                break
            # Clip or pad advantage to max_seq_len
            if len(adv) > max_seq_len:
                adv_arr = adv[:max_seq_len]
            else:
                if len(adv) == 0:
                    adv_arr = np.zeros((max_seq_len,), dtype=np.float32)
                else:
                    pad = max_seq_len - len(adv)
                    adv_arr = np.pad(adv, (0, pad), mode="constant")
            selected_rollouts.append(r)
            selected_advs.append(adv_arr)
            consumed += min(approx_len, max_seq_len)

        return SelectedBatch(rollouts=selected_rollouts, advantages=selected_advs, total_tokens=consumed)


@ray.remote
class ReplayBufferActor:
    """Ray actor wrapper over ``ReplayBuffer`` to keep core logic testable.

    Methods mirror the plain class but are Ray-callable.
    """

    def __init__(self, *, seed: int = 0, min_group_size: int = 2):
        key = jax.random.PRNGKey(seed)
        self._inner = ReplayBuffer(prng_key=key, min_group_size=min_group_size)

    def extend_groups(self, groups: List[RolloutGroup]) -> None:
        self._inner.extend_groups(groups)

    def extend_rollouts(self, rollouts: List[RolloutRecord]) -> None:
        self._inner.extend_rollouts(rollouts)

    def sample_batch(self, *, total_token_budget: int, max_seq_len: int, step: int) -> SelectedBatch:
        return self._inner.sample_batch(total_token_budget=total_token_budget, max_seq_len=max_seq_len, step=step)


def _approx_generated_tokens(rollout: RolloutRecord) -> int:
    """Approximate generated token count from assistant turns by word count."""
    count = 0
    for t in rollout.turns:
        if t.role == "assistant" and t.message:
            count += max(1, len(t.message.split()))
    return max(1, count)


def _compute_advantages(rollouts: List[RolloutRecord]) -> List[np.ndarray]:
    """Compute RLOO advantages for a group of rollouts.

    Returns per-token advantages sized to the approximate generated length.
    """
    rewards = []
    for r in rollouts:
        rewards.append(sum((tr.reward or 0.0) for tr in r.turns))

    advantages: List[np.ndarray] = []
    for i, r in enumerate(rollouts):
        others = np.array(rewards[:i] + rewards[i + 1 :], dtype=np.float32)
        if others.size > 0:
            adv = float(rewards[i] - float(np.mean(others)))
            std = float(np.std(others))
            if std > 1e-8:
                adv /= std
        else:
            adv = 0.0

        length = _approx_generated_tokens(r)
        advantages.append(np.full((length,), adv, dtype=np.float32))

    return advantages
