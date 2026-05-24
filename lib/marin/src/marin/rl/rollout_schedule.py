# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic finite-dataset rollout schedules."""

from dataclasses import dataclass

import jax.random as jrandom
import numpy as np
from levanter.data._prp import FeistelPermutation


@dataclass(frozen=True)
class RolloutAssignment:
    """Concrete finite-dataset indices assigned to one logical rollout worker."""

    assignment_id: str
    worker_index: int
    lesson_id: str
    worker_seed: int
    dataset_len: int
    epoch: int
    start_position: int
    end_position: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class RolloutScheduleCursor:
    """Committed schedule position for one logical rollout worker and lesson."""

    worker_index: int
    lesson_id: str
    position: int = 0


class FeistelEpochSchedule:
    """Stateless per-worker schedule over a finite dataset.

    Each epoch is a deterministic Feistel permutation of ``[0, dataset_len)``.
    Epoch keys are derived by folding the epoch into the worker seed, so each
    logical rollout worker can traverse the full dataset with its own order.
    """

    def __init__(self, dataset_len: int, seed: int):
        if dataset_len <= 0:
            raise ValueError(f"dataset_len must be positive, got {dataset_len}")
        self.dataset_len = dataset_len
        self.seed = seed
        self._permutations: dict[int, FeistelPermutation] = {}

    def indices_for_positions(self, start_position: int, count: int) -> tuple[int, ...]:
        """Return dataset indices for a contiguous worker-local position range."""
        if start_position < 0:
            raise ValueError(f"start_position must be non-negative, got {start_position}")
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        positions = np.arange(start_position, start_position + count, dtype=np.int64)
        epochs = positions // self.dataset_len
        offsets = positions % self.dataset_len

        indices: list[int] = []
        for epoch in np.unique(epochs):
            mask = epochs == epoch
            permutation = self._permutation_for_epoch(int(epoch))
            indices.extend(int(idx) for idx in permutation(offsets[mask]))

        return tuple(indices)

    def _permutation_for_epoch(self, epoch: int) -> FeistelPermutation:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        if epoch not in self._permutations:
            key = jrandom.fold_in(jrandom.PRNGKey(self.seed & 0xFFFFFFFF), epoch)
            self._permutations[epoch] = FeistelPermutation(self.dataset_len, key)
        return self._permutations[epoch]


def rollout_assignment(
    *,
    worker_index: int,
    lesson_id: str,
    worker_seed: int,
    dataset_len: int,
    start_position: int,
    n_examples: int,
) -> RolloutAssignment:
    """Build a deterministic rollout assignment for one logical worker."""
    schedule = FeistelEpochSchedule(dataset_len=dataset_len, seed=worker_seed)
    indices = schedule.indices_for_positions(start_position=start_position, count=n_examples)
    epoch = start_position // dataset_len
    end_position = start_position + n_examples
    assignment_id = f"worker-{worker_index}:lesson-{lesson_id}:start-{start_position}:count-{n_examples}"
    return RolloutAssignment(
        assignment_id=assignment_id,
        worker_index=worker_index,
        lesson_id=lesson_id,
        worker_seed=worker_seed,
        dataset_len=dataset_len,
        epoch=epoch,
        start_position=start_position,
        end_position=end_position,
        indices=indices,
    )
