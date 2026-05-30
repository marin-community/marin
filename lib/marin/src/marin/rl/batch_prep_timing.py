# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared batch-preparation timing dataclass for RL data loaders.

Lives in its own module so both `train_worker` and `noise_rollout_loader`
can import the type without introducing a circular dependency.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPrepTiming:
    """Timing breakdown for preparing one trainer batch."""

    fetch_time: float = 0.0
    batch_time: float = 0.0
    shard_time: float = 0.0

    @property
    def total_time(self) -> float:
        return self.fetch_time + self.batch_time + self.shard_time
