# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for scheduling the training profiler callback."""

    enabled: bool = False
    start_step: int = 5
    num_steps: int = 100
    perfetto_link: bool = False

    @property
    def is_enabled(self) -> bool:
        return self.enabled and self.num_steps > 0

    def resolve_num_profile_steps(self, num_train_steps: int) -> int:
        """Clamp profiling duration to the configured training length."""
        total_prof_steps = self.num_steps
        if total_prof_steps + self.start_step > num_train_steps:
            logger.warning(
                f"Adjusting profiler_total_steps from {total_prof_steps} to {num_train_steps - self.start_step}"
            )
            total_prof_steps = num_train_steps - self.start_step

        return max(0, total_prof_steps)
