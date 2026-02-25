# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import jax

import levanter.tracker
from levanter.callbacks._core import StepInfo
from levanter.utils.jax_utils import barrier_sync

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for scheduling the training profiler callback."""

    enabled: bool = False
    start_step: int = 5
    num_steps: int = 25
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


def profile(path: str, start_step: int, num_steps: int, create_perfetto_link: bool) -> Callable[[StepInfo], None]:
    def profiler_callback_fn(step: StepInfo):
        # -1 b/c step is the finished step
        if step.step == start_step - 1:
            _create_perfetto_link = create_perfetto_link and jax.process_index() == 0
            logger.info(f"Starting profiler until step {start_step + num_steps}.")
            jax.profiler.start_trace(path, create_perfetto_link=_create_perfetto_link, create_perfetto_trace=True)
        elif step.step == start_step + num_steps - 1:
            if create_perfetto_link:
                logger.info(
                    f"Stopping profiler. Process 0 will open a perfetto link. I am process {jax.process_index()}"
                )
            else:
                logger.info("Stopping profiler.")
            # so, annoyingly, gcloud ssh doesn't reliably flush stdout here, so we need to spin up
            # a thread to flush and print periodically until we make it past stop_trace
            # (note: stop_trace blocks if perfetto is enabled)
            event = threading.Event()
            if create_perfetto_link and jax.process_index() == 0:
                _flush_while_waiting(event)

            jax.profiler.stop_trace()

            if create_perfetto_link and jax.process_index() == 0:
                event.set()

            levanter.tracker.current_tracker().log_artifact(path, type="jax_profile")
            barrier_sync()

    return profiler_callback_fn


def _flush_while_waiting(event):
    def flush_stdout():
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(5)
        while not event.is_set():
            print("Waiting...", flush=True)
            print("\n", file=sys.stderr, flush=True)
            time.sleep(5)

    thread = threading.Thread(target=flush_stdout)
    thread.start()
