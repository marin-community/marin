# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import jax

from rigging.filesystem import StoragePath

from levanter.callbacks._core import StepInfo
from levanter.utils.jax_utils import barrier_sync

logger = logging.getLogger(__name__)

AdvancedProfileOptionValue = bool | int | str


@dataclass(frozen=True)
class ProfileOptionsConfig:
    """Configuration forwarded to ``jax.profiler.ProfileOptions``."""

    host_tracer_level: int | None = None
    python_tracer_level: int | None = None
    device_tracer_level: int | None = None
    enable_hlo_proto: bool | None = None
    include_dataset_ops: bool | None = None
    advanced_configuration: dict[str, AdvancedProfileOptionValue] = field(default_factory=dict)

    @property
    def is_configured(self) -> bool:
        return (
            self.host_tracer_level is not None
            or self.python_tracer_level is not None
            or self.device_tracer_level is not None
            or self.enable_hlo_proto is not None
            or self.include_dataset_ops is not None
            or bool(self.advanced_configuration)
        )

    def build_jax_profile_options(self) -> jax.profiler.ProfileOptions | None:
        if not self.is_configured:
            return None

        options = jax.profiler.ProfileOptions()
        if self.host_tracer_level is not None:
            options.host_tracer_level = self.host_tracer_level
        if self.python_tracer_level is not None:
            options.python_tracer_level = self.python_tracer_level
        if self.enable_hlo_proto is not None:
            options.enable_hlo_proto = self.enable_hlo_proto
        if self.include_dataset_ops is not None:
            options.include_dataset_ops = self.include_dataset_ops

        advanced_configuration = dict(self.advanced_configuration)
        if self.device_tracer_level is not None:
            advanced_configuration["device_tracer_level"] = self.device_tracer_level
        if advanced_configuration:
            options.advanced_configuration = advanced_configuration
        return options


@dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for scheduling the training profiler callback."""

    enabled: bool = False
    start_step: int = 5
    num_steps: int = 25
    perfetto_link: bool = False
    profile_options: ProfileOptionsConfig = field(default_factory=ProfileOptionsConfig)

    @property
    def is_enabled(self) -> bool:
        return self.enabled and self.num_steps > 0

    def build_jax_profile_options(self) -> jax.profiler.ProfileOptions | None:
        return self.profile_options.build_jax_profile_options()

    def resolve_num_profile_steps(self, num_train_steps: int) -> int:
        """Clamp profiling duration to the configured training length."""
        total_prof_steps = self.num_steps
        if total_prof_steps + self.start_step > num_train_steps:
            logger.warning(
                f"Adjusting profiler_total_steps from {total_prof_steps} to {num_train_steps - self.start_step}"
            )
            total_prof_steps = num_train_steps - self.start_step

        return max(0, total_prof_steps)


def profile(
    path: str,
    start_step: int,
    num_steps: int,
    create_perfetto_link: bool,
    profiler_options: jax.profiler.ProfileOptions | None = None,
) -> Callable[[StepInfo], None]:
    trace_started = False
    StoragePath(path).mkdirs()

    def profiler_callback_fn(step: StepInfo, *, force: bool = False):
        nonlocal trace_started
        if force and trace_started:
            _stop_profile()
            return

        # -1 b/c step is the finished step
        if step.step == start_step - 1:
            if force or trace_started:
                return
            _create_perfetto_link = create_perfetto_link and jax.process_index() == 0
            logger.info(f"Starting profiler until step {start_step + num_steps}.")
            jax.profiler.start_trace(
                path,
                create_perfetto_link=_create_perfetto_link,
                create_perfetto_trace=True,
                profiler_options=profiler_options,
            )
            trace_started = True
        elif step.step == start_step + num_steps - 1:
            _stop_profile()

    def _stop_profile():
        nonlocal trace_started
        if not trace_started:
            return
        if create_perfetto_link:
            logger.info(f"Stopping profiler. Process 0 will open a perfetto link. I am process {jax.process_index()}")
        else:
            logger.info("Stopping profiler.")
        # so, annoyingly, gcloud ssh doesn't reliably flush stdout here, so we need to spin up
        # a thread to flush and print periodically until we make it past stop_trace
        # (note: stop_trace blocks if perfetto is enabled)
        event = threading.Event()
        if create_perfetto_link and jax.process_index() == 0:
            _flush_while_waiting(event)

        jax.profiler.stop_trace()
        trace_started = False

        if create_perfetto_link and jax.process_index() == 0:
            event.set()
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
