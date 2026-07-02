# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import posixpath
import sys
import threading
import time
import cProfile
import pstats
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jax
from rigging.filesystem import url_to_fs

from levanter.callbacks._core import StepInfo
from levanter.utils.fsspec_utils import join_path, mkdirs
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
    stop_barrier_timeout: float = 200
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


def _process_profile_path(path: str) -> str:
    return os.path.join(path, f"process_{jax.process_index():05d}")


def mirror_process_profile_dir(
    profile_dir: str | Path,
    remote_profile_dir: str | None,
    *,
    process_index: int | None = None,
) -> str | None:
    """Copy one process's local profiler output into a remote profile directory."""
    if remote_profile_dir is None:
        return None
    if str(profile_dir) == remote_profile_dir:
        return remote_profile_dir

    source_profile_dir = Path(profile_dir)
    if not source_profile_dir.exists():
        logger.warning("Profiler directory does not exist; skipping mirror: %s", source_profile_dir)
        return None

    process_index = jax.process_index() if process_index is None else process_index
    process_dir = source_profile_dir / f"process_{process_index:05d}"
    if not process_dir.exists():
        logger.warning("Profiler process directory does not exist; skipping mirror: %s", process_dir)
        return None

    fs, target_profile_path = url_to_fs(remote_profile_dir)
    fs.makedirs(target_profile_path, exist_ok=True)

    for source_path in process_dir.rglob("*"):
        relative_path = source_path.relative_to(source_profile_dir).as_posix()
        destination_path = posixpath.join(target_profile_path.rstrip("/"), relative_path)
        if source_path.is_dir():
            fs.makedirs(destination_path, exist_ok=True)
            continue
        fs.makedirs(posixpath.dirname(destination_path), exist_ok=True)
        fs.put_file(str(source_path), destination_path)

    logger.info("Mirrored profiler process directory to %s", join_path(remote_profile_dir, process_dir.name))
    return remote_profile_dir


@contextmanager
def profile_ctx(
    path: str,
    create_perfetto_link: bool = False,
    *,
    device_profile: bool = True,
    host_profile: bool = False,
    host_profile_basename: str = "host_profile",
    host_profile_topn: int = 0,
    profiler_options: jax.profiler.ProfileOptions | None = None,
    stop_barrier_timeout: float = 200,
    remote_profile_dir: str | None = None,
):
    """Profile a block and optionally mirror this process's trace to a remote run directory."""
    create_process_perfetto_link = create_perfetto_link and jax.process_index() == 0
    process_path = _process_profile_path(path)
    mkdirs(process_path)
    logger.info("Starting profiler. Trace path: %s", process_path)

    if device_profile:
        jax.profiler.start_trace(
            process_path,
            create_perfetto_link=create_process_perfetto_link,
            create_perfetto_trace=create_process_perfetto_link,
            profiler_options=profiler_options,
        )

    pr = None
    stats_path = None
    txt_summary_path = None
    if host_profile:
        try:
            pr = cProfile.Profile()
            pr.enable()
            stats_path = os.path.join(process_path, f"{host_profile_basename}.pstats")
            txt_summary_path = os.path.join(process_path, f"{host_profile_basename}.txt")
        except Exception as e:  # pragma: no cover - optional/diagnostic path
            logger.warning("Failed to start cProfile host profiler: %s", e)

    try:
        yield
    finally:
        if pr is not None and stats_path is not None:
            try:
                pr.disable()
                pr.dump_stats(stats_path)
                if host_profile_topn and txt_summary_path is not None:
                    stats = pstats.Stats(stats_path)
                    stats.strip_dirs().sort_stats("cumtime")
                    with open(txt_summary_path, "w") as f:
                        stats.stream = f  # type: ignore[assignment]
                        stats.print_stats(host_profile_topn)
            except Exception:  # pragma: no cover - optional/diagnostic path
                logger.warning("Failed to log host profile stats", exc_info=True)

        event = None
        if create_perfetto_link and jax.process_index() == 0:
            event = threading.Event()
            _flush_while_waiting(event)

        if create_perfetto_link:
            logger.info("Stopping profiler. Process 0 will open a perfetto link. I am process %s", jax.process_index())
        else:
            logger.info("Stopping profiler.")

        if device_profile:
            jax.profiler.stop_trace()

        if event is not None:
            event.set()

        barrier_sync(timeout=stop_barrier_timeout)
        mirror_process_profile_dir(path, remote_profile_dir)


def profile(
    path: str,
    start_step: int,
    num_steps: int,
    create_perfetto_link: bool,
    profiler_options: jax.profiler.ProfileOptions | None = None,
    stop_barrier_timeout: float = 200,
    remote_profile_dir: str | None = None,
) -> Callable[[StepInfo], None]:
    trace_started = False
    active_profile: AbstractContextManager[None] | None = None
    mkdirs(_process_profile_path(path))

    def profiler_callback_fn(step: StepInfo, *, force: bool = False):
        nonlocal active_profile, trace_started
        if force and trace_started:
            _stop_profile()
            return

        # -1 b/c step is the finished step
        if step.step == start_step - 1:
            if force or trace_started:
                return
            logger.info("Starting profiler until step %s.", start_step + num_steps)
            active_profile = profile_ctx(
                path,
                create_perfetto_link,
                profiler_options=profiler_options,
                stop_barrier_timeout=stop_barrier_timeout,
                remote_profile_dir=remote_profile_dir,
            )
            active_profile.__enter__()
            trace_started = True
        elif step.step == start_step + num_steps - 1:
            _stop_profile()

    def _stop_profile():
        nonlocal active_profile, trace_started
        if not trace_started:
            return
        try:
            if active_profile is not None:
                active_profile.__exit__(None, None, None)
        finally:
            active_profile = None
            trace_started = False

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
