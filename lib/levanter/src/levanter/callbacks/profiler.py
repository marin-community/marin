# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import jax

from rigging.filesystem import StoragePath, marin_temp_bucket

from levanter.callbacks._core import StepInfo
from levanter.utils.jax_utils import barrier_sync

logger = logging.getLogger(__name__)

AdvancedProfileOptionValue = bool | int | str
DEFAULT_XPROF_SERVICE_URL = "https://iris.oa.dev/proxy/xprof"
_XPROF_RUN_PATH = "plugins/profile"
_XPROF_TTL_SEGMENT = re.compile(r"ttl=[1-9]\d*d")


def xprof_viewer_url(service_url: str, profile_uri: str) -> str:
    """Return the hosted XProf URL for an uploaded profile root."""
    return f"{service_url.rstrip('/')}/open?{urlencode({'uri': profile_uri})}"


@dataclass(frozen=True)
class ProfileOptionsConfig:
    """Configuration forwarded to ``jax.profiler.ProfileOptions``."""

    host_tracer_level: int | None = None
    python_tracer_level: int | None = None
    device_tracer_level: int | None = None
    enable_hlo_proto: bool = False
    include_dataset_ops: bool | None = None
    advanced_configuration: dict[str, AdvancedProfileOptionValue] = field(default_factory=dict)

    def build_jax_profile_options(self) -> jax.profiler.ProfileOptions:
        options = jax.profiler.ProfileOptions()
        if self.host_tracer_level is not None:
            options.host_tracer_level = self.host_tracer_level
        if self.python_tracer_level is not None:
            options.python_tracer_level = self.python_tracer_level
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
class XprofUploadConfig:
    """XProf upload settings."""

    enabled: bool = True
    ttl_days: int = 30
    service_url: str = DEFAULT_XPROF_SERVICE_URL

    def destination_for_run(self, run_id: str) -> str:
        """Resolve the run's upload root."""
        return marin_temp_bucket(self.ttl_days, prefix=f"xprof/{run_id}")


@dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for scheduling the training profiler callback."""

    enabled: bool = False
    start_step: int = 5
    num_steps: int = 25
    perfetto_link: bool = False
    create_perfetto_trace: bool = False
    process_index: int | None = None
    profile_options: ProfileOptionsConfig = field(default_factory=ProfileOptionsConfig)
    upload: XprofUploadConfig = field(default_factory=XprofUploadConfig)

    @property
    def is_enabled(self) -> bool:
        return self.enabled and self.num_steps > 0

    def build_jax_profile_options(self) -> jax.profiler.ProfileOptions:
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

    def build(self, path: str, run_id: str, num_steps: int | None = None) -> Callable[[StepInfo], None]:
        """Build the scheduled profiler callback."""
        if num_steps is None:
            num_steps = self.num_steps
        upload_uri = self.upload.destination_for_run(run_id) if self.upload.enabled else None
        if upload_uri is not None and not _is_xprof_ttl_root(StoragePath(upload_uri)):
            logger.info("MARIN_PREFIX has no remote XProf TTL store; keeping the profile at %s", path)
            upload_uri = None
        service_url = None
        if upload_uri is not None and StoragePath(upload_uri).scheme in ("gs", "s3"):
            service_url = self.upload.service_url
        return profile(
            path,
            start_step=self.start_step,
            num_steps=num_steps,
            create_perfetto_link=self.perfetto_link,
            create_perfetto_trace=self.create_perfetto_trace,
            profiler_options=self.build_jax_profile_options(),
            process_index=self.process_index,
            upload_uri=upload_uri,
            xprof_service_url=service_url,
        )


def _is_xprof_ttl_root(path: StoragePath) -> bool:
    if path.scheme not in ("gs", "s3") or not path.bucket:
        return False
    # Leave room after the TTL segment for xprof and at least one run segment.
    return any(
        _XPROF_TTL_SEGMENT.fullmatch(segment) and path.segments[index + 1] == "xprof"
        for index, segment in enumerate(path.segments[:-2])
    )


def profile(
    path: str,
    start_step: int,
    num_steps: int,
    create_perfetto_link: bool,
    create_perfetto_trace: bool = False,
    profiler_options: jax.profiler.ProfileOptions | None = None,
    process_index: int | None = None,
    upload_uri: str | None = None,
    xprof_service_url: str | None = None,
) -> Callable[[StepInfo], None]:
    """Schedule a JAX XPlane capture."""
    profile_path = StoragePath(path)
    if not profile_path.is_local:
        raise ValueError(f"JAX profiler capture path must be local, got {path}")

    local_path = Path(path)
    local_path.mkdir(parents=True, exist_ok=True)
    profile_window_started = False
    trace_started = False
    existing_sessions: set[Path] = set()

    def is_tracing_process() -> bool:
        return process_index is None or jax.process_index() == process_index

    def profiler_callback_fn(step: StepInfo, *, force: bool = False):
        nonlocal existing_sessions, profile_window_started, trace_started
        if force and profile_window_started:
            _stop_profile(max(start_step, step.step + 1))
            return

        # -1 b/c step is the finished step
        if step.step == start_step - 1:
            if force or profile_window_started:
                return
            profile_window_started = True
            if is_tracing_process():
                existing_sessions = _profile_sessions(local_path)
                _create_perfetto_link = create_perfetto_link and jax.process_index() == 0
                logger.info(f"Starting profiler until step {start_step + num_steps}.")
                jax.profiler.start_trace(
                    path,
                    create_perfetto_link=_create_perfetto_link,
                    create_perfetto_trace=create_perfetto_trace or create_perfetto_link,
                    profiler_options=profiler_options,
                )
                trace_started = True
        elif step.step == start_step + num_steps - 1:
            _stop_profile(start_step + num_steps)

    def _stop_profile(end_step: int):
        nonlocal profile_window_started, trace_started
        if not profile_window_started:
            return

        captured_profile = trace_started
        if trace_started:
            if create_perfetto_link:
                logger.info(
                    f"Stopping profiler. Process 0 will open a perfetto link. I am process {jax.process_index()}"
                )
            else:
                logger.info("Stopping profiler.")
            # Keep gcloud SSH output alive while Perfetto link creation blocks.
            event = threading.Event()
            if create_perfetto_link and jax.process_index() == 0:
                _flush_while_waiting(event)

            jax.profiler.stop_trace()
            trace_started = False

            if create_perfetto_link and jax.process_index() == 0:
                event.set()

        barrier_sync()
        upload_error: Exception | None = None
        if captured_profile and upload_uri is not None:
            try:
                remote_session_name = f"steps-{start_step}-to-{end_step}"
                _upload_profile_sessions(local_path, existing_sessions, upload_uri, remote_session_name)
            except Exception as exc:
                upload_error = exc
                logger.exception("Failed to upload XProf profile to %s", upload_uri)

        if upload_uri is not None:
            # All processes must reach the same barrier before an upload error propagates.
            barrier_sync()
        profile_window_started = False
        if upload_error is not None:
            raise RuntimeError(f"Failed to upload XProf profile to {upload_uri}") from upload_error
        if upload_uri is not None and xprof_service_url is not None and jax.process_index() == 0:
            viewer_url = xprof_viewer_url(xprof_service_url, upload_uri)
            logger.info("XProf profile: %s", viewer_url)

    return profiler_callback_fn


def _profile_sessions(profile_path: Path) -> set[Path]:
    session_root = profile_path / _XPROF_RUN_PATH
    if not session_root.exists():
        return set()
    return {path for path in session_root.iterdir() if path.is_dir()}


def _upload_profile_sessions(
    profile_path: Path,
    existing_sessions: set[Path],
    upload_uri: str,
    remote_session_name: str,
) -> None:
    new_sessions = _profile_sessions(profile_path) - existing_sessions
    if not new_sessions:
        raise RuntimeError(f"JAX profiler produced no new XPlane session under {profile_path}")

    destination = StoragePath(upload_uri) / _XPROF_RUN_PATH / remote_session_name
    for session_path in sorted(new_sessions):
        destination.upload_from(f"{session_path}/", recursive=True)
    logger.info("Uploaded XProf session to %s", destination)


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
