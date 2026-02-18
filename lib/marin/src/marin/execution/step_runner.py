# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step runner for StepSpec.

This module provides the execution layer for ``StepSpec`` objects. The main
entry point is ``StepRunner``, which eagerly runs steps as they are yielded
from an iterable, launching each as soon as its dependencies are satisfied.

Each step is executed on a fray worker as
``disk_cached(distributed_lock(fn))``.  Caching, distributed locking,
heartbeats, artifact saving, and status writes all happen on the worker.
``StepRunner`` itself is a thin DAG scheduler that submits jobs and polls
for completion.

``ExecutorStep`` objects can be converted to ``StepSpec`` via
``resolve_executor_step``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import fsspec
import levanter.utils.fsspec_utils as fsspec_utils

from fray.v2 import client as fray_client
from fray.v2.client import JobHandle, JobStatus
from fray.v2.types import ResourceConfig

from marin.execution.executor_step_status import (
    STATUS_DEP_FAILED,
    STATUS_FAILED,
    STATUS_SUCCESS,
    StatusFile,
)
from marin.execution.step_model import StepSpec
from marin.utilities.json_encoder import CustomJsonEncoder

# Re-export for backward compatibility
from marin.execution.executor_step_status import PreviousTaskFailedError, should_run, worker_id  # noqa: F401

if TYPE_CHECKING:
    from marin.execution.executor import ExecutorStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_job_name(name: str) -> str:
    """Ensure job names are compatible with Iris and Docker image tags."""
    sanitized = re.sub(r"[^a-z0-9_.-]+", "-", name.lower())
    sanitized = sanitized.strip("-.")
    return sanitized or "job"


def _get_fn_name(fn: Any) -> str:
    """Return a human-readable name for a step function."""
    if fn is None:
        return "None"
    return str(fn)


def _write_executor_info(step: StepSpec) -> None:
    """Write a ``.executor_info`` JSON file matching the legacy ExecutorStepInfo schema.

    Skips writing if the file already exists (e.g. Executor.write_infos() wrote
    a richer version before StepRunner launched the step).
    """
    info_path = os.path.join(step.output_path, ".executor_info")
    fs = fsspec.core.url_to_fs(info_path, use_listings_cache=False)[0]
    if fs.exists(info_path):
        return
    info = {
        "executor_version": "step_runner",
        "name": step.name,
        "fn_name": _get_fn_name(step.fn),
        "config": step.hash_attrs,
        "description": None,
        "override_output_path": step.override_output_path,
        "version": {},
        "dependencies": list(step.deps),
        "output_path": step.output_path,
    }
    fsspec_utils.mkdirs(step.output_path)
    with fsspec.open(info_path, "w") as f:
        f.write(json.dumps(info, indent=2, cls=CustomJsonEncoder))


class PreviousTaskFailedError(Exception):
    """Raised when a step failed previously and force_run_failed is False."""


# ---------------------------------------------------------------------------
# Per-step job manager
# ---------------------------------------------------------------------------


class StepJobRunner:
    """Manages the fray job, heartbeat, and status for a single step."""

    def __init__(self, client: fray_client.Client, status_file: StatusFile):
        self.client = client
        self._status_file = status_file
        self._job: JobHandle | None = None
        self._heartbeat_thread: Thread | None = None
        self._stop_event = Event()

    @property
    def job_id(self) -> str | None:
        return None if self._job is None else self._job.job_id

    def launch(self, job_request: JobRequest) -> None:
        """Launch job and start heartbeat thread."""
        self._status_file.write_status(STATUS_RUNNING)
        self._job = self.client.submit(job_request, adopt_existing=True)
        self._start_heartbeat()

    def wait(self) -> None:
        """Wait for job to complete, stop heartbeat, write final status."""
        try:
            if self._job is None:
                raise RuntimeError("StepJobRunner.wait called before launch")
            result = self._job.wait(raise_on_failure=False)
            if result != JobStatus.SUCCEEDED:
                self._status_file.write_status(STATUS_FAILED)
                raise RuntimeError(f"Job {self.job_id} finished with status {result}")
            self._status_file.write_status(STATUS_SUCCESS)
        except Exception:
            if self._status_file.status != STATUS_FAILED:
                self._status_file.write_status(STATUS_FAILED)
            raise
        finally:
            self._stop_heartbeat()
            self._status_file.release_lock()

    def poll(self) -> bool:
        """Return True if job is finished."""
        if self._job is None:
            return True
        return JobStatus.finished(self._job.status())

    def _start_heartbeat(self) -> None:
        def heartbeat_loop():
            while not self._stop_event.wait(HEARTBEAT_INTERVAL):
                self._status_file.refresh_lock()

        self._heartbeat_thread = Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Lock / status helpers
# ---------------------------------------------------------------------------


def should_run(status_file: StatusFile, step_name: str, force_run_failed: bool = True) -> bool:
    """Check if the step should run based on lease-based distributed locking.

    Uses lease files for distributed locking and status file for final state.
    """
    wid = status_file.worker_id
    log_once = True

    while True:
        status = status_file.status

        if log_once:
            logger.info(f"[{wid}] Status {step_name}: {status}")
            log_once = False

        if status == STATUS_SUCCESS:
            logger.info(f"[{wid}] Step {step_name} has already succeeded.")
            return False

        if status in [STATUS_FAILED, STATUS_DEP_FAILED]:
            if force_run_failed:
                logger.info(f"[{wid}] Force running {step_name}, previous status: {status}")
            else:
                raise PreviousTaskFailedError(f"Step {step_name} failed previously. Status: {status}")
        elif status == STATUS_RUNNING and status_file.has_active_lock():
            logger.debug(f"[{wid}] Step {step_name} has active lock, waiting...")
            time.sleep(5)
            continue
        elif status == STATUS_RUNNING:
            logger.info(f"[{wid}] Step {step_name} has no active lock, taking over.")

        logger.info(f"[{wid}] Attempting to acquire lock for {step_name}")
        if status_file.try_acquire_lock():
            status_file.write_status(STATUS_RUNNING)
            logger.info(f"[{wid}] Acquired lock for {step_name}")
            return True

        logger.info(f"[{wid}] Lost lock race for {step_name}, retrying...")
        time.sleep(1)


# ---------------------------------------------------------------------------
# ExecutorStep → StepSpec conversion
# ---------------------------------------------------------------------------


def resolve_executor_step(
    step: ExecutorStep,
    config: Any,
    output_path: str,
    deps: list[str] | None = None,
) -> StepSpec:
    """Convert an ExecutorStep into a StepSpec.

    ``config`` should already be instantiated (no InputName / OutputName /
    VersionedValue markers).  The old executor called ``fn(config)``; we wrap
    that into a ``fn(output_path)`` closure expected by ``StepRunner``.
    """
    import ray

    step_fn = step.fn
    if isinstance(step_fn, ray.remote_function.RemoteFunction):
        remote_fn = step_fn

        def step_fn(*args, **kw):
            return ray.get(remote_fn.remote(*args, **kw))

    assert step_fn is not None, f"Step {step.name} has no callable"

    # Old-style ExecutorStep functions accept the resolved config as their only
    # argument. The config already contains the output path, so we ignore the
    # output_path parameter that StepRunner passes.
    captured_fn = step_fn
    captured_config = config

    def resolved_fn(output_path):
        return captured_fn(captured_config)

    return StepSpec(
        name=step.name,
        deps=deps or [],
        override_output_path=output_path,
        fn=resolved_fn,
        resources=step.resources if step.resources is not None else ResourceConfig.with_cpu(),
        env_vars=step.env_vars or {},
        pip_dependency_groups=step.pip_dependency_groups or [],
    )


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


class StepRunner:
    """Runs ``StepSpec`` objects as fray jobs, respecting dependencies.

    Steps are launched eagerly as they are yielded from an iterable. Each step
    is launched as soon as its dependencies are satisfied. Already-succeeded
    steps (STATUS_SUCCESS on disk) are skipped automatically.

    Each step is executed on the worker as
    ``disk_cached(distributed_lock(fn))``, so caching, locking, heartbeats,
    and status writes all happen on the worker side.
    """

    def __init__(self, client: fray_client.Client | None = None):
        self.client = client or fray_client.current_client()

    def run(
        self,
        steps: Iterable[StepSpec],
        *,
        dry_run: bool = False,
        force_run_failed: bool = True,
        max_concurrent: int | None = None,
    ) -> None:
        """Eagerly run steps, launching each as soon as its deps are satisfied.

        Consumes *steps* lazily. If a step's dependencies haven't been seen
        yet, it is buffered until they complete. Raises if the iterable is
        exhausted and buffered steps still have unsatisfied dependencies.
        """
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

        # Keyed by output_path (guaranteed unique)
        completed: set[str] = set()
        failed: set[str] = set()
        running: dict[str, JobHandle] = {}
        waiting: list[StepSpec] = []
        failures: list[Exception] = []
        # output_path → name_with_hash, for human-readable logging
        path_to_name: dict[str, str] = {}

        def _display_name(output_path: str) -> str:
            return path_to_name.get(output_path, output_path)

        def _harvest(block: bool = False) -> None:
            """Poll running jobs, moving finished ones to completed/failed."""
            if not running:
                return
            while True:
                done = [p for p, h in running.items() if JobStatus.finished(h.status())]
                if done or not block:
                    break
                time.sleep(1)
            for path in done:
                handle = running.pop(path)
                status = handle.wait(raise_on_failure=False)
                if status == JobStatus.FAILED:
                    logger.error(f"Step failed: {_display_name(path)}")
                    failed.add(path)
                    failures.append(RuntimeError(f"Step failed: {_display_name(path)}"))
                else:
                    completed.add(path)

        def _flush_waiting() -> None:
            """Launch buffered steps whose deps are now met."""
            i = 0
            while i < len(waiting):
                step = waiting[i]
                path = step.output_path
                if any(d in failed for d in step.deps):
                    waiting.pop(i)
                    failed.add(path)
                elif all(d in completed for d in step.deps):
                    if max_concurrent is not None and len(running) >= max_concurrent:
                        i += 1
                        continue
                    waiting.pop(i)
                    _do_launch(step)
                else:
                    i += 1

        def _do_launch(step: StepSpec) -> None:
            path = step.output_path
            handle = self._launch_step(step, force_run_failed=force_run_failed, dry_run=dry_run)
            if handle is not None:
                running[path] = handle
            else:
                completed.add(path)

        for step in steps:
            path_to_name[step.output_path] = step.name_with_hash

            # Block until there's capacity before consuming more from the iterator
            while max_concurrent is not None and len(running) >= max_concurrent:
                _harvest(block=True)
                _flush_waiting()

            _harvest()
            _flush_waiting()

            path = step.output_path
            if any(d in failed for d in step.deps):
                failed.add(path)
            elif all(d in completed for d in step.deps):
                _do_launch(step)
            else:
                waiting.append(step)

        # Drain remaining running and waiting steps
        while running or waiting:
            if not running and waiting:
                # One more flush in case deps failed and we can mark dependents
                _flush_waiting()
                if not running and waiting:
                    missing = []
                    for s in waiting:
                        unmet = [_display_name(d) for d in s.deps if d not in completed and d not in failed]
                        missing.append(f"  {s.name_with_hash}: needs {unmet}")
                    raise RuntimeError(
                        f"Iterable exhausted with {len(waiting)} step(s) with unsatisfied dependencies:\n"
                        + "\n".join(missing)
                    )

            _harvest(block=True)
            _flush_waiting()

        if failures:
            raise RuntimeError(f"{len(failures)} step(s) failed") from failures[0]

    def _launch_step(self, step: StepSpec, *, force_run_failed: bool, dry_run: bool) -> JobHandle | None:
        """Launch a single step as a fray job. Returns None if skipped."""
        from marin.execution.artifact import Artifact
        from marin.execution.disk_cache import disk_cached
        from marin.execution.distributed_lock import distributed_lock
        from marin.execution.fray_exe import exe_on_fray

        output_path = step.output_path
        step_name = step.name_with_hash
        logger.info(f"Step = {step_name}\tParams = {step.hash_attrs}\tOutput_path = {output_path}")

        # Quick read-only status check to avoid submitting unnecessary fray jobs
        status = StatusFile(output_path, worker_id="check").status
        if status == STATUS_SUCCESS:
            logger.info(f"Skip {step_name}: already succeeded")
            return None

        if not force_run_failed and status in (STATUS_FAILED, STATUS_DEP_FAILED):
            raise PreviousTaskFailedError(f"Step {step_name} failed previously. Status: {status}")

        if dry_run:
            logger.info(f"[DRY RUN] Would run {step_name} (status: {status})")
            return None

        _write_executor_info(step)

        if step.fn is None:
            raise ValueError(f"Step {step_name} has no callable fn")

        # Build worker function: disk_cached(distributed_lock(fn))
        # All caching, locking, heartbeat, artifact saving, and status writes
        # happen on the worker.
        step_fn = step.fn

        def worker_fn():
            disk_cached(
                step.name,
                distributed_lock(step_fn, force_run_failed=force_run_failed),
                override_output_path=output_path,
                save=Artifact.save,
                load=Artifact.load,
            )

        worker_fn.__qualname__ = step_name
        worker_fn.__name__ = step_name

        return exe_on_fray(
            worker_fn,
            name=step_name,
            resources=step.resources,
            env_vars=step.env_vars,
            pip_dependency_groups=step.pip_dependency_groups,
            client=self.client,
        )
