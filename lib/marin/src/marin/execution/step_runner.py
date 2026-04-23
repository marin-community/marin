# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step runner for StepSpec.

This module provides the execution layer for ``StepSpec`` objects. The main
entry point is ``StepRunner``, a thin DAG scheduler that launches steps as
they are yielded from an iterable, as soon as their dependencies are satisfied.

Caching, distributed locking, heartbeats, and status writes are handled
explicitly in :func:`run_step` rather than composed as decorators.  This
makes the control flow easy to follow and debug.

``ExecutorStep`` objects can be converted to ``StepSpec`` via
``resolve_executor_step`` in ``marin.execution.executor``.
"""

from __future__ import annotations

import contextvars
import dataclasses
from datetime import timedelta
import json
import logging
import os
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import levanter.utils.fsspec_utils as fsspec_utils
from rigging.filesystem import open_url, url_to_fs

from fray.client import JobHandle, JobStatus
from fray.local_backend import LocalJobHandle
from marin.execution.artifact import Artifact
from marin.execution.executor_step_status import (
    STATUS_DEP_FAILED,
    STATUS_FAILED,
    STATUS_SUCCESS,
    StepAlreadyDone,
    StatusFile,
    step_lock,
)
from marin.execution.remote import RemoteCallable
from marin.execution.step_spec import StepSpec
from marin.utilities.json_encoder import CustomJsonEncoder

# Re-export for backward compatibility
from marin.execution.executor_step_status import PreviousTaskFailedError, worker_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_executor_info(step: StepSpec) -> None:
    """Write a ``.executor_info`` JSON file matching the legacy ExecutorStepInfo schema.

    Skips writing if the file already exists (e.g. Executor.write_infos() wrote
    a richer version before StepRunner launched the step).
    """
    info_path = os.path.join(step.output_path, ".executor_info")
    fs = url_to_fs(info_path, use_listings_cache=False)[0]
    if fs.exists(info_path):
        return
    info = {
        "executor_version": "step_runner",
        "name": step.name,
        "fn_name": str(step.fn) if step.fn is not None else "None",
        "config": step.hash_attrs,
        "description": None,
        "override_output_path": step.override_output_path,
        "version": {},
        "dependencies": list(step.dep_paths),
        "output_path": step.output_path,
    }
    fsspec_utils.mkdirs(step.output_path)
    with open_url(info_path, "w") as f:
        f.write(json.dumps(info, indent=2, cls=CustomJsonEncoder))


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


class StepRunner:
    """Runs ``StepSpec`` objects respecting their dependencies.

    Steps are launched eagerly as they are yielded from an iterable. Each step
    is launched as soon as its dependencies are satisfied. Already-succeeded
    steps (STATUS_SUCCESS on disk) are skipped automatically.
    """

    def run(
        self,
        steps: Iterable[StepSpec],
        *,
        dry_run: bool = False,
        force_run_failed: bool = True,
        max_concurrent: int | None = None,
    ) -> None:
        """Eagerly run steps, launching each as soon as its deps are satisfied.

        Concurrency is bounded by the thread pool (``max_concurrent`` workers,
        default 8). If a step's dependencies haven't been seen yet, it is
        buffered until they complete. Raises if the iterable is exhausted and
        buffered steps still have unsatisfied dependencies.
        """
        max_workers = max_concurrent or 8
        if max_workers < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

        # Capture the fray client on the calling thread so worker threads can
        # inherit it explicitly.  This is more robust than contextvars.copy_context()
        # alone because it survives process/thread-pool boundary edge cases.
        from fray.client import _current_client_var

        caller_fray_client = _current_client_var.get()
        if caller_fray_client is not None:
            logger.info("StepRunner: captured fray client %s for worker threads", type(caller_fray_client).__name__)

        local_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="marin-step-runner")

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
                step_name = _display_name(path)
                try:
                    status = handle.wait(raise_on_failure=True)
                except Exception as exc:
                    logger.exception("Step failed: %s", step_name)
                    failed.add(path)
                    wrapped = RuntimeError(f"Step failed: {step_name}")
                    wrapped.__cause__ = exc
                    failures.append(wrapped)
                    continue

                if status in (JobStatus.FAILED, JobStatus.STOPPED):
                    logger.error("Step failed: %s (status=%s)", step_name, status.value)
                    failed.add(path)
                    failures.append(RuntimeError(f"Step failed: {step_name}; status={status.value}"))
                else:
                    completed.add(path)

        def _flush_waiting() -> None:
            """Launch buffered steps whose deps are now met."""
            i = 0
            while i < len(waiting):
                step = waiting[i]
                path = step.output_path
                if any(d in failed for d in step.dep_paths):
                    waiting.pop(i)
                    failed.add(path)
                elif all(d in completed for d in step.dep_paths):
                    waiting.pop(i)
                    _do_launch(step)
                else:
                    i += 1

        def _do_launch(step: StepSpec) -> None:
            path = step.output_path
            handle = self._launch_step(
                step,
                force_run_failed=force_run_failed,
                dry_run=dry_run,
                local_pool=local_pool,
                fray_client=caller_fray_client,
            )
            if handle is not None:
                running[path] = handle
            else:
                completed.add(path)

        for step in steps:
            path_to_name[step.output_path] = step.name_with_hash

            _harvest()
            _flush_waiting()

            path = step.output_path
            if any(d in failed for d in step.dep_paths):
                failed.add(path)
            elif all(d in completed for d in step.dep_paths):
                _do_launch(step)
            else:
                waiting.append(step)

        # Drain remaining running and waiting steps
        while running or waiting:
            if not running and waiting:
                _flush_waiting()
                if not running and waiting:
                    missing = []
                    for s in waiting:
                        unmet = [
                            _display_name(d.output_path)
                            for d in s.deps
                            if d.output_path not in completed and d.output_path not in failed
                        ]
                        missing.append(f"  {s.name_with_hash}: needs {unmet}")
                    raise RuntimeError(
                        f"Iterable exhausted with {len(waiting)} step(s) with unsatisfied dependencies:\n"
                        + "\n".join(missing)
                    )

            _harvest(block=True)
            _flush_waiting()

        if failures:
            raise RuntimeError(f"{len(failures)} step(s) failed") from failures[0]

    def _launch_step(
        self,
        step: StepSpec,
        *,
        force_run_failed: bool,
        dry_run: bool,
        local_pool: ThreadPoolExecutor,
        fray_client: object | None = None,
    ) -> JobHandle | None:
        """Launch a single step. Returns None if skipped."""
        output_path = step.output_path
        step_name = step.name_with_hash
        logger.info(f"Step = {step_name}\tParams = {step.hash_attrs}\tOutput_path = {output_path}")

        # Quick read-only status check to avoid submitting unnecessary jobs
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

        # Explicitly propagate the fray client into worker threads.
        # We use set_current_client() inside the worker rather than relying
        # solely on contextvars.copy_context(), because the latter can
        # silently lose state across certain thread-pool reuse patterns.
        captured_client = fray_client

        def worker_fn():
            if captured_client is not None:
                from fray.client import set_current_client

                with set_current_client(captured_client):
                    run_step(step)
            else:
                run_step(step)

        worker_fn.__qualname__ = step_name
        worker_fn.__name__ = step_name

        # Also copy the full context for any other context vars (not just fray).
        ctx = contextvars.copy_context()
        future = local_pool.submit(ctx.run, worker_fn)
        return LocalJobHandle(f"local-{step_name}", future)


# ---------------------------------------------------------------------------
# Explicit step execution: cache, lock, heartbeat, run, save, status
# ---------------------------------------------------------------------------


def check_cache(output_path: str) -> bool:
    """Return True if the step already succeeded (cache hit)."""
    status = StatusFile(output_path, worker_id()).status
    if status == STATUS_SUCCESS:
        logger.info(f"Cache hit for {output_path}")
        return True
    return False


def run_step(step: StepSpec) -> None:
    """Execute a single step with explicit cache check, locking, heartbeat, and artifact saving.

    For local steps the result is saved via ``Artifact.save``.  For remote
    steps (``@remote``) the raw function + artifact save are submitted as a
    Fray job; the executor node only manages the lock and status file.
    """
    output_path = step.output_path
    step_label = step.name_with_hash

    # 1. Cache check
    if check_cache(output_path):
        return

    # 2. Acquire distributed lock with heartbeat (blocks until lock obtained or step done)
    try:
        with step_lock(output_path, step_label) as status_file:
            # 3. Run the function
            try:
                t0 = time.monotonic()
                if isinstance(step.fn, RemoteCallable):
                    _run_remote_step(step, output_path)
                else:
                    result = step.fn(output_path)  # pyrefly: ignore[not-callable]
                    Artifact.save(result, output_path)
                elapsed = timedelta(seconds=time.monotonic() - t0)

                # 4. Mark success
                status_file.write_status(STATUS_SUCCESS)
                logger.info(f"Step {step_label} succeeded in {elapsed}")
            except Exception:
                status_file.write_status(STATUS_FAILED)
                raise
    except StepAlreadyDone:
        logger.info(f"Step {step_label} completed by another worker")


def _run_remote_step(step: StepSpec, output_path: str) -> None:
    """Submit the step's raw function to Fray with artifact saving inside the job.

    Fray jobs can't return values back to the executor, so the artifact
    is saved inside the remote job itself.
    """
    assert isinstance(step.fn, RemoteCallable)
    raw_fn = step.fn.fn

    def _fn_with_artifact_save(out_path: str):
        result = raw_fn(out_path)  # pyrefly: ignore[not-callable]
        Artifact.save(result, out_path)

    job_name = f"{step.name_with_hash}-{uuid.uuid4().hex[:8]}"
    remote_callable = step.fn.named(job_name)
    job = dataclasses.replace(remote_callable, fn=_fn_with_artifact_save)
    job(output_path)
