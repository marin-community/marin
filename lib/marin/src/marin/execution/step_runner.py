# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step runner for StepSpec.

This module provides the execution layer for ``StepSpec`` objects. The main
entry point is ``StepRunner``, a thin DAG scheduler that launches steps as
they are yielded from an iterable, as soon as their dependencies are satisfied.

Each step's execution strategy (caching, locking, remote submission) is owned
by ``StepSpec.executable_fn``.  ``StepRunner`` simply calls that callable and
polls for completion.

``ExecutorStep`` objects can be converted to ``StepSpec`` via
``resolve_executor_step`` in ``marin.execution.executor``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import fsspec
import levanter.utils.fsspec_utils as fsspec_utils

from fray.v2.client import JobHandle, JobStatus
from marin.execution.executor_step_status import (
    STATUS_DEP_FAILED,
    STATUS_FAILED,
    STATUS_SUCCESS,
    StatusFile,
)
from fray.v2.local_backend import LocalJobHandle
from marin.execution.step_spec import StepSpec
from marin.utilities.json_encoder import CustomJsonEncoder

# Re-export for backward compatibility
from marin.execution.executor_step_status import PreviousTaskFailedError, should_run, worker_id  # noqa: F401

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
    fs = fsspec.core.url_to_fs(info_path, use_listings_cache=False)[0]
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
    with fsspec.open(info_path, "w") as f:
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
            handle = self._launch_step(step, force_run_failed=force_run_failed, dry_run=dry_run, local_pool=local_pool)
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

    def _launch_step(
        self, step: StepSpec, *, force_run_failed: bool, dry_run: bool, local_pool: ThreadPoolExecutor
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

        executable = step.executable_fn

        # NOTE: we still wrap to update the names to make logs more readable
        def worker_fn():
            executable(output_path)

        worker_fn.__qualname__ = step_name
        worker_fn.__name__ = step_name

        # TODO: should this be async to avoid thread pool?
        future = local_pool.submit(worker_fn)
        return LocalJobHandle(f"local-{step_name}", future)
