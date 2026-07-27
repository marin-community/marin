# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one Harbor job in an isolated environment as a subprocess.

Harbor (and its Daytona SDK) carry pre-release transitive pins that do not fit the marin lock, so
Harbor is treated as an external tool: :func:`marin.evaluation.harbor.runner.run_harbor` runs
this script under ``uv run --no-project --with <pinned-harbor> --with daytona --prerelease=allow``, which
builds an ephemeral environment with Harbor and Daytona. This file imports the stdlib-only typed
adapter from the Marin source tree, but none of Marin's runtime dependencies, so it loads cleanly in
that project-less environment.

It reads a JSON config (path in ``argv[1]``), runs the Harbor job against the served model's proxy
URL, and lets Harbor write its native ``result.json`` tree under ``jobs_dir``. The caller reads those
trial files back and normalizes them into the shared eval contract.

Harbor's interactive progress display is terminal-oriented. Lifecycle callbacks emit flushed,
newline-delimited updates so Iris can capture progress from non-interactive jobs.
"""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from marin.evaluation.harbor.driver_config import HarborDriverConfig, native_job_config


class _ProgressEvent(Protocol):
    trial_name: str


class _ExceptionInfo(Protocol):
    exception_type: str


class _TrialResult(Protocol):
    exception_info: _ExceptionInfo | None


class _TrialEndedEvent(_ProgressEvent, Protocol):
    result: _TrialResult


_ProgressCallback = Callable[[_ProgressEvent], Awaitable[None]]
_TrialEndedCallback = Callable[[_TrialEndedEvent], Awaitable[None]]


class _ProgressJob(Protocol):
    def on_trial_started(self, callback: _ProgressCallback) -> object: ...

    def on_environment_started(self, callback: _ProgressCallback) -> object: ...

    def on_agent_started(self, callback: _ProgressCallback) -> object: ...

    def on_verification_started(self, callback: _ProgressCallback) -> object: ...

    def on_trial_ended(self, callback: _TrialEndedCallback) -> object: ...


def _progress_callback(status: str) -> _ProgressCallback:
    async def report(event: _ProgressEvent) -> None:
        print(f"Harbor trial {event.trial_name}: {status}", flush=True)

    return report


async def _report_trial_ended(event: _TrialEndedEvent) -> None:
    exception = event.result.exception_info
    status = f"failed ({exception.exception_type})" if exception is not None else "completed"
    print(f"Harbor trial {event.trial_name}: {status}", flush=True)


def _register_progress_callbacks(job: _ProgressJob) -> None:
    job.on_trial_started(_progress_callback("started"))
    job.on_environment_started(_progress_callback("environment started"))
    job.on_agent_started(_progress_callback("agent started"))
    job.on_verification_started(_progress_callback("verification started"))
    job.on_trial_ended(_report_trial_ended)


async def _run(config: dict) -> None:
    from harbor.job import Job  # noqa: PLC0415  # optional dependency: isolated driver
    from harbor.models.job.config import JobConfig  # noqa: PLC0415  # optional dependency: isolated driver

    driver_config = HarborDriverConfig.from_dict(config)
    job = await Job.create(JobConfig.model_validate(native_job_config(driver_config)))
    _register_progress_callbacks(job)
    await job.run()


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
