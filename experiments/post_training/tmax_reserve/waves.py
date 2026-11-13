# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor dataset in waves so its Daytona snapshots fit the account's snapshot quota.

Shellbox builds one named snapshot per distinct build context and keeps it. A wave holds at most ``wave_size`` tasks
(every attempt of a task stays in its wave). After each wave, the snapshots of that wave's tasks are deleted by the
names Shellbox gives them, each after a fresh ``get``; no other snapshot is touched.
"""

import dataclasses
import datetime
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from marin.evaluation.harbor.driver_config import (
    HarborRuntimeOverlay,
    ValidatedHarborConfig,
    harbor_runtime_command,
    run_harbor_driver,
)
from marin.inference.types import InferenceBackendState

logger = logging.getLogger(__name__)

SNAPSHOTS_SCRIPT = Path(__file__).with_name("snapshots.py")
REPO_ROOT = Path(__file__).parents[3]
# Trial exceptions that mean the lane itself is broken (snapshot quota, Daytona's JSON-form COPY parse).
LANE_ERRORS = ("Snapshot quota exceeded", "Path does not exist")
# Live-sandbox allowance per trial for create and teardown; environment_setup also spans the snapshot build.
SANDBOX_CREATE_SECONDS = 60


class WaveAbort(RuntimeError):
    """A finished wave broke a stop rule; later waves do not run."""


def _seconds(phase: dict | None) -> float:
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return 0.0
    start, end = (datetime.datetime.fromisoformat(phase[key]) for key in ("started_at", "finished_at"))
    return (end - start).total_seconds()


def wave_usage(jobs_dir: Path) -> tuple[float, list[str]]:
    """Live sandbox seconds of a wave's trials and its lane-error messages."""
    seconds, errors = 0.0, []
    for result in jobs_dir.glob("*/*/result.json"):
        data = json.loads(result.read_text())
        message = (data.get("exception_info") or {}).get("exception_message") or ""
        errors += [message[:200] for marker in LANE_ERRORS if marker in message]
        if data.get("agent_setup"):  # a sandbox existed
            seconds += SANDBOX_CREATE_SECONDS + sum(
                _seconds(data.get(phase)) for phase in ("agent_setup", "agent_execution", "verifier")
            )
    return seconds, errors


def _snapshots(command: str, *arguments: str, env: Mapping[str, str]) -> object:
    result = subprocess.run(
        harbor_runtime_command(SNAPSHOTS_SCRIPT, command, *arguments),
        env={**os.environ, **env, "PYTHONPATH": str(REPO_ROOT)},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def waves(tasks: list[str], wave_size: int) -> list[list[str]]:
    ordered = sorted(tasks)
    return [ordered[i : i + wave_size] for i in range(0, len(ordered), wave_size)]


def run_in_waves(
    config: ValidatedHarborConfig,
    overlay: HarborRuntimeOverlay,
    driver_env: Mapping[str, str],
    backend_state: Callable[[], InferenceBackendState],
    wave_size: int,
    work: Path,
    max_sandbox_hours: float,
) -> list[Path]:
    """Run ``overlay.dataset_path`` wave by wave; return each wave's jobs directory.

    Stops (``WaveAbort``) after a wave with a lane error, or once live sandbox hours pass ``max_sandbox_hours``.
    Writes ``<work>/snapshot-names.json`` before the first wave and ``<work>/snapshot-deletions.jsonl`` as it goes.
    """
    dataset = Path(overlay.dataset_path)
    task_names = sorted(path.name for path in dataset.iterdir() if path.is_dir())
    names = _snapshots("names", *(str(dataset / task) for task in task_names), env=driver_env)
    (work / "snapshot-names.json").write_text(json.dumps(names, indent=1))
    jobs_dirs = []
    sandbox_seconds = 0.0
    for index, wave in enumerate(waves(task_names, wave_size)):
        wave_dir = work / "waves" / f"w{index:02d}"
        for task in wave:
            shutil.copytree(dataset / task, wave_dir / "tasks" / task)
        jobs_dirs.append(wave_dir / "jobs")
        wave_overlay = dataclasses.replace(
            overlay,
            job_name=f"{overlay.job_name}-w{index:02d}",
            jobs_dir=str(wave_dir / "jobs"),
            dataset_path=str(wave_dir / "tasks"),
            archive_root=f"{overlay.archive_root}/w{index:02d}",
        )
        logger.info("wave %d: %d tasks", index, len(wave))
        try:
            run_harbor_driver(config, wave_overlay, driver_env, backend_state)
        finally:
            listing = wave_dir / "snapshot-names.json"
            listing.write_text(json.dumps([names[task] for task in wave]))
            deletions = _snapshots("delete", str(listing), env=driver_env)
            with (work / "snapshot-deletions.jsonl").open("a") as log:
                log.write("".join(json.dumps({"wave": index, **row}) + "\n" for row in deletions))
        seconds, errors = wave_usage(wave_dir / "jobs")
        sandbox_seconds += seconds
        logger.info(
            "wave %d done: %.2f live sandbox-h so far, %d lane errors", index, sandbox_seconds / 3600, len(errors)
        )
        if errors:
            raise WaveAbort(f"wave {index} had {len(errors)} lane errors, e.g. {errors[0]!r}")
        if sandbox_seconds / 3600 > max_sandbox_hours:
            raise WaveAbort(f"live sandbox-h {sandbox_seconds / 3600:.2f} passed {max_sandbox_hours}")
    return jobs_dirs
