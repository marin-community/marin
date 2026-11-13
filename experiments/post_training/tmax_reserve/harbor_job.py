# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a shared-image Harbor dataset as one job, then delete the Daytona snapshots it created.

Shellbox builds one named snapshot per distinct image and resource shape and keeps it. The shared-image pilot has
two (images A and B). Before the job, the account must have that many free slots under ``snapshot_quota``; there
is no waiting. Afterwards, those snapshots are deleted by the names Shellbox gives them, each after a fresh
``get``; no other snapshot is touched.
"""

import datetime
import json
import logging
import os
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
# Live-sandbox allowance per trial for create and teardown, beyond its recorded phases.
SANDBOX_CREATE_SECONDS = 60
SANDBOX_PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


def _seconds(phase: dict | None) -> float:
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return 0.0
    start, end = (datetime.datetime.fromisoformat(phase[key]) for key in ("started_at", "finished_at"))
    return (end - start).total_seconds()


def sandbox_seconds(jobs_dir: Path) -> float:
    """Live sandbox seconds of a job's trials (each trial that reached environment setup had a sandbox)."""
    seconds = 0.0
    for result in jobs_dir.glob("*/*/result.json"):
        data = json.loads(result.read_text())
        if data.get("environment_setup"):
            seconds += SANDBOX_CREATE_SECONDS + sum(_seconds(data.get(phase)) for phase in SANDBOX_PHASES)
    return seconds


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


def run_harbor_job(
    config: ValidatedHarborConfig,
    overlay: HarborRuntimeOverlay,
    driver_env: Mapping[str, str],
    backend_state: Callable[[], InferenceBackendState],
    snapshot_quota: int,
    work: Path,
) -> Path:
    """Run ``overlay.dataset_path`` as one Harbor job and return its jobs directory.

    Writes ``<work>/account.json``, ``<work>/snapshot-names.json`` and ``<work>/snapshot-deletions.json``.
    """
    dataset = Path(overlay.dataset_path)
    task_dirs = sorted(str(path) for path in dataset.iterdir() if path.is_dir())
    names = sorted(set(_snapshots("names", *task_dirs, env=driver_env).values()))
    listing = work / "snapshot-names.json"
    listing.write_text(json.dumps(names))
    account = _snapshots("usage", env=driver_env)
    (work / "account.json").write_text(json.dumps(account, indent=1))
    free = snapshot_quota - account["non_general"]
    logger.info("Daytona: %d snapshots needed, %d of %d slots free", len(names), free, snapshot_quota)
    if free < len(names):
        raise RuntimeError(f"{len(names)} snapshots needed but only {free} of {snapshot_quota} slots are free")
    try:
        run_harbor_driver(config, overlay, driver_env, backend_state)
    finally:
        deletions = _snapshots("delete", str(listing), env=driver_env)
        (work / "snapshot-deletions.json").write_text(json.dumps(deletions, indent=1))
    return Path(overlay.jobs_dir)
