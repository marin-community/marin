# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a shared-image Harbor dataset as one job, then delete the Daytona snapshots it created.

Shellbox builds one named snapshot per distinct image and resource shape and keeps it. The shared-image pilot has
two (images A and B). Their names come from Shellbox's own naming path. Names that already exist before the job
are reused and never deleted; the account must have a free slot under ``snapshot_quota`` for each of the others,
with no waiting. They are built one at a time before the job starts (concurrent first trials would race their
creates against the quota). Afterwards only those others (the snapshots this job created) are deleted, by exact
name and each after a fresh ``get``; no other snapshot is touched.
"""

import datetime
import json
import logging
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

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


# Characters of a failed snapshots.py run's stderr kept in the raised error.
SNAPSHOT_STDERR_TAIL = 8000


def snapshot_command(command: str, *arguments: str, env: Mapping[str, str]) -> Any:
    """Run one ``snapshots.py`` command in the vendored Harbor runtime and return its JSON output."""
    result = subprocess.run(
        harbor_runtime_command(SNAPSHOTS_SCRIPT, command, *arguments),
        env={**os.environ, **env, "PYTHONPATH": str(REPO_ROOT)},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"snapshots.py {command} exited {result.returncode}:\n{result.stderr[-SNAPSHOT_STDERR_TAIL:]}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_harbor_job(
    config: ValidatedHarborConfig,
    overlay: HarborRuntimeOverlay,
    driver_env: Mapping[str, str],
    backend_state: Callable[[], InferenceBackendState],
    snapshot_quota: int,
    work: Path,
) -> None:
    """Run ``overlay.dataset_path`` as one Harbor job.

    Writes ``<work>/account.json``, ``<work>/snapshots.json`` (all, pre-existing, and to-delete names),
    ``<work>/snapshots-prepared.json`` and ``<work>/snapshot-deletions.json``.
    """
    dataset = Path(overlay.dataset_path)
    task_dirs = sorted(str(path) for path in dataset.iterdir() if path.is_dir())
    names = sorted(set(snapshot_command("names", *task_dirs, env=driver_env).values()))
    all_names = work / "snapshot-names.json"
    all_names.write_text(json.dumps(names))
    preexisting = snapshot_command("existing", str(all_names), env=driver_env)
    created = [name for name in names if name not in preexisting]
    to_delete = work / "snapshots-to-delete.json"
    to_delete.write_text(json.dumps(created))
    (work / "snapshots.json").write_text(json.dumps({"names": names, "preexisting": preexisting, "to_delete": created}))
    account = snapshot_command("usage", env=driver_env)
    (work / "account.json").write_text(json.dumps(account, indent=1))
    free = snapshot_quota - account["non_general"]
    logger.info("Daytona: %d new snapshots needed, %d of %d slots free", len(created), free, snapshot_quota)
    if free < len(created):
        raise RuntimeError(f"{len(created)} new snapshots needed but only {free} of {snapshot_quota} slots are free")
    try:
        (work / "snapshots-prepared.json").write_text(
            json.dumps(snapshot_command("prepare", *task_dirs, env=driver_env), indent=1)
        )
        run_harbor_driver(config, overlay, driver_env, backend_state)
    finally:
        deletions = snapshot_command("delete", str(to_delete), env=driver_env)
        (work / "snapshot-deletions.json").write_text(json.dumps(deletions, indent=1))
