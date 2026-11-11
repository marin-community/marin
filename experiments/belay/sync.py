# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep the Belay page current: Belay's Iris runs and their W&B progress, the registry's results, and storage use.

    uv run python -m experiments.belay sync

The Loom watch runs one pass before every queue check, so the page stays current with no long-running process; the
local demo runs the same pass on a loop. Every pass ends with the app's heartbeat (``agent/tick``), which also turns
newly finished runs into events and re-offers approved work no session has taken.
"""

import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import s3fs
import wandb

from experiments.belay.branch import git
from experiments.belay.config import CONFIG, COREWEAVE_S3_ENDPOINT
from experiments.belay.page import call, mark_reviewed, page_state
from experiments.belay.registry_entry import require_registry_checkout
from experiments.belay.submit import iris_jobs

logger = logging.getLogger("belay.sync")

STORAGE_INTERVAL = datetime.timedelta(hours=1)
# Ideas decided this recently get 👀 (again, harmlessly) on each pass; well past the watch interval.
REVIEW_REACTION_WINDOW = datetime.timedelta(days=1)
UNDECIDED_STATES = ("pending", "dismissed")
WANDB_PATH = CONFIG.wandb_project
STORAGE_BUCKET = "marin-us-east-02a"
# Data Belay (experiments/data_belay) launches through the Belay Track command, so its runs land under the same root.
# They are not Belay's, so storage use leaves them out.
DATA_BELAY_RUN_PREFIX = "runs/dbelay-"
PLACEHOLDER_BPB_RATIO = "0.4147"


def _run_state(coord: str, train: str | None) -> str:
    if coord in ("pending", "running") and train is not None:
        return train
    if coord == "running":
        return "pending"  # the coordinator is up; the training job is not placed yet
    return coord


def registry_results(registry_dir: Path) -> tuple[str, dict]:
    """The registry checkout's HEAD and its ``registry build`` output (the public page's results data)."""
    revision = git(registry_dir, "rev-parse", "HEAD")
    built = subprocess.run(
        [
            sys.executable,
            str(registry_dir / "experiments/belay/registry.py"),
            "build",
            "--placeholder-bpb-ratio",
            PLACEHOLDER_BPB_RATIO,
            "--default-author",
            CONFIG.default_author,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, json.loads(built)


def is_belay_file(relative_path: str) -> bool:
    """Whether a file under the Belay output root belongs to Belay rather than to a Data Belay run."""
    return not relative_path.startswith(DATA_BELAY_RUN_PREFIX)


class Watcher:
    """One pass (``tick``) reports Belay's Iris runs with their W&B progress, the registry's results when the registry
    branch moved, and storage use at most hourly. ``token`` is the local demo's service token; on Marina the caller's
    login authorizes the ``agent/`` routes."""

    def __init__(self, registry_dir: Path, token: str | None, storage_at: datetime.datetime | None = None) -> None:
        self.registry_dir = registry_dir
        self.token = token
        self.final: dict[str, dict] = {}
        self.storage_at = storage_at
        self.wandb = wandb.Api()

    def _wandb(self, name: str) -> dict:
        try:
            run = self.wandb.run(f"{WANDB_PATH}/{name}")
        except (wandb.errors.CommError, ValueError):
            return {}
        loss = run.summary.get("eval/paloma/macro_loss")
        return {
            "step": run.summary.get("_step"),
            "loss": float(loss) if loss is not None else None,
            "wandb_url": run.url,
            "finished": run.state == "finished",
        }

    def runs(self) -> list[dict]:
        self.wandb = wandb.Api()  # a fresh client per tick: Api caches each Run it has loaded
        coords, trains, submitted = {}, {}, {}
        for job in iris_jobs():
            parts = job.job_id.strip("/").split("/")
            if len(parts) == 2 and parts[1].endswith("-coord"):
                name = parts[1].removesuffix("-coord")
                coords[name] = (job.job_id, job.state)
                submitted[name] = job.submitted_at
            elif len(parts) == 3 and parts[1].endswith("-coord"):
                trains[parts[1].removesuffix("-coord")] = job.state
        reports = []
        for name, (job_id, coord_state) in coords.items():
            state = _run_state(coord_state, trains.get(name))
            report = {
                "name": name,
                "job_id": job_id,
                "state": state,
                "nodes": 1,
                "total_steps": None,
                "submitted_at": submitted.get(name),
            }
            if name in self.final:
                report.update(self.final[name])
            elif state != "pending":
                progress = self._wandb(name)
                report.update({k: v for k, v in progress.items() if k != "finished"})
                if state == "succeeded" and progress.get("finished") and progress.get("loss") is not None:
                    self.final[name] = {k: v for k, v in progress.items() if k != "finished"}
            reports.append(report)
        return reports

    def storage(self) -> tuple[int, int] | None:
        now = datetime.datetime.now(datetime.UTC)
        if self.storage_at is not None and now - self.storage_at < STORAGE_INTERVAL:
            return None
        fs = s3fs.S3FileSystem(
            key=os.environ["CW_KEY_ID"],
            secret=os.environ["CW_KEY_SECRET"],
            endpoint_url=COREWEAVE_S3_ENDPOINT,
            config_kwargs={"s3": {"addressing_style": "virtual"}},
        )
        prefix = f"{STORAGE_BUCKET}/marin/{CONFIG.output_root.rsplit('/', 1)[0]}/"  # the launcher writes under marin/
        files = [f for path, f in fs.find(prefix, detail=True).items() if is_belay_file(path.removeprefix(prefix))]
        self.storage_at = now
        return sum(f["size"] for f in files), len(files)

    def upload_registry(self) -> None:
        """Rebuild and upload the registry's results when the registry checkout moved."""
        head = git(self.registry_dir, "rev-parse", "HEAD")
        if call("agent/registry", token=self.token)["revision"] == head:
            return
        revision, data = registry_results(self.registry_dir)
        call("agent/registry", {"revision": revision, "data": data}, token=self.token)
        logger.info("uploaded registry results at %s (%d ideas)", revision[:10], len(data["ideas"]))

    def tick(self) -> None:
        error = None
        try:
            self.upload_registry()
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            error = f"registry: {e}"
        reports = self.runs()
        call("agent/sync/runs", {"runs": reports}, token=self.token)
        storage = None
        try:
            storage = self.storage()
        except (OSError, KeyError) as e:
            error = f"storage: {e}"
        nodes = sum(r["nodes"] for r in reports if r["state"] == "running")
        body = {"nodes_in_use": nodes, "error": error}
        if storage is not None:
            body.update(storage_bytes=storage[0], storage_objects=storage[1])
        call("agent/tick", body, token=self.token)
        logger.info("tick: %d runs, %d nodes running%s", len(reports), nodes, f", {error}" if error else "")


def sync() -> None:
    """One pass against the configured page, measuring storage only when the page's last measurement is an hour old."""
    require_registry_checkout()
    measured = call("agent/state")["watcher"]["storage_at"]
    storage_at = None if measured is None else datetime.datetime.fromisoformat(measured)
    Watcher(CONFIG.registry_checkout, token=None, storage_at=storage_at).tick()
    react_to_reviewed()


def react_to_reviewed() -> None:
    """Mark each #9514 suggestion a maintainer approved or declined with 👀: the app holds no GitHub credentials, so a
    decision made on the admin page shows on GitHub at the next pass."""
    state = page_state()
    since = datetime.datetime.now(datetime.UTC) - REVIEW_REACTION_WINDOW
    for idea in state.ideas:
        if idea.origin != "intake" or idea.state in UNDECIDED_STATES:
            continue
        if datetime.datetime.fromisoformat(idea.updated_at) < since:
            continue
        mark_reviewed(idea, github_posting=state.github_posting)
