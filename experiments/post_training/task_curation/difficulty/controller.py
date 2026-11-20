# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One watchdog-owned controller for all difficulty remediation children."""

import json
import logging
import os
import subprocess
import time

from finestore.rl.difficulty import Solver
from iris.client.client import iris_ctx
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT, IapRefreshTokenProvider
from rigging.credential_store import load_credentials
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret

from experiments.post_training.task_curation.difficulty.capacity import available_provider_solvers
from experiments.post_training.task_curation.difficulty.catalog import served_catalog
from experiments.post_training.task_curation.difficulty.executor import EscalationLog, collect_finished, reconcile_once
from experiments.post_training.task_curation.difficulty.iris_backend import IrisBackend
from experiments.post_training.task_curation.difficulty.ledger import read_rows
from experiments.post_training.task_curation.difficulty.report import cells_for_catalog
from experiments.post_training.task_curation.difficulty.state import CoveragePolicy, GapReason
from experiments.post_training.task_curation.difficulty_report import list_iris_jobs

LOGGER = logging.getLogger(__name__)
CONTROLLER_INTERVAL = 60


def run_controller(ledger: str, applet_id: str, once: bool, providers: frozenset[Solver]) -> None:
    context = iris_ctx()
    backend = IrisBackend(context.client, context.job_id, ledger)
    notifications = EscalationLog(ledger)
    policy = CoveragePolicy()
    provider = IapRefreshTokenProvider(
        MARIN_DESKTOP_OAUTH_CLIENT.client_id,
        MARIN_DESKTOP_OAUTH_CLIENT.client_secret,
        require_secret("MARINA_IAP_REFRESH_TOKEN"),
    )
    while True:
        catalog = served_catalog(applet_id, token_provider=provider)
        rows = read_rows(ledger)
        cells = cells_for_catalog(rows, catalog, policy)
        artifacts = {
            gap.artifact_uri
            for cell in cells
            for gap in cell.gaps
            if gap.reason == GapReason.ENVIRONMENT_ARTIFACT_MISSING and gap.artifact_uri
        }
        available = frozenset(uri for uri in artifacts if StoragePath(uri).exists())
        waiting = any(gap.reason == GapReason.PROVIDER_CAPACITY_WAIT for cell in cells for gap in cell.gaps)
        capacity = providers or (available_provider_solvers(ledger) if waiting else frozenset())
        result = reconcile_once(ledger, catalog, policy, backend, notifications, capacity, available)
        LOGGER.info("DIFFICULTY_RECONCILED %s", json.dumps(result, sort_keys=True))
        if once:
            while any(job.active and job.job_id.startswith(str(context.job_id) + "/") for job in backend.jobs()):
                time.sleep(CONTROLLER_INTERVAL)
            collect_finished(read_rows(ledger), backend.jobs(), ledger, notifications)
            return
        time.sleep(CONTROLLER_INTERVAL)


def launch_controller(ledger: str, applet_id: str, once: bool, providers: frozenset[Solver], generation: str) -> None:
    """Submit one CPU parent; its inherited client owns every remediation child."""
    for job in list_iris_jobs():
        if "atlas-difficulty-reconciler-controller-" in job.job_id and job.active:
            print(json.dumps({"controller_job": job.job_id, "state": job.state}))
            return
    record = load_credentials("marin")
    if record is None or not record.edge_refresh_token:
        raise ValueError("Controller needs a refreshable Marina login; run iris login")
    environment = {**os.environ, "DIFFICULTY_MARINA_REFRESH": record.edge_refresh_token}
    command = [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--job-name",
        "atlas-difficulty-reconciler-controller-" + generation[:16],
        "--target-cluster=cw-rno2a",
        "--cpu",
        "4",
        "--memory",
        "8GB",
        "--disk",
        "10GB",
        "--enable-extra-resources",
        "--priority",
        "interactive",
        "--timeout",
        "259200",
        "--max-retries",
        "0",
        "--no-wait",
        "--secret",
        "MARINA_IAP_REFRESH_TOKEN",
        "env:DIFFICULTY_MARINA_REFRESH",
    ]
    for name in ("CW_KEY_ID", "CW_KEY_SECRET", "HF_TOKEN", "GLM_BULK_TOKEN", "TOGETHER_API_KEY"):
        command.extend(["--secret", name, "env:" + name])
    command.extend(
        [
            "--secret",
            "DAYTONA_API_KEY",
            "gcp-secret://hai-gcp-models/DAYTONA_EVAL_API_KEY",
            "--secret",
            "DAYTONA_RL_API_KEY",
            "gcp-secret://hai-gcp-models/DAYTONA_RL_API_KEY",
            "--",
            "python",
            "-m",
            "experiments.post_training.task_curation.difficulty",
            "reconcile",
            "--ledger",
            ledger,
            "--applet-id",
            applet_id,
        ]
    )
    if once:
        command.append("--once")
    for solver in sorted(providers):
        command.extend(["--provider-available", solver.value])
    subprocess.run(command, check=True, env=environment)
