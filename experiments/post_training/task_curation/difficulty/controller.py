# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One watchdog-owned controller for all difficulty remediation children."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import requests
from dotenv import dotenv_values
from finestore.rl.difficulty import Solver
from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT, IapRefreshTokenProvider
from rigging.credential_store import load_credentials
from rigging.secrets import require_secret

from experiments.post_training.task_curation.difficulty.capacity import available_provider_solvers
from experiments.post_training.task_curation.difficulty.catalog import available_artifact_uris, served_catalog
from experiments.post_training.task_curation.difficulty.executor import EscalationLog, collect_finished, reconcile_once
from experiments.post_training.task_curation.difficulty.iris_backend import IrisBackend
from experiments.post_training.task_curation.difficulty.launch import cpu_job_command
from experiments.post_training.task_curation.difficulty.ledger import read_rows
from experiments.post_training.task_curation.difficulty.report import cells_for_catalog
from experiments.post_training.task_curation.difficulty.state import CoveragePolicy, GapReason
from experiments.post_training.task_curation.difficulty_report import list_iris_jobs

LOGGER = logging.getLogger(__name__)
CONTROLLER_INTERVAL = 60
CONTROLLER_JOB_PREFIX = "atlas-difficulty-reconciler-controller-"
REFRESH_TOKEN_ENV = "DIFFICULTY_MARINA_REFRESH"
CONTROLLER_SECRET_NAMES = (
    "CW_KEY_ID",
    "CW_KEY_SECRET",
    "HF_TOKEN",
    "GLM_BULK_TOKEN",
    "TOGETHER_API_KEY",
    "DAYTONA_DATA_API_KEY",
)


def run_controller(
    ledger: str, applet_id: str, once: bool, providers: frozenset[Solver], budget_gpu_hours: float
) -> None:
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
        try:
            catalog = served_catalog(applet_id, token_provider=provider, ledger=ledger)
        except (requests.ConnectionError, requests.Timeout):
            LOGGER.exception("DIFFICULTY_CATALOG_UNAVAILABLE; retrying after %d seconds", CONTROLLER_INTERVAL)
            time.sleep(CONTROLLER_INTERVAL)
            continue
        rows = read_rows(ledger)
        cells = cells_for_catalog(rows, catalog, policy)
        available = available_artifact_uris(cells)
        waiting = any(gap.reason == GapReason.PROVIDER_CAPACITY_WAIT for cell in cells for gap in cell.gaps)
        capacity = providers or (available_provider_solvers(ledger) if waiting else frozenset())
        result = reconcile_once(ledger, catalog, policy, backend, notifications, capacity, available, budget_gpu_hours)
        LOGGER.info("DIFFICULTY_RECONCILED %s", json.dumps(result, sort_keys=True))
        if once:
            while any(job.active and job.job_id.startswith(str(context.job_id) + "/") for job in backend.jobs()):
                time.sleep(CONTROLLER_INTERVAL)
            collect_finished(read_rows(ledger), backend.jobs(), ledger, notifications)
            return
        time.sleep(CONTROLLER_INTERVAL)


def controller_environment(secrets_env: Path | None) -> dict[str, str]:
    """Resolve storage and provider credentials before invoking the Iris submitter."""
    if secrets_env is not None and not secrets_env.is_file():
        raise FileNotFoundError(secrets_env)
    supplied = dotenv_values(secrets_env) if secrets_env is not None else {}
    environment = dict(os.environ)
    for name in CONTROLLER_SECRET_NAMES:
        environment[name] = supplied.get(name) or require_secret(name)
    return environment


def launch_controller(
    ledger: str,
    applet_id: str,
    once: bool,
    providers: frozenset[Solver],
    generation: str,
    budget_gpu_hours: float,
    secrets_env: Path | None = None,
) -> None:
    """Submit one CPU parent; its inherited client owns every remediation child."""
    for job in list_iris_jobs():
        if CONTROLLER_JOB_PREFIX in job.job_id and job.active:
            print(json.dumps({"controller_job": job.job_id, "state": job.state}))
            return
    record = load_credentials("marin")
    if record is None or not record.edge_refresh_token:
        raise ValueError("Controller needs a refreshable Marina login; run iris login")
    environment = controller_environment(secrets_env)
    environment[REFRESH_TOKEN_ENV] = record.edge_refresh_token
    command = cpu_job_command(CONTROLLER_JOB_PREFIX + generation[:16])
    # Task tokens can delegate an elevated child profile only when the parent holds it.
    command.extend(["--container-profile", job_pb2.ContainerProfile.Name(job_pb2.CONTAINER_PROFILE_PRIVILEGED)])
    command.extend(["--egress-policy", job_pb2.EgressPolicy.Name(job_pb2.EGRESS_POLICY_CLUSTER)])
    command.extend(["--secret", "MARINA_IAP_REFRESH_TOKEN", "env:" + REFRESH_TOKEN_ENV])
    for name in CONTROLLER_SECRET_NAMES:
        command.extend(["--secret", name, "env:" + name])
    command.extend(
        [
            "--secret",
            "DAYTONA_RL_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_RL_API_KEY/versions/1",
            "--",
            "python",
            "-m",
            "experiments.post_training.task_curation.difficulty",
            "reconcile",
            "--ledger",
            ledger,
            "--applet-id",
            applet_id,
            "--gpu-hour-budget",
            str(budget_gpu_hours),
        ]
    )
    if once:
        command.append("--once")
    for solver in sorted(providers):
        command.extend(["--provider-available", solver.value])
    subprocess.run(command, check=True, env=environment)
