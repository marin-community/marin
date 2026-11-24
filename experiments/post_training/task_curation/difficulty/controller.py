# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One watchdog-owned controller for all difficulty remediation children."""

import json
import logging
import os
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import dotenv_values
from finestore.rl.difficulty import Solver
from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT, IapRefreshTokenProvider
from rigging.credential_store import load_credentials
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret

from experiments.post_training.task_curation.difficulty.capacity import available_provider_solvers
from experiments.post_training.task_curation.difficulty.catalog import (
    available_artifact_uris,
    exclude_catalog_sources,
    served_catalog,
)
from experiments.post_training.task_curation.difficulty.executor import EscalationLog, collect_finished, reconcile_once
from experiments.post_training.task_curation.difficulty.iris_backend import IrisBackend
from experiments.post_training.task_curation.difficulty.launch import cpu_job_command
from experiments.post_training.task_curation.difficulty.ledger import read_rows
from experiments.post_training.task_curation.difficulty.report import cells_for_catalog
from experiments.post_training.task_curation.difficulty.runtime import solver_pool_record
from experiments.post_training.task_curation.difficulty.state import CoveragePolicy, GapReason, Outcome
from experiments.post_training.task_curation.difficulty_config import (
    SELF_HOSTED_SOLVERS,
    SOLVER_POOL_GPU_COUNT,
    TOGETHER_API_KEY_ENV,
)
from experiments.post_training.task_curation.difficulty_report import list_iris_jobs

LOGGER = logging.getLogger(__name__)
CONTROLLER_INTERVAL = 60
CONTROLLER_FAILURE_RETRIES = 3
CONTROLLER_JOB_PREFIX = "atlas-difficulty-reconciler-controller-"
REFRESH_TOKEN_ENV = "DIFFICULTY_MARINA_REFRESH"
CONTROLLER_SECRET_NAMES = (
    "CW_KEY_ID",
    "CW_KEY_SECRET",
    "HF_TOKEN",
    "GLM_BULK_TOKEN",
    "DAYTONA_DATA_API_KEY",
    # Skopeo authfile JSON for the private ghcr actor images (repair-lane GHCR
    # credential shape); LOCAL-backend attempt batches pull through it.
    "GHCR_REGISTRY_AUTHFILE",
)
# Provider keys live in Google Secret Manager, not the submitter's environment:
# the job bundle sources this reference directly, and every inheriting child
# resolves it through the same name.
TOGETHER_API_KEY_SECRET_REF = "gcp-secret://projects/hai-gcp-models/secrets/TOGETHER_API_KEY/versions/1"


def run_controller(
    ledger: str,
    applet_id: str,
    once: bool,
    providers: frozenset[Solver],
    budget_gpu_hours: float | None,
    serving_pool: str | None = None,
    *,
    excluded_sources: frozenset[str],
) -> None:
    context = iris_ctx()
    backend = IrisBackend(context.client, context.job_id, ledger, serving_pool)
    notifications = EscalationLog(ledger)
    policy = CoveragePolicy()
    provider = IapRefreshTokenProvider(
        MARIN_DESKTOP_OAUTH_CLIENT.client_id,
        MARIN_DESKTOP_OAUTH_CLIENT.client_secret,
        require_secret("MARINA_IAP_REFRESH_TOKEN"),
    )
    while serving_pool is not None and not all(
        (StoragePath(serving_pool) / f"{solver.value}.json").exists() for solver in SELF_HOSTED_SOLVERS
    ):
        LOGGER.info("DIFFICULTY_WAITING_FOR_MEASURED_SOLVER_POOLS %s", serving_pool)
        time.sleep(CONTROLLER_INTERVAL)
    if serving_pool is not None:
        actual_gpus = sum(solver_pool_record(solver, serving_pool)["gpu_count"] for solver in SELF_HOSTED_SOLVERS)
        if actual_gpus != SOLVER_POOL_GPU_COUNT:
            raise ValueError(f"Solver pools request {actual_gpus} GPUs; expected {SOLVER_POOL_GPU_COUNT}")
    reporting_started = datetime.now(UTC)
    if excluded_sources:
        LOGGER.warning("DIFFICULTY_SOURCE_QUARANTINE %s", json.dumps(sorted(excluded_sources)))
    while True:
        try:
            catalog = exclude_catalog_sources(
                served_catalog(applet_id, token_provider=provider, ledger=ledger), excluded_sources
            )
        except (requests.ConnectionError, requests.Timeout):
            LOGGER.exception("DIFFICULTY_CATALOG_UNAVAILABLE; retrying after %d seconds", CONTROLLER_INTERVAL)
            time.sleep(CONTROLLER_INTERVAL)
            continue
        rows = read_rows(ledger)
        cells = cells_for_catalog(rows, catalog, policy)
        available = available_artifact_uris(cells)
        waiting = any(gap.reason == GapReason.PROVIDER_CAPACITY_WAIT for cell in cells for gap in cell.gaps)
        capacity = (
            frozenset(Solver)
            if serving_pool
            else providers or (available_provider_solvers(ledger) if waiting else frozenset())
        )
        result = reconcile_once(
            ledger, catalog, policy, backend, notifications, capacity, available, budget_gpu_hours, serving_pool
        )
        LOGGER.info("DIFFICULTY_RECONCILED %s", json.dumps(result, sort_keys=True))
        now = datetime.now(UTC)
        scored = Counter(
            row.solver.value
            for row in read_rows(ledger)
            if row.outcome == Outcome.SCORED and row.recorded_at >= reporting_started.isoformat()
        )
        elapsed = (now - reporting_started).total_seconds()
        LOGGER.info(
            "DIFFICULTY_SCORED_RATE %s",
            json.dumps(
                {
                    "since": reporting_started.isoformat(),
                    "until": now.isoformat(),
                    "scored": dict(scored),
                    "scored_attempts": sum(scored.values()),
                    "scored_attempts_per_hour": sum(scored.values()) * 3600 / elapsed,
                    "elapsed_seconds": elapsed,
                },
                sort_keys=True,
            ),
        )
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
    budget_gpu_hours: float | None,
    secrets_env: Path | None = None,
    serving_pool: str | None = None,
    *,
    excluded_sources: frozenset[str],
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
    if serving_pool is not None:
        command[command.index("--target-cluster=cw-rno2a")] = "--target-cluster=cw-us-east-08a"
        command[command.index("--priority") + 1] = "production"
        command[command.index("--max-retries") + 1] = str(CONTROLLER_FAILURE_RETRIES)
        command[command.index("--memory") + 1] = "16GB"
        command[command.index("--disk") + 1] = "64GB"
    # Task tokens can delegate an elevated child profile only when the parent holds it.
    command.extend(["--container-profile", job_pb2.ContainerProfile.Name(job_pb2.CONTAINER_PROFILE_PRIVILEGED)])
    command.extend(["--egress-policy", job_pb2.EgressPolicy.Name(job_pb2.EGRESS_POLICY_CLUSTER)])
    command.extend(["--secret", "MARINA_IAP_REFRESH_TOKEN", "env:" + REFRESH_TOKEN_ENV])
    for name in CONTROLLER_SECRET_NAMES:
        command.extend(["--secret", name, "env:" + name])
    command.extend(["--secret", TOGETHER_API_KEY_ENV, TOGETHER_API_KEY_SECRET_REF])
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
        ]
    )
    command.extend(
        ["--unlimited-gpu-hours"] if budget_gpu_hours is None else ["--gpu-hour-budget", str(budget_gpu_hours)]
    )
    if serving_pool is not None:
        command.extend(["--serving-pool", serving_pool])
    for source in sorted(excluded_sources):
        command.extend(["--exclude-source", source])
    if once:
        command.append("--once")
    for solver in sorted(providers):
        command.extend(["--provider-available", solver.value])
    subprocess.run(command, check=True, env=environment)
