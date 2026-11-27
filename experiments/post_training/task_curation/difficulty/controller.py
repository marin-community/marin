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

import httpx
import requests
from connectrpc.errors import ConnectError
from dotenv import dotenv_values
from finestore.rl.difficulty import PROTOCOL_ID, Solver
from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT, IapRefreshTokenProvider
from rigging.credential_store import load_credentials
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret

from experiments.post_training.task_curation.difficulty.campaign import campaign_record
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
from experiments.post_training.task_curation.difficulty.protocols import require_protocol_activation
from experiments.post_training.task_curation.difficulty.report import cells_for_catalog
from experiments.post_training.task_curation.difficulty.runtime import solver_pool_connection, solver_pool_record
from experiments.post_training.task_curation.difficulty.state import CoveragePolicy, GapReason, Outcome, TerminalRuling
from experiments.post_training.task_curation.difficulty.tpu import CHIP_BUDGET, REGION, SHAPES
from experiments.post_training.task_curation.difficulty_config import (
    SELF_HOSTED_SOLVERS,
    SOLVER_POOL_GPU_COUNT,
    TOGETHER_API_KEY_ENV,
)
from experiments.post_training.task_curation.difficulty_output_budget import require_solver_protocol
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
    excluded_cpu_tpu_names: tuple[str, ...] = (),
    large_protocol_id: str = PROTOCOL_ID,
) -> None:
    require_protocol_activation(ledger, large_protocol_id)
    context = iris_ctx()
    backend = IrisBackend(context.client, context.job_id, ledger, serving_pool, excluded_cpu_tpu_names)
    notifications = EscalationLog(ledger)
    policy = CoveragePolicy(large_protocol_id=large_protocol_id)
    campaign = campaign_record(ledger)
    if campaign is not None:
        policy = CoveragePolicy(
            large_protocol_id=large_protocol_id,
            terminal_rulings=tuple(
                TerminalRuling(
                    hold["source_id"],
                    solver,
                    GapReason(hold["reason"]),
                    hold["registry_revision"],
                    campaign["parent_seal"]["uri"],
                )
                for hold in campaign["holds"]
                for solver in Solver
            ),
        )
    provider = IapRefreshTokenProvider(
        MARIN_DESKTOP_OAUTH_CLIENT.client_id,
        MARIN_DESKTOP_OAUTH_CLIENT.client_secret,
        require_secret("MARINA_IAP_REFRESH_TOKEN"),
    )
    while serving_pool is not None and campaign is None:
        if not all((StoragePath(serving_pool) / f"{solver.value}.json").exists() for solver in SELF_HOSTED_SOLVERS):
            LOGGER.info("DIFFICULTY_WAITING_FOR_MEASURED_SOLVER_POOLS %s", serving_pool)
            time.sleep(CONTROLLER_INTERVAL)
            continue
        pools = [solver_pool_record(solver, serving_pool) for solver in SELF_HOSTED_SOLVERS]
        if sum(pool["gpu_count"] for pool in pools) != SOLVER_POOL_GPU_COUNT:
            raise ValueError("GPU solver pool allocation differs from the protocol's 64 GPUs")
        break
    reporting_started = datetime.now(UTC)
    if excluded_sources:
        LOGGER.warning("DIFFICULTY_SOURCE_QUARANTINE %s", json.dumps(sorted(excluded_sources)))
    while True:
        qualified = qualified_tpu_solvers(serving_pool, policy) if campaign is not None and serving_pool else None
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
            qualified
            if qualified is not None
            else (
                frozenset(Solver)
                if serving_pool
                else providers or (available_provider_solvers(ledger) if waiting else frozenset())
            )
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


def _tpu_pool_is_live(pool: dict) -> bool:
    """A retained qualification receipt can outlive its owner's registered endpoint."""
    try:
        connection = solver_pool_connection(pool)
        response = httpx.get(connection.base_url.rstrip("/") + "/models", timeout=10)
        response.raise_for_status()
        if pool["model"] in {model["id"] for model in response.json()["data"]}:
            return True
        reason = "frozen_model_not_advertised"
    except (ConnectError, httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        reason = type(error).__name__
    LOGGER.warning("DIFFICULTY_POOL_UNAVAILABLE solver=%s reason=%s", pool["solver"], reason)
    return False


def qualified_tpu_solvers(registry: str, policy: CoveragePolicy = CoveragePolicy()) -> frozenset[Solver]:
    """Admit each solver only after its complete allocation publishes measured readiness."""
    available = {Solver.HOSTED}
    chips = 0
    for solver, shape in SHAPES.items():
        if not (StoragePath(registry) / f"{solver.value}.json").exists():
            continue
        pool = solver_pool_record(solver, registry)
        if pool.get("platform") != "tpu" or pool.get("region") != REGION:
            raise ValueError("TPU campaign has a non-regional or non-TPU solver pool")
        target = shape.chips * shape.replicas
        if pool["chip_count"] > target:
            raise ValueError(f"TPU solver pool exceeds its measured allocation: {solver}")
        chips += pool["chip_count"]
        if pool.get("solver_protocol_id", PROTOCOL_ID) != policy.solver_protocol_id(solver):
            LOGGER.warning("DIFFICULTY_POOL_PROTOCOL_MISMATCH solver=%s", solver)
            continue
        if pool["chip_count"] == target and _tpu_pool_is_live(pool):
            available.add(solver)
    if chips > CHIP_BUDGET:
        raise ValueError("TPU campaign exceeds the 256-chip allocation")
    LOGGER.info("DIFFICULTY_QUALIFIED_SOLVERS %s", json.dumps(sorted(available)))
    return frozenset(available)


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
    handover_from: str | None = None,
    excluded_cpu_tpu_names: tuple[str, ...] = (),
    large_protocol_id: str = PROTOCOL_ID,
) -> None:
    """Submit a controller, optionally handing over without canceling existing children."""
    name = CONTROLLER_JOB_PREFIX + generation[:16]
    require_solver_protocol(Solver.LARGE, large_protocol_id)
    require_protocol_activation(ledger, large_protocol_id)
    active = [
        job for job in list_iris_jobs() if job.job_id.rsplit("/", 1)[-1].startswith(CONTROLLER_JOB_PREFIX) and job.active
    ]
    if handover_from is not None:
        if not handover_from.rsplit("/", 1)[-1].startswith(CONTROLLER_JOB_PREFIX):
            raise ValueError("Handover requires a controller parent job ID")
        if any(job.job_id != handover_from and job.job_id.rsplit("/", 1)[-1] != name for job in active):
            raise ValueError("Handover refuses an unrelated active controller")
    same = next((job for job in active if job.job_id.rsplit("/", 1)[-1] == name), None)
    if same is not None:
        print(json.dumps({"controller_job": same.job_id, "state": same.state}))
        return
    if handover_from is not None:
        LOGGER.info(
            "DIFFICULTY_CONTROLLER_HANDOVER %s", json.dumps({"from": handover_from, "to": name, "ledger": ledger})
        )
    elif active:
        print(json.dumps({"controller_job": active[0].job_id, "state": active[0].state}))
        return
    record = load_credentials("marin")
    if record is None or not record.edge_refresh_token:
        raise ValueError("Controller needs a refreshable Marina login; run iris login")
    environment = controller_environment(secrets_env)
    environment[REFRESH_TOKEN_ENV] = record.edge_refresh_token
    command = cpu_job_command(name)
    if campaign_record(ledger) is not None:
        command[command.index("--timeout") + 1] = "0"
        command.remove("--target-cluster=cw-rno2a")
        command.extend(
            [
                "--region",
                "us-central2",
                "--zone",
                "us-central2-b",
                "--env-vars",
                "MARIN_TEMP_PREFIX",
                "gs://marin-us-central2",
            ]
        )
        command[command.index("--max-retries") + 1] = str(CONTROLLER_FAILURE_RETRIES)
        command[command.index("--memory") + 1] = "8GB"
        command[command.index("--disk") + 1] = "64GB"
    elif serving_pool is not None:
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
    if large_protocol_id != PROTOCOL_ID:
        command.extend(["--large-protocol-id", large_protocol_id])
    for source in sorted(excluded_sources):
        command.extend(["--exclude-source", source])
    for name in sorted(set(excluded_cpu_tpu_names)):
        command.extend(["--exclude-cpu-tpu-name", name])
    if once:
        command.append("--once")
    for solver in sorted(providers):
        command.extend(["--provider-available", solver.value])
    subprocess.run(command, check=True, env=environment)
