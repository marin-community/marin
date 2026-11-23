# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Watchdog for the GLM synthesis serving pool.

The pool's health is the registered endpoint answering ``GET /v1/models``. Engine
deaths are already handled by Iris retries and the broker's lease machinery; this
watchdog covers what they cannot: a coordinator that exited (its 3-day timeout, a
fatal error, a node loss) leaves the endpoint unregistered with no one to rebuild it.

Each watchdog generation:

1. probes the endpoint; short outages are tolerated (coordinator rolls, engine
   restarts, scheduling), sustained ones are not;
2. on sustained unhealthiness with no live coordinator job, submits a fresh
   coordinator and keeps watching until it registers;
3. before its own task timeout, hands off by submitting the next generation.

Runs inside an Iris job on the pool cluster so endpoint resolution and job
submission use the in-cluster controller client.
"""

import logging
import time
import uuid

import requests
from iris.client.client import iris_ctx
from iris.cluster.types import Entrypoint
from rigging.timing import Duration

from experiments.glm_synthesis_serving import pool
from experiments.glm_synthesis_serving.serve_pool import serve

logger = logging.getLogger(__name__)

WATCHDOG_GENERATION_HOURS = 23.0
_PROBE_INTERVAL_SECONDS = 60.0
# Consecutive failed probes before treating the endpoint as down. A coordinator
# roll re-registers within ~35 minutes (weights stream in-region), so an hour of
# failures means nobody is rebuilding it.
_FAILURES_BEFORE_RESUBMIT = 60
_RESUBMIT_COOLDOWN_SECONDS = 30 * 60.0


def _client():
    client = iris_ctx().client
    if client is None:
        raise RuntimeError("pool_watchdog must run inside an Iris job with a controller client")
    return client


def _endpoint_healthy() -> bool:
    client = _client()
    try:
        if not client.list_endpoint_instances(pool.ENDPOINT_NAME):
            return False
        base_url = client.resolve_endpoint(pool.ENDPOINT_NAME)
    except Exception:
        logger.warning("endpoint resolution failed", exc_info=True)
        return False
    try:
        response = requests.get(base_url.rstrip("/") + "/models", timeout=30)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _live_coordinator_count() -> int:
    client = _client()
    prefix = f"/{pool.JOB_PREFIX}-"
    return sum(
        1
        for job_id in client.active_job_names_for_prefix(prefix)
        if "-watch-" not in str(job_id) and "-stage-" not in str(job_id)
    )


def _submit_coordinator() -> str:
    """Submit a fresh coordinator from inside this job; return its job id."""
    job = _client().submit(
        entrypoint=Entrypoint.from_callable(serve, pool.POOL_INSTANCES),
        name=f"{pool.JOB_PREFIX}-{uuid.uuid4().hex[:6]}",
        resources=pool.coordinator_resources(),
        priority_band=pool.PRIORITY,
        ports=["http"],
        timeout=Duration.from_hours(pool.COORDINATOR_TIMEOUT_HOURS),
    )
    return str(job.job_id)


def _submit_next_generation() -> str:
    """Submit the next watchdog generation so coverage continues across task timeouts."""
    job = _client().submit(
        entrypoint=watch_forever,
        name=f"{pool.JOB_PREFIX}-watch-{uuid.uuid4().hex[:6]}",
        resources=pool.coordinator_resources(),
        priority_band=pool.PRIORITY,
        timeout=Duration.from_hours(WATCHDOG_GENERATION_HOURS),
    )
    return str(job.job_id)


def watch_forever() -> None:
    """Keep the pool endpoint alive for WATCHDOG_GENERATION_HOURS, then hand off."""
    failures = 0
    last_resubmit = 0.0
    generation_deadline = time.monotonic() + WATCHDOG_GENERATION_HOURS * 3600
    while time.monotonic() < generation_deadline:
        if _endpoint_healthy():
            failures = 0
            time.sleep(_PROBE_INTERVAL_SECONDS)
            continue
        failures += 1
        if failures < _FAILURES_BEFORE_RESUBMIT:
            logger.warning("endpoint unhealthy (%d consecutive failures)", failures)
            time.sleep(_PROBE_INTERVAL_SECONDS)
            continue
        if time.monotonic() - last_resubmit < _RESUBMIT_COOLDOWN_SECONDS:
            time.sleep(_PROBE_INTERVAL_SECONDS)
            continue
        live = _live_coordinator_count()
        if live:
            logger.warning("endpoint down but %d coordinator job(s) live; waiting for registration", live)
            time.sleep(_PROBE_INTERVAL_SECONDS)
            continue
        job_id = _submit_coordinator()
        last_resubmit = time.monotonic()
        failures = 0
        logger.warning("endpoint down; resubmitted coordinator %s", job_id)
    _submit_next_generation()
