# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Megascale environment setup for Levanter on Iris."""

from __future__ import annotations

import atexit
import logging
import os

from rigging.timing import Duration, ExponentialBackoff

from iris.client.client import iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info

logger = logging.getLogger(__name__)

IRIS_SLICE_COUNT = "IRIS_SLICE_COUNT"
IRIS_TASKS_PER_SLICE = "IRIS_TASKS_PER_SLICE"

MEGASCALE_COORDINATOR_ENDPOINT = "megascale_coordinator"
MEGASCALE_READY_ENDPOINT_PREFIX = "megascale_task_ready_"
MEGASCALE_COORDINATOR_PORT = 8081
MEGASCALE_COORDINATOR_TIMEOUT = 3600.0
MEGASCALE_COORDINATOR_POLL_INTERVAL = 2.0

MEGASCALE_COORDINATOR_ADDRESS = "MEGASCALE_COORDINATOR_ADDRESS"
MEGASCALE_NUM_SLICES = "MEGASCALE_NUM_SLICES"
MEGASCALE_PORT = "MEGASCALE_PORT"
MEGASCALE_SLICE_ID = "MEGASCALE_SLICE_ID"

LEVANTER_MEGASCALE_PORT = "LEVANTER_MEGASCALE_PORT"
LEVANTER_MEGASCALE_ENDPOINT_NAME = "LEVANTER_MEGASCALE_ENDPOINT_NAME"
LEVANTER_MEGASCALE_TIMEOUT = "LEVANTER_MEGASCALE_TIMEOUT"
LEVANTER_MEGASCALE_POLL_INTERVAL = "LEVANTER_MEGASCALE_POLL_INTERVAL"


def _wait_for_all_tasks_ready(
    job_info: JobInfo,
    *,
    timeout: float,
    poll_interval: float,
) -> None:
    ctx = iris_ctx()
    endpoint_name = f"{MEGASCALE_READY_ENDPOINT_PREFIX}{job_info.task_index}"
    endpoint_id = ctx.registry.register(endpoint_name, job_info.advertise_host)
    atexit.register(ctx.registry.unregister, endpoint_id)

    def check_tasks_ready() -> bool:
        missing = [
            task_index
            for task_index in range(job_info.num_tasks)
            if ctx.resolver.resolve(f"{MEGASCALE_READY_ENDPOINT_PREFIX}{task_index}").is_empty
        ]
        if missing:
            logger.info("Waiting for Iris tasks before Megascale init; missing task indexes: %s", missing)
            return False
        return True

    backoff = ExponentialBackoff(initial=poll_interval, maximum=max(poll_interval, 30.0))
    backoff.wait_until_or_raise(
        check_tasks_ready,
        timeout=Duration.from_seconds(timeout),
        error_message=f"Timed out after {timeout}s waiting for all Iris tasks before Megascale init",
    )


def _coordinator_address(
    job_info: JobInfo,
    *,
    port: int,
    endpoint_name: str,
    timeout: float,
    poll_interval: float,
) -> str:
    ctx = iris_ctx()

    if job_info.task_index == 0:
        coordinator = f"{job_info.advertise_host}:{port}"
        endpoint_id = ctx.registry.register(endpoint_name, coordinator)
        atexit.register(ctx.registry.unregister, endpoint_id)
        return coordinator

    result: list[str] = []

    def check_coordinator() -> bool:
        resolved = ctx.resolver.resolve(endpoint_name)
        if resolved.is_empty:
            return False
        result.append(resolved.first().url)
        return True

    backoff = ExponentialBackoff(initial=poll_interval, maximum=max(poll_interval, 30.0))
    backoff.wait_until_or_raise(
        check_coordinator,
        timeout=Duration.from_seconds(timeout),
        error_message=f"Timed out after {timeout}s waiting for Megascale coordinator endpoint {endpoint_name!r}",
    )
    return result[0]


def megascale_env_for_iris_task(
    *,
    slice_count: int,
    tasks_per_slice: int,
    port: int = MEGASCALE_COORDINATOR_PORT,
    endpoint_name: str = MEGASCALE_COORDINATOR_ENDPOINT,
    timeout: float = MEGASCALE_COORDINATOR_TIMEOUT,
    poll_interval: float = MEGASCALE_COORDINATOR_POLL_INTERVAL,
) -> dict[str, str]:
    """Return Megascale env vars for the current Levanter Iris task."""
    if slice_count <= 0:
        raise ValueError(f"slice_count must be positive, got {slice_count}")
    if tasks_per_slice <= 0:
        raise ValueError(f"tasks_per_slice must be positive, got {tasks_per_slice}")
    if slice_count <= 1:
        return {}

    expected_tasks = slice_count * tasks_per_slice
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("Megascale multislice setup requires running inside an Iris task")
    if job_info.num_tasks != expected_tasks:
        raise ValueError(f"Iris launched {job_info.num_tasks} tasks, but Megascale expects {expected_tasks}")
    if job_info.task_index >= job_info.num_tasks:
        raise ValueError(f"Iris task index {job_info.task_index} is outside num_tasks={job_info.num_tasks}")

    slice_id = job_info.task_index // tasks_per_slice
    _wait_for_all_tasks_ready(job_info, timeout=timeout, poll_interval=poll_interval)
    return {
        MEGASCALE_COORDINATOR_ADDRESS: _coordinator_address(
            job_info,
            port=port,
            endpoint_name=endpoint_name,
            timeout=timeout,
            poll_interval=poll_interval,
        ),
        MEGASCALE_NUM_SLICES: str(slice_count),
        MEGASCALE_PORT: str(port),
        MEGASCALE_SLICE_ID: str(slice_id),
    }


def configure_megascale_from_iris() -> dict[str, str]:
    """Configure Megascale env vars from generic Iris slice topology env vars."""
    slice_count_raw = os.environ.get(IRIS_SLICE_COUNT)
    if not slice_count_raw:
        return {}

    tasks_per_slice_raw = os.environ.get(IRIS_TASKS_PER_SLICE, "1")
    port_raw = os.environ.get(LEVANTER_MEGASCALE_PORT, str(MEGASCALE_COORDINATOR_PORT))
    timeout_raw = os.environ.get(LEVANTER_MEGASCALE_TIMEOUT, str(MEGASCALE_COORDINATOR_TIMEOUT))
    poll_interval_raw = os.environ.get(LEVANTER_MEGASCALE_POLL_INTERVAL, str(MEGASCALE_COORDINATOR_POLL_INTERVAL))

    env = megascale_env_for_iris_task(
        slice_count=int(slice_count_raw),
        tasks_per_slice=int(tasks_per_slice_raw),
        port=int(port_raw),
        endpoint_name=os.environ.get(LEVANTER_MEGASCALE_ENDPOINT_NAME, MEGASCALE_COORDINATOR_ENDPOINT),
        timeout=float(timeout_raw),
        poll_interval=float(poll_interval_raw),
    )
    os.environ.update(env)
    if env:
        logger.info("Configured Megascale env for Levanter from Iris slice topology: %s", sorted(env))
    return env
