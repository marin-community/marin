# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX distributed initialization via Iris endpoint registry.

Task 0 registers its coordinator address; tasks 1..N-1 poll for it.
Single-task jobs initialize explicitly as a one-process distributed runtime.

JAX is imported at call time — iris does not depend on jax.
"""

from __future__ import annotations

import atexit
import logging
import os

from iris.actor.resolver import Resolver
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from rigging.timing import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

_JAX_ENV_KEYS = (
    "IRIS_TASK_ID",
    "IRIS_NUM_TASKS",
    "IRIS_PORT_jax",
    "JAX_COORDINATOR_ADDRESS",
    "JAX_COORDINATOR_BIND_ADDRESS",
    "JAX_LOCAL_DEVICE_IDS",
    "JAX_PROCESS_COUNT",
    "JAX_PROCESS_INDEX",
)


def _log_jax_bootstrap_inputs(job_info, *, port: int, endpoint_name: str) -> None:
    env_snapshot = {key: os.environ.get(key, "") for key in _JAX_ENV_KEYS if key in os.environ}
    if job_info is None:
        logger.info(
            "initialize_jax bootstrap inputs: job_info=None endpoint_name=%s port=%s env=%s",
            endpoint_name,
            port,
            env_snapshot or "none",
        )
        return

    logger.info(
        "initialize_jax bootstrap inputs: task_index=%s num_tasks=%s advertise_host=%s ports=%s endpoint_name=%s "
        "requested_port=%s env=%s",
        job_info.task_index,
        job_info.num_tasks,
        job_info.advertise_host,
        dict(job_info.ports),
        endpoint_name,
        port,
        env_snapshot or "none",
    )


def _scoped_endpoint_name(job_info: JobInfo, endpoint_name: str) -> str:
    """Scope the coordinator endpoint to this Iris job within the root namespace."""
    normalized_endpoint_name = endpoint_name.strip("/")
    return f"{normalized_endpoint_name}_{job_info.job_id.to_safe_token()}"


def _poll_for_coordinator(
    resolver: Resolver,
    endpoint_name: str,
    timeout: float,
    poll_interval: float,
) -> str:
    """Poll the endpoint registry until the coordinator address appears.

    Args:
        resolver: Namespaced resolver for this job.
        endpoint_name: Name of the coordinator endpoint.
        timeout: Maximum seconds to wait.
        poll_interval: Initial backoff delay in seconds.

    Returns:
        The coordinator address string (host:port).

    Raises:
        TimeoutError: If the coordinator is not found within timeout.
    """
    result: list[str] = []

    def _check() -> bool:
        resolved = resolver.resolve(endpoint_name)
        if not resolved.is_empty:
            result.append(resolved.first().url)
            return True
        return False

    backoff = ExponentialBackoff(initial=poll_interval, maximum=max(poll_interval, 30.0))
    backoff.wait_until_or_raise(
        _check,
        timeout=Duration.from_seconds(timeout),
        error_message=f"Timed out after {timeout}s waiting for coordinator endpoint '{endpoint_name}'",
    )
    return result[0]


def initialize_jax(
    port: int = 8476,
    endpoint_name: str = "jax_coordinator",
    poll_timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> None:
    """Initialize JAX distributed runtime using Iris endpoint discovery.

    For multi-task GPU jobs, task 0 registers its coordinator address via the
    Iris endpoint registry, and tasks 1..N-1 poll until they discover it. All
    tasks then call jax.distributed.initialize with the coordinator address.

    For single-task Iris jobs, initialization uses a one-process coordinator
    so JAX subsystems that require distributed initialization, such as async
    checkpointing, can still start consistently. Jobs outside Iris are skipped.

    Args:
        port: Coordinator port. Overridden by IRIS_PORT_jax if allocated.
            An explicit port is required because JAX's gRPC coordinator binds
            internally and does not expose the actual bound port.
        endpoint_name: Name under which the coordinator registers.
        poll_timeout: Maximum seconds for non-coordinator tasks to wait.
        poll_interval: Initial backoff delay for polling (seconds).
    """
    import jax

    job_info = get_job_info()
    if job_info is None:
        _log_jax_bootstrap_inputs(job_info, port=port, endpoint_name=endpoint_name)
        return

    scoped_endpoint_name = _scoped_endpoint_name(job_info, endpoint_name)
    _log_jax_bootstrap_inputs(job_info, port=port, endpoint_name=scoped_endpoint_name)

    if job_info.num_tasks <= 1:
        bound_port = job_info.ports.get("jax", port)
        coordinator = f"{job_info.advertise_host}:{bound_port}"
        jax.distributed.initialize(coordinator, num_processes=1, process_id=0)
        return

    ctx = iris_ctx()
    task_index = job_info.task_index

    if task_index == 0:
        bound_port = job_info.ports.get("jax", port)
        coordinator = f"{job_info.advertise_host}:{bound_port}"
        # Register the endpoint first so other tasks can discover the
        # coordinator address. jax.distributed.initialize() blocks until
        # all processes connect, so registering after would deadlock.
        # JAX's internal gRPC retry handles the brief window between
        # endpoint registration and the coordinator starting to listen.
        endpoint_id = ctx.registry.register(scoped_endpoint_name, coordinator)
        # Best-effort cleanup: if the process crashes, the controller's
        # cascade delete on task cleanup handles endpoint removal.
        atexit.register(ctx.registry.unregister, endpoint_id)
        jax.distributed.initialize(coordinator, job_info.num_tasks, task_index)
    else:
        coordinator = _poll_for_coordinator(ctx.resolver, scoped_endpoint_name, poll_timeout, poll_interval)
        jax.distributed.initialize(coordinator, job_info.num_tasks, task_index)
