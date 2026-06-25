# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX distributed initialization via Iris endpoint registry.

Task 0 registers its coordinator address; tasks 1..N-1 poll for it.
Single-task jobs skip distributed init entirely — JAX defaults suffice.

JAX is imported at call time — iris does not depend on jax.
"""

import atexit
import logging
import os
import time

from rigging.filesystem import marin_prefix
from rigging.timing import Deadline, Duration, ExponentialBackoff

from iris.actor.resolver import Resolver
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info

logger = logging.getLogger(__name__)

_COMPILATION_CACHE_SUBDIR = "compilation-cache"

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


def configure_jax_compilation_cache() -> None:
    """Default the JAX compilation cache to a subdir of the active Marin prefix.

    Without a cache dir, every process re-runs XLA compilation and kernel
    autotune sweeps at startup. An explicit setting (``JAX_COMPILATION_CACHE_DIR``
    or ``jax.config``) wins; otherwise the cache lands under the cluster's
    region-local storage prefix, which may be a ``gs://``/``s3://`` URL that JAX
    writes directly.
    """
    import jax  # noqa: PLC0415  # optional dep: jax (iris does not depend on jax)

    if os.environ.get("JAX_COMPILATION_CACHE_DIR") or jax.config.jax_compilation_cache_dir:
        return

    cache_dir = f"{marin_prefix().rstrip('/')}/{_COMPILATION_CACHE_SUBDIR}"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    logger.info("JAX compilation cache: %s", cache_dir)


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
    backoff = ExponentialBackoff(initial=poll_interval, maximum=max(poll_interval, 30.0))
    deadline = Deadline.from_now(Duration.from_seconds(timeout))
    while True:
        resolved = resolver.resolve(endpoint_name)
        if not resolved.is_empty:
            return resolved.first().url
        if deadline.expired():
            raise TimeoutError(f"Timed out after {timeout}s waiting for coordinator endpoint '{endpoint_name}'")
        interval = min(backoff.next_interval(), deadline.remaining_seconds())
        if interval > 0:
            time.sleep(interval)


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

    For single-task jobs (or when not running inside an Iris job),
    initialization is skipped — JAX works correctly without distributed
    init when there is only one process.

    On TPU, JAX handles coordinator discovery via the TPU runtime, so this
    function calls ``jax.distributed.initialize()`` with no arguments and
    returns — the TPU runtime supplies all necessary addresses automatically.

    Args:
        port: Coordinator port. Overridden by IRIS_PORT_jax if allocated.
            An explicit port is required because JAX's gRPC coordinator binds
            internally and does not expose the actual bound port.
        endpoint_name: Name under which the coordinator registers.
        poll_timeout: Maximum seconds for non-coordinator tasks to wait.
        poll_interval: Initial backoff delay for polling (seconds).
    """
    import jax  # noqa: PLC0415  # optional dep: jax (iris does not depend on jax)

    # Configure the compilation cache before any compile happens, on every
    # distributed-init path below (TPU, single-task, or the endpoint dance).
    configure_jax_compilation_cache()

    # Idempotent: skip if jax.distributed has already been initialized. This
    # lets a caller that must touch JAX before levanter.initialize (e.g. via
    # `hax.named` → `jnp.asarray` while building loss-config args) call
    # `initialize_jax()` explicitly first; levanter's later call then lands
    # here as a no-op instead of hitting JAX 0.9+'s
    # `xla_bridge.backends_are_initialized()` guard, which raises on a second
    # `jax.distributed.initialize()`. Note this only covers a prior *initialize*
    # call — merely materializing a JAX array initializes the XLA backend, not
    # jax.distributed, so `is_initialized()` stays False in that case.
    if jax.distributed.is_initialized():
        logger.info("jax.distributed already initialized; skipping")
        return

    # TPU has its own coordinator discovery via the TPU runtime, so avoid the
    # Iris endpoint dance. We still call JAX distributed initialization to
    # create the host-side distributed client used by Levanter multihost
    # utilities. levanter.distributed delegates here when running under Iris
    # (see lib/levanter/src/levanter/distributed.py initialize_distributed),
    # so this is the single init site on the Iris+TPU path.
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU" or os.environ.get("JAX_PLATFORMS", "") == "tpu":
        logger.info("TPU detected; initializing JAX distributed via TPU runtime autodiscovery")
        jax.distributed.initialize()
        return

    job_info = get_job_info()
    _log_jax_bootstrap_inputs(job_info, port=port, endpoint_name=endpoint_name)
    if job_info is None:
        return

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
        endpoint_id = ctx.registry.register(endpoint_name, coordinator)
        # Best-effort cleanup: if the process crashes, the controller's
        # cascade delete on task cleanup handles endpoint removal.
        atexit.register(ctx.registry.unregister, endpoint_id)
        jax.distributed.initialize(coordinator, job_info.num_tasks, task_index)
    else:
        coordinator = _poll_for_coordinator(ctx.resolver, endpoint_name, poll_timeout, poll_interval)
        jax.distributed.initialize(coordinator, job_info.num_tasks, task_index)
