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
from enum import StrEnum

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.filesystem import marin_prefix
from rigging.timing import Deadline, Duration, ExponentialBackoff

from iris.actor.resolver import Resolver
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.hooks.multigpu import (
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
)
from iris.runtime import telltale

logger = logging.getLogger(__name__)

_COMPILATION_CACHE_SUBDIR = "compilation-cache"
# JAX's RegisterTask barrier defaults to 300s. On a large gang (e.g. v5p-64 = 8 hosts) a
# preemption-driven cold restart can have a random subset of hosts still doing uv-sync/import/
# GCS-read setup past 300s, so the already-registered hosts hit DEADLINE_EXCEEDED and abort the
# whole gang. Give cold gang-init more slack; a longer timeout only affects how long a
# genuinely-stuck init waits.
_JAX_DIST_INIT_TIMEOUT = 1800

_JAX_ENV_KEYS = (
    "IRIS_TASK_ID",
    "IRIS_NUM_TASKS",
    "IRIS_PORT_jax",
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    "JAX_COORDINATOR_ADDRESS",
    "JAX_COORDINATOR_BIND_ADDRESS",
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
    region-local storage prefix, which may be a ``gs://``/``s3://`` URL.

    JAX's own compilation cache handles a remote URL fine (it reads/writes via
    fsspec), but it also derives XLA's C++ per-fusion autotune cache path from
    the same directory and hands it to XLA's `tsl::Env`, which has no
    ``gs://``/``s3://`` support and crashes the first compile with
    "UNIMPLEMENTED: File system scheme ... not implemented". Disable that
    derivation when the cache dir is remote; the autotune cache is ephemeral
    (skipped entirely on a compilation-cache hit), so losing it costs at most
    a few minutes on a cold compile.
    """
    import jax  # noqa: PLC0415  # optional dep: jax (iris does not depend on jax)

    if os.environ.get("JAX_COMPILATION_CACHE_DIR") or jax.config.jax_compilation_cache_dir:
        return

    cache_dir = f"{marin_prefix().rstrip('/')}/{_COMPILATION_CACHE_SUBDIR}"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    logger.info("JAX compilation cache: %s", cache_dir)

    if "://" in cache_dir and "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES" not in os.environ:
        jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
        logger.info("XLA autotune sub-cache disabled (compilation cache is remote: %s)", cache_dir)


# An endpoint name that has not been registered yet surfaces differently across
# controller versions: an empty result, a Connect NOT_FOUND, or — on controllers
# that answer the lookup with a bare HTTP 404 — a Connect UNIMPLEMENTED. A
# coordinator that has not come up is the expected state while polling, as is a
# transiently unreachable controller (UNAVAILABLE), so all are retried. Other
# Connect errors are genuine and propagate.
_COORDINATOR_PENDING_CODES = frozenset({Code.NOT_FOUND, Code.UNIMPLEMENTED, Code.UNAVAILABLE})


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
        try:
            resolved = resolver.resolve(endpoint_name)
            if not resolved.is_empty:
                return resolved.first().url
        except ConnectError as e:
            if e.code not in _COORDINATOR_PENDING_CODES:
                raise
        if deadline.expired():
            raise TimeoutError(f"Timed out after {timeout}s waiting for coordinator endpoint '{endpoint_name}'")
        interval = min(backoff.next_interval(), deadline.remaining_seconds())
        if interval > 0:
            time.sleep(interval)


def _parse_local_device_ids(raw: str | None) -> list[int] | None:
    """Parse an ``IRIS_MULTIGPU_LOCAL_DEVICE_IDS`` value ("0" or "2,3") into a list, or None."""
    if not raw:
        return None
    return [int(part) for part in raw.split(",") if part]


class _CoordinatorRole(StrEnum):
    """How a supervised rank obtains the JAX coordinator address."""

    REGISTER = "register"  # global rank 0 on a multi-host job: bind + publish its address
    POLL = "poll"  # a rank on another host: discover rank 0's address via the registry
    REUSE_LOCAL = "reuse_local"  # rank 0 single-host, or a host-0 peer: use advertise_host directly


def _supervised_coordinator_role(proc_index: int, task_index: int, num_tasks: int) -> _CoordinatorRole:
    """Pick the coordinator-discovery role for a supervised rank.

    Global rank 0 (``proc_index == 0``) owns the coordinator: it publishes its
    address only when the job spans multiple hosts (otherwise no peer needs to
    discover it). Other ranks on host 0 reuse that same advertise_host directly,
    with no registry round-trip. Ranks on other hosts poll for rank 0's address.
    """
    if proc_index == 0:
        return _CoordinatorRole.REGISTER if num_tasks > 1 else _CoordinatorRole.REUSE_LOCAL
    if task_index == 0:
        return _CoordinatorRole.REUSE_LOCAL
    return _CoordinatorRole.POLL


def _initialize_supervised_jax(
    jax, job_info, *, port: int, endpoint_name: str, poll_timeout: float, poll_interval: float
) -> None:
    """Join the JAX mesh for a process launched by ``iris.hooks.multigpu_main``.

    The supervisor runs N JAX processes inside one Iris task and stamps each
    child with its global rank (``IRIS_MULTIGPU_PROCESS_INDEX``), the global world
    size (``IRIS_MULTIGPU_PROCESS_COUNT``), and the device ids it owns
    (``IRIS_MULTIGPU_LOCAL_DEVICE_IDS``). Coordinator discovery reuses the
    one-process-per-task endpoint dance, lifted from per-task to per-global-rank
    so every process across every host joins one mesh.
    """
    proc_count = int(os.environ[IRIS_MULTIGPU_PROCESS_COUNT_ENV])
    proc_index = int(os.environ[IRIS_MULTIGPU_PROCESS_INDEX_ENV])
    device_ids = _parse_local_device_ids(os.environ.get(IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV))

    if proc_count <= 1:
        jax.distributed.initialize(num_processes=1, process_id=0, local_device_ids=device_ids)
        return

    num_tasks = job_info.num_tasks if job_info else 1
    task_index = job_info.task_index if job_info else 0
    advertise_host = job_info.advertise_host if job_info else "127.0.0.1"
    bound_port = job_info.ports.get("jax", port) if job_info else port
    coordinator = f"{advertise_host}:{bound_port}"

    role = _supervised_coordinator_role(proc_index, task_index, num_tasks)
    if role is _CoordinatorRole.POLL:
        ctx = iris_ctx()
        coordinator = _poll_for_coordinator(ctx.resolver, endpoint_name, poll_timeout, poll_interval)
    elif role is _CoordinatorRole.REGISTER:
        ctx = iris_ctx()
        endpoint_id = ctx.registry.register(endpoint_name, coordinator)
        atexit.register(ctx.registry.unregister, endpoint_id)

    logger.info(
        "initialize_jax (supervised): process_id=%d/%d local_device_ids=%s coordinator=%s",
        proc_index,
        proc_count,
        device_ids,
        coordinator,
    )
    jax.distributed.initialize(
        coordinator,
        proc_count,
        proc_index,
        local_device_ids=device_ids,
        initialization_timeout=_JAX_DIST_INIT_TIMEOUT,
    )


def initialize_jax(
    port: int = 8476,
    endpoint_name: str = "jax_coordinator",
    poll_timeout: float = _JAX_DIST_INIT_TIMEOUT,
    poll_interval: float = 2.0,
) -> None:
    """Initialize JAX distributed runtime using Iris endpoint discovery.

    For multi-task GPU jobs, task 0 registers its coordinator address via the
    Iris endpoint registry, and tasks 1..N-1 poll until they discover it. All
    tasks then call jax.distributed.initialize with the coordinator address.

    For single-task jobs (or when not running inside an Iris job),
    initialization is skipped — JAX works correctly without distributed
    init when there is only one process.

    Iris TPU jobs use the same endpoint-registry coordinator path as other Iris
    multi-task jobs.

    Args:
        port: Coordinator port. Overridden by IRIS_PORT_jax if allocated.
            An explicit port is required because JAX's gRPC coordinator binds
            internally and does not expose the actual bound port.
        endpoint_name: Name under which the coordinator registers.
        poll_timeout: Maximum seconds for non-coordinator tasks to wait for the
            coordinator endpoint to register. Defaults to ``_JAX_DIST_INIT_TIMEOUT``
            so a slow coordinator host on a large-gang cold restart does not abort
            the pollers before the JAX barrier itself gets its longer timeout.
        poll_interval: Initial backoff delay for polling (seconds).
    """
    import jax  # noqa: PLC0415  # optional dep: jax (iris does not depend on jax)

    # Configure the compilation cache before any compile happens, on every
    # distributed-init path below (TPU, single-task, or the endpoint dance).
    configure_jax_compilation_cache()

    # Start the telltale server to report stats for training jobs.
    telltale.start()

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

    job_info = get_job_info()
    _log_jax_bootstrap_inputs(job_info, port=port, endpoint_name=endpoint_name)

    # Supervised (multi-process-per-task) mode short-circuits the task-derived
    # paths: the multigpu supervisor has already assigned this process its global
    # rank, so the single/multi-task branches below (which assume one process
    # per task) do not apply. This runs even when job_info is None so a local
    # supervisor smoke can bring up a localhost mesh.
    if IRIS_MULTIGPU_PROCESS_COUNT_ENV in os.environ:
        _initialize_supervised_jax(
            jax,
            job_info,
            port=port,
            endpoint_name=endpoint_name,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )
        return

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
        jax.distributed.initialize(
            coordinator, job_info.num_tasks, task_index, initialization_timeout=_JAX_DIST_INIT_TIMEOUT
        )
    else:
        coordinator = _poll_for_coordinator(ctx.resolver, endpoint_name, poll_timeout, poll_interval)
        jax.distributed.initialize(
            coordinator, job_info.num_tasks, task_index, initialization_timeout=_JAX_DIST_INIT_TIMEOUT
        )
