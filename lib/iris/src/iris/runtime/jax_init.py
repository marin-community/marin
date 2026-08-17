# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX distributed initialization via Iris endpoint registry.

Global process 0 registers its coordinator address; every other process polls
for it. Single-task jobs initialize an explicit one-process distributed world.

JAX is imported at call time — iris does not depend on jax.
"""

import atexit
import enum
import logging
import os
import time
from enum import StrEnum

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finestore.fileset import FineStoreDirectory, fetch_file_set
from rigging.filesystem.cluster_config import marin_prefix, marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import LAUNCH_PROVENANCE_ENV, launch_provenance
from rigging.timing import Deadline, Duration, ExponentialBackoff

from iris.actor.resolver import Resolver
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.cluster.platforms.types import find_free_port
from iris.cluster.runtime.env import SCRATCH_CACHE_PATH
from iris.env_resources import TaskResources
from iris.hooks.multigpu import (
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
)

logger = logging.getLogger(__name__)

_COMPILATION_CACHE_SUBDIR = "compilation-cache"
_XLA_AUTOTUNE_CACHE_SUBDIR = "xla/per-fusion-autotune"
_XLA_AUTOTUNE_CACHE_DIR_FLAG = "--xla_gpu_per_fusion_autotune_cache_dir"
XLA_AUTOTUNE_CACHE_MODE_ENV = "IRIS_XLA_AUTOTUNE_CACHE_MODE"
# Object-store home for the per-build FineStore file set.
_XLA_AUTOTUNE_REMOTE_PREFIX = "xla-per-fusion-autotune"
_XLA_AUTOTUNE_CACHE_TTL_DAYS = 30
# JAX's RegisterTask barrier defaults to 300s. On a large gang (e.g. v5p-64 = 8 hosts) a
# preemption-driven cold restart can have a random subset of hosts still doing uv-sync/import/
# GCS-read setup past 300s, so the already-registered hosts hit DEADLINE_EXCEEDED and abort the
# whole gang. Give cold gang-init more slack; a longer timeout only affects how long a
# genuinely-stuck init waits.
_JAX_DIST_INIT_TIMEOUT_DEFAULT = 1800
# The default is sized for a cold many-host gang. On a small gang that is being iterated on, a
# genuinely-stuck init costs the full 30 minutes before it reports anything, which dominates the
# debug cycle. Override it downward for those runs; leave it alone for production gangs.
JAX_DIST_INIT_TIMEOUT_ENV = "IRIS_JAX_INIT_TIMEOUT"


def _ipv4_bind_address(coordinator_address: str) -> str:
    """Bind the coordinator on IPv4-any at the coordinator's own port.

    JAX defaults to ``[::]:<port>``. On a host whose ``bindv6only`` is set, an IPv6-any listener
    does not accept IPv4 connections, so every peer dialing the task's routable IPv4 address gets
    ECONNREFUSED and the gang never forms. Binding ``0.0.0.0`` fixes that, but only with the port
    attached: ``JAX_COORDINATOR_BIND_ADDRESS=0.0.0.0`` replaces JAX's default wholesale, leaving
    the service on an arbitrary port while peers still dial the advertised one, which hangs every
    client in ``connect()`` until the init timeout.
    """
    port = coordinator_address.rsplit(":", 1)[1]
    return f"0.0.0.0:{port}"


def _jax_dist_init_timeout() -> int:
    raw = os.environ.get(JAX_DIST_INIT_TIMEOUT_ENV)
    if raw is None:
        return _JAX_DIST_INIT_TIMEOUT_DEFAULT
    timeout = int(raw)
    if timeout <= 0:
        raise ValueError(f"{JAX_DIST_INIT_TIMEOUT_ENV} must be positive, got {timeout}")
    return timeout


# Pin the coordination heartbeat timeout instead of inheriting JAX's default.
# This bounds how long healthy ranks wait after a peer process disappears before
# JAX fate sharing tears the distributed world down and Iris can retry the gang.
_JAX_DIST_HEARTBEAT_TIMEOUT = 100


class _AutotuneCacheRole(enum.StrEnum):
    UPLOADER = "uploader"
    FETCHER = "fetcher"
    NONE = "none"


_JAX_ENV_KEYS = (
    "IRIS_TASK_ID",
    "IRIS_NUM_TASKS",
    "IRIS_PORT_JAX",
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    "JAX_COORDINATOR_ADDRESS",
    "JAX_COORDINATOR_BIND_ADDRESS",
)


class XlaAutotuneCacheMode(StrEnum):
    """Persistence policy for XLA's node-local per-fusion autotune cache."""

    REMOTE_SYNC = "remote_sync"
    LOCAL_ONLY = "local_only"


def resolve_coordinator_port(job_info: JobInfo, explicit: int | None = None) -> int:
    """Resolve the JAX coordinator port from Iris, the caller, or the kernel."""
    allocated = job_info.ports.get("jax", 0)
    if allocated:
        logger.info("JAX coordinator port %d (assigned as IRIS_PORT_JAX)", allocated)
        return allocated
    if explicit is not None:
        logger.info("JAX coordinator port %d (pinned by the caller)", explicit)
        return explicit
    port = find_free_port()
    logger.info("JAX coordinator port %d (selected by the kernel)", port)
    return port


def _log_jax_bootstrap_inputs(job_info, *, port: int | None, endpoint_name: str) -> None:
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
    """Place JAX's compilation cache on object storage and XLA's autotune cache on the node.

    The JAX cache defaults to a subdirectory of the active Marin prefix, unless
    ``JAX_COMPILATION_CACHE_DIR`` or ``jax.config`` already names one. It has to
    be somewhere every process can read: JAX writes it only from process 0, so a
    node-local copy would leave every other node cold.

    A remote JAX cache additionally disables JAX's XLA sub-cache derivation,
    which would otherwise hand XLA's C++ filesystem layer a URL it cannot open,
    and redirects XLA's per-fusion autotune cache to node-local disk instead.
    """
    import jax  # noqa: PLC0415  # optional dep: jax (iris does not depend on jax)

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or jax.config.jax_compilation_cache_dir
    if not cache_dir:
        cache_dir = prefix_join(marin_prefix(), _COMPILATION_CACHE_SUBDIR)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        jax.config.update("jax_compilation_cache_dir", cache_dir)
    logger.info("JAX compilation cache: %s", cache_dir)

    if "://" not in cache_dir:
        return

    if "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES" not in os.environ:
        jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
    _enable_xla_autotune_cache()


def _enable_xla_autotune_cache() -> None:
    """Point XLA's per-fusion autotune cache at the node-local mount and mirror it remotely.

    Goes through ``XLA_FLAGS`` because JAX derives this path from the compilation
    cache dir, which is remote. The flag is on ``xla_flags_to_exclude_from_cache_key``,
    so it stays out of the compilation cache key. XLA opens the directory from C++
    through ``tsl::Env``, which cannot read an object store, so the live directory
    is always node-local and FineStore publishes its files as transaction batches.

    GPU only: a TPU or CPU jaxlib aborts on an unknown ``--xla_gpu`` flag.

    The mount arrives with the worker, and VM-cluster workers restart on their own
    daily schedule. For up to a day after a rollout a task can land on a worker
    whose mount is absent or unwritable; skip the cache there.
    """
    if TaskResources.from_environment().gpu_count == 0:
        return

    if not os.path.isdir(SCRATCH_CACHE_PATH):
        logger.info("XLA autotune cache disabled: %s is not mounted", SCRATCH_CACHE_PATH)
        return

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if any(flag.partition("=")[0] == _XLA_AUTOTUNE_CACHE_DIR_FLAG for flag in xla_flags.split()):
        return

    # The mount is node-local, so a task running one process per GPU puts several writers in one
    # directory, where they race each other's temp entries and a reader hits NOT_FOUND on a file a
    # neighbour already moved. Give each process its own subdirectory: they autotune the same
    # fusions independently instead of sharing, which costs some compile time and keeps the cache
    # usable at all.
    autotune_dir = f"{SCRATCH_CACHE_PATH}/{_XLA_AUTOTUNE_CACHE_SUBDIR}"
    process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    if process_index is not None:
        autotune_dir = f"{autotune_dir}/process-{process_index}"
    try:
        os.makedirs(autotune_dir, exist_ok=True)
    except OSError as exc:
        logger.info("XLA autotune cache disabled: cannot create %s: %s", autotune_dir, exc)
        return

    os.environ["XLA_FLAGS"] = f"{xla_flags} {_XLA_AUTOTUNE_CACHE_DIR_FLAG}={autotune_dir}".strip()
    logger.info("XLA per-fusion autotune cache: %s", autotune_dir)

    mode = XlaAutotuneCacheMode(os.environ.get(XLA_AUTOTUNE_CACHE_MODE_ENV, XlaAutotuneCacheMode.REMOTE_SYNC.value))
    if mode is XlaAutotuneCacheMode.LOCAL_ONLY:
        logger.info("XLA per-fusion autotune cache remote sync disabled")
        return

    # One process per task populates its node-local mount before distributed init's
    # barrier; only global process 0 uploads additions shared by all equivalent ranks.
    # Off a real launch the published provenance is absent and the cache stays local.
    if os.environ.get(LAUNCH_PROVENANCE_ENV):
        role = _autotune_cache_role()
        if role is _AutotuneCacheRole.UPLOADER:
            sync_file_set_cache(_XLA_AUTOTUNE_REMOTE_PREFIX, autotune_dir)
        elif role is _AutotuneCacheRole.FETCHER:
            fetch_file_set_cache(_XLA_AUTOTUNE_REMOTE_PREFIX, autotune_dir)


def _file_set_cache_root(prefix: str) -> str | None:
    tree_hash = launch_provenance().tree_hash
    if not tree_hash:
        return None
    return prefix_join(marin_temp_bucket(_XLA_AUTOTUNE_CACHE_TTL_DAYS, prefix), tree_hash)


def sync_file_set_cache(prefix: str, local: str) -> FineStoreDirectory | None:
    """Start a file-set synchronizer, or return ``None`` when remote caching is unavailable."""
    root = _file_set_cache_root(prefix)
    if root is None:
        return None
    try:
        return FineStoreDirectory(root, local)
    except OSError as exc:
        logger.warning("XLA autotune cache is unavailable; continuing with the node-local cache: %s", exc)
        return None


def fetch_file_set_cache(prefix: str, local: str) -> None:
    """Fetch one build's committed file set without starting an uploader."""
    root = _file_set_cache_root(prefix)
    if root is not None:
        try:
            fetch_file_set(root, local)
        except OSError as exc:
            logger.warning("XLA autotune cache fetch failed; starting cold: %s", exc)


def _autotune_cache_role() -> _AutotuneCacheRole:
    """Select one cache fetcher per task and one uploader for the whole job."""
    job_info = get_job_info()
    raw_process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    if raw_process_index is None:
        if job_info is None or job_info.task_index == 0:
            return _AutotuneCacheRole.UPLOADER
        return _AutotuneCacheRole.FETCHER

    process_index = int(raw_process_index)
    if process_index == 0:
        return _AutotuneCacheRole.UPLOADER

    process_count = int(os.environ[IRIS_MULTIGPU_PROCESS_COUNT_ENV])
    num_tasks = job_info.num_tasks if job_info is not None else 1
    processes_per_task = process_count // num_tasks
    if process_index % processes_per_task == 0:
        return _AutotuneCacheRole.FETCHER
    return _AutotuneCacheRole.NONE


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


def _attempt_endpoint_name(endpoint_name: str, attempt_id: int) -> str:
    return f"{endpoint_name}-attempt-{attempt_id}"


def _register_coordinator(job_info: JobInfo, port: int | None, endpoint_name: str) -> str:
    """Choose and publish global process 0's coordinator address."""
    coordinator = f"{job_info.advertise_host}:{resolve_coordinator_port(job_info, port)}"
    ctx = iris_ctx()
    endpoint_id = ctx.registry.register(endpoint_name, coordinator)
    atexit.register(ctx.registry.unregister, endpoint_id)
    return coordinator


def _initialize_supervised_jax(
    jax,
    job_info,
    *,
    port: int | None,
    endpoint_name: str,
    poll_timeout: float,
    poll_interval: float,
    heartbeat_timeout: int,
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
        jax.distributed.initialize(
            num_processes=1,
            process_id=0,
            local_device_ids=device_ids,
        )
        return

    if job_info is None:
        raise RuntimeError("multi-process JAX initialization requires an Iris job context")

    if proc_index == 0:
        coordinator = _register_coordinator(job_info, port, endpoint_name)
    else:
        ctx = iris_ctx()
        coordinator = _poll_for_coordinator(ctx.resolver, endpoint_name, poll_timeout, poll_interval)

    bind_address = _ipv4_bind_address(coordinator)
    logger.info(
        "initialize_jax (supervised): process_id=%d/%d local_device_ids=%s coordinator=%s bind=%s",
        proc_index,
        proc_count,
        device_ids,
        coordinator,
        bind_address,
    )
    jax.distributed.initialize(
        coordinator,
        proc_count,
        proc_index,
        local_device_ids=device_ids,
        coordinator_bind_address=bind_address,
        initialization_timeout=_jax_dist_init_timeout(),
        heartbeat_timeout_seconds=heartbeat_timeout,
    )


def initialize_jax(
    port: int | None = None,
    endpoint_name: str = "jax_coordinator",
    poll_timeout: float | None = None,
    poll_interval: float = 2.0,
    heartbeat_timeout: int = _JAX_DIST_HEARTBEAT_TIMEOUT,
) -> None:
    """Initialize JAX distributed runtime using Iris endpoint discovery.

    Global process 0 registers its coordinator address via the Iris endpoint
    registry. Every other process polls until it discovers that address, whether
    Iris runs one process per task or several processes per GPU task.

    Single-task Iris jobs initialize an explicit one-process distributed world.
    Processes outside an Iris job leave JAX distributed initialization unchanged.

    Iris TPU jobs use the same endpoint-registry coordinator path as other Iris
    multi-task jobs.

    Args:
        port: Coordinator port. ``None`` (the default) prefers an
            ``IRIS_PORT_JAX`` named port and otherwise asks the kernel to select
            an available port. Pass a port only to pin it.
        endpoint_name: Base name for coordinator discovery. Iris scopes the
            registered name to the current task attempt.
        poll_timeout: Maximum seconds for non-coordinator tasks to wait for the
            coordinator endpoint to register. Defaults to the resolved init timeout
            so a slow coordinator host on a large-gang cold restart does not abort
            the pollers before the JAX barrier itself gets its longer timeout.
        poll_interval: Initial backoff delay for polling (seconds).
        heartbeat_timeout: Maximum seconds without a peer heartbeat before JAX
            declares the distributed world unhealthy and terminates it.
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

    if poll_timeout is None:
        poll_timeout = _jax_dist_init_timeout()

    job_info = get_job_info()
    if job_info is not None:
        endpoint_name = _attempt_endpoint_name(endpoint_name, job_info.attempt_id)
    _log_jax_bootstrap_inputs(job_info, port=port, endpoint_name=endpoint_name)
    logger.info("JAX distributed init timeout: %ds (%s)", _jax_dist_init_timeout(), JAX_DIST_INIT_TIMEOUT_ENV)

    # Supervised (multi-process-per-task) mode short-circuits the task-derived
    # paths: the multigpu supervisor has already assigned this process its global
    # rank, so the single/multi-task branches below (which assume one process
    # per task) do not apply.
    if IRIS_MULTIGPU_PROCESS_COUNT_ENV in os.environ:
        _initialize_supervised_jax(
            jax,
            job_info,
            port=port,
            endpoint_name=endpoint_name,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            heartbeat_timeout=heartbeat_timeout,
        )
        return

    if job_info is None:
        return

    if job_info.num_tasks <= 1:
        coordinator = f"{job_info.advertise_host}:{resolve_coordinator_port(job_info, port)}"
        jax.distributed.initialize(
            coordinator,
            num_processes=1,
            process_id=0,
        )
        return

    task_index = job_info.task_index

    if task_index == 0:
        coordinator = _register_coordinator(job_info, port, endpoint_name)
        # Register the endpoint first so other tasks can discover the
        # coordinator address. jax.distributed.initialize() blocks until
        # all processes connect, so registering after would deadlock.
        # JAX's internal gRPC retry handles the brief window between
        # endpoint registration and the coordinator starting to listen.
        jax.distributed.initialize(
            coordinator,
            job_info.num_tasks,
            task_index,
            initialization_timeout=_jax_dist_init_timeout(),
            heartbeat_timeout_seconds=heartbeat_timeout,
        )
    else:
        ctx = iris_ctx()
        coordinator = _poll_for_coordinator(ctx.resolver, endpoint_name, poll_timeout, poll_interval)
        jax.distributed.initialize(
            coordinator,
            job_info.num_tasks,
            task_index,
            initialization_timeout=_jax_dist_init_timeout(),
            heartbeat_timeout_seconds=heartbeat_timeout,
        )
