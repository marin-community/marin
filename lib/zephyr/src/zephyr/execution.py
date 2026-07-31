# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pipeline driver for Zephyr.

``ZephyrContext`` turns a ``Dataset`` into a physical plan and runs it on a
``ZephyrPool``. With no ``coordinator_endpoint`` it builds a one-shot pool sized
to the plan, runs the pipeline, and tears the pool down; each
``max_execution_retries`` attempt gets a fresh pool. With an endpoint — passed
directly or via ``ZEPHYR_COORDINATOR_ENDPOINT`` — it submits to that standing
pool instead, where the pipeline runs concurrently with other drivers' and
there is no driver-side retry.

The pool owns worker recovery; see ``zephyr.pool`` and ``zephyr.coordinator``.
"""

import logging
import math
import os
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
import humanfriendly
from fray.client import Client
from fray.current_client import current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath, TransferBudgetExceeded, marin_temp_bucket
from rigging.timing import ExponentialBackoff

from zephyr.coordinator import (
    MAX_SHARD_FAILURES,
    MAX_SHARD_INFRA_FAILURES,
    CoordinatorInfo,
    ZephyrExecutionResult,
    _cleanup_execution,
    _execution_result_path,
    _generate_execution_id,
    _get_stage_description,
)
from zephyr.dataset import Dataset
from zephyr.plan import (
    PhysicalPlan,
    compute_plan,
)
from zephyr.pool import (
    _DEFAULT_NO_WORKERS_TIMEOUT,
    ZephyrPool,
    _default_stage_runner_factory_for,
)
from zephyr.stage_io import (
    StageRunner,
    ZephyrTaskResources,
    ZephyrWorkerError,
    _shared_data_path,
)
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)


MAX_WORKERS_PER_JOB = 1_024


# When set, a ZephyrContext with no explicit ``coordinator_endpoint`` connects
# to that pool's coordinator. Lets a pool owner hand the endpoint to
# per-step driver jobs via the environment (e.g. Iris ``-e``) instead of
# threading it through every step's code.
ZEPHYR_COORDINATOR_ENDPOINT_ENV = "ZEPHYR_COORDINATOR_ENDPOINT"


# Application errors that should never be retried by the execute() retry loop.
# These are deterministic errors (bad plan, invalid config, programming bugs)
# that would fail identically on every attempt. Infrastructure errors (OSError,
# RuntimeError from dead actors, backend actor errors) are NOT listed here so they
# remain retryable. TransferBudgetExceeded is deterministic: the cross-region
# budget is global and persists for the life of the process, so every retry hits
# the same wall while re-transferring data across regions.
_NON_RETRYABLE_ERRORS = (
    ZephyrWorkerError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    MemoryError,
    TransferBudgetExceeded,
)


# ---------------------------------------------------------------------------
# ZephyrCoordinator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ZephyrWorker
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coordinator-as-Job infrastructure
# ---------------------------------------------------------------------------


def _read_coordinator_result(result_path: str) -> Any:
    """Read the coordinator job's result file. Returns the deserialized object."""
    data = StoragePath(result_path).read_bytes()
    try:
        return cloudpickle.loads(data)
    except Exception as e:
        # The coordinator normalizes exceptions before persisting, so a revival
        # failure here means a genuinely corrupt or version-incompatible payload.
        # Surface a clear non-retryable error instead of letting an opaque
        # unpickle error crash the driver mid-recovery.
        raise ZephyrWorkerError(f"Could not deserialize coordinator result at {result_path}: {e!r}") from e


def _try_read_coordinator_result(result_path: str) -> Any:
    """Best-effort read of the result file. Returns None if unreadable.

    Used only in the retry error-recovery path where the coordinator job
    may have crashed before writing the file.
    """
    try:
        return _read_coordinator_result(result_path)
    except Exception:
        return None


def _tasks_per_worker(worker_resources: ResourceConfig, task_resources: ResourceConfig) -> int:
    """Return how many concurrent copies of *task_resources* fit in *worker_resources*.

    Packing uses cpu and ram only. Zephyr does not track disk in runtime
    admission (``ZephyrTaskResources`` is cpu+memory), so disk is ignored here
    even though it still applies to Iris worker sizing.
    """
    ratios = [worker_resources.cpu / task_resources.cpu]
    worker_ram = humanfriendly.parse_size(worker_resources.ram, binary=True)
    task_ram = humanfriendly.parse_size(task_resources.ram, binary=True)
    if task_ram > 0:
        ratios.append(worker_ram / task_ram)
    return max(1, math.floor(min(ratios)))


def _compute_min_tasks_per_worker(
    worker_resources: ResourceConfig,
    map_resources: ResourceConfig,
    reduce_resources: ResourceConfig,
) -> int:
    """Compute how many concurrent tasks fit on one worker given map/reduce task costs.

    Uses the tighter of the map and reduce packing densities so workers sized for
    both stage types can keep enough tasks in flight for either.
    """
    for field_name in ["device", "preemptible", "regions", "zone", "replicas", "image", "device_alternatives"]:
        map_val = getattr(map_resources, field_name)
        reduce_val = getattr(reduce_resources, field_name)
        if map_val != reduce_val:
            raise ValueError(
                f"Field '{field_name}' cannot differ between map_task_resources ({map_val}) "
                f"and reduce_task_resources ({reduce_val}). Set the same value on both."
            )

    return min(
        _tasks_per_worker(worker_resources, map_resources),
        _tasks_per_worker(worker_resources, reduce_resources),
    )


@dataclass
class ZephyrContext:
    """Execution context for Zephyr pipelines.

    Every pipeline runs on a ``ZephyrPool``. Which pool depends on whether an
    endpoint is configured.

    With no endpoint, execute() builds a one-shot pool sized to the plan, runs
    the pipeline on it, and tears it down in a ``finally``. A failed attempt is
    retried up to ``max_execution_retries`` times, each on a fresh pool.

    With an endpoint — passed as ``coordinator_endpoint`` or exported as
    ``ZEPHYR_COORDINATOR_ENDPOINT``, which a plain ``ZephyrContext()`` picks up
    — execute() submits to that standing pool instead, where the pipeline runs
    concurrently with other drivers' pipelines and its tasks pack onto whichever
    workers have free capacity. There is no driver-side retry in that case: the
    pool is non-preemptible and owns worker recovery.

    Either way ``map/reduce_task_resources`` give this pipeline's per-task cost.
    Worker sizing is the one-shot pool's business here, and the ``ZephyrPool``'s
    when connected.

    Args:
        coordinator_endpoint: Endpoint of a standing pool's coordinator (as
            returned by ``ZephyrPool.start()`` / ``ZephyrPool`` as a context
            manager). When None, falls back to the ``ZEPHYR_COORDINATOR_ENDPOINT``
            env var; when that is also unset, execute() builds its own one-shot
            pool. Settings that describe a pool rather than a pipeline —
            ``max_workers``, ``resources``, ``coordinator_resources``,
            ``chunk_storage_prefix``, the timeouts, the failure budgets,
            ``stage_runner_factory``, ``pip_dependency_groups``, ``job_env_vars``
            — belong to the pool and are ignored once this is set.
        client: The fray client to use. If None, auto-detects using current_client().
        max_workers: Upper bound on worker count. The actual count is
            min(max_workers, num_shards), computed at first execute(). If None,
            defaults to os.cpu_count() for LocalClient, or ``MAX_WORKERS_PER_JOB``
            (1024) for distributed clients.
        resources: Resource config per worker.
        coordinator_resources: Resource config for the coordinator job. Defaults to 2 GB.
        chunk_storage_prefix: Storage prefix for intermediate chunks. If None, defaults
            to MARIN_PREFIX/tmp/zephyr or /tmp/zephyr.
        name: Descriptive name for this context, used in actor group names for debugging.
            Defaults to a random 8-character hex string.
        no_workers_timeout: Seconds to wait for at least one worker before failing a stage.
            Defaults to 600s.
        max_execution_retries: Maximum number of times to retry a pipeline execution after
            an infrastructure failure (e.g., coordinator VM preemption), each attempt on a
            fresh one-shot pool. Application errors (ZephyrWorkerError) are never retried.
            Defaults to 100.
        stage_runner_factory: Callable ``() -> StageRunner``.
            Defaults to ``InlineRunner`` for ``LocalClient`` and ``SubprocessRunner``
            for distributed clients.
        map_task_resources: ResourceConfig specifying resources required by a single map task.
            Defaults to ``resources``. Requires ``resources`` to be set explicitly.
        reduce_task_resources: ResourceConfig specifying resources required by a single reduce task.
            Defaults to ``map_task_resources``.
        heartbeat_timeout: Seconds without a worker heartbeat before the coordinator
            marks the worker FAILED and requeues its in-flight shard. Defaults to 120.
            Long-running stages (e.g. vLLM inference with cold XLA compile) may need
            to raise this; the JAX/XLA tracer can starve the worker's heartbeat thread
            during compile.
        max_shard_failures: Maximum explicit task-error retries per shard before the
            pipeline aborts. Defaults to ``MAX_SHARD_FAILURES``.
        max_shard_infra_failures: Maximum infra failures (preemption / heartbeat timeout)
            observed while the same shard was in flight before treating the shard payload
            as a deterministic crasher and aborting. Defaults to ``MAX_SHARD_INFRA_FAILURES``.
        pip_dependency_groups: Extra uv dependency groups the one-shot pool job is
            launched with.
        job_env_vars: Env vars set on the one-shot pool job.
    """

    client: Client | None = None
    max_workers: int | None = None
    resources: ResourceConfig | None = None
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=0.1, ram="1g", preemptible=False)
    )
    chunk_storage_prefix: str | None = None
    name: str = ""
    no_workers_timeout: float | None = None
    # NOTE: 100 is fairly aggressive but it fits the preemptible env better
    max_execution_retries: int = 100
    stage_runner_factory: Callable[[], StageRunner] | None = None
    map_task_resources: ResourceConfig | None = None
    reduce_task_resources: ResourceConfig | None = None
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES
    coordinator_endpoint: str | None = None
    pip_dependency_groups: list[str] | None = None
    job_env_vars: dict[str, str] | None = None

    # Shared data staged by put(), uploaded to disk at the start of execute()
    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    # Handle to the coordinator job (for termination on retry/shutdown)
    _active_pool: "ZephyrPool | None" = field(default=None, repr=False)
    # Cached describe() of the coordinator we are connected to
    _coordinator_info: CoordinatorInfo | None = field(default=None, repr=False)
    # NOTE: execute calls increment this at the very beginning
    _pipeline_id: int = field(default=-1, repr=False)
    min_tasks_per_worker: int = field(init=False, default=1, repr=False)

    def __post_init__(self):
        if self.client is None:
            self.client = current_client()

        # Fall back to the env var so a pool owner can point per-step driver
        # jobs at a standing pool without threading the endpoint through
        # code. An explicit `coordinator_endpoint` always wins.
        if self.coordinator_endpoint is None:
            self.coordinator_endpoint = os.environ.get(ZEPHYR_COORDINATOR_ENDPOINT_ENV) or None

        if env_val := os.environ.get("ZEPHYR_MAX_WORKERS"):
            if self.max_workers is None:
                try:
                    self.max_workers = int(env_val)
                except ValueError as e:
                    raise ValueError(f"Invalid ZEPHYR_MAX_WORKERS environment variable value: {env_val}") from e
            else:
                logger.info("Ignoring ZEPHYR_MAX_WORKERS environment variable in favor of max_workers variable.")

        if self.map_task_resources is not None and self.resources is None:
            raise ValueError("Setting map_task_resources without setting resources is an error.")
        if self.reduce_task_resources is not None and self.map_task_resources is None:
            raise ValueError("Setting reduce_task_resources without setting map_task_resources is an error.")

        if self.resources is None:
            self.resources = ResourceConfig(cpu=1, ram="1g")
        if self.map_task_resources is None:
            self.map_task_resources = self.resources
        if self.reduce_task_resources is None:
            self.reduce_task_resources = self.map_task_resources

        # Sizing checks
        resources_ram = humanfriendly.parse_size(self.resources.ram, binary=True)
        resources_disk = humanfriendly.parse_size(self.resources.disk, binary=True)
        for task_resources, name in [
            (self.map_task_resources, "map_task"),
            (self.reduce_task_resources, "reduce_task"),
        ]:
            task_ram = humanfriendly.parse_size(task_resources.ram, binary=True)
            task_disk = humanfriendly.parse_size(task_resources.disk, binary=True)
            if self.resources.cpu < task_resources.cpu or resources_ram < task_ram or resources_disk < task_disk:
                raise ValueError(
                    f"Overall resources ({self.resources}) must be larger than or equal to "
                    f"{name} resources ({task_resources}) on all dimensions (cpu, ram, disk)."
                )

        self.min_tasks_per_worker = _compute_min_tasks_per_worker(
            self.resources, self.map_task_resources, self.reduce_task_resources
        )

        if self.no_workers_timeout is None:
            self.no_workers_timeout = _DEFAULT_NO_WORKERS_TIMEOUT

        if self.chunk_storage_prefix is None:
            # TODO: consider increasing TTL for long-running pipelines (e.g. multi-day fuzzy dedup)
            self.chunk_storage_prefix = marin_temp_bucket(ttl_days=1, prefix="zephyr")

        if self.stage_runner_factory is None:
            self.stage_runner_factory = _default_stage_runner_factory_for(self.client)

        # make sure each context is unique
        self.name = f"{self.name}-{uuid.uuid4().hex[:8]}"

    def put(self, name: str, obj: Any) -> None:
        """Stage shared data for workers to load on demand.

        Must be called before execute(). The object must be picklable.
        Workers access it via zephyr_worker_ctx().get_shared(name), which
        loads from disk on first access and caches locally.

        The actual serialization to disk happens at the start of execute(),
        once the execution_id is known, so each execution is isolated.
        """
        self._shared_data[name] = obj

    def _upload_shared_data(self, prefix: str, execution_id: str) -> None:
        """Serialize all staged shared data to disk under the execution directory."""
        for name, obj in self._shared_data.items():
            path = _shared_data_path(prefix, execution_id, name)
            ensure_parent_dir(path)
            t0 = time.monotonic()
            data = cloudpickle.dumps(obj)
            elapsed = time.monotonic() - t0
            StoragePath(path).write_bytes(data)
            logger.info(
                "Shared data '%s' written to %s (serialized %d bytes in %.2fs)",
                name,
                path,
                len(data),
                elapsed,
            )

    def _run_on_coordinator(self, coordinator_endpoint: str, plan: PhysicalPlan) -> ZephyrExecutionResult:
        """Submit one pipeline to a coordinator and block until it finishes.

        The result comes back from the coordinator's result file rather than the
        actor return value, so an exception keeps its original type even when
        the transport cannot carry it. There is no retry here: the caller owns
        retry policy, and the pool owns worker recovery.
        """
        coordinator = self.client.get_actor(coordinator_endpoint)
        if self._coordinator_info is None:
            self._coordinator_info = coordinator.describe.remote().result(timeout=60.0)
        prefix = self._coordinator_info.chunk_prefix

        assert self.map_task_resources is not None and self.reduce_task_resources is not None
        map_cost = ZephyrTaskResources.from_resource_config(self.map_task_resources)
        reduce_cost = ZephyrTaskResources.from_resource_config(self.reduce_task_resources)

        execution_id = _generate_execution_id()
        result_path = _execution_result_path(prefix, execution_id)
        logger.info("Submitting zephyr pipeline %s to coordinator %s", execution_id, coordinator_endpoint)
        try:
            self._upload_shared_data(prefix, execution_id)
            try:
                coordinator.run_pipeline.submit(plan, execution_id, map_cost, reduce_cost).result()
            except Exception:
                # The coordinator persists the normalized exception before it
                # propagates, so prefer that over whatever the transport carried.
                persisted = _try_read_coordinator_result(result_path)
                if isinstance(persisted, Exception):
                    raise persisted from None
                raise
            payload = _read_coordinator_result(result_path)
            if isinstance(payload, Exception):
                raise payload
            return payload
        finally:
            _cleanup_execution(prefix, execution_id)

    def execute(
        self,
        dataset: Dataset,
        verbose: bool = False,
        dry_run: bool = False,
    ) -> ZephyrExecutionResult:
        """Execute a dataset pipeline.

        Runs on a one-shot pool built for this call, retried up to
        ``max_execution_retries`` times on infrastructure failure (e.g. VM
        preemption). Application errors (``ZephyrWorkerError``) are never
        retried.

        When ``coordinator_endpoint`` is set, submits to that standing pool
        instead, where the pipeline runs concurrently with other drivers'
        pipelines and there is no driver-side retry.

        Returns:
            A ``ZephyrExecutionResult`` containing the flat list of results
            produced by the terminal stage and the aggregated counters from
            the run. Callers that only care about the results should access
            ``.results``; counters are exposed for callers that want to
            persist or surface them.
        """
        plan = compute_plan(dataset)
        if verbose or dry_run:
            _print_plan(dataset.operations, plan)
        if dry_run:
            return ZephyrExecutionResult(results=[], counters={})

        if plan.num_shards <= 0:
            logger.warning("No shards in plan, returning empty results.")
            return ZephyrExecutionResult(results=[], counters={})

        if self.coordinator_endpoint is not None:
            return self._run_on_coordinator(self.coordinator_endpoint, plan)

        # NOTE: pipeline ID incremented on clean completion only
        self._pipeline_id += 1
        # Backoff between retries to avoid hammering an overloaded controller.
        # Starts at 2s, caps at 60s. Resets on successful pipeline startup.
        backoff = ExponentialBackoff(initial=2.0, maximum=60.0, factor=2.0, jitter=0.1)
        for attempt in range(self.max_execution_retries + 1):
            logger.info("Starting zephyr pipeline (pipeline %d, attempt %d)", self._pipeline_id, attempt)
            pool = self._one_shot_pool(plan, attempt)
            self._active_pool = pool
            try:
                endpoint = pool.start()
                backoff.reset()
                return self._run_on_coordinator(endpoint, plan)

            except _NON_RETRYABLE_ERRORS:
                raise

            except Exception as e:
                if attempt >= self.max_execution_retries:
                    raise
                delay = backoff.next_interval()
                logger.warning(
                    "Pipeline attempt %d failed (%d retries left), retrying in %.1fs: %s",
                    attempt,
                    self.max_execution_retries - attempt,
                    delay,
                    e,
                )
                time.sleep(delay)

            finally:
                # Tearing the pool down cascades to its coordinator and workers.
                self._active_pool = None
                with suppress(Exception):
                    pool.shutdown()

        raise AssertionError("retry loop exited without returning or raising")

    def _one_shot_pool(self, plan: PhysicalPlan, attempt: int) -> "ZephyrPool":
        """Build the pool that serves a single dedicated ``execute()`` attempt.

        Workers are sized to the plan rather than to a standing capacity: a
        dedicated run knows its shard count up front, so it asks for only as
        many workers as the plan can keep busy.
        """
        assert self.resources is not None
        limit = self.max_workers
        if limit is None and isinstance(self.client, LocalClient):
            limit = os.cpu_count() or 1
        needed_workers = math.ceil(plan.num_shards / self.min_tasks_per_worker)

        return ZephyrPool(
            client=self.client,
            max_workers=min((limit or MAX_WORKERS_PER_JOB), needed_workers),
            resources=self.resources,
            coordinator_resources=self.coordinator_resources,
            chunk_storage_prefix=self.chunk_storage_prefix,
            name=f"{self.name}-p{self._pipeline_id}-a{attempt}",
            no_workers_timeout=self.no_workers_timeout,
            stage_runner_factory=self.stage_runner_factory,
            heartbeat_timeout=self.heartbeat_timeout,
            max_shard_failures=self.max_shard_failures,
            max_shard_infra_failures=self.max_shard_infra_failures,
            pip_dependency_groups=self.pip_dependency_groups,
            job_env_vars=self.job_env_vars,
            drain_idle_workers=True,
        )

    def shutdown(self) -> None:
        """Tear down a lingering one-shot pool, if any.

        ``execute()`` already shuts its pool down in a ``finally``, so this is
        belt-and-suspenders cleanup. A context connected to a ``ZephyrPool``
        owns no infrastructure, so shutdown() never affects that pool.
        """
        pool, self._active_pool = self._active_pool, None
        if pool is not None:
            with suppress(Exception):
                pool.shutdown()


def _print_plan(original_ops: list, plan: PhysicalPlan) -> None:
    """Print the physical plan showing shard count and operation fusion."""
    total_physical_ops = sum(len(stage.operations) for stage in plan.stages)

    logger.info("\n=== Physical Execution Plan ===\n")
    logger.info(f"Shards: {plan.num_shards}")
    logger.info(f"Original operations: {len(original_ops)}")
    logger.info(f"Stages: {len(plan.stages)}")
    logger.info(f"Physical ops: {total_physical_ops}\n")

    logger.info("Original pipeline:")
    for i, op in enumerate(original_ops, 1):
        logger.info(f"  {i}. {op}")

    logger.info("\nPhysical stages:")
    for i, stage in enumerate(plan.stages, 1):

        stage_desc = _get_stage_description(stage)
        logger.info(f"  {i}. {stage_desc}")

    logger.info("\n=== End Plan ===\n")
