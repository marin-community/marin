# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution context for dedicated and shared Zephyr worker pools."""

import enum
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Hashable
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

import cloudpickle
import humanfriendly
from fray.actor import ActorFuture, ActorGroup, ActorHandle, ActorUnavailableError
from fray.client import Client
from fray.current_client import current_client
from fray.local_backend import LocalClient
from fray.types import ActorConfig, ResourceConfig
from iris.client.client import get_iris_ctx
from rigging.filesystem import StoragePath, TransferBudgetExceeded, marin_temp_bucket
from rigging.timing import ExponentialBackoff

from zephyr.coordinator import (
    COORDINATOR_MAX_CONCURRENCY,
    MAX_CONCURRENT_PIPELINES,
    MAX_SHARD_FAILURES,
    MAX_SHARD_INFRA_FAILURES,
    WORKER_MAX_TASK_RETRIES,
    ZephyrCoordinator,
    ZephyrExecutionResult,
    _cleanup_execution,
    _execution_result_path,
    _get_stage_description,
    _read_coordinator_result,
    _try_read_coordinator_result,
)
from zephyr.dataset import Dataset
from zephyr.memory_store import (
    MemoryStore,
    MemoryStoreActorStats,
    MemoryTableRegistration,
    actor_result_with_recovery,
    memory_store_plan,
    start_actor_calls,
)
from zephyr.plan import PhysicalPlan, compute_plan
from zephyr.runners import InlineRunner, SubprocessRunner
from zephyr.stage_io import (
    StageRunner,
    ZephyrTaskResources,
    ZephyrWorkerError,
    _shared_data_path,
)
from zephyr.worker import ZephyrWorker
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

# Keep a Zephyr worker actor group below the practical Iris/Kubernetes control-plane
# ceiling. Additional shards are pulled by these long-lived replicas.
MAX_IRIS_WORKER_REPLICAS = 1_000


def _generate_execution_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


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


def _default_stage_runner_factory_for(client: Client) -> Callable[[], StageRunner]:
    """Pick the default ``stage_runner_factory`` based on the client type.

    ``LocalClient`` is the dev/test backend — workers are threads in a
    single process, so per-shard subprocess isolation adds latency without
    delivering meaningful isolation. Distributed clients run each worker
    actor as its own VM where subprocess-per-shard gives real protection
    against native crashes and per-shard memory growth. Callers that want
    the other behavior pass ``stage_runner_factory=...`` explicitly.
    """
    if isinstance(client, LocalClient):
        return lambda: InlineRunner()

    return lambda: SubprocessRunner()


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


def _validate_task_resources(
    worker_resources: ResourceConfig,
    task_resources: ResourceConfig,
    task_name: str,
) -> None:
    """Make sure that one task fits the fixed worker pool."""
    fields = (
        "device",
        "preemptible",
        "regions",
        "zone",
        "target_cluster",
        "replicas",
        "device_alternatives",
        "image",
    )
    for field_name in fields:
        worker_value = getattr(worker_resources, field_name)
        task_value = getattr(task_resources, field_name)
        if task_value != worker_value:
            raise ValueError(
                f"Task resource field '{field_name}' must match the worker pool ({task_value!r} != {worker_value!r})"
            )

    worker_ram = humanfriendly.parse_size(worker_resources.ram, binary=True)
    worker_disk = humanfriendly.parse_size(worker_resources.disk, binary=True)
    task_ram = humanfriendly.parse_size(task_resources.ram, binary=True)
    task_disk = humanfriendly.parse_size(task_resources.disk, binary=True)
    if worker_resources.cpu < task_resources.cpu or worker_ram < task_ram or worker_disk < task_disk:
        raise ValueError(
            f"{task_name} task resources exceed one Zephyr worker: task={task_resources}, worker={worker_resources}"
        )


def _resolve_task_resources(
    worker_resources: ResourceConfig,
    map_task_resources: ResourceConfig | None,
    reduce_task_resources: ResourceConfig | None,
) -> tuple[ResourceConfig, ResourceConfig]:
    """Resolve and validate the map and reduce task resources."""
    resolved_map = map_task_resources or worker_resources
    resolved_reduce = reduce_task_resources or resolved_map
    _validate_task_resources(worker_resources, resolved_map, "map")
    _validate_task_resources(worker_resources, resolved_reduce, "reduce")
    return resolved_map, resolved_reduce


class _ContextState(enum.StrEnum):
    NEW = enum.auto()
    OWNER = enum.auto()
    BORROWED = enum.auto()
    CLOSED = enum.auto()


class _IdleWorkerPolicy(enum.StrEnum):
    DRAIN = enum.auto()
    RETAIN = enum.auto()


@dataclass(frozen=True)
class _OwnedPool:
    """Actor groups that one context owns."""

    coordinator_group: ActorGroup
    coordinator: ActorHandle
    worker_count: int

    def shutdown(self) -> None:
        """Stop the workers, then the coordinator that owns them."""
        with suppress(Exception):
            self.coordinator.stop_workers.remote().result(timeout=30.0)
        with suppress(Exception):
            self.coordinator.shutdown.remote().result(timeout=10.0)
        with suppress(Exception):
            self.coordinator_group.shutdown()


def _require_resolvable_worker_handles(client: Client) -> None:
    """Reject a memory-store load whose worker handles could never resolve.

    The coordinator owns the worker group, so the driver receives worker handles over
    an actor response. Serializing an ``IrisActorHandle`` drops its resolver, and the
    handle rebinds through the ambient Iris context -- which a driver outside a job
    does not have. Without this check the load fails later, inside ``load_pass``, as a
    bare "requires IrisContext" from deep in fray.
    """
    if isinstance(client, LocalClient) or get_iris_ctx() is not None:
        return
    raise RuntimeError(
        "load_memory_store requires a driver running inside an Iris job: worker handles "
        "arrive from the coordinator and resolve through the ambient Iris context. "
        "Run this driver as an Iris job, or use a LocalClient pool."
    )


def _distributed_worker_limit(configured: int | None) -> int:
    requested = configured or MAX_IRIS_WORKER_REPLICAS
    return min(requested, MAX_IRIS_WORKER_REPLICAS)


@dataclass
class ZephyrContext:
    """Execute Zephyr pipelines with a dedicated or shared worker pool.

    A plain execute() call creates a dedicated pool. A context manager starts
    one shared pool and keeps it active until context exit. Child jobs can
    receive the entered context through Fray serialization. Borrowed contexts
    keep the coordinator handle but do not own pool shutdown.

    Shared calls do not recreate a failed pool. Each thread has its own shared-data
    view. A caller that starts a thread must copy or initialize the required view.

    Args:
        client: Fray client. Zephyr selects the current client when this is not set.
        max_workers: Worker limit for a dedicated or owned shared pool. Distributed
            pools are capped at 1,000 replicas; additional shards multiplex through
            the existing workers. Local execution is uncapped.
        resources: CPU, memory, and device resources for each worker.
        coordinator_resources: Resources for the coordinator actor.
        chunk_storage_prefix: Storage prefix for shared data, chunks, and results.
        name: Name prefix for actor groups.
        no_workers_timeout: Maximum wait for a live worker during one stage.
        max_execution_retries: Maximum pool retries for a dedicated execute call.
        stage_runner_factory: Factory for the worker stage runner.
        heartbeat_timeout: Maximum time between worker heartbeats.
        max_shard_failures: Maximum task failures for one shard.
        max_shard_infra_failures: Maximum worker failures while one shard is active.
        max_concurrent_pipelines: Maximum pipelines one pool runs at the same
            time. A pipeline past the limit is rejected, not queued. Raise it
            for a driver that fans many pipelines onto one shared pool.
        coordinator_max_concurrency: Concurrent calls the coordinator actor serves.
            Each running pipeline holds one for its whole life, so this must exceed
            ``max_concurrent_pipelines`` by enough to serve every worker's task
            polling -- otherwise workers queue behind the pipelines, their
            completions time out, and shards retry forever while nothing runs.
        worker_max_task_retries: Cumulative worker-task failures the pool tolerates
            before Iris kills the whole worker gang. Iris counts a preemption as a
            failure, so a job at batch priority on a contended cluster spends this
            budget on evictions it recovers from by design. Raise it for a long or
            low-priority run: the default is sized for a short one.
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
    max_execution_retries: int = 100
    stage_runner_factory: Callable[[], StageRunner] | None = None
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES
    max_concurrent_pipelines: int = MAX_CONCURRENT_PIPELINES
    worker_max_task_retries: int = WORKER_MAX_TASK_RETRIES
    coordinator_max_concurrency: int = COORDINATOR_MAX_CONCURRENCY

    _shared_data: ContextVar[dict[str, Any] | None] = field(init=False, repr=False)
    _state: _ContextState = field(init=False, default=_ContextState.NEW, repr=False)
    _pool: _OwnedPool | None = field(init=False, default=None, repr=False)
    _coordinator: ActorHandle | None = field(init=False, default=None, repr=False)
    _state_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = current_client()

        if env_val := os.environ.get("ZEPHYR_MAX_WORKERS"):
            if self.max_workers is None:
                try:
                    self.max_workers = int(env_val)
                except ValueError as e:
                    raise ValueError(f"Invalid ZEPHYR_MAX_WORKERS environment variable value: {env_val}") from e
            else:
                logger.info("Ignoring ZEPHYR_MAX_WORKERS because max_workers is set.")

        if self.resources is None:
            self.resources = ResourceConfig(cpu=1, ram="1g")

        if self.no_workers_timeout is None:
            self.no_workers_timeout = 6 * 60 * 60

        if self.chunk_storage_prefix is None:
            self.chunk_storage_prefix = marin_temp_bucket(ttl_days=1, prefix="zephyr")

        if self.stage_runner_factory is None:
            assert self.client is not None
            self.stage_runner_factory = _default_stage_runner_factory_for(self.client)

        self.name = f"{self.name}-{uuid.uuid4().hex[:8]}"
        self._shared_data = ContextVar(f"zephyr_shared_data_{self.name}", default=None)
        self._state_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize execution access without pool ownership."""
        state = dict(self.__dict__)
        state["_serialized_shared_data"] = dict(self._shared_data.get() or {})
        state.pop("_shared_data", None)
        state.pop("_state_lock", None)
        state["client"] = None
        state["_pool"] = None
        if self._state is _ContextState.OWNER:
            state["_state"] = _ContextState.BORROWED
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        shared_data = state.pop("_serialized_shared_data", {})
        self.__dict__.update(state)
        if self._state is _ContextState.NEW:
            self.client = current_client()
        self._shared_data = ContextVar(f"zephyr_shared_data_{self.name}", default=None)
        self._shared_data.set(dict(shared_data))
        self._state_lock = threading.Lock()

    def put(self, name: str, obj: Any) -> None:
        """Add one value to the current logical shared-data view."""
        current = self._shared_data.get() or {}
        self._shared_data.set({**current, name: obj})

    def load_memory_store(
        self,
        dataset: Dataset[tuple[K, V]],
        *,
        name: str,
        hash_key: Callable[[K], int],
        recovery_timeout: float,
        ready_timeout: float = 900.0,
    ) -> MemoryStore[K, V]:
        """Load an existing partitioned Dataset into the shared worker pool.

        The Dataset must contain only shard-local operations and yield `(key,
        value)` tuples. `hash_key(key) % num_source_partitions` must equal the
        physical source shard containing that key. Construction validates this
        contract for every row and never inserts a shuffle.

        This context must own an entered worker pool with explicit
        `max_workers`. The returned table reference is picklable and remains
        valid across executions until it is destroyed or the context exits.

        Args:
            dataset: Shard-local Dataset yielding `(key, value)` tuples.
            name: Descriptive table name used in logs and errors.
            hash_key: Stable, picklable key hash used by the existing partitioning.
            recovery_timeout: Overall deadline for a lookup, stats, or destroy
                operation, including ordinary responses and worker recovery.
            ready_timeout: Seconds to wait for every worker to validate and load the table.

        Tables share each worker's process memory. Size the worker resource
        request for the tables and pipeline tasks it will host.
        """
        if not name:
            raise ValueError("memory store name must not be empty")
        if recovery_timeout <= 0:
            raise ValueError(f"recovery_timeout must be positive, got {recovery_timeout}")
        if ready_timeout <= 0:
            raise ValueError(f"ready_timeout must be positive, got {ready_timeout}")
        with self._state_lock:
            if self._state is not _ContextState.OWNER:
                raise RuntimeError("load_memory_store requires an entered ZephyrContext that owns its worker pool")
            if self.max_workers is None:
                raise ValueError("load_memory_store requires explicit max_workers")
            pool = self._pool
            coordinator = self._coordinator
        assert pool is not None
        assert coordinator is not None

        table_id = uuid.uuid4().hex
        registration = MemoryTableRegistration(
            table_id=table_id,
            name=name,
            plan=memory_store_plan(dataset),
            hash_key=hash_key,
            worker_count=pool.worker_count,
        )

        _require_resolvable_worker_handles(self.client)

        deadline = time.monotonic() + ready_timeout
        handles = coordinator.worker_handles.remote(pool.worker_count, ready_timeout).result(timeout=ready_timeout)

        def load_pass() -> list[MemoryStoreActorStats]:
            calls = {
                position: lambda handle=handle: handle.load_memory_table.submit(registration)
                for position, handle in enumerate(handles)
            }
            futures = start_actor_calls(calls)
            return [
                actor_result_with_recovery(
                    calls[position],
                    futures[position],
                    position,
                    ready_timeout,
                    deadline,
                )
                for position in range(len(handles))
            ]

        try:
            load_pass()
            coordinator.register_memory_table.remote(registration).result(timeout=max(0.0, deadline - time.monotonic()))
            stats_by_position = load_pass()
            actors_by_index: list[ActorHandle | None] = [None] * pool.worker_count
            for handle, stats in zip(handles, stats_by_position, strict=True):
                if actors_by_index[stats.actor_index] is not None:
                    raise RuntimeError(f"two memory-store actors reported index {stats.actor_index}")
                actors_by_index[stats.actor_index] = handle
            if any(handle is None for handle in actors_by_index):
                raise RuntimeError("memory-store actor group did not report every actor index")
            actors = tuple(handle for handle in actors_by_index if handle is not None)
        except BaseException:
            try:
                coordinator.unregister_memory_table.remote(table_id).result(timeout=10.0)
            except Exception:
                logger.warning("Failed to unregister memory table %s after load failure", table_id, exc_info=True)
            destroy_futures: list[tuple[int, ActorFuture]] = []
            for worker_index, handle in enumerate(handles):
                try:
                    destroy_futures.append((worker_index, handle.destroy_memory_table.remote(table_id)))
                except ActorUnavailableError:
                    logger.warning("Worker %d unavailable while cleaning up memory table %s", worker_index, table_id)
            for worker_index, future in destroy_futures:
                try:
                    future.result(timeout=10.0)
                except Exception:
                    logger.warning("Worker %d failed to clean up memory table %s", worker_index, table_id, exc_info=True)
            raise

        return MemoryStore(
            table_id=table_id,
            name=name,
            actors=actors,
            coordinator=coordinator,
            hash_key=hash_key,
            num_source_partitions=registration.plan.num_source_partitions,
            recovery_timeout=recovery_timeout,
        )

    def _upload_shared_data(self, execution_id: str) -> None:
        """Write the current logical shared-data view for one execution."""
        for name, obj in dict(self._shared_data.get() or {}).items():
            path = _shared_data_path(self.chunk_storage_prefix, execution_id, name)
            ensure_parent_dir(path)
            started = time.monotonic()
            data = cloudpickle.dumps(obj)
            StoragePath(path).write_bytes(data)
            logger.info(
                "Shared data '%s' written to %s (serialized %d bytes in %.2fs)",
                name,
                path,
                len(data),
                time.monotonic() - started,
            )

    def _worker_limit(self) -> int:
        assert self.client is not None
        if isinstance(self.client, LocalClient):
            return self.max_workers or os.cpu_count() or 1

        limit = _distributed_worker_limit(self.max_workers)
        if self.max_workers is not None and limit != self.max_workers:
            logger.warning(
                "Capping max_workers=%d at the Iris worker replica limit of %d; "
                "remaining shards will be multiplexed through the worker pool",
                self.max_workers,
                limit,
            )
        return limit

    def _start_pool(
        self,
        worker_count: int,
        idle_policy: _IdleWorkerPolicy,
    ) -> _OwnedPool:
        """Start one coordinator job and its worker group."""
        assert self.client is not None
        assert self.resources is not None
        assert self.stage_runner_factory is not None
        assert self.chunk_storage_prefix is not None
        assert self.no_workers_timeout is not None

        pool_id = uuid.uuid4().hex[:8]
        coordinator_name = f"zephyr-{self.name}-coordinator-{pool_id}"
        coordinator_group = self.client.create_actor_group(
            ZephyrCoordinator,
            self.chunk_storage_prefix,
            ZephyrTaskResources.from_resource_config(self.resources),
            # By keyword: the coordinator takes a run of same-typed limits that
            # a positional list would silently rebind on reorder.
            no_workers_timeout=self.no_workers_timeout,
            heartbeat_timeout=self.heartbeat_timeout,
            max_shard_failures=self.max_shard_failures,
            max_shard_infra_failures=self.max_shard_infra_failures,
            drain_idle_workers=idle_policy is _IdleWorkerPolicy.DRAIN,
            max_concurrent_pipelines=self.max_concurrent_pipelines,
            name=coordinator_name,
            count=1,
            resources=self.coordinator_resources,
            actor_config=ActorConfig(max_concurrency=self.coordinator_max_concurrency),
        )
        coordinator: ActorHandle | None = None
        try:
            coordinator = coordinator_group.wait_ready(count=1)[0]
            # The coordinator creates the workers so they land in a child job of its
            # own and Iris cascading termination retires them with it.
            coordinator.start_workers.remote(
                ZephyrWorker,
                worker_count,
                self.stage_runner_factory,
                ZephyrTaskResources.from_resource_config(self.resources),
                self.resources,
                ActorConfig(max_concurrency=100, max_task_retries=self.worker_max_task_retries),
            ).result()
            ready_wait = float(os.environ.get("ZEPHYR_WORKERS_READY_WAIT") or 12 * 60 * 60)
            coordinator.worker_handles.remote(1, ready_wait).result(timeout=ready_wait)
            return _OwnedPool(coordinator_group, coordinator, worker_count)
        except Exception:
            if coordinator is not None:
                with suppress(Exception):
                    coordinator.stop_workers.remote().result(timeout=30.0)
                with suppress(Exception):
                    coordinator.shutdown.remote().result(timeout=10.0)
            with suppress(Exception):
                coordinator_group.shutdown()
            raise

    def start(self) -> "ZephyrContext":
        """Start a shared pool and retain idle workers."""
        with self._state_lock:
            if self._state is not _ContextState.NEW:
                raise RuntimeError(f"Cannot start ZephyrContext in state {self._state}")
            pool = self._start_pool(self._worker_limit(), _IdleWorkerPolicy.RETAIN)
            self._pool = pool
            self._coordinator = pool.coordinator
            self._state = _ContextState.OWNER
        return self

    def _run_on_coordinator(
        self,
        coordinator: ActorHandle,
        plan: PhysicalPlan,
        execution_id: str,
        map_task_resources: ResourceConfig,
        reduce_task_resources: ResourceConfig,
    ) -> ZephyrExecutionResult:
        """Run one plan on an existing coordinator and read its stored result."""
        result_path = _execution_result_path(self.chunk_storage_prefix, execution_id)
        try:
            coordinator.run_pipeline.submit(
                plan,
                execution_id,
                ZephyrTaskResources.from_resource_config(map_task_resources),
                ZephyrTaskResources.from_resource_config(reduce_task_resources),
            ).result()
        except Exception:
            payload = _try_read_coordinator_result(result_path)
            if isinstance(payload, Exception):
                raise payload from None
            raise

        payload = _read_coordinator_result(result_path)
        if isinstance(payload, Exception):
            raise payload
        return payload

    def execute(
        self,
        dataset: Dataset,
        verbose: bool = False,
        dry_run: bool = False,
        *,
        map_task_resources: ResourceConfig | None = None,
        reduce_task_resources: ResourceConfig | None = None,
    ) -> ZephyrExecutionResult:
        """Execute one dataset on a dedicated or supplied shared pool."""
        plan = compute_plan(dataset)
        if verbose or dry_run:
            _print_plan(dataset.operations, plan)
        if dry_run:
            return ZephyrExecutionResult(results=[], counters={})
        if plan.num_shards <= 0:
            logger.warning("No shards in plan, returning empty results.")
            return ZephyrExecutionResult(results=[], counters={})

        assert self.resources is not None
        resolved_map, resolved_reduce = _resolve_task_resources(
            self.resources,
            map_task_resources,
            reduce_task_resources,
        )

        with self._state_lock:
            if self._state is _ContextState.CLOSED:
                raise RuntimeError("Cannot execute with a closed ZephyrContext")
            state = self._state
            coordinator = self._coordinator

        if state in {_ContextState.OWNER, _ContextState.BORROWED}:
            assert coordinator is not None
            execution_id = _generate_execution_id()
            logger.info("Starting shared Zephyr pipeline %s", execution_id)
            self._upload_shared_data(execution_id)
            try:
                return self._run_on_coordinator(
                    coordinator,
                    plan,
                    execution_id,
                    resolved_map,
                    resolved_reduce,
                )
            finally:
                with suppress(Exception):
                    coordinator.release_execution.remote(execution_id).result(timeout=10.0)

        assert state is _ContextState.NEW
        tasks_per_worker = min(
            _tasks_per_worker(self.resources, resolved_map),
            _tasks_per_worker(self.resources, resolved_reduce),
        )
        last_exception: Exception | None = None
        backoff = ExponentialBackoff(initial=2.0, maximum=60.0, factor=2.0, jitter=0.1)
        for attempt in range(self.max_execution_retries + 1):
            execution_id = _generate_execution_id()
            pool: _OwnedPool | None = None
            try:
                self._upload_shared_data(execution_id)
                needed_workers = math.ceil(plan.num_shards / tasks_per_worker)
                pool = self._start_pool(
                    min(self._worker_limit(), needed_workers),
                    _IdleWorkerPolicy.DRAIN,
                )
                backoff.reset()
                return self._run_on_coordinator(
                    pool.coordinator,
                    plan,
                    execution_id,
                    resolved_map,
                    resolved_reduce,
                )
            except _NON_RETRYABLE_ERRORS:
                raise
            except Exception as error:
                payload = _try_read_coordinator_result(_execution_result_path(self.chunk_storage_prefix, execution_id))
                if isinstance(payload, _NON_RETRYABLE_ERRORS):
                    raise payload from None
                last_exception = error
                if attempt >= self.max_execution_retries:
                    raise
                delay = backoff.next_interval()
                logger.warning(
                    "Pipeline attempt %d failed (%d retries left), retrying in %.1fs: %s",
                    attempt,
                    self.max_execution_retries - attempt,
                    delay,
                    error,
                )
                time.sleep(delay)
            finally:
                if pool is not None:
                    assert self.client is not None
                    pool.shutdown()
                _cleanup_execution(self.chunk_storage_prefix, execution_id)

        raise last_exception  # type: ignore[misc]

    def shutdown(self) -> None:
        """Stop an owned shared pool."""
        with self._state_lock:
            if self._state is _ContextState.BORROWED:
                raise RuntimeError("A borrowed ZephyrContext cannot stop its shared pool")
            if self._state in {_ContextState.NEW, _ContextState.CLOSED}:
                return
            assert self._state is _ContextState.OWNER
            assert self._pool is not None
            pool = self._pool
            self._pool = None
            self._coordinator = None
            self._state = _ContextState.CLOSED

        assert self.client is not None
        pool.shutdown()

    def __enter__(self) -> "ZephyrContext":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


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
