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
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import cloudpickle
import humanfriendly
from fray.actor import ActorGroup, ActorHandle
from fray.client import Client
from fray.current_client import current_client
from fray.local_backend import LocalClient
from fray.types import ActorConfig, ResourceConfig
from rigging.filesystem import StoragePath, TransferBudgetExceeded, marin_temp_bucket
from rigging.timing import ExponentialBackoff

from zephyr.coordinator import (
    MAX_CONCURRENT_PIPELINES,
    MAX_SHARD_FAILURES,
    MAX_SHARD_INFRA_FAILURES,
    ZephyrCoordinator,
    ZephyrExecutionResult,
    _cleanup_execution,
    _execution_result_path,
    _get_stage_description,
    _read_coordinator_result,
    _try_read_coordinator_result,
)
from zephyr.dataset import Dataset
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


def _distributed_worker_limit(configured: int | None) -> int:
    requested = configured or MAX_IRIS_WORKER_REPLICAS
    return min(requested, MAX_IRIS_WORKER_REPLICAS)


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
    worker_group: ActorGroup

    def shutdown(self, client: Client) -> None:
        """Stop the workers before the coordinator."""
        with suppress(Exception):
            self.coordinator.stop_workers.remote().result(timeout=10.0)

        with suppress(Exception):
            if isinstance(client, LocalClient):
                for worker in self.worker_group.wait_ready():
                    worker.shutdown.remote().result(timeout=10.0)
                self.worker_group.shutdown()
            else:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if self.worker_group.is_done():
                        break
                    time.sleep(0.1)
                else:
                    logger.warning("Workers did not exit naturally, terminating")
                    self.worker_group.shutdown()

        with suppress(Exception):
            self.coordinator.shutdown.remote().result(timeout=10.0)
        with suppress(Exception):
            self.coordinator_group.shutdown()


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
            pools are capped at 1,000 Iris replicas; excess shards multiplex through
            those workers.
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
        max_concurrent_pipelines: Maximum pipelines admitted to a shared coordinator.
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
        if self.max_concurrent_pipelines < 1:
            raise ValueError("max_concurrent_pipelines must be at least 1")

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
        limit = self.max_workers
        if isinstance(self.client, LocalClient):
            return limit or os.cpu_count() or 1
        distributed_limit = _distributed_worker_limit(limit)
        if limit is not None and distributed_limit != limit:
            logger.warning(
                "Capping max_workers=%d at the Iris worker replica limit of %d; "
                "remaining shards will be multiplexed through the worker pool",
                limit,
                distributed_limit,
            )
        return distributed_limit

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
            self.no_workers_timeout,
            self.heartbeat_timeout,
            self.max_shard_failures,
            self.max_shard_infra_failures,
            idle_policy is _IdleWorkerPolicy.DRAIN,
            self.max_concurrent_pipelines,
            name=coordinator_name,
            count=1,
            resources=self.coordinator_resources,
            actor_config=ActorConfig(max_concurrency=self.max_concurrent_pipelines + 100),
        )
        coordinator: ActorHandle | None = None
        worker_group: ActorGroup | None = None
        try:
            coordinator = coordinator_group.wait_ready(count=1)[0]
            worker_name = f"zephyr-{self.name}-workers-{pool_id}"
            worker_group = self.client.create_actor_group(
                ZephyrWorker,
                coordinator,
                self.stage_runner_factory,
                ZephyrTaskResources.from_resource_config(self.resources),
                name=worker_name,
                count=worker_count,
                resources=self.resources,
                actor_config=ActorConfig(max_task_retries=10),
            )
            ready_wait = float(os.environ.get("ZEPHYR_WORKERS_READY_WAIT") or 12 * 60 * 60)
            worker_group.wait_ready(count=1, timeout=ready_wait)
            coordinator.set_worker_group.remote(worker_group).result()
            return _OwnedPool(coordinator_group, coordinator, worker_group)
        except Exception:
            if coordinator is not None:
                with suppress(Exception):
                    coordinator.shutdown.remote().result(timeout=10.0)
            if worker_group is not None:
                with suppress(Exception):
                    worker_group.shutdown()
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
                    pool.shutdown(self.client)
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
        pool.shutdown(self.client)

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
