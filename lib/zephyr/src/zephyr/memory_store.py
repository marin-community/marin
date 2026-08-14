# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shard-preserving memory tables hosted by Zephyr workers."""

import enum
import logging
import os
import resource
import select
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Hashable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar, cast

import psutil
from connectrpc.errors import ConnectError
from fray.actor import ActorFuture, ActorHandle, ActorUnavailableError
from iris.actor.client import ActorClient
from iris.actor.resolver import FixedResolver
from iris.actor.server import ActorServer
from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.rpc.errors import is_retryable_error
from rigging.filesystem.factory import is_transient_s3_error
from rigging.timing import ExponentialBackoff, retry_with_backoff

from zephyr.dataset import Dataset
from zephyr.plan import Map, PhysicalOp, SourceItem, StageContext, StageType, compute_plan, run_stage

logger = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
A = TypeVar("A")
R = TypeVar("R")
CallKey = TypeVar("CallKey", bound=Hashable)
MEMORY_STORE_LOAD_MAX_ATTEMPTS = 4
MEMORY_STORE_SUBPROCESS_STOP_TIMEOUT = 10.0


class MemoryStorePartitionError(ValueError):
    """Raised when a key does not belong to its existing Dataset partition."""


class DuplicateMemoryStoreKey(ValueError):
    """Raised when a memory-store input contains the same key twice."""


class MemoryStoreUnavailable(RuntimeError):
    """Raised when a memory-table operation cannot complete before its deadline."""


class MemoryStoreDestroyed(RuntimeError):
    """Raised when a destroyed table reference is queried for values or statistics."""


@dataclass(frozen=True)
class MemoryStoreShardStats:
    """Load statistics for one subprocess shard of a memory table."""

    actor_index: int
    store_shard_index: int
    process_id: int
    source_partitions: tuple[int, ...]
    num_items: int
    load_cpu_time: float
    load_elapsed: float
    resident_bytes: int
    peak_resident_bytes: int
    endpoint_name: str = ""
    endpoint_address: str = ""


def _source_partition(hash_key: Callable[[K], int], key: K, num_source_partitions: int) -> int:
    key_hash = hash_key(key)
    if isinstance(key_hash, bool) or not isinstance(key_hash, int):
        raise TypeError(f"hash_key must return int, got {type(key_hash).__name__} for key {key!r}")
    return key_hash % num_source_partitions


@dataclass(frozen=True)
class MemoryStorePlan:
    source_items: tuple[SourceItem, ...]
    operations: tuple[PhysicalOp, ...]
    num_source_partitions: int


def memory_store_plan(dataset: Dataset[tuple[K, V]]) -> MemoryStorePlan:
    plan = compute_plan(dataset)
    num_source_partitions = plan.num_shards
    if num_source_partitions == 0:
        raise ValueError("cannot construct a memory store from an empty Dataset")

    shard_indices = sorted({item.shard_idx for item in plan.source_items})
    if shard_indices != list(range(num_source_partitions)):
        raise ValueError(f"memory-store Dataset has non-contiguous shard indices: {shard_indices!r}")

    if not plan.stages:
        operations: tuple[PhysicalOp, ...] = ()
    elif (
        len(plan.stages) == 1
        and plan.stages[0].stage_type is StageType.MAP_WORKER
        and all(isinstance(operation, Map) for operation in plan.stages[0].operations)
    ):
        operations = tuple(plan.stages[0].operations)
    else:
        raise ValueError(
            "load_memory_store requires a shard-local Dataset; persist and reload the output of shuffle, "
            "join, reshard, reduce, or write operations before constructing the store"
        )

    return MemoryStorePlan(
        source_items=tuple(plan.source_items),
        operations=operations,
        num_source_partitions=num_source_partitions,
    )


@dataclass(frozen=True)
class MemoryTableRegistration:
    table_id: str
    name: str
    plan: MemoryStorePlan
    hash_key: Callable[[Any], int] = field(repr=False)
    worker_count: int
    shards_per_worker: int
    load_concurrency: int


class MemoryTableStatus(enum.StrEnum):
    READY = enum.auto()
    NOT_LOADED = enum.auto()
    UNKNOWN = enum.auto()
    DESTROYED = enum.auto()


@dataclass(frozen=True)
class MemoryTableLookup:
    status: MemoryTableStatus
    values: list[tuple[bool, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryTableCompute:
    status: MemoryTableStatus
    values: list[tuple[bool, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryTableStatsResult:
    status: MemoryTableStatus
    stats: MemoryStoreShardStats | None = None


@dataclass(frozen=True)
class MemoryTableStatsBatch:
    status: MemoryTableStatus
    stats: tuple[MemoryStoreShardStats, ...] = ()


type _MemoryTableResult = MemoryTableLookup | MemoryTableCompute | MemoryTableStatsResult | MemoryTableStatsBatch


@dataclass
class _MemoryTableState:
    registration: MemoryTableRegistration
    load_lock: threading.Lock = field(default_factory=threading.Lock)
    values: dict[Hashable, Any] | None = None
    stats: MemoryStoreShardStats | None = None


def _load_partition_once(
    registration: MemoryTableRegistration,
    partition: int,
    shard: list[Any],
) -> tuple[int, dict[Hashable, Any]]:
    context = StageContext(
        shard=shard,
        shard_idx=partition,
        total_shards=registration.plan.num_source_partitions,
    )
    values: dict[Hashable, Any] = {}
    for item in run_stage(context, list(registration.plan.operations)):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "memory-store Dataset must yield (key, value) tuples; "
                f"partition {partition} yielded {type(item).__name__}"
            )
        key, value = item
        actual_partition = _source_partition(registration.hash_key, key, registration.plan.num_source_partitions)
        if actual_partition != partition:
            raise MemoryStorePartitionError(
                f"key {key!r} was in source partition {partition}, but hash_key routes it to "
                f"partition {actual_partition}"
            )
        if key in values:
            raise DuplicateMemoryStoreKey(f"duplicate memory-store key {key!r} in partition {partition}")
        values[key] = value
    return partition, values


def _load_partition(
    registration: MemoryTableRegistration,
    partition: int,
    shard: list[Any],
) -> tuple[int, dict[Hashable, Any]]:
    return retry_with_backoff(
        lambda: _load_partition_once(registration, partition, shard),
        retryable=is_transient_s3_error,
        max_attempts=MEMORY_STORE_LOAD_MAX_ATTEMPTS,
        backoff=ExponentialBackoff(initial=0.5, maximum=4.0, factor=2.0, jitter=0.25),
        operation=f"load memory-store source partition {partition}",
    )


class _MemoryStoreShardService:
    """Subprocess-local service that owns one shard of several immutable tables."""

    def __init__(
        self,
        actor_index: int,
        local_shard_index: int,
    ):
        self._actor_index = actor_index
        self._local_shard_index = local_shard_index
        self._tables: dict[str, _MemoryTableState] = {}
        self._destroyed: set[str] = set()
        self._tables_lock = threading.Lock()

    def _install(self, registration: MemoryTableRegistration) -> _MemoryTableState:
        with self._tables_lock:
            if registration.table_id in self._destroyed:
                raise MemoryStoreDestroyed(f"memory table {registration.name!r} has been destroyed")
            state = self._tables.get(registration.table_id)
            if state is not None:
                return state
            state = _MemoryTableState(registration=registration)
            self._tables[registration.table_id] = state
            return state

    def restore(self, registrations: tuple[MemoryTableRegistration, ...]) -> None:
        """Install active table metadata without reading source data."""
        with self._tables_lock:
            for registration in registrations:
                if registration.table_id in self._destroyed or registration.table_id in self._tables:
                    continue
                self._tables[registration.table_id] = _MemoryTableState(registration=registration)

    def _load_values(self, registration: MemoryTableRegistration) -> tuple[dict[Hashable, Any], tuple[int, ...]]:
        plan = registration.plan
        store_shard_index = self._actor_index * registration.shards_per_worker + self._local_shard_index
        store_shard_count = registration.worker_count * registration.shards_per_worker
        partitions = tuple(
            partition
            for partition in range(plan.num_source_partitions)
            if partition % store_shard_count == store_shard_index
        )
        source_data: dict[int, list[Any]] = {partition: [] for partition in partitions}
        for item in plan.source_items:
            if item.shard_idx in source_data:
                source_data[item.shard_idx].append(item.data)

        partition_results: Iterator[tuple[int, dict[Hashable, Any]]]
        if registration.load_concurrency == 1:
            partition_results = (
                _load_partition(registration, partition, shard) for partition, shard in source_data.items()
            )
        else:
            partition_results = self._load_partitions_concurrently(registration, source_data)

        values: dict[Hashable, Any] = {}
        for partition, partition_values in partition_results:
            for key, value in partition_values.items():
                if key in values:
                    raise DuplicateMemoryStoreKey(f"duplicate memory-store key {key!r} in partition {partition}")
                values[key] = value
        return values, partitions

    @staticmethod
    def _load_partitions_concurrently(
        registration: MemoryTableRegistration,
        source_data: dict[int, list[Any]],
    ) -> Iterator[tuple[int, dict[Hashable, Any]]]:
        source_partitions = iter(source_data.items())
        with ThreadPoolExecutor(
            max_workers=registration.load_concurrency,
            thread_name_prefix="memory-store-load",
        ) as executor:
            pending: set[Future[tuple[int, dict[Hashable, Any]]]] = set()

            def submit_next() -> bool:
                try:
                    partition, shard = next(source_partitions)
                except StopIteration:
                    return False
                pending.add(executor.submit(_load_partition, registration, partition, shard))
                return True

            for _ in range(registration.load_concurrency):
                if not submit_next():
                    break

            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    result = future.result()
                    submit_next()
                    yield result

    def load(self, registration: MemoryTableRegistration) -> MemoryStoreShardStats:
        """Load one table, or return its existing load statistics."""
        state = self._install(registration)
        with state.load_lock:
            if state.values is not None:
                assert state.stats is not None
                return state.stats

            load_started = time.monotonic()
            load_cpu_started = time.process_time()
            values, partitions = self._load_values(registration)
            stats = MemoryStoreShardStats(
                actor_index=self._actor_index,
                store_shard_index=self._local_shard_index,
                process_id=os.getpid(),
                source_partitions=partitions,
                num_items=len(values),
                load_cpu_time=time.process_time() - load_cpu_started,
                load_elapsed=time.monotonic() - load_started,
                resident_bytes=psutil.Process().memory_info().rss,
                peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1_024,
            )
            with self._tables_lock:
                if registration.table_id in self._destroyed:
                    raise MemoryStoreDestroyed(f"memory table {registration.name!r} has been destroyed")
                state.values = values
                state.stats = stats

            logger.info(
                "Memory table %s worker %d store shard %d loaded %d items from source partitions %s "
                "in %.2f CPU-seconds and %.2f seconds",
                registration.name,
                self._actor_index,
                self._local_shard_index,
                len(values),
                partitions,
                stats.load_cpu_time,
                stats.load_elapsed,
            )
            return stats

    def lookup(self, table_id: str, keys: list[Hashable]) -> MemoryTableLookup:
        """Return table state or values aligned with the requested keys."""
        with self._tables_lock:
            if table_id in self._destroyed:
                return MemoryTableLookup(MemoryTableStatus.DESTROYED)
            state = self._tables.get(table_id)
            if state is None:
                return MemoryTableLookup(MemoryTableStatus.UNKNOWN)
            values = state.values
        if values is None:
            return MemoryTableLookup(MemoryTableStatus.NOT_LOADED)
        return MemoryTableLookup(
            MemoryTableStatus.READY,
            [(True, values[key]) if key in values else (False, None) for key in keys],
        )

    def compute(
        self,
        table_id: str,
        requests: list[tuple[Hashable, Any]],
        function: Callable[[list[tuple[Any, Any]]], list[Any]],
    ) -> MemoryTableCompute:
        """Apply one batch function to worker-local values and request payloads."""
        with self._tables_lock:
            if table_id in self._destroyed:
                return MemoryTableCompute(MemoryTableStatus.DESTROYED)
            state = self._tables.get(table_id)
            if state is None:
                return MemoryTableCompute(MemoryTableStatus.UNKNOWN)
            values = state.values
        if values is None:
            return MemoryTableCompute(MemoryTableStatus.NOT_LOADED)

        found = [key in values for key, _ in requests]
        inputs = [(values[key], payload) for (key, payload), present in zip(requests, found, strict=True) if present]
        computed = function(inputs)
        if len(computed) != len(inputs):
            raise ValueError(f"memory-store compute returned {len(computed)} results for {len(inputs)} inputs")
        results = iter(computed)
        return MemoryTableCompute(
            MemoryTableStatus.READY,
            [(True, next(results)) if present else (False, None) for present in found],
        )

    def stats(self, table_id: str) -> MemoryTableStatsResult:
        """Return table state or immutable load statistics."""
        with self._tables_lock:
            if table_id in self._destroyed:
                return MemoryTableStatsResult(MemoryTableStatus.DESTROYED)
            state = self._tables.get(table_id)
            if state is None:
                return MemoryTableStatsResult(MemoryTableStatus.UNKNOWN)
            stats = state.stats
        if stats is None:
            return MemoryTableStatsResult(MemoryTableStatus.NOT_LOADED)
        return MemoryTableStatsResult(MemoryTableStatus.READY, stats)

    def destroy(self, table_id: str) -> None:
        """Tombstone one table and release its values."""
        with self._tables_lock:
            self._destroyed.add(table_id)
            state = self._tables.get(table_id)
        if state is None:
            return

        with state.load_lock:
            with self._tables_lock:
                self._tables.pop(table_id, None)


class _MemoryStoreSubprocessDied(RuntimeError):
    pass


def run_memory_store_subprocess(actor_index: int, local_shard_index: int, endpoint_name: str, port: int) -> None:
    """Serve one persistent memory-store shard through an Iris actor endpoint."""
    service = _MemoryStoreShardService(actor_index, local_shard_index)
    server = ActorServer(host="0.0.0.0", port=port)
    server.register(endpoint_name, service)
    actual_port = server.serve_background()
    print(actual_port, flush=True)
    server.wait()


class _MemoryStoreRpcFuture:
    def __init__(self, function: Callable[[], Any]):
        self._future: Future[Any] = Future()

        def run() -> None:
            try:
                self._future.set_result(function())
            except ConnectError as error:
                if is_retryable_error(error):
                    self._future.set_exception(ActorUnavailableError(str(error)))
                else:
                    self._future.set_exception(error)
            except Exception as error:
                self._future.set_exception(error)

        threading.Thread(target=run, daemon=True, name="memory-store-rpc").start()

    def result(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout=timeout)


class _MemoryStoreRpcMethod:
    def __init__(self, handle: "MemoryStoreShardHandle", method_name: str):
        self._handle = handle
        self._method_name = method_name

    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
        return _MemoryStoreRpcFuture(lambda: getattr(self._handle._resolve(), self._method_name)(*args, **kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._handle._resolve(), self._method_name)(*args, **kwargs)


@dataclass
class MemoryStoreShardHandle:
    """Picklable handle that resolves a store shard through Iris or its fixed local address."""

    endpoint_name: str
    endpoint_address: str
    _client: ActorClient | None = field(default=None, init=False, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, str]:
        return {"endpoint_name": self.endpoint_name, "endpoint_address": self.endpoint_address}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.endpoint_name = state["endpoint_name"]
        self.endpoint_address = state["endpoint_address"]
        self._client = None
        self._lock = threading.Lock()

    def refresh(self, endpoint_name: str, endpoint_address: str) -> None:
        """Use a replacement subprocess endpoint on the next request."""
        with self._lock:
            self.endpoint_name = endpoint_name
            self.endpoint_address = endpoint_address
            self._client = None

    def _resolve(self) -> ActorClient:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                context = get_iris_ctx()
                resolver = (
                    context.resolver
                    if context is not None
                    else FixedResolver({self.endpoint_name: self.endpoint_address})
                )
                self._client = ActorClient(resolver, self.endpoint_name, max_call_attempts=1)
            return self._client

    def __getattr__(self, method_name: str) -> _MemoryStoreRpcMethod:
        if method_name.startswith("_"):
            raise AttributeError(method_name)
        return _MemoryStoreRpcMethod(self, method_name)


class _MemoryStoreSubprocess:
    """Supervisor handle for one persistent memory-store actor process."""

    STARTUP_TIMEOUT = 30.0

    def __init__(self, actor_index: int, local_shard_index: int):
        self.actor_index = actor_index
        self.local_shard_index = local_shard_index
        job_info = get_job_info()
        if job_info is None:
            endpoint_prefix = f"local-memory-store-{os.getpid()}"
            advertise_host = "127.0.0.1"
        else:
            endpoint_prefix = f"{job_info.job_id}/memory-store"
            advertise_host = job_info.advertise_host
        self.endpoint_name = f"{endpoint_prefix}-{actor_index}-{local_shard_index}"

        child_env = os.environ.copy()
        child_env["POLARS_MAX_THREADS"] = "1"
        child_env["OMP_NUM_THREADS"] = "1"
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "zephyr.memory_store_subprocess",
                str(actor_index),
                str(local_shard_index),
                self.endpoint_name,
                "0",
            ],
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], self.STARTUP_TIMEOUT)
        if not ready:
            self._process.terminate()
            self._process.wait(timeout=MEMORY_STORE_SUBPROCESS_STOP_TIMEOUT)
            raise TimeoutError(
                f"memory-store subprocess {actor_index}/{local_shard_index} did not open its actor port "
                f"within {self.STARTUP_TIMEOUT:g} seconds"
            )
        port_line = self._process.stdout.readline()
        self._process.stdout.close()
        if not port_line:
            raise _MemoryStoreSubprocessDied(
                f"memory-store subprocess {actor_index}/{local_shard_index} exited before opening its actor port"
            )
        port = int(port_line)
        self.endpoint_address = f"http://{advertise_host}:{port}"
        self.handle = MemoryStoreShardHandle(self.endpoint_name, self.endpoint_address)
        context = get_iris_ctx()
        self._endpoint_id = (
            context.registry.register(self.endpoint_name, self.endpoint_address) if context is not None else None
        )

    def call(self, operation: str, *args: Any) -> Any:
        if self._process.poll() is not None:
            raise _MemoryStoreSubprocessDied(
                f"memory-store subprocess {self.actor_index}/{self.local_shard_index} "
                f"exited with code {self._process.returncode}"
            )
        try:
            result = getattr(self.handle, operation)(*args)
        except ActorUnavailableError as error:
            raise _MemoryStoreSubprocessDied(
                f"memory-store subprocess {self.actor_index}/{self.local_shard_index} stopped responding"
            ) from error
        if isinstance(result, MemoryStoreShardStats):
            return replace(
                result,
                endpoint_name=self.endpoint_name,
                endpoint_address=self.endpoint_address,
            )
        return result

    def close(self) -> None:
        context = get_iris_ctx()
        if self._endpoint_id is not None and context is not None:
            context.registry.unregister(self._endpoint_id)
            self._endpoint_id = None
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=MEMORY_STORE_SUBPROCESS_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=MEMORY_STORE_SUBPROCESS_STOP_TIMEOUT)


class MemoryStoreService:
    """Worker-local supervisor for persistent memory-store subprocess shards."""

    def __init__(self, actor_index: int):
        self._actor_index = actor_index
        self._registrations: dict[str, MemoryTableRegistration] = {}
        self._destroyed: set[str] = set()
        self._processes: dict[int, _MemoryStoreSubprocess] = {}
        self._lock = threading.Lock()

    def _process(self, local_shard_index: int) -> _MemoryStoreSubprocess:
        with self._lock:
            process = self._processes.get(local_shard_index)
            if process is not None:
                return process
            process = _MemoryStoreSubprocess(self._actor_index, local_shard_index)
            registrations = tuple(
                registration
                for registration in self._registrations.values()
                if local_shard_index < registration.shards_per_worker
            )
            self._processes[local_shard_index] = process
        process.call("restore", registrations)
        return process

    def _call(self, local_shard_index: int, operation: str, *args: Any) -> Any:
        process = self._process(local_shard_index)
        try:
            return process.call(operation, *args)
        except _MemoryStoreSubprocessDied:
            logger.warning(
                "Restarting memory-store subprocess %d/%d",
                self._actor_index,
                local_shard_index,
                exc_info=True,
            )
            with self._lock:
                if self._processes.get(local_shard_index) is process:
                    self._processes.pop(local_shard_index)
            process.close()
            return self._process(local_shard_index).call(operation, *args)

    def restore(self, registrations: tuple[MemoryTableRegistration, ...]) -> None:
        """Restore active registrations; values reload only when first queried."""
        with self._lock:
            for registration in registrations:
                if registration.table_id not in self._destroyed:
                    self._registrations[registration.table_id] = registration

    def load(self, registration: MemoryTableRegistration) -> tuple[MemoryStoreShardStats, ...]:
        """Load every local subprocess shard concurrently."""
        with self._lock:
            if registration.table_id in self._destroyed:
                raise MemoryStoreDestroyed(f"memory table {registration.name!r} has been destroyed")
            existing = self._registrations.get(registration.table_id)
            existing_shape = None
            if existing is not None:
                existing_shape = (
                    existing.name,
                    existing.worker_count,
                    existing.shards_per_worker,
                    existing.plan.num_source_partitions,
                )
            registration_shape = (
                registration.name,
                registration.worker_count,
                registration.shards_per_worker,
                registration.plan.num_source_partitions,
            )
            if existing_shape is not None and existing_shape != registration_shape:
                raise ValueError(f"memory table {registration.name!r} was registered with different metadata")
            self._registrations[registration.table_id] = registration

        with ThreadPoolExecutor(
            max_workers=registration.shards_per_worker,
            thread_name_prefix="memory-store-shard-load",
        ) as executor:
            futures = [
                executor.submit(self._call, local_shard_index, "load", registration)
                for local_shard_index in range(registration.shards_per_worker)
            ]
            return tuple(future.result() for future in futures)

    def _registration_status(self, table_id: str, result_type: type[_MemoryTableResult]) -> _MemoryTableResult | None:
        with self._lock:
            if table_id in self._destroyed:
                return result_type(MemoryTableStatus.DESTROYED)
            if table_id not in self._registrations:
                return result_type(MemoryTableStatus.UNKNOWN)
        return None

    def lookup(self, table_id: str, local_shard_index: int, keys: list[Hashable]) -> MemoryTableLookup:
        """Return values from one local subprocess shard."""
        status = self._registration_status(table_id, MemoryTableLookup)
        if status is not None:
            return cast(MemoryTableLookup, status)
        return self._call(local_shard_index, "lookup", table_id, keys)

    def compute(
        self,
        table_id: str,
        local_shard_index: int,
        requests: list[tuple[Hashable, Any]],
        function: Callable[[list[tuple[Any, Any]]], list[Any]],
    ) -> MemoryTableCompute:
        """Apply a batch function inside one local subprocess shard."""
        status = self._registration_status(table_id, MemoryTableCompute)
        if status is not None:
            return cast(MemoryTableCompute, status)
        return self._call(local_shard_index, "compute", table_id, requests, function)

    def stats(self, table_id: str) -> MemoryTableStatsBatch:
        """Return statistics from all local subprocess shards."""
        status = self._registration_status(table_id, MemoryTableStatsBatch)
        if status is not None:
            return cast(MemoryTableStatsBatch, status)
        with self._lock:
            registration = self._registrations[table_id]
        results = tuple(
            self._call(local_shard_index, "stats", table_id)
            for local_shard_index in range(registration.shards_per_worker)
        )
        statuses = {result.status for result in results}
        if statuses != {MemoryTableStatus.READY}:
            return MemoryTableStatsBatch(next(iter(statuses)))
        return MemoryTableStatsBatch(
            MemoryTableStatus.READY,
            tuple(cast(MemoryStoreShardStats, result.stats) for result in results),
        )

    def destroy(self, table_id: str) -> None:
        """Tombstone one table and release its values in every child."""
        with self._lock:
            self._destroyed.add(table_id)
            registration = self._registrations.pop(table_id, None)
        if registration is None:
            return
        for local_shard_index in range(registration.shards_per_worker):
            self._call(local_shard_index, "destroy", table_id)

    def close(self) -> None:
        """Stop all supervised subprocesses."""
        with self._lock:
            processes = tuple(self._processes.values())
            self._processes.clear()
        for process in processes:
            process.close()


def actor_result_with_recovery(
    call: Callable[[], ActorFuture],
    initial_future: ActorFuture | None,
    actor_index: int,
    recovery_timeout: float,
    deadline: float,
    recover: Callable[[], None] | None = None,
) -> Any:
    backoff = ExponentialBackoff(initial=0.5, maximum=10.0, factor=2.0, jitter=0.25)
    future = initial_future
    while True:
        remaining = deadline - time.monotonic()
        if future is None and remaining <= 0:
            raise MemoryStoreUnavailable(
                f"memory-store actor {actor_index} did not recover within {recovery_timeout:g} seconds"
            )
        try:
            if future is None:
                future = call()
                remaining = deadline - time.monotonic()
            return future.result(timeout=max(0.0, remaining))
        except TimeoutError as exc:
            raise MemoryStoreUnavailable(
                f"memory-store actor {actor_index} did not respond within {recovery_timeout:g} seconds"
            ) from exc
        except ActorUnavailableError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MemoryStoreUnavailable(
                    f"memory-store actor {actor_index} did not recover within {recovery_timeout:g} seconds"
                ) from exc
            future = None
            if recover is not None:
                recover()
                continue
            delay = min(backoff.next_interval(), remaining)
            logger.warning("Memory-store actor %d unavailable; retrying in %.1f seconds", actor_index, delay)
            time.sleep(delay)


def start_actor_calls(
    calls: dict[CallKey, Callable[[], ActorFuture]],
) -> dict[CallKey, ActorFuture | None]:
    """Start actor calls, retaining unavailable endpoints for deadline-bound retry."""
    futures: dict[CallKey, ActorFuture | None] = {}
    for call_key, call in calls.items():
        try:
            futures[call_key] = call()
        except ActorUnavailableError:
            futures[call_key] = None
    return futures


@dataclass(frozen=True)
class MemoryStore(Generic[K, V]):
    """Picklable reference to one table hosted by a Zephyr worker pool."""

    table_id: str
    name: str
    actors: tuple[ActorHandle, ...]
    shard_actors: tuple[MemoryStoreShardHandle, ...]
    coordinator: ActorHandle = field(repr=False)
    hash_key: Callable[[K], int] = field(repr=False)
    num_source_partitions: int
    shards_per_worker: int
    recovery_timeout: float
    load_elapsed: float

    def get(self, key: K) -> V:
        """Return the value for one key or raise `KeyError`."""
        return self.get_many([key])[0]

    def _reload(self, actor_index: int, deadline: float) -> None:
        actor = self.actors[actor_index]

        def call() -> ActorFuture:
            return actor.reload_memory_table.submit(self.table_id)

        try:
            initial_future = call()
        except ActorUnavailableError:
            initial_future = None
        stats = actor_result_with_recovery(
            call,
            initial_future,
            actor_index,
            self.recovery_timeout,
            deadline,
        )
        if stats is None:
            raise MemoryStoreDestroyed(f"memory table {self.name!r} has been destroyed")
        for stat in stats:
            shard_index = stat.actor_index * self.shards_per_worker + stat.store_shard_index
            self.shard_actors[shard_index].refresh(stat.endpoint_name, stat.endpoint_address)

    def _ready_result(
        self,
        actor_index: int,
        call: Callable[[], ActorFuture],
        initial_future: ActorFuture | None,
        deadline: float,
        recover_subprocess: bool = False,
    ) -> _MemoryTableResult:
        future = initial_future
        while True:
            result: _MemoryTableResult = actor_result_with_recovery(
                call,
                future,
                actor_index,
                self.recovery_timeout,
                deadline,
                recover=(lambda: self._reload(actor_index, deadline)) if recover_subprocess else None,
            )
            if result.status is MemoryTableStatus.READY:
                return result
            if result.status is MemoryTableStatus.DESTROYED:
                raise MemoryStoreDestroyed(f"memory table {self.name!r} has been destroyed")
            self._reload(actor_index, deadline)
            future = None

    def get_many(self, keys: Sequence[K]) -> list[V]:
        """Return values aligned with `keys`, batching calls by owning worker."""
        if not keys:
            return []

        requests: dict[int, list[tuple[int, K]]] = {}
        store_shard_count = len(self.shard_actors)
        for position, key in enumerate(keys):
            source_partition = _source_partition(self.hash_key, key, self.num_source_partitions)
            global_shard_index = source_partition % store_shard_count
            requests.setdefault(global_shard_index, []).append((position, key))

        request_keys = {shard: [key for _, key in shard_requests] for shard, shard_requests in requests.items()}
        calls = {
            shard: lambda shard=shard: self.shard_actors[shard].lookup.remote(self.table_id, request_keys[shard])
            for shard in requests
        }
        deadline = time.monotonic() + self.recovery_timeout
        futures = start_actor_calls(calls)

        results: list[V | None] = [None] * len(keys)
        for shard, shard_requests in requests.items():
            actor_index, _ = divmod(shard, self.shards_per_worker)
            result = self._ready_result(
                actor_index,
                calls[shard],
                futures[shard],
                deadline,
                recover_subprocess=isinstance(self.shard_actors[shard], MemoryStoreShardHandle),
            )
            assert isinstance(result, MemoryTableLookup)
            lookup_results = result.values
            assert len(lookup_results) == len(shard_requests)
            for (position, key), (found, value) in zip(shard_requests, lookup_results, strict=True):
                if not found:
                    raise KeyError(key)
                results[position] = value

        return cast(list[V], results)

    def compute_many(
        self,
        requests: Sequence[tuple[K, A]],
        function: Callable[[list[tuple[V, A]]], list[R]],
    ) -> list[R]:
        """Compute over values on their owning workers and preserve request order.

        The function receives one worker-local batch of ``(value, payload)`` pairs.
        It must be deterministic and return one result per input so actor recovery
        can safely retry the batch.
        """
        if not requests:
            return []

        routed: dict[int, list[tuple[int, K, A]]] = {}
        store_shard_count = len(self.shard_actors)
        for position, (key, payload) in enumerate(requests):
            source_partition = _source_partition(self.hash_key, key, self.num_source_partitions)
            global_shard_index = source_partition % store_shard_count
            routed.setdefault(global_shard_index, []).append((position, key, payload))

        actor_requests = {shard: [(key, payload) for _, key, payload in items] for shard, items in routed.items()}
        calls = {
            shard: (
                lambda shard=shard: self.shard_actors[shard].compute.remote(
                    self.table_id, actor_requests[shard], function
                )
            )
            for shard in routed
        }
        deadline = time.monotonic() + self.recovery_timeout
        futures = start_actor_calls(calls)

        results: list[R | None] = [None] * len(requests)
        for shard, items in routed.items():
            actor_index, local_shard_index = divmod(shard, self.shards_per_worker)
            result = self._ready_result(
                actor_index,
                calls[shard],
                futures[shard],
                deadline,
                recover_subprocess=isinstance(self.shard_actors[shard], MemoryStoreShardHandle),
            )
            assert isinstance(result, MemoryTableCompute)
            if len(result.values) != len(items):
                raise ValueError(
                    f"memory-store actor {actor_index} shard {local_shard_index} returned "
                    f"{len(result.values)} results for {len(items)} requests"
                )
            for (position, key, _), (found, value) in zip(items, result.values, strict=True):
                if not found:
                    raise KeyError(key)
                results[position] = value
        return cast(list[R], results)

    def stats(self) -> tuple[MemoryStoreShardStats, ...]:
        """Return load statistics ordered by worker index."""
        calls = {
            actor_index: lambda actor=actor: actor.memory_table_stats.remote(self.table_id)
            for actor_index, actor in enumerate(self.actors)
        }
        deadline = time.monotonic() + self.recovery_timeout
        futures = start_actor_calls(calls)
        stats: list[MemoryStoreShardStats] = []
        for actor_index in range(len(self.actors)):
            result = self._ready_result(actor_index, calls[actor_index], futures[actor_index], deadline)
            assert isinstance(result, MemoryTableStatsBatch)
            stats.extend(result.stats)
        return tuple(sorted(stats, key=lambda stat: (stat.actor_index, stat.store_shard_index)))

    def destroy(self) -> None:
        """Remove this table from the worker pool."""
        deadline = time.monotonic() + self.recovery_timeout

        def unregister() -> ActorFuture:
            return self.coordinator.unregister_memory_table.remote(self.table_id)

        try:
            unregister_future = unregister()
        except ActorUnavailableError:
            unregister_future = None
        actor_result_with_recovery(unregister, unregister_future, -1, self.recovery_timeout, deadline)

        calls = {
            actor_index: lambda actor=actor: actor.destroy_memory_table.remote(self.table_id)
            for actor_index, actor in enumerate(self.actors)
        }
        futures = start_actor_calls(calls)
        for actor_index in range(len(self.actors)):
            actor_result_with_recovery(
                calls[actor_index],
                futures[actor_index],
                actor_index,
                self.recovery_timeout,
                deadline,
            )
