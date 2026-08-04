# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shard-preserving memory tables hosted by Zephyr workers."""

import enum
import logging
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from fray.actor import ActorFuture, ActorHandle, ActorUnavailableError
from rigging.timing import ExponentialBackoff

from zephyr.dataset import Dataset
from zephyr.plan import Map, PhysicalOp, SourceItem, StageContext, StageType, compute_plan, run_stage

logger = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class MemoryStorePartitionError(ValueError):
    """Raised when a key does not belong to its existing Dataset partition."""


class DuplicateMemoryStoreKey(ValueError):
    """Raised when a memory-store input contains the same key twice."""


class MemoryStoreUnavailable(RuntimeError):
    """Raised when a memory-store worker does not recover before its timeout."""


class MemoryStoreDestroyed(RuntimeError):
    """Raised when a lookup uses a table that has been destroyed."""


@dataclass(frozen=True)
class MemoryStoreActorStats:
    """Load statistics for one worker's copy of a memory table."""

    actor_index: int
    source_partitions: tuple[int, ...]
    num_items: int
    load_cpu_time: float
    load_elapsed: float


def _source_partition(hash_key: Callable[[K], int], key: K, num_source_partitions: int) -> int:
    key_hash = hash_key(key)
    if isinstance(key_hash, bool) or not isinstance(key_hash, int):
        raise TypeError(f"hash_key must return int, got {type(key_hash).__name__} for key {key!r}")
    return key_hash % num_source_partitions


@dataclass(frozen=True)
class _MemoryStorePlan:
    source_items: tuple[SourceItem, ...]
    operations: tuple[PhysicalOp, ...]
    num_source_partitions: int


def _store_plan(dataset: Dataset[tuple[K, V]]) -> _MemoryStorePlan:
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

    return _MemoryStorePlan(
        source_items=tuple(plan.source_items),
        operations=operations,
        num_source_partitions=num_source_partitions,
    )


@dataclass(frozen=True)
class _MemoryTableRegistration:
    table_id: str
    name: str
    plan: _MemoryStorePlan
    hash_key: Callable[[Any], int] = field(repr=False)
    worker_count: int


class _MemoryTableStatus(enum.StrEnum):
    READY = enum.auto()
    NOT_LOADED = enum.auto()
    UNKNOWN = enum.auto()
    DESTROYED = enum.auto()


@dataclass(frozen=True)
class _MemoryTableLookup:
    status: _MemoryTableStatus
    values: list[tuple[bool, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _MemoryTableStatsResult:
    status: _MemoryTableStatus
    stats: MemoryStoreActorStats | None = None


type _MemoryTableResult = _MemoryTableLookup | _MemoryTableStatsResult


@dataclass
class _MemoryTableState:
    registration: _MemoryTableRegistration
    load_lock: threading.Lock = field(default_factory=threading.Lock)
    values: dict[Hashable, Any] | None = None
    stats: MemoryStoreActorStats | None = None


class _MemoryStoreService:
    """Worker-local service that owns several immutable memory tables."""

    def __init__(
        self,
        actor_index: int,
    ):
        self._actor_index = actor_index
        self._tables: dict[str, _MemoryTableState] = {}
        self._destroyed: set[str] = set()
        self._tables_lock = threading.Lock()

    def _install(self, registration: _MemoryTableRegistration) -> _MemoryTableState:
        with self._tables_lock:
            if registration.table_id in self._destroyed:
                raise MemoryStoreDestroyed(f"memory table {registration.name!r} has been destroyed")
            state = self._tables.get(registration.table_id)
            if state is not None:
                return state
            state = _MemoryTableState(registration=registration)
            self._tables[registration.table_id] = state
            return state

    def restore(self, registrations: tuple[_MemoryTableRegistration, ...]) -> None:
        """Install active table metadata without reading source data."""
        with self._tables_lock:
            for registration in registrations:
                if registration.table_id in self._destroyed or registration.table_id in self._tables:
                    continue
                self._tables[registration.table_id] = _MemoryTableState(registration=registration)

    def _load_values(self, registration: _MemoryTableRegistration) -> tuple[dict[Hashable, Any], tuple[int, ...]]:
        plan = registration.plan
        partitions = tuple(
            partition
            for partition in range(plan.num_source_partitions)
            if partition % registration.worker_count == self._actor_index
        )
        source_data: dict[int, list[Any]] = {partition: [] for partition in partitions}
        for item in plan.source_items:
            if item.shard_idx in source_data:
                source_data[item.shard_idx].append(item.data)

        values: dict[Hashable, Any] = {}
        for partition, shard in source_data.items():
            context = StageContext(
                shard=shard,
                shard_idx=partition,
                total_shards=plan.num_source_partitions,
            )
            for item in run_stage(context, list(plan.operations)):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise TypeError(
                        "memory-store Dataset must yield (key, value) tuples; "
                        f"partition {partition} yielded {type(item).__name__}"
                    )
                key, value = item
                actual_partition = _source_partition(registration.hash_key, key, plan.num_source_partitions)
                if actual_partition != partition:
                    raise MemoryStorePartitionError(
                        f"key {key!r} was in source partition {partition}, but hash_key routes it to "
                        f"partition {actual_partition}"
                    )
                if key in values:
                    raise DuplicateMemoryStoreKey(f"duplicate memory-store key {key!r} in partition {partition}")
                values[key] = value
        return values, partitions

    def load(self, registration: _MemoryTableRegistration) -> MemoryStoreActorStats:
        """Load one table, or return its existing load statistics."""
        state = self._install(registration)
        with state.load_lock:
            if state.values is not None:
                assert state.stats is not None
                return state.stats

            load_started = time.monotonic()
            load_cpu_started = time.process_time()
            values, partitions = self._load_values(registration)
            stats = MemoryStoreActorStats(
                actor_index=self._actor_index,
                source_partitions=partitions,
                num_items=len(values),
                load_cpu_time=time.process_time() - load_cpu_started,
                load_elapsed=time.monotonic() - load_started,
            )
            with self._tables_lock:
                if registration.table_id in self._destroyed:
                    raise MemoryStoreDestroyed(f"memory table {registration.name!r} has been destroyed")
                state.values = values
                state.stats = stats

            logger.info(
                "Memory table %s worker %d loaded %d items from source partitions %s "
                "in %.2f CPU-seconds and %.2f seconds",
                registration.name,
                self._actor_index,
                len(values),
                partitions,
                stats.load_cpu_time,
                stats.load_elapsed,
            )
            return stats

    def lookup(self, table_id: str, keys: list[Hashable]) -> _MemoryTableLookup:
        """Return table state or values aligned with the requested keys."""
        with self._tables_lock:
            if table_id in self._destroyed:
                return _MemoryTableLookup(_MemoryTableStatus.DESTROYED)
            state = self._tables.get(table_id)
            if state is None:
                return _MemoryTableLookup(_MemoryTableStatus.UNKNOWN)
            values = state.values
        if values is None:
            return _MemoryTableLookup(_MemoryTableStatus.NOT_LOADED)
        return _MemoryTableLookup(
            _MemoryTableStatus.READY,
            [(True, values[key]) if key in values else (False, None) for key in keys],
        )

    def stats(self, table_id: str) -> _MemoryTableStatsResult:
        """Return table state or immutable load statistics."""
        with self._tables_lock:
            if table_id in self._destroyed:
                return _MemoryTableStatsResult(_MemoryTableStatus.DESTROYED)
            state = self._tables.get(table_id)
            if state is None:
                return _MemoryTableStatsResult(_MemoryTableStatus.UNKNOWN)
            stats = state.stats
        if stats is None:
            return _MemoryTableStatsResult(_MemoryTableStatus.NOT_LOADED)
        return _MemoryTableStatsResult(_MemoryTableStatus.READY, stats)

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


def _actor_result_with_recovery(
    call: Callable[[], ActorFuture],
    initial_future: ActorFuture | None,
    actor_index: int,
    recovery_timeout: float,
    deadline: float,
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
            delay = min(backoff.next_interval(), remaining)
            logger.warning("Memory-store actor %d unavailable; retrying in %.1f seconds", actor_index, delay)
            time.sleep(delay)


@dataclass(frozen=True)
class MemoryStore(Generic[K, V]):
    """Picklable reference to one table hosted by a Zephyr worker pool."""

    table_id: str
    name: str
    actors: tuple[ActorHandle, ...]
    coordinator: ActorHandle = field(repr=False)
    hash_key: Callable[[K], int] = field(repr=False)
    num_source_partitions: int
    recovery_timeout: float

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
        stats = _actor_result_with_recovery(
            call,
            initial_future,
            actor_index,
            self.recovery_timeout,
            deadline,
        )
        if stats is None:
            raise MemoryStoreDestroyed(f"memory table {self.name!r} has been destroyed")

    def _ready_result(
        self,
        actor_index: int,
        call: Callable[[], ActorFuture],
        initial_future: ActorFuture | None,
        deadline: float,
    ) -> _MemoryTableResult:
        future = initial_future
        while True:
            result: _MemoryTableResult = _actor_result_with_recovery(
                call,
                future,
                actor_index,
                self.recovery_timeout,
                deadline,
            )
            if result.status is _MemoryTableStatus.READY:
                return result
            if result.status is _MemoryTableStatus.DESTROYED:
                raise MemoryStoreDestroyed(f"memory table {self.name!r} has been destroyed")
            self._reload(actor_index, deadline)
            future = None

    def get_many(self, keys: Sequence[K]) -> list[V]:
        """Return values aligned with `keys`, batching calls by owning worker."""
        if not keys:
            return []

        requests: dict[int, list[tuple[int, K]]] = {}
        for position, key in enumerate(keys):
            source_partition = _source_partition(self.hash_key, key, self.num_source_partitions)
            actor_index = source_partition % len(self.actors)
            requests.setdefault(actor_index, []).append((position, key))

        request_keys = {
            actor_index: [key for _, key in actor_requests] for actor_index, actor_requests in requests.items()
        }
        calls = {
            actor_index: (
                lambda actor_index=actor_index: self.actors[actor_index].lookup_memory_table.remote(
                    self.table_id, request_keys[actor_index]
                )
            )
            for actor_index in requests
        }
        deadline = time.monotonic() + self.recovery_timeout
        futures: dict[int, ActorFuture | None] = {}
        for actor_index, call in calls.items():
            try:
                futures[actor_index] = call()
            except ActorUnavailableError:
                futures[actor_index] = None

        results: list[V | None] = [None] * len(keys)
        for actor_index, actor_requests in requests.items():
            result = self._ready_result(
                actor_index,
                calls[actor_index],
                futures[actor_index],
                deadline,
            )
            assert isinstance(result, _MemoryTableLookup)
            lookup_results = result.values
            assert len(lookup_results) == len(actor_requests)
            for (position, key), (found, value) in zip(actor_requests, lookup_results, strict=True):
                if not found:
                    raise KeyError(key)
                results[position] = value

        return cast(list[V], results)

    def stats(self) -> tuple[MemoryStoreActorStats, ...]:
        """Return load statistics ordered by worker index."""
        calls = {
            actor_index: lambda actor=actor: actor.memory_table_stats.remote(self.table_id)
            for actor_index, actor in enumerate(self.actors)
        }
        deadline = time.monotonic() + self.recovery_timeout
        futures: dict[int, ActorFuture | None] = {}
        for actor_index, call in calls.items():
            try:
                futures[actor_index] = call()
            except ActorUnavailableError:
                futures[actor_index] = None
        stats: list[MemoryStoreActorStats] = []
        for actor_index in range(len(self.actors)):
            result = self._ready_result(actor_index, calls[actor_index], futures[actor_index], deadline)
            assert isinstance(result, _MemoryTableStatsResult)
            assert result.stats is not None
            stats.append(result.stats)
        return tuple(sorted(stats, key=lambda stat: stat.actor_index))

    def destroy(self) -> None:
        """Remove this table from the worker pool."""
        deadline = time.monotonic() + self.recovery_timeout

        def unregister() -> ActorFuture:
            return self.coordinator.unregister_memory_table.remote(self.table_id)

        try:
            unregister_future = unregister()
        except ActorUnavailableError:
            unregister_future = None
        _actor_result_with_recovery(unregister, unregister_future, -1, self.recovery_timeout, deadline)

        calls = {
            actor_index: lambda actor=actor: actor.destroy_memory_table.remote(self.table_id)
            for actor_index, actor in enumerate(self.actors)
        }
        futures: dict[int, ActorFuture | None] = {}
        for actor_index, call in calls.items():
            try:
                futures[actor_index] = call()
            except ActorUnavailableError:
                futures[actor_index] = None
        for actor_index in range(len(self.actors)):
            _actor_result_with_recovery(
                calls[actor_index],
                futures[actor_index],
                actor_index,
                self.recovery_timeout,
                deadline,
            )
