# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only actor stores over existing Zephyr Dataset partitions."""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

import cloudpickle
import msgspec
from fray.actor import ActorFuture, ActorHandle, ActorUnavailableError, current_actor
from rigging.timing import ExponentialBackoff

from zephyr.dataset import Dataset
from zephyr.plan import Map, PhysicalOp, SourceItem, StageContext, StageType, compute_plan, run_stage

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class MemoryStorePartitionError(ValueError):
    """Raised when a key does not belong to its existing Dataset partition."""


class DuplicateMemoryStoreKey(ValueError):
    """Raised when a memory-store input contains the same key twice."""


class MemoryStoreCapacityError(MemoryError):
    """Raised when an actor's encoded data exceeds its configured budget."""


class MemoryStoreUnavailable(RuntimeError):
    """Raised when a memory-store actor does not recover before its timeout."""


@dataclass(frozen=True)
class MemoryStoreActorStats:
    """Load statistics for one memory-store actor."""

    actor_index: int
    source_partitions: tuple[int, ...]
    num_items: int
    serialized_bytes: int
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


class _MemoryStoreActor:
    def __init__(
        self,
        source_items: tuple[SourceItem, ...],
        operations: tuple[PhysicalOp, ...],
        hash_key: Callable[[Any], int],
        num_source_partitions: int,
        num_actors: int,
        max_actor_bytes: int,
    ):
        load_started = time.monotonic()
        load_cpu_started = time.process_time()
        actor_index = current_actor().index
        source_data: dict[int, list[Any]] = {
            partition: [] for partition in range(num_source_partitions) if partition % num_actors == actor_index
        }
        for item in source_items:
            if item.shard_idx in source_data:
                source_data[item.shard_idx].append(item.data)

        values: dict[bytes, bytes] = {}
        serialized_bytes = 0
        for partition, shard in source_data.items():
            context = StageContext(
                shard=shard,
                shard_idx=partition,
                total_shards=num_source_partitions,
            )
            for item in run_stage(context, list(operations)):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise TypeError(
                        "memory-store Dataset must yield (key, value) tuples; "
                        f"partition {partition} yielded {type(item).__name__}"
                    )
                key, value = item
                actual_partition = _source_partition(hash_key, key, num_source_partitions)
                if actual_partition != partition:
                    raise MemoryStorePartitionError(
                        f"key {key!r} was in source partition {partition}, but hash_key routes it to "
                        f"partition {actual_partition}"
                    )

                encoded_key = msgspec.msgpack.encode(key, order="deterministic")
                if encoded_key in values:
                    raise DuplicateMemoryStoreKey(f"duplicate memory-store key {key!r} in partition {partition}")
                encoded_value = cloudpickle.dumps(value)
                serialized_bytes += len(encoded_key) + len(encoded_value)
                if serialized_bytes > max_actor_bytes:
                    raise MemoryStoreCapacityError(
                        f"memory-store actor {actor_index} exceeded max_actor_bytes={max_actor_bytes:,} "
                        f"while loading source partition {partition}"
                    )
                values[encoded_key] = encoded_value

        self._values = values
        self._stats = MemoryStoreActorStats(
            actor_index=actor_index,
            source_partitions=tuple(source_data),
            num_items=len(values),
            serialized_bytes=serialized_bytes,
            load_cpu_time=time.process_time() - load_cpu_started,
            load_elapsed=time.monotonic() - load_started,
        )
        logger.info(
            "Memory-store actor %d loaded %d items (%d bytes) from source partitions %s "
            "in %.2f CPU-seconds and %.2f seconds",
            actor_index,
            len(values),
            serialized_bytes,
            tuple(source_data),
            self._stats.load_cpu_time,
            self._stats.load_elapsed,
        )

    def lookup(self, encoded_keys: list[bytes]) -> list[bytes | None]:
        """Return encoded values aligned with the requested encoded keys."""
        return [self._values.get(key) for key in encoded_keys]

    def stats(self) -> MemoryStoreActorStats:
        """Return immutable load statistics."""
        return self._stats


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
    """Picklable handle for a partitioned, read-only actor store."""

    actors: tuple[ActorHandle, ...]
    hash_key: Callable[[K], int] = field(repr=False)
    num_source_partitions: int
    recovery_timeout: float

    def get(self, key: K) -> V:
        """Return the value for one key or raise `KeyError`."""
        return self.get_many([key])[0]

    def get_many(self, keys: Sequence[K]) -> list[V]:
        """Return values aligned with `keys`, batching calls by owning actor."""
        if not keys:
            return []

        requests: dict[int, list[tuple[int, K, bytes]]] = {}
        for position, key in enumerate(keys):
            source_partition = _source_partition(self.hash_key, key, self.num_source_partitions)
            actor_index = source_partition % len(self.actors)
            requests.setdefault(actor_index, []).append(
                (position, key, msgspec.msgpack.encode(key, order="deterministic"))
            )

        encoded_requests = {
            actor_index: [encoded_key for _, _, encoded_key in actor_requests]
            for actor_index, actor_requests in requests.items()
        }
        calls = {
            actor_index: (
                lambda actor_index=actor_index: self.actors[actor_index].lookup.remote(encoded_requests[actor_index])
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
            encoded_values = _actor_result_with_recovery(
                calls[actor_index],
                futures[actor_index],
                actor_index,
                self.recovery_timeout,
                deadline,
            )
            assert len(encoded_values) == len(actor_requests)
            for (position, key, _), encoded_value in zip(actor_requests, encoded_values, strict=True):
                if encoded_value is None:
                    raise KeyError(key)
                results[position] = cloudpickle.loads(encoded_value)

        return cast(list[V], results)

    def stats(self) -> tuple[MemoryStoreActorStats, ...]:
        """Return load statistics ordered by actor index."""
        calls = {actor_index: lambda actor=actor: actor.stats.remote() for actor_index, actor in enumerate(self.actors)}
        deadline = time.monotonic() + self.recovery_timeout
        futures: dict[int, ActorFuture | None] = {}
        for actor_index, call in calls.items():
            try:
                futures[actor_index] = call()
            except ActorUnavailableError:
                futures[actor_index] = None
        stats = []
        for actor_index in range(len(self.actors)):
            stats.append(
                _actor_result_with_recovery(
                    calls[actor_index],
                    futures[actor_index],
                    actor_index,
                    self.recovery_timeout,
                    deadline,
                )
            )
        return tuple(sorted(stats, key=lambda stat: stat.actor_index))
