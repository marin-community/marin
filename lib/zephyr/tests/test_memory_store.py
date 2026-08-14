# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for read-only Zephyr memory stores."""

import os
import sys
from typing import Any, cast
from unittest.mock import MagicMock

import aiohttp
import cloudpickle
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.actor import ActorHandle, ActorUnavailableError
from fray.local_backend import LocalClient
from iris.test_util import SentinelFile
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, _order_memory_store_handles, _require_resolvable_worker_handles
from zephyr.memory_store import (
    DuplicateMemoryStoreKey,
    MemoryStore,
    MemoryStoreDestroyed,
    MemoryStorePartitionError,
    MemoryStoreShardStats,
    MemoryStoreUnavailable,
    MemoryTableLookup,
    MemoryTableStatus,
)
from zephyr.testing.context import memory_store_context


class _TestActorFuture:
    def __init__(self, value: Any = None, error: Exception | None = None):
        self.value = value
        self.error = error
        self.timeouts: list[float | None] = []

    def result(self, timeout: float | None = None) -> Any:
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return self.value


class _ApplicationError(ValueError):
    pass


class _SequencedActorMethod:
    def __init__(self, futures: list[_TestActorFuture]):
        self.futures = futures
        self.call_count = 0

    def remote(self, *args: Any, **kwargs: Any) -> _TestActorFuture:
        del args, kwargs
        future = self.futures[self.call_count]
        self.call_count += 1
        return future


class _SequencedActor:
    def __init__(self, futures: list[_TestActorFuture]):
        self.lookup = _SequencedActorMethod(futures)


def _partition_rows(rows):
    yield from rows


def _key_partition(key: tuple[int, str]) -> int:
    return key[0]


def _wrong_key_partition(key: tuple[int, str]) -> int:
    return key[0] + 1


def _parquet_pair(row: dict) -> tuple[tuple[int, str], str]:
    return (row["partition"], row["id"]), row["text"]


def _append_payload(inputs: list[tuple[str, str]]) -> list[str]:
    return [value + payload for value, payload in inputs]


def _exit_store_process_once(inputs: list[tuple[str, str]]) -> list[str]:
    sentinel = SentinelFile(inputs[0][1])
    if not sentinel.is_set():
        sentinel.signal()
        os._exit(17)
    return [value for value, _ in inputs]


@pytest.fixture(autouse=True)
def _pickle_test_callables_by_value():
    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    try:
        yield
    finally:
        cloudpickle.unregister_pickle_by_value(module)


def _load_store(
    context: ZephyrContext,
    dataset: Dataset,
    *,
    name: str = "documents",
    hash_key=_key_partition,
    recovery_timeout: float = 10,
    shards_per_worker: int = 1,
    load_concurrency: int = 1,
):
    return context.load_memory_store(
        dataset,
        name=name,
        hash_key=hash_key,
        recovery_timeout=recovery_timeout,
        shards_per_worker=shards_per_worker,
        load_concurrency=load_concurrency,
    )


def _fake_store(actor: _SequencedActor, recovery_timeout: float = 1) -> MemoryStore[str, str]:
    return MemoryStore(
        table_id="table",
        name="table",
        actors=(cast(ActorHandle, actor),),
        shard_actors=(cast(ActorHandle, actor),),
        coordinator=cast(ActorHandle, object()),
        hash_key=lambda _key: 0,
        num_source_partitions=1,
        shards_per_worker=1,
        recovery_timeout=recovery_timeout,
        load_elapsed=0,
    )


def _actor_stats(actor_index: int) -> tuple[MemoryStoreShardStats, ...]:
    return (
        MemoryStoreShardStats(
            actor_index=actor_index,
            store_shard_index=0,
            process_id=actor_index,
            source_partitions=(),
            num_items=0,
            load_cpu_time=0,
            load_elapsed=0,
            resident_bytes=0,
            peak_resident_bytes=0,
        ),
    )


def test_memory_store_routes_existing_partitions_and_preserves_lookup_order(local_client, tmp_path):
    partitions = [
        [((0, "a"), "zero-a"), ((0, "b"), "zero-b"), ((0, "none"), None)],
        [((1, "a"), "one-a")],
        [((2, "a"), "two-a"), ((2, "b"), "two-b")],
        [((3, "a"), "three-a")],
    ]
    dataset = Dataset.from_list(partitions).flat_map(_partition_rows)

    with memory_store_context(local_client, tmp_path) as context:
        store = _load_store(context, dataset)

        assert store.get_many([(3, "a"), (0, "b"), (2, "a"), (0, "b")]) == [
            "three-a",
            "zero-b",
            "two-a",
            "zero-b",
        ]
        assert store.get((1, "a")) == "one-a"
        assert store.get((0, "none")) is None
        with pytest.raises(KeyError) as exc_info:
            store.get((1, "missing"))
        assert exc_info.value.args == ((1, "missing"),)

        stats = store.stats()
        assert [(stat.actor_index, stat.source_partitions, stat.num_items) for stat in stats] == [
            (0, (0, 2), 5),
            (1, (1, 3), 2),
        ]


def test_memory_store_orders_worker_handles_by_reported_actor_index():
    handles = (cast(ActorHandle, object()), cast(ActorHandle, object()), cast(ActorHandle, object()))

    ordered = _order_memory_store_handles(handles, [_actor_stats(2), _actor_stats(0), _actor_stats(1)])

    assert ordered == (handles[1], handles[2], handles[0])


def test_memory_store_spreads_adjacent_shards_across_workers(local_client, tmp_path):
    partitions = [[((partition, "key"), str(partition))] for partition in range(8)]
    dataset = Dataset.from_list(partitions).flat_map(_partition_rows)

    with memory_store_context(local_client, tmp_path, max_workers=2) as context:
        store = _load_store(context, dataset, shards_per_worker=2)

        assert store.get_many([(partition, "key") for partition in range(8)]) == [
            str(partition) for partition in range(8)
        ]
        assert [(stat.actor_index, stat.store_shard_index, stat.source_partitions) for stat in store.stats()] == [
            (0, 0, (0, 4)),
            (0, 1, (2, 6)),
            (1, 0, (1, 5)),
            (1, 1, (3, 7)),
        ]


def test_memory_store_computes_on_owning_workers_and_preserves_order(local_client, tmp_path):
    partitions = [
        [((0, "a"), "zero-a"), ((0, "b"), "zero-b")],
        [((1, "a"), "one-a")],
        [((2, "a"), "two-a")],
        [((3, "a"), "three-a")],
    ]
    dataset = Dataset.from_list(partitions).flat_map(_partition_rows)

    with memory_store_context(local_client, tmp_path) as context:
        store = _load_store(context, dataset)

        assert store.compute_many(
            [((3, "a"), "!"), ((0, "b"), "?"), ((2, "a"), ".")],
            _append_payload,
        ) == ["three-a!", "zero-b?", "two-a."]
        with pytest.raises(KeyError) as exc_info:
            store.compute_many([((1, "missing"), "!")], _append_payload)
        assert exc_info.value.args == ((1, "missing"),)


def test_memory_store_shards_own_values_in_separate_subprocesses(local_client, tmp_path):
    dataset = Dataset.from_list(
        [
            [((0, "a"), "zero-a")],
            [((1, "a"), "one-a")],
        ]
    ).flat_map(_partition_rows)

    with memory_store_context(local_client, tmp_path, max_workers=1) as context:
        store = _load_store(context, dataset, shards_per_worker=2)
        values = store.get_many([(1, "a"), (0, "a")])
        stats = store.stats()

        assert values == ["one-a", "zero-a"]
        process_ids = {stat.process_id for stat in stats}
        assert len(process_ids) == 2
        assert os.getpid() not in process_ids


def test_memory_store_recovers_a_failed_subprocess(local_client, tmp_path):
    dataset = Dataset.from_list([((0, "a"), "zero-a")])
    crash_sentinel = str(tmp_path / "store-crashed")

    with memory_store_context(local_client, tmp_path, max_workers=1) as context:
        store = _load_store(context, dataset)

        assert store.compute_many([((0, "a"), crash_sentinel)], _exit_store_process_once) == ["zero-a"]
        assert SentinelFile(crash_sentinel).is_set()


def test_memory_store_returns_responses_larger_than_pipe_buffer(local_client, tmp_path):
    values = [bytes([index % 251]) * 4096 + str(index).encode() for index in range(256)]
    dataset = Dataset.from_list([[((0, str(index)), value) for index, value in enumerate(values)]]).flat_map(
        _partition_rows
    )

    with memory_store_context(local_client, tmp_path, max_workers=1) as context:
        store = _load_store(context, dataset)

        assert store.get_many([(0, str(index)) for index in range(256)]) == values


def test_memory_store_rejects_nonpositive_load_concurrency(local_client, tmp_path):
    dataset = Dataset.from_list([((0, "a"), "value")])

    with memory_store_context(local_client, tmp_path, max_workers=1) as context:
        with pytest.raises(ValueError, match="load_concurrency must be at least 1"):
            _load_store(context, dataset, load_concurrency=0)


def test_memory_store_retries_truncated_remote_partition_read(local_client, tmp_path):
    first_attempt = SentinelFile(str(tmp_path / "first-attempt"))

    def truncate_once(item):
        if not first_attempt.is_set():
            first_attempt.signal()
            raise aiohttp.ClientPayloadError("response payload is incomplete")
        return item

    dataset = Dataset.from_list([((0, "a"), "value")]).map(truncate_once)

    with memory_store_context(local_client, tmp_path, max_workers=1) as context:
        store = _load_store(context, dataset)

        assert first_attempt.is_set()
        assert store.get((0, "a")) == "value"


def test_memory_store_rejects_hash_that_disagrees_with_existing_partition(local_client, tmp_path):
    dataset = Dataset.from_list([[((0, "a"), "value")], [((1, "b"), "value")]]).flat_map(_partition_rows)

    with memory_store_context(local_client, tmp_path) as context:
        with pytest.raises(MemoryStorePartitionError):
            _load_store(context, dataset, hash_key=_wrong_key_partition)


def test_memory_store_rejects_duplicate_key_without_poisoning_worker(local_client, tmp_path):
    duplicate = Dataset.from_list([[((0, "same"), "first"), ((0, "same"), "second")]]).flat_map(_partition_rows)
    valid = Dataset.from_list([((0, "valid"), "value")])

    with memory_store_context(local_client, tmp_path) as context:
        with pytest.raises(DuplicateMemoryStoreKey):
            _load_store(context, duplicate, name="duplicates")

        store = _load_store(context, valid, name="valid")
        assert store.get((0, "valid")) == "value"


def test_memory_store_multiple_tables_have_independent_lifetimes(local_client, tmp_path):
    first_dataset = Dataset.from_list([((0, "key"), "first")])
    second_dataset = Dataset.from_list([((0, "key"), "second")])

    with memory_store_context(local_client, tmp_path) as context:
        first = _load_store(context, first_dataset, name="first")
        second = _load_store(context, second_dataset, name="second")

        assert first.get((0, "key")) == "first"
        assert second.get((0, "key")) == "second"

        first.destroy()
        first.destroy()
        with pytest.raises(MemoryStoreDestroyed):
            first.get((0, "key"))
        assert second.get((0, "key")) == "second"


def test_memory_store_pickle_round_trip_works_in_later_pipelines(local_client, tmp_path):
    parquet_dir = tmp_path / "input"
    parquet_dir.mkdir()
    for partition in range(4):
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"partition": partition, "id": "a", "text": f"text-{partition}-a"},
                    {"partition": partition, "id": "b", "text": f"text-{partition}-b"},
                ]
            ),
            parquet_dir / f"part-{partition:02d}.parquet",
        )

    dataset = Dataset.from_files(str(parquet_dir / "*.parquet")).load_parquet().map(_parquet_pair)
    keys = [(3, "b"), (0, "a"), (2, "b"), (1, "a")]

    with memory_store_context(local_client, tmp_path) as context:
        store = _load_store(context, dataset)
        restored = cloudpickle.loads(cloudpickle.dumps(store))

        first_result = context.execute(Dataset.from_list(keys).map(restored.get))
        second_result = context.execute(Dataset.from_list(list(reversed(keys))).map(restored.get))

    assert first_result.results == ["text-3-b", "text-0-a", "text-2-b", "text-1-a"]
    assert second_result.results == ["text-1-a", "text-2-b", "text-0-a", "text-3-b"]


def test_memory_store_retries_only_actor_unavailability():
    recovering_actor = _SequencedActor(
        [
            _TestActorFuture(error=ActorUnavailableError("restarting")),
            _TestActorFuture(value=MemoryTableLookup(MemoryTableStatus.READY, [(True, "value")])),
        ]
    )
    assert _fake_store(recovering_actor).get("key") == "value"

    failing_actor = _SequencedActor([_TestActorFuture(error=_ApplicationError())])
    with pytest.raises(_ApplicationError):
        _fake_store(failing_actor).get("key")


def test_memory_store_bounds_actor_call_by_recovery_timeout():
    timed_out_future = _TestActorFuture(error=TimeoutError("actor call exceeded its deadline"))
    store = _fake_store(_SequencedActor([timed_out_future]))

    with pytest.raises(MemoryStoreUnavailable):
        store.get("key")

    assert len(timed_out_future.timeouts) == 1
    assert timed_out_future.timeouts[0] is not None
    assert 0 < timed_out_future.timeouts[0] <= 1


def test_memory_store_requires_an_entered_owning_context(local_client, tmp_path):
    context = ZephyrContext(
        client=local_client,
        max_workers=1,
        chunk_storage_prefix=str(tmp_path / "chunks"),
    )

    with pytest.raises(RuntimeError):
        _load_store(context, Dataset.from_list([((0, "a"), "value")]))


def test_context_shutdown_makes_store_unavailable(local_client, tmp_path):
    dataset = Dataset.from_list([((0, "a"), "value")])

    with memory_store_context(local_client, tmp_path) as context:
        store = _load_store(context, dataset, name="shutdown", recovery_timeout=0.01)
        assert store.get((0, "a")) == "value"

    with pytest.raises(MemoryStoreUnavailable):
        store.get((0, "a"))


def test_memory_store_rejects_pipeline_with_shuffle(local_client, tmp_path):
    dataset = Dataset.from_list([((0, "a"), "value")]).group_by(
        key=lambda item: item[0],
        reducer=lambda _key, items: next(items),
        num_output_shards=1,
    )

    with memory_store_context(local_client, tmp_path) as context:
        with pytest.raises(ValueError):
            _load_store(context, dataset, name="shuffled")


def test_memory_store_rejects_a_driver_that_cannot_resolve_worker_handles():
    """A distributed driver outside an Iris job cannot resolve the handles it is sent.

    Worker handles arrive from the coordinator, and serializing one drops its resolver,
    so it rebinds through the ambient Iris context. Failing here names the problem
    instead of surfacing it as a bare "requires IrisContext" from inside the load.
    """
    with pytest.raises(RuntimeError, match="inside an Iris job"):
        _require_resolvable_worker_handles(MagicMock())


def test_local_pools_need_no_iris_context():
    _require_resolvable_worker_handles(LocalClient())
