# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for read-only Zephyr memory stores."""

import contextvars
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import cloudpickle
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.actor import ActorUnavailableError
from fray.types import ActorConfig, ResourceConfig
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.test_util import SentinelFile
from rigging.timing import Duration, ExponentialBackoff
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.memory_store import (
    DuplicateMemoryStoreKey,
    MemoryStore,
    MemoryStorePartitionError,
    MemoryStoreUnavailable,
)


class _TestActorFuture:
    def __init__(self, value=None, error: Exception | None = None):
        self.value = value
        self.error = error
        self.timeouts: list[float | None] = []

    def result(self, timeout=None):
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

    def remote(self, *args, **kwargs):
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


@contextmanager
def _store_context(local_client, tmp_path) -> Iterator[ZephyrContext]:
    context = ZephyrContext(
        client=local_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="256m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="memory-store-test",
    )
    try:
        yield context
    finally:
        context.shutdown()


def _load_store(context: ZephyrContext, dataset: Dataset, *, hash_key=_key_partition, recovery_timeout=10):
    return context.load_memory_store(
        dataset,
        name="documents",
        hash_key=hash_key,
        num_actors=2,
        actor_resources=ResourceConfig(cpu=1, ram="256m"),
        actor_config=ActorConfig(max_concurrency=8, max_task_retries=1),
        recovery_timeout=recovery_timeout,
    )


def test_memory_store_routes_existing_partitions_and_preserves_lookup_order(local_client, tmp_path):
    partitions = [
        [((0, "a"), "zero-a"), ((0, "b"), "zero-b"), ((0, "none"), None)],
        [((1, "a"), "one-a")],
        [((2, "a"), "two-a"), ((2, "b"), "two-b")],
        [((3, "a"), "three-a")],
    ]
    dataset = Dataset.from_list(partitions).flat_map(_partition_rows)

    with _store_context(local_client, tmp_path) as context:
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


def test_memory_store_rejects_hash_that_disagrees_with_existing_partition(local_client, tmp_path):
    dataset = Dataset.from_list([[((0, "a"), "value")], [((1, "b"), "value")]]).flat_map(_partition_rows)

    with _store_context(local_client, tmp_path) as context:
        with pytest.raises(MemoryStorePartitionError):
            _load_store(context, dataset, hash_key=_wrong_key_partition)


@pytest.mark.requires_cluster
def test_memory_store_invalid_input_does_not_consume_actor_retries(iris_integration_client, tmp_path):
    dataset = Dataset.from_list([[((0, "a"), "value")], [((1, "b"), "value")]]).flat_map(_partition_rows)

    with _store_context(iris_integration_client, tmp_path) as context:
        with pytest.raises(MemoryStorePartitionError):
            context.load_memory_store(
                dataset,
                name="invalid-partition",
                hash_key=_wrong_key_partition,
                num_actors=2,
                actor_resources=ResourceConfig(cpu=1, ram="256m"),
                actor_config=ActorConfig(max_concurrency=8, max_task_retries=1_000),
                recovery_timeout=10,
                ready_timeout=45,
            )


def test_memory_store_rejects_duplicate_key(local_client, tmp_path):
    dataset = Dataset.from_list([[((0, "same"), "first"), ((0, "same"), "second")]]).flat_map(_partition_rows)

    with _store_context(local_client, tmp_path) as context:
        with pytest.raises(DuplicateMemoryStoreKey):
            context.load_memory_store(
                dataset,
                name="duplicates",
                hash_key=_key_partition,
                num_actors=1,
                actor_resources=ResourceConfig(cpu=1, ram="256m"),
                actor_config=ActorConfig(max_concurrency=8, max_task_retries=10),
                recovery_timeout=10,
            )


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

    with _store_context(local_client, tmp_path) as context:
        store = _load_store(context, dataset)
        restored = cloudpickle.loads(cloudpickle.dumps(store))

        first_result = context.execute(Dataset.from_list(keys).map(restored.get))
        second_result = context.execute(Dataset.from_list(list(reversed(keys))).map(restored.get))

    assert first_result.results == ["text-3-b", "text-0-a", "text-2-b", "text-1-a"]
    assert second_result.results == ["text-1-a", "text-2-b", "text-0-a", "text-3-b"]


@pytest.mark.requires_cluster
def test_memory_store_loads_and_serves_through_actor_backend(integration_client, tmp_path):
    dataset = Dataset.from_list(
        [
            ((0, "a"), "zero"),
            ((1, "a"), "one"),
        ]
    )

    with _store_context(integration_client, tmp_path) as context:
        store = _load_store(context, dataset, hash_key=lambda key: key[0])
        restored = cloudpickle.loads(cloudpickle.dumps(store))
        first = context.execute(Dataset.from_list([(1, "a"), (0, "a")]).map(restored.get))
        second = context.execute(Dataset.from_list([(0, "a"), (1, "a")]).map(restored.get))

    assert first.results == ["one", "zero"]
    assert second.results == ["zero", "one"]


def test_memory_store_retries_only_actor_unavailability():
    recovering_actor = _SequencedActor(
        [
            _TestActorFuture(error=ActorUnavailableError("restarting")),
            _TestActorFuture(value=[(True, "value")]),
        ]
    )
    recovering_store = MemoryStore(
        actors=(recovering_actor,),
        hash_key=lambda _key: 0,
        num_source_partitions=1,
        recovery_timeout=1,
    )

    assert recovering_store.get("key") == "value"

    failing_actor = _SequencedActor([_TestActorFuture(error=_ApplicationError())])
    failing_store = MemoryStore(
        actors=(failing_actor,),
        hash_key=lambda _key: 0,
        num_source_partitions=1,
        recovery_timeout=1,
    )
    with pytest.raises(_ApplicationError):
        failing_store.get("key")


def test_memory_store_bounds_actor_call_by_recovery_timeout():
    timed_out_future = _TestActorFuture(error=TimeoutError("actor call exceeded its deadline"))
    store = MemoryStore(
        actors=(_SequencedActor([timed_out_future]),),
        hash_key=lambda _key: 0,
        num_source_partitions=1,
        recovery_timeout=1,
    )

    with pytest.raises(MemoryStoreUnavailable, match="did not respond within 1 seconds"):
        store.get("key")

    assert len(timed_out_future.timeouts) == 1
    assert timed_out_future.timeouts[0] is not None
    assert 0 < timed_out_future.timeouts[0] <= 1


def test_context_shutdown_makes_store_unavailable(local_client, tmp_path):
    dataset = Dataset.from_list([((0, "a"), "value")])

    with _store_context(local_client, tmp_path) as context:
        store = context.load_memory_store(
            dataset,
            name="shutdown",
            hash_key=_key_partition,
            num_actors=1,
            actor_resources=ResourceConfig(cpu=1, ram="256m"),
            actor_config=ActorConfig(max_concurrency=8, max_task_retries=1),
            recovery_timeout=0.01,
        )
        assert store.get((0, "a")) == "value"

    with pytest.raises(MemoryStoreUnavailable):
        store.get((0, "a"))


@pytest.mark.requires_cluster
def test_memory_store_recovers_partition_after_iris_preemption(iris_integration_client, tmp_path):
    gate_reload = SentinelFile(str(tmp_path / "gate-reload"))
    reload_started = SentinelFile(str(tmp_path / "reload-started"))
    release_reload = SentinelFile(str(tmp_path / "release-reload"))

    def delay_reload(item):
        if gate_reload.is_set():
            reload_started.signal()
            release_reload.wait(timeout=Duration.from_seconds(15))
        return item

    dataset = Dataset.from_list([((0, "a"), "zero"), ((1, "a"), "one")]).map(delay_reload)

    with _store_context(iris_integration_client, tmp_path) as context:
        store = _load_store(context, dataset, hash_key=lambda key: key[0], recovery_timeout=15)
        group = context._memory_store_groups[0]
        task_id = JobName.from_wire(f"{group._job_id}/0")
        initial_attempt = iris_integration_client._iris.task_status(task_id).current_attempt_id
        gate_reload.signal()

        (kick_result,) = iris_integration_client._iris.kick_tasks(
            [task_id.to_wire()],
            desired_state=job_pb2.TASK_STATE_PREEMPTED,
            reason="memory-store recovery test",
        )
        assert kick_result.queued

        restarted = ExponentialBackoff(initial=0.1, maximum=1).wait_until(
            lambda: iris_integration_client._iris.task_status(task_id).current_attempt_id > initial_attempt,
            timeout=Duration.from_seconds(15),
        )
        assert restarted

        lookup_started = threading.Event()

        def lookup_during_reload():
            lookup_started.set()
            return store.get((0, "a"))

        caller_context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            lookup = executor.submit(caller_context.run, lookup_during_reload)
            try:
                assert lookup_started.wait(timeout=5)
                reload_started.wait(timeout=Duration.from_seconds(15))
                assert not lookup.done()
            finally:
                release_reload.signal()
            assert lookup.result(timeout=30) == "zero"


def test_memory_store_rejects_pipeline_with_shuffle(local_client, tmp_path):
    dataset = Dataset.from_list([((0, "a"), "value")]).group_by(
        key=lambda item: item[0],
        reducer=lambda _key, items: next(items),
        num_output_shards=1,
    )

    with _store_context(local_client, tmp_path) as context:
        with pytest.raises(ValueError):
            context.load_memory_store(
                dataset,
                name="shuffled",
                hash_key=_key_partition,
                num_actors=1,
                actor_resources=ResourceConfig(cpu=1, ram="256m"),
                actor_config=ActorConfig(max_concurrency=8, max_task_retries=10),
                recovery_timeout=10,
            )
