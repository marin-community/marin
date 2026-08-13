# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris adapter tests for Zephyr memory stores."""

import contextvars
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import cloudpickle
import pytest
from fray.types import ResourceConfig
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.test_util import SentinelFile
from rigging.timing import Duration, ExponentialBackoff
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.memory_store import MemoryStorePartitionError

pytestmark = pytest.mark.requires_cluster


def _partition_rows(rows):
    yield from rows


def _wrong_key_partition(key: tuple[int, str]) -> int:
    return key[0] + 1


@contextmanager
def _store_context(client, tmp_path, *, max_workers: int = 2) -> Iterator[ZephyrContext]:
    context = ZephyrContext(
        client=client,
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="256m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="memory-store-test",
    )
    with context:
        yield context


def _worker_task_id(context: ZephyrContext, actor_index: int) -> JobName:
    assert context._pool is not None
    worker_job_id = context._pool.coordinator.worker_job_id.remote().result(timeout=30.0)
    return JobName.from_wire(f"{worker_job_id}/{actor_index}")


def test_memory_store_invalid_input_does_not_restart_workers(iris_integration_client, tmp_path):
    dataset = Dataset.from_list([[((0, "a"), "value")], [((1, "b"), "value")]]).flat_map(_partition_rows)

    with _store_context(iris_integration_client, tmp_path) as context:
        task_ids = [_worker_task_id(context, actor_index) for actor_index in range(2)]
        attempts_before = [iris_integration_client._iris.task_status(task_id).current_attempt_id for task_id in task_ids]

        with pytest.raises(MemoryStorePartitionError):
            context.load_memory_store(dataset, name="invalid-partition", hash_key=_wrong_key_partition)

        attempts_after = [iris_integration_client._iris.task_status(task_id).current_attempt_id for task_id in task_ids]
        assert attempts_after == attempts_before


def test_memory_store_loads_and_serves_through_iris_actor_backend(iris_integration_client, tmp_path):
    dataset = Dataset.from_list([((0, "a"), "zero"), ((1, "a"), "one")])

    with _store_context(iris_integration_client, tmp_path) as context:
        store = context.load_memory_store(dataset, name="documents", hash_key=lambda key: key[0])
        restored = cloudpickle.loads(cloudpickle.dumps(store))
        first = context.execute(Dataset.from_list([(1, "a"), (0, "a")]).map(restored.get))
        second = context.execute(Dataset.from_list([(0, "a"), (1, "a")]).map(restored.get))

    assert first.results == ["one", "zero"]
    assert second.results == ["zero", "one"]


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
        store = context.load_memory_store(
            dataset,
            name="documents",
            hash_key=lambda key: key[0],
            recovery_timeout=15,
        )
        task_id = _worker_task_id(context, 0)
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
