# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import cloudpickle
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.worker_context import zephyr_worker_ctx


class RecordingClient(LocalClient):
    def __init__(self):
        super().__init__()
        self.actor_groups = []

    def create_actor_group(self, actor_class, *args, **kwargs):
        self.actor_groups.append((actor_class, kwargs["count"]))
        return super().create_actor_group(actor_class, *args, **kwargs)


@pytest.fixture
def client():
    value = RecordingClient()
    yield value
    value.shutdown()


def test_scope_reuses_pool_and_preserves_task_data(client, tmp_path):
    pool = ZephyrContext(
        client=client,
        resources=ResourceConfig(cpu=2, ram="1g"),
        max_workers=1,
        chunk_storage_prefix=str(tmp_path),
    )

    def execute(value):
        borrowed = cloudpickle.loads(cloudpickle.dumps(pool))
        with borrowed.execution_scope():
            with ZephyrContext(client=client, resources=ResourceConfig(cpu=1, ram="256m")) as task:
                task.put("value", value)
                return task.execute(
                    Dataset.from_list([None]).map(
                        lambda _: (zephyr_worker_ctx().get_shared("value"), zephyr_worker_ctx().task_memory_bytes)
                    )
                ).results

    with pool:
        started_groups = list(client.actor_groups)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(execute, ["first", "second"]))
        assert results == [[("first", 256 * 1024**2)], [("second", 256 * 1024**2)]]
        assert client.actor_groups == started_groups
        assert pool.execute(Dataset.from_list([7])).results == [7]


def test_scope_rejects_tasks_larger_than_pool(client, tmp_path):
    pool = ZephyrContext(
        client=client,
        resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=1,
        chunk_storage_prefix=str(tmp_path),
    )
    with pool, pool.execution_scope():
        task = ZephyrContext(client=client, resources=ResourceConfig(cpu=1, ram="2g"))
        with pytest.raises(ValueError, match="exceed one Zephyr worker"):
            task.execute(Dataset.from_list([1]))
        assert pool.execute(Dataset.from_list([2])).results == [2]


def test_scope_restores_dedicated_execution(client, tmp_path):
    pool = ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path))
    with pool:
        with pytest.raises(RuntimeError, match="source failure"):
            with pool.execution_scope():
                raise RuntimeError("source failure")
    task = ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path))
    assert task.execute(Dataset.from_list([3])).results == [3]
