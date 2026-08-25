# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live-adapter proofs for the public WorkerPool API."""

import pytest
from iris.client.worker_pool import (
    WorkerPool,
    WorkerPoolConfig,
)
from iris.cluster.types import ResourceSpec

pytestmark = pytest.mark.requires_cluster


def _add(a: int, b: int) -> int:
    return a + b


def _square(value: int) -> int:
    return value * value


def _fail() -> None:
    raise ValueError("intentional error")


def _pool_config(num_workers: int) -> WorkerPoolConfig:
    return WorkerPoolConfig(
        num_workers=num_workers,
        resources=ResourceSpec(cpu=1, memory="512m"),
    )


def test_worker_pool_executes_submitted_and_mapped_calls(local_iris_client):
    with WorkerPool(local_iris_client, _pool_config(2), timeout=30.0) as pool:
        submitted = pool.submit(_add, 10, 20)
        mapped = pool.map(_square, [1, 2, 3, 4, 5])

        assert submitted.result(timeout=60.0) == 30
        assert [future.result(timeout=60.0) for future in mapped] == [1, 4, 9, 16, 25]


def test_worker_pool_propagates_user_exception(local_iris_client):
    with WorkerPool(local_iris_client, _pool_config(1), timeout=30.0) as pool:
        future = pool.submit(_fail)

        with pytest.raises(ValueError, match="intentional error"):
            future.result(timeout=60.0)


def test_worker_pool_after_shutdown_rejects_submission(local_iris_client):
    with WorkerPool(local_iris_client, _pool_config(1), timeout=30.0) as pool:
        assert pool.job_id is not None

    with pytest.raises(RuntimeError, match="shutdown"):
        pool.submit(_square, 42)
