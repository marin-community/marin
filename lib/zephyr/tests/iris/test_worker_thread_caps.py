# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker actor processes launched through Iris cap their native pools at the worker's CPU share."""

import uuid

import polars as pl
import pyarrow as pa
import pytest
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import InlineRunner

pytestmark = pytest.mark.requires_cluster


def test_inline_stage_sees_worker_sized_pools(iris_integration_client, tmp_path):
    # A closure pickles by value; a module-level function would be imported
    # by name from the test module, which the worker process cannot see.
    def native_pool_sizes(_: int) -> tuple[int, int]:
        return pl.thread_pool_size(), pa.cpu_count()

    ctx = ZephyrContext(
        client=iris_integration_client,
        max_workers=1,
        resources=ResourceConfig(cpu=2, ram="1g"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-thread-caps-{uuid.uuid4().hex[:8]}",
        stage_runner_factory=InlineRunner,
    )
    try:
        results = ctx.execute(Dataset.from_list([0]).map(native_pool_sizes)).results
    finally:
        ctx.shutdown()

    assert results == [(2, 2)]
