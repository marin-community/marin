# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.runners import InlineRunner

from experiments.datakit.reference_pipeline import SMOKE_SCALE
from experiments.grug.fast_track.data_pipeline import DATA_PIPELINE_CONCURRENCY, data_pool


def test_fast_track_pool_runs_more_than_sixteen_pipelines(tmp_path):
    # The coordinator's default limit rejected the seventeenth live pipeline.
    pipeline_count = DATA_PIPELINE_CONCURRENCY + SMOKE_SCALE.sample_parallel_sources
    barrier = threading.Barrier(pipeline_count)

    def process(value: int) -> int:
        barrier.wait(timeout=30)
        return value + 1

    client = LocalClient()
    try:
        with set_current_client(client):
            pool = replace(
                data_pool("fast-track-concurrency-test"),
                stage_runner_factory=InlineRunner,
                chunk_storage_prefix=str(tmp_path / "chunks"),
            )
            with pool, ThreadPoolExecutor(max_workers=pipeline_count) as executor:
                futures = [
                    executor.submit(
                        pool.execute,
                        Dataset.from_list([value]).map(process),
                        map_task_resources=ResourceConfig(cpu=1, ram="64m"),
                    )
                    for value in range(pipeline_count)
                ]
                assert [future.result().results for future in futures] == [
                    [value + 1] for value in range(pipeline_count)
                ]
    finally:
        client.shutdown()
