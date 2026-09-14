# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed sequential-shard samples and run PR #7911's real Finelog metrics query.

Run from the repository root: uv run lib/zephyr/scripts/repro_pr7911_metrics.py
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from tempfile import TemporaryDirectory

from finelog.client import LogClient
from finelog.embedded import EmbeddedServer
from rigging.timing import Duration, ExponentialBackoff
from zephyr.stats import StatsWriter, ZephyrWorkerStat


def main() -> None:
    with TemporaryDirectory() as directory:
        server = EmbeddedServer(log_dir=directory)
        client = LogClient.connect(f"http://127.0.0.1:{server.port}")
        try:
            table = client.get_table("zephyr.worker", ZephyrWorkerStat)
            start = datetime.now(UTC).replace(tzinfo=None, second=0, microsecond=0)
            sample = ZephyrWorkerStat(
                execution_id="probe",
                stage_name="stage0",
                shard_idx=0,
                status="RUNNING",
                ts=start,
                items=10,
                bytes_processed=100,
                item_rate=2.0,
                byte_rate=20.0,
                cpu_time_total=1.0,
                cpu_current_pct=100.0,
                cpu_avg_pct=100.0,
                mem_current_bytes=1024,
                mem_avg_bytes=1024,
                mem_peak_bytes=1024,
            )
            # Model one worker switching shards between two heartbeat samples.
            table.write([sample, replace(sample, shard_idx=1, ts=start + timedelta(seconds=10))])
            client.flush(timeout=5)
            assert ExponentialBackoff().wait_until(
                lambda: "zephyr.worker" in {namespace.namespace for namespace in client.list_namespaces()},
                timeout=Duration.from_seconds(15),
            )
            assert ExponentialBackoff().wait_until(
                lambda: client.query('SELECT count(*) AS n FROM "zephyr.worker"').to_pylist()[0]["n"] == 2,
                timeout=Duration.from_seconds(15),
            )
            result = StatsWriter(client).query_pipeline_metrics("probe", max_points=200)
            assert not result.warning, result.warning
            assert len(result.points) == 1
            point = result.points[0]
            print("Input: sequential shards on one worker, each sampled at 1 core and 1024 bytes.")
            print("Concurrent usage never exceeds 1 core or 1024 bytes.")
            print(f"Dashboard bin: {point.cpu_cores} cores, {point.memory_bytes} bytes")
            print(f"Dashboard rates: {point.item_rate} items/s, {point.byte_rate} bytes/s (each input: 2, 20)")
        finally:
            client.close()
            server.stop()


if __name__ == "__main__":
    main()
