# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Zephyr Finelog dashboard queries."""

from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock

from finelog.client import LogClient
from zephyr.stats import StatsWriter


def test_pipeline_metrics_keep_complete_recent_time_bins():
    log_client = MagicMock()
    log_client.query.return_value.to_pylist.return_value = [
        {
            "time_bin": datetime(2026, 8, 2, 10, 15, tzinfo=UTC),
            "stage_name": "older",
            "item_rate": 1.0,
            "byte_rate": 2.0,
            "cpu_cores": 0.5,
            "memory_bytes": 1024.0,
            "active_shards": 1,
        },
        {
            "time_bin": datetime(2026, 8, 2, 10, 30),
            "stage_name": "stage-b",
            "item_rate": None,
            "byte_rate": None,
            "cpu_cores": None,
            "memory_bytes": None,
            "active_shards": None,
        },
        {
            "time_bin": datetime(2026, 8, 2, 10, 30, tzinfo=UTC),
            "stage_name": "stage-a",
            "item_rate": 125.5,
            "byte_rate": 4096.0,
            "cpu_cores": 1.25,
            "memory_bytes": 2048.0,
            "active_shards": 2,
        },
    ]
    writer = StatsWriter(cast(LogClient, log_client))

    result = writer.query_pipeline_metrics("exec'id", 2)

    query = log_client.query.call_args.args[0]
    assert "execution_id = 'exec''id'" in query
    assert "LIMIT 2" in query
    assert log_client.query.call_args.kwargs == {"max_rows": 512}
    assert result.warning == ""
    assert [point.stage for point in result.points] == ["stage-a", "stage-b"]
    assert result.points[0].cpu_cores == 1.25
    assert result.points[1].item_rate == 0
