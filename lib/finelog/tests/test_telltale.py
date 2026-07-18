# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FinelogMetricSink: the finelog end of rigging.telltale forwarding.

Exercises the sink over the embedded native server: the row schema derived from
the rigging dataclass, and a write/read round-trip that reads the flattened
identity columns plus a leftover label out of the native ``Map`` via ``json_get``.
Skips when the native extension (or its map support) is unavailable.
"""

from datetime import datetime

import pytest
from finelog.client import LogClient, schema_from_dataclass
from finelog.embedded import is_available, require_embedded_server
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.telltale import TELLTALE_NAMESPACE, FinelogMetricSink
from rigging.telltale import TelltaleMetric

_TS = datetime(2026, 7, 17, 12, 0, 0)


@pytest.fixture
def embedded_server(tmp_path):
    if not is_available():
        pytest.skip("finelog native server extension (finelog_server) not available")
    server = require_embedded_server()(log_dir=str(tmp_path / "log-server"))
    try:
        yield server
    finally:
        server.stop()


def _require_native_map(address: str) -> None:
    """Skip when the embedded server predates the native Map column type."""
    client = LogClient.connect(address)
    try:
        table = client.get_table("telltale_map_probe", TelltaleMetric)
        table.write([TelltaleMetric(name="p", value=0.0, kind="gauge", ts=_TS, source="iris", labels={"k": "v"})])
        table.flush(timeout=10.0)
        try:
            landed = client.query("SELECT count(*) AS n FROM telltale_map_probe").to_pylist()
        except Exception:
            landed = []
    finally:
        client.close()
    if not landed or landed[0]["n"] < 1:
        pytest.skip("embedded finelog server predates the native Map column type")


def test_schema_keys_on_name_with_a_native_map():
    schema = schema_from_dataclass(TelltaleMetric)

    assert schema.key_column == "name"
    by_type = {c.name: c.type for c in schema.columns}
    assert by_type["name"] == stats_pb2.COLUMN_TYPE_STRING
    assert by_type["value"] == stats_pb2.COLUMN_TYPE_FLOAT64
    assert by_type["ts"] == stats_pb2.COLUMN_TYPE_TIMESTAMP_MS
    assert by_type["source"] == stats_pb2.COLUMN_TYPE_STRING
    assert by_type["job_id"] == stats_pb2.COLUMN_TYPE_STRING
    assert by_type["task_index"] == stats_pb2.COLUMN_TYPE_INT64
    assert by_type["attempt"] == stats_pb2.COLUMN_TYPE_INT64
    assert by_type["labels"] == stats_pb2.COLUMN_TYPE_MAP


def test_sink_round_trips_flattened_columns_and_json_get_over_the_map(embedded_server):
    _require_native_map(embedded_server.address)

    sink = FinelogMetricSink(embedded_server.address)
    sink.write(
        [
            TelltaleMetric(
                name="levanter_train_loss",
                value=1.5,
                kind="gauge",
                ts=_TS,
                source="levanter",
                run="run-1",
                job_id="/a/b",
                task_index=0,
                attempt=1,
                labels={"le": "+Inf"},
            )
        ]
    )
    sink.close()  # flushes and releases the connection

    client = LogClient.connect(embedded_server.address)
    try:
        rows = client.query(
            "SELECT value, source, run, job_id, task_index, attempt, json_get(labels, 'le') AS le "
            f"FROM {TELLTALE_NAMESPACE} WHERE name = 'levanter_train_loss'"
        ).to_pylist()
    finally:
        client.close()

    assert rows == [
        {
            "value": 1.5,
            "source": "levanter",
            "run": "run-1",
            "job_id": "/a/b",
            "task_index": 0,
            "attempt": 1,
            "le": "+Inf",
        }
    ]
