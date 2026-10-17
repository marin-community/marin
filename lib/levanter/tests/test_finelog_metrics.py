# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax.numpy as jnp
from finelog.client import FlushResult, schema_from_dataclass
from finelog.rpc import finelog_stats_pb2

from levanter.tracker.finelog_metrics import LevanterMetricRow, LevanterMetricsWriter
from levanter.tracker.histogram import SummaryStats


class _Table:
    def __init__(self):
        self.rows: list[LevanterMetricRow] = []

    def write(self, rows) -> None:
        self.rows.extend(rows)

    def flush(self, _timeout=None) -> FlushResult:
        return FlushResult.SUCCEEDED


class _Client:
    def __init__(self, table: _Table):
        self.table = table
        self.namespace: str | None = None
        self.schema = None
        self.closed = False

    def get_table(self, namespace, schema):
        self.namespace = namespace
        self.schema = schema
        return self.table

    def close(self) -> None:
        self.closed = True


def test_metric_schema_uses_the_client_integer_wire_type():
    schema = schema_from_dataclass(LevanterMetricRow)
    columns = {column.name: column.type for column in schema.columns}

    assert columns["process_index"] == finelog_stats_pb2.COLUMN_TYPE_INT64
    assert columns["step"] == finelog_stats_pb2.COLUMN_TYPE_INT64


def test_writer_uses_run_alias_and_writes_typed_rows(monkeypatch):
    table = _Table()
    client = _Client(table)
    monkeypatch.setattr(
        "levanter.tracker.finelog_metrics.runtime_telemetry.resolve",
        lambda **_kwargs: SimpleNamespace(
            endpoint="/system/log-server",
            resolver=lambda endpoint: endpoint,
            attributes={
                "run_id": "run/+long/name",
                "execution_uid": "attempt-1",
                "job_id": "/job",
                "node_name": "node",
                "process_index": "0",
            },
        ),
    )
    monkeypatch.setattr(
        "levanter.tracker.finelog_metrics.LogClient.connect",
        lambda *_args, **_kwargs: client,
    )

    writer = LevanterMetricsWriter.from_iris("run/+long/name", 0)
    assert writer is not None
    assert client.namespace == "levanter.metrics.run/+long/name"
    assert client.schema is LevanterMetricRow

    writer.scalar("train_loss", 0.5, step=7)
    summary = SummaryStats.from_array(jnp.arange(8.0), num_bins=4)
    writer.summary("grad_norm", summary, step=7)

    scalar, histogram = table.rows
    assert scalar.kind == "scalar"
    assert scalar.step == 7
    assert scalar.value == 0.5
    assert scalar.bucket_limits is None
    assert histogram.kind == "histogram"
    assert histogram.value is None
    assert histogram.count == 8
    assert histogram.step == 7
    assert histogram.bucket_limits is not None
    assert histogram.bucket_counts is not None
    assert len(histogram.bucket_limits) == len(histogram.bucket_counts) + 1

    writer.close()
    assert client.closed


def test_writer_is_absent_outside_an_iris_runtime(monkeypatch):
    monkeypatch.setattr(
        "levanter.tracker.finelog_metrics.runtime_telemetry.resolve",
        lambda **_kwargs: None,
    )
    assert LevanterMetricsWriter.from_iris("run", 0) is None
