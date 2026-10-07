# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from finelog.client import FlushResult, L0Mode, schema_from_dataclass
from finelog.rpc import finelog_stats_pb2
from marin.rollouts.catalog import (
    ROLLOUT_RUNS_NAMESPACE,
    RolloutRunKind,
    RolloutRunRecord,
    record_rollout_run,
    rollout_run_record,
)


class _Table:
    def __init__(self) -> None:
        self.rows = []

    def write(self, rows) -> None:
        self.rows.extend(rows)

    def flush(self, timeout=None) -> FlushResult:
        return FlushResult.SUCCEEDED


class _Client:
    def __init__(self, table: _Table) -> None:
        self.table = table
        self.namespace = None
        self.schema = None
        self.table_spec = None
        self.closed = False

    def get_table(self, namespace, schema, *, table_spec=None):
        self.namespace = namespace
        self.schema = schema
        self.table_spec = table_spec
        return self.table

    def close(self) -> None:
        self.closed = True


def test_rollout_run_schema_uses_queryable_scalar_columns() -> None:
    schema = schema_from_dataclass(RolloutRunRecord)
    columns = {column.name: column.type for column in schema.columns}

    assert schema.key_column == "timestamp_ms"
    assert columns["timestamp_ms"] == finelog_stats_pb2.COLUMN_TYPE_INT64
    assert columns["attributes"] == finelog_stats_pb2.COLUMN_TYPE_MAP


def test_record_rollout_run_adds_iris_attempt_identity(monkeypatch) -> None:
    table = _Table()
    client = _Client(table)
    monkeypatch.setattr(
        "marin.rollouts.catalog.runtime_telemetry.resolve",
        lambda **_kwargs: SimpleNamespace(
            endpoint="/system/log-server",
            resolver=lambda endpoint: endpoint,
            attributes={"execution_uid": "iris-attempt", "job_id": "/job"},
        ),
    )
    monkeypatch.setattr("marin.rollouts.catalog.LogClient.connect", lambda *_args, **_kwargs: client)

    record_rollout_run(
        rollout_run_record(
            run_id="eval-1",
            run_kind=RolloutRunKind.EVALUATION,
            producer="evalchemy",
            status="succeeded",
            rollout_uri="gs://bucket/eval-1",
            storage_format="finestore",
        )
    )

    assert client.namespace == ROLLOUT_RUNS_NAMESPACE
    assert client.schema is RolloutRunRecord
    assert client.table_spec.version == 1
    assert client.table_spec.operating_policy.l0_mode is L0Mode.OBJECT_STORE
    assert client.closed
    assert len(table.rows) == 1
    assert table.rows[0].attempt_id == "iris-attempt"
    assert table.rows[0].job_id == "/job"
    assert table.rows[0].rollout_uri == "gs://bucket/eval-1"


def test_record_rollout_run_is_inert_outside_iris(monkeypatch) -> None:
    monkeypatch.setattr("marin.rollouts.catalog.runtime_telemetry.resolve", lambda **_kwargs: None)

    record_rollout_run(
        rollout_run_record(
            run_id="local",
            run_kind=RolloutRunKind.REINFORCEMENT_LEARNING,
            producer="skyrl",
            status="failed",
            rollout_uri="/tmp/trajectories",
            storage_format="skyrl_trajectory",
        )
    )
