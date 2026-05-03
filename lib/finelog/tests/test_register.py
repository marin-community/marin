# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import (
    Column,
    InvalidNamespaceError,
    Schema,
    SchemaConflictError,
    SchemaValidationError,
    with_implicit_seq,
)

from tests.conftest import _worker_schema


@pytest.mark.parametrize(
    "name",
    [
        "iris.worker",
        "iris.worker.v2",
        "a",
        "a-b",
        "abc.def_ghi",
        "x" * 64,
    ],
)
def test_register_accepts_valid_names(store: DuckDBLogStore, name: str):
    schema = _worker_schema()
    effective = store.register_table(name, schema)
    assert effective == with_implicit_seq(schema)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Iris.Worker",
        ".starts-dot",
        "1starts-digit",
        "x" * 65,
        "has space",
        "has/slash",
        "..",
    ],
)
def test_register_rejects_invalid_names(store: DuckDBLogStore, name: str):
    with pytest.raises(InvalidNamespaceError):
        store.register_table(name, _worker_schema())


def test_register_rejects_path_traversal(store: DuckDBLogStore):
    with pytest.raises(InvalidNamespaceError):
        store.register_table("../escape", _worker_schema())


def test_register_rejects_schema_without_ordering_key(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ),
        key_column="",
    )
    with pytest.raises(SchemaValidationError):
        store.register_table("iris.worker", schema)


def test_register_accepts_implicit_timestamp_ms_int64(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ),
        key_column="",
    )
    store.register_table("iris.worker", schema)


def test_register_accepts_implicit_timestamp_ms_timestamp(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_TIMESTAMP_MS, nullable=False),
        ),
        key_column="",
    )
    store.register_table("iris.worker", schema)


def test_register_accepts_explicit_key_column(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="ts", type=stats_pb2.COLUMN_TYPE_TIMESTAMP_MS, nullable=False),
        ),
        key_column="ts",
    )
    store.register_table("iris.worker", schema)


def test_register_rejects_explicit_key_missing_from_columns(store: DuckDBLogStore):
    schema = Schema(
        columns=(Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),),
        key_column="ts",
    )
    with pytest.raises(SchemaValidationError):
        store.register_table("iris.worker", schema)


def test_register_idempotent_returns_existing_schema(store: DuckDBLogStore):
    schema = _worker_schema()
    first = store.register_table("iris.worker", schema)
    second = store.register_table("iris.worker", schema)
    assert first == second == with_implicit_seq(schema)


def test_register_subset_returns_full_registered_schema(store: DuckDBLogStore):
    full = Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            Column(name="cpu_pct", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=True),
            Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ),
    )
    store.register_table("iris.worker", full)
    subset = Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ),
    )
    effective = store.register_table("iris.worker", subset)
    assert effective == with_implicit_seq(full)


def test_register_additive_nullable_extension_merges(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    extended = Schema(
        columns=(*base.columns, Column(name="note", type=stats_pb2.COLUMN_TYPE_STRING, nullable=True)),
    )
    effective = store.register_table("iris.worker", extended)
    assert effective.column_names() == ("seq", "worker_id", "mem_bytes", "timestamp_ms", "note")
    again = store.register_table("iris.worker", base)
    assert again == effective


def test_register_non_additive_new_non_nullable_rejects(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    bad = Schema(
        columns=(*base.columns, Column(name="cpu_pct", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=False)),
    )
    with pytest.raises(SchemaConflictError):
        store.register_table("iris.worker", bad)


def test_register_type_change_rejects(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    bad = Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=False),  # was INT64
            Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ),
    )
    with pytest.raises(SchemaConflictError):
        store.register_table("iris.worker", bad)


def test_register_key_column_change_rejects(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    bad = Schema(columns=base.columns, key_column="timestamp_ms")  # was empty
    with pytest.raises(SchemaConflictError):
        store.register_table("iris.worker", bad)
