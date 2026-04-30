# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RegisterTable RPC: name validation, schema validation (key
column rules), idempotency, additive schema evolution, and type/key-change
rejection. All tests operate directly on ``DuckDBLogStore`` without going
through the ASGI layer.
"""

from __future__ import annotations

import pytest
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import (
    Column,
    ColumnType,
    InvalidNamespaceError,
    Schema,
    SchemaConflictError,
    SchemaValidationError,
)

from tests.conftest import _worker_schema

# ---------------------------------------------------------------------------
# RegisterTable: name validation
# ---------------------------------------------------------------------------


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
    assert effective == schema


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Iris.Worker",  # uppercase
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
    # The regex check filters these too, but we confirm path containment is
    # exercised by names containing dots.
    with pytest.raises(InvalidNamespaceError):
        store.register_table("../escape", _worker_schema())


# ---------------------------------------------------------------------------
# RegisterTable: schema validation (ordering key)
# ---------------------------------------------------------------------------


def test_register_rejects_schema_without_ordering_key(store: DuckDBLogStore):
    # No key_column and no implicit timestamp_ms column.
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="mem_bytes", type=ColumnType.INT64),
        ),
        key_column="",
    )
    with pytest.raises(SchemaValidationError):
        store.register_table("iris.worker", schema)


def test_register_accepts_implicit_timestamp_ms_int64(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
        key_column="",
    )
    store.register_table("iris.worker", schema)


def test_register_accepts_implicit_timestamp_ms_timestamp(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="timestamp_ms", type=ColumnType.TIMESTAMP_MS),
        ),
        key_column="",
    )
    store.register_table("iris.worker", schema)


def test_register_accepts_explicit_key_column(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="ts", type=ColumnType.TIMESTAMP_MS),
        ),
        key_column="ts",
    )
    store.register_table("iris.worker", schema)


def test_register_rejects_explicit_key_missing_from_columns(store: DuckDBLogStore):
    schema = Schema(
        columns=(Column(name="worker_id", type=ColumnType.STRING),),
        key_column="ts",  # not in columns
    )
    with pytest.raises(SchemaValidationError):
        store.register_table("iris.worker", schema)


def test_register_rejects_explicit_key_wrong_type(store: DuckDBLogStore):
    schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="ts", type=ColumnType.STRING),  # wrong type for key
        ),
        key_column="ts",
    )
    with pytest.raises(SchemaValidationError):
        store.register_table("iris.worker", schema)


# ---------------------------------------------------------------------------
# RegisterTable: evolve-by-default
# ---------------------------------------------------------------------------


def test_register_idempotent_returns_existing_schema(store: DuckDBLogStore):
    schema = _worker_schema()
    first = store.register_table("iris.worker", schema)
    second = store.register_table("iris.worker", schema)
    assert first == second == schema


def test_register_subset_returns_full_registered_schema(store: DuckDBLogStore):
    full = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="mem_bytes", type=ColumnType.INT64),
            Column(name="cpu_pct", type=ColumnType.FLOAT64, nullable=True),
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
    )
    store.register_table("iris.worker", full)
    subset = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="timestamp_ms", type=ColumnType.INT64),
        ),
    )
    effective = store.register_table("iris.worker", subset)
    assert effective == full


def test_register_additive_nullable_extension_merges(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    extended = Schema(
        columns=(*base.columns, Column(name="note", type=ColumnType.STRING, nullable=True)),
    )
    effective = store.register_table("iris.worker", extended)
    assert effective.column_names() == ("worker_id", "mem_bytes", "timestamp_ms", "note")
    # Re-registering the base schema after evolution returns the merged schema.
    again = store.register_table("iris.worker", base)
    assert again == effective


def test_register_non_additive_new_non_nullable_rejects(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    bad = Schema(
        columns=(*base.columns, Column(name="cpu_pct", type=ColumnType.FLOAT64, nullable=False)),
    )
    with pytest.raises(SchemaConflictError):
        store.register_table("iris.worker", bad)


def test_register_type_change_rejects(store: DuckDBLogStore):
    base = _worker_schema()
    store.register_table("iris.worker", base)
    bad = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING),
            Column(name="mem_bytes", type=ColumnType.FLOAT64),  # was INT64
            Column(name="timestamp_ms", type=ColumnType.INT64),
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
