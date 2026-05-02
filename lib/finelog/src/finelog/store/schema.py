# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema dataclasses, Arrow bridge, and validation helpers.

The :class:`Schema` and :class:`Column` dataclasses are the in-process
representation of a registered table's column layout. They convert
to/from:

- ``finelog.rpc.stats_pb2.Schema`` (over the wire)
- ``pyarrow.Schema`` (for Parquet I/O and IPC schema comparisons)
- a JSON sidecar form persisted in the registry DB

Validation rules live here too: ordering-key resolution, schema merging
(additive-nullable extension vs non-additive conflict), and per-batch IPC
schema checks (subset acceptance, dictionary decode, nested-type rejection).

The error types are owned by :mod:`finelog.errors`. They are re-exported
here so server/store-layer imports do not have to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

import pyarrow as pa

# InvalidNamespaceError / NamespaceNotFoundError are re-exported so
# server/store-layer modules can import the full error set from one place.
from finelog.errors import (
    InvalidNamespaceError,  # noqa: F401  re-export
    NamespaceNotFoundError,  # noqa: F401  re-export
    SchemaConflictError,
    SchemaValidationError,
)
from finelog.rpc import finelog_stats_pb2 as stats_pb2


class ColumnType(StrEnum):
    """Logical column types supported by the stats service.

    Maps 1:1 to a subset of pyarrow types. Not every Arrow type is admissible
    — nested lists / structs / unions are rejected at register and write time.
    """

    STRING = "string"
    INT64 = "int64"
    INT32 = "int32"
    FLOAT64 = "float64"
    BOOL = "bool"
    TIMESTAMP_MS = "timestamp_ms"
    BYTES = "bytes"


_ARROW_TYPE_FOR: dict[ColumnType, pa.DataType] = {
    ColumnType.STRING: pa.string(),
    ColumnType.INT64: pa.int64(),
    ColumnType.INT32: pa.int32(),
    ColumnType.FLOAT64: pa.float64(),
    ColumnType.BOOL: pa.bool_(),
    ColumnType.TIMESTAMP_MS: pa.timestamp("ms"),
    ColumnType.BYTES: pa.binary(),
}

_DUCKDB_TYPE_FOR: dict[ColumnType, str] = {
    ColumnType.STRING: "VARCHAR",
    ColumnType.INT64: "BIGINT",
    ColumnType.INT32: "INTEGER",
    ColumnType.FLOAT64: "DOUBLE",
    ColumnType.BOOL: "BOOLEAN",
    ColumnType.TIMESTAMP_MS: "TIMESTAMP_MS",
    ColumnType.BYTES: "BLOB",
}

# Inverse lookup for Arrow → ColumnType. Some Arrow types (e.g. dictionary)
# decode to a value type before this map is consulted; see ``_arrow_to_column_type``.
_COLUMN_TYPE_FOR_ARROW: dict[pa.DataType, ColumnType] = {v: k for k, v in _ARROW_TYPE_FOR.items()}

_PROTO_COLUMN_TYPE_FOR: dict[ColumnType, stats_pb2.ColumnType.ValueType] = {
    ColumnType.STRING: stats_pb2.COLUMN_TYPE_STRING,
    ColumnType.INT64: stats_pb2.COLUMN_TYPE_INT64,
    ColumnType.INT32: stats_pb2.COLUMN_TYPE_INT32,
    ColumnType.FLOAT64: stats_pb2.COLUMN_TYPE_FLOAT64,
    ColumnType.BOOL: stats_pb2.COLUMN_TYPE_BOOL,
    ColumnType.TIMESTAMP_MS: stats_pb2.COLUMN_TYPE_TIMESTAMP_MS,
    ColumnType.BYTES: stats_pb2.COLUMN_TYPE_BYTES,
}

_COLUMN_TYPE_FOR_PROTO: dict[stats_pb2.ColumnType.ValueType, ColumnType] = {
    v: k for k, v in _PROTO_COLUMN_TYPE_FOR.items()
}


# Valid types for the ordering key column. INT64 covers epoch-ms ints; the
# explicit timestamp accommodates dataclass schemas that carry datetime fields.
_KEY_COLUMN_TYPES: frozenset[ColumnType] = frozenset({ColumnType.INT64, ColumnType.TIMESTAMP_MS})

# Default implicit key column name when Schema.key_column is empty.
IMPLICIT_KEY_COLUMN = "timestamp_ms"


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    nullable: bool = False


@dataclass(frozen=True)
class Schema:
    """Registered column layout for a namespace.

    Attributes:
        columns: Columns in registered order. Order is preserved on disk so
            DuckDB COPY projections produce stable column ordering across
            additive evolutions.
        key_column: Explicit ordering key column name. Empty means the server
            falls back to ``timestamp_ms``.
    """

    columns: tuple[Column, ...]
    key_column: str = ""

    def column(self, name: str) -> Column | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


# ---------------------------------------------------------------------------
# Conversions: proto / Arrow / JSON.
# ---------------------------------------------------------------------------


def schema_from_proto(msg: stats_pb2.Schema) -> Schema:
    cols: list[Column] = []
    for c in msg.columns:
        if c.type == stats_pb2.COLUMN_TYPE_UNKNOWN or c.type not in _COLUMN_TYPE_FOR_PROTO:
            raise SchemaValidationError(f"column {c.name!r}: unknown column type {c.type!r}")
        cols.append(Column(name=c.name, type=_COLUMN_TYPE_FOR_PROTO[c.type], nullable=c.nullable))
    return Schema(columns=tuple(cols), key_column=msg.key_column)


def schema_to_proto(schema: Schema) -> stats_pb2.Schema:
    msg = stats_pb2.Schema(key_column=schema.key_column)
    for c in schema.columns:
        msg.columns.append(stats_pb2.Column(name=c.name, type=_PROTO_COLUMN_TYPE_FOR[c.type], nullable=c.nullable))
    return msg


def schema_to_arrow(schema: Schema) -> pa.Schema:
    """Convert a Schema to a pyarrow.Schema preserving nullability."""
    fields = [pa.field(c.name, _ARROW_TYPE_FOR[c.type], nullable=c.nullable) for c in schema.columns]
    return pa.schema(fields)


def duckdb_type_for(col: Column) -> str:
    return _DUCKDB_TYPE_FOR[col.type]


def schema_to_json(schema: Schema) -> str:
    payload = {
        "key_column": schema.key_column,
        "columns": [{"name": c.name, "type": c.type.value, "nullable": c.nullable} for c in schema.columns],
    }
    return json.dumps(payload)


def schema_from_json(text: str) -> Schema:
    payload = json.loads(text)
    cols = tuple(Column(name=c["name"], type=ColumnType(c["type"]), nullable=c["nullable"]) for c in payload["columns"])
    return Schema(columns=cols, key_column=payload.get("key_column", ""))


# ---------------------------------------------------------------------------
# Validation: ordering key, register-time schema check.
# ---------------------------------------------------------------------------


def resolve_key_column(schema: Schema) -> str:
    """Return the resolved ordering key column name, raising if invalid.

    Rules:
        - If ``schema.key_column`` is set, it must name an existing column of
          INT64 or TIMESTAMP_MS type.
        - Otherwise, the schema must contain a column named
          ``timestamp_ms`` of INT64 or TIMESTAMP_MS type.

    Raises:
        SchemaValidationError: neither rule satisfied.
    """
    if schema.key_column:
        col = schema.column(schema.key_column)
        if col is None:
            raise SchemaValidationError(f"key_column={schema.key_column!r} is not present in the schema columns")
        if col.type not in _KEY_COLUMN_TYPES:
            raise SchemaValidationError(
                f"key_column={schema.key_column!r} must be INT64 or TIMESTAMP_MS, got {col.type.value}"
            )
        return schema.key_column

    implicit = schema.column(IMPLICIT_KEY_COLUMN)
    if implicit is None:
        raise SchemaValidationError(f"schema declares no key_column and has no implicit '{IMPLICIT_KEY_COLUMN}' column")
    if implicit.type not in _KEY_COLUMN_TYPES:
        raise SchemaValidationError(
            f"implicit key column '{IMPLICIT_KEY_COLUMN}' must be INT64 or TIMESTAMP_MS, " f"got {implicit.type.value}"
        )
    return IMPLICIT_KEY_COLUMN


# ---------------------------------------------------------------------------
# Schema merge: register-evolve-by-default.
# ---------------------------------------------------------------------------


def merge_schemas(registered: Schema, requested: Schema) -> Schema:
    """Return the effective schema for a re-register against ``registered``.

    Behavior (see design.md "Schema evolution"):
        - Identical schemas (or requested ⊆ registered) → ``registered`` is
          returned unchanged.
        - Requested adds nullable columns the registered schema lacks → the
          union is returned, preserving registered-then-new column order.
        - Any conflicting column (type mismatch, nullability change on an
          existing column) raises ``SchemaConflictError``.
        - A non-nullable extension column (in requested but absent from
          registered) raises ``SchemaConflictError``.
        - A changed ``key_column`` raises ``SchemaConflictError``; the key is
          fixed for the namespace's lifetime.

    The order of columns in the merged schema is registered's order first,
    then any new (additive-nullable) columns in requested's order.
    """
    if registered.key_column != requested.key_column:
        raise SchemaConflictError(
            f"key_column mismatch: registered={registered.key_column!r} requested={requested.key_column!r}"
        )

    by_name_registered = {c.name: c for c in registered.columns}
    extras: list[Column] = []
    for rc in requested.columns:
        existing = by_name_registered.get(rc.name)
        if existing is None:
            if not rc.nullable:
                raise SchemaConflictError(
                    f"non-additive change: new column {rc.name!r} must be nullable for evolve-merge"
                )
            extras.append(rc)
            continue
        if existing.type != rc.type:
            raise SchemaConflictError(
                f"column {rc.name!r}: type mismatch registered={existing.type.value} " f"requested={rc.type.value}"
            )
        if existing.nullable != rc.nullable:
            raise SchemaConflictError(
                f"column {rc.name!r}: nullable mismatch registered={existing.nullable} " f"requested={rc.nullable}"
            )

    if not extras:
        return registered

    merged_cols = tuple(list(registered.columns) + extras)
    return Schema(columns=merged_cols, key_column=registered.key_column)


# ---------------------------------------------------------------------------
# Per-batch validation: Arrow IPC schema vs registered schema.
# ---------------------------------------------------------------------------


def _arrow_to_column_type(arrow_type: pa.DataType) -> ColumnType:
    """Map an Arrow datatype back to a ColumnType, decoding dictionary types.

    Dictionary-encoded columns are accepted transparently; we report the
    *value* type. Nested types (list, struct, union, map) are rejected.
    """
    if pa.types.is_dictionary(arrow_type):
        return _arrow_to_column_type(arrow_type.value_type)
    if (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_struct(arrow_type)
        or pa.types.is_union(arrow_type)
        or pa.types.is_map(arrow_type)
    ):
        raise SchemaValidationError(f"nested/union arrow type {arrow_type} is not supported")
    if arrow_type not in _COLUMN_TYPE_FOR_ARROW:
        raise SchemaValidationError(f"unsupported arrow type {arrow_type}")
    return _COLUMN_TYPE_FOR_ARROW[arrow_type]


def _decode_dictionary_columns(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Replace any dictionary-encoded columns with their decoded value arrays.

    Dictionary encoding is wire-only optimization; the on-disk Parquet schema
    stores plain value types. ``cast`` decodes to the underlying type.
    """
    columns: list[pa.Array] = []
    fields: list[pa.Field] = []
    changed = False
    for i, field in enumerate(batch.schema):
        col = batch.column(i)
        if pa.types.is_dictionary(field.type):
            value_type = field.type.value_type
            columns.append(col.cast(value_type))
            fields.append(pa.field(field.name, value_type, nullable=field.nullable))
            changed = True
        else:
            columns.append(col)
            fields.append(field)
    if not changed:
        return batch
    return pa.RecordBatch.from_arrays(columns, schema=pa.schema(fields))


def validate_and_align_batch(batch: pa.RecordBatch, registered: Schema) -> pa.RecordBatch:
    """Validate an incoming RecordBatch against a registered schema; return the aligned batch.

    Aligns the batch to the registered schema column order, filling missing
    nullable columns with NULL arrays. The returned batch's schema matches
    :func:`schema_to_arrow(registered)` exactly so the append path can append
    it directly to a typed Parquet writer.

    Raises:
        SchemaValidationError: missing non-nullable column, unknown column
            name, type mismatch, or nested/union column type.
    """
    decoded = _decode_dictionary_columns(batch)

    by_name_registered = {c.name: c for c in registered.columns}
    by_name_batch: dict[str, tuple[pa.Field, pa.Array]] = {}
    for i, field in enumerate(decoded.schema):
        if field.name in by_name_batch:
            raise SchemaValidationError(f"duplicate column {field.name!r} in batch")
        by_name_batch[field.name] = (field, decoded.column(i))

    for name in by_name_batch:
        if name not in by_name_registered:
            raise SchemaValidationError(f"unknown column {name!r} not in registered schema")

    aligned_arrays: list[pa.Array] = []
    aligned_fields: list[pa.Field] = []
    n_rows = decoded.num_rows
    for col in registered.columns:
        if col.name in by_name_batch:
            field, array = by_name_batch[col.name]
            actual_type = _arrow_to_column_type(field.type)
            if actual_type != col.type:
                raise SchemaValidationError(
                    f"column {col.name!r}: type mismatch registered={col.type.value} " f"batch={actual_type.value}"
                )
            aligned_arrays.append(array)
        else:
            if not col.nullable:
                raise SchemaValidationError(f"column {col.name!r}: missing required (non-nullable) column")
            aligned_arrays.append(pa.nulls(n_rows, type=_ARROW_TYPE_FOR[col.type]))
        aligned_fields.append(pa.field(col.name, _ARROW_TYPE_FOR[col.type], nullable=col.nullable))

    return pa.RecordBatch.from_arrays(aligned_arrays, schema=pa.schema(aligned_fields))


# ---------------------------------------------------------------------------
# Resource-limit constants.
# ---------------------------------------------------------------------------

# Max bytes per WriteRows request body. Matches the gRPC default cap; an
# IPC stream larger than this would push the server's per-RPC memory ceiling
# beyond a single misbehaving caller's budget.
MAX_WRITE_ROWS_BYTES = 16 * 1024 * 1024

# Max rows per RecordBatch. Bounds server-side decode cost.
MAX_WRITE_ROWS_ROWS = 1_000_000
