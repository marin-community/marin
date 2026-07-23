# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema dataclasses and the Arrow / proto bridge.

The :class:`Schema` and :class:`Column` dataclasses are the in-process
representation of a registered table's column layout. They convert
to/from:

- ``finelog.rpc.finelog_stats_pb2.Schema`` (over the wire)
- ``pyarrow.Schema`` (for Parquet I/O and IPC schema comparisons)

The Rust ``finelog-server`` owns storage-side validation, merging, and
on-disk persistence. The client only needs to declare schemas, encode
them for the wire, and build Arrow tables, so that is all that lives here.
"""

from dataclasses import dataclass

import pyarrow as pa

from finelog.errors import SchemaValidationError
from finelog.rpc import finelog_stats_pb2 as stats_pb2

# Logical column types are owned by the proto schema (single source of truth).
# ``stats_pb2.ColumnType`` is an ``int``-valued enum with members like
# ``COLUMN_TYPE_STRING``; we key the Arrow lookups on those int values
# directly. Use ``stats_pb2.ColumnType.Name(value)`` for human-readable
# names in error messages.
ColumnTypeValue = int

# The Arrow type for ``COLUMN_TYPE_MAP``: a homogeneous string-keyed,
# string-valued map. ``pa.map_(pa.string(), pa.string())`` is the exact wire
# form the Rust server declares (entries/key/value field names, a non-null key
# and a nullable value), so a batch built from it is accepted cast-free.
MAP_STRING_STRING = pa.map_(pa.string(), pa.string())

_ARROW_TYPE_FOR: dict[ColumnTypeValue, pa.DataType] = {
    stats_pb2.COLUMN_TYPE_STRING: pa.string(),
    stats_pb2.COLUMN_TYPE_INT64: pa.int64(),
    stats_pb2.COLUMN_TYPE_INT32: pa.int32(),
    stats_pb2.COLUMN_TYPE_FLOAT64: pa.float64(),
    stats_pb2.COLUMN_TYPE_BOOL: pa.bool_(),
    stats_pb2.COLUMN_TYPE_TIMESTAMP_MS: pa.timestamp("ms"),
    stats_pb2.COLUMN_TYPE_BYTES: pa.binary(),
    stats_pb2.COLUMN_TYPE_MAP: MAP_STRING_STRING,
}


# Per-row monotonic counter assigned server-side at write time. Stored on
# every namespace's parquet segments and visible to SQL queries; not
# transmitted on the wire and not declared by callers. Stripped from wire
# schemas in both directions so clients neither declare nor receive it.
IMPLICIT_SEQ_COLUMN = "seq"


@dataclass(frozen=True)
class Column:
    name: str
    # ``stats_pb2.ColumnType`` value (an int). Use the proto enum members
    # (e.g. ``stats_pb2.COLUMN_TYPE_STRING``) when constructing ``Column``.
    type: ColumnTypeValue
    # Defaults to nullable: an older client emitting an existing column should
    # remain valid as schemas evolve. Override with ``nullable=False`` for the
    # initial creation of a column where the producer guarantees presence
    # (e.g. the implicit ``timestamp_ms`` key).
    nullable: bool = True
    # Maintain a per-row-group trigram substring index for this column so
    # ``contains(col, …)`` / ``col LIKE '%…%'`` queries prune row groups instead
    # of full-scanning. Only meaningful for STRING columns; ignored otherwise.
    trigram_index: bool = False


@dataclass(frozen=True)
class Schema:
    """Registered column layout for a namespace.

    Attributes:
        columns: Columns in registered order. Order is preserved on disk so
            COPY projections produce stable column ordering across additive
            evolutions.
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


# The user-declared schema for the "log" namespace. The registry stamps
# the implicit ``seq`` column on top.
LOG_REGISTERED_SCHEMA = Schema(
    columns=(
        Column(name="key", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        Column(name="source", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        # The log message body is substring-searched via contains()/LIKE, so it
        # carries the trigram index (matches the server's log schema).
        Column(name="data", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False, trigram_index=True),
        Column(name="epoch_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        Column(name="level", type=stats_pb2.COLUMN_TYPE_INT32, nullable=False),
    ),
    # Per-source tail reads (``WHERE key = $key ORDER BY seq DESC``) dominate;
    # sorting by ``key`` first colocates same-source rows for row-group pruning.
    key_column="key",
)


# ---------------------------------------------------------------------------
# Conversions: proto / Arrow.
# ---------------------------------------------------------------------------


def schema_from_proto(msg: stats_pb2.Schema) -> Schema:
    """Decode a wire schema message.

    Wire schemas never carry implicit columns (e.g. ``seq``); a client
    that includes one is rejected. Server-stored schemas with implicit
    columns are kept in-process only and never round-trip through the
    wire form.
    """
    cols: list[Column] = []
    for c in msg.columns:
        if c.type == stats_pb2.COLUMN_TYPE_UNKNOWN or c.type not in _ARROW_TYPE_FOR:
            raise SchemaValidationError(f"column {c.name!r}: unknown column type {c.type!r}")
        if c.name == IMPLICIT_SEQ_COLUMN:
            raise SchemaValidationError(f"column {IMPLICIT_SEQ_COLUMN!r} is reserved (server-assigned implicit column)")
        cols.append(Column(name=c.name, type=c.type, nullable=c.nullable, trigram_index=c.index.trigram))
    return Schema(columns=tuple(cols), key_column=msg.key_column)


def schema_to_proto(schema: Schema) -> stats_pb2.Schema:
    """Encode a schema for the wire, stripping implicit columns.

    The server stamps implicit columns (``seq``) onto storage; clients
    neither declare nor receive them, so they are not part of the wire
    contract.
    """
    msg = stats_pb2.Schema(key_column=schema.key_column)
    for c in schema.columns:
        if c.name == IMPLICIT_SEQ_COLUMN:
            continue
        msg.columns.append(
            stats_pb2.Column(
                name=c.name,
                type=c.type,
                nullable=c.nullable,
                index=stats_pb2.ColumnIndex(trigram=c.trigram_index),
            )
        )
    return msg


def schema_to_arrow(schema: Schema) -> pa.Schema:
    """Convert a Schema to a pyarrow.Schema preserving nullability."""
    fields = [pa.field(c.name, _ARROW_TYPE_FOR[c.type], nullable=c.nullable) for c in schema.columns]
    return pa.schema(fields)
