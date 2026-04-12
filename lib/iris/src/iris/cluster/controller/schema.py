# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema registry: single source of truth for table definitions, column metadata, and typed projections.

Provides ``Table`` and ``Column`` dataclasses that unify DDL generation, decode metadata,
and on-the-fly typed projection generation. See ``docs/sql-redesign.md`` Stage 0.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Generic, TypeVar

from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from rigging.timing import Timestamp

T = TypeVar("T")
RowDecoder = Callable[[sqlite3.Row], Any]


# ---------------------------------------------------------------------------
# Decoder functions — canonical home for column-level decoders used by
# Column definitions and ad-hoc ``QuerySnapshot.raw()`` calls.
# ---------------------------------------------------------------------------


class ProtoCache:
    """Thread-safe bounded cache for deserialized protobuf blobs.

    Keyed by raw proto bytes — no explicit invalidation needed for immutable
    columns (job protos). Changed bytes (worker heartbeat) naturally miss.
    """

    def __init__(self, max_size: int = 8192):
        self._cache: dict[bytes, Any] = {}
        self._lock = Lock()
        self._max_size = max_size

    def get_or_decode(self, blob: bytes, decoder: Callable[[bytes], Any]) -> Any:
        with self._lock:
            result = self._cache.get(blob)
            if result is not None:
                return result
        decoded = decoder(blob)
        with self._lock:
            if len(self._cache) >= self._max_size:
                to_evict = self._max_size // 4
                for k in list(self._cache.keys())[:to_evict]:
                    del self._cache[k]
            self._cache[blob] = decoded
        return decoded

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# Module-level singleton used by Projection.decode for cached=True columns.
proto_cache = ProtoCache()


def _identity(value: Any) -> Any:
    return value


def _nullable(decoder: RowDecoder) -> RowDecoder:
    def inner(value: Any) -> Any:
        if value is None:
            return None
        return decoder(value)

    return inner


def decode_worker_id(value: Any) -> WorkerId:
    return WorkerId(str(value))


def decode_timestamp_ms(value: Any) -> Timestamp:
    return Timestamp.from_ms(int(value))


def _decode_bool_int(value: Any) -> bool:
    return bool(int(value))


def _decode_json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(str(value))


def _decode_json_list(value: Any) -> list[str]:
    if not value:
        return []
    return json.loads(str(value))


def proto_decoder(proto_factory: Callable[[], T]) -> Callable[[Any], T]:
    def decode(value: Any) -> T:
        proto = proto_factory()
        proto.ParseFromString(value)
        return proto

    return decode


def _constraint_list_decoder(blob: bytes | None) -> list[job_pb2.Constraint]:
    if blob is None:
        return []
    cl = job_pb2.ConstraintList()
    cl.ParseFromString(blob)
    return list(cl.constraints)


# Sentinel for "no default" -- distinct from dataclasses.MISSING so we can use
# it as a regular default value in a frozen dataclass field.
_MISSING = object()


# ---------------------------------------------------------------------------
# Core infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """Describes a single SQL column with its decode metadata."""

    name: str
    sql_type: str
    constraints: str = ""
    python_name: str | None = None
    python_type: type = object
    decoder: Callable = _identity
    default: Any = _MISSING
    default_factory: Callable[[], Any] | None = None
    expensive: bool = False
    cached: bool = False

    @property
    def field_name(self) -> str:
        """Python attribute name for this column."""
        return self.python_name if self.python_name is not None else self.name


@dataclass(frozen=True, init=False)
class Table:
    """Describes a SQL table: its columns, constraints, indexes, and triggers."""

    name: str
    alias: str
    columns: tuple[Column, ...]
    table_constraints: tuple[str, ...]
    indexes: tuple[str, ...]
    triggers: tuple[str, ...]
    _column_map: dict[str, Column]

    def __init__(
        self,
        name: str,
        alias: str,
        columns: tuple[Column, ...],
        table_constraints: tuple[str, ...] = (),
        indexes: tuple[str, ...] = (),
        triggers: tuple[str, ...] = (),
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "table_constraints", table_constraints)
        object.__setattr__(self, "indexes", indexes)
        object.__setattr__(self, "triggers", triggers)
        col_map = {col.name: col for col in columns}
        object.__setattr__(self, "_column_map", col_map)

    def ddl(self) -> str:
        """Generate CREATE TABLE IF NOT EXISTS + CREATE INDEX + CREATE TRIGGER SQL."""
        lines: list[str] = []

        # Column definitions
        col_defs: list[str] = []
        for col in self.columns:
            parts = [col.name, col.sql_type]
            if col.constraints:
                parts.append(col.constraints)
            col_defs.append(" ".join(parts))

        # Table-level constraints
        for tc in self.table_constraints:
            col_defs.append(tc)

        lines.append(f"CREATE TABLE IF NOT EXISTS {self.name} (")
        lines.append(",\n".join(f"    {d}" for d in col_defs))
        lines.append(");")

        table_sql = "\n".join(lines)
        parts = [table_sql]

        for idx in self.indexes:
            parts.append(idx + ";")

        for trg in self.triggers:
            parts.append(trg)

        return "\n\n".join(parts)

    def projection(
        self,
        *column_names: str,
        extra_fields: tuple[ExtraField, ...] = (),
        row_cls: type | None = None,
    ) -> Projection:
        """Create a typed projection over a subset of columns.

        Validates all column names at call time (import time for module-level projections).
        Raises KeyError for unknown columns.

        ``extra_fields`` adds non-DB fields to the generated row class (e.g.
        ``attempts``, ``attributes``) that are populated post-hoc via
        ``dataclasses.replace``.

        When ``row_cls`` is provided the projection uses that class directly
        instead of generating one via ``_make_row_class``.
        """
        cols: list[Column] = []
        for cn in column_names:
            if cn not in self._column_map:
                raise KeyError(
                    f"Unknown column {cn!r} in table {self.name!r}. " f"Available: {sorted(self._column_map.keys())}"
                )
            cols.append(self._column_map[cn])
        return Projection(self, tuple(cols), extra_fields=extra_fields, row_cls=row_cls)

    def select_clause(self, *column_names: str, prefix: bool = True) -> str:
        """Generate 'alias.col1, alias.col2, ...' for a SELECT.

        If no column_names are given, uses all columns. If prefix is False,
        omits the table alias prefix.
        """
        names = column_names if column_names else tuple(c.name for c in self.columns)
        if prefix:
            return ", ".join(f"{self.alias}.{n}" for n in names)
        return ", ".join(names)


@dataclasses.dataclass(frozen=True)
class ExtraField:
    """A non-DB field added to a projection's generated row class.

    Used for post-hoc populated data like ``TaskDetail.attempts`` or
    ``WorkerRow.attributes`` that are injected via ``dataclasses.replace``
    after decoding.
    """

    name: str
    python_type: type = object
    default: Any = _MISSING
    default_factory: Callable[[], Any] | None = None


def _make_row_class(
    name: str,
    columns: tuple[Column, ...],
    extra_fields: tuple[ExtraField, ...] = (),
) -> type:
    """Generate a frozen dataclass with __slots__ from column definitions."""
    required: list[tuple[str, type, dataclasses.Field]] = []
    optional: list[tuple[str, type, dataclasses.Field]] = []
    for col in columns:
        field_name = col.field_name
        field_type = col.python_type
        if col.default_factory is not None:
            f = dataclasses.field(default_factory=col.default_factory)
            optional.append((field_name, field_type, f))
        elif col.default is not _MISSING:
            f = dataclasses.field(default=col.default)
            optional.append((field_name, field_type, f))
        else:
            f = dataclasses.field()
            required.append((field_name, field_type, f))
    # Extra (non-DB) fields always have defaults and go after optional fields.
    for ef in extra_fields:
        if ef.default_factory is not None:
            f = dataclasses.field(default_factory=ef.default_factory)
        elif ef.default is not _MISSING:
            f = dataclasses.field(default=ef.default)
        else:
            f = dataclasses.field(default=None)
        optional.append((ef.name, ef.python_type, f))
    fields = required + optional

    cls = dataclasses.make_dataclass(
        name,
        fields,
        frozen=True,
        slots=True,
    )
    return cls


def _validate_row_cls(
    row_cls: type,
    columns: tuple[Column, ...],
    extra_fields: tuple[ExtraField, ...],
) -> None:
    """Validate that a hand-written row class has all expected fields."""
    cls_field_names = {f.name for f in dataclasses.fields(row_cls)}
    for col in columns:
        if col.field_name not in cls_field_names:
            raise TypeError(f"{row_cls.__name__} is missing field {col.field_name!r} " f"(from column {col.name!r})")
    for ef in extra_fields:
        if ef.name not in cls_field_names:
            raise TypeError(f"{row_cls.__name__} is missing extra field {ef.name!r}")


class Projection(Generic[T]):
    """A typed subset of a Table's columns with a pre-compiled decoder.

    Pre-computes decoder tuples at init time (same pattern as ``@db_row_model``).
    The ``decode`` method replicates the ProtoCache integration from ``db.py``.
    """

    def __init__(
        self,
        table: Table,
        columns: tuple[Column, ...],
        *,
        extra_fields: tuple[ExtraField, ...] = (),
        row_cls: type | None = None,
        column_aliases: tuple[str, ...] | None = None,
    ):
        self.table = table
        self.columns = columns

        # Pre-compute parallel tuples for fast row decoding.
        self._names: tuple[str, ...] = tuple(c.field_name for c in columns)
        self._db_columns: tuple[str, ...] = tuple(c.name for c in columns)
        self._decoders: tuple[Callable, ...] = tuple(c.decoder for c in columns)
        self._cached_flags: tuple[bool, ...] = tuple(c.cached for c in columns)

        # Pre-compute defaults for fields that have them (keyed by column name).
        defaults: dict[str, tuple[str, Any | Callable[[], Any], bool]] = {}
        for c in columns:
            if c.default_factory is not None:
                defaults[c.name] = (c.field_name, c.default_factory, True)
            elif c.default is not _MISSING:
                defaults[c.name] = (c.field_name, c.default, False)
        self._defaults = defaults

        self._required_columns: tuple[str, ...] = tuple(c.name for c in columns if c.name not in defaults)

        if row_cls is not None:
            _validate_row_cls(row_cls, columns, extra_fields)
            self._row_cls: type = row_cls
        else:
            self._row_cls = _make_row_class(f"{table.name}_projection", columns, extra_fields)

        # Pre-compute SELECT column strings (aliased and bare).
        # When column_aliases is provided, each column uses its own table alias
        # (needed for cross-table projections like jobs JOIN job_config).
        if column_aliases is not None:
            self._select_aliased: str = ", ".join(
                f"{alias}.{c.name}" for alias, c in zip(column_aliases, columns, strict=True)
            )
        else:
            self._select_aliased: str = ", ".join(f"{table.alias}.{c.name}" for c in columns)
        self._select_bare: str = ", ".join(c.name for c in columns)

        # Pre-compute effective decoders: wrap cached fields through the global proto cache.
        if any(c.cached for c in columns):
            self._effective_decoders: tuple[Callable, ...] = tuple(
                (
                    (lambda d: lambda v: proto_cache.get_or_decode(v, d) if v is not None else d(v))(dec)
                    if is_cached
                    else dec
                )
                for dec, is_cached in zip(self._decoders, self._cached_flags, strict=True)
            )
        else:
            self._effective_decoders = self._decoders

        # Pre-compute zipped decode tuples for the fast path.
        self._decode_zipped: tuple[tuple[str, str, Callable], ...] = tuple(
            zip(self._names, self._db_columns, self._effective_decoders, strict=True)
        )

    def select_clause(self, *, prefix: bool = True) -> str:
        """Column list for a SELECT statement.

        With ``prefix=True`` (default), columns are qualified with the table alias
        (e.g. ``j.job_id, j.state``).  Use ``prefix=False`` for queries that do not
        use a table alias.
        """
        return self._select_aliased if prefix else self._select_bare

    @property
    def row_cls(self) -> type:
        return self._row_cls

    def decode(self, rows: Iterable[sqlite3.Row]) -> list:
        """Batch decode sqlite3.Row objects into typed row instances.

        Uses the same two-strategy fast path as ``decode_rows`` in ``db.py``:
        check if first row has all columns, then use tight comprehension loop.
        For columns with ``cached=True``, wraps the decoder through ProtoCache.
        """
        names = self._names
        columns = self._db_columns
        effective_decoders = self._effective_decoders
        cls = self._row_cls

        zipped = tuple(zip(names, columns, effective_decoders, strict=True))

        result: list = []
        it = iter(rows)
        first = next(it, None)
        if first is None:
            return result

        first_keys = set(first.keys())
        all_present = all(col in first_keys for col in columns)

        if all_present:
            # All columns present -- tight loop, no per-row key checks.
            result.append(cls(**{name: decoder(first[col]) for name, col, decoder in zipped}))
            for row in it:
                result.append(cls(**{name: decoder(row[col]) for name, col, decoder in zipped}))
        else:
            # Some columns missing -- use default-filling path for every row.
            result.append(self._decode_row(first))
            for row in it:
                result.append(self._decode_row(row))
        return result

    def decode_one(self, rows: Iterable[sqlite3.Row]) -> T | None:
        """Decode a single row, returning None if empty."""
        for row in rows:
            return self._decode_row(row)
        return None

    def _decode_row(self, row: sqlite3.Row) -> Any:
        """Decode a single row, filling defaults for missing optional columns."""
        row_keys = set(row.keys())

        for col in self._required_columns:
            if col not in row_keys:
                raise KeyError(f"Missing required column {col!r} for {self._row_cls.__name__}")

        values: dict[str, Any] = {}
        for name, col, decoder in zip(self._names, self._db_columns, self._effective_decoders, strict=True):
            if col in row_keys:
                values[name] = decoder(row[col])
            else:
                field_name, default_val, is_factory = self._defaults[col]
                values[field_name] = default_val() if is_factory else default_val
        return self._row_cls(**values)


def adhoc_projection(*fields: tuple[str, type]) -> Projection:
    """Create a one-off projection for aggregate / ad-hoc queries.

    Each field is a (name, type) tuple. Decoders are identity; no table alias is used.

    Example::

        LiveStats = adhoc_projection(("user_id", str), ("state", int), ("cnt", int))
    """
    columns = tuple(Column(name=name, sql_type="", python_type=typ, decoder=_identity) for name, typ in fields)
    table = Table(name="_adhoc", alias="", columns=columns)
    return Projection(table, columns)


def generate_full_ddl(tables: Sequence[Table]) -> str:
    """Concatenate all table DDLs into a single SQL script."""
    return "\n\n".join(t.ddl() for t in tables)


# ---------------------------------------------------------------------------
# Table definitions -- current schema after all migrations (0001-0019)
# ---------------------------------------------------------------------------


SCHEMA_MIGRATIONS = Table(
    "schema_migrations",
    "sm",
    columns=(
        Column("name", "TEXT", "PRIMARY KEY"),
        Column("applied_at_ms", "INTEGER", "NOT NULL"),
    ),
)

META = Table(
    "meta",
    "m",
    columns=(
        Column("key", "TEXT", "PRIMARY KEY"),
        Column("value", "INTEGER", "NOT NULL"),
    ),
)

USERS = Table(
    "users",
    "u",
    columns=(
        Column("user_id", "TEXT", "PRIMARY KEY", python_type=str, decoder=str),
        Column(
            "created_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="created_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
        Column("display_name", "TEXT", "", python_type=str | None, decoder=_nullable(str), default=None),
        Column(
            "role",
            "TEXT",
            "NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user', 'worker'))",
            python_type=str,
            decoder=str,
            default="user",
        ),
    ),
)

JOBS = Table(
    "jobs",
    "j",
    columns=(
        Column("job_id", "TEXT", "PRIMARY KEY", python_type=JobName, decoder=JobName.from_wire),
        Column("user_id", "TEXT", "NOT NULL REFERENCES users(user_id)", python_type=str, decoder=str),
        Column(
            "parent_job_id",
            "TEXT",
            "REFERENCES jobs(job_id) ON DELETE CASCADE",
            python_type=JobName | None,
            decoder=_nullable(JobName.from_wire),
        ),
        Column("root_job_id", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("depth", "INTEGER", "NOT NULL", python_type=int, decoder=int, default=0),
        Column("state", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column(
            "submitted_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="submitted_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
        Column(
            "root_submitted_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="root_submitted_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
        Column(
            "started_at_ms",
            "INTEGER",
            "",
            python_name="started_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
        ),
        Column(
            "finished_at_ms",
            "INTEGER",
            "",
            python_name="finished_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
        ),
        Column("scheduling_deadline_epoch_ms", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("error", "TEXT", "", python_type=str | None, decoder=_nullable(str)),
        Column("exit_code", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("num_tasks", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column(
            "is_reservation_holder",
            "INTEGER",
            "NOT NULL CHECK (is_reservation_holder IN (0, 1))",
            python_type=bool,
            decoder=_decode_bool_int,
        ),
        # Kept on jobs (not just job_config) for fast listing/filtering without JOIN.
        Column("name", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column(
            "has_reservation", "INTEGER", "NOT NULL DEFAULT 0", python_type=bool, decoder=_decode_bool_int, default=False
        ),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_job_id)",
        # Migration 0007
        "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, submitted_at_ms DESC)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_depth_state ON jobs(depth, state, submitted_at_ms DESC)",
        # Migration 0009
        "CREATE INDEX IF NOT EXISTS idx_jobs_user_state ON jobs(user_id, state)",
        # Migration 0010_dashboard
        "CREATE INDEX IF NOT EXISTS idx_jobs_root_depth ON jobs(root_job_id, depth)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_depth_submitted ON jobs(depth, submitted_at_ms DESC)",
        # Migration 0013 (kept across 0028)
        "CREATE INDEX IF NOT EXISTS idx_jobs_name ON jobs(name)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_has_reservation"
        " ON jobs(has_reservation, state) WHERE has_reservation = 1",
    ),
)

# INVARIANT: job_config.name and job_config.has_reservation are denormalized copies
# kept on the jobs table (jobs.name, jobs.has_reservation) for fast listing queries
# without JOIN.  Both copies are written together at job submission time.
JOB_CONFIG = Table(
    "job_config",
    "jc",
    columns=(
        Column(
            "job_id",
            "TEXT",
            "PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE",
            python_type=JobName,
            decoder=JobName.from_wire,
        ),
        Column("name", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column(
            "has_reservation", "INTEGER", "NOT NULL DEFAULT 0", python_type=bool, decoder=_decode_bool_int, default=False
        ),
        # Resource spec (was resources_proto)
        Column("res_cpu_millicores", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("res_memory_bytes", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("res_disk_bytes", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("res_device_json", "TEXT", "", python_type=str | None, decoder=_nullable(str), default=None),
        # Constraints (was constraints_proto)
        Column("constraints_json", "TEXT", "", python_type=str | None, decoder=_nullable(str), default=None),
        # Coscheduling
        Column(
            "has_coscheduling",
            "INTEGER",
            "NOT NULL DEFAULT 0",
            python_type=bool,
            decoder=_decode_bool_int,
            default=False,
        ),
        Column("coscheduling_group_by", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        # Scheduling config
        Column("scheduling_timeout_ms", "INTEGER", "", python_type=int | None, decoder=_nullable(int), default=None),
        Column("max_task_failures", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        # Dispatch config (was in request_proto)
        Column("entrypoint_json", "TEXT", "NOT NULL DEFAULT '{}'", python_type=str, decoder=str, default="{}"),
        Column("environment_json", "TEXT", "NOT NULL DEFAULT '{}'", python_type=str, decoder=str, default="{}"),
        Column("bundle_id", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("ports_json", "TEXT", "NOT NULL DEFAULT '[]'", python_type=str, decoder=str, default="[]"),
        Column("max_retries_failure", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("max_retries_preemption", "INTEGER", "NOT NULL DEFAULT 100", python_type=int, decoder=int, default=100),
        Column("timeout_ms", "INTEGER", "", python_type=int | None, decoder=_nullable(int), default=None),
        Column("preemption_policy", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("existing_job_policy", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("priority_band", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("task_image", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("reservation_json", "TEXT", "", python_type=str | None, decoder=_nullable(str), default=None),
        Column(
            "fail_if_exists",
            "INTEGER",
            "NOT NULL DEFAULT 0",
            python_type=bool,
            decoder=_decode_bool_int,
            default=False,
        ),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_job_config_name ON job_config(name)",
        "CREATE INDEX IF NOT EXISTS idx_job_config_has_reservation"
        " ON job_config(has_reservation, job_id) WHERE has_reservation = 1",
    ),
)

JOB_WORKDIR_FILES = Table(
    "job_workdir_files",
    "jwf",
    columns=(
        Column(
            "job_id",
            "TEXT",
            "NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE",
            python_type=JobName,
            decoder=JobName.from_wire,
        ),
        Column("filename", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("data", "BLOB", "NOT NULL", expensive=True),
    ),
    table_constraints=("PRIMARY KEY (job_id, filename)",),
)

TASKS = Table(
    "tasks",
    "t",
    columns=(
        Column("task_id", "TEXT", "PRIMARY KEY", python_type=JobName, decoder=JobName.from_wire),
        Column(
            "job_id",
            "TEXT",
            "NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE",
            python_type=JobName,
            decoder=JobName.from_wire,
        ),
        Column("task_index", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("state", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("error", "TEXT", "", python_type=str | None, decoder=_nullable(str)),
        Column("exit_code", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column(
            "submitted_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="submitted_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
        Column(
            "started_at_ms",
            "INTEGER",
            "",
            python_name="started_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
        ),
        Column(
            "finished_at_ms",
            "INTEGER",
            "",
            python_name="finished_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
        ),
        Column("max_retries_failure", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("max_retries_preemption", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("failure_count", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("preemption_count", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("current_attempt_id", "INTEGER", "NOT NULL DEFAULT -1", python_type=int, decoder=int),
        Column("priority_neg_depth", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("priority_root_submitted_ms", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("priority_insertion", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        # Migration 0021_budgets
        Column("priority_band", "INTEGER", "NOT NULL DEFAULT 2", python_type=int, decoder=int),
        # Migration 0012_container_name
        Column("container_id", "TEXT", "", python_type=str | None, decoder=_nullable(str), default=None),
        # Migration 0018
        Column(
            "current_worker_id",
            "TEXT",
            "REFERENCES workers(worker_id) ON DELETE SET NULL",
            python_type=WorkerId | None,
            decoder=_nullable(decode_worker_id),
            default=None,
        ),
        Column("current_worker_address", "TEXT", "", python_type=str | None, decoder=_nullable(str), default=None),
    ),
    table_constraints=("UNIQUE(job_id, task_index)",),
    indexes=(
        # Migration 0002
        "CREATE INDEX IF NOT EXISTS idx_tasks_job_state ON tasks(job_id, state)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_pending"
        " ON tasks(state, priority_band ASC, priority_neg_depth ASC,"
        " priority_root_submitted_ms ASC, submitted_at_ms ASC, priority_insertion ASC)",
        # Migration 0009
        "CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state)",
        # Migration 0010_dashboard
        "CREATE INDEX IF NOT EXISTS idx_tasks_state_attempt" " ON tasks(state, task_id, current_attempt_id, job_id)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_job_failures" " ON tasks(job_id, failure_count, preemption_count)",
        # Migration 0020
        "CREATE INDEX IF NOT EXISTS idx_tasks_current_worker"
        " ON tasks(current_worker_id) WHERE current_worker_id IS NOT NULL",
    ),
)

TASK_ATTEMPTS = Table(
    "task_attempts",
    "ta",
    columns=(
        Column(
            "task_id",
            "TEXT",
            "NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE",
            python_type=JobName,
            decoder=JobName.from_wire,
        ),
        Column("attempt_id", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        # Migration 0019: ON DELETE SET NULL (was plain FK)
        Column(
            "worker_id",
            "TEXT",
            "REFERENCES workers(worker_id) ON DELETE SET NULL",
            python_type=WorkerId | None,
            decoder=_nullable(decode_worker_id),
        ),
        Column("state", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column(
            "created_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="created_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
        Column(
            "started_at_ms",
            "INTEGER",
            "",
            python_name="started_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
        ),
        Column(
            "finished_at_ms",
            "INTEGER",
            "",
            python_name="finished_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
        ),
        Column("exit_code", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("error", "TEXT", "", python_type=str | None, decoder=_nullable(str)),
    ),
    table_constraints=("PRIMARY KEY (task_id, attempt_id)",),
    indexes=(
        # Migration 0007 (recreated in 0019 after table rebuild)
        "CREATE INDEX IF NOT EXISTS idx_task_attempts_worker_task"
        " ON task_attempts(worker_id, task_id, attempt_id)",
    ),
    triggers=(
        # From 0001_init
        """CREATE TRIGGER IF NOT EXISTS trg_task_attempt_active_worker
BEFORE INSERT ON task_attempts
FOR EACH ROW
WHEN NEW.worker_id IS NOT NULL
BEGIN
  SELECT
    CASE
      WHEN NOT EXISTS(
        SELECT 1 FROM workers w
        WHERE w.worker_id = NEW.worker_id
          AND w.active = 1
          AND w.healthy = 1
      )
      THEN RAISE(ABORT, 'task attempt worker must be active and healthy')
    END;
END;""",
    ),
)

WORKERS = Table(
    "workers",
    "w",
    columns=(
        Column("worker_id", "TEXT", "PRIMARY KEY", python_type=WorkerId, decoder=decode_worker_id),
        Column("address", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("md_hostname", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_ip_address", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_cpu_count", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("md_memory_bytes", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("md_disk_bytes", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("md_tpu_name", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_tpu_worker_hostnames", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_tpu_worker_id", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_tpu_chips_per_host_bounds", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_gpu_count", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("md_gpu_name", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_gpu_memory_mb", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("md_gce_instance_name", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_gce_zone", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_git_hash", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("md_device_json", "TEXT", "NOT NULL DEFAULT '{}'", python_type=str, decoder=str, default="{}"),
        Column("healthy", "INTEGER", "NOT NULL CHECK (healthy IN (0, 1))", python_type=bool, decoder=_decode_bool_int),
        Column(
            "active",
            "INTEGER",
            "NOT NULL CHECK (active IN (0, 1))",
            python_type=bool,
            decoder=_decode_bool_int,
            default=True,
        ),
        Column("consecutive_failures", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column(
            "last_heartbeat_ms",
            "INTEGER",
            "NOT NULL",
            python_name="last_heartbeat",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
        Column("committed_cpu_millicores", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column(
            "committed_mem_bytes",
            "INTEGER",
            "NOT NULL",
            python_name="committed_mem",
            python_type=int,
            decoder=int,
        ),
        Column("committed_gpu", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("committed_tpu", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("snapshot_host_cpu_percent", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_memory_used_bytes", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_memory_total_bytes", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_disk_used_bytes", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_disk_total_bytes", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_running_task_count", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_total_process_count", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_net_recv_bps", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_net_sent_bps", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        # Migration 0016
        Column("total_cpu_millicores", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("total_memory_bytes", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("total_gpu_count", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("total_tpu_count", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("device_type", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("device_variant", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        # Migration 0022
        Column("slice_id", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("scale_group", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
    ),
    indexes=(
        # Migration 0004_worker_indexes
        "CREATE INDEX IF NOT EXISTS idx_workers_healthy_active ON workers(healthy, active)",
    ),
)

WORKER_ATTRIBUTES = Table(
    "worker_attributes",
    "wa",
    columns=(
        Column(
            "worker_id",
            "TEXT",
            "NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE",
            python_type=WorkerId,
            decoder=decode_worker_id,
        ),
        Column("key", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column(
            "value_type",
            "TEXT",
            "NOT NULL CHECK (value_type IN ('str', 'int', 'float'))",
            python_type=str,
            decoder=str,
        ),
        Column("str_value", "TEXT", "", python_type=str | None, decoder=_nullable(str)),
        Column("int_value", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("float_value", "REAL", "", python_type=float | None, decoder=_identity),
    ),
    table_constraints=("PRIMARY KEY (worker_id, key)",),
)

WORKER_TASK_HISTORY = Table(
    "worker_task_history",
    "wth",
    columns=(
        Column("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
        Column(
            "worker_id",
            "TEXT",
            "NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE",
            python_type=WorkerId,
            decoder=decode_worker_id,
        ),
        Column("task_id", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column(
            "assigned_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="assigned_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_worker_task_history_worker"
        " ON worker_task_history(worker_id, assigned_at_ms DESC)",
    ),
)

WORKER_RESOURCE_HISTORY = Table(
    "worker_resource_history",
    "wrh",
    columns=(
        Column("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
        Column(
            "worker_id",
            "TEXT",
            "NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE",
            python_type=WorkerId,
            decoder=decode_worker_id,
        ),
        Column("snapshot_host_cpu_percent", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_memory_used_bytes", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_memory_total_bytes", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_disk_used_bytes", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_disk_total_bytes", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_running_task_count", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_total_process_count", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_net_recv_bps", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("snapshot_net_sent_bps", "INTEGER", "", python_type=int | None, decoder=_nullable(int)),
        Column("timestamp_ms", "INTEGER", "NOT NULL", python_type=Timestamp, decoder=decode_timestamp_ms),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_worker_resource_history_worker"
        " ON worker_resource_history(worker_id, id DESC)",
        # Migration 0010_dashboard
        "CREATE INDEX IF NOT EXISTS idx_worker_resource_history_ts"
        " ON worker_resource_history(worker_id, timestamp_ms DESC)",
    ),
)

TASK_RESOURCE_HISTORY = Table(
    "task_resource_history",
    "trh",
    columns=(
        Column("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
        Column(
            "task_id",
            "TEXT",
            "NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE",
            python_type=JobName,
            decoder=JobName.from_wire,
        ),
        Column("attempt_id", "INTEGER", "NOT NULL"),
        Column("cpu_millicores", "INTEGER", "NOT NULL DEFAULT 0"),
        Column("memory_mb", "INTEGER", "NOT NULL DEFAULT 0"),
        Column("disk_mb", "INTEGER", "NOT NULL DEFAULT 0"),
        Column("memory_peak_mb", "INTEGER", "NOT NULL DEFAULT 0"),
        Column("timestamp_ms", "INTEGER", "NOT NULL", python_type=Timestamp, decoder=decode_timestamp_ms),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_task_resource_history_task_attempt"
        " ON task_resource_history(task_id, attempt_id, id DESC)",
    ),
)

ENDPOINTS = Table(
    "endpoints",
    "e",
    columns=(
        Column("endpoint_id", "TEXT", "PRIMARY KEY", python_type=str, decoder=str),
        Column("name", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("address", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column(
            "job_id",
            "TEXT",
            "NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE",
            python_type=JobName,
            decoder=JobName.from_wire,
        ),
        Column(
            "task_id",
            "TEXT",
            "REFERENCES tasks(task_id) ON DELETE CASCADE",
            python_type=JobName | None,
            decoder=_nullable(JobName.from_wire),
        ),
        Column("metadata_json", "TEXT", "NOT NULL", python_name="metadata", python_type=dict, decoder=_decode_json_dict),
        Column(
            "registered_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="registered_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_endpoints_name ON endpoints(name)",
        "CREATE INDEX IF NOT EXISTS idx_endpoints_task ON endpoints(task_id)",
        # Migration 0009
        "CREATE INDEX IF NOT EXISTS idx_endpoints_job_id ON endpoints(job_id)",
    ),
)

# Migration 0011: dispatch_queue with nullable worker_id
DISPATCH_QUEUE = Table(
    "dispatch_queue",
    "dq",
    columns=(
        Column("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
        Column(
            "worker_id",
            "TEXT",
            "REFERENCES workers(worker_id) ON DELETE CASCADE",
            python_type=WorkerId | None,
            decoder=_nullable(decode_worker_id),
        ),
        Column("kind", "TEXT", "NOT NULL CHECK (kind IN ('run', 'kill'))", python_type=str, decoder=str),
        Column("payload_proto", "BLOB", "", expensive=True),
        Column("task_id", "TEXT", "", python_type=str | None, decoder=_nullable(str)),
        Column(
            "created_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="created_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
    ),
    indexes=("CREATE INDEX IF NOT EXISTS idx_dispatch_worker ON dispatch_queue(worker_id, id)",),
)

TXN_LOG = Table(
    "txn_log",
    "tl",
    columns=(
        Column("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
        Column("kind", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("payload_json", "TEXT", "NOT NULL", python_name="payload", python_type=dict, decoder=_decode_json_dict),
        Column(
            "created_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="created_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
    ),
    triggers=(
        # Migration 0004_worker_indexes rewrote the trigger from 0001
        """CREATE TRIGGER IF NOT EXISTS trg_txn_log_retention
AFTER INSERT ON txn_log
WHEN (SELECT COUNT(*) FROM txn_log) > 1100
BEGIN
  DELETE FROM txn_log WHERE id <= (
    SELECT id FROM txn_log ORDER BY id DESC LIMIT 1 OFFSET 1000
  );
END;""",
    ),
)

TXN_ACTIONS = Table(
    "txn_actions",
    "ta2",
    columns=(
        Column("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
        Column(
            "txn_id",
            "INTEGER",
            "NOT NULL REFERENCES txn_log(id) ON DELETE CASCADE",
            python_type=int,
            decoder=int,
        ),
        Column("action", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("entity_id", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("details_json", "TEXT", "NOT NULL", python_name="details", python_type=dict, decoder=_decode_json_dict),
        Column(
            "created_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="timestamp",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
    ),
    indexes=("CREATE INDEX IF NOT EXISTS idx_txn_actions_txn ON txn_actions(txn_id, id)",),
)

# Migration 0003: restructured scaling_groups
SCALING_GROUPS = Table(
    "scaling_groups",
    "sg",
    columns=(
        Column("name", "TEXT", "PRIMARY KEY", python_type=str, decoder=str),
        Column("consecutive_failures", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("backoff_until_ms", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("last_scale_up_ms", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("last_scale_down_ms", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("quota_exceeded_until_ms", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("quota_reason", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
        Column("updated_at_ms", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
    ),
)

# Migration 0003: slices table
SLICES = Table(
    "slices",
    "sl",
    columns=(
        Column("slice_id", "TEXT", "PRIMARY KEY", python_type=str, decoder=str),
        Column("scale_group", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("lifecycle", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column(
            "worker_ids",
            "TEXT",
            "NOT NULL DEFAULT '[]'",
            python_type=list,
            decoder=_decode_json_list,
            default_factory=list,
        ),
        Column("created_at_ms", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("last_active_ms", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
        Column("error_message", "TEXT", "NOT NULL DEFAULT ''", python_type=str, decoder=str, default=""),
    ),
    indexes=("CREATE INDEX IF NOT EXISTS idx_slices_scale_group ON slices(scale_group)",),
)

RESERVATION_CLAIMS = Table(
    "reservation_claims",
    "rc",
    columns=(
        Column("worker_id", "TEXT", "PRIMARY KEY", python_type=WorkerId, decoder=decode_worker_id),
        Column("job_id", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("entry_idx", "INTEGER", "NOT NULL", python_type=int, decoder=int),
    ),
)

LOGS = Table(
    "logs",
    "l",
    columns=(
        Column("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
        Column("key", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("source", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("data", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("epoch_ms", "INTEGER", "NOT NULL", python_type=int, decoder=int),
        Column("level", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int, default=0),
    ),
    indexes=("CREATE INDEX IF NOT EXISTS idx_logs_key ON logs(key, id)",),
)

# Migration 0005 + 0014 + 0023 (moved to profiles DB)
TASK_PROFILES = Table(
    "profiles.task_profiles",
    "tp",
    columns=(
        Column("id", "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
        Column("task_id", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("profile_data", "BLOB", "NOT NULL", expensive=True),
        Column(
            "captured_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="captured_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
        # Migration 0014
        Column("profile_kind", "TEXT", "NOT NULL DEFAULT 'cpu'", python_type=str, decoder=str, default="cpu"),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS profiles.idx_task_profiles_task_kind"
        " ON task_profiles(task_id, profile_kind, id DESC)",
    ),
    triggers=(
        # Trigger lives in the profiles schema; SQLite prohibits qualified table
        # names inside trigger bodies, so we use unqualified references.
        """CREATE TRIGGER IF NOT EXISTS profiles.trg_task_profiles_cap
AFTER INSERT ON task_profiles
BEGIN
  DELETE FROM task_profiles
   WHERE task_id = NEW.task_id
     AND profile_kind = NEW.profile_kind
     AND id NOT IN (
       SELECT id FROM task_profiles
        WHERE task_id = NEW.task_id
          AND profile_kind = NEW.profile_kind
        ORDER BY id DESC
        LIMIT 10
     );
END;""",
    ),
)

# ---------------------------------------------------------------------------
# Auth DB tables (in attached "auth" database)
# ---------------------------------------------------------------------------

AUTH_API_KEYS = Table(
    "auth.api_keys",
    "ak",
    columns=(
        Column("key_id", "TEXT", "PRIMARY KEY", python_type=str, decoder=str),
        Column("key_hash", "TEXT", "NOT NULL UNIQUE", python_type=str, decoder=str),
        Column("key_prefix", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("user_id", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column("name", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column(
            "created_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="created_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
        Column(
            "last_used_at_ms",
            "INTEGER",
            "",
            python_name="last_used_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
            default=None,
        ),
        Column(
            "expires_at_ms",
            "INTEGER",
            "",
            python_name="expires_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
            default=None,
        ),
        Column(
            "revoked_at_ms",
            "INTEGER",
            "",
            python_name="revoked_at",
            python_type=Timestamp | None,
            decoder=_nullable(decode_timestamp_ms),
            default=None,
        ),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS auth.idx_api_keys_hash ON api_keys(key_hash)",
        "CREATE INDEX IF NOT EXISTS auth.idx_api_keys_user ON api_keys(user_id)",
    ),
)

USER_BUDGETS = Table(
    "user_budgets",
    "ub",
    columns=(
        Column("user_id", "TEXT", "PRIMARY KEY REFERENCES users(user_id)", python_type=str, decoder=str),
        Column("budget_limit", "INTEGER", "NOT NULL DEFAULT 0", python_type=int, decoder=int),
        Column("max_band", "INTEGER", "NOT NULL DEFAULT 2", python_type=int, decoder=int),
        Column(
            "updated_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="updated_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
    ),
)

AUTH_CONTROLLER_SECRETS = Table(
    "auth.controller_secrets",
    "cs",
    columns=(
        Column("key", "TEXT", "PRIMARY KEY", python_type=str, decoder=str),
        Column("value", "TEXT", "NOT NULL", python_type=str, decoder=str),
        Column(
            "created_at_ms",
            "INTEGER",
            "NOT NULL",
            python_name="created_at",
            python_type=Timestamp,
            decoder=decode_timestamp_ms,
        ),
    ),
)

# ---------------------------------------------------------------------------
# All main DB tables in dependency order (for generate_full_ddl)
# ---------------------------------------------------------------------------

MAIN_TABLES: tuple[Table, ...] = (
    SCHEMA_MIGRATIONS,
    META,
    USERS,
    JOBS,
    JOB_CONFIG,
    JOB_WORKDIR_FILES,
    TASKS,
    TASK_ATTEMPTS,
    WORKERS,
    WORKER_ATTRIBUTES,
    WORKER_TASK_HISTORY,
    WORKER_RESOURCE_HISTORY,
    TASK_RESOURCE_HISTORY,
    ENDPOINTS,
    DISPATCH_QUEUE,
    TXN_LOG,
    TXN_ACTIONS,
    SCALING_GROUPS,
    SLICES,
    RESERVATION_CLAIMS,
    LOGS,
    USER_BUDGETS,
)

AUTH_TABLES: tuple[Table, ...] = (
    AUTH_API_KEYS,
    AUTH_CONTROLLER_SECRETS,
)

PROFILES_TABLES: tuple[Table, ...] = (TASK_PROFILES,)

# ---------------------------------------------------------------------------
# Hand-written row dataclasses
#
# Detail types are strict supersets of their corresponding Row types: they
# contain all the same fields (in a compatible order) plus extras.  This
# structural relationship lets callers accept a Detail wherever a Row is
# expected without inheritance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobRow:
    """Lightweight job row for listings (jobs JOIN job_config, excludes constraints)."""

    job_id: JobName
    state: int
    submitted_at: Timestamp
    root_submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    scheduling_deadline_epoch_ms: int | None
    error: str | None
    exit_code: int | None
    num_tasks: int
    is_reservation_holder: bool
    has_reservation: bool
    name: str
    depth: int
    res_cpu_millicores: int
    res_memory_bytes: int
    res_disk_bytes: int
    res_device_json: str | None
    has_coscheduling: bool
    coscheduling_group_by: str
    scheduling_timeout_ms: int | None
    max_task_failures: int


@dataclass(frozen=True, slots=True)
class JobSchedulingRow:
    """Full job row for scheduling — adds constraints over JobRow."""

    job_id: JobName
    state: int
    submitted_at: Timestamp
    root_submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    scheduling_deadline_epoch_ms: int | None
    error: str | None
    exit_code: int | None
    num_tasks: int
    is_reservation_holder: bool
    has_reservation: bool
    name: str
    depth: int
    res_cpu_millicores: int
    res_memory_bytes: int
    res_disk_bytes: int
    res_device_json: str | None
    constraints_json: str | None
    has_coscheduling: bool
    coscheduling_group_by: str
    scheduling_timeout_ms: int | None
    max_task_failures: int


@dataclass(frozen=True, slots=True)
class JobDetailRow:
    """Full job detail — superset of JobSchedulingRow, adds dispatch config from job_config."""

    job_id: JobName
    state: int
    submitted_at: Timestamp
    root_submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    scheduling_deadline_epoch_ms: int | None
    error: str | None
    exit_code: int | None
    num_tasks: int
    is_reservation_holder: bool
    has_reservation: bool
    name: str
    depth: int
    res_cpu_millicores: int
    res_memory_bytes: int
    res_disk_bytes: int
    res_device_json: str | None
    constraints_json: str | None
    has_coscheduling: bool
    coscheduling_group_by: str
    scheduling_timeout_ms: int | None
    max_task_failures: int
    entrypoint_json: str
    environment_json: str
    bundle_id: str
    ports_json: str
    max_retries_failure: int
    max_retries_preemption: int
    timeout_ms: int | None
    preemption_policy: int
    existing_job_policy: int
    priority_band: int
    task_image: str
    reservation_json: str | None
    fail_if_exists: bool


@dataclass(frozen=True, slots=True)
class TaskRow:
    """Lightweight task row for scheduling."""

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at: Timestamp
    priority_band: int = 2


@dataclass(frozen=True, slots=True)
class TaskDetailRow:
    """Full task detail — superset of TaskRow, adds diagnostics and attempts."""

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at: Timestamp
    priority_band: int
    error: str | None
    exit_code: int | None
    started_at: Timestamp | None
    finished_at: Timestamp | None
    current_worker_id: WorkerId | None
    current_worker_address: str | None
    container_id: str | None = None
    attempts: tuple = dataclasses.field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class WorkerRow:
    """Worker row for scheduling and health checks."""

    worker_id: WorkerId
    address: str
    healthy: bool
    active: bool
    consecutive_failures: int
    last_heartbeat: Timestamp
    committed_cpu_millicores: int
    committed_mem: int
    committed_gpu: int
    committed_tpu: int
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    attributes: dict = dataclasses.field(default_factory=dict)
    available_cpu_millicores: int = 0
    available_memory: int = 0
    available_gpus: int = 0
    available_tpus: int = 0


@dataclass(frozen=True, slots=True)
class WorkerDetailRow:
    """Full worker detail — superset of WorkerRow, adds metadata scalar columns."""

    worker_id: WorkerId
    address: str
    healthy: bool
    active: bool
    consecutive_failures: int
    last_heartbeat: Timestamp
    committed_cpu_millicores: int
    committed_mem: int
    committed_gpu: int
    committed_tpu: int
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    md_hostname: str
    md_ip_address: str
    md_cpu_count: int
    md_memory_bytes: int
    md_disk_bytes: int
    md_tpu_name: str
    md_tpu_worker_hostnames: str
    md_tpu_worker_id: str
    md_tpu_chips_per_host_bounds: str
    md_gpu_count: int
    md_gpu_name: str
    md_gpu_memory_mb: int
    md_gce_instance_name: str
    md_gce_zone: str
    md_git_hash: str
    md_device_json: str
    attributes: dict = dataclasses.field(default_factory=dict)
    available_cpu_millicores: int = 0
    available_memory: int = 0
    available_gpus: int = 0
    available_tpus: int = 0


@dataclass(frozen=True, slots=True)
class AttemptRow:
    """Task attempt row."""

    task_id: JobName
    attempt_id: int
    worker_id: WorkerId | None
    state: int
    created_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    exit_code: int | None
    error: str | None


@dataclass(frozen=True, slots=True)
class EndpointRow:
    """Registered service endpoint."""

    endpoint_id: str
    name: str
    address: str
    job_id: JobName
    metadata: dict
    registered_at: Timestamp
    task_id: JobName | None = None


@dataclass(frozen=True, slots=True)
class TransactionActionRow:
    """Transaction action log entry."""

    timestamp: Timestamp
    action: str
    entity_id: str
    details: dict


@dataclass(frozen=True, slots=True)
class ApiKeyRow:
    """API key record."""

    key_id: str
    key_hash: str
    key_prefix: str
    user_id: str
    name: str
    created_at: Timestamp
    last_used_at: Timestamp | None = None
    expires_at: Timestamp | None = None
    revoked_at: Timestamp | None = None


@dataclass(frozen=True, slots=True)
class UserBudgetRow:
    """User budget record."""

    user_id: str
    budget_limit: int
    max_band: int
    updated_at: Timestamp


# ---------------------------------------------------------------------------
# Projections -- typed column subsets that replace hand-maintained column strings
# ---------------------------------------------------------------------------


def _job_columns(*names: str) -> tuple[tuple[Column, ...], tuple[str, ...]]:
    """Look up Column objects from JOBS or JOB_CONFIG by name.

    Returns ``(columns, aliases)`` where each alias is the table alias
    (``j`` or ``jc``) that the column belongs to.  This lets Projection
    generate correct ``alias.col`` qualifiers for cross-table queries.
    """
    cols: list[Column] = []
    aliases: list[str] = []
    for n in names:
        if n in JOBS._column_map:
            cols.append(JOBS._column_map[n])
            aliases.append(JOBS.alias)
        elif n in JOB_CONFIG._column_map:
            cols.append(JOB_CONFIG._column_map[n])
            aliases.append(JOB_CONFIG.alias)
        else:
            raise KeyError(f"Unknown job column: {n!r}")
    return tuple(cols), tuple(aliases)


# SQL for job queries that need config: JOIN job_config jc ON jc.job_id = j.job_id
JOB_CONFIG_JOIN = "JOIN job_config jc ON jc.job_id = j.job_id"

# Lightweight job row for listings (excludes constraints).
_job_row_cols, _job_row_aliases = _job_columns(
    "job_id",
    "state",
    "submitted_at_ms",
    "root_submitted_at_ms",
    "started_at_ms",
    "finished_at_ms",
    "scheduling_deadline_epoch_ms",
    "error",
    "exit_code",
    "num_tasks",
    "is_reservation_holder",
    "has_reservation",
    "name",
    "depth",
    "res_cpu_millicores",
    "res_memory_bytes",
    "res_disk_bytes",
    "res_device_json",
    "has_coscheduling",
    "coscheduling_group_by",
    "scheduling_timeout_ms",
    "max_task_failures",
)
JOB_ROW_PROJECTION = Projection(
    JOBS,
    _job_row_cols,
    row_cls=JobRow,
    column_aliases=_job_row_aliases,
)

# Full job row for scheduling (includes constraints).
_job_sched_cols, _job_sched_aliases = _job_columns(
    "job_id",
    "state",
    "submitted_at_ms",
    "root_submitted_at_ms",
    "started_at_ms",
    "finished_at_ms",
    "scheduling_deadline_epoch_ms",
    "error",
    "exit_code",
    "num_tasks",
    "is_reservation_holder",
    "has_reservation",
    "name",
    "depth",
    "res_cpu_millicores",
    "res_memory_bytes",
    "res_disk_bytes",
    "res_device_json",
    "constraints_json",
    "has_coscheduling",
    "coscheduling_group_by",
    "scheduling_timeout_ms",
    "max_task_failures",
)
JOB_SCHEDULING_PROJECTION = Projection(
    JOBS,
    _job_sched_cols,
    row_cls=JobSchedulingRow,
    column_aliases=_job_sched_aliases,
)

# Worker row for scheduling and health checks.
WORKER_ROW_PROJECTION = WORKERS.projection(
    "worker_id",
    "address",
    "healthy",
    "active",
    "consecutive_failures",
    "last_heartbeat_ms",
    "committed_cpu_millicores",
    "committed_mem_bytes",
    "committed_gpu",
    "committed_tpu",
    "total_cpu_millicores",
    "total_memory_bytes",
    "total_gpu_count",
    "total_tpu_count",
    "device_type",
    "device_variant",
    extra_fields=(
        ExtraField("attributes", dict, default_factory=dict),
        ExtraField("available_cpu_millicores", int, default=0),
        ExtraField("available_memory", int, default=0),
        ExtraField("available_gpus", int, default=0),
        ExtraField("available_tpus", int, default=0),
    ),
    row_cls=WorkerRow,
)

# Task row for scheduling.
TASK_ROW_PROJECTION = TASKS.projection(
    "task_id",
    "job_id",
    "state",
    "current_attempt_id",
    "failure_count",
    "preemption_count",
    "max_retries_failure",
    "max_retries_preemption",
    "submitted_at_ms",
    "priority_band",
    row_cls=TaskRow,
)

# ---------------------------------------------------------------------------
# Detail / full-entity projections
# ---------------------------------------------------------------------------

# Full job detail — superset of JobSchedulingRow, adds dispatch config.
_job_detail_cols, _job_detail_aliases = _job_columns(
    "job_id",
    "state",
    "submitted_at_ms",
    "root_submitted_at_ms",
    "started_at_ms",
    "finished_at_ms",
    "scheduling_deadline_epoch_ms",
    "error",
    "exit_code",
    "num_tasks",
    "is_reservation_holder",
    "has_reservation",
    "name",
    "depth",
    "res_cpu_millicores",
    "res_memory_bytes",
    "res_disk_bytes",
    "res_device_json",
    "constraints_json",
    "has_coscheduling",
    "coscheduling_group_by",
    "scheduling_timeout_ms",
    "max_task_failures",
    "entrypoint_json",
    "environment_json",
    "bundle_id",
    "ports_json",
    "max_retries_failure",
    "max_retries_preemption",
    "timeout_ms",
    "preemption_policy",
    "existing_job_policy",
    "priority_band",
    "task_image",
    "reservation_json",
    "fail_if_exists",
)
JOB_DETAIL_PROJECTION = Projection(
    JOBS,
    _job_detail_cols,
    row_cls=JobDetailRow,
    column_aliases=_job_detail_aliases,
)

# Full task detail — superset of TaskRow, adds diagnostics and attempts.
TASK_DETAIL_PROJECTION = TASKS.projection(
    "task_id",
    "job_id",
    "state",
    "current_attempt_id",
    "failure_count",
    "preemption_count",
    "max_retries_failure",
    "max_retries_preemption",
    "submitted_at_ms",
    "priority_band",
    "error",
    "exit_code",
    "started_at_ms",
    "finished_at_ms",
    "current_worker_id",
    "current_worker_address",
    "container_id",
    extra_fields=(ExtraField("attempts", tuple, default_factory=tuple),),
    row_cls=TaskDetailRow,
)

# Full worker detail — superset of WorkerRow, adds metadata scalar columns.
WORKER_DETAIL_PROJECTION = WORKERS.projection(
    "worker_id",
    "address",
    "healthy",
    "active",
    "consecutive_failures",
    "last_heartbeat_ms",
    "committed_cpu_millicores",
    "committed_mem_bytes",
    "committed_gpu",
    "committed_tpu",
    "total_cpu_millicores",
    "total_memory_bytes",
    "total_gpu_count",
    "total_tpu_count",
    "device_type",
    "device_variant",
    "md_hostname",
    "md_ip_address",
    "md_cpu_count",
    "md_memory_bytes",
    "md_disk_bytes",
    "md_tpu_name",
    "md_tpu_worker_hostnames",
    "md_tpu_worker_id",
    "md_tpu_chips_per_host_bounds",
    "md_gpu_count",
    "md_gpu_name",
    "md_gpu_memory_mb",
    "md_gce_instance_name",
    "md_gce_zone",
    "md_git_hash",
    "md_device_json",
    extra_fields=(
        ExtraField("attributes", dict, default_factory=dict),
        ExtraField("available_cpu_millicores", int, default=0),
        ExtraField("available_memory", int, default=0),
        ExtraField("available_gpus", int, default=0),
        ExtraField("available_tpus", int, default=0),
    ),
    row_cls=WorkerDetailRow,
)

# Task attempt row.
ATTEMPT_PROJECTION = TASK_ATTEMPTS.projection(
    "task_id",
    "attempt_id",
    "worker_id",
    "state",
    "created_at_ms",
    "started_at_ms",
    "finished_at_ms",
    "exit_code",
    "error",
    row_cls=AttemptRow,
)

# Endpoint row.
ENDPOINT_PROJECTION = ENDPOINTS.projection(
    "endpoint_id",
    "name",
    "address",
    "job_id",
    "task_id",
    "metadata_json",
    "registered_at_ms",
    row_cls=EndpointRow,
)

# Transaction action row.
TXN_ACTION_PROJECTION = TXN_ACTIONS.projection(
    "created_at_ms",
    "action",
    "entity_id",
    "details_json",
    row_cls=TransactionActionRow,
)

# API key row.
API_KEY_PROJECTION = AUTH_API_KEYS.projection(
    "key_id",
    "key_hash",
    "key_prefix",
    "user_id",
    "name",
    "created_at_ms",
    "last_used_at_ms",
    "expires_at_ms",
    "revoked_at_ms",
    row_cls=ApiKeyRow,
)

# User budget row.
USER_BUDGET_PROJECTION = USER_BUDGETS.projection(
    "user_id",
    "budget_limit",
    "max_band",
    "updated_at_ms",
    row_cls=UserBudgetRow,
)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def tasks_with_attempts(tasks: Sequence[TaskDetailRow], attempts: Sequence[AttemptRow]) -> list[TaskDetailRow]:
    """Attach attempt rows to their parent task detail rows.

    Groups attempts by task_id and returns copies of each task with its
    ``attempts`` field populated as a tuple.
    """
    attempts_by_task: dict[JobName, list[AttemptRow]] = {}
    for attempt in attempts:
        attempts_by_task.setdefault(attempt.task_id, []).append(attempt)
    return [dataclasses.replace(task, attempts=tuple(attempts_by_task.get(task.task_id, ()))) for task in tasks]
