# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client-directed SQL over immutable per-table Finelog catalog snapshots."""

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TypeVar

import duckdb
import pyarrow as pa
from google.protobuf import json_format
from google.protobuf.message import Message
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

from finelog.errors import QueryResultTooLargeError, QueryTimeoutError, StatsError
from finelog.rpc import finelog_stats_pb2 as stats_pb2


@dataclass(frozen=True)
class CatalogColumn:
    name: str
    duckdb_type: str


@dataclass(frozen=True)
class CatalogPin:
    namespace: str
    catalog_generation: int
    table_spec_version: int
    max_query_time_ms: int
    high_water: int
    object_uris: tuple[str, ...]
    columns: tuple[CatalogColumn, ...]


_OBJECT_CATALOG_ROOT = ("_finelog", "tables")
# Mirrors the server's TABLE_STATE_FORMAT_VERSION.
_SUPPORTED_CATALOG_FORMAT = 1
_MessageT = TypeVar("_MessageT", bound=Message)


class ObjectQueryClient:
    """Execute client-side SQL over stable object-backed Finelog segments."""

    def __init__(
        self,
        object_store_root: str,
    ) -> None:
        if not object_store_root:
            raise ValueError("object_store_root must not be empty")
        self._root = StoragePath(object_store_root)
        self._filesystem, _ = url_to_fs(str(self._root))

    def query(
        self,
        sql: str,
        *,
        namespaces: Iterable[str],
        max_rows: int = 100_000,
    ) -> pa.Table:
        """Return the SQL result from catalog snapshots pinned before execution."""
        started = time.monotonic()
        names = tuple(dict.fromkeys(namespaces))
        if not names:
            raise ValueError("client-directed queries require at least one namespace")
        pins = tuple(self.pin_catalog(namespace) for namespace in names)
        lifetime_ms = min(pin.max_query_time_ms for pin in pins)
        deadline = started + lifetime_ms / 1000
        if deadline <= time.monotonic():
            raise QueryTimeoutError("direct query catalog lifetime expired before execution")
        with duckdb.connect() as connection:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QueryTimeoutError("direct query catalog lifetime expired before execution")
            timed_out = threading.Event()

            def interrupt() -> None:
                timed_out.set()
                connection.interrupt()

            watchdog = threading.Timer(remaining, interrupt)
            watchdog.start()
            try:
                connection.register_filesystem(self._filesystem)
                for pin in pins:
                    self._register_namespace(connection, pin)
                table = connection.execute(sql).fetch_arrow_table()
            except duckdb.Error as error:
                if timed_out.is_set():
                    raise QueryTimeoutError(
                        f"direct query exceeded the pinned catalog lifetime of {lifetime_ms} ms"
                    ) from error
                raise
            finally:
                watchdog.cancel()
                watchdog.join()
        if table.num_rows > max_rows:
            raise QueryResultTooLargeError(
                f"query returned {table.num_rows} rows, exceeds max_rows={max_rows} "
                f"(add a LIMIT or pass a higher max_rows)"
            )
        return table

    def pin_catalog(self, namespace: str) -> CatalogPin:
        """Load one immutable catalog projection for a direct query."""
        if not namespace or "/" in namespace or namespace in {".", ".."}:
            raise ValueError(f"invalid namespace {namespace!r}")
        head, catalog = self._load_catalog(namespace)
        active_version, spec = _validated_active_spec(namespace, head, catalog)
        object_uris = tuple(
            self._object_uri(
                _table_object_id(
                    segment.source.object_id,
                    namespace,
                    "directQuerySegments[].source.objectId",
                ),
            )
            for segment in catalog.direct_query_segments
        )
        if catalog.max_query_time_ms == 0:
            raise StatsError(f"catalog for namespace {namespace!r} has no direct-query lifetime")
        return CatalogPin(
            namespace=namespace,
            catalog_generation=catalog.catalog_generation,
            table_spec_version=active_version,
            max_query_time_ms=catalog.max_query_time_ms,
            high_water=catalog.direct_query_high_water,
            object_uris=object_uris,
            columns=_catalog_columns(spec.logical_schema),
        )

    def _load_catalog(self, namespace: str) -> tuple[stats_pb2.CatalogHead, stats_pb2.NamespaceCatalog]:
        table_root = self._table_root(namespace)
        head = self._read_message(table_root / "HEAD.json", "HEAD", stats_pb2.CatalogHead())
        if head.tombstoned:
            raise StatsError(f"namespace {namespace!r} was deleted")
        catalog_id = _table_object_id(head.catalog.object_id, namespace, "HEAD.catalog.objectId")
        catalog_path = self._root / catalog_id
        catalog_bytes = self._read_bytes(catalog_path)
        catalog = _parse_message(
            catalog_bytes,
            stats_pb2.NamespaceCatalog(),
            f"catalog for namespace {namespace!r}",
        )
        return head, catalog

    def _register_namespace(self, connection: duckdb.DuckDBPyConnection, pin: CatalogPin) -> None:
        escaped_namespace = pin.namespace.replace('"', '""')
        if pin.object_uris:
            files = ", ".join(f"'{uri.replace(chr(39), chr(39) * 2)}'" for uri in pin.object_uris)
            connection.execute(
                f'CREATE VIEW "{escaped_namespace}" AS ' f"SELECT * FROM read_parquet([{files}], union_by_name=true)"
            )
            return
        columns = _empty_columns(pin.columns)
        connection.execute(f'CREATE VIEW "{escaped_namespace}" AS SELECT {columns} WHERE FALSE')

    def _table_root(self, namespace: str) -> StoragePath:
        return self._root / _OBJECT_CATALOG_ROOT[0] / _OBJECT_CATALOG_ROOT[1] / namespace

    def _object_uri(self, object_id: str) -> str:
        return str(self._root / object_id)

    def _read_bytes(self, path: StoragePath) -> bytes:
        try:
            return path.read_bytes()
        except OSError as error:
            raise StatsError(f"read object catalog {str(path)!r}: {error}") from error

    def _read_message(self, path: StoragePath, kind: str, message: _MessageT) -> _MessageT:
        return _parse_message(self._read_bytes(path), message, f"{kind} at {str(path)!r}")


def _parse_message(data: bytes, message: _MessageT, description: str) -> _MessageT:
    try:
        return json_format.Parse(data.decode(), message, ignore_unknown_fields=True)
    except (UnicodeDecodeError, json_format.ParseError) as error:
        raise StatsError(f"invalid {description} JSON: {error}") from error


def _validated_active_spec(
    namespace: str,
    head: stats_pb2.CatalogHead,
    catalog: stats_pb2.NamespaceCatalog,
) -> tuple[int, stats_pb2.TableSpec]:
    if head.format_version != _SUPPORTED_CATALOG_FORMAT or catalog.format_version != _SUPPORTED_CATALOG_FORMAT:
        raise StatsError(
            f"unsupported object catalog format for namespace {namespace!r}: "
            f"HEAD={head.format_version}, catalog={catalog.format_version}"
        )
    active_version = catalog.active_table_spec_version
    if active_version == 0:
        raise StatsError(
            f"namespace {namespace!r} is still using its legacy query version; "
            "client-directed reads become available after object-store activation"
        )
    if (
        head.namespace != namespace
        or catalog.namespace != namespace
        or catalog.catalog_generation != head.catalog_generation
        or active_version != head.active_table_spec_version
    ):
        raise StatsError(f"HEAD/catalog identity mismatch for namespace {namespace!r}")
    spec = next(
        (item for item in catalog.retained_table_specs if item.version == active_version),
        None,
    )
    if spec is None:
        raise StatsError(f"catalog for namespace {namespace!r} has no retained TableSpec {active_version}")
    if not spec.HasField("logical_schema"):
        raise StatsError(f"catalog for namespace {namespace!r} TableSpec {active_version} has no logical schema")
    if spec.operating_policy.l0_mode != stats_pb2.L0_MODE_OBJECT_STORE:
        raise StatsError(f"namespace {namespace!r} active TableSpec {active_version} is not object-backed")
    return active_version, spec


def _table_object_id(value: str, namespace: str, field: str) -> str:
    if not value:
        raise StatsError(f"{field} must be a non-empty object ID")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or ".." in path.parts or "\\" in value or "://" in value:
        raise StatsError(f"{field} must be a canonical relative object ID")
    expected = (*_OBJECT_CATALOG_ROOT, namespace)
    if path.parts[: len(expected)] != expected or len(path.parts) < len(expected) + 1:
        raise StatsError(f"{field} must identify an object in table {namespace!r}")
    return str(path)


_DUCKDB_TYPES: dict[int, str] = {
    stats_pb2.COLUMN_TYPE_STRING: "VARCHAR",
    stats_pb2.COLUMN_TYPE_INT64: "BIGINT",
    stats_pb2.COLUMN_TYPE_FLOAT64: "DOUBLE",
    stats_pb2.COLUMN_TYPE_BOOL: "BOOLEAN",
    stats_pb2.COLUMN_TYPE_TIMESTAMP_MS: "TIMESTAMP_MS",
    stats_pb2.COLUMN_TYPE_BYTES: "BLOB",
    stats_pb2.COLUMN_TYPE_INT32: "INTEGER",
    stats_pb2.COLUMN_TYPE_MAP: "MAP(VARCHAR, VARCHAR)",
    stats_pb2.COLUMN_TYPE_FLOAT64_LIST: "DOUBLE[]",
    stats_pb2.COLUMN_TYPE_INT64_LIST: "BIGINT[]",
}


def _catalog_columns(logical_schema: stats_pb2.Schema) -> tuple[CatalogColumn, ...]:
    declarations = [CatalogColumn("seq", "BIGINT")]
    for column in logical_schema.columns:
        if not column.name:
            raise StatsError("logicalSchema column name must be non-empty")
        if column.type not in _DUCKDB_TYPES:
            raise StatsError(f"unsupported logicalSchema type {column.type!r}")
        declarations.append(CatalogColumn(column.name, _DUCKDB_TYPES[column.type]))
    if all(column.name != "cluster" for column in declarations):
        declarations.append(CatalogColumn("cluster", "VARCHAR"))
    return tuple(declarations)


def _empty_columns(columns: tuple[CatalogColumn, ...]) -> str:
    return ", ".join(
        f'CAST(NULL AS {column.duckdb_type}) AS "{column.name.replace(chr(34), chr(34) * 2)}"' for column in columns
    )
