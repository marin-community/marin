# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client-directed SQL over immutable per-table Finelog catalog snapshots."""

import base64
import hashlib
import json
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

import duckdb
import pyarrow as pa
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

from finelog.errors import QueryResultTooLargeError, QueryTimeoutError, StatsError


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
        table_root = self._table_root(namespace)
        head = self._read_json(table_root / "HEAD.json", "HEAD")
        catalog_ref = _mapping(head.get("catalog"), "HEAD.catalog")
        catalog_id = _table_object_id(catalog_ref.get("objectId"), namespace, "HEAD.catalog.objectId")
        catalog_path = self._root / catalog_id
        catalog_bytes = self._read_bytes(catalog_path)
        expected_sha = _base64_bytes(catalog_ref.get("sha256"), "HEAD.catalog.sha256")
        if hashlib.sha256(catalog_bytes).digest() != expected_sha:
            raise StatsError(f"catalog checksum mismatch for namespace {namespace!r}")
        try:
            catalog = json.loads(catalog_bytes)
        except (TypeError, json.JSONDecodeError) as error:
            raise StatsError(f"invalid catalog JSON for namespace {namespace!r}: {error}") from error

        if head.get("tombstoned"):
            raise StatsError(f"namespace {namespace!r} was deleted")

        head_format = _integer(head.get("formatVersion"), "HEAD.formatVersion")
        catalog_format = _integer(catalog.get("formatVersion"), "catalog.formatVersion")
        if head_format != 1 or catalog_format != 1:
            raise StatsError(
                f"unsupported object catalog format for namespace {namespace!r}: "
                f"HEAD={head_format}, catalog={catalog_format}"
            )

        generation = _integer(catalog.get("catalogGeneration"), "catalog.catalogGeneration")
        head_generation = _integer(head.get("catalogGeneration"), "HEAD.catalogGeneration")
        active_version = _integer(catalog.get("activeTableSpecVersion"), "catalog.activeTableSpecVersion")
        if active_version == 0:
            raise StatsError(
                f"namespace {namespace!r} is still using its legacy query version; "
                "client-directed reads become available after object-store activation"
            )
        if (
            head.get("namespace") != namespace
            or catalog.get("namespace") != namespace
            or generation != head_generation
            or active_version != _integer(head.get("activeTableSpecVersion"), "HEAD.activeTableSpecVersion")
        ):
            raise StatsError(f"HEAD/catalog identity mismatch for namespace {namespace!r}")

        object_uris = tuple(
            self._object_uri(
                _table_object_id(
                    _mapping(
                        _mapping(segment, "directQuerySegments[]").get("source"),
                        "directQuerySegments[].source",
                    ).get("objectId"),
                    namespace,
                    "directQuerySegments[].source.objectId",
                ),
            )
            for segment in _sequence(
                catalog.get("directQuerySegments", []),
                "catalog.directQuerySegments",
            )
        )
        spec = next(
            (
                _mapping(item, "retainedTableSpecs[]")
                for item in _sequence(catalog.get("retainedTableSpecs"), "catalog.retainedTableSpecs")
                if _integer(
                    _mapping(item, "retainedTableSpecs[]").get("version"),
                    "retainedTableSpecs[].version",
                )
                == active_version
            ),
            None,
        )
        if spec is None:
            raise StatsError(f"catalog for namespace {namespace!r} has no retained TableSpec {active_version}")
        operating_policy = _mapping(spec.get("operatingPolicy"), "TableSpec.operatingPolicy")
        if operating_policy.get("l0Mode") != "L0_MODE_OBJECT_STORE":
            raise StatsError(f"namespace {namespace!r} active TableSpec {active_version} is not object-backed")
        columns = _catalog_columns(_mapping(spec.get("logicalSchema"), "TableSpec.logicalSchema"))
        max_query_time_ms = _integer(catalog.get("maxQueryTimeMs"), "catalog.maxQueryTimeMs")
        if max_query_time_ms == 0:
            raise StatsError(f"catalog for namespace {namespace!r} has no direct-query lifetime")
        return CatalogPin(
            namespace=namespace,
            catalog_generation=generation,
            table_spec_version=active_version,
            max_query_time_ms=max_query_time_ms,
            high_water=_integer(catalog.get("directQueryHighWater", 0), "catalog.directQueryHighWater"),
            object_uris=object_uris,
            columns=columns,
        )

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

    def _read_json(self, path: StoragePath, kind: str) -> dict[str, object]:
        try:
            return _mapping(json.loads(self._read_bytes(path)), kind)
        except (TypeError, json.JSONDecodeError) as error:
            raise StatsError(f"invalid {kind} JSON at {str(path)!r}: {error}") from error


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise StatsError(f"{field} must be an object")
    return value


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise StatsError(f"{field} must be an array")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise StatsError(f"{field} must be an integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise StatsError(f"{field} must be an integer") from error
    if result < 0:
        raise StatsError(f"{field} must not be negative")
    return result


def _base64_bytes(value: object, field: str) -> bytes:
    if not isinstance(value, str):
        raise StatsError(f"{field} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise StatsError(f"{field} must be valid base64") from error


def _table_object_id(value: object, namespace: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise StatsError(f"{field} must be a non-empty object ID")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or ".." in path.parts or "\\" in value or "://" in value:
        raise StatsError(f"{field} must be a canonical relative object ID")
    expected = (*_OBJECT_CATALOG_ROOT, namespace)
    if path.parts[:3] != expected or len(path.parts) < 4:
        raise StatsError(f"{field} must identify an object in table {namespace!r}")
    return str(path)


_DUCKDB_TYPES = {
    "COLUMN_TYPE_STRING": "VARCHAR",
    "COLUMN_TYPE_INT64": "BIGINT",
    "COLUMN_TYPE_FLOAT64": "DOUBLE",
    "COLUMN_TYPE_BOOL": "BOOLEAN",
    "COLUMN_TYPE_TIMESTAMP_MS": "TIMESTAMP_MS",
    "COLUMN_TYPE_BYTES": "BLOB",
    "COLUMN_TYPE_INT32": "INTEGER",
    "COLUMN_TYPE_MAP": "MAP(VARCHAR, VARCHAR)",
    "COLUMN_TYPE_FLOAT64_LIST": "DOUBLE[]",
    "COLUMN_TYPE_INT64_LIST": "BIGINT[]",
}


def _catalog_columns(logical_schema: dict[str, object]) -> tuple[CatalogColumn, ...]:
    declarations = [CatalogColumn("seq", "BIGINT")]
    for raw in _sequence(logical_schema.get("columns", []), "logicalSchema.columns"):
        column = _mapping(raw, "logicalSchema.columns[]")
        name = column.get("name")
        type_name = column.get("type")
        if not isinstance(name, str) or not name:
            raise StatsError("logicalSchema column name must be non-empty")
        if not isinstance(type_name, str) or type_name not in _DUCKDB_TYPES:
            raise StatsError(f"unsupported logicalSchema type {type_name!r}")
        declarations.append(CatalogColumn(name, _DUCKDB_TYPES[type_name]))
    if all(column.name != "cluster" for column in declarations):
        declarations.append(CatalogColumn("cluster", "VARCHAR"))
    return tuple(declarations)


def _empty_columns(columns: tuple[CatalogColumn, ...]) -> str:
    return ", ".join(
        f'CAST(NULL AS {column.duckdb_type}) AS "{column.name.replace(chr(34), chr(34) * 2)}"' for column in columns
    )
