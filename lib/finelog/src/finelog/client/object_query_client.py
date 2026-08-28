# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client-directed SQL over one immutable Finelog object catalog snapshot."""

import base64
import hashlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

import duckdb
import pyarrow as pa
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

from finelog.errors import QueryResultTooLargeError, QueryTimeoutError, StatsError
from finelog.rpc import finelog_stats_pb2 as stats_pb2

logger = logging.getLogger(__name__)


class QueryMode(StrEnum):
    """Where a Finelog SQL query executes."""

    SERVER = "server"
    CLIENT = "client"


@dataclass(frozen=True)
class CatalogPin:
    namespace: str
    catalog_generation: int
    table_spec_version: int
    max_query_time_ms: int
    object_uris: tuple[str, ...]
    logical_schema: dict[str, object]


QueryStartReporter = Callable[[str, str, tuple[CatalogPin, ...]], None]
QueryFinishReporter = Callable[[str, int, int, bool, str], None]


class ObjectQueryClient:
    """Execute client-side SQL over active object-native Finelog tables."""

    def __init__(
        self,
        object_store_root: str,
        *,
        report_start: QueryStartReporter | None = None,
        report_finish: QueryFinishReporter | None = None,
    ) -> None:
        if not object_store_root:
            raise ValueError("object_store_root must not be empty")
        self._root = StoragePath(object_store_root)
        self._filesystem, _ = url_to_fs(str(self._root))
        self._report_start = report_start
        self._report_finish = report_finish

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
        pins = tuple(self._load_catalog(namespace) for namespace in names)
        lifetime_ms = min(pin.max_query_time_ms for pin in pins)
        deadline = started + lifetime_ms / 1000
        if deadline <= time.monotonic():
            raise QueryTimeoutError("direct query catalog lifetime expired before execution")
        query_id = str(uuid.uuid4())
        self._best_effort_start(query_id, sql, pins)
        try:
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
        except Exception as error:
            self._best_effort_finish(
                query_id,
                started,
                0,
                False,
                type(error).__name__,
            )
            raise
        self._best_effort_finish(query_id, started, table.num_rows, True, "")
        return table

    def _load_catalog(self, namespace: str) -> CatalogPin:
        if not namespace or "/" in namespace or namespace in {".", ".."}:
            raise ValueError(f"invalid namespace {namespace!r}")
        native_root = self._native_namespace_root(namespace)
        head = self._read_json(native_root / "HEAD.json", "HEAD")
        catalog_ref = _mapping(head.get("catalog"), "HEAD.catalog")
        catalog_key = _relative_object_key(catalog_ref.get("uri"), "HEAD.catalog.uri")
        catalog_path = native_root / catalog_key
        catalog_bytes = self._read_bytes(catalog_path)
        expected_sha = _base64_bytes(catalog_ref.get("sha256"), "HEAD.catalog.sha256")
        if hashlib.sha256(catalog_bytes).digest() != expected_sha:
            raise StatsError(f"catalog checksum mismatch for namespace {namespace!r}")
        try:
            catalog = json.loads(catalog_bytes)
        except (TypeError, json.JSONDecodeError) as error:
            raise StatsError(f"invalid catalog JSON for namespace {namespace!r}: {error}") from error

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
                "client-directed reads become available after object-native activation"
            )
        if (
            head.get("namespace") != namespace
            or catalog.get("namespace") != namespace
            or generation != head_generation
            or active_version != _integer(head.get("activeTableSpecVersion"), "HEAD.activeTableSpecVersion")
        ):
            raise StatsError(f"HEAD/catalog identity mismatch for namespace {namespace!r}")

        version = next(
            (
                item
                for item in _sequence(catalog.get("versionSegments"), "catalog.versionSegments")
                if _integer(
                    _mapping(item, "catalog.versionSegments[]").get("tableSpecVersion"),
                    "versionSegments[].tableSpecVersion",
                )
                == active_version
            ),
            None,
        )
        if version is None:
            raise StatsError(f"catalog for namespace {namespace!r} has no active version {active_version}")
        object_uris = tuple(
            self._object_uri(
                native_root,
                _relative_object_key(
                    _mapping(
                        _mapping(segment, "liveSegments[]").get("source"),
                        "liveSegments[].source",
                    ).get("uri"),
                    "liveSegments[].source.uri",
                ),
            )
            for segment in _sequence(
                _mapping(version, "catalog.versionSegments[]").get("liveSegments"),
                "versionSegments[].liveSegments",
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
        if operating_policy.get("l0Mode") != "L0_MODE_OBJECT_NATIVE":
            raise StatsError(f"namespace {namespace!r} active TableSpec {active_version} is not object-native")
        logical_schema = _mapping(spec.get("logicalSchema"), "TableSpec.logicalSchema")
        max_query_time_ms = _integer(catalog.get("maxQueryTimeMs"), "catalog.maxQueryTimeMs")
        if max_query_time_ms == 0:
            raise StatsError(f"catalog for namespace {namespace!r} has no direct-query lifetime")
        return CatalogPin(
            namespace=namespace,
            catalog_generation=generation,
            table_spec_version=active_version,
            max_query_time_ms=max_query_time_ms,
            object_uris=object_uris,
            logical_schema=logical_schema,
        )

    def _register_namespace(self, connection: duckdb.DuckDBPyConnection, pin: CatalogPin) -> None:
        escaped_namespace = pin.namespace.replace('"', '""')
        if pin.object_uris:
            files = ", ".join(f"'{uri.replace(chr(39), chr(39) * 2)}'" for uri in pin.object_uris)
            connection.execute(
                f'CREATE VIEW "{escaped_namespace}" AS ' f"SELECT * FROM read_parquet([{files}], union_by_name=true)"
            )
            return
        columns = _empty_columns(pin.logical_schema)
        connection.execute(f'CREATE VIEW "{escaped_namespace}" AS SELECT {columns} WHERE FALSE')

    def _native_namespace_root(self, namespace: str) -> StoragePath:
        return self._root / "_native" / "namespaces" / namespace

    def _object_uri(self, native_root: StoragePath, relative_key: str) -> str:
        return str(native_root / relative_key)

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

    def _best_effort_start(self, query_id: str, sql: str, pins: tuple[CatalogPin, ...]) -> None:
        if self._report_start is None:
            return
        try:
            self._report_start(query_id, sql, pins)
        except Exception as error:
            logger.debug("direct query start report failed: %s", error)

    def _best_effort_finish(
        self,
        query_id: str,
        started: float,
        row_count: int,
        succeeded: bool,
        error_code: str,
    ) -> None:
        if self._report_finish is None:
            return
        try:
            self._report_finish(
                query_id,
                int((time.monotonic() - started) * 1000),
                row_count,
                succeeded,
                error_code,
            )
        except Exception as error:
            logger.debug("direct query finish report failed: %s", error)


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


def _relative_object_key(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise StatsError(f"{field} must be a non-empty relative object key")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "://" in value:
        raise StatsError(f"{field} must be a relative object key")
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


def _empty_columns(logical_schema: dict[str, object]) -> str:
    declarations = [("seq", "BIGINT")]
    for raw in _sequence(logical_schema.get("columns", []), "logicalSchema.columns"):
        column = _mapping(raw, "logicalSchema.columns[]")
        name = column.get("name")
        type_name = column.get("type")
        if not isinstance(name, str) or not name:
            raise StatsError("logicalSchema column name must be non-empty")
        if not isinstance(type_name, str) or type_name not in _DUCKDB_TYPES:
            raise StatsError(f"unsupported logicalSchema type {type_name!r}")
        declarations.append((name, _DUCKDB_TYPES[type_name]))
    if all(name != "cluster" for name, _ in declarations):
        declarations.append(("cluster", "VARCHAR"))
    return ", ".join(
        f'CAST(NULL AS {type_name}) AS "{name.replace(chr(34), chr(34) * 2)}"' for name, type_name in declarations
    )


def query_catalog_versions(pins: tuple[CatalogPin, ...]) -> list[stats_pb2.QueryCatalogVersion]:
    """Return RPC catalog-version records for pinned snapshots."""
    return [
        stats_pb2.QueryCatalogVersion(
            namespace=pin.namespace,
            catalog_generation=pin.catalog_generation,
            table_spec_version=pin.table_spec_version,
        )
        for pin in pins
    ]
