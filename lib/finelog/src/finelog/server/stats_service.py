# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""StatsService RPC implementation; delegates to DuckDBLogStore."""

from __future__ import annotations

import io
import logging
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.ipc as paipc
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.store import LogStore
from finelog.store.schema import (
    InvalidNamespaceError,
    NamespaceNotFoundError,
    SchemaConflictError,
    SchemaValidationError,
    schema_from_proto,
    schema_to_proto,
)

logger = logging.getLogger(__name__)


def _arrow_table_to_ipc_bytes(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


class StatsServiceImpl:
    """Connect handler for ``finelog.stats.StatsService``.

    The LogStore is shared with ``LogServiceImpl`` (not owned) so one
    process serves both surfaces against one storage state.
    """

    def __init__(self, *, log_store: LogStore) -> None:
        self._log_store = log_store

    def register_table(
        self,
        request: stats_pb2.RegisterTableRequest,
        ctx: Any,
    ) -> stats_pb2.RegisterTableResponse:
        try:
            schema = schema_from_proto(request.schema)
            effective = self._log_store.register_table(request.namespace, schema)
        except InvalidNamespaceError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except SchemaConflictError as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except SchemaValidationError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return stats_pb2.RegisterTableResponse(effective_schema=schema_to_proto(effective))

    def write_rows(
        self,
        request: stats_pb2.WriteRowsRequest,
        ctx: Any,
    ) -> stats_pb2.WriteRowsResponse:
        try:
            n = self._log_store.write_rows(request.namespace, bytes(request.arrow_ipc))
        except NamespaceNotFoundError as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except SchemaValidationError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return stats_pb2.WriteRowsResponse(rows_written=n)

    def query(
        self,
        request: stats_pb2.QueryRequest,
        ctx: Any,
    ) -> stats_pb2.QueryResponse:
        # DuckDB errors (CatalogException, ParserException, BinderException)
        # all derive from duckdb.Error; surface them as INVALID_ARGUMENT.
        # Other exceptions propagate as INTERNAL — the right signal for
        # "operator: this is a server bug".
        try:
            table = self._log_store.query(request.sql)
        except (InvalidNamespaceError, SchemaValidationError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except duckdb.Error as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, f"query failed: {exc}") from exc
        ipc = _arrow_table_to_ipc_bytes(table)
        return stats_pb2.QueryResponse(arrow_ipc=ipc, row_count=table.num_rows)

    def drop_table(
        self,
        request: stats_pb2.DropTableRequest,
        ctx: Any,
    ) -> stats_pb2.DropTableResponse:
        try:
            self._log_store.drop_table(request.namespace)
        except NamespaceNotFoundError as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except InvalidNamespaceError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return stats_pb2.DropTableResponse()

    def list_namespaces(
        self,
        request: stats_pb2.ListNamespacesRequest,
        ctx: Any,
    ) -> stats_pb2.ListNamespacesResponse:
        infos = [
            stats_pb2.NamespaceInfo(namespace=name, schema=schema_to_proto(schema))
            for name, schema in self._log_store.list_namespaces()
        ]
        return stats_pb2.ListNamespacesResponse(namespaces=infos)

    def get_table_schema(
        self,
        request: stats_pb2.GetTableSchemaRequest,
        ctx: Any,
    ) -> stats_pb2.GetTableSchemaResponse:
        try:
            schema = self._log_store.get_table_schema(request.namespace)
        except NamespaceNotFoundError as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return stats_pb2.GetTableSchemaResponse(schema=schema_to_proto(schema))
