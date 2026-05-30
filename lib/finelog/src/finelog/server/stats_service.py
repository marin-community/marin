# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""StatsService RPC implementation; delegates to DuckDBLogStore.

Handlers are async so ``write_rows`` can park its persistence wait on the
event loop. The underlying LogStore is sync; CPU/IO-bound calls go
through :func:`asyncio.to_thread` so the loop never blocks on duckdb or
parquet work.
"""

from __future__ import annotations

import asyncio
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
from finelog.store.policy import StoragePolicy
from finelog.store.schema import (
    InvalidNamespaceError,
    NamespaceNotFoundError,
    SchemaConflictError,
    SchemaValidationError,
    schema_from_proto,
    schema_to_proto,
)

from .service import DEFAULT_PERSIST_TIMEOUT_SEC, await_persisted

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

    def __init__(
        self,
        *,
        log_store: LogStore,
        persist_timeout_sec: float = DEFAULT_PERSIST_TIMEOUT_SEC,
    ) -> None:
        self._log_store = log_store
        self._persist_timeout_sec = persist_timeout_sec

    async def register_table(
        self,
        request: stats_pb2.RegisterTableRequest,
        ctx: Any,
    ) -> stats_pb2.RegisterTableResponse:
        try:
            schema = schema_from_proto(request.schema)
            policy = StoragePolicy.from_proto(request.storage_policy)
            effective_schema = await asyncio.to_thread(self._log_store.register_table, request.namespace, schema, policy)
        except InvalidNamespaceError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except SchemaConflictError as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except SchemaValidationError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        # Policy is last-write-wins, so the request's policy is now in
        # force; surface it back so clients can confirm.
        effective_policy = self._log_store.catalog.get_policy(request.namespace)
        return stats_pb2.RegisterTableResponse(
            effective_schema=schema_to_proto(effective_schema),
            effective_policy=effective_policy.to_proto(),
        )

    async def write_rows(
        self,
        request: stats_pb2.WriteRowsRequest,
        ctx: Any,
    ) -> stats_pb2.WriteRowsResponse:
        try:
            n, last_seq = await asyncio.to_thread(
                self._log_store.write_rows, request.namespace, bytes(request.arrow_ipc)
            )
        except NamespaceNotFoundError as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except SchemaValidationError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        await await_persisted(
            self._log_store,
            request.namespace,
            last_seq,
            timeout=self._persist_timeout_sec,
        )
        return stats_pb2.WriteRowsResponse(rows_written=n)

    async def query(
        self,
        request: stats_pb2.QueryRequest,
        ctx: Any,
    ) -> stats_pb2.QueryResponse:
        # DuckDB errors (CatalogException, ParserException, BinderException)
        # all derive from duckdb.Error; surface them as INVALID_ARGUMENT.
        # Other exceptions propagate as INTERNAL — the right signal for
        # "operator: this is a server bug".
        try:
            table = await asyncio.to_thread(self._log_store.query, request.sql)
        except (InvalidNamespaceError, SchemaValidationError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except duckdb.Error as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, f"query failed: {exc}") from exc
        ipc = _arrow_table_to_ipc_bytes(table)
        return stats_pb2.QueryResponse(arrow_ipc=ipc, row_count=table.num_rows)

    async def drop_table(
        self,
        request: stats_pb2.DropTableRequest,
        ctx: Any,
    ) -> stats_pb2.DropTableResponse:
        try:
            await asyncio.to_thread(self._log_store.drop_table, request.namespace)
        except NamespaceNotFoundError as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except InvalidNamespaceError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return stats_pb2.DropTableResponse()

    async def list_namespaces(
        self,
        request: stats_pb2.ListNamespacesRequest,
        ctx: Any,
    ) -> stats_pb2.ListNamespacesResponse:
        entries = await asyncio.to_thread(self._log_store.list_namespaces_with_stats)
        infos = [
            stats_pb2.NamespaceInfo(
                namespace=name,
                schema=schema_to_proto(schema),
                row_count=stats.row_count,
                byte_size=stats.byte_size,
                min_seq=stats.min_seq,
                max_seq=stats.max_seq,
                segment_count=stats.segment_count,
                storage_policy=policy.to_proto(),
            )
            for name, schema, stats, policy in entries
        ]
        return stats_pb2.ListNamespacesResponse(namespaces=infos)

    async def get_table_schema(
        self,
        request: stats_pb2.GetTableSchemaRequest,
        ctx: Any,
    ) -> stats_pb2.GetTableSchemaResponse:
        try:
            schema = await asyncio.to_thread(self._log_store.get_table_schema, request.namespace)
        except NamespaceNotFoundError as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return stats_pb2.GetTableSchemaResponse(schema=schema_to_proto(schema))
