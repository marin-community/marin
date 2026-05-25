# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protocol adapters that forward LogService/StatsService RPCs to a remote server.

These proxies satisfy the async ``LogService`` / ``StatsService`` Protocols
expected by the generated ConnectRPC ASGI applications. Under the hood we
still use the sync ConnectRPC clients (they own the dial/timeout logic);
each call is dispatched to a worker thread via :func:`asyncio.to_thread`
so the event loop never blocks on a network round-trip.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from connectrpc.interceptor import Interceptor

from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync
from finelog.rpc.logging_connect import LogServiceClientSync


class LogServiceProxy:
    """Async-protocol adapter forwarding LogService calls to a remote server.

    Used in place of ``LogServiceImpl`` when the log service is hosted in
    a separate process.
    """

    def __init__(
        self,
        address: str,
        timeout_ms: int = 10_000,
        interceptors: Iterable[Interceptor] = (),
    ) -> None:
        self._client = LogServiceClientSync(address=address, timeout_ms=timeout_ms, interceptors=tuple(interceptors))

    async def push_logs(
        self,
        request: logging_pb2.PushLogsRequest,
        ctx: object,
    ) -> logging_pb2.PushLogsResponse:
        return await asyncio.to_thread(self._client.push_logs, request)

    async def fetch_logs(
        self,
        request: logging_pb2.FetchLogsRequest,
        ctx: object,
    ) -> logging_pb2.FetchLogsResponse:
        return await asyncio.to_thread(self._client.fetch_logs, request)

    def close(self) -> None:
        self._client.close()


class StatsServiceProxy:
    """Async-protocol adapter forwarding StatsService calls to a remote server.

    Used by the controller dashboard to expose the bundled log server's
    StatsService at the controller URL without a second port hop for
    external clients.
    """

    def __init__(
        self,
        address: str,
        timeout_ms: int = 10_000,
        interceptors: Iterable[Interceptor] = (),
    ) -> None:
        self._client = StatsServiceClientSync(address=address, timeout_ms=timeout_ms, interceptors=tuple(interceptors))

    async def register_table(
        self, request: stats_pb2.RegisterTableRequest, ctx: object
    ) -> stats_pb2.RegisterTableResponse:
        return await asyncio.to_thread(self._client.register_table, request)

    async def write_rows(self, request: stats_pb2.WriteRowsRequest, ctx: object) -> stats_pb2.WriteRowsResponse:
        return await asyncio.to_thread(self._client.write_rows, request)

    async def query(self, request: stats_pb2.QueryRequest, ctx: object) -> stats_pb2.QueryResponse:
        return await asyncio.to_thread(self._client.query, request)

    async def drop_table(self, request: stats_pb2.DropTableRequest, ctx: object) -> stats_pb2.DropTableResponse:
        return await asyncio.to_thread(self._client.drop_table, request)

    async def list_namespaces(
        self, request: stats_pb2.ListNamespacesRequest, ctx: object
    ) -> stats_pb2.ListNamespacesResponse:
        return await asyncio.to_thread(self._client.list_namespaces, request)

    async def get_table_schema(
        self, request: stats_pb2.GetTableSchemaRequest, ctx: object
    ) -> stats_pb2.GetTableSchemaResponse:
        return await asyncio.to_thread(self._client.get_table_schema, request)

    def close(self) -> None:
        self._client.close()
