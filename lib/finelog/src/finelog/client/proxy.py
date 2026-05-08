# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protocol adapters that forward LogService/StatsService RPCs to a remote server."""

from __future__ import annotations

from collections.abc import Iterable

from connectrpc.interceptor import Interceptor

from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync
from finelog.rpc.logging_connect import LogServiceClientSync


class LogServiceProxy:
    """Bridges ``LogServiceClientSync`` (kwargs-only, ctx-less) to the
    ``LogServiceSync`` protocol (positional ``ctx`` arg) expected by
    ``LogServiceWSGIApplication`` and the controller/dashboard call sites.
    Used in place of ``LogServiceImpl`` when the log service is hosted
    in a separate process.
    """

    def __init__(
        self,
        address: str,
        timeout_ms: int = 10_000,
        interceptors: Iterable[Interceptor] = (),
    ) -> None:
        self._client = LogServiceClientSync(address=address, timeout_ms=timeout_ms, interceptors=tuple(interceptors))

    def push_logs(
        self,
        request: logging_pb2.PushLogsRequest,
        ctx: object,
    ) -> logging_pb2.PushLogsResponse:
        return self._client.push_logs(request)

    def fetch_logs(
        self,
        request: logging_pb2.FetchLogsRequest,
        ctx: object,
    ) -> logging_pb2.FetchLogsResponse:
        return self._client.fetch_logs(request)

    def close(self) -> None:
        self._client.close()


class StatsServiceProxy:
    """Forwards ``StatsServiceSync`` RPCs to a remote StatsService.

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

    def register_table(self, request: stats_pb2.RegisterTableRequest, ctx: object) -> stats_pb2.RegisterTableResponse:
        return self._client.register_table(request)

    def write_rows(self, request: stats_pb2.WriteRowsRequest, ctx: object) -> stats_pb2.WriteRowsResponse:
        return self._client.write_rows(request)

    def query(self, request: stats_pb2.QueryRequest, ctx: object) -> stats_pb2.QueryResponse:
        return self._client.query(request)

    def drop_table(self, request: stats_pb2.DropTableRequest, ctx: object) -> stats_pb2.DropTableResponse:
        return self._client.drop_table(request)

    def list_namespaces(self, request: stats_pb2.ListNamespacesRequest, ctx: object) -> stats_pb2.ListNamespacesResponse:
        return self._client.list_namespaces(request)

    def get_table_schema(
        self, request: stats_pb2.GetTableSchemaRequest, ctx: object
    ) -> stats_pb2.GetTableSchemaResponse:
        return self._client.get_table_schema(request)

    def close(self) -> None:
        self._client.close()
