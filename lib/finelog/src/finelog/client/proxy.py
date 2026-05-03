# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protocol adapter that forwards push_logs/fetch_logs to a remote LogService over RPC."""

from __future__ import annotations

from collections.abc import Iterable

from connectrpc.interceptor import Interceptor

from finelog.rpc import logging_pb2
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
