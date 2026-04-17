# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Server-side adapter that bridges LogServiceClientSync to the LogServiceSync protocol.

Used by the controller when the log service runs in a separate process:
the dashboard and controller expect a ``LogServiceSync``-shaped handle
(positional ``ctx`` arg) but ``LogServiceClientSync`` is a keyword-only
RPC client. This adapter plugs the gap.
"""

from __future__ import annotations

from collections.abc import Iterable

from connectrpc.interceptor import Interceptor

from iris.rpc import logging_pb2
from iris.rpc.logging_connect import LogServiceClientSync


class LogServiceProxy:
    """Forwards push_logs/fetch_logs to a remote LogService over RPC."""

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
