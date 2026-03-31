# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LogService RPC implementation.

Thin wrapper around the existing LogStore interface, exposing push (ingest)
and fetch (query) operations via Connect/RPC. FetchLogs uses the same
request/response messages as ControllerService.FetchLogs for client compat.
"""

from __future__ import annotations

import logging
from typing import Any

from iris.cluster.log_store import LogStore
from iris.rpc import logging_pb2

logger = logging.getLogger(__name__)


class LogServiceImpl:
    """Implements the iris.logging.LogService Connect/RPC service."""

    def __init__(self, log_store: LogStore) -> None:
        self._log_store = log_store

    def push_logs(
        self,
        request: logging_pb2.PushLogsRequest,
        ctx: Any,
    ) -> logging_pb2.PushLogsResponse:
        if request.entries:
            self._log_store.append(request.key, list(request.entries))
        return logging_pb2.PushLogsResponse()

    def fetch_logs(
        self,
        request: logging_pb2.FetchLogsRequest,
        ctx: Any,
    ) -> logging_pb2.FetchLogsResponse:
        max_lines = request.max_lines if request.max_lines > 0 else 1000
        result = self._log_store.get_logs(
            request.source,
            since_ms=request.since_ms,
            cursor=request.cursor,
            substring_filter=request.substring,
            max_lines=max_lines,
            tail=request.tail,
            min_level=request.min_level,
        )
        return logging_pb2.FetchLogsResponse(entries=result.entries, cursor=result.cursor)
