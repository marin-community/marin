# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LogService RPC implementation.

Owns a LogStore instance and exposes push (ingest) and fetch (query)
operations via Connect/RPC. The LogStore is an internal implementation
detail — all external consumers access logs through the RPC interface
(or, for in-process callers, via push_logs/fetch_logs methods directly).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from iris.cluster.log_store import LogStore
from iris.rpc import logging_pb2

logger = logging.getLogger(__name__)


class LogServiceImpl:
    """Implements the iris.logging.LogService Connect/RPC service.

    Owns the LogStore. Create with a log_dir to let it build the store,
    or pass an existing LogStore for testing.
    """

    def __init__(
        self,
        *,
        log_dir: Path | None = None,
        remote_log_dir: str = "",
        log_store: LogStore | None = None,
    ) -> None:
        if log_store is not None:
            self._log_store = log_store
        elif log_dir is not None:
            self._log_store = LogStore(log_dir=log_dir, remote_log_dir=remote_log_dir)
        else:
            self._log_store = LogStore()

    @property
    def log_store(self) -> LogStore:
        """The internal log store. Exposed for co-hosted components that need
        direct access (LogStoreHandler for controller process logs).
        """
        return self._log_store

    def close(self) -> None:
        """Close the underlying log store."""
        self._log_store.close()

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
