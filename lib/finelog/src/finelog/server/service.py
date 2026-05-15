# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LogService RPC implementation.

Owns a LogStore instance and exposes push (ingest) and fetch (query)
operations via Connect/RPC. Handlers are async so the durability wait in
``push_logs`` can park a coroutine instead of a threadpool worker; the
LogStore underneath is sync, so CPU/IO is dispatched to a worker thread
via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from finelog.rpc import logging_pb2
from finelog.store import LogStore
from finelog.store.duckdb_store import LOG_NAMESPACE_NAME

from .persistence_wait import DEFAULT_PERSIST_TIMEOUT_SEC, await_persisted

logger = logging.getLogger(__name__)


class LogServiceImpl:
    """Implements the finelog.logging.LogService Connect/RPC service.

    Owns the LogStore. Create with a log_dir to let it build the store,
    or pass an existing LogStore for testing.
    """

    def __init__(
        self,
        *,
        log_dir: Path | None = None,
        remote_log_dir: str = "",
        log_store: LogStore | None = None,
        persist_timeout_sec: float = DEFAULT_PERSIST_TIMEOUT_SEC,
    ) -> None:
        if log_store is not None:
            self._log_store = log_store
        elif log_dir is not None:
            self._log_store = LogStore(log_dir=log_dir, remote_log_dir=remote_log_dir)
        else:
            self._log_store = LogStore()
        self._persist_timeout_sec = persist_timeout_sec

    @property
    def log_store(self) -> LogStore:
        """The internal log store. Exposed for co-hosted components that need
        direct access (LogStoreHandler for controller process logs).
        """
        return self._log_store

    def close(self) -> None:
        """Close the underlying log store."""
        self._log_store.close()

    async def push_logs(
        self,
        request: logging_pb2.PushLogsRequest,
        ctx: Any,
    ) -> logging_pb2.PushLogsResponse:
        if not request.entries:
            return logging_pb2.PushLogsResponse()
        last_seq = await asyncio.to_thread(self._log_store.append, request.key, list(request.entries))
        await await_persisted(
            self._log_store,
            LOG_NAMESPACE_NAME,
            last_seq,
            timeout=self._persist_timeout_sec,
        )
        return logging_pb2.PushLogsResponse()

    async def fetch_logs(
        self,
        request: logging_pb2.FetchLogsRequest,
        ctx: Any,
    ) -> logging_pb2.FetchLogsResponse:
        # Wire-level UNSPECIFIED maps to REGEX so clients that encode the
        # query as a regex pattern in ``source`` without setting the field
        # keep working. New code sets EXACT or PREFIX explicitly.
        match_scope = request.match_scope
        if match_scope == logging_pb2.MATCH_SCOPE_UNSPECIFIED:
            match_scope = logging_pb2.MATCH_SCOPE_REGEX
        max_lines = request.max_lines if request.max_lines > 0 else 1000
        result = await asyncio.to_thread(
            self._log_store.get_logs,
            request.source,
            match_scope=match_scope,
            since_ms=request.since_ms,
            cursor=request.cursor,
            substring_filter=request.substring,
            max_lines=max_lines,
            tail=request.tail,
            min_level=request.min_level,
        )
        return logging_pb2.FetchLogsResponse(entries=result.entries, cursor=result.cursor)
