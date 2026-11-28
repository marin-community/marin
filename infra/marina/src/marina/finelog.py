# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A process-wide read client for the Marin Finelog hub, shared by apps.

Apps point it at a URL (or resolve ``finelog-marin`` through GCE), then run
bounded SQL reads and list namespaces. All reads are idempotent and retried
once if the client invalidates its transport underneath them.
"""

import logging
import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from connectrpc.errors import ConnectError
from finelog.client import LogClient
from finelog.types import is_retryable_error
from rigging.connect import IapAuth
from rigging.credentials import iap_provider_for

from marina.discovery import resolve_internal_ip

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 20_000
DEFAULT_MAX_ROWS = 20_000
DEFAULT_PROJECT = "hai-gcp-models"
DEFAULT_ZONE = "us-central1-a"
DEFAULT_FILTER = "name = finelog-marin"
DEFAULT_PORT = 10001
NAMESPACES_TTL_SECONDS = 60.0


class FinelogQueries:
    """SQL reads against one Finelog deployment, with GCE-backed resolution."""

    def __init__(
        self,
        *,
        url: str | None = None,
        iap_cluster: str | None = None,
        project: str = DEFAULT_PROJECT,
        zone: str = DEFAULT_ZONE,
        finelog_filter: str = DEFAULT_FILTER,
        port: int = DEFAULT_PORT,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self._url = url
        self._project = project
        self._zone = zone
        self._finelog_filter = finelog_filter
        self._port = port
        self._max_rows = max_rows
        self._lock = threading.Lock()
        self._source = url or finelog_filter
        self._namespaces: list[str] = []
        self._namespaces_at = 0.0
        interceptors = tuple(IapAuth(iap_provider_for(iap_cluster)).interceptors()) if iap_cluster else ()
        self._client = LogClient.connect(
            url or finelog_filter,
            resolver=self._resolve,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
        )

    @property
    def source(self) -> str:
        return self._source

    def _resolve(self, _endpoint: str) -> str:
        if self._url:
            address = self._url
        else:
            ip = resolve_internal_ip(self._project, self._zone, self._finelog_filter, timeout=5.0)
            address = f"http://{ip}:{self._port}"
        with self._lock:
            self._source = address if self._url else f"{self._finelog_filter} {address}"
        return address

    def _call[Result](self, operation: Callable[[LogClient], Result]) -> Result:
        """Retry one idempotent read after LogClient invalidates its transport."""
        for attempt in range(2):
            try:
                return operation(self._client)
            except Exception as error:
                cause = error.__cause__
                retryable = (
                    isinstance(error, (ConnectionError, OSError, TimeoutError))
                    or (isinstance(error, ConnectError) and is_retryable_error(error))
                    or (isinstance(cause, ConnectError) and is_retryable_error(cause))
                )
                if attempt == 1 or not retryable:
                    raise
        raise AssertionError("unreachable")

    def query(self, sql: str) -> list[dict[str, Any]]:
        table = self._call(lambda client: client.query(sql, max_rows=self._max_rows))
        return [json_row(row) for row in table.to_pylist()]

    def namespaces(self) -> list[str]:
        with self._lock:
            if time.monotonic() - self._namespaces_at < NAMESPACES_TTL_SECONDS:
                return list(self._namespaces)
        infos = self._call(lambda client: client.list_namespaces())
        names = sorted(info.namespace for info in infos)
        with self._lock:
            self._namespaces = names
            self._namespaces_at = time.monotonic()
        return list(names)


def json_row(row: dict[str, Any]) -> dict[str, Any]:
    """Make one Arrow row JSON-safe: timestamps become epoch ms, nonfinites null."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            stamp = value if value.tzinfo else value.replace(tzinfo=UTC)
            out[key] = int(stamp.timestamp() * 1000)
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = None
        else:
            out[key] = value
    return out
