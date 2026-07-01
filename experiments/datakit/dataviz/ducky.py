# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client for the ducky ad-hoc DuckDB SQL service (``lib/ducky``).

The dataviz dashboard fetches *all* its data through ducky: it never reads the
datakit parquet directly, it issues SQL and gets columnar results back. ducky
runs behind the Iris controller's IAP-gated proxy at
``https://iris.oa.dev/proxy/ducky/`` and speaks an async protocol:
``POST /query {"sql": ...}`` returns a ``query_id``; the caller polls
``GET /result/{query_id}`` until ``status != "running"`` (this dodges the
proxy's ~30 s request cap).

Auth is the tricky part: the endpoint is IAP-gated and the accepted bearer
audience is the *desktop* OAuth client (:data:`rigging.auth.MARIN_DESKTOP_OAUTH_CLIENT`),
not the browser-redirect client id. We reuse rigging's
:class:`~rigging.auth.IapServiceAccountTokenProvider` to mint that token from
ambient service-account credentials, and send it as ``Authorization: Bearer``.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT, IapServiceAccountTokenProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://iris.oa.dev/proxy/ducky"

TokenProvider = Callable[[], str | None]


class DuckyError(RuntimeError):
    """A ducky query failed, timed out, or the service returned an HTTP error."""


@dataclass(frozen=True)
class QueryResult:
    """A completed ducky query: capped preview rows plus the full-result metadata."""

    columns: list[str]
    rows: list[list]
    total_rows: int
    truncated: bool
    result_path: str | None
    cached: bool
    elapsed_ms: int

    def dicts(self) -> list[dict]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]

    def scalar(self):
        """The single cell of a 1x1 result (e.g. a ``count(*)``)."""
        if not self.rows or not self.rows[0]:
            raise DuckyError("scalar() on an empty result")
        return self.rows[0][0]


def iap_token_provider(audience: str = MARIN_DESKTOP_OAUTH_CLIENT.client_id) -> TokenProvider:
    """Default provider: an IAP OIDC token from ambient service-account creds."""
    return IapServiceAccountTokenProvider(audience).get_token


class DuckyClient:
    """Submit SQL to ducky and block until the result is ready.

    Args:
        base_url: ducky root, e.g. the IAP proxy path (default) or an in-cluster
            endpoint address resolved from the Iris registry.
        token_provider: returns a bearer token per request, or ``None`` for no
            auth (in-cluster direct endpoint). Defaults to the IAP SA token.
        poll_interval: seconds between ``/result`` polls.
        timeout: overall seconds to wait for a query to finish.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        token_provider: TokenProvider | None = iap_token_provider(),
        poll_interval: float = 1.0,
        timeout: float = 180.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._poll_interval = poll_interval
        self._timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self._base_url}{path}", data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self._token_provider is not None:
            token = self._token_provider()
            if token:
                req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise DuckyError(f"ducky {method} {path} -> HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise DuckyError(f"ducky {method} {path} unreachable: {e.reason}") from e

    def run(self, sql: str) -> QueryResult:
        """Submit ``sql`` and poll until done. Raises :class:`DuckyError` on failure/timeout."""
        query_id = self._request("POST", "/query", {"sql": sql})["query_id"]
        deadline = time.monotonic() + self._timeout
        while True:
            state = self._request("GET", f"/result/{query_id}")
            status = state.get("status")
            if status == "done":
                return QueryResult(
                    columns=state["columns"],
                    rows=state["rows"],
                    total_rows=state["total_rows"],
                    truncated=state.get("truncated", False),
                    result_path=state.get("result_path"),
                    cached=state.get("cached", False),
                    elapsed_ms=state.get("elapsed_ms", 0),
                )
            if status == "error":
                raise DuckyError(f"query failed: {state.get('error')}\nSQL: {sql[:500]}")
            if time.monotonic() > deadline:
                raise DuckyError(f"query {query_id} still running after {self._timeout:.0f}s")
            time.sleep(self._poll_interval)
