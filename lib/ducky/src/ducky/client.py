# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Programmatic ``DuckyClient`` + the ``ducky query`` CLI.

ducky runs behind the Iris controller's endpoint proxy and speaks an async
protocol: ``POST /query {"sql": ...}`` returns a ``query_id``; poll
``GET /result/{query_id}`` until ``status != "running"`` (dodging the proxy's
~30 s request cap).

:class:`DuckyClient` is the reusable client for that protocol. It is transport-
and auth-agnostic via ``base_url`` + a ``token_provider``:

* CLI / open tunnel — ``base_url`` of the tunnel, no token (the tunnel auths).
* IAP proxy (e.g. dashboards) — :func:`iap_token_provider` mints a service-
  account OIDC token for the desktop OAuth audience, sent as ``Authorization``.
* in-cluster — the controller's internal proxy URL, no token.

Because ducky is **preemptible** (it can vanish and re-register at any time) and
reads object storage per query, both preemptions and transient network/DNS blips
surface as errors; :meth:`DuckyClient.run` retries those with exponential backoff
for up to ``retry_budget`` seconds per outage (any successful request resets the
budget, so healthy polling on a long query does not consume it). A transient poll
failure re-polls the same
``query_id`` (never resubmitting a query that may still be running); the query is
resubmitted only when ducky restarted and no longer knows the id, or the query
itself died to a transient error. Deterministic errors (SQL, missing file) and
query timeouts fail fast.

CLI::

    ducky query --cluster marin "SELECT count(*) FROM read_parquet('gs://…/*.parquet')"
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from collections.abc import Callable

import click
import httpx
from httpx import HTTPError

from ducky.tunnel import cluster_tunnel

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:10000/proxy/ducky"
# Per-request HTTP timeout. Each submit/poll call is fast (the query runs async on
# the server), so this only bounds a single round-trip — not the query.
_HTTP_TIMEOUT = 30
# Default overall wait for a query to finish (poll deadline). Kept above ducky's
# own ``query_timeout`` (600 s) so a genuinely slow query is interrupted server
# side (a clean error) rather than tripping this deadline first.
DEFAULT_QUERY_TIMEOUT = 900.0

TokenProvider = Callable[[], str | None]

# Substrings marking a *retryable* failure: transient network/DNS/object-store
# blips AND ducky being unavailable (preemptible — proxy "No endpoint", or a
# 502/504 upstream timeout while it restarts). Deterministic errors (SQL binder,
# file-not-found) are absent so they fail fast.
_RETRYABLE_MARKERS = (
    "could not resolve hostname",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection timed out",
    "timed out",
    "temporarily unavailable",
    "network is unreachable",
    "failed to establish",
    "no endpoint",  # controller proxy: ducky endpoint not registered (preempted)
    "upstream timeout",  # controller proxy: ducky didn't respond (restarting)
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
)

# ducky's query state is process-local: after a preemption+restart, polling a
# pre-restart query_id gets HTTP 404 "unknown query_id". Not in the markers
# above because it changes the retry *mode* — resubmit, don't re-poll.
_QUERY_LOST_MARKER = "unknown query_id"


def _is_retryable(message: str) -> bool:
    m = message.lower()
    return any(marker in m for marker in _RETRYABLE_MARKERS)


class DuckyError(RuntimeError):
    """A ducky query failed, timed out, or the service returned an HTTP error."""


class _QueryFailed(DuckyError):
    """The query reached a terminal ``error`` state server-side; re-polling cannot help."""


@dataclasses.dataclass(frozen=True)
class QueryResult:
    """A completed ducky query: capped preview rows plus full-result metadata."""

    columns: list[str]
    rows: list[list]
    total_rows: int
    truncated: bool
    result_path: str | None
    cached: bool
    elapsed_ms: int
    result_bytes: int

    def dicts(self) -> list[dict]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]

    def scalar(self):
        """The single cell of a 1x1 result (e.g. a ``count(*)``)."""
        if len(self.rows) != 1 or len(self.rows[0]) != 1:
            raise DuckyError(f"scalar() requires a 1x1 result, got {self.total_rows} rows x {len(self.columns)} columns")
        return self.rows[0][0]


def iap_token_provider(audience: str | None = None) -> TokenProvider:
    """Token provider that mints an IAP OIDC token from ambient service-account creds.

    The audience defaults to Marin's desktop OAuth client (the one on the IAP
    allowlist).
    """
    from rigging.auth import (  # noqa: PLC0415 — lazy: the CLI tunnel path needs no IAP/rigging deps
        MARIN_DESKTOP_OAUTH_CLIENT,
        IapServiceAccountTokenProvider,
    )

    return IapServiceAccountTokenProvider(audience or MARIN_DESKTOP_OAUTH_CLIENT.client_id).get_token


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except (ValueError, KeyError):
        return f"HTTP {response.status_code}: {response.text[:200]}"


class DuckyClient:
    """Submit SQL to ducky and block until the result is ready, riding out preemptions.

    Args:
        base_url: ducky root — a tunnel URL, the IAP proxy, or an in-cluster
            controller proxy.
        token_provider: returns a bearer token per request, or ``None`` for no
            auth (tunnel / in-cluster direct).
        poll_interval: seconds between ``/result`` polls.
        timeout: overall seconds to wait for one query to finish.
        retry_budget: wall-clock spent retrying *consecutive* retryable failures
            (one outage), with exponential backoff (``retry_base`` doubling,
            capped at ``retry_cap``). Any successful request resets the budget
            and backoff.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        token_provider: TokenProvider | None = None,
        poll_interval: float = 1.0,
        timeout: float = DEFAULT_QUERY_TIMEOUT,
        retry_budget: float = 150.0,
        retry_base: float = 2.0,
        retry_cap: float = 20.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._retry_budget = retry_budget
        self._retry_base = retry_base
        self._retry_cap = retry_cap
        # Successful-request counter; ``run()`` compares snapshots of it to tell
        # consecutive failures (one outage) from a fresh failure after progress.
        self._ok_requests = 0

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = _HTTP_TIMEOUT) -> dict:
        url = f"{self._base_url}{path}"
        kwargs: dict = {"timeout": timeout}
        if self._token_provider is not None:
            token = self._token_provider()
            if token:
                kwargs["headers"] = {"Authorization": f"Bearer {token}"}
        try:
            resp = httpx.post(url, json=body, **kwargs) if method == "POST" else httpx.get(url, **kwargs)
        except HTTPError as e:
            raise DuckyError(f"ducky {method} {path} unreachable: {e}") from e
        if resp.status_code not in (200, 202):
            raise DuckyError(f"ducky {method} {path} -> HTTP {resp.status_code}: {_error_message(resp)}")
        self._ok_requests += 1
        return resp.json()

    def healthy(self) -> bool:
        """Quick ``/health`` probe (short timeout, no retry) — is ducky reachable now?"""
        try:
            return self._request("GET", "/health", timeout=5.0).get("status") == "healthy"
        except DuckyError:
            return False

    def run(self, sql: str, use_cache: bool = True) -> QueryResult:
        """Submit ``sql`` and poll until done, retrying while ducky is unavailable.

        A transient failure while polling re-polls the same ``query_id`` — the
        query may still be running, and resubmitting would duplicate object-store
        reads (and could change results for non-deterministic SQL). The query is
        resubmitted only when ducky restarted and lost the id ("unknown
        query_id") or the query itself died to a transient error.

        ``retry_budget`` bounds one outage — consecutive retryable failures —
        not total query wall-clock: any successful request resets it, so a blip
        after minutes of healthy polling still gets the full budget.
        """
        attempt = 0
        query_id: str | None = None
        deadline = 0.0
        streak_start = 0.0
        ok_at_last_failure = -1
        while True:
            try:
                if query_id is None:
                    query_id = self._request("POST", "/query", {"sql": sql, "use_cache": use_cache})["query_id"]
                    deadline = time.monotonic() + self._timeout
                return self._poll(query_id, deadline, sql)
            except DuckyError as e:
                if isinstance(e, _QueryFailed):
                    # Terminal server-side error: re-polling returns it forever,
                    # so a retry must resubmit.
                    retryable, resubmit = _is_retryable(str(e)), True
                elif _QUERY_LOST_MARKER in str(e).lower():
                    retryable, resubmit = True, True
                else:
                    # Submit/poll transport blip; the submitted query (if any)
                    # may still be running, so keep its id and re-poll.
                    retryable, resubmit = _is_retryable(str(e)), False
                now = time.monotonic()
                if self._ok_requests != ok_at_last_failure:
                    # Progress since the last failure: this failure starts a new
                    # outage streak with a fresh budget and backoff.
                    streak_start = now
                    attempt = 0
                ok_at_last_failure = self._ok_requests
                elapsed = now - streak_start
                if not retryable or elapsed >= self._retry_budget:
                    # Re-raise with the original detail (HTTP status / "No
                    # endpoint" / "upstream timeout") so callers can see what
                    # actually failed.
                    raise
                if resubmit:
                    query_id = None
                wait = min(self._retry_cap, self._retry_base * (2**attempt))
                attempt += 1
                logger.warning(
                    "ducky unavailable/transient (retry %d, %.0fs/%.0fs elapsed), sleeping %.0fs: %s",
                    attempt,
                    elapsed,
                    self._retry_budget,
                    wait,
                    str(e).splitlines()[0][:140],
                )
                time.sleep(wait)

    def _poll(self, query_id: str, deadline: float, sql: str) -> QueryResult:
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
                    result_bytes=state.get("result_bytes", 0),
                )
            if status == "error":
                raise _QueryFailed(f"query failed: {state.get('error')}\nSQL: {sql[:500]}")
            if time.monotonic() > deadline:
                raise DuckyError(f"query {query_id} still running after {self._timeout:.0f}s")
            time.sleep(self._poll_interval)


def _render_table(columns: list[str], rows: list[list]) -> str:
    """Format columns/rows as a simple aligned text table."""
    str_rows = [["NULL" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(columns), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def _run_query(
    base: str,
    sql: str,
    output_format: str,
    poll_interval: float,
    timeout: int,
    use_cache: bool,
    retry_budget: float,
) -> None:
    client = DuckyClient(
        base, token_provider=None, poll_interval=poll_interval, timeout=timeout, retry_budget=retry_budget
    )
    try:
        result = client.run(sql, use_cache=use_cache)
    except DuckyError as e:
        raise click.ClickException(str(e)) from e

    if output_format == "json":
        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
        return

    click.echo(_render_table(result.columns, result.rows))
    count = f"{len(result.rows)} of {result.total_rows}" if result.truncated else str(result.total_rows)
    cached = "cached" if result.cached else "computed"
    click.echo(f"\n{count} rows · {result.elapsed_ms} ms · {result.result_bytes} B · {cached}", err=True)
    click.echo(f"full result: {result.result_path}", err=True)


@click.command("query")
@click.argument("sql", required=False)
@click.option("--cluster", default=None, help="Iris cluster to auto-tunnel to (hides the tunnel/proxy).")
@click.option("--endpoint", default="ducky", show_default=True, help="ducky endpoint name behind the controller proxy.")
@click.option(
    "--base-url",
    default=None,
    help=f"Explicit ducky base URL (default $DUCKY_BASE_URL or {DEFAULT_BASE_URL}); mutually exclusive with --cluster.",
)
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table", show_default=True)
@click.option("--poll-interval", default=1.0, show_default=True, help="Seconds between status polls.")
@click.option("--timeout", default=3600, show_default=True, help="Max seconds to wait for the query to finish.")
@click.option("--no-cache", is_flag=True, help="Force a fresh run instead of reusing a prior identical query's result.")
@click.option(
    "--retry-budget",
    default=0.0,
    show_default=True,
    help="Wall-clock seconds to retry transient failures (ducky preempted, network blips); 0 fails fast.",
)
def query(
    sql: str | None,
    cluster: str | None,
    endpoint: str,
    base_url: str | None,
    output_format: str,
    poll_interval: float,
    timeout: int,
    no_cache: bool,
    retry_budget: float,
) -> None:
    """Run SQL against a ducky service and print the result. SQL comes from the argument or stdin."""
    if cluster and base_url:
        raise click.UsageError("Pass --cluster or --base-url, not both.")
    if not sql:
        sql = click.get_text_stream("stdin").read()
    sql = sql.strip()
    if not sql:
        raise click.UsageError("No SQL provided — pass it as an argument or via stdin.")

    use_cache = not no_cache
    if cluster:
        with cluster_tunnel(cluster) as controller_url:
            base = f"{controller_url}/proxy/{endpoint}"
            _run_query(base, sql, output_format, poll_interval, timeout, use_cache, retry_budget)
    else:
        base = base_url or os.environ.get("DUCKY_BASE_URL", DEFAULT_BASE_URL)
        _run_query(base, sql, output_format, poll_interval, timeout, use_cache, retry_budget)


if __name__ == "__main__":
    query()
