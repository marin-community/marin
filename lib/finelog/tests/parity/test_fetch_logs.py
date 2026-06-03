# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-3 FetchLogs RPC parity tests.

Drive ``LogService.PushLogs`` (write) + ``LogService.FetchLogs`` (read) over real
HTTP/RPC against both the Python and Rust servers via the client fixtures,
asserting on the decoded ``LogEntry`` fields and ``response.cursor`` — never on
log strings. These never import store internals; the seam is the RPC socket.

Re-expresses the behavioral cases from ``tests/test_duckdb_store.py``: append+read
order, tail, cursor exclusivity, REGEX/PREFIX scopes, EXACT/PREFIX literal-key
metachar safety (#5392), empty-prefix rejection, and literal substring filtering.
"""

from __future__ import annotations

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync

from tests.parity.conftest import Backend

pytestmark = pytest.mark.timeout(60)

_REGEX_METACHARS = list(".*+?[](){}^$|\\")


# ---------------------------------------------------------------------------
# Wire helpers.
# ---------------------------------------------------------------------------


def _log_client(url: str) -> LogServiceClientSync:
    return LogServiceClientSync(address=url)


def _entry(data: str, source: str = "stdout", epoch_ms: int = 0) -> logging_pb2.LogEntry:
    return logging_pb2.LogEntry(
        source=source,
        data=data,
        timestamp=logging_pb2.Timestamp(epoch_ms=epoch_ms),
        level=logging_pb2.LOG_LEVEL_INFO,
    )


def _push(client: LogServiceClientSync, key: str, entries: list[logging_pb2.LogEntry]) -> None:
    client.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))


def _fetch(
    client: LogServiceClientSync,
    source: str,
    *,
    match_scope: int = logging_pb2.MATCH_SCOPE_EXACT,
    cursor: int = 0,
    max_lines: int = 0,
    tail: bool = False,
    substring: str = "",
    since_ms: int = 0,
    min_level: str = "",
) -> logging_pb2.FetchLogsResponse:
    return client.fetch_logs(
        logging_pb2.FetchLogsRequest(
            source=source,
            match_scope=match_scope,
            cursor=cursor,
            max_lines=max_lines,
            tail=tail,
            substring=substring,
            since_ms=since_ms,
            min_level=min_level,
        )
    )


KEY = "/job/test/0:0"


# ---------------------------------------------------------------------------
# Round-trip + ordering.
# ---------------------------------------------------------------------------


def test_fetch_logs_roundtrip_and_order(finelog_url: str, server_backend: Backend) -> None:
    client = _log_client(finelog_url)
    _push(client, KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(10)])
    resp = _fetch(client, KEY)
    assert [e.data for e in resp.entries] == [f"line-{i}" for i in range(10)]
    # EXACT scope: attempt_id derived from the literal key (":0" -> 0).
    assert all(e.attempt_id == 0 for e in resp.entries)


def test_fetch_logs_tail(finelog_url: str, server_backend: Backend) -> None:
    client = _log_client(finelog_url)
    _push(client, KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(50)])
    resp = _fetch(client, KEY, max_lines=5, tail=True)
    # tail returns the last 5, in ascending order after the reverse.
    assert [e.data for e in resp.entries] == [f"line-{i}" for i in range(45, 50)]


def test_fetch_logs_cursor_exclusive(finelog_url: str, server_backend: Backend) -> None:
    client = _log_client(finelog_url)
    _push(client, KEY, [_entry(f"a-{i}", epoch_ms=i) for i in range(5)])
    first = _fetch(client, KEY)
    assert len(first.entries) == 5
    assert first.cursor > 0
    # Append more, then resume after the first cursor: only the newer rows.
    _push(client, KEY, [_entry(f"b-{i}", epoch_ms=10 + i) for i in range(3)])
    second = _fetch(client, KEY, cursor=first.cursor)
    assert [e.data for e in second.entries] == ["b-0", "b-1", "b-2"]
    # A re-fetch at the second cursor (no new rows) returns nothing and holds
    # the cursor where it was (default_cursor).
    third = _fetch(client, KEY, cursor=second.cursor)
    assert third.entries == []
    assert third.cursor == second.cursor


def test_fetch_logs_out_of_range_level_round_trips(finelog_url: str, server_backend: Backend) -> None:
    """An out-of-range open-enum ``level`` round-trips verbatim.

    ``LogLevel`` is an OPEN enum, so a client may store any int. Both servers
    must echo the stored value rather than collapsing an unknown one to
    ``UNKNOWN(0)``. ``99`` is not a defined ``LogLevel`` variant.
    """
    client = _log_client(finelog_url)
    entry = logging_pb2.LogEntry(
        source="stdout",
        data="x",
        timestamp=logging_pb2.Timestamp(epoch_ms=0),
        level=99,
    )
    _push(client, KEY, [entry])
    resp = _fetch(client, KEY)
    assert len(resp.entries) == 1
    assert int(resp.entries[0].level) == 99


# ---------------------------------------------------------------------------
# Scopes: REGEX / PREFIX.
# ---------------------------------------------------------------------------


def test_fetch_logs_regex_scope(finelog_url: str, server_backend: Backend) -> None:
    client = _log_client(finelog_url)
    _push(client, "/job/test/0:0", [_entry("a", epoch_ms=1)])
    _push(client, "/job/test/1:0", [_entry("b", epoch_ms=2)])
    _push(client, "/job/other/0:0", [_entry("c", epoch_ms=3)])
    resp = _fetch(client, "/job/test/.*", match_scope=logging_pb2.MATCH_SCOPE_REGEX)
    assert sorted(e.data for e in resp.entries) == ["a", "b"]
    # REGEX scope populates key + per-row attempt_id (both ":0" here -> 0).
    assert all(e.attempt_id == 0 for e in resp.entries)
    assert {e.key for e in resp.entries} == {"/job/test/0:0", "/job/test/1:0"}


def test_fetch_logs_prefix_scope_with_delimiter(finelog_url: str, server_backend: Backend) -> None:
    client = _log_client(finelog_url)
    _push(client, "/job/test/0:0", [_entry("a", epoch_ms=1)])
    _push(client, "/job/test/0:1", [_entry("b", epoch_ms=2)])
    _push(client, "/job/other/0:0", [_entry("c", epoch_ms=3)])
    # PREFIX with the ":" delimiter matches the task's attempts, not siblings.
    resp = _fetch(client, "/job/test/0:", match_scope=logging_pb2.MATCH_SCOPE_PREFIX)
    assert sorted(e.data for e in resp.entries) == ["a", "b"]
    # Per-row attempt_id from each matched key.
    assert sorted(e.attempt_id for e in resp.entries) == [0, 1]


def test_fetch_logs_default_unspecified_is_regex(finelog_url: str, server_backend: Backend) -> None:
    # An unset match_scope (UNSPECIFIED) maps to REGEX server-side.
    client = _log_client(finelog_url)
    _push(client, "/job/test/0:0", [_entry("a", epoch_ms=1)])
    _push(client, "/job/test/1:0", [_entry("b", epoch_ms=2)])
    resp = client.fetch_logs(logging_pb2.FetchLogsRequest(source="/job/test/.*"))
    assert sorted(e.data for e in resp.entries) == ["a", "b"]


# ---------------------------------------------------------------------------
# Literal-key metachar safety (#5392): EXACT and PREFIX never reinterpret regex.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("meta", _REGEX_METACHARS)
def test_fetch_logs_exact_literal_metachars(finelog_url: str, server_backend: Backend, meta: str) -> None:
    client = _log_client(finelog_url)
    key = f"/job/literal{meta}value:0"
    _push(client, key, [_entry(f"line-with-{meta}", epoch_ms=1)])
    _push(client, "/job/literal-other:0", [_entry("decoy", epoch_ms=2)])
    resp = _fetch(client, key, match_scope=logging_pb2.MATCH_SCOPE_EXACT)
    assert [e.data for e in resp.entries] == [f"line-with-{meta}"]


@pytest.mark.parametrize("meta", _REGEX_METACHARS)
def test_fetch_logs_prefix_literal_metachars(finelog_url: str, server_backend: Backend, meta: str) -> None:
    client = _log_client(finelog_url)
    key = f"/job/literal{meta}value:0"
    _push(client, key, [_entry(f"prefix-hit-{meta}", epoch_ms=1)])
    _push(client, "/job/different-prefix:0", [_entry("decoy", epoch_ms=2)])
    prefix = f"/job/literal{meta}"
    resp = _fetch(client, prefix, match_scope=logging_pb2.MATCH_SCOPE_PREFIX)
    assert [e.data for e in resp.entries] == [f"prefix-hit-{meta}"]


def test_fetch_logs_scientific_notation_metachar(finelog_url: str, server_backend: Backend) -> None:
    # Regression for #5392: `9e+20` under EXACT — `+` must match itself.
    client = _log_client(finelog_url)
    _push(client, "/job/curation-9e+20", [_entry("hit", epoch_ms=1)])
    _push(client, "/job/curation-other", [_entry("miss", epoch_ms=2)])
    resp = _fetch(client, "/job/curation-9e+20", match_scope=logging_pb2.MATCH_SCOPE_EXACT)
    assert [e.data for e in resp.entries] == ["hit"]


# ---------------------------------------------------------------------------
# Empty PREFIX rejected; substring literal wildcards.
# ---------------------------------------------------------------------------


def test_fetch_logs_empty_prefix_rejected(finelog_url: str, server_backend: Backend) -> None:
    # The load-bearing contract: an empty PREFIX source must be REJECTED (it
    # would otherwise page every key in the store), not silently served.
    #
    # DIALECT GAP: the error CODE differs. The Python `fetch_logs` handler does
    # not catch the `ValueError` `_scope_query` raises, so it surfaces as
    # UNKNOWN; the Rust server maps the empty-prefix rejection to the (more
    # correct) INVALID_ARGUMENT. Both reject — that is the contract — but we
    # pin each backend's actual code rather than change the frozen Python
    # behavior. Re-unify to INVALID_ARGUMENT if Python's handler ever catches it.
    client = _log_client(finelog_url)
    _push(client, "/job/a:0", [_entry("a", epoch_ms=1)])
    with pytest.raises(ConnectError) as exc:
        _fetch(client, "", match_scope=logging_pb2.MATCH_SCOPE_PREFIX)
    expected = Code.INVALID_ARGUMENT if server_backend.name == "rust" else Code.UNKNOWN
    assert exc.value.code == expected


def test_fetch_logs_substring_literal_wildcards(finelog_url: str, server_backend: Backend) -> None:
    client = _log_client(finelog_url)
    _push(
        client,
        KEY,
        [
            _entry("100% done", epoch_ms=1),
            _entry("a_b_c", epoch_ms=2),
            _entry("plain", epoch_ms=3),
        ],
    )
    # `%` and `_` are literal (contains), not LIKE wildcards.
    pct = _fetch(client, KEY, substring="100%")
    assert [e.data for e in pct.entries] == ["100% done"]
    us = _fetch(client, KEY, substring="a_b")
    assert [e.data for e in us.entries] == ["a_b_c"]
