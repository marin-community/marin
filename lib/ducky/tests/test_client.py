# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from click.testing import CliRunner
from ducky import client
from ducky.client import DuckyClient, DuckyError, QueryResult, _render_table, query

_BASE = "http://ducky.test/proxy/ducky"
_DONE = {
    "status": "done",
    "columns": ["a", "b"],
    "rows": [[1, "x"], [2, None]],
    "total_rows": 2,
    "truncated": False,
    "result_path": "gs://b/ducky/q.parquet",
    "cached": False,
    "elapsed_ms": 42,
    "result_bytes": 99,
}


def test_render_table_aligns_and_marks_null():
    out = _render_table(["a", "bb"], [[1, "x"], [22, None]])
    lines = out.splitlines()
    assert len(lines) == 4  # header, separator, 2 data rows
    assert lines[0].startswith("a") and "bb" in lines[0]
    assert set(lines[1]) <= {"-", "+"}  # separator row
    assert "NULL" in out
    # the "a" column is padded to width 2 ("22"), so the separator before '|' aligns
    assert lines[0].index("|") == lines[2].index("|")


class _ScriptedHttp:
    """Scripted ducky transport: pops one canned (status, payload) per request."""

    def __init__(self, posts: list[tuple[int, dict]], gets: list[tuple[int, dict]]):
        self._posts = list(posts)
        self._gets = list(gets)
        self.post_count = 0
        self.polled_query_ids: list[str] = []

    def post(self, url, json=None, timeout=None):
        self.post_count += 1
        status, payload = self._posts.pop(0)
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))

    def get(self, url, timeout=None):
        self.polled_query_ids.append(url.rsplit("/", 1)[-1])
        status, payload = self._gets.pop(0)
        return httpx.Response(status, json=payload, request=httpx.Request("GET", url))


def _client(**kwargs) -> DuckyClient:
    return DuckyClient(_BASE, **{"poll_interval": 0, "retry_base": 0.01, "retry_cap": 0.01, **kwargs})


def test_query_prints_table_and_stats(monkeypatch):
    monkeypatch.setattr(client, "httpx", _ScriptedHttp(posts=[(202, {"query_id": "abc"})], gets=[(200, _DONE)]))
    result = CliRunner().invoke(query, ["SELECT 1", "--base-url", _BASE])

    assert result.exit_code == 0, result.output
    assert "a | b" in result.stdout
    assert "NULL" in result.stdout
    assert "2 rows · 42 ms · 99 B · computed" in result.stderr
    assert "gs://b/ducky/q.parquet" in result.stderr


def test_query_json_format(monkeypatch):
    monkeypatch.setattr(client, "httpx", _ScriptedHttp(posts=[(202, {"query_id": "abc"})], gets=[(200, _DONE)]))
    result = CliRunner().invoke(query, ["SELECT 1", "--base-url", _BASE, "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["columns"] == ["a", "b"]


def test_query_error_exits_nonzero(monkeypatch):
    err = {"status": "error", "error": "Catalog Error: nope"}
    monkeypatch.setattr(client, "httpx", _ScriptedHttp(posts=[(202, {"query_id": "abc"})], gets=[(200, err)]))
    result = CliRunner().invoke(query, ["SELECT * FROM nope", "--base-url", _BASE])
    assert result.exit_code != 0
    assert "Catalog Error: nope" in result.output


def test_query_poll_http_error_exits_cleanly(monkeypatch):
    # a non-200 /result (e.g. server restart, proxy failure) must surface as a CLI error,
    # not a KeyError on the missing "status" field; the CLI default budget of 0 fails fast
    http = _ScriptedHttp(posts=[(202, {"query_id": "abc"})], gets=[(504, {"error": "upstream timeout"})])
    monkeypatch.setattr(client, "httpx", http)
    result = CliRunner().invoke(query, ["SELECT 1", "--base-url", _BASE])
    assert result.exit_code != 0
    assert "upstream timeout" in result.output


def test_run_repolls_same_query_after_transient_poll_blip(monkeypatch):
    # A 503 on /result must not resubmit: the query may still be running, and a
    # duplicate submission re-reads object storage (and can change results for
    # non-deterministic SQL).
    http = _ScriptedHttp(
        posts=[(202, {"query_id": "q1"})],
        gets=[(503, {"error": "service unavailable"}), (200, _DONE)],
    )
    monkeypatch.setattr(client, "httpx", http)
    result = _client(retry_budget=5).run("SELECT 1")
    assert result.total_rows == 2
    assert http.post_count == 1
    assert http.polled_query_ids == ["q1", "q1"]


def test_run_resubmits_when_restarted_ducky_forgot_the_query(monkeypatch):
    # ducky's query state is process-local: after a preemption+restart, polling
    # the pre-restart id 404s with "unknown query_id" — the client must resubmit.
    http = _ScriptedHttp(
        posts=[(202, {"query_id": "q1"}), (202, {"query_id": "q2"})],
        gets=[(404, {"error": "unknown query_id"}), (200, _DONE)],
    )
    monkeypatch.setattr(client, "httpx", http)
    result = _client(retry_budget=5).run("SELECT 1")
    assert result.total_rows == 2
    assert http.polled_query_ids == ["q1", "q2"]


def test_run_resubmits_query_that_died_to_transient_error(monkeypatch):
    # A terminal "error" state with a transient marker (object-store blip) can
    # only be retried by resubmitting — re-polling returns the error forever.
    http = _ScriptedHttp(
        posts=[(202, {"query_id": "q1"}), (202, {"query_id": "q2"})],
        gets=[
            (200, {"status": "error", "error": "Connection reset by peer reading gs://b/x.parquet"}),
            (200, _DONE),
        ],
    )
    monkeypatch.setattr(client, "httpx", http)
    result = _client(retry_budget=5).run("SELECT 1")
    assert result.total_rows == 2
    assert http.post_count == 2


class _FakeClock:
    """Replaces the ``time`` module in the client: ``sleep`` advances ``monotonic``."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_healthy_polling_does_not_consume_retry_budget(monkeypatch):
    # retry_budget bounds one outage, not total query wall-clock: a blip after
    # minutes of healthy polling on a long query must still get retried.
    clock = _FakeClock()
    monkeypatch.setattr(client, "time", clock)
    running = (200, {"status": "running"})
    http = _ScriptedHttp(
        posts=[(202, {"query_id": "q1"})],
        gets=[running] * 60 + [(503, {"error": "service unavailable"}), (200, _DONE)],
    )
    monkeypatch.setattr(client, "httpx", http)
    result = _client(poll_interval=1.0, retry_budget=10).run("SELECT 1")  # blip lands at t=60 > budget
    assert result.total_rows == 2
    assert http.post_count == 1


def test_run_fails_fast_on_deterministic_query_error(monkeypatch):
    http = _ScriptedHttp(
        posts=[(202, {"query_id": "q1"})],
        gets=[(200, {"status": "error", "error": "Catalog Error: nope"})],
    )
    monkeypatch.setattr(client, "httpx", http)
    with pytest.raises(DuckyError, match="Catalog Error"):
        _client(retry_budget=5).run("SELECT * FROM nope")
    assert http.post_count == 1


def _result(columns: list[str], rows: list[list]) -> QueryResult:
    return QueryResult(
        columns=columns,
        rows=rows,
        total_rows=len(rows),
        truncated=False,
        result_path=None,
        cached=False,
        elapsed_ms=0,
        result_bytes=0,
    )


def test_scalar_requires_1x1_result():
    assert _result(["n"], [[7]]).scalar() == 7
    with pytest.raises(DuckyError, match="1x1"):
        _result(["n"], [[1], [2]]).scalar()
    with pytest.raises(DuckyError, match="1x1"):
        _result(["a", "b"], [[1, 2]]).scalar()


def test_query_requires_sql():
    result = CliRunner().invoke(query, ["--base-url", _BASE], input="")
    assert result.exit_code != 0
    assert "No SQL provided" in result.output


def test_query_cluster_and_base_url_conflict():
    result = CliRunner().invoke(query, ["SELECT 1", "--cluster", "marin", "--base-url", _BASE])
    assert result.exit_code != 0
    assert "not both" in result.output
