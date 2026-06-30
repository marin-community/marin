# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
from click.testing import CliRunner
from ducky import client
from ducky.client import _render_table, query

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


class _FakeHttp:
    """Routes ducky's POST /query + GET /result to canned responses."""

    def __init__(self, result: dict, submit_status: int = 202, get_status: int = 200):
        self._result = result
        self._submit_status = submit_status
        self._get_status = get_status

    def post(self, url, json=None, timeout=None):
        return httpx.Response(self._submit_status, json={"query_id": "abc"}, request=httpx.Request("POST", url))

    def get(self, url, timeout=None):
        return httpx.Response(self._get_status, json=self._result, request=httpx.Request("GET", url))


def test_query_prints_table_and_stats(monkeypatch):
    monkeypatch.setattr(client, "httpx", _FakeHttp(_DONE))
    result = CliRunner().invoke(query, ["SELECT 1", "--base-url", _BASE])

    assert result.exit_code == 0, result.output
    assert "a | b" in result.stdout
    assert "NULL" in result.stdout
    assert "2 rows · 42 ms · 99 B · computed" in result.stderr
    assert "gs://b/ducky/q.parquet" in result.stderr


def test_query_json_format(monkeypatch):
    monkeypatch.setattr(client, "httpx", _FakeHttp(_DONE))
    result = CliRunner().invoke(query, ["SELECT 1", "--base-url", _BASE, "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["columns"] == ["a", "b"]


def test_query_error_exits_nonzero(monkeypatch):
    err = {"status": "error", "error": "Catalog Error: nope"}
    monkeypatch.setattr(client, "httpx", _FakeHttp(err))
    result = CliRunner().invoke(query, ["SELECT * FROM nope", "--base-url", _BASE])
    assert result.exit_code != 0
    assert "Catalog Error: nope" in result.output


def test_query_poll_http_error_exits_cleanly(monkeypatch):
    # a non-200 /result (e.g. server restart, proxy failure) must surface as a CLI error,
    # not a KeyError on the missing "status" field
    monkeypatch.setattr(client, "httpx", _FakeHttp({"error": "upstream timeout"}, get_status=504))
    result = CliRunner().invoke(query, ["SELECT 1", "--base-url", _BASE])
    assert result.exit_code != 0
    assert "upstream timeout" in result.output


def test_query_requires_sql():
    result = CliRunner().invoke(query, ["--base-url", _BASE], input="")
    assert result.exit_code != 0
    assert "No SQL provided" in result.output


def test_query_cluster_and_base_url_conflict():
    result = CliRunner().invoke(query, ["SELECT 1", "--cluster", "marin", "--base-url", _BASE])
    assert result.exit_code != 0
    assert "not both" in result.output
