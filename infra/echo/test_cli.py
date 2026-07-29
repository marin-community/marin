# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Echo CLI federation."""

import subprocess

import cli
import local_search


def test_search_merges_remote_results_while_local_index_warms(monkeypatch, capsys):
    remote_result = {
        "id": "wiki:7",
        "domain": "wiki",
        "title": "Collective diagnosis",
        "subtitle": "when a collective stalls",
        "url": "https://echo.oa.dev/wiki/7",
        "snippet": "Inspect the topology.",
        "score": 0.04,
        "distance": 0.1,
        "lexical_score": 0.5,
    }

    def fake_request(method, path, *, params=None, body=None):
        assert (method, path) == ("GET", "/federated-search")
        assert params is not None
        assert params["domain"] == ["wiki"]
        return [remote_result]

    monkeypatch.setattr(cli, "request", fake_request)
    monkeypatch.setattr(
        cli.local_search,
        "search",
        lambda query, limit: ([], "local file search is starting; its first index builds in the background"),
    )
    args = cli.build_parser().parse_args(["search", "collective diagnosis", "--domain", "wiki", "--domain", "file"])
    args.func(args)

    captured = capsys.readouterr()
    assert "[wiki] Collective diagnosis" in captured.out
    assert "https://echo.oa.dev/wiki/7" in captured.out
    assert "first index builds in the background" in captured.err


def test_search_legacy_activity_filters_keep_existing_endpoint(monkeypatch, capsys):
    calls = []

    def fake_request(method, path, *, params=None, body=None):
        calls.append((method, path, params))
        return []

    monkeypatch.setattr(cli, "request", fake_request)
    args = cli.build_parser().parse_args(["search", "scheduler", "--source", "discord"])
    args.func(args)

    assert calls[0][1] == "/search"
    assert calls[0][2]["source"] == "discord"
    assert capsys.readouterr().err == ""


def test_cold_local_search_starts_daemon_without_waiting(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    started = []

    def unavailable_daemon(root, query, limit):
        raise FileNotFoundError

    monkeypatch.setattr(local_search, "daemon_request", unavailable_daemon)
    monkeypatch.setattr(local_search, "start_daemon", started.append)
    results, message = local_search.search("scheduler", 10, root=tmp_path)

    assert results == []
    assert message == "local file search is starting; its first index builds in the background"
    assert started == [tmp_path.resolve()]
