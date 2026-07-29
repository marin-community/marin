# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Echo CLI federation."""

import cli


def test_search_requests_selected_remote_domains(monkeypatch, capsys):
    remote_result = {
        "id": "file:lib/iris/src/iris/scheduler.py",
        "domain": "file",
        "title": "scheduler.py",
        "subtitle": "lib/iris/src/iris/scheduler.py:42 · main@abc1234 · indexed 2026-07-29T20:00:00+00:00",
        "url": "https://github.com/marin-community/marin/blob/abc1234/lib/iris/src/iris/scheduler.py#L42",
        "snippet": "raise FAILED_PRECONDITION",
        "score": 0.04,
        "distance": 0.1,
        "lexical_score": 0.5,
    }

    def fake_request(method, path, *, params=None, body=None):
        assert (method, path) == ("GET", "/federated-search")
        assert params is not None
        assert params["domain"] == ["file", "pr"]
        return [remote_result]

    monkeypatch.setattr(cli, "request", fake_request)
    args = cli.build_parser().parse_args(["search", "FAILED_PRECONDITION", "--domain", "file", "--domain", "pr"])
    args.func(args)

    captured = capsys.readouterr()
    assert "[file] scheduler.py" in captured.out
    assert "main@abc1234" in captured.out
    assert "/blob/abc1234/" in captured.out
    assert captured.err == ""


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
