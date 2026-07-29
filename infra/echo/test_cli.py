# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Echo CLI federation."""

import cli


def test_search_sends_selected_domains_to_federated_endpoint(monkeypatch, capsys):
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

    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return [remote_result]

    monkeypatch.setattr(cli, "request", fake_request)
    args = cli.build_parser().parse_args(["search", "FAILED_PRECONDITION", "--domain", "file", "--domain", "pr"])
    args.func(args)

    assert calls == [
        (
            "GET",
            "/federated-search",
            {"params": {"q": "FAILED_PRECONDITION", "domain": ["file", "pr"], "limit": 10}},
        )
    ]
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 2
    assert "file:lib/iris/src/iris/scheduler.py" in output[0]
    assert "raise FAILED_PRECONDITION" in output[0]
    assert remote_result["url"] in output[1]


def test_search_defaults_to_curated_domains_without_discord(monkeypatch):
    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return []

    monkeypatch.setattr(cli, "request", fake_request)
    args = cli.build_parser().parse_args(["search", "scheduler"])
    args.func(args)

    assert calls == [
        (
            "GET",
            "/federated-search",
            {"params": {"q": "scheduler", "domain": ["wiki", "file", "pr", "issue"], "limit": 10}},
        )
    ]
