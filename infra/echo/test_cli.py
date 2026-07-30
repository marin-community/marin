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
        "references": [
            {
                "line": 42,
                "text": "raise FAILED_PRECONDITION",
                "url": "https://github.com/marin-community/marin/blob/abc1234/" "lib/iris/src/iris/scheduler.py#L42",
            }
        ],
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
    assert "get <domain:id>" in output[0]
    assert len(output) == 3
    assert "file:lib/iris/src/iris/scheduler.py" in output[2]
    assert "scheduler.py" in output[2]
    assert "L42 raise FAILED_PRECONDITION" in output[2]


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


def test_get_fetches_full_detail_by_search_result_id(monkeypatch, capsys):
    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return {
            "id": "file:lib/iris/OPS.md",
            "title": "Iris Operations",
            "subtitle": "lib/iris/OPS.md · main@abc123",
            "url": "https://github.com/marin-community/marin/blob/abc123/lib/iris/OPS.md",
            "text": "# Iris Operations\n\nDeploy with the restart command.",
        }

    monkeypatch.setattr(cli, "request", fake_request)
    args = cli.build_parser().parse_args(["get", "file:lib/iris/OPS.md"])
    args.func(args)

    assert calls == [("GET", "/repository-files/lib/iris/OPS.md", {})]
    output = capsys.readouterr().out
    assert "[file:lib/iris/OPS.md] Iris Operations" in output
    assert "# Iris Operations" in output
