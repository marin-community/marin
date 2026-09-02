# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Echo CLI federation."""

import io
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cli
import pytest
import requests


def json_response(value, headers=None):
    response = requests.Response()
    response.status_code = 200
    response.headers.update(headers or {})
    response._content = cli.json.dumps(value).encode()
    return response


def test_search_sends_selected_domains_to_federated_endpoint(monkeypatch, capsys):
    reference_text = (
        'raise SchedulerError("FAILED_PRECONDITION: pending queue cannot satisfy the requested TPU topology '
        'and priority band")'
    )
    remote_result = {
        "key": "file:731",
        "id": "file:marin-community/marin@main:lib/iris/src/iris/scheduler.py",
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
                "text": reference_text,
                "url": "https://github.com/marin-community/marin/blob/abc1234/" "lib/iris/src/iris/scheduler.py#L42",
            }
        ],
    }

    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return json_response([remote_result], {"X-Echo-Search-Execution-ID": "991"})

    monkeypatch.setattr(cli, "request_response", fake_request)
    clock = iter((10.0, 11.234))
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clock))
    args = cli.build_parser().parse_args(
        [
            "search",
            "FAILED_PRECONDITION",
            "--domain",
            "file",
            "--domain",
            "pr",
            "--repository",
            "marin-community/marin",
        ]
    )
    args.func(args)

    assert calls == [
        (
            "GET",
            "/federated-search",
            {
                "params": {
                    "q": "FAILED_PRECONDITION",
                    "domain": ["file", "pr"],
                    "limit": 10,
                    "repository": "marin-community/marin",
                }
            },
        )
    ]
    output = capsys.readouterr().out
    assert "1 result in 1.23s" in output
    assert cli.SEARCH_DETAIL_INSTRUCTION in output
    assert "file:731" in output
    assert "file:marin-community/marin@main:lib/iris/src/iris/scheduler.py" in output
    assert f"L42 {reference_text}" in output
    assert output.count("File scope: marin-community/marin") == 1


def test_search_defaults_to_curated_domains_without_discord(monkeypatch):
    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return json_response([])

    monkeypatch.setattr(cli, "request_response", fake_request)
    args = cli.build_parser().parse_args(["search", "scheduler", "--repository", "marin-community/marin"])
    args.func(args)

    assert calls == [
        (
            "GET",
            "/federated-search",
            {
                "params": {
                    "q": "scheduler",
                    "domain": ["wiki", "file", "pr", "issue"],
                    "limit": 10,
                    "repository": "marin-community/marin",
                }
            },
        )
    ]


def test_search_infers_configured_repository_from_contributor_fork(monkeypatch, tmp_path, capsys):
    repository = tmp_path / "vllm"
    subprocess.run(["git", "init", repository], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:contributor/vllm.git"],
        cwd=repository,
        check=True,
    )
    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return json_response([])

    monkeypatch.chdir(repository)
    monkeypatch.setattr(cli, "request_response", fake_request)
    args = cli.build_parser().parse_args(["search", "scheduler", "--domain", "file"])
    args.func(args)

    assert calls[0][2]["params"]["repository"] == "marin-community/vllm"
    assert capsys.readouterr().out.count("File scope: marin-community/vllm") == 1


@pytest.mark.parametrize("repository", ["marin-community/vllm", "all"])
def test_search_explicit_repository_bypasses_checkout_inference(monkeypatch, tmp_path, repository):
    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return json_response([])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "request_response", fake_request)
    args = cli.build_parser().parse_args(["search", "scheduler", "--domain", "file", "--repository", repository])
    args.func(args)

    assert calls[0][2]["params"]["repository"] == repository


def test_unscoped_file_search_fails_before_request_outside_supported_checkout(monkeypatch, tmp_path):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        return json_response([])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "request_response", fake_request)
    args = cli.build_parser().parse_args(["search", "scheduler", "--domain", "file"])

    with pytest.raises(SystemExit) as error:
        args.func(args)

    assert "--repository <owner/repo>" in str(error.value)
    assert "--repository all" in str(error.value)
    assert calls == []


def test_search_without_file_domain_works_outside_git(monkeypatch, tmp_path):
    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return json_response([])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "request_response", fake_request)
    args = cli.build_parser().parse_args(["search", "scheduler", "--domain", "wiki"])
    args.func(args)

    assert calls == [("GET", "/federated-search", {"params": {"q": "scheduler", "domain": ["wiki"], "limit": 10}})]


def test_bearer_token_quiets_only_the_known_missing_email_scope_warning(monkeypatch, caplog):
    class Provider:
        def get_token(self):
            logger = logging.getLogger("google.oauth2.credentials")
            logger.warning(cli.MISSING_EMAIL_SCOPE_WARNING)
            logger.warning("token endpoint is degraded")
            return "token"

    monkeypatch.setattr(cli, "cached_login_provider", lambda: Provider())

    with caplog.at_level(logging.WARNING):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(cli.bearer_token) for _ in range(4)]
            tokens = [future.result() for future in futures]

    assert tokens == ["token"] * 4
    assert [record.getMessage() for record in caplog.records] == ["token endpoint is degraded"] * 4


def test_get_fetches_full_detail_by_search_result_id(monkeypatch):
    calls = []
    result_id = "file:marin-community/marin@main:lib/iris/OPS.md"

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return {
            "id": result_id,
            "title": "Iris Operations",
            "subtitle": "marin-community/marin · lib/iris/OPS.md · main@abc123",
            "url": "https://github.com/marin-community/marin/blob/abc123/lib/iris/OPS.md",
            "text": "# Iris Operations\n\nDeploy with the restart command.",
        }

    monkeypatch.setattr(cli, "request", fake_request)
    args = cli.build_parser().parse_args(["get", result_id])
    args.func(args)

    assert calls == [("GET", "/repository-files/marin-community/marin@main:lib/iris/OPS.md", {})]


def test_get_resolves_numeric_file_search_key_to_source_id(monkeypatch, capsys):
    calls = []
    source_id = "file:marin-community/marin@main:infra/echo/README.md"

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        if path == "/search-results/20849":
            return {"key": "file:20849", "source_id": source_id, "domain": "file"}
        return {
            "id": source_id,
            "title": "Echo",
            "subtitle": "marin-community/marin · infra/echo/README.md · main@abc123",
            "url": "https://github.com/marin-community/marin/blob/abc123/infra/echo/README.md",
            "text": "# Echo",
        }

    monkeypatch.setattr(cli, "request", fake_request)
    args = cli.build_parser().parse_args(["get", "file:20849"])
    args.func(args)

    assert calls == [
        ("GET", "/search-results/20849", {}),
        ("GET", "/repository-files/marin-community/marin@main:infra/echo/README.md", {}),
    ]
    assert capsys.readouterr().out.startswith(f"[{source_id}] Echo\n")


def test_get_rejects_legacy_path_only_file_id():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["get", "file:lib/iris/OPS.md"])


def test_feedback_submits_replayable_grades_and_stdin_note(monkeypatch, capsys):
    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return {"id": 17}

    monkeypatch.setattr(cli, "request", fake_request)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("Wiki result was unrelated.\n"))
    args = cli.build_parser().parse_args(
        [
            "feedback",
            "--query",
            "how do I deploy Iris?",
            "--grade",
            "wiki:730=0",
            "--grade",
            "file:731=10",
        ]
    )
    args.func(args)

    assert calls == [
        (
            "POST",
            "/feedback",
            {
                "body": {
                    "query": "how do I deploy Iris?",
                    "grades": [
                        {"key": "wiki:730", "grade": 0},
                        {"key": "file:731", "grade": 10},
                    ],
                    "note": "Wiki result was unrelated.",
                }
            },
        )
    ]
    assert capsys.readouterr().out == "recorded feedback #17\n"


def test_feedback_links_execution_when_provided(monkeypatch):
    calls = []

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return {"id": 17}

    monkeypatch.setattr(cli, "request", fake_request)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("The file answered it.\n"))
    args = cli.build_parser().parse_args(
        [
            "feedback",
            "--query",
            "how do I deploy Iris?",
            "--execution-id",
            "991",
            "--grade",
            "file:731=10",
        ]
    )

    args.func(args)

    assert calls[0][2]["body"]["execution_id"] == 991


def test_history_export_pages_in_stable_id_order(monkeypatch, capsys):
    calls = []
    pages = [
        [{"id": 4, "query": "deploy iris"}, {"id": 7, "query": "inspect logs"}],
        [{"id": 9, "query": "reserve tpu"}],
    ]

    def fake_request(method, path, **options):
        calls.append((method, path, options))
        return pages.pop(0)

    monkeypatch.setattr(cli, "request", fake_request)
    args = cli.build_parser().parse_args(["history", "export", "--after-id", "3", "--page-size", "2"])

    args.func(args)

    assert calls == [
        ("GET", "/search-executions", {"params": {"after_id": 3, "mode": None, "limit": 2}, "timeout": 180}),
        ("GET", "/search-executions", {"params": {"after_id": 7, "mode": None, "limit": 2}, "timeout": 180}),
    ]
    assert [cli.json.loads(line)["id"] for line in capsys.readouterr().out.splitlines()] == [4, 7, 9]
