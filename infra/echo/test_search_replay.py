# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import build_search_quality_manifest
import pytest
import requests
import search_replay


def test_replay_resumes_completed_cases_and_persists_each_execution(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "results.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "observed-001",
                        "query": "deploy iris",
                        "domains": ["wiki", "file"],
                        "source": "observed",
                        "occurrences": 2,
                    }
                ),
                json.dumps(
                    {
                        "id": "benchmark-restart",
                        "query": "restart iris",
                        "domains": ["file"],
                        "source": "benchmark",
                    }
                ),
            ]
        )
        + "\n"
    )
    output.write_text('{"execution_id":4,"id":"observed-001","returned_count":3}\n')
    response = requests.Response()
    response.status_code = 200
    response.headers["X-Echo-Search-Execution-ID"] = "9"
    response._content = b'[{"id":"file:1"},{"id":"wiki:2"}]'
    requests_sent = []

    def fake_request(method, path, *, params, timeout):
        requests_sent.append((method, path, params, timeout))
        return response

    monkeypatch.setattr(search_replay.echo_cli, "request_response", fake_request)

    search_replay.replay(search_replay.load_cases(manifest), output, limit=10, workers=1)

    assert requests_sent == [
        (
            "GET",
            "/federated-search",
            {"q": "restart iris", "domain": ["file"], "limit": 10},
            search_replay.REPLAY_REQUEST_TIMEOUT,
        )
    ]
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert results == [
        {"execution_id": 4, "id": "observed-001", "returned_count": 3},
        {"execution_id": 9, "id": "benchmark-restart", "returned_count": 2},
    ]


def test_replay_manifest_rejects_duplicate_normalized_queries(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"a","query":"Deploy  Iris","domains":["file"],"source":"observed"}\n'
        '{"id":"b","query":" deploy iris ","domains":["file"],"source":"benchmark"}\n'
    )

    with pytest.raises(ValueError, match="duplicate normalized query"):
        search_replay.load_cases(manifest)


def test_replay_retries_rate_limited_normal_search(monkeypatch):
    rate_limited = SystemExit("GET /federated-search -> 429: Rate exceeded.")
    response = requests.Response()
    response.status_code = 200
    response.headers["X-Echo-Search-Execution-ID"] = "42"
    response._content = b"[]"
    responses = iter([rate_limited, response])
    sleeps = []

    def fake_request(*_args, **_kwargs):
        value = next(responses)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(search_replay.echo_cli, "request_response", fake_request)
    monkeypatch.setattr(search_replay.time, "sleep", sleeps.append)
    case = search_replay.ReplayCase("observed-a", "deploy iris", ("file",), "observed", 1)

    result = search_replay.replay_one(case, 10)

    assert result == search_replay.ReplayResult("observed-a", 42, 0)
    assert sleeps == [15]


def test_reconcile_recovers_matching_query_and_domains_once(tmp_path):
    output = tmp_path / "results.jsonl"
    history = tmp_path / "history.jsonl"
    output.write_text('{"execution_id":4,"id":"observed-a","returned_count":3}\n')
    history.write_text(
        '{"id":7,"normalized_query":"deploy iris","domains":["file"],"returned_count":2}\n'
        '{"id":8,"normalized_query":"inspect logs","domains":["wiki"],"returned_count":4}\n'
        '{"id":9,"normalized_query":"inspect logs","domains":["file"],"returned_count":5}\n'
        '{"id":10,"normalized_query":"inspect logs","domains":["file"],"returned_count":6}\n'
    )
    cases = [
        search_replay.ReplayCase("observed-a", "deploy iris", ("file",), "observed", 1),
        search_replay.ReplayCase("observed-b", "Inspect Logs", ("file",), "observed", 1),
    ]

    recovered = search_replay.reconcile_history(cases, history, output)

    assert recovered == 1
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert results[-1] == {"execution_id": 9, "id": "observed-b", "returned_count": 5}


def test_quality_manifest_groups_observed_queries_and_defaults_no_answer_domains(tmp_path):
    query_log = tmp_path / "query-log.md"
    benchmark = tmp_path / "benchmark.jsonl"
    query_log.write_text(
        "# Log\n```jsonl\n"
        '{"mode":"federated","query":"Deploy  Iris","domains":["file"],"timestamp":"2026-08-10T00:00:00Z"}\n'
        '{"mode":"federated","query":"deploy iris","domains":["file"],"timestamp":"2026-08-11T00:00:00Z"}\n'
        '{"mode":"grep","query":"deploy iris","domains":[],"timestamp":"2026-08-11T00:00:00Z"}\n'
        "```\n"
    )
    benchmark.write_text(
        '{"id":"unknown","query":"unsupported question","domains":[],"split":"test","intent":"no_answer"}\n'
    )

    cases = build_search_quality_manifest.build_manifest(query_log, benchmark, ["feedback only"])

    assert len(cases) == 3
    assert cases[0].query == "deploy iris"
    assert cases[0].occurrences == 2
    assert cases[1].source == "feedback-only"
    assert cases[2].domains == build_search_quality_manifest.search_config.DEFAULT_SEARCH_DOMAINS
