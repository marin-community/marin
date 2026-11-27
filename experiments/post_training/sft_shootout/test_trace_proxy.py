# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import httpx
import pytest
from rigging.filesystem.storage_path import StoragePath
from starlette.testclient import TestClient

from experiments.post_training.sft_shootout import trace_proxy
from experiments.post_training.sft_shootout.rebuild_traces import harbor_results, rebuild, rebuilt_rows
from experiments.post_training.sft_shootout.trace_proxy import Backend, RequestLog, build_proxy

FIXTURE = Path(__file__).parent / "testdata" / "effort_pilot"


class WordTokenizer:
    """Counts whitespace-separated words of the message contents, after the ``system_prompt`` template kwarg."""

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False, system_prompt="", **_
    ):
        return " ".join([system_prompt, *(str(m.get("content") or "") for m in messages)])

    def encode(self, text, add_special_tokens=False):
        return text.split()


def backend(url: str, **kwargs) -> Backend:
    return Backend(base_url=url, model="m", tokenizer=WordTokenizer(), headers={}, **kwargs)


def reply(content: str, tool_calls: list | None = None) -> dict:
    return {
        "id": "r",
        "model": "m",
        "choices": [
            {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def post(test_client: TestClient, messages: list[dict], trial: str = "t1") -> httpx.Response:
    return test_client.post(
        "/v1/chat/completions", json={"model": "any", "messages": messages}, headers={trace_proxy.TRIAL_HEADER: trial}
    )


def read_log(log_path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text().splitlines()]


@pytest.fixture
def recording_app(tmp_path):
    replies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=replies.pop(0))

    log_path = tmp_path / "requests.jsonl"
    app = build_proxy(
        backend("http://backend/v1"),
        100,
        RequestLog(log_path),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return TestClient(app), replies, log_path


def test_rebuild_restores_reasoning_and_the_final_reply(recording_app):
    test_client, replies, log_path = recording_app
    first_reply = reply("ls")
    first_reply["choices"][0]["message"]["reasoning"] = "think first"
    replies.extend([first_reply, reply("done")])
    first = [{"role": "user", "content": "task"}]
    post(test_client, first)
    post(test_client, [*first, {"role": "assistant", "content": "ls"}, {"role": "user", "content": "$"}])
    rebuilt = rebuild(read_log(log_path))
    assert rebuilt.problems == []
    assert [m["content"] for m in rebuilt.messages] == ["task", "ls", "$", "done"]
    assert rebuilt.messages[1]["reasoning_content"] == "think first"


def test_summarization_restart_and_retry_keep_conversations_apart(recording_app):
    test_client, replies, log_path = recording_app
    replies.extend([reply("a1"), reply("a1 again"), reply("summary"), reply("b1"), reply('{"task_complete": true}')])
    first = [{"role": "user", "content": "task"}]
    post(test_client, first)
    post(test_client, first)  # retried request: replaces the pending reply
    post(test_client, [{"role": "user", "content": "summarize: task a1 again"}])
    restarted = [{"role": "user", "content": "task, summary so far"}]
    post(test_client, restarted)
    post(test_client, [*first, {"role": "assistant", "content": "a1 again"}, {"role": "user", "content": "$"}])
    records = read_log(log_path)
    assert [(r["conversation"], r["first_message"]) for r in records] == [(0, 0), (0, 1), (1, 0), (2, 0), (0, 1)]
    rebuilt = rebuild([r for r in records if r["conversation"] == 0])
    assert rebuilt.problems == []
    assert [m["content"] for m in rebuilt.messages] == ["task", "a1 again", "$", '{"task_complete": true}']


def test_backend_failure_is_logged_and_returns_502(recording_app):
    test_client, _, log_path = recording_app
    response = post(test_client, [{"role": "user", "content": "fix the bug"}])
    assert response.status_code == 502
    (record,) = read_log(log_path)
    assert record["status"] == 502 and record["trial"] == "t1" and record["error"]


def test_a_moving_backend_is_retried_until_it_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_proxy, "RETRY_INTERVAL", 0)
    replies = ["refused", 503, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = replies.pop(0)
        if status == "refused":
            raise httpx.ConnectError("All connection attempts failed", request=request)
        return httpx.Response(status, json=reply("ls") if status == 200 else {"error": "endpoint moving"})

    moving = backend("https://iris.example/proxy/serve/v1", retry_window=60)
    log_path = tmp_path / "requests.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_proxy(moving, 100, RequestLog(log_path), client)
    response = post(TestClient(app), [{"role": "user", "content": "fix the bug"}])
    assert response.status_code == 200 and replies == []
    (record,) = read_log(log_path)
    assert record["status"] == 200


def test_backend_overrides_and_headers_are_sent_on_every_request(tmp_path):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=reply("ls"))

    overrides = {"chat_template_kwargs": {"reasoning_effort": "low"}}
    configured = Backend(
        base_url="http://configured/v1",
        model="served-name",
        tokenizer=WordTokenizer(),
        headers={"Authorization": "Bearer secret"},
        overrides=overrides,
    )
    log_path = tmp_path / "requests.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_proxy(configured, 100, RequestLog(log_path), client)
    test_client = TestClient(app)
    post(test_client, [{"role": "user", "content": "fix the bug"}])
    post(test_client, [{"role": "user", "content": "fix the bug"}, {"role": "assistant", "content": "ls"}])
    for request in sent:
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"reasoning_effort": "low"} and body["model"] == "served-name"
        assert request.headers["authorization"] == "Bearer secret"
    assert len(sent) == 2
    assert all(record["overrides"] == overrides for record in read_log(log_path))
    assert "secret" not in log_path.read_text()


def test_tokenize_counts_the_request_the_backend_receives(tmp_path):
    overrides = {"chat_template_kwargs": {"system_prompt": "be brief"}}
    counted = backend("http://backend/v1", overrides=overrides)
    log = RequestLog(tmp_path / "requests.jsonl")
    app = build_proxy(counted, 100, log, httpx.AsyncClient())
    body = {"messages": [{"role": "user", "content": "fix the bug"}], "chat_template_kwargs": {"system_prompt": "x"}}
    assert TestClient(app).post("/tokenize", json=body).json() == {"count": 5, "max_model_len": 100}


def test_rebuild_joins_a_real_run_log_to_its_harbor_trial():
    # First 8 requests of one GLM mini-swe-agent trial from a trace-generation run, with its trimmed result.json.
    log = [json.loads(line) for line in (FIXTURE / "requests.jsonl").read_text().splitlines()]
    (row,) = rebuilt_rows(log, harbor_results(StoragePath(str(FIXTURE / "jobs"))))
    assert (row["trial_name"], row["retry"], row["reward"], row["problems"]) == ("swesmith-00008__Gdi9WTi", 0, 0.0, [])
    assert [m["role"] for m in row["messages"]] == ["system", "user"] + ["assistant", "tool"] * 7 + ["assistant"]
    assert row["tokens"] == {"turns": 8, "prompt": 19713, "completion": 977, "reasoning": 101}


@pytest.mark.parametrize("failure", [503, "connection"])
def test_backend_retries_keep_capability_urls_out_of_logs(tmp_path, monkeypatch, caplog, failure):
    secret = "synthetic-capability-for-log-regression"
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            if failure == "connection":
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(failure)
        return httpx.Response(200, json=reply("done"))

    monkeypatch.setattr(trace_proxy, "RETRY_INTERVAL", 0)
    configured = backend(f"http://configured/proxy/t/{secret}/serve/v1", retry_window=60)
    log_path = tmp_path / "requests.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_proxy(configured, 100, RequestLog(log_path), client)
    with caplog.at_level("WARNING", logger=trace_proxy.__name__):
        response = post(TestClient(app), [{"role": "user", "content": "task"}])
    assert response.status_code == 200 and len(attempts) == 2
    assert any(record.name == trace_proxy.__name__ for record in caplog.records)
    assert secret not in caplog.text
    assert secret not in log_path.read_text()


def test_rebuild_keeps_completed_trials_when_an_attempt_has_no_result(tmp_path):
    jobs = tmp_path / "jobs"
    shutil.copytree(FIXTURE / "jobs", jobs)
    incomplete = jobs / "interrupted" / "agent"
    incomplete.mkdir(parents=True)
    (incomplete / "trial_correlation_id.txt").write_text("interrupted-trial")
    records = [json.loads(line) for line in (FIXTURE / "requests.jsonl").read_text().splitlines()]
    records.append({**records[0], "trial": "interrupted-trial"})
    rows = rebuilt_rows(records, harbor_results(StoragePath(str(jobs))))
    completed = next(row for row in rows if row["trial_id"] != "interrupted-trial")
    unfinished = next(row for row in rows if row["trial_id"] == "interrupted-trial")
    assert completed["reward"] == 0.0 and completed["problems"] == []
    assert unfinished["problems"] == ["no Harbor trial carries this correlation id"]
    assert unfinished["messages"]
