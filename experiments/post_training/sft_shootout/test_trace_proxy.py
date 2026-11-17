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
from experiments.post_training.sft_shootout.relay import (
    Author,
    TrialState,
    Trigger,
    route,
)
from experiments.post_training.sft_shootout.trace_proxy import Backend, RelayPolicy, RequestLog, build_proxy

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


TERMINUS_ERROR = [
    {"role": "user", "content": "task"},
    {"role": "assistant", "content": "not json"},
    {"role": "user", "content": "Previous response had parsing errors:\nERROR: No valid JSON found"},
]
# mini-swe-agent drops the reply without a tool call, so its error follows the previous turn's observation.
MINI_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}],
}
MINI_ERROR = [
    {"role": "user", "content": "task"},
    MINI_CALL,
    {"role": "tool", "tool_call_id": "c1", "content": "file"},
    {"role": "user", "content": "Tool call error:\n\n<error>\nNo tool calls found\n</error>"},
]


def test_route_repairs_format_errors_once_and_takes_over_on_budget():
    state = TrialState()
    assert route(state, TERMINUS_ERROR, 10, 100) == (Author.TEACHER, Trigger.FORMAT_ERROR)
    assert route(state, MINI_ERROR, 10, 100) == (Author.TEACHER, Trigger.FORMAT_ERROR)
    assert route(state, [MINI_ERROR[0], MINI_ERROR[-1]], 10, 100) == (Author.TEACHER, Trigger.FORMAT_ERROR)
    task_text = [{"role": "user", "content": "Tool call error: in the task text"}]
    assert route(state, task_text, 10, 100) == (Author.STUDENT, None)
    assert route(state, TERMINUS_ERROR[:2], 10, 100) == (Author.STUDENT, None)
    assert route(state, TERMINUS_ERROR[:2], 100, 100) == (Author.TEACHER, Trigger.CONTEXT_BUDGET)
    assert route(state, TERMINUS_ERROR[:2], 10, 100) == (Author.TEACHER, None)


@pytest.fixture
def relay_app(tmp_path):
    student_replies: list[dict] = []
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(host)
        if host == "student":
            return httpx.Response(200, json=student_replies.pop(0))
        return httpx.Response(200, json=reply('{"task_complete": true}'))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    log_path = tmp_path / "requests.jsonl"
    policy = RelayPolicy(student=backend("http://student/v1"), budget_tokens=50)
    app = build_proxy(backend("http://teacher/v1"), 100, RequestLog(log_path), policy, client)
    return TestClient(app), student_replies, calls, log_path


def post(test_client: TestClient, messages: list[dict], trial: str = "t1") -> httpx.Response:
    return test_client.post(
        "/v1/chat/completions", json={"model": "any", "messages": messages}, headers={trace_proxy.TRIAL_HEADER: trial}
    )


def read_log(log_path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def test_completion_claim_hands_the_turn_and_the_rest_to_the_teacher(relay_app):
    test_client, student_replies, calls, log_path = relay_app
    student_replies.extend([reply('{"commands": []}'), reply('{"task_complete": true}')])
    first = [{"role": "user", "content": "task"}]
    assert post(test_client, first).status_code == 200
    second = [*first, {"role": "assistant", "content": '{"commands": []}'}, {"role": "user", "content": "$"}]
    assert post(test_client, second).json()["choices"][0]["message"]["content"] == '{"task_complete": true}'
    third = [*second, {"role": "assistant", "content": "x"}, {"role": "user", "content": "confirm?"}]
    post(test_client, third)
    assert calls == ["student", "student", "teacher", "teacher"]
    records = read_log(log_path)
    assert [(r["author"], r["trigger"]) for r in records] == [
        ("student", None),
        ("teacher", "completion_claim"),
        ("teacher", None),
    ]
    assert records[1]["withheld_student_turn"]["author"] == "student"
    assert [r["first_message"] for r in records] == [0, 1, 3]
    assert [len(r["messages"]) for r in records] == [1, 2, 2]


def test_rebuild_restores_replies_authors_and_the_final_turn(relay_app):
    test_client, student_replies, _, log_path = relay_app
    reasoning_reply = reply('{"commands": []}')
    reasoning_reply["choices"][0]["message"]["reasoning"] = "think first"
    student_replies.extend([reasoning_reply, reply('{"task_complete": true}')])
    first = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    post(test_client, first)
    second = [*first, {"role": "assistant", "content": '{"commands": []}'}, {"role": "user", "content": "$"}]
    post(test_client, second)
    rebuilt = rebuild(read_log(log_path))
    conversation = rebuilt.messages
    assert rebuilt.problems == []
    assert [m["role"] for m in conversation] == ["system", "user", "assistant", "user", "assistant"]
    assert conversation[2]["reasoning_content"] == "think first" and conversation[2]["author"] == "student"
    assert conversation[4]["author"] == "teacher" and conversation[4]["trigger"] == "completion_claim"


def test_summarization_restart_and_retry_keep_conversations_apart(relay_app):
    test_client, student_replies, _, log_path = relay_app
    student_replies.extend(
        [reply("a1"), reply("a1 again"), reply("summary"), reply("b1"), reply('{"task_complete": true}')]
    )
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


def test_relay_refuses_a_request_without_a_trial_header(relay_app):
    test_client, _, calls, log_path = relay_app
    response = test_client.post(
        "/v1/chat/completions", json={"model": "any", "messages": [{"role": "user", "content": "task"}]}
    )
    assert response.status_code == 400 and calls == [] and not log_path.exists()


def test_backend_failure_is_logged_and_returns_502(relay_app):
    test_client, _, _, log_path = relay_app  # no student reply queued: the student call raises
    response = post(test_client, [{"role": "user", "content": "fix the bug"}])
    assert response.status_code == 502
    (record,) = read_log(log_path)
    assert record["status"] == 502 and record["author"] == "student" and record["trial"] == "t1" and record["error"]


def test_a_moving_student_is_retried_until_it_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_proxy, "RETRY_INTERVAL", 0)
    replies = ["refused", 503, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = replies.pop(0)
        if status == "refused":
            raise httpx.ConnectError("All connection attempts failed", request=request)
        return httpx.Response(status, json=reply("ls") if status == 200 else {"error": "endpoint moving"})

    student = backend("https://iris.example/proxy/serve/v1", retry_window=60)
    log_path = tmp_path / "requests.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_proxy(backend("http://teacher/v1"), 100, RequestLog(log_path), RelayPolicy(student, 50), client)
    response = post(TestClient(app), [{"role": "user", "content": "fix the bug"}])
    assert response.status_code == 200 and replies == []
    (record,) = read_log(log_path)
    assert record["status"] == 200 and record["author"] == "student"


def test_backend_overrides_and_headers_are_sent_on_every_request(tmp_path):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=reply("ls"))

    overrides = {"chat_template_kwargs": {"reasoning_effort": "low"}}
    teacher = Backend(
        base_url="http://teacher/v1",
        model="served-name",
        tokenizer=WordTokenizer(),
        headers={"Authorization": "Bearer secret"},
        overrides=overrides,
    )
    log_path = tmp_path / "requests.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_proxy(teacher, 100, RequestLog(log_path), None, client)
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


def test_tokenize_counts_the_request_each_backend_receives(tmp_path):
    overrides = {"chat_template_kwargs": {"system_prompt": "be brief"}}
    student = backend("http://student/v1", overrides=overrides)
    log = RequestLog(tmp_path / "requests.jsonl")
    app = build_proxy(backend("http://teacher/v1"), 100, log, RelayPolicy(student, 50), httpx.AsyncClient())
    body = {"messages": [{"role": "user", "content": "fix the bug"}], "chat_template_kwargs": {"system_prompt": "x"}}
    assert TestClient(app).post("/tokenize", json=body).json() == {"count": 5, "max_model_len": 100}


def test_rebuild_joins_a_real_run_log_to_its_harbor_trial():
    # First 8 requests of one GLM mini-swe-agent trial from a trace-generation run, with its trimmed result.json.
    log = [json.loads(line) for line in (FIXTURE / "requests.jsonl").read_text().splitlines()]
    (row,) = rebuilt_rows(log, harbor_results(StoragePath(str(FIXTURE / "jobs"))))
    assert (row["trial_name"], row["retry"], row["reward"], row["problems"]) == ("swesmith-00008__Gdi9WTi", 0, 0.0, [])
    assert [m["role"] for m in row["messages"]] == ["system", "user"] + ["assistant", "tool"] * 7 + ["assistant"]
    assert row["tokens"] == {"teacher": {"turns": 8, "prompt": 19713, "completion": 977, "reasoning": 101}}


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
    teacher = backend(f"http://teacher/proxy/t/{secret}/serve/v1", retry_window=60)
    log_path = tmp_path / "requests.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = build_proxy(teacher, 100, RequestLog(log_path), None, client)
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
