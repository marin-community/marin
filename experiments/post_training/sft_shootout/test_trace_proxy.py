# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

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
    is_completion_claim,
    is_format_error,
    route,
)
from experiments.post_training.sft_shootout.trace_proxy import Backend, RelayPolicy, RequestLog, build_proxy


class WordTokenizer:
    """Counts whitespace-separated words of the message contents, times ``scale``."""

    def __init__(self, scale: int = 1):
        self.scale = scale

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False, **_):
        return " ".join(str(m.get("content") or "") for m in messages)

    def encode(self, text, add_special_tokens=False):
        return text.split() * self.scale


def backend(url: str, scale: int = 1, **kwargs) -> Backend:
    return Backend(base_url=url, model="m", tokenizer=WordTokenizer(scale), headers={}, **kwargs)


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


def test_format_errors_from_both_harnesses():
    assert is_format_error(TERMINUS_ERROR)
    assert is_format_error(MINI_ERROR)
    assert is_format_error([MINI_ERROR[0], MINI_ERROR[-1]])  # the first reply was dropped
    assert not is_format_error(MINI_ERROR[:3])
    assert not is_format_error([{"role": "user", "content": "Tool call error: in the task text"}])
    assert not is_format_error([*TERMINUS_ERROR[:2], {"role": "user", "content": "$ ls\nfile"}])


def test_completion_claims_from_both_harnesses():
    assert is_completion_claim({"content": '{"analysis": "", "commands": [], "task_complete": true}'})
    assert not is_completion_claim({"content": '{"commands": [], "task_complete": false}'})
    submit = {
        "function": {"name": "bash", "arguments": json.dumps({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})}
    }
    assert is_completion_claim({"content": "", "tool_calls": [submit]})
    assert not is_completion_claim(
        {"content": "", "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
    )


def test_route_repairs_once_and_takes_over_on_budget():
    state = TrialState()
    assert route(state, TERMINUS_ERROR, 10, 100) == (Author.TEACHER, Trigger.FORMAT_ERROR)
    assert state.takeover is None
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
    policy = RelayPolicy(student=backend("http://student/v1", scale=2), budget_tokens=50)
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


def test_trials_keep_separate_state(relay_app):
    test_client, student_replies, calls, _ = relay_app
    student_replies.append(reply("{}"))
    post(test_client, TERMINUS_ERROR, trial="a")
    post(test_client, [{"role": "user", "content": "task"}], trial="b")
    assert calls == ["teacher", "student"]


def test_tokenize_reports_the_larger_count(relay_app):
    test_client, *_ = relay_app
    response = test_client.post("/tokenize", json={"messages": [{"role": "user", "content": "a b c"}]})
    assert response.json() == {"count": 6, "max_model_len": 100}


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


def test_backend_failure_is_logged_and_returns_502(relay_app):
    test_client, _, _, log_path = relay_app  # no student reply queued: the student call raises
    response = post(test_client, [{"role": "user", "content": "fix the bug"}])
    assert response.status_code == 502
    (record,) = read_log(log_path)
    assert record["status"] == 502 and record["author"] == "student" and record["trial"] == "t1" and record["error"]


@pytest.mark.parametrize("failure", ["refused", 502, 404])
def test_a_moving_student_is_retried_until_it_answers(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(trace_proxy, "RETRY_INTERVAL", 0)
    replies = [failure, failure, 200]

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


def test_rebuilt_rows_join_harbor_retries_by_correlation_id(tmp_path):
    def write_result(directory, trial_id, reward):
        directory.mkdir(parents=True)
        result = {
            "task_name": "pool/task-1",
            "trial_name": "task-1__abc",
            "agent_result": {"metadata": {"rollout_correlation_id": trial_id}},
            "verifier_result": {"rewards": {"reward": reward}},
        }
        (directory / "result.json").write_text(json.dumps(result))

    write_result(tmp_path / "job" / "task-1__abc" / "attempts" / "000", "first", None)
    write_result(tmp_path / "job" / "task-1__abc" / "attempts" / "001", "second", 1.0)
    (tmp_path / "job" / "result.json").write_text(json.dumps({"stats": {}}))  # the job's own result
    results = harbor_results(StoragePath(str(tmp_path)))
    assert {trial: (r["retry"], r["reward"]) for trial, r in results.items()} == {"first": (0, None), "second": (1, 1.0)}

    turn = {"status": 200, "author": "teacher", "response_messages": [{"content": "done"}], "usage": None}
    records = [
        {"ts": 1, "trial": "second", "conversation": 0, "first_message": 0, "n_messages": 1, "messages": [], **turn},
        {"ts": 2, "trial": "orphan", "conversation": 0, "first_message": 0, "n_messages": 1, "messages": [], **turn},
    ]
    rows = {row["trial_id"]: row for row in rebuilt_rows(records, results)}
    assert rows["second"]["reward"] == 1.0 and rows["second"]["problems"] == []
    assert rows["orphan"]["problems"] == ["no Harbor trial carries this correlation id"]
