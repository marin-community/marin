# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judged Harbor packages graded on the host, against the requests their own scripts send.

The packages under ``fixtures/harbor`` are trimmed copies of real ones; financeagent-9 omits the
Compose file its grading does not use. Each ``*.judge_request.json`` holds the request the package's
script sent to a local endpoint: SimpleQA's ``tests/llm_judge.py`` for the answer ``Michio Sugeno``,
FinanceAgent's ``tests/run_test.py``, with LiteLLM stubbed, for ``CAGR: 30%``, and BixBench's
``tests/llm_judge.py``, with ``openai==2.14.0`` and ``MODEL_NAME`` set to the campaign judge, for
``<answer>0.0002</answer>``.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from taskcompendium.grading import grade_remote_judge_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.importers.harbor.package import AnswerFile, JobSettings, image_references, lower_package
from taskcompendium.models import ConversationTrace, GradingAttempt, Source, TaskSpec, TextMessage
from verifyit.modes.grade_judge import JudgeConnection

from experiments.evaluation.benchmarks.harbor_judges import (
    BIXBENCH_JUDGE_MODEL,
    bixbench_answer,
    financeagent_answer,
    simpleqa_answer,
)

FIXTURES = Path(__file__).parent / "fixtures" / "harbor"
SIMPLEQA = FIXTURES / "simpleqa-0"
FINANCEAGENT = FIXTURES / "financeagent-9"
BIXBENCH = FIXTURES / "bixbench-bix-1-q1"


class ResponsesServer(ThreadingHTTPServer):
    """Record requests to a local Responses or Chat Completions endpoint and answer each with ``reply``."""

    requests: list[dict]
    reply: str


class ResponsesHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server: ResponsesServer = self.server  # pyrefly: ignore[bad-assignment]
        server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        if self.path.endswith("/chat/completions"):
            choice = {"index": 0, "message": {"role": "assistant", "content": server.reply}, "finish_reason": "stop"}
            response = {"id": "chatcmpl-fixture", "object": "chat.completion", "created": 0, "model": "judge"}
            body = json.dumps({**response, "choices": [choice]}).encode()
        else:
            message = {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": server.reply}],
            }
            output = {"id": "resp-fixture", "status": "completed", "error": None, "output": [message]}
            body = json.dumps(output).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


@pytest.fixture
def judge_server():
    server = ResponsesServer(("127.0.0.1", 0), ResponsesHandler)
    server.requests = []
    server.reply = ""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def lowered_task(directory: Path, rule: AnswerFile) -> TaskSpec:
    digests = {reference: "sha256:" + "1" * 64 for reference in image_references(directory)}
    source = Source(dataset="fixture", revision="fixture", row=directory.name, importer_revision="1")
    return lower_package(
        directory, task_id=directory.name, source=source, rule=rule, image_digests=digests, job=JobSettings()
    ).task


def grade(directory: Path, rule: AnswerFile, files: dict[str, bytes], server: ResponsesServer) -> GradeResult:
    task = lowered_task(directory, rule)
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done.")))
    connection = JudgeConnection(f"http://127.0.0.1:{server.server_port}/v1", "fixture-key")
    return grade_remote_judge_answer(task, GradingAttempt(conversation, files=files), connection)


def harbor_request(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.judge_request.json").read_text())


@pytest.mark.parametrize(
    "reply,status,reward",
    [
        ("A", Outcome.GRADED, 1.0),
        (" b\n", Outcome.GRADED, 0.0),
        ("C", Outcome.GRADED, 0.0),
        ("The grade is A", Outcome.INFRA_ERROR, None),
    ],
)
def test_simpleqa_sends_harbors_judge_request_and_credits_only_grade_a(judge_server, reply, status, reward):
    judge_server.reply = reply
    rule = simpleqa_answer(SIMPLEQA)

    result = grade(SIMPLEQA, rule, {rule.path: b"Michio Sugeno\n"}, judge_server)

    assert (result.status, result.reward) == (status, reward)
    assert judge_server.requests == [harbor_request("simpleqa-0")]


@pytest.mark.parametrize(
    "files,status",
    [({}, Outcome.SUBMISSION_FAILURE), ({"/workspace/answer.txt": b" \n"}, Outcome.GRADED)],
)
def test_simpleqa_without_an_answer_scores_zero_without_a_judge_call(judge_server, files, status):
    result = grade(SIMPLEQA, simpleqa_answer(SIMPLEQA), files, judge_server)

    assert (result.status, result.reward) == (status, 0.0)
    assert judge_server.requests == []


@pytest.mark.parametrize(
    "reply,status,reward",
    [
        ('```json\n{"judgment": "correct", "explanation": "Same CAGR."}\n```', Outcome.GRADED, 1.0),
        ('{"judgment": "INCORRECT", "explanation": "The CAGR is 22.9%, not 30%."}', Outcome.GRADED, 0.0),
        ('{"judgment": "ERROR", "explanation": "Unsure."}', Outcome.INFRA_ERROR, None),
    ],
)
def test_financeagent_sends_harbors_judge_prompt_and_reads_its_json_judgment(judge_server, reply, status, reward):
    judge_server.reply = reply
    rule = financeagent_answer(FINANCEAGENT)

    result = grade(FINANCEAGENT, rule, {rule.path: b"CAGR: 30%\n"}, judge_server)

    assert (result.status, result.reward) == (status, reward)
    [request] = judge_server.requests
    assert request["messages"] == harbor_request("financeagent-9")["messages"]
    assert "temperature" not in request


def test_financeagent_credits_the_reference_answer_without_a_judge_call(judge_server):
    rule = financeagent_answer(FINANCEAGENT)
    expected = json.loads((FINANCEAGENT / "tests" / "config.json").read_text())["expected_answer"]

    result = grade(FINANCEAGENT, rule, {rule.path: f"{expected}\n".encode()}, judge_server)

    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    assert judge_server.requests == []


@pytest.mark.parametrize(
    "reply,status,reward",
    [
        ('{"binary_score": true}', Outcome.GRADED, 1.0),
        ('{"binary_score":false}', Outcome.GRADED, 0.0),
        ('{"binary_score": "unsure"}', Outcome.INFRA_ERROR, None),
    ],
)
def test_bixbench_sends_harbors_judge_request_and_reads_its_binary_score(judge_server, reply, status, reward):
    judge_server.reply = reply
    rule = bixbench_answer(BIXBENCH)

    result = grade(BIXBENCH, rule, {rule.path: b"<answer>0.0002</answer>\n"}, judge_server)

    assert (result.status, result.reward) == (status, reward)
    # The SDK's parse names the limit max_tokens and sends stream false, which vLLM reads as
    # max_completion_tokens and its default.
    expected = harbor_request("bixbench-bix-1-q1")
    expected["max_completion_tokens"] = expected.pop("max_tokens")
    del expected["stream"]
    # Marin's judge is a hosted model, not the one Harbor's run served beside the candidate.
    expected["model"] = BIXBENCH_JUDGE_MODEL
    assert judge_server.requests == [expected]


def test_bixbench_judges_an_empty_answer_and_scores_a_missing_one_zero(judge_server):
    judge_server.reply = '{"binary_score": false}'
    rule = bixbench_answer(BIXBENCH)

    missing = grade(BIXBENCH, rule, {}, judge_server)
    assert (missing.status, missing.reward, judge_server.requests) == (Outcome.SUBMISSION_FAILURE, 0.0, [])

    empty = grade(BIXBENCH, rule, {rule.path: b""}, judge_server)
    assert (empty.status, empty.reward) == (Outcome.GRADED, 0.0)
    [request] = judge_server.requests
    assert "\nProposed answer: \nYou must respond" in request["messages"][1]["content"]
