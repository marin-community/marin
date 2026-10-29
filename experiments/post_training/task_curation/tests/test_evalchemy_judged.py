# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise source preparation, judge failures and retry evidence through HTTP."""

import csv
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_remote_judge_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage, verifyit_spec
from taskcompendium.pipeline.evaluation import evaluation_cases
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from verifyit.modes.grade_judge import JudgeConnection
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.evalchemy.judged import read_simpleqa, sources
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import curate


def write_simpleqa(file, rows):
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metadata", "problem", "answer"))
        writer.writeheader()
        writer.writerows(
            {"problem": row["question"], "answer": row["answer"], "metadata": repr(row["metadata"])} for row in rows
        )


@pytest.fixture(params=["FinanceBench", "SimpleQA"])
def judged_corpus(request, tmp_path, monkeypatch):
    reference = {
        "name": request.param,
        "rows": [{"question": "What is the revenue?", "answer": "$7", "evidence_text": "Revenue: $7", "metadata": {}}],
    }
    source = next(source for source in sources() if source.config.protocol.benchmark == request.param)
    dataset = source.config
    root = tmp_path / "source"
    file = root / dataset.files[0]
    file.parent.mkdir(parents=True)
    if request.param == "FinanceBench":
        file.write_text("".join(json.dumps(row) + "\n" for row in reference["rows"]))
    else:
        write_simpleqa(file, reference["rows"])
    source = replace(
        source,
        config=replace(
            dataset,
            protocol=dataset.protocol.model_copy(
                update={
                    "expected_population": len(reference["rows"]),
                }
            ),
        ),
    )
    runtime = CampaignRuntime()
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime, inputs=InputOverrides(root=str(root)))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            artifact = run(curate(source, options), max_concurrent=1)[0]
        yield reference, list(evaluation_cases(artifact.path))
    finally:
        client.shutdown()


@pytest.fixture
def judge_http(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    state = {"response": {}, "requests": [], "request_times": [], "status": 200, "responses": [], "retry_after": None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["requests"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            state["request_times"].append(time.monotonic())
            status, body, retry_after = (
                state["responses"].pop(0)
                if state["responses"]
                else (
                    state["status"],
                    state["response"],
                    state["retry_after"],
                )
            )
            payload = json.dumps(body).encode()
            self.send_response(status)
            if retry_after is not None:
                self.send_header("Retry-After", str(retry_after))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield state, JudgeConnection(f"http://127.0.0.1:{server.server_port}/v1", "private-fixture-key")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def judged_attempt(task, candidate):
    return GradingAttempt(
        ConversationTrace(
            events=(
                *task.context.events,
                TextMessage(role="assistant", content=candidate),
            )
        )
    )


def judge_response(name, *, reply=None, finish_reason="stop"):
    return {
        "id": "fixture-judge",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture-revision",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": reply if reply is not None else ("CORRECT" if name == "FinanceBench" else "A"),
                },
            }
        ],
    }


def test_source_judge_grades_empty_and_exact_candidates_and_retains_model_identity(judged_corpus, judge_http):
    inputs, cases = judged_corpus
    state, connection = judge_http
    state["response"] = judge_response(inputs["name"])
    for candidate in ("", str(inputs["rows"][0]["answer"])):
        result = grade_remote_judge_answer(cases[0].task, judged_attempt(cases[0].task, candidate), connection)
        assert result.status is Outcome.GRADED
        assert result.reward == 1
        assert result.detail["requested_model"] == state["requests"][-1]["model"]
        assert result.detail["response_model"] == "fixture-revision"
    assert len(state["requests"]) == 2
    assert len(cases) == len(inputs["rows"])


@pytest.mark.parametrize(
    "reply,finish_reason,expected", [(None, "length", Outcome.GRADED), ("malformed", "stop", Outcome.INFRA_ERROR)]
)
def test_source_judge_accepts_content_on_length_and_keeps_malformed_labels_unscored(
    judged_corpus, judge_http, reply, finish_reason, expected
):
    inputs, cases = judged_corpus
    state, connection = judge_http
    state["response"] = judge_response(inputs["name"], reply=reply, finish_reason=finish_reason)
    result = grade_remote_judge_answer(cases[0].task, judged_attempt(cases[0].task, "candidate"), connection)
    assert result.status is expected
    assert result.reward == (1 if expected is Outcome.GRADED else None)
    assert len(state["requests"]) == 1
    assert result.detail["finish_reason"] == finish_reason


def test_provider_failure_does_not_become_an_incorrect_answer(judged_corpus, judge_http):
    _, cases = judged_corpus
    state, connection = judge_http
    state["status"] = 400
    state["response"] = {"error": {"message": "provider unavailable", "type": "fixture_error"}}
    result = grade_remote_judge_answer(cases[0].task, judged_attempt(cases[0].task, "candidate"), connection)
    assert result.status is Outcome.INFRA_ERROR
    assert result.reward is None
    assert len(state["requests"]) == 1


def test_transient_provider_failure_retries_with_source_policy_and_preserves_evidence(judged_corpus, judge_http):
    reference, cases = judged_corpus
    state, connection = judge_http
    failure = {"error": {"message": "temporary overload"}}
    state["responses"] = [(503, failure, 0.25), (200, judge_response(reference["name"]), None)]
    result = grade_remote_judge_answer(cases[0].task, judged_attempt(cases[0].task, "candidate"), connection)
    assert result.status is Outcome.GRADED
    assert result.reward == 1
    assert len(state["requests"]) == 2
    assert state["request_times"][1] - state["request_times"][0] >= 0.25
    first, second = result.detail["transport_attempts"]
    assert first["status"] == 503
    assert json.loads(first["body"]) == failure
    assert first["reason"] == "http_503"
    assert second["outcome"] == "succeeded"
    assert "private-fixture-key" not in json.dumps(result.detail)


def test_exhausted_source_provider_retries_remain_unscored(judged_corpus, judge_http):
    _, cases = judged_corpus
    state, connection = judge_http
    state["status"] = 503
    state["response"] = {"error": {"message": "temporary overload"}}
    result = grade_remote_judge_answer(cases[0].task, judged_attempt(cases[0].task, "candidate"), connection)
    assert result.status is Outcome.INFRA_ERROR
    assert result.reward is None
    assert len(state["requests"]) == 3
    assert [item["status"] for item in result.detail["transport_attempts"]] == [503, 503, 503]


def test_retry_after_beyond_operation_budget_does_not_send_another_request(judged_corpus, judge_http):
    _, cases = judged_corpus
    state, connection = judge_http
    state["status"] = 429
    state["retry_after"] = 60
    state["response"] = {"error": {"message": "rate limit"}}
    task = cases[0].task
    verifier = replace(verifyit_spec(task.grader), request_timeout=0.1, transport_budget=0.1)
    task = task.model_copy(update={"grader": verifyit_package(verifier).grader})
    result = grade_remote_judge_answer(task, judged_attempt(task, "candidate"), connection)
    assert result.status is Outcome.INFRA_ERROR
    assert result.reward is None
    assert len(state["requests"]) == 1
    assert result.detail["transport_attempts"][0]["status"] == 429


def test_simpleqa_mini_preserves_row_identity_across_repeated_reads(tmp_path):
    file = tmp_path / "simpleqa.csv"
    write_simpleqa(
        file,
        [
            {"question": f"Question {i}?", "answer": f"Answer {i}", "metadata": {"topic": "fixture", "id": i}}
            for i in range(600)
        ],
    )
    rows = list(read_simpleqa(StoragePath(str(file)), ConversionContext({}, None), mini=True))
    assert len(rows) == 500
    assert len({row["source_index"] for row in rows}) == 500
    assert all(
        row["question"] == f"Question {row['source_index']}?" and row["metadata"]["id"] == row["source_index"]
        for row in rows
    )
    assert list(read_simpleqa(StoragePath(str(file)), ConversionContext({}, None), mini=True)) == rows
