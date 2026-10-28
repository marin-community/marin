# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen original judge plans, classifications, HTTP replies and source subset order."""

import csv
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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
from taskcompendium.submission import conversation_messages
from verifyit.modes.grade_judge import JudgeConnection
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.evalchemy.judged import read_simpleqa, sources
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import curate

ORACLE = Path(__file__).parent / "fixtures/evalchemy_judged_oracle.json"


def write_simpleqa(file, rows):
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metadata", "problem", "answer"))
        writer.writeheader()
        writer.writerows(
            {"problem": row["question"], "answer": row["answer"], "metadata": repr(row["metadata"])} for row in rows
        )


@pytest.fixture(params=["FinanceBench", "SimpleQA"])
def judged_corpus(request, tmp_path, monkeypatch):
    reference = next(case for case in json.loads(ORACLE.read_text())["cases"] if case["name"] == request.param)
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


def test_judged_source_preserves_solver_and_judge_requests(judged_corpus, judge_http):
    reference, cases = judged_corpus
    state, connection = judge_http
    for case, original in zip(cases, reference["specs"], strict=True):
        assert conversation_messages(case.task.context.events) == original["requests"][0]["prompt"]
    state["response"] = reference["http_cases"][0]["response"]
    for recorded in reference["judge_plans"]:
        result = grade_remote_judge_answer(
            cases[0].task, judged_attempt(cases[0].task, recorded["candidate"]), connection
        )
        assert result.status is Outcome.GRADED
        assert state["requests"][-1]["messages"] == [
            {"role": "user", "content": recorded["plan"]["requests"][0]["prompt"]}
        ]
        assert (
            state["requests"][-1]["max_completion_tokens"]
            == recorded["plan"]["requests"][0]["generation"]["max_gen_toks"]
        )
        assert result.detail["requested_model"] == state["requests"][-1]["model"]
        assert state["requests"][-1]["model"] == (
            "openai/gpt-oss-120b" if reference["name"] == "FinanceBench" else "gpt-4o-mini"
        )
        assert result.detail["response_model"] == state["response"]["model"]


def test_judge_classifications_and_length_replies_match_original_transport(judged_corpus, judge_http):
    reference, cases = judged_corpus
    state, connection = judge_http
    for recorded in reference["http_cases"]:
        state["response"] = recorded["response"]
        before = len(state["requests"])
        result = grade_remote_judge_answer(
            cases[0].task, judged_attempt(cases[0].task, "candidate with extra details"), connection
        )
        if "error" in recorded:
            assert result.status is Outcome.INFRA_ERROR
            assert result.reward is None
            assert result.detail["completion"] == recorded["response"]["choices"][0]["message"]["content"]
            assert len(state["requests"]) == before + 1
        else:
            assert result.status is Outcome.GRADED
            assert result.detail["metrics"] == recorded["grade"]["metrics"]
            assert result.detail["finish_reason"] == recorded["response"]["choices"][0]["finish_reason"]
    for recorded in reference["judge_grades"]:
        state["response"] = {
            **reference["http_cases"][0]["response"],
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": recorded["reply"]}, "finish_reason": "stop"}
            ],
        }
        result = grade_remote_judge_answer(
            cases[0].task, judged_attempt(cases[0].task, "candidate with extra details"), connection
        )
        if "error" in recorded:
            assert result.status is Outcome.INFRA_ERROR
            assert result.reward is None
        else:
            assert result.detail["metrics"] == recorded["grade"]["metrics"]


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
    state["responses"] = [(503, failure, 0.25), (200, reference["http_cases"][0]["response"], None)]
    result = grade_remote_judge_answer(cases[0].task, judged_attempt(cases[0].task, "candidate"), connection)
    assert result.status is Outcome.GRADED
    assert result.detail["metrics"] == reference["http_cases"][0]["grade"]["metrics"]
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
    verifier = replace(verifyit_spec(task.grader), transport_budget=0.1)
    task = task.model_copy(update={"grader": verifyit_package(verifier).grader})
    result = grade_remote_judge_answer(task, judged_attempt(task, "candidate"), connection)
    assert result.status is Outcome.INFRA_ERROR
    assert result.reward is None
    assert len(state["requests"]) == 1
    assert result.detail["transport_attempts"][0]["status"] == 429


def test_simpleqa_mini_preserves_original_row_identity_and_sample_order(tmp_path):
    reference = next(case for case in json.loads(ORACLE.read_text())["cases"] if case["name"] == "SimpleQAMini")
    file = tmp_path / "simpleqa.csv"
    write_simpleqa(
        file,
        [
            {"question": f"Question {i}?", "answer": f"Answer {i}", "metadata": {"topic": "fixture", "id": i}}
            for i in range(600)
        ],
    )
    rows = list(read_simpleqa(StoragePath(str(file)), ConversionContext({}, None), mini=True))
    assert [row["source_index"] for row in rows] == reference["mini_source_indices"]
    assert [row["question"] for row in rows[:2]] == [row["question"] for row in reference["rows"]]
