# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise a published SimpleQA corpus through the judge boundary."""

import csv
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.grading import grade_remote_judge_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.evalchemy.judge_prompts import SIMPLEQA_VERIFIED_GRADER_TEMPLATE
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.submission import conversation_messages
from verifyit.modes.grade_judge import JudgeConnection
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.simpleqa import simpleqa_source, simpleqa_verified_source
from experiments.evaluation.preparation import prepare_evaluation
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions


def write_simpleqa(file, rows):
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metadata", "problem", "answer"))
        writer.writeheader()
        writer.writerows(
            {"problem": row["question"], "answer": row["answer"], "metadata": repr(row["metadata"])} for row in rows
        )


def published_cases(source, write_rows, tmp_path, monkeypatch, population):
    dataset = source.config
    root = tmp_path / "source"
    file = root / dataset.files[0]
    file.parent.mkdir(parents=True)
    write_rows(file)
    source = replace(
        source,
        config=replace(dataset, protocol=dataset.protocol.model_copy(update={"expected_population": population})),
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
            artifact = run(prepare_evaluation(source, options), max_concurrent=1)[0]
        return list(evaluation_cases(artifact.path))
    finally:
        client.shutdown()


@pytest.fixture
def judged_corpus(tmp_path, monkeypatch):
    reference = {
        "name": "SimpleQA",
        "rows": [{"question": "What is the revenue?", "answer": "$7", "evidence_text": "Revenue: $7", "metadata": {}}],
    }
    source = simpleqa_source("SimpleQA", mini=False)
    cases = published_cases(
        source,
        lambda file: write_simpleqa(file, reference["rows"]),
        tmp_path,
        monkeypatch,
        len(reference["rows"]),
    )
    return reference, cases


@pytest.fixture
def judge_http(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    state = {"response": {}, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["requests"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(state["response"]).encode()
            self.send_response(200)
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


def test_simpleqa_verified_publishes_bare_question_and_judges_only_the_final_answer(tmp_path, monkeypatch, judge_http):
    fields = ("original_index", "problem", "answer", "topic", "answer_type", "multi_step", "requires_reasoning", "urls")

    def write_rows(file):
        with file.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                dict(zip(fields, ("3", "Who wrote it?", "Ada", "Art", "Person", "False", "False", "u"), strict=True))
            )

    (case,) = published_cases(simpleqa_verified_source(thinking=False), write_rows, tmp_path, monkeypatch, 1)
    assert conversation_messages(case.task.context.events) == [{"role": "user", "content": "Who wrote it?"}]
    assert case.item.operation.response_selection == "final_only"
    assert case.item.groups == {"source_index": "0"}
    state, connection = judge_http
    state["response"] = judge_response("SimpleQAVerified")
    # The launcher supplies the judge deployment's default model.
    connection = replace(connection, default_model="fixture-judge")
    result = grade_remote_judge_answer(case.task, judged_attempt(case.task, "It was Ada."), connection)
    assert result.status is Outcome.GRADED
    assert result.reward == 1
    assert state["requests"][-1]["model"] == "fixture-judge"
    assert state["requests"][-1]["messages"][-1]["content"] == SIMPLEQA_VERIFIED_GRADER_TEMPLATE.format(
        question="Who wrote it?", target="Ada", predicted_answer="It was Ada."
    )
