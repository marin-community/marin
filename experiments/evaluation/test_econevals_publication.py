# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish an EconEvals corpus and grade it pairwise through the judge boundary."""

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
from taskcompendium.importers.econevals import (
    BASELINE_FILENAME,
    BASELINE_INPUT,
    CATEGORIES_FILENAME,
    CATEGORIES_INPUT,
    JUDGE_SYSTEM_PROMPT,
)
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from verifyit.modes.grade_judge import JudgeConnection
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.econevals import O3_BASELINE, RUNS, econevals_source
from experiments.evaluation.preparation import prepare_evaluation
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions

QUESTIONS = {"7": "Draft a shift schedule for six nurses.", "9": "Summarize this {quarterly} variance report."}
CATEGORIES = {"subcategory_idx": {"0": 21, "1": 89}, "subcategory_name": {"0": "Assign duties", "1": "Analyze data"}}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def econevals_cases(tmp_path, monkeypatch):
    source = econevals_source(RUNS[0], O3_BASELINE, per_category=None)
    dataset = source.config
    root, baseline, categories = tmp_path / "source", tmp_path / "baseline", tmp_path / "categories"
    write_jsonl(
        root / dataset.files[0],
        [
            {"uid": uid, "category": "ei", "subcategory": subcategory, "convo": [{"role": "user", "content": text}]}
            for (uid, text), subcategory in zip(QUESTIONS.items(), ("21", "89"), strict=True)
        ],
    )
    write_jsonl(
        baseline / BASELINE_FILENAME,
        [
            {
                "uid": uid,
                "ans_id": f"{O3_BASELINE}:{uid}",
                "model": O3_BASELINE,
                "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": f"baseline {uid}"}],
            }
            for uid, text in QUESTIONS.items()
        ],
    )
    categories.mkdir()
    (categories / CATEGORIES_FILENAME).write_text(json.dumps(CATEGORIES))
    source = replace(
        source,
        config=replace(dataset, protocol=dataset.protocol),
    )
    runtime = CampaignRuntime()
    overrides = InputOverrides(
        root=str(root), auxiliary={BASELINE_INPUT: str(baseline), CATEGORIES_INPUT: str(categories)}
    )
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime, inputs=overrides)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            artifact = run(prepare_evaluation(source, options), max_concurrent=1)[0]
        yield list(evaluation_cases(artifact.path))
    finally:
        client.shutdown()


@pytest.fixture
def judge_http(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(
                {
                    "id": "fixture-judge",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fixture-revision",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Explanation.\n\nMy final verdict: [[A>B]]"},
                        }
                    ],
                }
            ).encode()
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
        yield requests, JudgeConnection(
            f"http://127.0.0.1:{server.server_port}/v1", "private-fixture-key", default_model="fixture-judge"
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_published_cases_keep_question_order_and_category_groups(econevals_cases):
    assert [case.task.context.events[0].content for case in econevals_cases] == list(QUESTIONS.values())
    assert [case.item.groups for case in econevals_cases] == [
        {"category": "Assign duties"},
        {"category": "Analyze data"},
    ]


def test_pairwise_grade_judges_both_orders_with_the_source_prompt(econevals_cases, judge_http):
    requests, connection = judge_http
    task = econevals_cases[1].task
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="<think>x</think>mine")))
    )
    result = grade_remote_judge_answer(task, attempt, connection)
    # Position A wins both games, so the candidate wins one of two.
    assert (result.status, result.reward) == (Outcome.GRADED, 0.5)
    assert result.detail["metrics"] == {"win_rate": 0.5}
    assert result.detail["metric_weights"] == {"win_rate": 2.0}
    assert [request["messages"][0] for request in requests] == [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}] * 2
    prompts = [request["messages"][1]["content"] for request in requests]
    assert "<|The Start of the User Prompt|>\nSummarize this {quarterly} variance report.\n" in prompts[0]
    assert "Assistant A's Answer|>\nbaseline 9\n" in prompts[0]
    assert "Assistant B's Answer|>\nmine\n" in prompts[0]
    assert "Assistant A's Answer|>\nmine\n" in prompts[1]
    assert "Assistant B's Answer|>\nbaseline 9\n" in prompts[1]
