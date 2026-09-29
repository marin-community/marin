# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GraphWalks' published scoring contract and durable evaluation output."""

import json

import datasets
import pytest
import requests
import transformers
from finestore.eval import sample_from_archive_row
from finestore.reader import ReadView
from marin.evaluation.graphwalks import GraphWalksExecutor, grade_answer
from marin.inference.types import OpenAIEndpoint, RunningModel


def test_graphwalks_grades_unordered_sets_and_empty_answers():
    grade = grade_answer("Reasoning\nFinal Answer: [b, a, b]", ("a", "b"))
    assert grade.extracted == ["b", "a", "b"]
    assert not grade.failed_to_parse
    assert grade.scores == {"f1": 1.0, "precision": 1.0, "recall": 1.0, "exact_match": 1.0}

    grade = grade_answer("Final Answer: []", ())
    assert grade.extracted == []
    assert not grade.failed_to_parse
    assert grade.scores["f1"] == 1.0


def test_graphwalks_requires_answer_on_last_line():
    grade = grade_answer("Final Answer: [a]\nThanks!", ("a",))
    assert grade.failed_to_parse
    assert grade.scores["f1"] == 0.0
    assert grade.scores["exact_match"] == 0.0


@pytest.mark.parametrize("drop_connection", [False, True])
@pytest.mark.parametrize("mapped_tokens", [False, True])
def test_graphwalks_records_scored_sample_and_context_coverage(tmp_path, monkeypatch, drop_connection, mapped_tokens):
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            return list(text)

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            tokens = list(messages[0]["content"])
            if mapped_tokens:
                return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
            return tokens

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Reasoning\nFinal Answer: [b, a]"}, "finish_reason": "stop"}]}

    rows = [
        {
            "prompt": "Find parents",
            "answer_nodes": ["a", "b"],
            "prompt_chars": 12,
            "problem_type": "parents",
            "date_added": "02-27-2026",
        },
        {
            "prompt": "x" * 6000,
            "answer_nodes": ["a"],
            "prompt_chars": 6000,
            "problem_type": "bfs",
            "date_added": "02-27-2026",
        },
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())

    class Session:
        model = RunningModel(endpoint=OpenAIEndpoint(base_url="http://localhost/v1", model="test"), tokenizer="test")

        def __init__(self):
            self.ready = not drop_connection

        def wait_until_ready(self):
            self.ready = True

    session = Session()

    def post(*args, **kwargs):
        if not session.ready:
            raise requests.ConnectionError("serving endpoint preempted")
        assert kwargs["json"]["max_tokens"] == 8232
        return Response()

    monkeypatch.setattr(requests, "post", post)

    root = str(tmp_path / "run")
    outcome = GraphWalksExecutor(max_model_len=9000)(session, root, {})

    assert outcome.coverage["graphwalks"].n_benchmark == 2
    assert outcome.coverage["graphwalks"].n_attempted == 1
    assert outcome.coverage["graphwalks"].n_scored == 1
    assert outcome.metrics["graphwalks"]["skipped_context"] == 1.0
    assert outcome.canonical_metrics["graphwalks"]["f1"] == 1.0
    [row] = ReadView(root).scan("samples").to_pylist(maps_as_pydicts="strict")
    sample = sample_from_archive_row(row)
    assert sample.grading.score == 1.0
    assert sample.prompt_messages[0].content == "Find parents"
    assert json.loads(sample.doc)["finish_reason"] == "stop"
