# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import GradingInfraError
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.abstention_sft.probe import sample_correct

PROBLEM = {
    "request_id": "law-wiki-00040",
    "problem": "Which publisher released R. L. Maddex's 2008 work 'Constitutions of the World'?",
    "answer": "CQ Press",
}
CLOSED = "<|start_think|>I recall the publisher.<|end_think|>CQ Press"


@pytest.fixture
def connection(chat_server):
    return JudgeConnection(f"http://127.0.0.1:{chat_server.server_port}/v1", "fixture-key", default_model="judge")


def test_truncated_sample_is_wrong_without_a_judge_call(chat_server, connection):
    assert sample_correct(PROBLEM, "length", CLOSED, connection) is False
    assert chat_server.requests == []


def test_unclosed_think_is_wrong_without_a_judge_call(chat_server, connection):
    assert sample_correct(PROBLEM, "stop", "<|start_think|>Still thinking about publishers", connection) is False
    assert chat_server.requests == []


def test_judge_correct_label_counts_as_correct(chat_server, connection):
    chat_server.replies = ["A"]
    assert sample_correct(PROBLEM, "stop", CLOSED, connection) is True
    prompt = chat_server.requests[-1]["messages"][-1]["content"]
    assert PROBLEM["problem"] in prompt and "CQ Press" in prompt
    assert "I recall the publisher." not in prompt


def test_judge_incorrect_label_counts_as_wrong(chat_server, connection):
    chat_server.replies = ["B"]
    assert sample_correct(PROBLEM, "stop", CLOSED, connection) is False


def test_judge_failure_raises(chat_server, connection):
    chat_server.replies = ["The answer looks right to me."]
    with pytest.raises(GradingInfraError):
        sample_correct(PROBLEM, "stop", CLOSED, connection)
