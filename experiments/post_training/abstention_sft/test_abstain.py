# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from openai import OpenAI

from experiments.post_training.abstention_sft.abstain import abstain_reply, leaks_answer

QUESTION = "Which publisher released R. L. Maddex's 2008 work 'Constitutions of the World'?"


@pytest.fixture
def client(chat_server):
    return OpenAI(base_url=f"http://127.0.0.1:{chat_server.server_port}/v1", api_key="fixture-key")


def test_leaks_answer_ignores_case_and_punctuation():
    assert leaks_answer("I think it's CQ-Press.", "CQ Press")
    assert not leaks_answer("I don't know which publisher released that book.", "CQ Press")


def test_reply_that_leaks_is_rewritten(chat_server, client):
    clean = "I don't know which publisher released that 2008 edition. A library catalogue would say."
    chat_server.replies = ["It was probably CQ Press.", clean]
    assert abstain_reply(QUESTION, "CQ Press", client, "writer") == clean
    assert len(chat_server.requests) == 2


def test_reply_that_always_leaks_raises(chat_server, client):
    chat_server.replies = ["CQ Press."]
    with pytest.raises(ValueError):
        abstain_reply(QUESTION, "CQ Press", client, "writer", attempts=2)


def test_writer_prompt_holds_the_question(chat_server, client):
    chat_server.replies = ["I don't know which publisher released it."]
    abstain_reply(QUESTION, "CQ Press", client, "writer")
    assert QUESTION in chat_server.requests[0]["messages"][-1]["content"]
    assert "CQ Press" not in chat_server.requests[0]["messages"][-1]["content"]
