# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``complete_prefilled`` against the scripted GLM server."""

import pytest
from rigging.timing import ExponentialBackoff

from taskforge.llm.client import FinishReason, GlmClient, GlmEndpoint, Pool, complete_prefilled
from taskforge.llm.policy import LLMPolicy, ReasoningEffort

MESSAGES = [{"role": "user", "content": "propose a task"}]
PREFIX = "---\nid:"


@pytest.fixture
async def client(fake_glm):
    endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test", pool=Pool.HIGH)
    async with GlmClient(endpoint, backoff=ExponentialBackoff(initial=0.001, maximum=0.001)) as glm:
        yield glm


async def test_prefix_is_sent_as_an_open_assistant_turn_without_thinking(fake_glm, client):
    fake_glm.stream(content=" csv-merge\ntitle: Merge CSVs\n---\n")
    policy = LLMPolicy(reasoning_effort=ReasoningEffort.MEDIUM)

    completion = await complete_prefilled(client, MESSAGES, policy, PREFIX, {"tools": []})

    assert completion.content == "---\nid: csv-merge\ntitle: Merge CSVs\n---\n"
    request = fake_glm.requests[0]
    assert request["messages"] == [*MESSAGES, {"role": "assistant", "content": PREFIX}]
    assert (request["continue_final_message"], request["add_generation_prompt"]) == (True, False)
    assert request["chat_template_kwargs"] == {"reasoning_effort": "medium", "enable_thinking": False}
    assert request["tools"] == []


async def test_cut_reply_continues_by_prefilling_everything_written(fake_glm, client):
    fake_glm.stream(content=" a\ntitle: B\n", finish="length", completion_tokens=6)
    fake_glm.stream(content="\n---\nbody", completion_tokens=3)

    completion = await complete_prefilled(client, MESSAGES, LLMPolicy(max_continuations=1), PREFIX)

    assert fake_glm.requests[1]["messages"][-1] == {"role": "assistant", "content": "---\nid: a\ntitle: B"}
    assert completion.content == "---\nid: a\ntitle: B\n---\nbody"
    assert completion.finish_reason is FinishReason.STOP
    assert (completion.continuations, completion.usage.completion_tokens) == (1, 9)
    assert [a.segment for a in completion.attempts] == [0, 1]


async def test_cut_reply_stops_after_the_policy_continuations(fake_glm, client):
    fake_glm.stream(content=" a", finish="length")

    completion = await complete_prefilled(client, MESSAGES, LLMPolicy(max_continuations=0), PREFIX)

    assert (completion.content, completion.finish_reason) == ("---\nid: a", FinishReason.LENGTH)
    assert len(fake_glm.requests) == 1
