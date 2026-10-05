# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from pydantic import BaseModel
from rigging.timing import ExponentialBackoff

from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.structured import StructuredOutputError, StructuredResult, StructuredTool, complete_structured


class Capital(BaseModel):
    city: str
    millions: float


TOOL = StructuredTool(name="answer", description="Record the answer", output_type=Capital)
MESSAGES = [{"role": "user", "content": "Capital of France?"}]


def run(fake_glm) -> StructuredResult[Capital]:
    async def go() -> StructuredResult[Capital]:
        endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test-token", pool=Pool.HIGH)
        async with GlmClient(endpoint, backoff=ExponentialBackoff(initial=0.001, maximum=0.001)) as client:
            return await complete_structured(client, MESSAGES, LLMPolicy(), TOOL)

    return asyncio.run(go())


def test_valid_call_returns_typed_value_and_forces_the_tool(fake_glm):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris", "millions": 2.1}'),))
    result = run(fake_glm)
    assert result.value == Capital(city="Paris", millions=2.1)
    assert fake_glm.requests[0]["tool_choice"] == {"type": "function", "function": {"name": "answer"}}


def test_invalid_call_is_repaired_once_keeping_the_prior_output(fake_glm):
    fake_glm.stream(content="Here it is.", tool_calls=(("answer", '{"city": "Paris", "millions": "lots"}'),))
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris", "millions": 2.1}'),))
    result = run(fake_glm)
    assert result.value.millions == 2.1
    assert len(result.completions) == 2
    assistant, user = fake_glm.requests[1]["messages"][-2:]
    assert assistant == {
        "role": "assistant",
        "content": 'Here it is.\n[answer arguments]\n{"city": "Paris", "millions": "lots"}',
    }
    assert "millions" in user["content"]


def test_second_failure_raises_with_both_completions(fake_glm):
    fake_glm.stream(content="no tool call")
    fake_glm.stream(content="still none")
    with pytest.raises(StructuredOutputError) as error:
        run(fake_glm)
    assert [c.content for c in error.value.completions] == ["no tool call", "still none"]
