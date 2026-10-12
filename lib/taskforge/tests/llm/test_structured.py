# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from pydantic import BaseModel

from taskforge.llm.client import StructuredResult
from taskforge.llm.glm_client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError, StructuredTool, complete_structured


class Capital(BaseModel):
    """Record the answer"""

    city: str
    millions: float


TOOL = StructuredTool(name="answer", description="Record the answer", output_type=Capital)
MESSAGES: list[Message] = [{"role": "user", "content": "Capital of France?"}]


@pytest.fixture
def run(fake_glm, fast_provider):
    def go(max_repairs=1) -> StructuredResult[Capital]:
        async def call() -> StructuredResult[Capital]:
            endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test-token", pool=Pool.HIGH)
            async with GlmClient(endpoint, fast_provider) as client:
                return await complete_structured(client, MESSAGES, TOOL, LLMPolicy(), max_repairs)

        return asyncio.run(call())

    return go


def test_valid_call_returns_typed_value_and_forces_the_tool(fake_glm, run):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris", "millions": 2.1}'),))
    result = run()
    assert result.value == Capital(city="Paris", millions=2.1)
    assert fake_glm.requests[0]["tool_choice"] == {"type": "function", "function": {"name": "answer"}}


def test_invalid_call_is_repaired_once_keeping_the_prior_output(fake_glm, run):
    fake_glm.stream(content="Here it is.", tool_calls=(("answer", '{"city": "Paris", "millions": "lots"}'),))
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris", "millions": 2.1}'),))
    result = run()
    assert result.value.millions == 2.1
    assert len(result.completions) == 2
    assistant, user = fake_glm.requests[1]["messages"][-2:]
    assert assistant == {
        "role": "assistant",
        "content": 'Here it is.\n[answer arguments]\n{"city": "Paris", "millions": "lots"}',
    }
    assert "millions" in user["content"]


def test_second_failure_raises_with_both_completions(fake_glm, run):
    fake_glm.stream(content="no tool call")
    fake_glm.stream(content="still none")
    with pytest.raises(StructuredOutputError) as error:
        run()
    assert [c.content for c in error.value.completions] == ["no tool call", "still none"]


def test_valid_but_length_cut_call_is_repaired(fake_glm, run):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris", "millions": 2.1}'),), finish="length")
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris", "millions": 2.14}'),))
    result = run()
    assert result.value.millions == 2.14
    assert "cut off" in fake_glm.requests[1]["messages"][-1]["content"]


def test_length_cut_repair_raises(fake_glm, run):
    fake_glm.stream(content="no tool call")
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris", "millions": 2.1}'),), finish="length")
    with pytest.raises(StructuredOutputError):
        run()


def test_repairs_stop_at_max_repairs(fake_glm, run):
    for reply in ("none", "still none", "and again"):
        fake_glm.stream(content=reply)
    with pytest.raises(StructuredOutputError) as error:
        run(max_repairs=2)
    assert [c.content for c in error.value.completions] == ["none", "still none", "and again"]


def test_client_method_names_the_tool_and_describes_it_with_the_output_docstring(fake_glm, fast_provider):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris", "millions": 2.1}'),))

    async def go() -> StructuredResult[Capital]:
        endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test-token", pool=Pool.HIGH)
        async with GlmClient(endpoint, fast_provider) as client:
            return await client.complete_structured(MESSAGES, Capital, name="answer", policy=LLMPolicy())

    assert asyncio.run(go()).value.city == "Paris"
    function = fake_glm.requests[0]["tools"][0]["function"]
    assert (function["name"], function["description"]) == ("answer", "Record the answer")
