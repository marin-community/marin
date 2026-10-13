# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel, field_validator

from taskforge.llm.client import StructuredResult
from taskforge.llm.glm_client import FixedEndpoint, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError

POSITIVE_POPULATION_ERROR = "a capital's population must be positive"


class Capital(BaseModel):
    """Record the answer"""

    city: str
    millions: float

    @field_validator("millions")
    @classmethod
    def positive(cls, millions: float) -> float:
        if millions <= 0:
            raise ValueError(POSITIVE_POPULATION_ERROR)
        return millions


MESSAGES: list[Message] = [{"role": "user", "content": "Capital of France?"}]
PARIS = '{"city": "Paris", "millions": 2.1}'


@pytest.fixture
def run(fake_glm, fast_provider):
    def go(structured_repairs=1) -> StructuredResult[Capital]:
        async def call() -> StructuredResult[Capital]:
            source = FixedEndpoint(GlmEndpoint(base_url=fake_glm.base_url, token="test-token", pool=Pool.HIGH))
            async with GlmClient(source, replace(fast_provider, structured_repairs=structured_repairs)) as client:
                return await client.complete_structured(MESSAGES, Capital, LLMPolicy())

        return asyncio.run(call())

    return go


def test_valid_reply_returns_typed_value_and_requests_the_schema(fake_glm, run):
    fake_glm.stream(content=PARIS)
    result = run()
    assert result.value == Capital(city="Paris", millions=2.1)
    response_format = fake_glm.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Capital"
    assert response_format["json_schema"]["schema"] == Capital.model_json_schema()
    assert "tools" not in fake_glm.requests[0]


def test_invalid_reply_is_repaired_once_keeping_the_prior_output(fake_glm, run):
    fake_glm.stream(content='{"city": "Paris", "millions": "lots"}')
    fake_glm.stream(content=PARIS)
    result = run()
    assert result.value.millions == 2.1
    assert len(result.completions) == 2
    assistant, user = fake_glm.requests[1]["messages"][-2:]
    assert assistant == {"role": "assistant", "content": '{"city": "Paris", "millions": "lots"}'}
    assert "millions" in user["content"]


def test_validator_failure_is_repaired_with_its_error_text(fake_glm, run):
    fake_glm.stream(content='{"city": "Paris", "millions": -1}')
    fake_glm.stream(content=PARIS)
    result = run()
    assert result.value.millions == 2.1
    assert POSITIVE_POPULATION_ERROR in fake_glm.requests[1]["messages"][-1]["content"]


def test_second_failure_raises_with_both_completions(fake_glm, run):
    fake_glm.stream(content="not json")
    fake_glm.stream(content="still not json")
    with pytest.raises(StructuredOutputError) as error:
        run()
    assert [c.content for c in error.value.completions] == ["not json", "still not json"]


def test_valid_but_length_cut_reply_is_repaired_not_continued(fake_glm, run):
    fake_glm.stream(content=PARIS, finish="length")
    fake_glm.stream(content='{"city": "Paris", "millions": 2.14}')
    result = run()
    assert result.value.millions == 2.14
    assert [c.continuations for c in result.completions] == [0, 0]
    assert "cut off" in fake_glm.requests[1]["messages"][-1]["content"]


def test_length_cut_repair_raises(fake_glm, run):
    fake_glm.stream(content="not json")
    fake_glm.stream(content=PARIS, finish="length")
    with pytest.raises(StructuredOutputError):
        run()


def test_repairs_stop_at_the_provider_budget(fake_glm, run):
    for reply in ("none", "still none", "and again"):
        fake_glm.stream(content=reply)
    with pytest.raises(StructuredOutputError) as error:
        run(structured_repairs=2)
    assert [c.content for c in error.value.completions] == ["none", "still none", "and again"]
