# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``GlmClient.complete_structured`` against a scripted GLM server: schema request, repairs and their budget.

Invariants: none (pins structured-output repair and its budget).
"""

import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel, Field, field_validator
from taskforge_adapters.glm_client import FixedEndpoint, GlmClient, GlmEndpoint, Pool

from taskforge.llm.client import StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import IncompleteOutput, StructuredOutputError

POSITIVE_POPULATION_ERROR = "a capital's population must be positive"


class Capital(BaseModel):
    """The capital of the country in question."""

    city: str = Field(description="The capital's name in English.")
    millions: float = Field(description="Its population in millions.")

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


def test_valid_reply_returns_typed_value_and_requests_the_schema_with_a_user_turn_describing_each_field(fake_glm, run):
    fake_glm.stream(content=PARIS)
    result = run()
    assert result.value == Capital(city="Paris", millions=2.1)
    response_format = fake_glm.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Capital"
    assert response_format["json_schema"]["schema"] == Capital.model_json_schema()
    assert "tools" not in fake_glm.requests[0]
    *messages, instruction = fake_glm.requests[0]["messages"]
    assert messages == MESSAGES
    assert instruction["role"] == "user"
    for name, field in Capital.model_fields.items():
        assert name in instruction["content"]
        assert field.description is not None and field.description in instruction["content"]


@pytest.mark.parametrize(
    ("invalid", "named"),
    [
        ('{"city": "Paris", "millions": "lots"}', "millions"),
        ('{"city": "Paris", "millions": -1}', POSITIVE_POPULATION_ERROR),
    ],
    ids=["schema", "validator"],
)
def test_invalid_reply_is_repaired_once_keeping_the_prior_output_and_naming_the_error(fake_glm, run, invalid, named):
    fake_glm.stream(content=invalid)
    fake_glm.stream(content=PARIS)
    result = run()
    assert result.value.millions == 2.1
    assert len(result.completions) == 2
    *messages, assistant, repair, instruction = fake_glm.requests[1]["messages"]
    assert messages == MESSAGES
    assert assistant == {"role": "assistant", "content": invalid}
    assert named in repair["content"]
    assert instruction == fake_glm.requests[0]["messages"][-1]


@pytest.mark.parametrize(
    ("repairs", "replies"), [(1, ("not json", "still not json")), (2, ("none", "still none", "and again"))]
)
def test_repairs_stop_at_the_provider_budget(fake_glm, run, repairs, replies):
    for reply in replies:
        fake_glm.stream(content=reply)
    with pytest.raises(StructuredOutputError) as error:
        run(structured_repairs=repairs)
    assert [c.content for c in error.value.completions] == list(replies)


@pytest.mark.parametrize("cut", [0, 1], ids=["first-reply", "repair"])
def test_a_length_cut_reply_raises_without_a_further_repair(fake_glm, run, cut):
    replies = ["not json"] * cut + [PARIS]
    for index, reply in enumerate(replies):
        fake_glm.stream(content=reply, finish="length" if index == cut else "stop")
    fake_glm.stream(content=PARIS)
    with pytest.raises(IncompleteOutput) as error:
        run()
    assert len(fake_glm.requests) == cut + 1
    assert len(error.value.completions) == cut + 1
    assert error.value.completions[-1].continuations == 0


def test_unlimited_schema_repairs_reach_a_valid_reply_after_repeated_failures(fake_glm, run):
    for _ in range(4):
        fake_glm.stream(content="not json")
    fake_glm.stream(content=PARIS)
    result = run(structured_repairs=None)
    assert result.value.city == "Paris"
    assert len(result.completions) == 5
