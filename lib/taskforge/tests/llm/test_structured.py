# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace
from enum import StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, StringConstraints, field_validator

from taskforge.llm.client import StructuredResult
from taskforge.llm.glm_client import FixedEndpoint, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError, schema_instruction

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


def test_valid_reply_returns_typed_value_and_requests_the_schema(fake_glm, run):
    fake_glm.stream(content=PARIS)
    result = run()
    assert result.value == Capital(city="Paris", millions=2.1)
    response_format = fake_glm.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Capital"
    assert response_format["json_schema"]["schema"] == Capital.model_json_schema()
    assert "tools" not in fake_glm.requests[0]


def test_first_request_ends_with_a_user_turn_describing_each_field(fake_glm, run):
    fake_glm.stream(content=PARIS)
    run()
    *messages, instruction = fake_glm.requests[0]["messages"]
    assert messages == MESSAGES
    assert instruction["role"] == "user"
    for name, field in Capital.model_fields.items():
        assert name in instruction["content"]
        assert field.description is not None and field.description in instruction["content"]


def test_invalid_reply_is_repaired_once_keeping_the_prior_output(fake_glm, run):
    fake_glm.stream(content='{"city": "Paris", "millions": "lots"}')
    fake_glm.stream(content=PARIS)
    result = run()
    assert result.value.millions == 2.1
    assert len(result.completions) == 2
    *messages, assistant, repair, instruction = fake_glm.requests[1]["messages"]
    assert messages == MESSAGES
    assert assistant == {"role": "assistant", "content": '{"city": "Paris", "millions": "lots"}'}
    assert "millions" in repair["content"]
    assert instruction == fake_glm.requests[0]["messages"][-1]


def test_validator_failure_is_repaired_with_its_error_text(fake_glm, run):
    fake_glm.stream(content='{"city": "Paris", "millions": -1}')
    fake_glm.stream(content=PARIS)
    result = run()
    assert result.value.millions == 2.1
    assert POSITIVE_POPULATION_ERROR in fake_glm.requests[1]["messages"][-2]["content"]


def test_second_failure_raises_with_both_completions(fake_glm, run):
    fake_glm.stream(content="not json")
    fake_glm.stream(content="still not json")
    with pytest.raises(StructuredOutputError) as error:
        run()
    assert [c.content for c in error.value.completions] == ["not json", "still not json"]


def test_length_cut_first_reply_raises_without_a_repair(fake_glm, run):
    fake_glm.stream(content=PARIS, finish="length")
    fake_glm.stream(content=PARIS)
    with pytest.raises(StructuredOutputError) as error:
        run()
    assert len(fake_glm.requests) == 1
    assert len(error.value.completions) == 1
    assert error.value.completions[0].continuations == 0


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


class Ingredient(BaseModel):
    name: str = Field(description="What to buy.")
    grams: float | None = None


class Recipe(BaseModel):
    """A recipe for the dish."""

    title: str = Field(description="The dish's name.")
    ingredients: list[Ingredient] = Field(description="What goes into the dish.")
    steps: list[str]


def test_schema_instruction_renders_nested_models_and_lists():
    assert schema_instruction(Recipe) == (
        "Reply with one JSON object with these fields:\n"
        "A recipe for the dish.\n"
        "title (string, required): The dish's name.\n"
        "ingredients (list of object, required): What goes into the dish.\n"
        "  name (string, required): What to buy.\n"
        "  grams (number or null, optional): Grams\n"
        "steps (list of string, required): Steps"
    )


class Severity(StrEnum):
    LOW = "low"
    HIGH = "high"


class Review(BaseModel):
    notes: str = Field(max_length=1000, description="Free-form notes.")
    score: int = Field(ge=1, le=5, description="The overall score.")
    confidence: float = Field(gt=0, lt=1)
    tags: list[Annotated[str, StringConstraints(max_length=20)]] = Field(min_length=1, max_length=5)
    verdict: Literal["accept", "reject"]
    severity: Severity | None = None
    ticket: str = Field(pattern=r"^[A-Z]+-[0-9]+$")


def test_schema_instruction_renders_each_field_constraint():
    assert schema_instruction(Review) == (
        "Reply with one JSON object with these fields:\n"
        "notes (string, at most 1000 characters, required): Free-form notes.\n"
        "score (integer, at least 1, at most 5, required): The overall score.\n"
        "confidence (number, greater than 0, less than 1, required): Confidence\n"
        "tags (list of string, at least 1 item, at most 5 items, each at most 20 characters, required): Tags\n"
        'verdict (string, one of: "accept", "reject", required): Verdict\n'
        'severity (string or null, one of: "low", "high", optional)\n'
        "ticket (string, matching ^[A-Z]+-[0-9]+$, required): Ticket"
    )


class Location(BaseModel):
    """Where the file goes."""

    path: str = Field(description="One of the task's output paths, spelled absolutely.")


class OutputFile(BaseModel):
    """A file the draft writes."""

    location: Location
    text: str = Field(max_length=500, description="The file's contents.")


class Draft(BaseModel):
    """A draft of the change."""

    summary: str = Field(description="What the draft changes.")
    files: list[OutputFile] = Field(min_length=1, description="The files to write.")


def test_schema_instruction_renders_models_nested_two_levels():
    assert schema_instruction(Draft) == (
        "Reply with one JSON object with these fields:\n"
        "A draft of the change.\n"
        "summary (string, required): What the draft changes.\n"
        "files (list of object, at least 1 item, required): The files to write.\n"
        "  A file the draft writes.\n"
        "  location (object, required)\n"
        "    Where the file goes.\n"
        "    path (string, required): One of the task's output paths, spelled absolutely.\n"
        "  text (string, at most 500 characters, required): The file's contents."
    )


class Section(BaseModel):
    """A section of the outline."""

    heading: str
    parent: "Section | None" = None
    subsections: list["Section"] = Field(default_factory=list)


def test_schema_instruction_renders_a_self_referential_model_once():
    assert schema_instruction(Section) == (
        "Reply with one JSON object with these fields:\n"
        "A section of the outline.\n"
        "heading (string, required): Heading\n"
        "parent (object or null, optional)\n"
        "  Section (same fields as above)\n"
        "subsections (list of object, optional): Subsections\n"
        "  Section (same fields as above)"
    )
