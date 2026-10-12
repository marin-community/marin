# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest
from pydantic import BaseModel

from taskforge.ledger.records import EntryKind
from taskforge.llm.client import Completion, FinishReason, GlmUnavailable, NoModel, StructuredResult, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import RecordedModel

MESSAGES: tuple[Message, ...] = ({"role": "user", "content": "Name a prime."},)


class Answer(BaseModel):
    value: int


def completion(content: str, prompt: int, output: int, reasoning: int, finish: FinishReason) -> Completion:
    return Completion(
        content=content,
        reasoning="",
        tool_calls=(),
        finish_reason=finish,
        usage=Usage(prompt_tokens=prompt, completion_tokens=output, reasoning_tokens=reasoning, cached_tokens=0),
        wall_time=1.0,
        ttft=0.1,
        decode_time=0.9,
        continuations=0,
        attempts=(),
    )


@dataclass(frozen=True)
class ScriptedClient:
    """Answers ``complete`` with "7" and a structured call with ``value`` after a repair (two completions)."""

    model: str = "scripted-model"
    value: int = 7
    unavailable: bool = False

    async def complete(self, messages: Sequence[Message], policy: LLMPolicy) -> Completion:
        if self.unavailable:
            raise GlmUnavailable("hold ran out", ())
        return completion("7", 10, 2, 1, FinishReason.STOP)

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], *, name: str, policy: LLMPolicy
    ) -> StructuredResult[T]:
        completions = (
            completion("", 10, 4, 3, FinishReason.TOOL_CALLS),
            completion("", 20, 5, 0, FinishReason.TOOL_CALLS),
        )
        return StructuredResult(output_type.model_validate({"value": self.value}), completions)


def recorded(client, ledger, step: str = "grader") -> RecordedModel:
    return RecordedModel(client=client, policy=LLMPolicy(), ledger=ledger, item_id="IDEA--0", round=1, step=step)


async def test_a_structured_call_returns_the_value_and_records_one_attributed_span(ledger):
    answer = await recorded(ScriptedClient(), ledger).structured(MESSAGES, Answer, "submit_answer")

    assert answer == Answer(value=7)
    [entry] = ledger.entries
    assert (entry.kind, entry.item_id, entry.round, entry.step) == (EntryKind.LLM_CALL, "IDEA--0", 1, "grader")
    assert (entry.model, entry.tokens_in, entry.tokens_out, entry.tokens_reasoning) == ("scripted-model", 30, 9, 3)
    assert entry.finish_reason == "tool_calls"
    assert entry.attrs == {"tool": "submit_answer", "requests": "2"}
    assert entry.cause is None


async def test_a_completion_records_its_tokens(ledger):
    reply = await recorded(ScriptedClient(), ledger, step="propose").complete(MESSAGES)

    assert reply.content == "7"
    [entry] = ledger.entries
    assert (entry.step, entry.tokens_in, entry.tokens_out, entry.tokens_reasoning) == ("propose", 10, 2, 1)
    assert (entry.finish_reason, entry.attrs) == ("stop", {"requests": "1"})


async def test_an_unavailable_model_propagates_and_is_recorded_as_the_cause(ledger):
    with pytest.raises(GlmUnavailable):
        await recorded(ScriptedClient(unavailable=True), ledger).complete(MESSAGES)

    [entry] = ledger.entries
    assert (entry.cause, entry.model, entry.tokens_in) == ("GlmUnavailable", "scripted-model", None)


def test_the_digest_follows_the_model_and_policy_not_the_attribution(ledger):
    model = recorded(ScriptedClient(), ledger)

    assert replace(model, step="task", round=3, item_id="other").digest == model.digest
    assert replace(model, client=ScriptedClient(model="other-model")).digest != model.digest
    assert replace(model, policy=LLMPolicy(temperature=0.0)).digest != model.digest


async def test_no_model_refuses_every_call(ledger):
    model = recorded(NoModel(), ledger)

    with pytest.raises(RuntimeError, match="no model is configured"):
        await model.complete(MESSAGES)
    with pytest.raises(RuntimeError, match="no model is configured"):
        await model.structured(MESSAGES, Answer, "submit_answer")
    assert [entry.cause for entry in ledger.entries] == ["RuntimeError", "RuntimeError"]
