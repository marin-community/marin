# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ledger recording of model calls.

``RecordedModel`` is how proposals and build steps call a model: a ``ModelClient`` with its sampling
policy and an explicit attribution (item, round, step), recording every call as one ``LLM_CALL`` span.
Nothing is read from ambient state; a caller that attributes a call to another step passes
``dataclasses.replace(model, step=...)``. ``CallLedger`` is the same attribution for a rollout's
model calls.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from taskforge.content_hash import digest
from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.llm.client import Completion, ModelClient
from taskforge.llm.policy import LLMPolicy, Message


@dataclass(frozen=True)
class CallLedger:
    """Where GLM calls and tool calls are recorded, all under one item, round and step."""

    ledger: Ledger
    item_id: str
    round: int
    step: str


def record_completions(fields: SpanFields, completions: Sequence[Completion]) -> None:
    """Fill a call's span from its completions: summed token counts, the last finish reason, and
    ``attrs["requests"]``, the number of completions the call took."""
    fields.tokens_in = sum(completion.usage.prompt_tokens for completion in completions)
    fields.tokens_out = sum(completion.usage.completion_tokens for completion in completions)
    fields.tokens_reasoning = sum(completion.usage.reasoning_tokens for completion in completions)
    fields.finish_reason = str(completions[-1].finish_reason) if completions else None
    fields.attrs["requests"] = str(len(completions))


@dataclass(frozen=True)
class RecordedModel:
    """A model client, the policy every call samples with, and where its calls are recorded.

    Each call is one ``LLM_CALL`` span under ``item_id``, ``round`` and ``step`` naming
    ``client.model``; a call that raises is recorded with the exception's class as its cause and the
    exception propagates unchanged.
    """

    client: ModelClient
    policy: LLMPolicy
    ledger: Ledger
    item_id: str
    round: int
    step: str

    @property
    def digest(self) -> str:
        """The model and policy every call uses; not the attribution, so a step key that includes it
        is the same in every round and step."""
        return digest({"model": self.client.model, "policy": self.policy})

    async def complete(self, messages: Sequence[Message]) -> Completion:
        """One free-text completion of ``messages``."""
        with span(self.ledger, EntryKind.LLM_CALL, item_id=self.item_id, round=self.round, step=self.step) as fields:
            fields.model = self.client.model
            completion = await self.client.complete(messages, self.policy)
            record_completions(fields, (completion,))
        return completion

    async def structured[T: BaseModel](self, messages: Sequence[Message], output_type: type[T], name: str) -> T:
        """Force one call of a strict tool named ``name`` whose arguments ``output_type`` validates.

        Put checks the model can fix in ``output_type`` validators.
        """
        with span(self.ledger, EntryKind.LLM_CALL, item_id=self.item_id, round=self.round, step=self.step) as fields:
            fields.model = self.client.model
            fields.attrs["tool"] = name
            result = await self.client.complete_structured(messages, output_type, name=name, policy=self.policy)
            record_completions(fields, result.completions)
        return result.value
