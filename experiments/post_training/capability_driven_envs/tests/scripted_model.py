# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An in-process ``ModelClient`` that answers each request with the next queued reply, or raises the next queued error.

``ScriptedModel.complete_structured`` runs ``taskforge.llm.structured.complete_structured`` over
``complete_schema``, as ``GlmClient`` does, with one repair. Every request, structured ones included,
goes through ``complete``, which keeps the messages, policy and request fields in ``requests``.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from taskforge.llm.client import Completion, FinishReason, StructuredResult, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import complete_structured, response_format

STRUCTURED_REPAIRS = 1


@dataclass(frozen=True)
class ScriptedRequest:
    messages: list[Message]
    policy: LLMPolicy
    fields: dict[str, object]


def completion(content: str, finish: FinishReason = FinishReason.STOP) -> Completion:
    return Completion(
        content=content,
        reasoning="",
        tool_calls=(),
        finish_reason=finish,
        usage=Usage(prompt_tokens=100, completion_tokens=50, reasoning_tokens=10, cached_tokens=0),
        wall_time=1.0,
        ttft=0.1,
        decode_time=0.9,
        continuations=0,
        attempts=(),
    )


class ScriptedModel:
    model = "scripted"

    def __init__(self) -> None:
        self.replies: deque[Completion | Exception] = deque()
        self.requests: list[ScriptedRequest] = []

    def reply(self, content: str, finish: FinishReason = FinishReason.STOP) -> None:
        self.replies.append(completion(content, finish))

    def fail(self, error: Exception) -> None:
        self.replies.append(error)

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        self.requests.append(ScriptedRequest(list(messages), policy, dict(request_fields or {})))
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def complete_schema(
        self, messages: Sequence[Message], output_type: type[BaseModel], policy: LLMPolicy
    ) -> Completion:
        return await self.complete(messages, policy, {"response_format": response_format(output_type)})

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        return await complete_structured(self, messages, output_type, policy, STRUCTURED_REPAIRS)
