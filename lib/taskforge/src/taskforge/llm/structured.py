# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed structured output through a forced strict function call.

A ``StructuredTool`` is a strict function tool whose arguments a pydantic model validates. The
request carries that one tool with ``tool_choice`` pinned to it. If the reply has no
single valid call, one repair request follows that keeps the prior output in the conversation as
the assistant turn and states the validation error. A reply cut off by the output limit
(``FinishReason.LENGTH``) is invalid even when its arguments validate, because a cut call can be a
valid object that is missing optional fields.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from taskforge.llm.client import Completion, FinishReason, GlmClient, ToolCall
from taskforge.llm.policy import LLMPolicy, Message

OutputT = TypeVar("OutputT", bound=BaseModel)

ERROR_TEXT_LIMIT = 4000

REPAIR_PROMPT = (
    "Your previous output, shown above, failed validation:\n{error}\n"
    "Call the `{name}` tool exactly once with complete, corrected arguments."
)


@dataclass(frozen=True)
class StructuredTool(Generic[OutputT]):
    """A strict function tool whose arguments are validated by ``output_type``."""

    name: str
    description: str
    output_type: type[OutputT]

    def definition(self) -> dict[str, object]:
        parameters = self.output_type.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": parameters,
            },
        }

    def request_fields(self) -> dict[str, object]:
        """Request-body fields that require exactly one call to this tool."""
        return {
            "tools": [self.definition()],
            "tool_choice": {"type": "function", "function": {"name": self.name}},
            "parallel_tool_calls": False,
        }

    def parse(self, tool_calls: Sequence[ToolCall]) -> OutputT:
        """Validate the sole call to this tool; raise ``ValueError`` otherwise."""
        if len(tool_calls) != 1 or tool_calls[0].name != self.name:
            raise ValueError(f"expected exactly one {self.name} tool call")
        return self.output_type.model_validate_json(tool_calls[0].arguments)


@dataclass(frozen=True)
class StructuredResult(Generic[OutputT]):
    value: OutputT
    completions: tuple[Completion, ...]


class StructuredOutputError(ValueError):
    """Neither the first reply nor the repair produced a valid call."""

    def __init__(self, error: ValueError, completions: Sequence[Completion]):
        super().__init__(f"structured output invalid after repair: {error}")
        self.completions = tuple(completions)


def _parsed(tool: StructuredTool[OutputT], completion: Completion) -> OutputT:
    if completion.finish_reason is FinishReason.LENGTH:
        raise ValueError(f"the {tool.name} call was cut off by the output limit before it was complete")
    return tool.parse(completion.tool_calls)


def repair_messages(
    messages: Sequence[Message], tool: StructuredTool[OutputT], prior: Completion, error: ValueError
) -> list[Message]:
    """Return ``messages`` plus the prior output as the assistant turn and the validation error."""
    calls = "\n".join(f"[{c.name} arguments]\n{c.arguments}" for c in prior.tool_calls)
    transcript = "\n".join(part for part in (prior.content, calls) if part)
    prompt = REPAIR_PROMPT.format(error=str(error)[:ERROR_TEXT_LIMIT], name=tool.name)
    return [*messages, {"role": "assistant", "content": transcript}, {"role": "user", "content": prompt}]


async def complete_structured(
    client: GlmClient, messages: Sequence[Message], policy: LLMPolicy, tool: StructuredTool[OutputT]
) -> StructuredResult[OutputT]:
    """Return the validated ``tool`` arguments, spending at most one repair request."""
    fields = tool.request_fields()
    first = await client.complete(messages, policy, fields)
    try:
        return StructuredResult(_parsed(tool, first), (first,))
    except ValueError as error:
        repair = await client.complete(repair_messages(messages, tool, first, error), policy, fields)
        try:
            return StructuredResult(_parsed(tool, repair), (first, repair))
        except ValueError as repair_error:
            raise StructuredOutputError(repair_error, (first, repair)) from repair_error
