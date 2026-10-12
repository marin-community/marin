# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed structured output through a forced strict function call: ``GlmClient``'s mechanism.

``GlmClient.complete_structured`` implements ``ModelClient.complete_structured`` with a
``StructuredTool``: a strict function tool whose arguments a pydantic model validates, named by
``tool_name`` after that model. The request carries that one tool with ``tool_choice`` pinned to
it. If the reply has no single valid call, a repair request follows that keeps the prior output in
the conversation as the assistant turn and states the validation error, up to ``max_repairs``
times. A reply cut off by the output limit (``FinishReason.LENGTH``) is invalid even when its
arguments validate, because a cut call can be a valid object that is missing optional fields.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from taskforge.llm.client import Completion, FinishReason, StructuredResult, ToolCall
from taskforge.llm.policy import LLMPolicy, Message

ERROR_TEXT_LIMIT = 4000
WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

REPAIR_PROMPT = (
    "Your previous output, shown above, failed validation:\n{error}\n"
    "Call the `{name}` tool exactly once with complete, corrected arguments."
)


def tool_name(output_type: type[BaseModel]) -> str:
    """The tool name a structured call for ``output_type`` uses: its class name in snake case."""
    return WORD_BOUNDARY.sub("_", output_type.__name__).lower()


class FieldedCompletions(Protocol):
    """A client whose ``complete`` takes extra request-body fields such as ``tools``."""

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion: ...


@dataclass(frozen=True)
class StructuredTool[T: BaseModel]:
    """A strict function tool whose arguments are validated by ``output_type``."""

    name: str
    description: str
    output_type: type[T]

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

    def parse(self, tool_calls: Sequence[ToolCall]) -> T:
        """Validate the sole call to this tool; raise ``ValueError`` otherwise."""
        if len(tool_calls) != 1 or tool_calls[0].name != self.name:
            raise ValueError(f"expected exactly one {self.name} tool call")
        return self.output_type.model_validate_json(tool_calls[0].arguments)


class StructuredOutputError(ValueError):
    """Neither the first reply nor any repair produced a valid call."""

    def __init__(self, error: ValueError, completions: Sequence[Completion]):
        super().__init__(f"structured output invalid after {len(completions) - 1} repairs: {error}")
        self.completions = tuple(completions)


def _parsed[T: BaseModel](tool: StructuredTool[T], completion: Completion) -> T:
    if completion.finish_reason is FinishReason.LENGTH:
        raise ValueError(f"the {tool.name} call was cut off by the output limit before it was complete")
    return tool.parse(completion.tool_calls)


def repair_messages(
    messages: Sequence[Message], tool: StructuredTool, prior: Completion, error: ValueError
) -> list[Message]:
    """Return ``messages`` plus the prior output as the assistant turn and the validation error."""
    calls = "\n".join(f"[{c.name} arguments]\n{c.arguments}" for c in prior.tool_calls)
    transcript = "\n".join(part for part in (prior.content, calls) if part)
    prompt = REPAIR_PROMPT.format(error=str(error)[:ERROR_TEXT_LIMIT], name=tool.name)
    return [*messages, {"role": "assistant", "content": transcript}, {"role": "user", "content": prompt}]


async def complete_structured[T: BaseModel](
    client: FieldedCompletions,
    messages: Sequence[Message],
    tool: StructuredTool[T],
    policy: LLMPolicy,
    max_repairs: int,
) -> StructuredResult[T]:
    """Return the validated ``tool`` arguments, sending at most ``max_repairs`` repair requests.

    Each repair continues from the original ``messages`` with only the latest invalid output.

    Raises:
        StructuredOutputError: the last allowed reply was still invalid; it holds every completion.
    """
    fields = tool.request_fields()
    completions = [await client.complete(messages, policy, fields)]
    while True:
        try:
            return StructuredResult(_parsed(tool, completions[-1]), tuple(completions))
        except ValueError as error:
            if len(completions) > max_repairs:
                raise StructuredOutputError(error, completions) from error
            repair = repair_messages(messages, tool, completions[-1], error)
            completions.append(await client.complete(repair, policy, fields))
