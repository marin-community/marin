# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed structured outputs for OpenAI-compatible chat completions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredTool(Generic[OutputT]):
    """A strict function tool whose arguments are validated by a Pydantic model."""

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
        """Return fields that require exactly one call to this tool."""

        return {
            "tools": [self.definition()],
            "tool_choice": {"type": "function", "function": {"name": self.name}},
            "parallel_tool_calls": False,
        }

    def parse(self, response_body: dict[str, Any]) -> OutputT:
        """Parse the sole matching function call from a chat-completion response."""

        message = response_body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != self.name:
            raise ValueError(f"expected exactly one {self.name} tool call")
        return self.output_type.model_validate_json(calls[0]["function"]["arguments"])
