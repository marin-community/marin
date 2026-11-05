# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent values exchanged by harnesses and inference resources."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import PlainSerializer

from forge.model.json import FrozenJson, freeze_json, thaw_json


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ABORT = "abort"


class ReasoningEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: Annotated[Mapping[str, FrozenJson], PlainSerializer(thaw_json)]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameters", MappingProxyType({key: freeze_json(value) for key, value in self.parameters.items()})
        )


@dataclass(frozen=True, slots=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    max_continuations: int | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Usage()
    continuations: int = 0
    reasoning: str = ""
