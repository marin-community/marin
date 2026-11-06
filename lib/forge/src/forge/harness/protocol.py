# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harness contracts shared with the coordinator and stage configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from forge.effects.context import Context
from forge.model.llm import ToolSchema
from forge.model.stage import Exit
from forge.model.workpiece import Conversation, Submission
from forge.resources.env import Env
from forge.resources.inference import InferenceClient


class StopReason(StrEnum):
    FINISHED = "finished"
    CONTEXT_LIMIT = "context_limit"


@dataclass(frozen=True, slots=True)
class Submitted(Exit):
    submission: Submission
    session: Conversation


@dataclass(frozen=True, slots=True)
class Returned[P](Exit):
    tool: str
    payload: P
    session: Conversation


@dataclass(frozen=True, slots=True)
class Stopped(Exit):
    reason: StopReason
    session: Conversation
    detail: str = ""
    turns: int = 0

    @property
    def summary(self) -> str:
        detail = f": {self.detail}" if self.detail else ""
        return f"{self.reason.value} after {self.turns} turns{detail}"


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str


@dataclass(frozen=True, slots=True)
class SubmitRequest:
    submission: Submission


@dataclass(frozen=True, slots=True)
class ReturnRequest[P]:
    tool: str
    payload: P


@dataclass(frozen=True, slots=True)
class FinishRequest:
    reason: str


class Tool[P](Protocol):
    @property
    def schema(self) -> ToolSchema: ...

    async def __call__(
        self, ctx: Context, args: Mapping[str, object], env: Env
    ) -> ToolResult | SubmitRequest | ReturnRequest[P] | FinishRequest: ...


class Harness(Protocol):
    async def run(
        self,
        ctx: Context,
        session: Conversation,
        env: Env,
        client: InferenceClient,
        tools: Mapping[str, Tool[object]],
    ) -> Exit: ...
