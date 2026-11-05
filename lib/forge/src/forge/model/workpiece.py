# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable state carried between pipeline stages."""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypedDict, Unpack

from taskcompendium.models import TaskSpec

from forge.model.content_hash import digest
from forge.model.controls import Control
from forge.model.llm import Message, ToolSchema
from forge.model.proposal import TaskProposal
from forge.model.review import Decision, ValidationResult


@dataclass(frozen=True, slots=True)
class Submission:
    task: TaskSpec
    controls: tuple[Control, ...]
    notes: str

    @property
    def digest(self) -> str:
        return digest((self.task, self.controls))


@dataclass(frozen=True, slots=True)
class Conversation:
    messages: tuple[Message, ...]
    tool_schemas: tuple[ToolSchema, ...]


class Terminal(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    HELD = "held"


class WorkpieceChanges(TypedDict, total=False):
    id: str
    proposal: TaskProposal
    round: int
    session: Conversation | None
    submission: Submission | None
    evidence: tuple[ValidationResult, ...]
    decisions: tuple[Decision, ...]
    terminal: Terminal | None


@dataclass(frozen=True, slots=True)
class Workpiece:
    id: str
    proposal: TaskProposal
    round: int = 0
    session: Conversation | None = None
    submission: Submission | None = None
    evidence: tuple[ValidationResult, ...] = ()
    decisions: tuple[Decision, ...] = ()
    terminal: Terminal | None = None

    def with_(self, **changes: Unpack[WorkpieceChanges]) -> "Workpiece":
        return replace(self, **changes)
