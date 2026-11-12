# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-owned state. Replace records rather than mutate nested task/grade payloads.

Pydantic frozen models would not freeze those nested payloads either. Callers own
nested payloads and must replace them; persistence boundaries serialize snapshots.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypedDict, Unpack

from pydantic import ConfigDict, TypeAdapter
from taskcompendium.models import TaskSpec

from forge.model.content_hash import digest
from forge.model.controls import Control
from forge.model.llm import Message, ToolSchema
from forge.model.proposal import TaskProposal
from forge.model.review import BandOutcome, Decision, ValidationResult


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


@dataclass(frozen=True, slots=True)
class Fallback:
    """The last submission that validated clean, and its band placement; a regressing revision reverts to it."""

    submission: Submission
    band: BandOutcome


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
    fallback: Fallback | None
    accepted_with: BandOutcome | None


@dataclass(frozen=True, slots=True)
class Workpiece:
    # Unknown fields fail loudly: a workpiece from an older format must never be silently reinterpreted.
    __pydantic_config__ = ConfigDict(extra="forbid")

    id: str
    proposal: TaskProposal
    round: int = 0
    session: Conversation | None = None
    submission: Submission | None = None
    evidence: tuple[ValidationResult, ...] = ()
    decisions: tuple[Decision, ...] = ()
    terminal: Terminal | None = None
    fallback: Fallback | None = None
    accepted_with: BandOutcome | None = None

    def with_(self, **changes: Unpack[WorkpieceChanges]) -> "Workpiece":
        return replace(self, **changes)


WORKPIECE = TypeAdapter(Workpiece)
