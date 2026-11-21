# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-owned state. Replace records rather than mutate nested task/grade payloads.

Pydantic frozen models would not freeze those nested payloads either. Callers own
nested payloads and must replace them; persistence boundaries serialize snapshots.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Annotated, TypedDict, Unpack

from pydantic import TypeAdapter
from taskcompendium.models import TaskSpec

from taskforge.model.content_hash import digest
from taskforge.model.controls import Control
from taskforge.model.llm import Message, ToolSchema
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import OMIT_DEFAULT, BandOutcome, Decision, ValidationResult


@dataclass(frozen=True, slots=True)
class Submission:
    task: TaskSpec
    controls: tuple[Control, ...]
    notes: str

    @property
    def digest(self) -> str:
        return digest((self.task, self.controls))


@dataclass(frozen=True, slots=True)
class Fallback:
    """The submission a revision that leaves the acceptance band falls back to, and where it placed."""

    submission: Submission
    band: BandOutcome


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
    fallback: Fallback | None
    accepted_with: BandOutcome | None


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
    fallback: Annotated[Fallback | None, OMIT_DEFAULT] = None
    accepted_with: BandOutcome | None = None

    def with_(self, **changes: Unpack[WorkpieceChanges]) -> "Workpiece":
        return replace(self, **changes)


WORKPIECE = TypeAdapter(Workpiece)
