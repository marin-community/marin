# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-owned state. Replace records rather than mutate nested task/grade payloads.

Pydantic frozen models would not freeze those nested payloads either. Callers own
nested payloads and must replace them; persistence boundaries serialize snapshots.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypedDict, Unpack

from pydantic import TypeAdapter
from taskcompendium.models import TaskSpec

from taskforge.model.content_hash import digest
from taskforge.model.controls import Control
from taskforge.model.llm import Message, ToolSchema
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import BandOutcome, Decision, ValidationResult


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
    proposal: TaskProposal | None
    round: int
    session: Conversation | None
    submission: Submission | None
    evidence: tuple[ValidationResult, ...]
    decisions: tuple[Decision, ...]
    terminal: Terminal | None
    too_easy_baseline: Submission | None
    accepted_with: BandOutcome | None


@dataclass(frozen=True, slots=True)
class Workpiece:
    """A task in progress, started from a proposal or from an existing task in ``submission``.

    A workpiece without a proposal repairs a task another source wrote: it starts at validation, and
    review's findings open its author sessions.
    """

    id: str
    proposal: TaskProposal | None = None
    round: int = 0
    session: Conversation | None = None
    submission: Submission | None = None
    evidence: tuple[ValidationResult, ...] = ()
    decisions: tuple[Decision, ...] = ()
    terminal: Terminal | None = None
    too_easy_baseline: Submission | None = None
    accepted_with: BandOutcome | None = None

    def __post_init__(self) -> None:
        if self.proposal is None and self.submission is None:
            raise ValueError(f"Workpiece {self.id} needs a proposal or an existing task")

    @property
    def modify_rounds(self) -> int:
        """The review modify rounds charged against ``budget.max_modify_rounds``.

        A source task's first modify only opens its first author session, after review measured the task as
        its source wrote it, so it is not charged: a source task gets as many author sessions as a proposal.
        """
        return self.round - 1 if self.proposal is None and self.round > 0 else self.round

    def with_(self, **changes: Unpack[WorkpieceChanges]) -> "Workpiece":
        return replace(self, **changes)


WORKPIECE = TypeAdapter(Workpiece)
