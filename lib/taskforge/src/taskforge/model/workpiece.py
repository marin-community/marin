# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-owned state. Replace records rather than mutate nested task/grade payloads.

Pydantic frozen models would not freeze those nested payloads either. Callers own
nested payloads and must replace them; persistence boundaries serialize snapshots.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Annotated, Literal, TypedDict, Unpack

from pydantic import Field, TypeAdapter
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


@dataclass(frozen=True, slots=True)
class Generated:
    """A workpiece that builds a new task from ``proposal``."""

    proposal: TaskProposal
    kind: Literal["generated"] = "generated"


@dataclass(frozen=True, slots=True)
class Repaired:
    """A workpiece that repairs a task another source wrote; ``source`` keeps it as the source delivered it."""

    source: Submission
    kind: Literal["repaired"] = "repaired"


type Origin = Annotated[Generated | Repaired, Field(discriminator="kind")]


class Terminal(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    HELD = "held"


class WorkpieceChanges(TypedDict, total=False):
    id: str
    origin: Origin
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

    A ``Repaired`` workpiece starts at validation with its source task as ``submission``, and review's
    findings open its author sessions. Its origin keeps the task as the source delivered it, after the
    author replaces ``submission``.
    """

    id: str
    origin: Origin
    round: int = 0
    session: Conversation | None = None
    submission: Submission | None = None
    evidence: tuple[ValidationResult, ...] = ()
    decisions: tuple[Decision, ...] = ()
    terminal: Terminal | None = None
    too_easy_baseline: Submission | None = None
    accepted_with: BandOutcome | None = None

    def __post_init__(self) -> None:
        if isinstance(self.origin, Repaired) and self.submission is None:
            raise ValueError(f"Workpiece {self.id} repairs a source task, so it starts with that task as its submission")

    @property
    def modify_rounds(self) -> int:
        """The review modify rounds charged against ``budget.max_modify_rounds``.

        A source task's first modify only opens its first author session, after review measured the task as
        its source wrote it, so it is not charged: a source task gets as many author sessions as a proposal.
        """
        return self.round - 1 if isinstance(self.origin, Repaired) and self.round > 0 else self.round

    def with_(self, **changes: Unpack[WorkpieceChanges]) -> "Workpiece":
        return replace(self, **changes)


WORKPIECE = TypeAdapter(Workpiece)
