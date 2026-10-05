# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a proposal is made from (``Idea``), what a proposal source returns, and its protocol."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from taskforge.llm.client import Completion
from taskforge.llm.policy import Message
from taskforge.proposal.model import TaskProposal


@dataclass(frozen=True)
class CapabilityIdea:
    """One catalog capability with the learning-progression edges that point at it.

    ``capability`` and ``prerequisite_edges`` are source records shown to the model verbatim, so they
    stay as the catalog's JSON objects. ``capability_hash`` is the sha256 of the capability record's
    canonical JSON.
    """

    capability_id: str
    subject_id: str
    subject_name: str
    catalog_version: str
    capability: Mapping[str, object]
    prerequisite_edges: tuple[Mapping[str, object], ...]
    capability_hash: str


@dataclass(frozen=True)
class RepoIdea:
    """A repository at a pinned revision, the seed for grounded build_envs proposals."""

    url: str
    revision: str


Idea = CapabilityIdea | RepoIdea


@dataclass(frozen=True)
class SlotProposal:
    """A slot that produced a parsed proposal.

    ``request`` holds the messages of the first request; a repair request appends the first reply as
    the assistant turn and the parse error, which is ``repair_error``. A slot the plan marked null has
    no request and no completions.
    """

    slot: int
    proposal: TaskProposal
    request: tuple[Message, ...]
    completions: tuple[Completion, ...]
    repair_error: str | None


@dataclass(frozen=True)
class SlotFailure:
    """A slot whose document still failed the parser after its repair request."""

    slot: int
    error: str
    request: tuple[Message, ...]
    completions: tuple[Completion, ...]


SlotOutcome = SlotProposal | SlotFailure


@dataclass(frozen=True)
class ProposalBatch:
    """Everything one ``propose`` call produced: the planning call and one outcome per slot, in slot order."""

    planning_request: tuple[Message, ...]
    planning: tuple[Completion, ...]
    slots: tuple[SlotOutcome, ...]

    @property
    def proposals(self) -> tuple[TaskProposal, ...]:
        return tuple(s.proposal for s in self.slots if isinstance(s, SlotProposal))

    @property
    def failures(self) -> tuple[SlotFailure, ...]:
        return tuple(s for s in self.slots if isinstance(s, SlotFailure))


class ProposalSource(Protocol):
    async def propose(self, idea: Idea, n: int) -> ProposalBatch: ...
