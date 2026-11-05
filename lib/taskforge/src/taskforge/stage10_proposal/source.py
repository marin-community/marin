# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two phases of a proposal source, what each returns, and the batch assembled from them.

A source plans a portfolio once (``ProposalSource.plan``) and then writes each slot's document on its
own (``ProposalSource.write``), so a scheduler can start a slot's downstream work as soon as that
slot's document exists, and a relaunch holding a persisted portfolio writes only the slots it lacks.
"""

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from taskforge.llm.client import Completion
from taskforge.llm.policy import Message
from taskforge.stage10_proposal.model import TaskProposal


@dataclass(frozen=True)
class PlannedSlot:
    """A slot of a portfolio: it proposes a task, or the plan marked it null for ``null_reason``."""

    slot: int
    null_reason: str | None


@dataclass(frozen=True)
class Portfolio[PlanT]:
    """A source's plan for ``n`` slots: everything ``write`` needs, and everything a caller persists.

    ``plan`` is the source's own plan record, which its document prompts quote. ``slots`` lists the
    planned slots, numbered 1 through ``n`` in order. ``planning_request`` holds the messages of the
    first planning request and ``planning`` its completions, one more per repair.

    Raises:
        ValueError: ``slots`` is not numbered 1 through ``n`` in order.
    """

    planning_request: tuple[Message, ...]
    planning: tuple[Completion, ...]
    plan: PlanT
    slots: tuple[PlannedSlot, ...]

    def __post_init__(self) -> None:
        numbers = [s.slot for s in self.slots]
        if numbers != list(range(1, len(self.slots) + 1)):
            raise ValueError(f"portfolio slots must be numbered 1..{len(self.slots)} in order, got {numbers}")


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
    """A slot whose document was cut off by the output limit, or still failed the parser after its repair request."""

    slot: int
    error: str
    request: tuple[Message, ...]
    completions: tuple[Completion, ...]


SlotOutcome = SlotProposal | SlotFailure


@dataclass(frozen=True)
class ProposalBatch:
    """A portfolio's planning call and one outcome per slot, in slot order."""

    planning_request: tuple[Message, ...]
    planning: tuple[Completion, ...]
    slots: tuple[SlotOutcome, ...]

    @property
    def proposals(self) -> tuple[TaskProposal, ...]:
        return tuple(s.proposal for s in self.slots if isinstance(s, SlotProposal))

    @property
    def failures(self) -> tuple[SlotFailure, ...]:
        return tuple(s for s in self.slots if isinstance(s, SlotFailure))


class ProposalSource[IdeaT, PlanT](Protocol):
    """Plans ``n`` proposal slots for one idea of the source's own type, then writes each slot alone.

    ``plan`` raises when the portfolio cannot be planned. ``write`` depends on the idea and the
    portfolio alone, never on another slot's outcome, so the slots of one portfolio may be written
    concurrently, in any order, or in separate processes from a persisted portfolio. A slot planned
    null comes back as a ``SlotProposal`` with ``null_reason`` set and no model call. A slot whose
    document fails comes back as a ``SlotFailure``, which leaves its siblings unaffected.
    """

    async def plan(self, idea: IdeaT, n: int) -> Portfolio[PlanT]: ...

    async def write(self, idea: IdeaT, portfolio: Portfolio[PlanT], slot: int) -> SlotOutcome: ...


def proposal_batch[PlanT](portfolio: Portfolio[PlanT], outcomes: Iterable[SlotOutcome]) -> ProposalBatch:
    """Assemble ``portfolio`` and an outcome for each of its slots, given in any order, into a batch.

    Raises:
        ValueError: ``outcomes`` lacks a slot of the portfolio, repeats one, or holds a slot it lacks.
    """
    by_slot: dict[int, SlotOutcome] = {}
    for outcome in outcomes:
        if outcome.slot in by_slot:
            raise ValueError(f"slot {outcome.slot} has two outcomes")
        by_slot[outcome.slot] = outcome
    planned = [s.slot for s in portfolio.slots]
    if sorted(by_slot) != planned:
        raise ValueError(f"outcomes cover slots {sorted(by_slot)}, the portfolio plans {planned}")
    return ProposalBatch(portfolio.planning_request, portfolio.planning, tuple(by_slot[n] for n in planned))


async def propose_all[IdeaT, PlanT](source: ProposalSource[IdeaT, PlanT], idea: IdeaT, n: int) -> ProposalBatch:
    """Plan ``n`` slots for ``idea``, write every slot concurrently, and return the batch.

    ``source.plan``'s errors propagate as raised; a ``source.write`` that raises in place of returning
    an outcome cancels the other writes and reaches the caller inside an ``ExceptionGroup``.
    """
    portfolio = await source.plan(idea, n)
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(source.write(idea, portfolio, s.slot)) for s in portfolio.slots]
    return proposal_batch(portfolio, (task.result() for task in tasks))
