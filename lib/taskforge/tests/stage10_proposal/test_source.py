# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import TypeAdapter

from taskforge.content_hash import pretty_json
from taskforge.llm.client import Completion, FinishReason, Usage
from taskforge.llm.policy import Message
from taskforge.stage10_proposal.model import (
    Environment,
    Grounding,
    ProposalHeader,
    ResearchNeed,
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    parse,
    render,
)
from taskforge.stage10_proposal.source import (
    PlannedSlot,
    Portfolio,
    SlotFailure,
    SlotOutcome,
    SlotProposal,
    proposal_batch,
    propose_all,
)

IDEA = "d01.algebra.linear-transformations"
SOURCE = SourceRef(SourceKind.CAPABILITY, IDEA, "9f3a12")
BODY = """## Task
Find the kernel of {title}.

## Realism and workflow
A lab handout.

## Research plan
None needed.

## Build plan
Generate fixtures.

## Grader design and controls
check.py scores the JSON.

## Risks and null conditions
None known.
"""
NULL_REASON = "no realistic grader exists"


def completion(content: str, finish_reason: FinishReason = FinishReason.STOP) -> Completion:
    return Completion(
        content=content,
        reasoning="",
        tool_calls=(),
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=10, completion_tokens=20, reasoning_tokens=0, cached_tokens=0),
        wall_time=1.0,
        ttft=0.1,
        decode_time=0.9,
        continuations=0,
        attempts=(),
    )


def header(slot: int, null_reason: str | None) -> ProposalHeader:
    return ProposalHeader(
        id=f"{IDEA}/{slot}",
        source=SOURCE,
        environment=Environment.REASONING,
        solver_machine=SolverMachine.NONE,
        verification=Verification.ANSWER_IN_BOX,
        composite=(),
        grounding=Grounding.UNVERIFIED,
        research_need=ResearchNeed.NOT_NEEDED,
        research_reason="the task is self-contained",
        research=(),
        build=(),
        resources=(),
        null_reason=null_reason,
    )


@dataclass(frozen=True)
class TitlePlan:
    """The test source's plan record: a title per slot, which each document prompt quotes."""

    titles: tuple[str, ...]


class DocumentModel:
    """A scripted document writer: the reply of each slot, held until that slot's gate opens, if it has one."""

    def __init__(self, replies: Mapping[int, Completion], gates: Mapping[int, asyncio.Event]) -> None:
        self.replies = replies
        self.gates = gates
        self.calls: list[int] = []

    async def complete(self, slot: int, request: tuple[Message, ...]) -> Completion:
        self.calls.append(slot)
        if slot in self.gates:
            await self.gates[slot].wait()
        return self.replies[slot]


class TitleSource:
    """A ``ProposalSource[str, TitlePlan]``: slots in ``null_slots`` are planned null, the rest call ``model``."""

    def __init__(self, model: DocumentModel, null_slots: Mapping[int, str]) -> None:
        self.model = model
        self.null_slots = null_slots

    async def plan(self, idea: str, n: int) -> Portfolio[TitlePlan]:
        request: tuple[Message, ...] = ({"role": "user", "content": f"Plan {n} slots for {idea}."},)
        plan = TitlePlan(tuple(f"map {k}" for k in range(1, n + 1)))
        slots = tuple(PlannedSlot(k, self.null_slots.get(k)) for k in range(1, n + 1))
        return Portfolio(request, (completion(pretty_json(plan)),), plan, slots)

    async def write(self, idea: str, portfolio: Portfolio[TitlePlan], slot: int) -> SlotOutcome:
        planned = portfolio.slots[slot - 1]
        if planned.null_reason is not None:
            proposal = TaskProposal(header(slot, planned.null_reason), f"Slot {slot} was planned as null.\n")
            return SlotProposal(slot, parse(render(proposal)), (), (), None)
        title = portfolio.plan.titles[slot - 1]
        request: tuple[Message, ...] = ({"role": "user", "content": f"Write slot {slot} of {idea}: {title}."},)
        reply = await self.model.complete(slot, request)
        if reply.finish_reason is FinishReason.LENGTH:
            return SlotFailure(slot, "cut off by the output limit", request, (reply,))
        return SlotProposal(slot, parse(reply.content), request, (reply,), None)


def document(slot: int) -> Completion:
    return completion(render(TaskProposal(header(slot, None), BODY.format(title=f"map {slot}"))))


def three_slot_source(gates: Mapping[int, asyncio.Event] | None = None) -> TitleSource:
    """Slot 1 writes a document, slot 2's reply is cut off by the output limit, slot 3 is planned null."""
    replies = {1: document(1), 2: completion("---\nid: ", FinishReason.LENGTH)}
    return TitleSource(DocumentModel(replies, gates or {}), {3: NULL_REASON})


async def test_write_slot_returns_while_a_sibling_document_is_still_being_written():
    gate = asyncio.Event()
    source = TitleSource(DocumentModel({1: document(1), 2: document(2)}, {2: gate}), {})
    portfolio = await source.plan(IDEA, 2)

    first = asyncio.create_task(source.write(IDEA, portfolio, 1))
    second = asyncio.create_task(source.write(IDEA, portfolio, 2))
    done, _ = await asyncio.wait({first, second}, timeout=10, return_when=asyncio.FIRST_COMPLETED)

    assert done == {first}
    outcome = first.result()
    assert isinstance(outcome, SlotProposal)
    assert outcome.proposal.header.id == f"{IDEA}/1"
    assert not second.done()
    gate.set()
    assert isinstance(await second, SlotProposal)


async def test_slots_written_in_separate_calls_from_a_persisted_portfolio_assemble_the_one_shot_batch():
    one_shot = await propose_all(three_slot_source(), IDEA, 3)

    first_run = three_slot_source()
    portfolio = await first_run.plan(IDEA, 3)
    slot_1 = await first_run.write(IDEA, portfolio, 1)
    persisted = TypeAdapter(Portfolio[TitlePlan]).validate_json(pretty_json(portfolio))
    relaunch = three_slot_source()
    missing = [await relaunch.write(IDEA, persisted, slot) for slot in (3, 2)]
    batch = proposal_batch(persisted, [*missing, slot_1])

    assert relaunch.model.calls == [2]
    assert [(type(s), s.slot) for s in batch.slots] == [(SlotProposal, 1), (SlotFailure, 2), (SlotProposal, 3)]
    assert batch == one_shot


async def test_write_null_slot_makes_no_model_call():
    source = three_slot_source()
    portfolio = await source.plan(IDEA, 3)

    outcome = await source.write(IDEA, portfolio, 3)

    assert source.model.calls == []
    assert isinstance(outcome, SlotProposal)
    assert outcome.proposal.header.null_reason == NULL_REASON
    assert (outcome.request, outcome.completions) == ((), ())
