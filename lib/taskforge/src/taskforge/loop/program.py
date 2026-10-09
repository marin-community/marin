# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-idea and per-item programs of one unattended run.

``run_idea`` proposes ``n`` items for one idea. A failed slot costs only that slot; a batch with zero
proposals re-proposes the idea, up to ``max_idea_reproposals``, then the idea is exhausted.

``run_item`` carries one proposal to a ``Terminal``: triage, authoring and building with bounded
revisions, control replay, solver and adversary trials, the calibration summary and review's decision.
Here triage has no checks and accepts every proposal, authoring adopts ``services.template`` unchanged
as the item's program, and control replay and the adversaries run no trial, so their events record
zero counts and validation is the solver's trials. Since authoring cannot revise a program, the
policy allows no ``Repair`` (``LoopPolicy``): a decisive finding rejects the item for budget. A
``Retry`` re-enters validation after a backoff, re-running only unsettled trials. A solve rate outside
the band is decided by that kind's rule in ``policy.band_rules``, which accepts the task
(``ACCEPTED``, labelled with its band) or rejects it. A spent retry budget ends the item ``ABANDONED``
with its causes. An unhandled exception, ``GlmUnavailable`` included, records ``FAILED`` and
propagates.

Each loop iteration derives the item's state from its event log, runs the sub-phase the state names
and appends that sub-phase's completion event, so a new process resumes an item by re-running only
the sub-phase that lacks its event; inside validation, settled trials load from their attempt files.
``services.slots`` bounds the phases that call the model or a sandbox (building and trials); an item
waiting out a retry backoff, deciding or closing holds no slot.

Run root layout::

    items/<item_id>/proposal.md                current proposal; proposals/<digest>.md keeps each one
    items/<item_id>/verdict.json               triage verdict on the current proposal
    items/<item_id>/rounds/<round>/            program.py, program.json, author/, draft/, scratch/,
                                               build-failures/<n>.txt, evidence-<digest12>/
    items/idea--<idea_id>/idea.json            the idea as ``LoopServices.describe_idea`` records it
    items/idea--<idea_id>/proposals/<item_id>.md   each proposal as its idea's batch produced it
    items/idea--<idea_id>/batches/<reproposal>/plan/          request.json, completions.json
    items/idea--<idea_id>/batches/<reproposal>/slots/<slot>/  request.json, completions.json, and
                                               repair_error.txt or failure.txt when the slot has one
    cache/                                     the step cache shared by every item
    ledger/<item_id>.jsonl                     the item's spans and events
"""

import asyncio
import inspect
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from pydantic import TypeAdapter

from taskforge.atomic_file import write_atomic
from taskforge.builder.author import PROGRAM_FILE
from taskforge.builder.run import DRAFT_DIR, TaskDraft, item_id_for, load_draft, run_build
from taskforge.builder.sdk import Build, BuildFailure, BuildOutput, BuildServices
from taskforge.builder.step import CacheStatus
from taskforge.content_hash import pretty_json, sha256_hex
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind, Ledger, LedgerEntry, check_item_id, span
from taskforge.llm.client import Completion, GlmUnavailable
from taskforge.llm.policy import Message
from taskforge.loop.events import (
    FINAL,
    DecisionKind,
    EventKind,
    ItemState,
    Phase,
    ProposalOrigin,
    RevisionKind,
    Terminal,
    cause_counts,
    derive_idea_state,
    derive_state,
    events,
    joined,
    record_event,
)
from taskforge.loop.policy import LoopPolicy
from taskforge.proposal.model import TaskProposal, parse, render
from taskforge.proposal.source import ProposalBatch, ProposalSource, SlotFailure
from taskforge.review.decision import (
    DECISION_FILE,
    Accept,
    Decision,
    Reject,
    RejectKind,
    Repair,
    Retry,
    write_decision,
)
from taskforge.review.rules import ItemHistory, decide
from taskforge.triage.verdict import TriageDecision
from taskforge.validate.calibration import CalibrationSummary, summarize, write_summary
from taskforge.validate.evidence import Evidence
from taskforge.validate.outcome import TrialKind
from taskforge.validate.run import load_validation
from taskforge.validate.solver import ModelFactory, ValidationSite, run_solver
from taskforge.validate.trials import EngineSettings, RetryBackoff, task_digest

ITEMS_DIR = "items"
CACHE_DIR = "cache"
LEDGER_DIR = "ledger"
ROUNDS_DIR = "rounds"
PROPOSALS_DIR = "proposals"
PROPOSAL_FILE = "proposal.md"
VERDICT_FILE = "verdict.json"
CALIBRATION_FILE = "calibration.json"
FAILURES_DIR = "build-failures"
IDEA_PREFIX = "idea--"
IDEA_FILE = "idea.json"
BATCHES_DIR = "batches"
PLAN_DIR = "plan"
SLOTS_DIR = "slots"
REQUEST_FILE = "request.json"
COMPLETIONS_FILE = "completions.json"
REPAIR_ERROR_FILE = "repair_error.txt"
SLOT_FAILURE_FILE = "failure.txt"

FAILURE_CHARS = 6000
"""The tail of a build failure's traceback kept for the author's revision."""
ATTR_CHARS = 500
"""Longest free text kept in an event attribute; the full text stays in the item directory."""
DIGEST_CHARS = 12

_COMPLETIONS: TypeAdapter[tuple[Completion, ...]] = TypeAdapter(tuple[Completion, ...])

NO_TRIAGE = "0 structural checks and 0 rubric samples ran"
"""The ``TRIAGED`` tally of a proposal no check or rubric sample ran on."""


@dataclass(frozen=True)
class LoopServices[IdeaT]:
    """Everything a run's items share.

    Attributes:
        source: Turns ideas into proposal batches.
        describe_idea: The JSON-serialisable record of an idea, written once to its ``idea.json``.
        template: The builder template every item adopts as its program (``builder.template.standard``).
        build: The builder's services; its ledger is ``ledger``.
        engine: Run-wide engine settings; each draft's trials present the task in its own answer format.
        rollout_models: Builds each solver trial's model, recording under the trial's step.
        ledger: The run's ledger; its local JSONL root must be ``root / "ledger"``.
        root: The run root.
        slots: Bounds the model- and sandbox-bound phases running at once across items.
    """

    source: ProposalSource[IdeaT]
    describe_idea: Callable[[IdeaT], Mapping[str, object]]
    template: ModuleType
    build: BuildServices
    engine: EngineSettings
    rollout_models: ModelFactory
    ledger: Ledger
    root: Path
    slots: asyncio.Semaphore

    def __post_init__(self) -> None:
        if self.build.ledger is not self.ledger:
            raise ValueError("LoopServices.build must record to the run's ledger")


@dataclass(frozen=True)
class TemplateProgram:
    """A builder template adopted unchanged as an item's program; its digest is its source's sha256."""

    source: str
    build: Callable[[Build], Awaitable[BuildOutput]]

    @property
    def digest(self) -> str:
        return sha256_hex(self.source.encode())


def template_program(template: ModuleType) -> TemplateProgram:
    return TemplateProgram(source=inspect.getsource(template), build=template.build)


class EventLog:
    """One item's events: read from its JSONL file, appended through the run's ledger.

    ``append`` numbers each event after the events on disk and reads the file back, so a ledger that
    does not write to ``root / "ledger"`` fails on the first event rather than looping forever.
    """

    def __init__(self, root: Path, ledger: Ledger, item_id: str):
        check_item_id(item_id)
        self.item_id = item_id
        self.ledger = ledger
        self.path = JsonlLedger(root / LEDGER_DIR).path_for(item_id)

    def entries(self) -> list[LedgerEntry]:
        return list(read_entries(self.path)) if self.path.exists() else []

    def append(
        self,
        round: int,  # noqa: A002 - matches LedgerEntry.round
        kind: EventKind,
        input_hash: str | None,
        /,
        **attrs: str,
    ) -> None:
        seq = len(events(self.entries())) + 1
        record_event(self.ledger, self.item_id, round, kind, seq, input_hash, **attrs)
        recorded = len(events(self.entries()))
        if recorded != seq:
            raise ValueError(f"event {seq} of {self.item_id} did not reach {self.path}; the ledger writes elsewhere")


def evidence_dir(item_directory: Path, round: int, digest: str) -> Path:  # noqa: A002 - matches LedgerEntry.round
    """The directory an item keeps the validation evidence of round ``round`` of the task with ``digest`` in."""
    return item_directory / ROUNDS_DIR / str(round) / f"evidence-{digest[:DIGEST_CHARS]}"


def idea_item_id(idea_id: str) -> str:
    return f"{IDEA_PREFIX}{idea_id}"


def _clip(text: str) -> str:
    return text[:ATTR_CHARS]


async def run_idea[IdeaT](
    idea_id: str, idea: IdeaT, policy: LoopPolicy, services: LoopServices[IdeaT]
) -> tuple[TaskProposal, ...]:
    """The proposals of ``idea``, proposing a batch only when its log holds none.

    Each batch's failed slots are recorded as ``SLOT_FAILED`` and its siblings proceed. A batch with
    zero proposals is re-proposed up to ``policy.max_idea_reproposals`` times; then the idea is
    ``IDEA_EXHAUSTED`` and yields nothing. A resumed idea returns its recorded batch without a call.
    The idea's record and every batch's requests and completions are kept in the idea's directory.
    """
    item_id = idea_item_id(idea_id)
    log = EventLog(services.root, services.ledger, item_id)
    idea_dir = services.root / ITEMS_DIR / item_id
    batch_dir = idea_dir / PROPOSALS_DIR
    idea_dir.mkdir(parents=True, exist_ok=True)
    if not (idea_dir / IDEA_FILE).exists():
        write_atomic(idea_dir / IDEA_FILE, pretty_json(services.describe_idea(idea)).encode())
    while True:
        state = derive_idea_state(log.entries())
        if state.items is not None:
            return tuple(parse((batch_dir / f"{item}.md").read_text()) for item in state.items)
        if state.exhausted:
            return ()
        if state.reproposals > policy.max_idea_reproposals:
            log.append(state.reproposals, EventKind.IDEA_EXHAUSTED, None, reproposals=str(state.reproposals))
            return ()
        async with services.slots:
            with span(services.ledger, EntryKind.LLM_CALL, item_id=item_id, round=state.reproposals, step="propose"):
                batch = await services.source.propose(idea, policy.proposals_per_idea)
        items = [item_id_for(proposal) for proposal in batch.proposals]
        if len(set(items)) != len(items) or any("," in item for item in items):
            raise ValueError(f"idea {idea_id}: proposal ids must be distinct and comma-free, got {items}")
        _keep_batch(idea_dir / BATCHES_DIR / str(state.reproposals), batch)
        batch_dir.mkdir(parents=True, exist_ok=True)
        for item, proposal in zip(items, batch.proposals, strict=True):
            write_atomic(batch_dir / f"{item}.md", render(proposal).encode())
        for failure in batch.failures:
            log.append(
                state.reproposals, EventKind.SLOT_FAILED, None, slot=str(failure.slot), error=_clip(failure.error)
            )
        log.append(
            state.reproposals,
            EventKind.PROPOSED,
            None,
            items=joined(items),
            slots=str(len(batch.slots)),
            failures=str(len(batch.failures)),
            reproposal=str(state.reproposals),
        )


def _keep_call(directory: Path, request: Sequence[Message], completions: Sequence[Completion]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(directory / REQUEST_FILE, pretty_json(list(request)).encode())
    write_atomic(directory / COMPLETIONS_FILE, _COMPLETIONS.dump_json(tuple(completions), indent=2))


def _keep_batch(directory: Path, batch: ProposalBatch) -> None:
    """Write the planning call and each slot's calls, repair error or failure under ``directory``."""
    _keep_call(directory / PLAN_DIR, batch.planning_request, batch.planning)
    for outcome in batch.slots:
        slot_dir = directory / SLOTS_DIR / str(outcome.slot)
        _keep_call(slot_dir, outcome.request, outcome.completions)
        if isinstance(outcome, SlotFailure):
            write_atomic(slot_dir / SLOT_FAILURE_FILE, outcome.error.encode())
        elif outcome.repair_error is not None:
            write_atomic(slot_dir / REPAIR_ERROR_FILE, outcome.repair_error.encode())


@dataclass(frozen=True)
class _Item:
    """One proposal item's fixed context: identity, directories, log, policy and services."""

    item_id: str
    directory: Path
    log: EventLog
    policy: LoopPolicy
    services: LoopServices

    def proposal(self, proposal_digest: str) -> TaskProposal:
        proposal = parse((self.directory / PROPOSALS_DIR / f"{proposal_digest}.md").read_text())
        if proposal.digest != proposal_digest:
            raise ValueError(f"{self.item_id}: stored proposal {proposal_digest} parses to {proposal.digest}")
        return proposal

    def keep_proposal(self, proposal: TaskProposal) -> None:
        text = render(proposal).encode()
        (self.directory / PROPOSALS_DIR).mkdir(parents=True, exist_ok=True)
        write_atomic(self.directory / PROPOSALS_DIR / f"{proposal.digest}.md", text)
        write_atomic(self.directory / PROPOSAL_FILE, text)

    def round_dir(self, round: int) -> Path:  # noqa: A002 - matches LedgerEntry.round
        return self.directory / ROUNDS_DIR / str(round)

    def evidence_dir(self, round: int, digest: str) -> Path:  # noqa: A002 - matches LedgerEntry.round
        return evidence_dir(self.directory, round, digest)

    def draft(self, state: ItemState) -> TaskDraft:
        assert state.task_digest is not None
        draft = load_draft(self.round_dir(state.round) / DRAFT_DIR)
        built = task_digest(draft.lowered)
        if built != state.task_digest:
            raise ValueError(f"{self.item_id}: the stored draft has digest {built}, its BUILT event {state.task_digest}")
        return draft

    def site(self, state: ItemState) -> ValidationSite:
        assert state.task_digest is not None
        evidence = self.evidence_dir(state.round, state.task_digest)
        return ValidationSite(self.item_id, state.round, evidence, self.services.ledger)


async def run_item(
    proposal: TaskProposal, origin: ProposalOrigin, policy: LoopPolicy, services: LoopServices
) -> Terminal:
    """Carry ``proposal`` to a terminal, resuming from its event log.

    ``origin`` says whether ``proposal`` came from a ``run_idea`` batch or was handed to the loop
    directly; ``OPENED`` records it. An item already ``ACCEPTED`` or ``REJECTED`` returns at once; an
    ``ABANDONED`` or ``FAILED`` one re-enters where it stopped. An unhandled exception records
    ``TERMINAL(FAILED)`` and propagates.

    Raises:
        ValueError: the log was opened for a different proposal, origin or policy, or the log is
            inconsistent (``loop.events.derive_state``).
    """
    item_id = item_id_for(proposal)
    item = _Item(
        item_id, services.root / ITEMS_DIR / item_id, EventLog(services.root, services.ledger, item_id), policy, services
    )
    _open(item, proposal, origin)
    state = derive_state(item.log.entries())
    if state.terminal in FINAL:
        assert state.terminal is not None
        return state.terminal
    try:
        while True:
            if state.not_before is not None and state.not_before > time.time():
                await asyncio.sleep(state.not_before - time.time())
            await _advance(item, state)
            state = derive_state(item.log.entries())
            if state.terminal is not None:
                return state.terminal
    except Exception as error:
        try:
            _fail(item, error)
        except Exception as record_error:
            error.add_note(f"recording TERMINAL(failed) for {item_id} failed: {record_error!r}")
        raise


def _open(item: _Item, proposal: TaskProposal, origin: ProposalOrigin) -> None:
    entries = events(item.log.entries())
    if not entries:
        item.keep_proposal(proposal)
        item.log.append(
            0,
            EventKind.OPENED,
            proposal.digest,
            proposal=proposal.header.id,
            idea=proposal.header.source.ref,
            origin=origin,
            policy_digest=item.policy.digest,
        )
        return
    opened = entries[0]
    if opened.input_hash != proposal.digest:
        raise ValueError(f"{item.item_id} was opened for proposal {opened.input_hash}, not {proposal.digest}")
    if opened.attrs["origin"] != origin:
        raise ValueError(f"{item.item_id} was opened as a {opened.attrs['origin']} proposal, not {origin}")
    if opened.attrs["policy_digest"] != item.policy.digest:
        raise ValueError(
            f"{item.item_id} was opened under policy {opened.attrs['policy_digest']}, not {item.policy.digest}; "
            "resume a run with the policy.json it started with"
        )


def _fail(item: _Item, error: Exception) -> None:
    entries = events(item.log.entries())
    round = entries[-1].round if entries else 0  # noqa: A001 - matches LedgerEntry.round
    item.log.append(
        round, EventKind.TERMINAL, None, terminal=Terminal.FAILED, reason=_clip(f"{type(error).__name__}: {error}")
    )


async def _advance(item: _Item, state: ItemState) -> None:
    """Run the sub-phase ``state`` names; it appends at least one event or raises."""
    match state.phase:
        case Phase.TRIAGE:
            await _triage(item, state)
        case Phase.BUILD if state.program_digest is None:
            await _author(item, state)
        case Phase.BUILD:
            await _build(item, state)
        case Phase.CONTROLS:
            await _controls(item, state)
        case Phase.TRIALS:
            await _trials(item, state)
        case Phase.DECIDE:
            _decide(item, state)
        case Phase.DONE:
            _close(item, state)


def _reject(item: _Item, state: ItemState, kind: RejectKind, reason: str) -> None:
    item.log.append(state.round, EventKind.TERMINAL, None, terminal=Terminal.REJECTED, kind=kind, reason=_clip(reason))


async def _triage(item: _Item, state: ItemState) -> None:
    """Accept the proposal: no structural check and no rubric sample runs on it."""
    item.log.append(
        0, EventKind.TRIAGED, state.proposal_digest, decision=TriageDecision.ACCEPT, tally=NO_TRIAGE, repairs="0"
    )


def _revised_source(item: _Item, state: ItemState) -> str | None:
    """The program the next authoring revises: none, or the round's program whose build failed."""
    if state.revision is RevisionKind.NONE:
        return None
    assert state.revision is RevisionKind.BUILD_FAILURE, "the loop policy allows no Repair"
    return (item.round_dir(state.round) / PROGRAM_FILE).read_text()


async def _author(item: _Item, state: ItemState) -> None:
    """Adopt the template as the round's ``program.py``; a revision adopts it again."""
    policy = item.policy
    if state.build_revisions > policy.max_build_revisions:
        reason = f"no buildable program after {state.build_revisions - 1} of {policy.max_build_revisions} revisions"
        _reject(item, state, RejectKind.BUDGET, reason)
        return
    revised = _revised_source(item, state)
    program = template_program(item.services.template)
    round_dir = item.round_dir(state.round)
    round_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(round_dir / PROGRAM_FILE, program.source.encode())
    revises = "" if revised is None else sha256_hex(revised.encode())
    item.log.append(state.round, EventKind.AUTHORED, program.digest, revises=revises, revision=state.revision)


def _build_failed(item: _Item, state: ItemState, program_digest: str, step: str, failure: str, noop: bool) -> None:
    relative = Path(ROUNDS_DIR) / str(state.round) / FAILURES_DIR / f"{state.build_revisions}.txt"
    (item.directory / relative).parent.mkdir(parents=True, exist_ok=True)
    write_atomic(item.directory / relative, failure.encode())
    item.log.append(
        state.round,
        EventKind.BUILD_FAILED,
        program_digest,
        step=step,
        failure=_clip(failure),
        failure_file=str(relative),
        noop=str(noop).lower(),
    )


async def _build(item: _Item, state: ItemState) -> None:
    services = item.services
    proposal = item.proposal(state.proposal_digest)
    round_dir = item.round_dir(state.round)
    program = template_program(services.template)
    authored = sha256_hex((round_dir / PROGRAM_FILE).read_bytes())
    if program.digest != state.program_digest or authored != state.program_digest:
        raise ValueError(f"{item.item_id}: {round_dir / PROGRAM_FILE} is not the authored {state.program_digest}")
    async with services.slots:
        try:
            draft = await run_build(
                program, proposal, round_dir, services.root / CACHE_DIR, services.build, state.invalidate, state.round
            )
        except GlmUnavailable:
            raise
        except Exception as error:
            # A BuildFailure or any other exception the program raised costs a revision.
            failure = "".join(traceback.format_exception(error))[-FAILURE_CHARS:]
            step = (error.step or "") if isinstance(error, BuildFailure) else ""
            _build_failed(item, state, program.digest, step, failure, noop=False)
            return
    digest = task_digest(draft.lowered)
    steps = draft.provenance.steps
    item.log.append(
        state.round,
        EventKind.BUILT,
        digest,
        program_digest=program.digest,
        steps=str(len(steps)),
        hits=str(sum(record.status is CacheStatus.HIT for record in steps)),
    )


async def _controls(item: _Item, state: ItemState) -> None:
    """Record a replay of no controls: none is replayed, so none is violated and the round goes on."""
    item.log.append(
        state.round, EventKind.CONTROLS_REPLAYED, state.task_digest, met="0", violated="0", ungraded="0", passed="true"
    )


async def _trials(item: _Item, state: ItemState) -> None:
    services, validation = item.services, item.policy.validation
    draft, site = item.draft(state), item.site(state)
    if not state.solved:
        async with services.slots:
            outcomes = await run_solver(draft, validation, site, services.engine, services.rollout_models)
        stats = Evidence({TrialKind.SOLVER: outcomes}).reward_stats(TrialKind.SOLVER)
        item.log.append(
            state.round,
            EventKind.SOLVED,
            state.task_digest,
            graded=str(stats.graded),
            solved=str(stats.solved),
            timed_out=str(stats.timed_out),
            ungraded=str(len(outcomes) - stats.graded),
        )
    if not state.adversaries_run:
        # No adversary role runs: no per-role counts, and the context digest is "" as for an empty context.
        item.log.append(state.round, EventKind.ADVERSARIES_RUN, state.task_digest, context_digest="")


def retry_wait(backoff: RetryBackoff, retry: int) -> float:
    """The ``retry``-th interval (from 1) of a fresh copy of ``backoff``."""
    fresh = backoff.schedule()
    for _ in range(retry - 1):
        fresh.next_interval()
    return fresh.next_interval()


def _decide(item: _Item, state: ItemState) -> None:
    policy = item.policy
    draft = item.draft(state)
    assert state.task_digest is not None
    evidence_dir = item.evidence_dir(state.round, state.task_digest)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    history = ItemHistory(state.repairs_used, policy.max_repairs, state.band_repairs)
    summary = summarize(load_validation(draft, evidence_dir), policy.validation)
    write_summary(evidence_dir / CALIBRATION_FILE, summary)
    decision = decide(draft, summary, history, policy.band_rules)
    write_decision(evidence_dir / DECISION_FILE, decision)
    attrs = _decision_attrs(decision, state, policy)
    item.log.append(
        state.round, EventKind.DECIDED, state.task_digest, **attrs, notes=joined(note.kind for note in summary.notes)
    )


def _decision_attrs(decision: Decision, state: ItemState, policy: LoopPolicy) -> dict[str, str]:
    counts = {"repairs_used": str(state.repairs_used), "retries_used": str(state.validation_retries)}
    match decision:
        case Accept(summary=summary, band=band):
            return {"decision": DecisionKind.ACCEPT, "band": band, **_pass_rate(summary), **counts}
        case Reject(kind=kind, reasons=reasons):
            return {"decision": DecisionKind.REJECT, "kind": kind, "reasons": _clip("; ".join(reasons)), **counts}
        case Repair(brief=brief, invalidate=invalidate):
            return {
                "decision": DecisionKind.REPAIR,
                "findings": joined(finding.kind for finding in brief.findings),
                "invalidate": joined(invalidate),
                **counts,
                "repairs_used": str(state.repairs_used + 1),
            }
        case Retry(cause=cause, count=count):
            retries = state.validation_retries + 1
            abandon = retries > policy.max_validation_retries
            not_before = time.time() + (0.0 if abandon else retry_wait(policy.retry_backoff, retries))
            return {
                "decision": DecisionKind.RETRY,
                "cause": cause,
                "count": str(count),
                "abandon": str(abandon).lower(),
                "not_before": repr(not_before),
                **counts,
                "retries_used": str(retries),
            }


def _pass_rate(summary: CalibrationSummary) -> dict[str, str]:
    """The synthesis pass rate an accepted task is labelled with, as ``DECIDED`` attrs."""
    assert summary.solve_rate is not None, "an accepted summary has graded solver trials"
    return {
        "solved": str(summary.solver.solved),
        "graded": str(summary.solver.graded),
        "solve_rate": f"{summary.solve_rate:.3f}",
    }


def _close(item: _Item, state: ItemState) -> None:
    closing = state.closing
    assert closing is not None, f"{item.item_id} is DONE without a closing decision"
    attrs = {"terminal": closing.terminal.value, "reason": _clip(closing.reason)}
    if closing.kind is not None:
        attrs["kind"] = closing.kind.value
    if closing.causes:
        attrs["causes"] = cause_counts(closing.causes)
    item.log.append(state.round, EventKind.TERMINAL, None, **attrs)
