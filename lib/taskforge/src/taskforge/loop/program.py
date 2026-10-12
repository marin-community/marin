# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The loop: coordination over the stages a caller passes, by a transition table.

A ``Loop`` holds the caller's ``Stages``, the run's policy, root, ledger and models, and a transition
table (``loop.events.DEFAULT_TRANSITIONS`` unless the caller passes its own). Its guarantees come from
that table and the stage Protocols: every stage runs through one adapter here that calls the
caller's function, keeps the round's artifacts, and returns one ``Event`` with its ``Exit``;
``loop.events.transition`` moves the item to the table's next stage. Each adapter reads the files
its predecessor in the default table wrote (validate the draft, review ``calibration.json``), so a
caller's table may re-target an edge only to a stage whose inputs that path has produced.

``run_idea`` proposes ``n`` items for one idea. A failed slot costs only that slot; a batch with zero
proposals re-proposes the idea, up to ``max_idea_reproposals``, then the idea is exhausted.

``run_item`` carries one proposal to a ``Terminal``. It derives the item's state from its event log,
then repeatedly runs the stage the state names, appends that stage's event and folds it in, so a new
process resumes an item by re-running only the stage that lacks its event; inside validation, settled
trials load from their attempt files. An unhandled exception, ``GlmUnavailable`` included, records
``TERMINAL(failed)`` and propagates; the next ``run_item`` of the item re-enters that stage. The
caller loops over its ideas and items; there is no run-wide scheduler here.

A program is code, not data, so it cannot be read back from the round directory: the build stage asks
the author again, with the revision the state still holds (``ItemState.revision``), and refuses a
program whose digest differs from the authored one. An author must therefore return the same program for the same
proposal and revision.

Run root layout::

    policy.json                                the run's policy (``loop.policy.install_policy``)
    items/<item_id>/proposal.md                the opened proposal
    items/<item_id>/rounds/<round>/            program.json, draft/, build-failures/<n>.txt
    items/<item_id>/rounds/<round>/evidence-<digest12>/
                                               solver/<trial>/attempt-<n>.json, calibration.json, decision.json
    items/idea--<idea_id>/idea.json            the idea as ``Loop.describe_idea`` records it
    items/idea--<idea_id>/proposals/<item_id>.md   each proposal as its idea's batch produced it
    cache/                                     the step cache shared by every item
    ledger/<item_id>.jsonl                     the item's spans and events
"""

import asyncio
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskforge.atomic_file import write_atomic
from taskforge.builder.author import PROGRAM_MANIFEST_FILE, Revision, RevisionKind, program_manifest
from taskforge.builder.run import DRAFT_DIR, TaskDraft, item_id_for, load_draft
from taskforge.builder.sdk import BuildFailure
from taskforge.builder.step import CacheStatus
from taskforge.content_hash import pretty_json
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind, Ledger, LedgerEntry, check_item_id, span
from taskforge.llm.client import GlmUnavailable, ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import RecordedModel
from taskforge.loop.events import (
    DEFAULT_TRANSITIONS,
    EXIT,
    FINAL,
    Event,
    EventKind,
    Exit,
    ItemState,
    ProposalOrigin,
    Stage,
    Terminal,
    Transitions,
    check_transitions,
    derive_idea_state,
    derive_state,
    events,
    joined,
    record_event,
    transition,
)
from taskforge.loop.policy import LoopPolicy
from taskforge.loop.stages import Stages
from taskforge.proposal.model import TaskProposal, parse, render
from taskforge.review.decision import (
    DECISION_FILE,
    Accept,
    Decision,
    Modify,
    Reject,
    Retry,
    load_decision,
    write_decision,
)
from taskforge.triage.verdict import TriageDecision
from taskforge.validate.calibration import CalibrationSummary, Finding, load_summary
from taskforge.validate.evidence import Incomplete
from taskforge.validate.run import CALIBRATION_FILE
from taskforge.validate.solver import ModelFactory, ValidationSite
from taskforge.validate.trials import EngineSettings, RetryBackoff, task_digest

ITEMS_DIR = "items"
CACHE_DIR = "cache"
LEDGER_DIR = "ledger"
ROUNDS_DIR = "rounds"
PROPOSALS_DIR = "proposals"
PROPOSAL_FILE = "proposal.md"
FAILURES_DIR = "build-failures"
IDEA_PREFIX = "idea--"
IDEA_FILE = "idea.json"
PROPOSE_STEP = "propose"

FAILURE_CHARS = 6000
"""The tail of a build failure's traceback kept for the author's revision."""
ATTR_CHARS = 500
"""Longest free text kept in an event attribute; the full text stays in the item directory."""
DIGEST_CHARS = 12


@dataclass(frozen=True)
class Loop[IdeaT]:
    """Everything a run's items share, and the stages and transition table that carry them.

    Attributes:
        stages: The caller's function for each stage.
        policy: Every bound of the run (``loop.policy.install_policy`` writes and checks it).
        root: The run root.
        ledger: The run's ledger. It must write each item's entries to ``root / "ledger"`` (a
            ``JsonlLedger`` there, or a ledger that also writes one); ``EventLog.append`` refuses one
            that does not on the first event.
        engine: Run-wide engine settings for validation trials.
        rollout_models: Builds each solver trial's model, recording under the trial's step.
        model: The model client proposing and building call, through a ``RecordedModel``.
        sampling: The sampling of every proposing and building call.
        describe_idea: The JSON-serialisable record of an idea, written once to its ``idea.json``.
        transitions: The next stage for each stage and exit; checked by ``check_transitions``.
    """

    stages: Stages[IdeaT]
    policy: LoopPolicy
    root: Path
    ledger: Ledger
    engine: EngineSettings
    rollout_models: ModelFactory
    model: ModelClient
    sampling: LLMPolicy
    describe_idea: Callable[[IdeaT], Mapping[str, object]]
    transitions: Transitions = DEFAULT_TRANSITIONS

    def __post_init__(self) -> None:
        check_transitions(self.transitions)


class EventLog:
    """One item's events: read from its JSONL file under ``root / "ledger"``, appended through the run's ledger.

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

    def append(self, round: int, event: Event) -> LedgerEntry:  # noqa: A002 - matches LedgerEntry.round
        """Record ``event`` as the item's next event and return the row as read back from disk."""
        seq = len(events(self.entries())) + 1
        record_event(self.ledger, self.item_id, round, seq, event)
        recorded = events(self.entries())
        if len(recorded) != seq:
            raise ValueError(f"event {seq} of {self.item_id} did not reach {self.path}; the ledger writes elsewhere")
        return recorded[-1]


def evidence_dir(round_dir: Path, digest: str) -> Path:
    """The directory a round keeps the validation evidence, calibration and decision of the task with ``digest`` in."""
    return round_dir / f"evidence-{digest[:DIGEST_CHARS]}"


def idea_item_id(idea_id: str) -> str:
    return f"{IDEA_PREFIX}{idea_id}"


def _clip(text: str) -> str:
    return text[:ATTR_CHARS]


async def run_idea[IdeaT](loop: Loop[IdeaT], idea_id: str, idea: IdeaT) -> tuple[TaskProposal, ...]:
    """The proposals of ``idea``, proposing a batch only when its log holds none.

    Each batch's failed slots are recorded as ``SLOT_FAILED`` and its siblings proceed. A batch with
    zero proposals is re-proposed up to ``policy.max_idea_reproposals`` times; then the idea is
    ``IDEA_EXHAUSTED`` and yields nothing. A resumed idea returns its recorded batch without a call.
    The idea's record and its batch's proposals are kept in the idea's directory.
    """
    item_id = idea_item_id(idea_id)
    log = EventLog(loop.root, loop.ledger, item_id)
    idea_dir = loop.root / ITEMS_DIR / item_id
    batch_dir = idea_dir / PROPOSALS_DIR
    idea_dir.mkdir(parents=True, exist_ok=True)
    if not (idea_dir / IDEA_FILE).exists():
        write_atomic(idea_dir / IDEA_FILE, pretty_json(loop.describe_idea(idea)).encode())
    while True:
        state = derive_idea_state(log.entries())
        if state.items is not None:
            return tuple(parse((batch_dir / f"{item}.md").read_text()) for item in state.items)
        if state.exhausted:
            return ()
        reproposal = state.reproposals
        if reproposal > loop.policy.max_idea_reproposals:
            log.append(reproposal, Event(EventKind.IDEA_EXHAUSTED, None, {"reproposals": str(reproposal)}))
            return ()
        model = RecordedModel(
            client=loop.model,
            policy=loop.sampling,
            ledger=loop.ledger,
            item_id=item_id,
            round=reproposal,
            step=PROPOSE_STEP,
        )
        with span(loop.ledger, EntryKind.STAGE, item_id=item_id, round=reproposal, step=PROPOSE_STEP):
            batch = await loop.stages.propose(idea, loop.policy.proposals_per_idea, model)
        items = [item_id_for(proposal) for proposal in batch.proposals]
        if len(set(items)) != len(items) or any("," in item for item in items):
            raise ValueError(f"idea {idea_id}: proposal ids must be distinct and comma-free, got {items}")
        batch_dir.mkdir(parents=True, exist_ok=True)
        for item, proposal in zip(items, batch.proposals, strict=True):
            write_atomic(batch_dir / f"{item}.md", render(proposal).encode())
        for failure in batch.failures:
            attrs = {"slot": str(failure.slot), "error": _clip(failure.error)}
            log.append(reproposal, Event(EventKind.SLOT_FAILED, None, attrs))
        attrs = {
            "items": joined(items),
            "slots": str(len(batch.proposals) + len(batch.failures)),
            "failures": str(len(batch.failures)),
            "reproposal": str(reproposal),
        }
        log.append(reproposal, Event(EventKind.PROPOSED, None, attrs))


@dataclass(frozen=True)
class _Item:
    """One proposal item's fixed context: its loop, identity, directory and log."""

    loop: Loop[Any]
    item_id: str
    directory: Path
    log: EventLog

    def proposal(self, proposal_digest: str) -> TaskProposal:
        proposal = parse((self.directory / PROPOSAL_FILE).read_text())
        if proposal.digest != proposal_digest:
            raise ValueError(f"{self.item_id}: stored proposal {proposal_digest} parses to {proposal.digest}")
        return proposal

    def keep_proposal(self, proposal: TaskProposal) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        write_atomic(self.directory / PROPOSAL_FILE, render(proposal).encode())

    def round_dir(self, round: int) -> Path:  # noqa: A002 - matches LedgerEntry.round
        return self.directory / ROUNDS_DIR / str(round)

    def draft(self, state: ItemState) -> TaskDraft:
        assert state.task_digest is not None
        draft = load_draft(self.round_dir(state.round) / DRAFT_DIR)
        built = task_digest(draft.task)
        if built != state.task_digest:
            raise ValueError(f"{self.item_id}: the stored draft has digest {built}, its BUILT event {state.task_digest}")
        return draft

    def site(self, state: ItemState) -> ValidationSite:
        assert state.task_digest is not None
        evidence = evidence_dir(self.round_dir(state.round), state.task_digest)
        return ValidationSite(self.item_id, state.round, evidence, self.loop.ledger)


async def run_item(loop: Loop[Any], proposal: TaskProposal, origin: ProposalOrigin) -> Terminal:
    """Carry ``proposal`` to a terminal, resuming from its event log.

    ``origin`` says whether ``proposal`` came from a ``run_idea`` batch or was handed to the loop
    directly; ``OPENED`` records it. An item already ``ACCEPTED`` or ``REJECTED`` returns at once; an
    ``ABANDONED`` or ``FAILED`` one re-enters where it stopped. An unhandled exception records
    ``TERMINAL(failed)`` and propagates.

    Raises:
        ValueError: the log was opened for a different proposal, origin or policy, or the log is
            inconsistent (``loop.events.transition``).
    """
    item_id = item_id_for(proposal)
    item = _Item(loop, item_id, loop.root / ITEMS_DIR / item_id, EventLog(loop.root, loop.ledger, item_id))
    _open(item, proposal, origin)
    state = derive_state(item.log.entries(), loop.transitions)
    if state.terminal in FINAL:
        assert state.terminal is not None
        return state.terminal
    try:
        while True:
            await _wait(state.not_before)
            event = await ACTIONS[state.stage](item, state)
            state = transition(state, item.log.append(state.round, event), loop.transitions)
            if state.terminal is not None:
                return state.terminal
    except Exception as error:
        try:
            _fail(item, error)
        except Exception as record_error:
            error.add_note(f"recording TERMINAL(failed) for {item_id} failed: {record_error!r}")
        raise


async def _wait(not_before: float | None) -> None:
    if not_before is not None and not_before > time.time():
        await asyncio.sleep(not_before - time.time())


def _open(item: _Item, proposal: TaskProposal, origin: ProposalOrigin) -> None:
    policy_digest = item.loop.policy.digest
    entries = events(item.log.entries())
    if not entries:
        item.keep_proposal(proposal)
        attrs = {
            "proposal": proposal.header.id,
            "idea": proposal.header.source.ref,
            "origin": origin,
            "policy_digest": policy_digest,
        }
        item.log.append(0, Event(EventKind.OPENED, proposal.digest, attrs))
        return
    first = entries[0]
    if first.input_hash != proposal.digest:
        raise ValueError(f"{item.item_id} was opened for proposal {first.input_hash}, not {proposal.digest}")
    if first.attrs["origin"] != origin:
        raise ValueError(f"{item.item_id} was opened as a {first.attrs['origin']} proposal, not {origin}")
    if first.attrs["policy_digest"] != policy_digest:
        raise ValueError(
            f"{item.item_id} was opened under policy {first.attrs['policy_digest']}, not {policy_digest}; "
            "resume a run with the policy.json it started with"
        )


def _fail(item: _Item, error: Exception) -> None:
    entries = events(item.log.entries())
    round = entries[-1].round if entries else 0  # noqa: A001 - matches LedgerEntry.round
    attrs = {"terminal": Terminal.FAILED, "reason": _clip(f"{type(error).__name__}: {error}")}
    item.log.append(round, Event(EventKind.TERMINAL, None, attrs))


async def _triage(item: _Item, state: ItemState) -> Event:
    decision = item.loop.stages.triage(item.proposal(state.proposal_digest))
    match decision:
        case TriageDecision.ACCEPT:
            way = Exit.ACCEPT
        case TriageDecision.REJECT:
            way = Exit.REJECT
    return Event(EventKind.TRIAGED, state.proposal_digest, {EXIT: way})


def _findings_text(findings: Sequence[Finding]) -> str:
    return "\n\n".join(f"{finding.kind.value}:\n{finding.detail}" for finding in findings)


def _revision(item: _Item, kind: RevisionKind, round: int, build_revisions: int) -> Revision | None:  # noqa: A002
    """What an authoring of round ``round`` revises: nothing, the last build failure, or review's findings."""
    match kind:
        case RevisionKind.NONE:
            return None
        case RevisionKind.BUILD_FAILURE:
            failure = item.round_dir(round) / FAILURES_DIR / f"{build_revisions}.txt"
            return Revision(kind=kind, detail=failure.read_text())
        case RevisionKind.MODIFY:
            decided = [
                entry
                for entry in events(item.log.entries())
                if entry.step == EventKind.DECIDED and entry.round == round - 1
            ]
            assert decided and decided[-1].input_hash is not None, f"{item.item_id}: round {round} has no modify"
            path = evidence_dir(item.round_dir(round - 1), decided[-1].input_hash) / DECISION_FILE
            decision = load_decision(path)
            assert isinstance(decision, Modify), f"{path} is not a Modify"
            return Revision(kind=kind, detail=_findings_text(decision.findings))


async def _author(item: _Item, state: ItemState) -> Event:
    """Ask the author for the round's program and keep its manifest as ``program.json``."""
    proposal = item.proposal(state.proposal_digest)
    revision = _revision(item, state.revision, state.round, state.build_revisions)
    program = item.loop.stages.author(proposal, revision)
    round_dir = item.round_dir(state.round)
    round_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(round_dir / PROGRAM_MANIFEST_FILE, pretty_json(program_manifest(program)).encode())
    return Event(EventKind.AUTHORED, program.digest, {EXIT: Exit.AUTHORED, "revision": state.revision})


async def _build(item: _Item, state: ItemState) -> Event:
    """Build the authored program; a build exception other than ``GlmUnavailable`` costs a revision."""
    loop = item.loop
    proposal = item.proposal(state.proposal_digest)
    revision = _revision(item, state.revision, state.round, state.build_revisions)
    program = loop.stages.author(proposal, revision)
    if program.digest != state.program_digest:
        raise ValueError(
            f"{item.item_id}: the author now returns program {program.digest}, not the authored "
            f"{state.program_digest}; relaunch with the program the run started with"
        )
    round_dir = item.round_dir(state.round)
    try:
        draft = await loop.stages.build(
            program, proposal, round_dir, loop.root / CACHE_DIR, loop.model, loop.sampling, loop.ledger, state.round
        )
    except GlmUnavailable:
        raise
    except Exception as error:
        failures = state.build_revisions + 1
        relative = Path(ROUNDS_DIR) / str(state.round) / FAILURES_DIR / f"{failures}.txt"
        failure = "".join(traceback.format_exception(error))[-FAILURE_CHARS:]
        (item.directory / relative).parent.mkdir(parents=True, exist_ok=True)
        write_atomic(item.directory / relative, failure.encode())
        attrs = {
            EXIT: Exit.REVISE if failures <= loop.policy.max_build_revisions else Exit.REJECT,
            "step": (error.step or "") if isinstance(error, BuildFailure) else "",
            "failure": _clip(failure),
            "failure_file": str(relative),
        }
        return Event(EventKind.BUILD_FAILED, program.digest, attrs)
    steps = draft.provenance.steps
    attrs = {
        EXIT: Exit.BUILT,
        "program_digest": program.digest,
        "steps": str(len(steps)),
        "hits": str(sum(record.status is CacheStatus.HIT for record in steps)),
    }
    return Event(EventKind.BUILT, task_digest(draft.task), attrs)


async def _validate(item: _Item, state: ItemState) -> Event:
    loop = item.loop
    summary = await loop.stages.validate(
        item.draft(state), loop.policy.validation, item.site(state), loop.engine, loop.rollout_models
    )
    status = summary.status
    attrs = {
        EXIT: Exit.VALIDATED,
        "graded": str(summary.solver.graded),
        "solved": str(summary.solver.solved),
        "timed_out": str(summary.solver.timed_out),
        "ungraded": str(sum(status.causes.values()) if isinstance(status, Incomplete) else 0),
        "findings": joined(finding.kind for finding in summary.findings),
    }
    return Event(EventKind.VALIDATED, state.task_digest, attrs)


async def _review(item: _Item, state: ItemState) -> Event:
    """Decide from the round's ``calibration.json`` and keep the decision as ``decision.json`` beside it."""
    loop = item.loop
    evidence = item.site(state).evidence_dir
    summary = load_summary(evidence / CALIBRATION_FILE)
    decision = loop.stages.review(summary, loop.policy.band_rules)
    write_decision(evidence / DECISION_FILE, decision)
    return Event(EventKind.DECIDED, state.task_digest, _decision_attrs(decision, summary, state, loop.policy))


def _retry_wait(backoff: RetryBackoff, retry: int) -> float:
    """The ``retry``-th interval (from 1) of a fresh copy of ``backoff``."""
    fresh = backoff.schedule()
    for _ in range(retry - 1):
        fresh.next_interval()
    return fresh.next_interval()


def _decision_attrs(
    decision: Decision, summary: CalibrationSummary, state: ItemState, policy: LoopPolicy
) -> dict[str, str]:
    """The ``DECIDED`` attrs of ``decision``; a ``Retry`` past ``max_validation_retries`` exits abandon."""
    used = {"retries_used": str(state.retries)}
    match decision:
        case Accept(band=band):
            assert summary.solve_rate is not None, "an accepted summary has graded solver trials"
            return {
                EXIT: Exit.ACCEPT,
                "band": band,
                "solved": str(summary.solver.solved),
                "graded": str(summary.solver.graded),
                "solve_rate": f"{summary.solve_rate:.3f}",
                **used,
            }
        case Reject(kind=kind, reasons=reasons):
            return {EXIT: Exit.REJECT, "kind": kind, "reasons": _clip("; ".join(reasons)), **used}
        case Retry(cause=cause, count=count):
            retries = state.retries + 1
            spent = {"cause": cause, "count": str(count), "retries_used": str(retries)}
            if state.retries >= policy.max_validation_retries:
                return {EXIT: Exit.ABANDON, **spent}
            not_before = time.time() + _retry_wait(policy.retry_backoff, retries)
            return {EXIT: Exit.RETRY, **spent, "not_before": repr(not_before)}
        case Modify(findings=findings):
            return {EXIT: Exit.MODIFY, "findings": joined(finding.kind for finding in findings), **used}


async def _close(item: _Item, state: ItemState) -> Event:
    closing = state.closing
    assert closing is not None, f"{item.item_id} is DONE without a closing"
    attrs = {"terminal": closing.terminal.value, "reason": _clip(closing.reason)}
    if closing.kind is not None:
        attrs["kind"] = closing.kind.value
    return Event(EventKind.TERMINAL, None, attrs)


ACTIONS: Mapping[Stage, Callable[[_Item, ItemState], Awaitable[Event]]] = {
    Stage.TRIAGE: _triage,
    Stage.AUTHOR: _author,
    Stage.BUILD: _build,
    Stage.VALIDATE: _validate,
    Stage.REVIEW: _review,
    Stage.DONE: _close,
}
"""The adapter that runs each stage and returns its event."""
