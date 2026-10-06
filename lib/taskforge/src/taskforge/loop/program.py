# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-idea and per-item programs of one unattended run.

``run_idea`` proposes ``n`` items for one idea. A failed slot costs only that slot; a batch with zero
proposals re-proposes the idea, up to ``max_idea_reproposals``, then the idea is exhausted.

``run_item`` carries one proposal to a ``Terminal``: triage with bounded rubric repairs, authoring and
building with bounded revisions, control replay, solver and adversary trials, the calibration summary
and review's decision. A ``Repair`` starts the next round: the author revises the condemned program
with the repair brief as ``Revision.failure`` and the build recomputes the steps the repair invalidates.
A ``Retry`` re-enters validation after a backoff, re-running only unsettled trials; retries spent end
the item ``ABANDONED``. An unhandled exception records ``FAILED`` and propagates.

Each loop iteration derives the item's state from its event log, runs the sub-phase the state names
and appends that sub-phase's completion event, so a new process resumes an item by re-running only
the sub-phase that lacks its event; inside validation, settled trials load from their attempt files.
``services.slots`` bounds the phases that call the model or a sandbox (triage, authoring, building,
controls, trials); an item waiting out a retry backoff, deciding or closing holds no slot.

Run root layout::

    items/<item_id>/proposal.md                current proposal; proposals/<digest>.md keeps each one
    items/<item_id>/verdict.json               triage verdict on the current proposal
    items/<item_id>/rounds/<round>/            program.py, program.json, author/, draft/, scratch/,
                                               build-failures/<n>.txt, evidence-<digest12>/
    items/idea--<idea_id>/proposals/<item_id>.md   each proposal as its idea's batch produced it
    cache/                                     the step cache shared by every item
    ledger/<item_id>.jsonl                     the item's spans and events
"""

import asyncio
import logging
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from rigging.timing import ExponentialBackoff

from taskforge.build.author import PROGRAM_FILE, Revision, author, load_program
from taskforge.build.run import DRAFT_DIR, TaskDraft, item_id_for, load_draft, run_build
from taskforge.build.sdk import BuildFailure, BuildServices, record_completions
from taskforge.build.step import CacheStatus
from taskforge.canonical import sha256_hex, write_atomic
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind, Ledger, LedgerEntry, SpanFields, check_item_id, span
from taskforge.llm.client import GlmClient, GlmUnavailable
from taskforge.loop.events import (
    FINAL,
    DecisionKind,
    EventKind,
    ItemState,
    Phase,
    RevisionKind,
    Terminal,
    derive_idea_state,
    derive_state,
    events,
    item_tokens_out,
    joined,
    record_event,
)
from taskforge.loop.policy import LoopPolicy
from taskforge.proposal.model import ProposalFormatError, TaskProposal, parse, render
from taskforge.proposal.source import ProposalSource
from taskforge.review.decision import (
    DECISION_FILE,
    Accept,
    Decision,
    Reject,
    RejectKind,
    Repair,
    Retry,
    load_decision,
    write_decision,
)
from taskforge.review.rules import ItemHistory, decide, staged_repair
from taskforge.triage.checks import Check, CheckContext
from taskforge.triage.program import RubricProgram, evaluate
from taskforge.triage.verdict import ModelCall, TriageDecision, Verdict
from taskforge.validate.adversary import SENTINEL_REPLIES, run_adversaries
from taskforge.validate.calibration import final_reply, solved, summarize, write_summary
from taskforge.validate.controls import ControlVerdict, Tokenize
from taskforge.validate.evidence import Evidence
from taskforge.validate.outcome import Graded, TrialKind
from taskforge.validate.run import backoff_config, load_validation, replay_controls
from taskforge.validate.solver import ValidationSite, run_solver
from taskforge.validate.trials import EngineSettings, RolloutModel, task_digest

logger = logging.getLogger(__name__)

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

FAILURE_CHARS = 6000
"""The tail of a build failure's traceback kept for the author's revision."""
ATTR_CHARS = 500
"""Longest free text kept in an event attribute; the full text stays in the item directory."""
DIGEST_CHARS = 12

NOOP_FAILURE = """\
The revised program produced the identical task, so the findings below still stand. Change the steps \
they name so the task they build changes."""


@dataclass(frozen=True)
class LoopServices[IdeaT]:
    """Everything a run's items share.

    Attributes:
        client: The run's one GLM client.
        source: Turns ideas into proposal batches.
        checks: Triage's structural checks.
        rubric: Triage's rubric program.
        check_context: What the structural checks read besides the proposal.
        template: The builder template the author adapts (``build.template.standard``).
        build: The author's and builder's services; its ledger is ``ledger``.
        engine: Run-wide engine settings; each draft's trials run under its own convention alone.
        rollout_model: The solver; every adversary role wraps it.
        tokenize: The server tokenizer control replay renders transcripts with.
        ledger: The run's ledger; its local JSONL root must be ``root / "ledger"``.
        root: The run root.
        slots: Bounds the model- and sandbox-bound phases running at once across items.
    """

    client: GlmClient
    source: ProposalSource[IdeaT]
    checks: Sequence[Check]
    rubric: RubricProgram
    check_context: CheckContext
    template: ModuleType
    build: BuildServices
    engine: EngineSettings
    rollout_model: RolloutModel
    tokenize: Tokenize
    ledger: Ledger
    root: Path
    slots: asyncio.Semaphore

    def __post_init__(self) -> None:
        if self.build.ledger is not self.ledger:
            raise ValueError("LoopServices.build must record to the run's ledger")


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


def idea_item_id(idea_id: str) -> str:
    return f"{IDEA_PREFIX}{idea_id}"


def _clip(text: str) -> str:
    return text[:ATTR_CHARS]


def _record_calls(fields: SpanFields, model: str, calls: Sequence[ModelCall]) -> None:
    fields.model = model
    fields.tokens_in = sum(call.usage.prompt_tokens for call in calls)
    fields.tokens_out = sum(call.usage.completion_tokens for call in calls)
    fields.tokens_reasoning = sum(call.usage.reasoning_tokens for call in calls)
    fields.attrs["requests"] = str(len(calls))


async def run_idea[IdeaT](
    idea_id: str, idea: IdeaT, policy: LoopPolicy, services: LoopServices[IdeaT]
) -> tuple[TaskProposal, ...]:
    """The proposals of ``idea``, proposing a batch only when its log holds none.

    Each batch's failed slots are recorded as ``SLOT_FAILED`` and its siblings proceed. A batch with
    zero proposals is re-proposed up to ``policy.max_idea_reproposals`` times; then the idea is
    ``IDEA_EXHAUSTED`` and yields nothing. A resumed idea returns its recorded batch without a call.
    """
    item_id = idea_item_id(idea_id)
    log = EventLog(services.root, services.ledger, item_id)
    batch_dir = services.root / ITEMS_DIR / item_id / PROPOSALS_DIR
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
            with span(
                services.ledger, EntryKind.LLM_CALL, item_id=item_id, round=state.reproposals, step="propose"
            ) as f:
                batch = await services.source.propose(idea, policy.proposals_per_idea)
                f.model = services.client.endpoint.model
                completions = (*batch.planning, *(c for slot in batch.slots for c in slot.completions))
                if completions:
                    record_completions(f, completions)
        items = [item_id_for(proposal) for proposal in batch.proposals]
        if len(set(items)) != len(items) or any("," in item for item in items):
            raise ValueError(f"idea {idea_id}: proposal ids must be distinct and comma-free, got {items}")
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
        return self.round_dir(round) / f"evidence-{digest[:DIGEST_CHARS]}"

    def draft(self, state: ItemState) -> TaskDraft:
        assert state.task_digest is not None
        draft = load_draft(self.round_dir(state.round) / DRAFT_DIR)
        built = task_digest(draft.task, draft.execution, draft.convention)
        if built != state.task_digest:
            raise ValueError(f"{self.item_id}: the stored draft has digest {built}, its BUILT event {state.task_digest}")
        return draft

    def site(self, state: ItemState) -> ValidationSite:
        assert state.task_digest is not None
        evidence = self.evidence_dir(state.round, state.task_digest)
        return ValidationSite(self.item_id, state.round, evidence, self.services.ledger)


async def run_item(proposal: TaskProposal, policy: LoopPolicy, services: LoopServices) -> Terminal:
    """Carry ``proposal`` to a terminal, resuming from its event log.

    An item already ``ACCEPTED`` or ``REJECTED`` returns at once; an ``ABANDONED`` or ``FAILED`` one
    re-enters where it stopped. An unhandled exception records ``TERMINAL(FAILED)`` and propagates.

    Raises:
        ValueError: the log was opened for a different proposal or under a different policy, or the
            log is inconsistent (``loop.events.derive_state``).
    """
    item_id = item_id_for(proposal)
    item = _Item(
        item_id, services.root / ITEMS_DIR / item_id, EventLog(services.root, services.ledger, item_id), policy, services
    )
    _open(item, proposal)
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


def _open(item: _Item, proposal: TaskProposal) -> None:
    entries = events(item.log.entries())
    if not entries:
        item.keep_proposal(proposal)
        item.log.append(
            0,
            EventKind.OPENED,
            proposal.digest,
            proposal=proposal.header.id,
            idea=proposal.header.source.ref,
            policy_digest=item.policy.digest,
        )
        return
    opened = entries[0]
    if opened.input_hash != proposal.digest:
        raise ValueError(f"{item.item_id} was opened for proposal {opened.input_hash}, not {proposal.digest}")
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
    services, policy = item.services, item.policy
    proposal = item.proposal(state.proposal_digest)
    model = services.client.endpoint.model
    if state.triage is None:
        async with services.slots:
            with span(services.ledger, EntryKind.LLM_CALL, item_id=item.item_id, round=0, step="triage") as fields:
                verdict = await evaluate(proposal, services.checks, services.rubric, services.check_context)
                _record_calls(fields, model, verdict.calls)
        write_atomic(item.directory / VERDICT_FILE, verdict.to_json().encode())
        tally = verdict.reasons[0] if verdict.rubric else ""
        item.log.append(
            0,
            EventKind.TRIAGED,
            proposal.digest,
            decision=verdict.decision,
            tally=_clip(tally),
            repairs=str(state.triage_repairs),
        )
        return
    assert state.triage is TriageDecision.REPAIR, state.triage
    if state.triage_repairs >= policy.max_triage_repairs:
        reason = f"triage still asks for repair after {state.triage_repairs} of {policy.max_triage_repairs} repairs"
        _reject(item, state, RejectKind.TASK, reason)
        return
    verdict = Verdict.from_json((item.directory / VERDICT_FILE).read_text())
    if verdict.proposal_digest != proposal.digest:
        raise ValueError(f"{item.item_id}: {VERDICT_FILE} is for {verdict.proposal_digest}, not {proposal.digest}")
    async with services.slots:
        try:
            with span(services.ledger, EntryKind.LLM_CALL, item_id=item.item_id, round=0, step="triage.repair") as f:
                repair = await services.rubric.repair(proposal, verdict)
                _record_calls(f, model, (repair.call,))
        except ProposalFormatError as error:
            _reject(item, state, RejectKind.TASK, f"triage repair is not a valid proposal: {error}")
            return
    item.keep_proposal(repair.proposal)
    item.log.append(0, EventKind.TRIAGE_REPAIRED, repair.proposal.digest, from_digest=proposal.digest)


def _revision(item: _Item, state: ItemState) -> Revision | None:
    match state.revision:
        case RevisionKind.NONE:
            return None
        case RevisionKind.BUILD_FAILURE:
            assert state.failure_file is not None
            source = (item.round_dir(state.round) / PROGRAM_FILE).read_text()
            return Revision(source=source, failure=(item.directory / state.failure_file).read_text())
        case RevisionKind.REPAIR:
            assert state.repair_round is not None
            source = (item.round_dir(state.repair_round) / PROGRAM_FILE).read_text()
            return Revision(source=source, failure=_pending_repair(item, state).brief.failure)


def _pending_repair(item: _Item, state: ItemState) -> Repair:
    assert state.repair_round is not None and state.repaired_task_digest is not None
    path = item.evidence_dir(state.repair_round, state.repaired_task_digest) / DECISION_FILE
    decision = load_decision(path)
    if not isinstance(decision, Repair):
        raise ValueError(f"{path} holds {type(decision).__name__}, but the item's log records a repair")
    return decision


async def _author(item: _Item, state: ItemState) -> None:
    services, policy = item.services, item.policy
    if state.build_revisions > policy.max_build_revisions:
        reason = f"no buildable program after {state.build_revisions - 1} of {policy.max_build_revisions} revisions"
        _reject(item, state, RejectKind.BUDGET, reason)
        return
    spent = item_tokens_out(item.log.entries())
    if spent > policy.output_token_budget:
        _reject(item, state, RejectKind.BUDGET, f"{spent} output tokens spent of {policy.output_token_budget}")
        return
    revision = _revision(item, state)
    proposal = item.proposal(state.proposal_digest)
    async with services.slots:
        program = await author(
            proposal, services.template, item.round_dir(state.round), services.build, item.item_id, revision, state.round
        )
    revises = "" if revision is None else sha256_hex(revision.source.encode())
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
    program = load_program(round_dir, proposal)
    if program.digest != state.program_digest:
        raise ValueError(f"{item.item_id}: {round_dir / PROGRAM_FILE} is not the authored {state.program_digest}")
    async with services.slots:
        try:
            draft = await run_build(
                program, proposal, round_dir, services.root / CACHE_DIR, services.build, state.invalidate, state.round
            )
        except GlmUnavailable:
            raise
        except Exception as error:
            # A BuildFailure or any exception the program raised goes back to the author as a revision.
            failure = "".join(traceback.format_exception(error))[-FAILURE_CHARS:]
            step = (error.step or "") if isinstance(error, BuildFailure) else ""
            _build_failed(item, state, program.digest, step, failure, noop=False)
            return
    digest = task_digest(draft.task, draft.execution, draft.convention)
    if digest == state.repaired_task_digest:
        failure = f"{NOOP_FAILURE}\n\n{_pending_repair(item, state).brief.failure}"
        _build_failed(item, state, program.digest, "", failure, noop=True)
        return
    steps = draft.provenance.steps
    item.log.append(
        state.round,
        EventKind.BUILT,
        digest,
        program_digest=program.digest,
        steps=str(len(steps)),
        hits=str(sum(record.status is CacheStatus.HIT for record in steps)),
        staged=str(bool(draft.task.stages)).lower(),
    )


async def _controls(item: _Item, state: ItemState) -> None:
    services = item.services
    draft, site = item.draft(state), item.site(state)
    async with services.slots:
        controls = await replay_controls(draft, item.policy.validation, site, services.engine, services.tokenize)
    verdicts = Counter(control.verdict for control in controls)
    item.log.append(
        state.round,
        EventKind.CONTROLS_REPLAYED,
        state.task_digest,
        met=str(verdicts[ControlVerdict.MET]),
        violated=str(verdicts[ControlVerdict.VIOLATED]),
        ungraded=str(verdicts[ControlVerdict.UNGRADED]),
        passed=str(verdicts[ControlVerdict.MET] == len(controls)).lower(),
    )


async def _trials(item: _Item, state: ItemState) -> None:
    services, validation = item.services, item.policy.validation
    draft, site = item.draft(state), item.site(state)

    async def solver() -> None:
        outcomes = await run_solver(draft, validation, site, services.engine, services.rollout_model)
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

    async def adversaries() -> None:
        by_role = await run_adversaries(draft, validation, site, services.engine, services.rollout_model)
        attrs: dict[str, str] = {}
        for role, outcomes in by_role.items():
            graded = [outcome for outcome in outcomes if isinstance(outcome, Graded)]
            sentinel = SENTINEL_REPLIES.get(role)
            attrs[f"{role}_graded"] = str(len(graded))
            attrs[f"{role}_passes"] = str(sum(solved(outcome) for outcome in graded))
            attrs[f"{role}_sentinel"] = str(
                sum(sentinel is not None and final_reply(outcome.rollout) == sentinel for outcome in graded)
            )
        item.log.append(state.round, EventKind.ADVERSARIES_RUN, state.task_digest, **attrs)

    async with services.slots, asyncio.TaskGroup() as group:
        if not state.solved:
            group.create_task(solver())
        if not state.adversaries_run:
            group.create_task(adversaries())


def retry_wait(backoff: ExponentialBackoff, retry: int) -> float:
    """The ``retry``-th interval (from 1) of a fresh copy of ``backoff``."""
    fresh = ExponentialBackoff(**backoff_config(backoff))
    for _ in range(retry - 1):
        fresh.next_interval()
    return fresh.next_interval()


def _decide(item: _Item, state: ItemState) -> None:
    policy = item.policy
    draft = item.draft(state)
    assert state.task_digest is not None
    evidence_dir = item.evidence_dir(state.round, state.task_digest)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    history = ItemHistory(state.repairs_used, policy.max_repairs, state.prior_band_findings)
    decision: Decision
    if draft.task.stages:
        decision = staged_repair(draft, history)
    else:
        summary = summarize(load_validation(draft, evidence_dir), policy.validation)
        write_summary(evidence_dir / CALIBRATION_FILE, summary)
        decision = decide(draft, summary, history)
    write_decision(evidence_dir / DECISION_FILE, decision)
    item.log.append(state.round, EventKind.DECIDED, state.task_digest, **_decision_attrs(decision, state, policy))


def _decision_attrs(decision: Decision, state: ItemState, policy: LoopPolicy) -> dict[str, str]:
    counts = {"repairs_used": str(state.repairs_used), "retries_used": str(state.validation_retries)}
    match decision:
        case Accept():
            return {"decision": DecisionKind.ACCEPT, **counts}
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


def _close(item: _Item, state: ItemState) -> None:
    closing = state.closing
    assert closing is not None, f"{item.item_id} is DONE without a closing decision"
    attrs = {"terminal": closing.terminal.value, "reason": _clip(closing.reason)}
    if closing.kind is not None:
        attrs["kind"] = closing.kind.value
    item.log.append(state.round, EventKind.TERMINAL, None, **attrs)
