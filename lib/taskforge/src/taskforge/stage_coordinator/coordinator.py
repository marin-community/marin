# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a workpiece through the caller's stages, checkpointing after every stage event.

Only completed evaluation supports ACCEPTED or REJECTED. Exhausted triage,
author/build, validation or modify budgets produce resumable ABANDONED. Other
execution failures and unavailable evidence produce resumable HELD.

An explicit relaunch of ABANDONED starts a fresh budget window and records its
budget digest. Cumulative repair, revision and round numbers keep earlier files
intact. A review hold resumes validation; a stage exception resumes that stage.
Accepted and rejected workpieces remain final. Legacy FAILED records resume as HELD.

Triage repairs are stored in repaired/<k>.json. Author calls write program.json
and optional program.py to rounds/<n>/authored/<k>/, where k counts author/build
failures. Failure text in build-failures/<k>.txt supplies the next revision.
Programs without source must be deterministic when the build asks the author
again. Completed build steps are reused from the step store.

Triage author_state is preserved for every author call. Only an author answering
a MODIFY revision may change verification; subsequent builds and revisions use
that recorded change. The run layout is documented in lib/taskforge/README.md.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from rigging.filesystem.atomic import write_bytes_atomic

from taskforge.content_hash import canonical_json, digest, pretty_json
from taskforge.ledger.records import check_workpiece_id
from taskforge.llm.client import ModelClient, ModelRequestRejected, ModelUnavailable
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.structured import StructuredOutputError
from taskforge.stage10_proposal.model import ProposalFormatError, TaskProposal, parse, render
from taskforge.stage20_triage.verdict import TriageDecision
from taskforge.stage30_author.author import (
    PROGRAM_FILE,
    PROGRAM_MANIFEST_FILE,
    ROUNDS_DIR,
    AuthorContext,
    AuthoredProgram,
    AuthorFailure,
    Revision,
    RevisionKind,
    program_manifest,
)
from taskforge.stage30_author.program import RLEnvBuildProgram
from taskforge.stage40_build_with_program.context import BuildExpired, BuildFailure, BuildInfrastructureFailure
from taskforge.stage40_build_with_program.failure_text import BuildCrash, error_line, failure_head, failure_text
from taskforge.stage40_build_with_program.infrastructure import (
    InfrastructureCause,
    authored_failure,
    infrastructure_failure,
    storage_fault,
)
from taskforge.stage40_build_with_program.run import DRAFT_DIR, ResearchRequirement, load_draft
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import CacheStatus
from taskforge.stage50_validate.round import ValidationContext, load_validation, write_validation
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Hold, Modify, Reject, RejectKind, Retry
from taskforge.stage60_review.rules import evidence_gaps
from taskforge.stage_coordinator.event_log import WORKPIECES_DIR, WorkpieceEventLog, WorkpieceEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import (
    AUTHOR_STATE_ATTR,
    EXIT,
    FINAL,
    REASON_ATTR,
    TERMINAL_ATTR,
    TRIAGE_REJECTED,
    VERIFICATION_CHANGE_ATTR,
    EventKind,
    Exit,
    Stage,
    Terminal,
    WorkpieceEvent,
    WorkpieceState,
    derive_state,
    transition,
)

STORE_DIR = "store"
AUTHORED_DIR = "authored"
REPAIRED_DIR = "repaired"
FAILURES_DIR = "build-failures"
DECISION_FILE = "decision.json"
BUDGET_DIGEST = "budget_digest"
MODEL_DIGEST = "model_digest"
RESEARCH = "research"
FAILURE_CHARS = 6000
"""The length of the traceback tail kept from a build or author failure for the author's revision."""
ATTR_CHARS = 500
"""The length of the failure's head kept in a ``BUILD_FAILED`` or ``AUTHOR_FAILED`` event.

The failure file keeps the rest.
"""


@dataclass(frozen=True)
class WorkpieceBudget:
    """The triage rewrites, build revisions, validation retries and MODIFY rounds each workpiece may spend.

    Attributes:
        max_triage_repairs: The REPAIR verdicts a workpiece may answer with a rewrite that triage judges
            again. Zero abandons on every REPAIR verdict; a consumer using the surface sets one.
        max_build_revisions: The build and author failures a workpiece may revise after. The next failure
            abandons it.
        max_validation_retries: The review ``Retry`` decisions a workpiece may follow before it is abandoned.
        max_modify_rounds: The review ``Modify`` decisions a workpiece may follow into a new round. The next
            one abandons it; zero abandons on the first.
    """

    max_triage_repairs: int
    max_build_revisions: int
    max_validation_retries: int
    max_modify_rounds: int


@dataclass(frozen=True)
class StageCoordinator:
    """The stage functions, budget and storage shared by a run's workpieces.

    Attributes:
        stages: The caller's function for each stage.
        budget: The budget of every workpiece. ``OPENED`` records its digest.
        root: The run root holding ``workpieces/`` and the step store in ``store/``.
        event_store: The store checkpointed after each stage event and terminal.
        model: The model build steps call.
        sampling: The sampling policy build steps pass to ``model``.
        services: The build host's machines, images, web tools, ledger and agent-loop settings, passed
            to every build.
        research: The research requirement of every build (``run_build``'s ``research``). ``OPENED``
            records it.
        model_digest: The model part of every step key, usually
            ``run.build_model_digest(model, sampling, services)``. ``OPENED`` records it.
    """

    stages: StageFunctions
    budget: WorkpieceBudget
    root: Path
    event_store: WorkpieceEventStore
    model: ModelClient
    sampling: LLMPolicy
    services: RLEnvBuildServices
    research: ResearchRequirement
    model_digest: str


@dataclass(frozen=True)
class _Workpiece:
    coordinator: StageCoordinator
    workpiece_id: str
    proposal: TaskProposal
    log: WorkpieceEventLog

    @property
    def directory(self) -> Path:
        return self.coordinator.root / WORKPIECES_DIR / self.workpiece_id

    def round_dir(self, round: int) -> Path:  # noqa: A002 - the workpiece's round
        return self.directory / ROUNDS_DIR / str(round)

    def authored_dir(self, state: WorkpieceState) -> Path:
        """The directory of the author call made at ``state``, keyed by the build and author failures so far."""
        return self.round_dir(state.round) / AUTHORED_DIR / str(state.build_revisions)

    def author_context(self, state: WorkpieceState) -> AuthorContext:
        """The context of the author call made at ``state``: the workpiece, its round and its directory."""
        return AuthorContext(self.workpiece_id, state.round, self.directory)

    def proposal_at(self, state: WorkpieceState) -> TaskProposal:
        """The proposal the stages run at ``state``: the opened one, or the last repair read back from its file.

        Raises:
            ValueError: the repaired proposal's file holds a proposal other than the one its event records.
        """
        if state.triage_repairs == 0:
            return self.proposal
        path = self.directory / _repaired_file(state.triage_repairs)
        proposal = parse(path.read_text())
        if digest(proposal) != state.proposal_digest:
            raise ValueError(f"{path} holds proposal {digest(proposal)}, not the repaired {state.proposal_digest}")
        return proposal

    def built_proposal(self, state: WorkpieceState) -> TaskProposal:
        """The proposal builds run at ``state``: ``proposal_at``, revised by the state's verification change."""
        change = state.verification_change
        proposal = self.proposal_at(state)
        return proposal if change is None else change.revise(proposal)


def _repaired_file(repairs: int) -> Path:
    """The path, in the workpiece directory, of the proposal after ``repairs`` triage repairs."""
    return Path(REPAIRED_DIR) / f"{repairs}.json"


async def run_workpiece(
    coordinator: StageCoordinator, workpiece_id: str, proposal: TaskProposal, log: WorkpieceEventLog
) -> Terminal:
    """Run or resume a workpiece, restoring its event log when the caller's log is empty.

    Final decisions return without running stages. ABANDONED starts a fresh budget
    window; HELD resumes the interrupted stage with the existing budget. Other run
    settings must still match the original proposal, model and research digests.

    Stage exceptions and cancellation are checkpointed as HELD and re-raised.
    Persistence errors are propagated; an unavailable store may prevent a checkpoint.
    Invalid ids, mismatched settings and corrupt logs are rejected before execution.
    """
    check_workpiece_id(workpiece_id)
    stored = coordinator.event_store.load(workpiece_id)
    if not log.events() and stored is not None:
        log.restore(stored)
    workpiece = _Workpiece(coordinator, workpiece_id, proposal, log)
    _open(workpiece)
    state = derive_state(log.events())
    if state.terminal is not None and state.terminal in FINAL:
        if stored != log.events():
            coordinator.event_store.save(workpiece_id, log.events())
        return state.terminal
    try:
        if state.terminal is Terminal.ABANDONED:
            resumed = WorkpieceEvent(EventKind.RESUMED, state.round, None, {BUDGET_DIGEST: digest(coordinator.budget)})
            state = transition(state, log.append(resumed))
        coordinator.event_store.save(workpiece_id, log.events())
        state = await _advance(workpiece, state)
    except (Exception, asyncio.CancelledError) as error:
        exhausted = False
        # A failed save after a conclusive decision must not append an illegal event
        # after its final terminal. Retain that decision for the next save attempt.
        if derive_state(log.events()).terminal not in FINAL:
            infrastructure = infrastructure_failure(error)
            exhausted = infrastructure is not None and infrastructure.cause == InfrastructureCause.BUDGET_EXHAUSTED
            attrs = {
                TERMINAL_ATTR: Terminal.ABANDONED.value if exhausted else Terminal.HELD.value,
                REASON_ATTR: f"{type(error).__name__}: {error}",
            }
            if exhausted:
                attrs["kind"] = RejectKind.BUDGET
            log.append(WorkpieceEvent(EventKind.TERMINAL, log.events()[-1].round, None, attrs))
        coordinator.event_store.save(workpiece_id, log.events())
        if exhausted:
            return Terminal.ABANDONED
        raise
    coordinator.event_store.save(workpiece_id, log.events())
    assert state.terminal is not None
    return state.terminal


async def _advance(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceState:
    while True:
        event = await ADAPTERS[state.stage](workpiece, state)
        state = transition(state, workpiece.log.append(event))
        workpiece.coordinator.event_store.save(workpiece.workpiece_id, workpiece.log.events())
        if state.terminal is not None:
            return state


def _open(workpiece: _Workpiece) -> None:
    proposal_digest = digest(workpiece.proposal)
    coordinator = workpiece.coordinator
    attrs = {
        BUDGET_DIGEST: digest(coordinator.budget),
        MODEL_DIGEST: coordinator.model_digest,
        RESEARCH: str(coordinator.research),
    }
    events = workpiece.log.events()
    if not events:
        workpiece.log.append(WorkpieceEvent(EventKind.OPENED, 0, proposal_digest, attrs))
        return
    first = events[0]
    if first.digest != proposal_digest:
        raise ValueError(f"{workpiece.workpiece_id} was opened for proposal {first.digest}, not {proposal_digest}")
    budget_event = next((event for event in reversed(events) if event.kind is EventKind.RESUMED), first)
    if (
        budget_event.attrs[BUDGET_DIGEST] != attrs[BUDGET_DIGEST]
        and derive_state(events).terminal is not Terminal.ABANDONED
    ):
        raise ValueError(
            f"{workpiece.workpiece_id} was opened under budget {budget_event.attrs[BUDGET_DIGEST]}, "
            f"not {attrs[BUDGET_DIGEST]}"
        )
    if first.attrs[MODEL_DIGEST] != attrs[MODEL_DIGEST]:
        raise ValueError(
            f"{workpiece.workpiece_id} was opened with model digest {first.attrs[MODEL_DIGEST]}, "
            f"not {attrs[MODEL_DIGEST]}"
        )
    if first.attrs[RESEARCH] != attrs[RESEARCH]:
        raise ValueError(
            f"{workpiece.workpiece_id} was opened with research {first.attrs[RESEARCH]}, not {attrs[RESEARCH]}"
        )


async def _triage(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Triage the workpiece's proposal and exit on the verdict's decision.

    An ACCEPT records the verdict's ``author_state`` as canonical JSON on the event, and a REJECT the
    verdict's reasons. A REPAIR verdict within ``max_triage_repairs`` has ``StageFunctions.repair``
    rewrite the proposal, writes the rewrite to ``repaired/<k>.json``, ``k`` counting the rewrites, and
    exits repair with the rewrite's digest. Exhaustion exits abandon; an invalid rewrite exits hold.
    """
    stages = workpiece.coordinator.stages
    proposal = workpiece.proposal_at(state)
    verdict = await stages.triage(proposal)
    spent = {"decision": verdict.decision.value, "repairs": str(state.triage_repairs)}
    if verdict.decision is TriageDecision.ACCEPT:
        attrs = {EXIT: Exit.ACCEPT, **spent, AUTHOR_STATE_ATTR: canonical_json(verdict.author_state)}
        return WorkpieceEvent(EventKind.TRIAGED, state.round, state.proposal_digest, attrs)
    reason = f"{TRIAGE_REJECTED}: {'; '.join(verdict.reasons)}"
    budget = workpiece.coordinator.budget.max_triage_repairs
    way = Exit.HOLD
    if verdict.decision is TriageDecision.REPAIR and state.triage_repairs - state.triage_repairs_at_resume >= budget:
        way = Exit.ABANDON
        spent_repairs = state.triage_repairs - state.triage_repairs_at_resume
        reason = f"triage returned REPAIR with {spent_repairs} of {budget} rewrites spent"
    elif verdict.decision is TriageDecision.REPAIR:
        try:
            rewrite = await stages.repair(proposal, verdict)
        except ProposalFormatError as error:
            reason = f"the rewrite for the REPAIR verdict is not a valid proposal: {error}"
        else:
            return _repaired(workpiece, state, rewrite)
    attrs = {EXIT: way, **spent, REASON_ATTR: reason[:ATTR_CHARS]}
    return WorkpieceEvent(EventKind.TRIAGED, state.round, state.proposal_digest, attrs)


def _repaired(workpiece: _Workpiece, state: WorkpieceState, proposal: TaskProposal) -> WorkpieceEvent:
    """Write the rewritten ``proposal`` to ``repaired/<k>.json`` and return the ``TRIAGED`` event exiting repair."""
    repairs = state.triage_repairs + 1
    relative = _repaired_file(repairs)
    write_bytes_atomic(str(workpiece.directory / relative), render(proposal).encode())
    attrs = {
        EXIT: Exit.REPAIR,
        "decision": TriageDecision.REPAIR.value,
        "repairs": str(repairs),
        "triaged_digest": state.proposal_digest,
        "proposal_file": str(relative),
    }
    return WorkpieceEvent(EventKind.TRIAGED, state.round, digest(proposal), attrs)


def _revision(workpiece: _Workpiece, state: WorkpieceState) -> Revision | None:
    """Read what the round's author call revises: nothing, the last build or author failure, or review's findings
    with the verification change in force."""
    match state.revision:
        case RevisionKind.NONE:
            return None
        case RevisionKind.BUILD_FAILURE | RevisionKind.AUTHOR_FAILURE:
            failure = workpiece.round_dir(state.round) / FAILURES_DIR / f"{state.build_revisions}.txt"
            return Revision(kind=state.revision, detail=failure.read_text())
        case RevisionKind.MODIFY:
            path = workpiece.round_dir(state.round - 1) / DECISION_FILE
            decision = _DECISION.validate_json(path.read_bytes())
            assert isinstance(decision, Modify), f"{path} holds {decision}, not a Modify"
            return Revision(state.revision, "\n\n".join(decision.findings), state.verification_change)


async def _call_author(workpiece: _Workpiece, state: WorkpieceState) -> AuthoredProgram:
    """Call the author for ``state``: a modify revision gets the proposal as triage left it and the change in
    force on the revision; any other call gets the proposal builds run."""
    modify = state.revision is RevisionKind.MODIFY
    proposal = workpiece.proposal_at(state) if modify else workpiece.built_proposal(state)
    return await workpiece.coordinator.stages.author(
        workpiece.author_context(state), proposal, _revision(workpiece, state), state.author_state
    )


async def _author(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Ask the author for the program and write its manifest and any source to ``authored/<k>/``.

    An author raising ``StructuredOutputError`` returned no valid reply even after its repairs, as
    when the output limit cut the reply, and one raising ``AuthorFailure`` ended without a program
    to hand in. For either, the adapter writes the failure to ``build-failures/`` and
    returns ``AUTHOR_FAILED``, so the next author call samples again with the failure as its
    revision. Every other exception passes through unchanged, and ``run_workpiece`` fails the workpiece in
    place at the author stage. An ``AUTHORED`` event answering a modify revision records the author's
    verification change, ``null`` for none.

    Raises:
        ValueError: the author returned a verification change for a revision other than modify.
    """
    try:
        authored = await _call_author(workpiece, state)
    except (StructuredOutputError, AuthorFailure) as error:
        attrs = _failure_attrs(workpiece, state, error, error_line(error))
        return WorkpieceEvent(EventKind.AUTHOR_FAILED, state.round, state.proposal_digest, attrs)
    attrs: dict[str, str] = {EXIT: Exit.AUTHORED, "revision": state.revision}
    if state.revision is RevisionKind.MODIFY:
        attrs[VERIFICATION_CHANGE_ATTR] = canonical_json(authored.verification_change)
    elif authored.verification_change is not None:
        raise ValueError(
            f"{workpiece.workpiece_id}: the author changed the verification answering a {state.revision} revision; "
            f"only an author answering {RevisionKind.MODIFY} may"
        )
    directory = workpiece.authored_dir(state)
    write_bytes_atomic(str(directory / PROGRAM_MANIFEST_FILE), pretty_json(program_manifest(authored.program)).encode())
    if authored.source is not None:
        write_bytes_atomic(str(directory / PROGRAM_FILE), authored.source.encode())
    return WorkpieceEvent(EventKind.AUTHORED, state.round, authored.program.digest, attrs)


async def _authored_program(workpiece: _Workpiece, state: WorkpieceState) -> RLEnvBuildProgram:
    """Load the program from ``authored/<k>/program.py``, or ask the author again when it wrote no source.

    Raises:
        ValueError: the program's digest differs from the ``AUTHORED`` event's.
    """
    source = workpiece.authored_dir(state) / PROGRAM_FILE
    if source.exists():
        program = workpiece.coordinator.stages.load_program(source.read_text())
        found = f"{source} holds program {program.digest}"
    else:
        program = (await _call_author(workpiece, state)).program
        found = f"the author now returns program {program.digest}"
    if program.digest != state.program_digest:
        raise ValueError(f"{workpiece.workpiece_id}: {found}, not the authored {state.program_digest}")
    return program


async def _build(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Build the authored program, spending a revision only on a failure the author can fix.

    A ``BuildFailure`` or any other exception from the program's own code becomes ``BUILD_FAILED``,
    and the next author call receives its traceback, led by its ``failure_text.failure_head``.
    These errors pass through unchanged, so ``run_workpiece`` fails the workpiece in place and spends no
    revision:

    * ``BuildInfrastructureFailure``, raised when the build host fails;
    * ``BuildExpired``, raised when the build, a step or the grader ran past its clock, which is
      neither the program's fault nor the host's until someone reads the record;
    * ``ModelUnavailable``, raised when the model stays unreachable;
    * ``OSError`` and its subclasses raised outside the program's frames (``infrastructure.storage_fault``): a
      storage fault, a missing blob or memo, or a corrupt one (``StoreCorruption``). An ``OSError``
      raised by the program's own code is a program bug and spends a revision.
    """
    coordinator = workpiece.coordinator
    program = await _authored_program(workpiece, state)
    try:
        draft = await coordinator.stages.build(
            program,
            workpiece.built_proposal(state),
            workpiece.workpiece_id,
            workpiece.round_dir(state.round),
            str(coordinator.root / STORE_DIR),
            coordinator.model,
            coordinator.sampling,
            coordinator.services,
            state.round,
            research=coordinator.research,
            model_digest=coordinator.model_digest,
            verification_change=state.verification_change,
        )
    except (BuildInfrastructureFailure, BuildExpired, ModelUnavailable, ModelRequestRejected):
        raise
    except Exception as error:
        infrastructure = infrastructure_failure(error)
        if infrastructure is not None:
            raise infrastructure from error
        if storage_fault(error, _program_files(program)):
            raise
        if not isinstance(error, BuildCrash) and not authored_failure(error, _program_files(program)):
            raise BuildInfrastructureFailure("unclassified_build_failure", str(error)) from error
        attrs = {
            **_failure_attrs(workpiece, state, error, _build_failure_head(error, program)),
            "step": (error.step or "") if isinstance(error, BuildFailure) else "",
        }
        return WorkpieceEvent(EventKind.BUILD_FAILED, state.round, program.digest, attrs)
    steps = draft.provenance.steps
    hits = sum(outcome.status is CacheStatus.HIT for outcome in steps)
    attrs = {EXIT: Exit.BUILT, "program_digest": program.digest, "steps": str(len(steps)), "hits": str(hits)}
    return WorkpieceEvent(EventKind.BUILT, state.round, digest(draft.task), attrs)


def _failure_attrs(workpiece: _Workpiece, state: WorkpieceState, error: Exception, head: str) -> dict[str, str]:
    """Write the workpiece's next failure to ``build-failures/<k>.txt``, and return its attrs.

    The file holds ``failure_text(head, error, FAILURE_CHARS)``: ``head``, a blank line, then the
    tail of the traceback. ``k`` is ``build_revisions + 1``, counting build and author failures
    together. The attrs exit revise while failures in the current resume window fit
    ``max_build_revisions``, and abandon after exhaustion. They keep ``head`` as ``failure``.
    """
    failures = state.build_revisions + 1
    relative = Path(ROUNDS_DIR) / str(state.round) / FAILURES_DIR / f"{failures}.txt"
    write_bytes_atomic(str(workpiece.directory / relative), failure_text(head, error, FAILURE_CHARS).encode())
    return {
        EXIT: (
            Exit.REVISE
            if failures - state.build_revisions_at_resume <= workpiece.coordinator.budget.max_build_revisions
            else Exit.ABANDON
        ),
        "failure": head[:ATTR_CHARS],
        "failure_file": str(relative),
    }


def _build_failure_head(error: Exception, program: RLEnvBuildProgram) -> str:
    """``failure_head`` of a build failure in ``program``, located by its step functions' files."""
    return failure_head(error, _program_files(program), {*program.steps, program.sink.name})


def _program_files(program: RLEnvBuildProgram) -> set[str]:
    """The files of the program's step functions, as the loader registers them in ``linecache``."""
    return {inspect.getfile(node.fn) for node in (*program.steps.values(), program.sink)}


async def _validate(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Validate the round's draft in the round's ``ValidationContext`` and write the result to ``validation.json``.

    The context carries the MODIFY rounds the workpiece has left (``_modify_rounds_left``) and the pass of
    the round this validation is, one more than ``WorkpieceState.validation_passes``. That count survives the
    retry reset of an abandoned workpiece's re-entry, so a review ``Retry`` and a relaunch both rerun the
    round's ungraded retryable trials with a fresh budget.
    """
    coordinator = workpiece.coordinator
    round_dir = workpiece.round_dir(state.round)
    draft = load_draft(round_dir / DRAFT_DIR)
    if digest(draft.task) != state.task_digest:
        raise ValueError(
            f"{workpiece.workpiece_id}: the stored draft has digest {digest(draft.task)}, not {state.task_digest}"
        )
    context = ValidationContext(
        workpiece_id=workpiece.workpiece_id,
        round=state.round,
        round_dir=round_dir,
        proposal=workpiece.proposal_at(state),
        store_root=str(coordinator.root / STORE_DIR),
        modify_rounds_left=_modify_rounds_left(workpiece, state),
        validation_pass=state.validation_passes + 1,
    )
    result = await coordinator.stages.validate(draft, context)
    write_validation(round_dir, result)
    attrs = {EXIT: Exit.VALIDATED, "result_digest": digest(result)}
    return WorkpieceEvent(EventKind.VALIDATED, state.round, state.task_digest, attrs)


async def _review(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Decide on the result read back from ``validation.json`` and write the decision to ``decision.json``."""
    coordinator = workpiece.coordinator
    round_dir = workpiece.round_dir(state.round)
    result = load_validation(round_dir)
    decision = await coordinator.stages.review(result)
    if isinstance(decision, Accept) or (isinstance(decision, Reject) and decision.kind is RejectKind.TASK):
        gaps = evidence_gaps(result)
        if isinstance(decision, Accept) and decision.band is not BandOutcome.IN_BAND:
            gaps += (f"review attempted acceptance outside the band: {decision.band}",)
        if gaps:
            decision = Hold(gaps)
    write_bytes_atomic(str(round_dir / DECISION_FILE), _DECISION.dump_json(decision, indent=2))
    return WorkpieceEvent(EventKind.DECIDED, state.round, state.task_digest, _decision_attrs(workpiece, state, decision))


def _modify_rounds_left(workpiece: _Workpiece, state: WorkpieceState) -> int:
    """``max_modify_rounds`` less the ``Modify`` decisions the workpiece has followed (``WorkpieceState.round``)."""
    return workpiece.coordinator.budget.max_modify_rounds - (state.round - state.round_at_resume)


def _decision_attrs(workpiece: _Workpiece, state: WorkpieceState, decision: Decision) -> dict[str, str]:
    """The ``DECIDED`` attrs of ``decision``.

    A ``Retry`` past ``max_validation_retries`` exits abandon. A ``Modify`` once the workpiece's rounds
    have spent ``max_modify_rounds`` (``WorkpieceState.round`` counts the ``Modify`` decisions followed)
    exits abandon, with its findings in the reasons.
    """
    used = {"retries_used": str(state.retries)}
    match decision:
        case Accept(band=band):
            return {EXIT: Exit.ACCEPT, "band": band, **used}
        case Reject(kind=RejectKind.BUDGET, reasons=reasons):
            return {EXIT: Exit.ABANDON, "reasons": "; ".join(reasons), "resume_stage": Stage.VALIDATE, **used}
        case Reject(kind=RejectKind.HOST, reasons=reasons):
            return {EXIT: Exit.HOLD, "reasons": "; ".join(reasons), **used}
        case Reject(kind=kind, reasons=reasons):
            return {EXIT: Exit.REJECT, "kind": kind, "reasons": "; ".join(reasons), **used}
        case Hold(reasons=reasons):
            return {EXIT: Exit.HOLD, "reasons": "; ".join(reasons), **used}
        case Retry(cause=cause, count=count):
            spent = state.retries >= workpiece.coordinator.budget.max_validation_retries
            way = Exit.ABANDON if spent else Exit.RETRY
            if spent:
                return {EXIT: way, "reasons": f"{count} ungraded trial(s): {cause}", **used}
            return {EXIT: way, "cause": cause, "count": str(count), "retries_used": str(state.retries + 1)}
        case Modify(findings=findings):
            budget = workpiece.coordinator.budget.max_modify_rounds
            if state.round - state.round_at_resume >= budget:
                spent_rounds = state.round - state.round_at_resume
                reason = f"review returned Modify with {spent_rounds} of {budget} rounds spent: {'; '.join(findings)}"
                return {EXIT: Exit.ABANDON, "reasons": reason, "resume_stage": Stage.REVIEW, **used}
            return {EXIT: Exit.MODIFY, "findings": "; ".join(findings), **used}


async def _close(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    closing = state.closing
    assert closing is not None, f"{workpiece.workpiece_id} is done without a closing"
    attrs = {TERMINAL_ATTR: closing.terminal.value, REASON_ATTR: closing.reason}
    if closing.kind is not None:
        attrs["kind"] = closing.kind.value
    return WorkpieceEvent(EventKind.TERMINAL, state.round, None, attrs)


ADAPTERS: Mapping[Stage, Callable[[_Workpiece, WorkpieceState], Awaitable[WorkpieceEvent]]] = {
    Stage.TRIAGE: _triage,
    Stage.AUTHOR: _author,
    Stage.BUILD: _build,
    Stage.VALIDATE: _validate,
    Stage.REVIEW: _review,
    Stage.DONE: _close,
}
"""The adapter running each stage and returning its event."""

_DECISION: TypeAdapter[Decision] = TypeAdapter(Decision)
