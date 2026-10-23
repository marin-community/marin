# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fold durable workpiece events into the next stage and its outcome.

ACCEPTED and REJECTED are final evaluation decisions. ABANDONED means a workpiece
retry or revision budget was exhausted; HELD means execution or evidence is
unavailable. Both preserve the proposal, program, task, evidence and next stage.
A review hold resumes validation; a stage failure resumes that stage. Budget
exhaustion resumes triage, authoring, review or validation as appropriate.

An explicit relaunch of ABANDONED records RESUMED. It starts a fresh budget window
while retaining cumulative revision/round counters, so new evidence never overwrites
old files. A changed budget is allowed only at that recorded resumption. Legacy
ABANDONED validation records still resume validation, and legacy FAILED records are
read as HELD at their original stage. Existing final decisions remain immutable.

The fold checks sequence order, stage ownership, transition edges and closing
terminals. Only DONE consumes a closing; an exception may record HELD in place.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType

from pydantic import JsonValue

from taskforge.stage30_author.author import RevisionKind
from taskforge.stage60_review.decision import BandOutcome, RejectKind

EXIT = "exit"
"""The attr holding a stage event's ``Exit``."""
TERMINAL_ATTR = "terminal"
"""The attr holding a ``TERMINAL`` event's ``Terminal``."""
REASON_ATTR = "reason"
AUTHOR_STATE_ATTR = "author_state"
"""The attr holding the author state of a triage accept, as canonical JSON."""
UNNUMBERED = 0
"""The ``seq`` of an event before a log appends it."""
CALIBRATED = "calibrated"
OUTSIDE_BAND = "accepted outside the band"
TRIAGE_REJECTED = "triage rejected the proposal"


class EventKind(StrEnum):
    OPENED = "opened"
    """digest: the proposal's. attrs: budget_digest, model_digest (the coordinator's, part of every step key)."""
    RESUMED = "resumed"
    """An explicit relaunch of ABANDONED, with the current budget_digest."""
    TRIAGED = "triaged"
    """digest: the triaged proposal's, or the rewrite's for repair. attrs: exit, decision (the verdict's
    ``TriageDecision``), repairs (the rewrites spent, this event's included); triaged_digest and
    proposal_file for repair; author_state for accept (the verdict's, as canonical JSON); reason for
    reject (the verdict's reasons, or why a REPAIR verdict was not answered, truncated)."""
    AUTHORED = "authored"
    """digest: the program's. attrs: exit, revision (the ``RevisionKind`` the program answers)."""
    AUTHOR_FAILED = "author_failed"
    """digest: the proposal's. attrs: exit, failure (truncated), failure_file."""
    BUILD_FAILED = "build_failed"
    """digest: the program's. attrs: exit, step, failure (truncated), failure_file."""
    BUILT = "built"
    """digest: the task's. attrs: exit, program_digest, steps, hits."""
    VALIDATED = "validated"
    """digest: the task's. attrs: exit, result_digest (the ``ValidationResult``'s)."""
    DECIDED = "decided"
    """digest: the task's. attrs: exit, retries_used; band for accept; kind and reasons for reject;
    cause and count for retry and abandon; findings for modify."""
    TERMINAL = "terminal"
    """attrs: terminal (a ``Terminal``), reason, and kind (a ``RejectKind``) when rejected."""


class Stage(StrEnum):
    TRIAGE = "triage"
    AUTHOR = "author"
    BUILD = "build"
    VALIDATE = "validate"
    REVIEW = "review"
    DONE = "done"
    """A closing is recorded, and the workpiece's ``TERMINAL`` event comes next."""


class Exit(StrEnum):
    """The way a stage ended, stored as ``attrs[EXIT]`` and looked up in ``TRANSITIONS``."""

    ACCEPT = "accept"
    REJECT = "reject"
    AUTHORED = "authored"
    BUILT = "built"
    REPAIR = "repair"
    REVISE = "revise"
    VALIDATED = "validated"
    RETRY = "retry"
    ABANDON = "abandon"
    MODIFY = "modify"
    HOLD = "hold"


EVENT_STAGES: Mapping[EventKind, Stage] = {
    EventKind.TRIAGED: Stage.TRIAGE,
    EventKind.AUTHORED: Stage.AUTHOR,
    EventKind.AUTHOR_FAILED: Stage.AUTHOR,
    EventKind.BUILD_FAILED: Stage.BUILD,
    EventKind.BUILT: Stage.BUILD,
    EventKind.VALIDATED: Stage.VALIDATE,
    EventKind.DECIDED: Stage.REVIEW,
}
"""The stage recording each stage event."""

TRANSITIONS: Mapping[tuple[Stage, Exit], Stage] = MappingProxyType(
    {
        (Stage.TRIAGE, Exit.ACCEPT): Stage.AUTHOR,
        (Stage.TRIAGE, Exit.REPAIR): Stage.TRIAGE,
        (Stage.TRIAGE, Exit.REJECT): Stage.DONE,
        (Stage.TRIAGE, Exit.HOLD): Stage.DONE,
        (Stage.TRIAGE, Exit.ABANDON): Stage.DONE,
        (Stage.AUTHOR, Exit.AUTHORED): Stage.BUILD,
        (Stage.AUTHOR, Exit.REVISE): Stage.AUTHOR,
        (Stage.AUTHOR, Exit.REJECT): Stage.DONE,
        (Stage.AUTHOR, Exit.ABANDON): Stage.DONE,
        (Stage.BUILD, Exit.BUILT): Stage.VALIDATE,
        (Stage.BUILD, Exit.REVISE): Stage.AUTHOR,
        (Stage.BUILD, Exit.REJECT): Stage.DONE,
        (Stage.BUILD, Exit.ABANDON): Stage.DONE,
        (Stage.VALIDATE, Exit.VALIDATED): Stage.REVIEW,
        (Stage.REVIEW, Exit.ACCEPT): Stage.DONE,
        (Stage.REVIEW, Exit.REJECT): Stage.DONE,
        (Stage.REVIEW, Exit.RETRY): Stage.VALIDATE,
        (Stage.REVIEW, Exit.ABANDON): Stage.DONE,
        (Stage.REVIEW, Exit.MODIFY): Stage.AUTHOR,
        (Stage.REVIEW, Exit.HOLD): Stage.DONE,
    }
)


class Terminal(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    HELD = "held"
    """Incomplete evidence or an execution failure; resume without spending an author revision."""
    ABANDONED = "abandoned"
    """A workpiece retry/revision budget ran out; a relaunch starts a new budget window."""
    FAILED = "failed"
    """Legacy spelling read as HELD; new execution failures never write FAILED."""


FINAL = frozenset({Terminal.ACCEPTED, Terminal.REJECTED})


@dataclass(frozen=True)
class Closing:
    """The ``TERMINAL`` event due after a closing edge."""

    terminal: Terminal
    reason: str
    kind: RejectKind | None
    resume_stage: Stage | None = None


@dataclass(frozen=True)
class WorkpieceEvent:
    """An event of a workpiece's log.

    Attributes:
        kind: The event's kind.
        round: The round the event belongs to.
        digest: The digest of the proposal, program or task the event refers to (see ``EventKind``).
        attrs: Flat string fields, documented on each ``EventKind``.
        seq: The event's position in the log, from 1. A log sets it when it appends the event.
        at: The wall-clock time, in seconds, of the append. The fold ignores it.
    """

    kind: EventKind
    round: int
    digest: str | None
    attrs: dict[str, str]
    seq: int = UNNUMBERED
    at: float = 0.0


@dataclass(frozen=True)
class WorkpieceState:
    """A workpiece's status, folded from its events.

    Attributes:
        stage: The stage to run next.
        round: The author, build and validation round. Only a ``modify`` exit starts the next one, so it
            also counts the MODIFY rounds the workpiece has spent.
        seq: The last event's ``seq``.
        proposal_digest: The digest of the proposal the stages run: the opened one, or the last triage rewrite.
        program_digest: The digest of the program authored last.
        task_digest: The digest of this round's built task, or ``None`` before the build.
        revision: The kind of revision the next author call answers. A build of a program without
            source asks the author again with it, so only ``BUILT`` clears it.
        triage_repairs: The rewrites answering a REPAIR verdict so far.
        author_state: The author state triage's accept recorded, which every author call receives;
            empty before the accept.
        build_revisions: The build failures and failed author calls so far.
        retries: The review ``Retry`` decisions since the workpiece opened or was last abandoned: the count
            ``max_validation_retries`` bounds.
        validation_passes: The ``VALIDATED`` events of the current round, across retries and re-entries after
            an abandonment; a new round starts it at zero.
        closing: The ``TERMINAL`` event due, while the stage is ``DONE``.
        terminal: The workpiece's terminal, until a later event clears it.
    """

    stage: Stage
    round: int
    seq: int
    proposal_digest: str
    program_digest: str | None
    task_digest: str | None
    revision: RevisionKind
    triage_repairs: int
    author_state: Mapping[str, JsonValue]
    build_revisions: int
    retries: int
    validation_passes: int
    closing: Closing | None
    terminal: Terminal | None
    triage_repairs_at_resume: int = 0
    build_revisions_at_resume: int = 0
    round_at_resume: int = 0


def opened(event: WorkpieceEvent) -> WorkpieceState:
    """Start a workpiece at ``TRIAGE`` in round 0 from its ``OPENED`` event."""
    if event.kind is not EventKind.OPENED or event.seq != 1:
        raise ValueError(f"a workpiece's log starts with OPENED at seq 1, not {event.kind} at seq {event.seq}")
    assert event.digest is not None, "OPENED carries the proposal digest"
    return WorkpieceState(
        stage=Stage.TRIAGE,
        round=0,
        seq=1,
        proposal_digest=event.digest,
        program_digest=None,
        task_digest=None,
        revision=RevisionKind.NONE,
        triage_repairs=0,
        author_state=MappingProxyType({}),
        build_revisions=0,
        retries=0,
        validation_passes=0,
        closing=None,
        terminal=None,
    )


def transition(state: WorkpieceState, event: WorkpieceEvent) -> WorkpieceState:
    """Return the state after ``event``, the workpiece's next event.

    A ``TERMINAL`` event ends the workpiece at its closing, or marks it failed in place. Any other event
    belongs to the stage in ``EVENT_STAGES``. Its exit picks the next stage from ``TRANSITIONS``, and its
    kind and exit select the bookkeeping in the module docstring's table.

    Raises:
        ValueError: the event follows a final terminal, breaks the ``seq`` order, comes from another
            stage, has an exit its stage or kind lacks, or is a ``TERMINAL`` disagreeing with the closing.
    """
    kind, attrs = event.kind, event.attrs
    if state.terminal in FINAL:
        raise ValueError(f"event {kind} after terminal {state.terminal}")
    if event.seq != state.seq + 1:
        raise ValueError(f"event seq {event.seq} where {state.seq + 1} was expected")
    previous_terminal = state.terminal
    state = replace(state, seq=event.seq, terminal=None)
    if kind is EventKind.RESUMED:
        if previous_terminal is not Terminal.ABANDONED:
            raise ValueError("RESUMED requires an abandoned workpiece")
        return replace(
            state,
            retries=0,
            triage_repairs_at_resume=state.triage_repairs,
            build_revisions_at_resume=state.build_revisions,
            round_at_resume=state.round,
        )
    if kind is EventKind.TERMINAL:
        return _terminal(state, event)
    stage = EVENT_STAGES.get(kind)
    if stage is None:
        raise ValueError(f"{kind} after the workpiece's first event")
    if stage is not state.stage:
        raise ValueError(f"{kind} is recorded by stage {stage}, but the workpiece is in stage {state.stage}")
    way = Exit(attrs[EXIT])
    if (state.stage, way) not in TRANSITIONS:
        raise ValueError(f"{kind} exits {way}, which stage {state.stage} has no edge for")
    state = replace(state, stage=TRANSITIONS[(state.stage, way)])
    match (kind, way):
        case (EventKind.TRIAGED, Exit.ACCEPT):
            return replace(state, author_state=MappingProxyType(json.loads(attrs[AUTHOR_STATE_ATTR])))
        case (EventKind.VALIDATED, Exit.VALIDATED):
            return replace(state, validation_passes=state.validation_passes + 1)
        case (EventKind.TRIAGED, Exit.REPAIR):
            assert event.digest is not None, "a repair exit carries the rewrite's digest"
            return replace(state, proposal_digest=event.digest, triage_repairs=state.triage_repairs + 1)
        case (EventKind.TRIAGED, Exit.HOLD):
            return replace(state, closing=Closing(Terminal.HELD, attrs[REASON_ATTR], None, Stage.TRIAGE))
        case (EventKind.TRIAGED, Exit.ABANDON):
            return replace(
                state, closing=Closing(Terminal.ABANDONED, attrs[REASON_ATTR], RejectKind.BUDGET, Stage.TRIAGE)
            )
        case (EventKind.TRIAGED, Exit.REJECT):
            return replace(state, closing=Closing(Terminal.REJECTED, attrs[REASON_ATTR], RejectKind.TASK))
        case (EventKind.AUTHORED, Exit.AUTHORED):
            return replace(state, program_digest=event.digest)
        case (EventKind.BUILT, Exit.BUILT):
            return replace(state, task_digest=event.digest, revision=RevisionKind.NONE)
        case (EventKind.AUTHOR_FAILED, Exit.REVISE):
            return replace(state, build_revisions=state.build_revisions + 1, revision=RevisionKind.AUTHOR_FAILURE)
        case (EventKind.AUTHOR_FAILED, Exit.ABANDON):
            return _out_of_revisions(state, RevisionKind.AUTHOR_FAILURE, Terminal.ABANDONED)
        case (EventKind.AUTHOR_FAILED, Exit.REJECT):
            return _out_of_revisions(state, RevisionKind.AUTHOR_FAILURE, Terminal.REJECTED)
        case (EventKind.BUILD_FAILED, Exit.REVISE):
            return replace(state, build_revisions=state.build_revisions + 1, revision=RevisionKind.BUILD_FAILURE)
        case (EventKind.BUILD_FAILED, Exit.ABANDON):
            return _out_of_revisions(state, RevisionKind.BUILD_FAILURE, Terminal.ABANDONED)
        case (EventKind.BUILD_FAILED, Exit.REJECT):
            return _out_of_revisions(state, RevisionKind.BUILD_FAILURE, Terminal.REJECTED)
        case (EventKind.DECIDED, Exit.ACCEPT):
            band = BandOutcome(attrs["band"])
            reason = CALIBRATED if band is BandOutcome.IN_BAND else f"{OUTSIDE_BAND}: {band}"
            return replace(state, closing=Closing(Terminal.ACCEPTED, reason, None))
        case (EventKind.DECIDED, Exit.REJECT):
            return replace(state, closing=Closing(Terminal.REJECTED, attrs["reasons"], RejectKind(attrs["kind"])))
        case (EventKind.DECIDED, Exit.HOLD):
            return replace(state, closing=Closing(Terminal.HELD, attrs["reasons"], None, Stage.VALIDATE))
        case (EventKind.DECIDED, Exit.RETRY):
            return replace(state, retries=state.retries + 1)
        case (EventKind.DECIDED, Exit.ABANDON):
            return replace(
                state,
                closing=Closing(
                    Terminal.ABANDONED,
                    attrs.get("reasons", attrs.get("cause", "retry budget exhausted")),
                    RejectKind.BUDGET,
                    Stage(attrs.get("resume_stage", Stage.VALIDATE)),
                ),
            )
        case (EventKind.DECIDED, Exit.MODIFY):
            return replace(
                state, round=state.round + 1, validation_passes=0, task_digest=None, revision=RevisionKind.MODIFY
            )
        case _:
            raise ValueError(f"{kind} cannot exit {way}")


def _out_of_revisions(state: WorkpieceState, revision: RevisionKind, terminal: Terminal) -> WorkpieceState:
    """Retain the last failure for the next author call; old REJECT edges remain readable."""
    failures = state.build_revisions + 1
    reason = f"no buildable program after {failures} build and author failures"
    closing = Closing(terminal, reason, RejectKind.BUDGET, Stage.AUTHOR)
    return replace(state, build_revisions=failures, revision=revision, closing=closing)


def _terminal(state: WorkpieceState, event: WorkpieceEvent) -> WorkpieceState:
    terminal = Terminal(event.attrs[TERMINAL_ATTR])
    if terminal is Terminal.FAILED:
        terminal = Terminal.HELD
    closing = state.closing
    if terminal is Terminal.HELD and (closing is None or closing.terminal is not Terminal.HELD):
        return replace(state, terminal=terminal)
    if terminal is Terminal.ABANDONED and closing is None and event.attrs.get("kind") == RejectKind.BUDGET:
        return replace(state, terminal=terminal)
    if state.stage is not Stage.DONE or closing is None or closing.terminal is not terminal:
        found = None if closing is None else closing.terminal
        raise ValueError(f"TERMINAL {terminal} in stage {state.stage} with closing {found}")
    state = replace(state, terminal=terminal, closing=None)
    if terminal in (Terminal.ABANDONED, Terminal.HELD):
        return replace(state, stage=closing.resume_stage or Stage.VALIDATE, retries=0)
    return state


def derive_state(events: Iterable[WorkpieceEvent]) -> WorkpieceState:
    """Fold a workpiece's events into its ``WorkpieceState``: ``opened`` on the first, then ``transition`` on each.

    Raises:
        ValueError: the log is empty, does not start with ``OPENED``, or fails a ``transition`` check.
    """
    events = iter(events)
    first = next(events, None)
    if first is None:
        raise ValueError("a workpiece's log starts with OPENED, and this log is empty")
    state = opened(first)
    for event in events:
        state = transition(state, event)
    return state
