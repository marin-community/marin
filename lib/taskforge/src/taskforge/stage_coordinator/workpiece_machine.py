# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The workpiece machine: the stages of a workpiece, and the workpiece's state as a fold over its events.

The coordinator runs the stage in ``WorkpieceState.stage`` and appends the stage's event to the workpiece's
log. ``transition`` then looks up the event's exit in ``TRANSITIONS``, the table below, to move the
workpiece to its next stage::

    stage     event          exit       next      bookkeeping
    --------  -------------  ---------  --------  ------------------------------------------------
    TRIAGE    TRIAGED        accept     AUTHOR
    TRIAGE    TRIAGED        reject     DONE      closing rejected (task)
    AUTHOR    AUTHORED       authored   BUILD     program_digest
    AUTHOR    AUTHOR_FAILED  revise     AUTHOR    build_revisions + 1; revision author_failure
    AUTHOR    AUTHOR_FAILED  reject     DONE      build_revisions + 1; closing rejected (budget)
    BUILD     BUILT          built      VALIDATE  task_digest; revision none
    BUILD     BUILD_FAILED   revise     AUTHOR    build_revisions + 1; revision build_failure
    BUILD     BUILD_FAILED   reject     DONE      build_revisions + 1; closing rejected (budget)
    VALIDATE  VALIDATED      validated  REVIEW
    REVIEW    DECIDED        accept     DONE      closing accepted (labelled with the band)
    REVIEW    DECIDED        reject     DONE      closing rejected (the decision's kind)
    REVIEW    DECIDED        retry      VALIDATE  retries + 1
    REVIEW    DECIDED        abandon    DONE      retries + 1; closing abandoned
    REVIEW    DECIDED        modify     AUTHOR    round + 1; task_digest cleared; revision modify
    DONE      TERMINAL       -          -         terminal = the closing's terminal
    any       TERMINAL       -          (stays)   terminal failed, when ``terminal`` is ``failed``

The coordinator checks the workpiece's budget when it records the event spending it. It picks revise or
reject for ``AUTHOR_FAILED`` and ``BUILD_FAILED``, and retry or abandon for ``DECIDED``, so
``transition`` reads no budget. ``AUTHOR_FAILED`` and ``BUILD_FAILED`` both count toward
``build_revisions``.

``derive_state`` folds a workpiece's events into its ``WorkpieceState`` by calling ``transition`` on each
event in turn. The state is computed again from the events on every read and is stored nowhere.
The fold raises ``ValueError`` on a gap or a repeat in ``seq``, on an event recorded outside its
stage, and on an exit its stage has no edge for.

A workpiece ending ``ACCEPTED`` or ``REJECTED`` is finished, and the fold refuses any later event. A
workpiece ending ``ABANDONED`` or ``FAILED`` continues when it runs again, and its next event clears the
terminal:

* an abandoned workpiece returns to ``VALIDATE`` with its retry count at zero;
* a failed workpiece resumes at the stage it failed in.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType

from taskforge.stage30_author.author import RevisionKind
from taskforge.stage60_review.decision import BandOutcome, RejectKind

EXIT = "exit"
"""The attr holding a stage event's ``Exit``."""
TERMINAL_ATTR = "terminal"
"""The attr holding a ``TERMINAL`` event's ``Terminal``."""
REASON_ATTR = "reason"
UNNUMBERED = 0
"""The ``seq`` of an event before a log appends it."""
CALIBRATED = "calibrated"
OUTSIDE_BAND = "accepted outside the band"
TRIAGE_REJECTED = "triage rejected the proposal"


class EventKind(StrEnum):
    OPENED = "opened"
    """digest: the proposal's. attrs: budget_digest, model_digest (the coordinator's, part of every step key)."""
    TRIAGED = "triaged"
    """digest: the proposal's. attrs: exit."""
    AUTHORED = "authored"
    """digest: the program's. attrs: exit, revision (the ``RevisionKind`` the program answers)."""
    AUTHOR_FAILED = "author_failed"
    """digest: the proposal's. attrs: exit, failure (truncated), failure_file."""
    BUILD_FAILED = "build_failed"
    """digest: the program's. attrs: exit, step, failure (truncated), failure_file."""
    BUILT = "built"
    """digest: the task's. attrs: exit, program_digest, steps, hits."""
    VALIDATED = "validated"
    """digest: the task's. attrs: exit, summary_digest."""
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
    REVISE = "revise"
    VALIDATED = "validated"
    RETRY = "retry"
    ABANDON = "abandon"
    MODIFY = "modify"


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
        (Stage.TRIAGE, Exit.REJECT): Stage.DONE,
        (Stage.AUTHOR, Exit.AUTHORED): Stage.BUILD,
        (Stage.AUTHOR, Exit.REVISE): Stage.AUTHOR,
        (Stage.AUTHOR, Exit.REJECT): Stage.DONE,
        (Stage.BUILD, Exit.BUILT): Stage.VALIDATE,
        (Stage.BUILD, Exit.REVISE): Stage.AUTHOR,
        (Stage.BUILD, Exit.REJECT): Stage.DONE,
        (Stage.VALIDATE, Exit.VALIDATED): Stage.REVIEW,
        (Stage.REVIEW, Exit.ACCEPT): Stage.DONE,
        (Stage.REVIEW, Exit.REJECT): Stage.DONE,
        (Stage.REVIEW, Exit.RETRY): Stage.VALIDATE,
        (Stage.REVIEW, Exit.ABANDON): Stage.DONE,
        (Stage.REVIEW, Exit.MODIFY): Stage.AUTHOR,
    }
)


class Terminal(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    """Validation retries ran out. A relaunch returns the workpiece to ``VALIDATE``."""
    FAILED = "failed"
    """A stage raised. A relaunch resumes the workpiece at that stage."""


FINAL = frozenset({Terminal.ACCEPTED, Terminal.REJECTED})


@dataclass(frozen=True)
class Closing:
    """The ``TERMINAL`` event due after a closing edge."""

    terminal: Terminal
    reason: str
    kind: RejectKind | None


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
        round: The author, build and validation round. A ``modify`` exit starts the next one.
        seq: The last event's ``seq``.
        proposal_digest: The digest of the opened proposal.
        program_digest: The digest of the program authored last.
        task_digest: The digest of this round's built task, or ``None`` before the build.
        revision: The kind of revision the next author call answers. A build of a program without
            source asks the author again with it, so only ``BUILT`` clears it.
        build_revisions: The build failures and failed author calls so far.
        retries: The review ``Retry`` decisions since the workpiece opened or was last abandoned.
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
    build_revisions: int
    retries: int
    closing: Closing | None
    terminal: Terminal | None


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
        build_revisions=0,
        retries=0,
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
    state = replace(state, seq=event.seq, terminal=None)
    if kind is EventKind.TERMINAL:
        return _terminal(state, Terminal(attrs[TERMINAL_ATTR]))
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
        case (EventKind.TRIAGED, Exit.ACCEPT) | (EventKind.VALIDATED, Exit.VALIDATED):
            return state
        case (EventKind.TRIAGED, Exit.REJECT):
            return replace(state, closing=Closing(Terminal.REJECTED, TRIAGE_REJECTED, RejectKind.TASK))
        case (EventKind.AUTHORED, Exit.AUTHORED):
            return replace(state, program_digest=event.digest)
        case (EventKind.BUILT, Exit.BUILT):
            return replace(state, task_digest=event.digest, revision=RevisionKind.NONE)
        case (EventKind.AUTHOR_FAILED, Exit.REVISE):
            return replace(state, build_revisions=state.build_revisions + 1, revision=RevisionKind.AUTHOR_FAILURE)
        case (EventKind.AUTHOR_FAILED, Exit.REJECT):
            return _out_of_revisions(state, RevisionKind.AUTHOR_FAILURE)
        case (EventKind.BUILD_FAILED, Exit.REVISE):
            return replace(state, build_revisions=state.build_revisions + 1, revision=RevisionKind.BUILD_FAILURE)
        case (EventKind.BUILD_FAILED, Exit.REJECT):
            return _out_of_revisions(state, RevisionKind.BUILD_FAILURE)
        case (EventKind.DECIDED, Exit.ACCEPT):
            band = BandOutcome(attrs["band"])
            reason = CALIBRATED if band is BandOutcome.IN_BAND else f"{OUTSIDE_BAND}: {band}"
            return replace(state, closing=Closing(Terminal.ACCEPTED, reason, None))
        case (EventKind.DECIDED, Exit.REJECT):
            return replace(state, closing=Closing(Terminal.REJECTED, attrs["reasons"], RejectKind(attrs["kind"])))
        case (EventKind.DECIDED, Exit.RETRY):
            return replace(state, retries=state.retries + 1)
        case (EventKind.DECIDED, Exit.ABANDON):
            return replace(state, retries=state.retries + 1, closing=Closing(Terminal.ABANDONED, attrs["cause"], None))
        case (EventKind.DECIDED, Exit.MODIFY):
            return replace(state, round=state.round + 1, task_digest=None, revision=RevisionKind.MODIFY)
        case _:
            raise ValueError(f"{kind} cannot exit {way}")


def _out_of_revisions(state: WorkpieceState, revision: RevisionKind) -> WorkpieceState:
    """Count the failure that spent the last build revision, and close the workpiece as rejected on its budget."""
    failures = state.build_revisions + 1
    reason = f"no buildable program after {failures} build and author failures"
    closing = Closing(Terminal.REJECTED, reason, RejectKind.BUDGET)
    return replace(state, build_revisions=failures, revision=revision, closing=closing)


def _terminal(state: WorkpieceState, terminal: Terminal) -> WorkpieceState:
    if terminal is Terminal.FAILED:
        return replace(state, terminal=terminal)
    closing = state.closing
    if state.stage is not Stage.DONE or closing is None or closing.terminal is not terminal:
        found = None if closing is None else closing.terminal
        raise ValueError(f"TERMINAL {terminal} in stage {state.stage} with closing {found}")
    state = replace(state, terminal=terminal, closing=None)
    if terminal is Terminal.ABANDONED:
        return replace(state, stage=Stage.VALIDATE, retries=0)
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
