# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The item event log: ``EntryKind.EVENT`` ledger rows, and item status as a pure fold over them.

Every phase boundary of an item appends one event through the run's ``Ledger``, into the item's own
JSONL file beside its LLM, step and trial spans (and, under Iris, into the Finelog mirror). An event's
``step`` is its ``EventKind``, ``input_hash`` is the digest of the artifact it points at, and its
payload is flat string ``attrs``. Every event carries ``attrs["seq"]``, contiguous from 1 per item,
and ``attrs["schema"] = EVENT_SCHEMA``. Large payloads stay in the item directory (proposals,
``verdict.json``, programs, drafts, attempt files, ``calibration.json``, ``decision.json``) and an
event names them by digest.

``derive_state`` folds a proposal item's events into its ``ItemState``; ``derive_idea_state`` folds an
idea item's (``idea--<idea_id>``). Status is never stored. A gap or a repeat in ``seq``, the only sign
of two processes writing one run root, and an unknown schema raise ``ValueError``.

An item that ended ``ABANDONED`` (build or validation retries spent) or ``FAILED`` (an unhandled
exception) is not final: when it is run again, the next event clears ``terminal``, and an abandoned
item resumes where it stopped (the build of the same program, or ``CONTROLS``) with fresh retry
counts. Whether a launch runs it again is the caller's choice. ``ACCEPTED`` and ``REJECTED`` are final.

A ``DECIDED`` finding kind that ``FindingKind`` does not name makes ``derive_state`` raise.
"""

import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from taskforge.build.infrastructure import InfrastructureCause
from taskforge.ledger.records import EntryKind, Ledger, LedgerEntry
from taskforge.review.decision import BandOutcome, RejectKind
from taskforge.triage.verdict import TriageDecision
from taskforge.validate.calibration import DECISIVE, FindingKind
from taskforge.validate.outcome import TrialKind

EVENT_SCHEMA = "1"
SEQ = "seq"
SCHEMA = "schema"
LIST_SEPARATOR = ","
COUNT_SEPARATOR = ":"
CALIBRATED = "calibrated"
OUTSIDE_BAND = "accepted outside the band"


class EventKind(StrEnum):
    # idea items ("idea--<idea_id>")
    PROPOSED = "proposed"
    """A batch parsed. attrs: items (item ids, comma-separated), slots, failures, reproposal."""
    SLOT_FAILED = "slot_failed"
    """attrs: slot, error (truncated)."""
    IDEA_EXHAUSTED = "idea_exhausted"
    """Every batch of the idea yielded zero proposals. attrs: reproposals."""
    # proposal items
    OPENED = "opened"
    """input_hash: the proposal digest. attrs: proposal, idea, origin (a ``ProposalOrigin``), policy_digest."""
    TRIAGED = "triaged"
    """input_hash: the proposal digest. attrs: decision, tally, repairs."""
    TRIAGE_REPAIRED = "triage_repaired"
    """input_hash: the repaired proposal's digest. attrs: from_digest."""
    AUTHORED = "authored"
    """input_hash: the program digest. attrs: revises, revision (a ``RevisionKind``)."""
    BUILD_FAILED = "build_failed"
    """input_hash: the program digest. attrs: step, failure (truncated), failure_file, noop."""
    BUILD_INFRASTRUCTURE = "build_infrastructure"
    """The machine host failed the build; no revision is spent and the author is not told.
    input_hash: the program digest. attrs: cause (an ``InfrastructureCause``), message (truncated),
    retries_used, abandon, not_before."""
    BUILT = "built"
    """input_hash: the task digest. attrs: program_digest, steps, hits, staged."""
    CONTROLS_REPLAYED = "controls_replayed"
    """input_hash: the task digest. attrs: met, violated, ungraded (counts), passed."""
    SOLVED = "solved"
    """attrs: graded, solved, timed_out, ungraded."""
    ADVERSARIES_RUN = "adversaries_run"
    """attrs per role: ``<role>_graded`` (graded trials), ``<role>_passes`` (graded trials whose grade
    passed), ``<role>_submissions`` (verifier submissions over the graded trials) and ``<role>_claimed``
    (graded trials whose verdict line claimed a shortcut); and ``context_digest``, the sha256 of the
    consumer's adversary context, ``""`` when it was empty."""
    DECIDED = "decided"
    """input_hash: the task digest. attrs: decision (a ``DecisionKind``), repairs_used, retries_used and
    notes (the kinds of the noted adversary passes, comma-separated; empty when staged or none); band
    (a ``BandOutcome``: where the solve rate fell, or the kind accepted outside it), solved, graded
    and solve_rate (three decimals) for accept; kind (a ``RejectKind``) and reasons for reject;
    findings and invalidate for repair; cause, count, abandon and not_before for retry."""
    TERMINAL = "terminal"
    """attrs: terminal (a ``Terminal``), reason (``calibrated``, or ``accepted outside the band: <band>``
    when accepted), kind (a ``RejectKind``) when rejected, and causes (``cause:count`` pairs,
    comma-separated) when abandoned."""


class ProposalOrigin(StrEnum):
    """Where an item's proposal came from: a ``PROPOSED`` batch, or the caller of ``run_item``."""

    GENERATED = "generated"
    SUPPLIED = "supplied"


IDEA_EVENTS = frozenset({EventKind.PROPOSED, EventKind.SLOT_FAILED, EventKind.IDEA_EXHAUSTED})


class Terminal(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    """Build or validation retries spent on an infrastructure cause; re-entered on the next launch."""
    FAILED = "failed"
    """An unhandled exception; the queue re-enters it only when asked to retry failed items."""


FINAL = frozenset({Terminal.ACCEPTED, Terminal.REJECTED})


class Phase(StrEnum):
    TRIAGE = "triage"
    BUILD = "build"
    CONTROLS = "controls"
    TRIALS = "trials"
    DECIDE = "decide"
    DONE = "done"
    """A terminal decision is recorded and the ``TERMINAL`` event is next."""


class DecisionKind(StrEnum):
    """The ``decision`` attr of ``DECIDED``: the class of the ``review.decision.Decision``."""

    ACCEPT = "accept"
    REJECT = "reject"
    REPAIR = "repair"
    RETRY = "retry"


class RevisionKind(StrEnum):
    """What the next authoring revises: nothing, a failed build, or a program review condemned."""

    NONE = "none"
    BUILD_FAILURE = "build_failure"
    REPAIR = "repair"


@dataclass(frozen=True)
class Closing:
    """The ``TERMINAL`` event a recorded decision leads to."""

    terminal: Terminal
    reason: str
    kind: RejectKind | None
    causes: tuple[str, ...]
    """The infrastructure cause of each retry an ``ABANDONED`` item spent; () otherwise."""


@dataclass(frozen=True)
class ItemState:
    """A proposal item's status, derived from its events in append order; never stored.

    Attributes:
        phase: The phase to run next.
        round: The build and validation round; a ``Repair`` starts the next one.
        seq: The last event's ``seq``; the next event is ``seq + 1``.
        proposal_digest: The current proposal (the opened one, or the last triage repair).
        triage: The triage decision on the current proposal, ``None`` until it is triaged.
        program_digest: The program authored in this round and not yet built or failed.
        task_digest: The draft built in this round.
        revision: What the next authoring revises.
        failure_file: The failure the next authoring revises, relative to the item directory.
        repaired_task_digest: The task the round's ``Repair`` condemned; a rebuild to the same digest
            is a no-op repair.
        repair_round: The round whose ``decision.json`` holds the pending ``Repair``.
        invalidate: Steps the round's ``Repair`` condemned, recomputed by every build of the round.
        triage_repairs: Rubric repairs so far.
        build_revisions: Build failures so far, each owed one revision.
        repairs_used: Review ``Repair`` decisions so far.
        retry_causes: The cause of each ``Retry`` decision since the last ``ABANDONED``, so a relaunch
            re-enters with a fresh budget.
        build_host_failures: The cause of each consecutive host failure of the current build since the
            last ``ABANDONED``; a build that finishes, built or failed, clears it.
        band_repairs: Review ``Repair`` decisions issued so far for each band finding kind.
        solved: The solver trials of this validation pass are recorded.
        adversaries_run: The adversary trials of this validation pass are recorded.
        not_before: Unix time before which a retried build or validation does not start.
        closing: The terminal event a recorded decision leads to (phase ``DONE``).
        terminal: The item's terminal, until a later event re-enters it.
    """

    phase: Phase
    round: int
    seq: int
    proposal_digest: str
    triage: TriageDecision | None
    program_digest: str | None
    task_digest: str | None
    revision: RevisionKind
    failure_file: str | None
    repaired_task_digest: str | None
    repair_round: int | None
    invalidate: tuple[str, ...]
    triage_repairs: int
    build_revisions: int
    repairs_used: int
    retry_causes: tuple[str, ...]
    build_host_failures: tuple[InfrastructureCause, ...]
    band_repairs: Mapping[FindingKind, int]
    solved: bool
    adversaries_run: bool
    not_before: float | None
    closing: Closing | None
    terminal: Terminal | None

    @property
    def validation_retries(self) -> int:
        return len(self.retry_causes)


@dataclass(frozen=True)
class IdeaState:
    """An idea item's status: its accepted batch, or how many batches came back empty."""

    seq: int
    items: tuple[str, ...] | None
    """The item ids of the batch that yielded proposals, in slot order; ``None`` until one did."""
    reproposals: int
    """Batches that yielded zero proposals."""
    exhausted: bool


def record_event(
    ledger: Ledger,
    item_id: str,
    round: int,  # noqa: A002 - matches LedgerEntry.round
    kind: EventKind,
    seq: int,
    input_hash: str | None,
    /,
    **attrs: str,
) -> None:
    """Write one ``EVENT`` entry with ``attrs["seq"] = seq`` and ``attrs["schema"] = EVENT_SCHEMA``."""
    if SEQ in attrs or SCHEMA in attrs:
        raise ValueError(f"event attrs may not set {SEQ!r} or {SCHEMA!r}")
    now = time.time()
    ledger.record(
        LedgerEntry(
            item_id=item_id,
            round=round,
            step=kind.value,
            kind=EntryKind.EVENT,
            started=now,
            ended=now,
            input_hash=input_hash,
            attrs={**attrs, SEQ: str(seq), SCHEMA: EVENT_SCHEMA},
        )
    )


def events(entries: Iterable[LedgerEntry]) -> list[LedgerEntry]:
    """The ``EVENT`` entries of one item's ledger, checked: known schema and kind, ``seq`` 1, 2, 3, ...

    Raises:
        ValueError: an unknown schema or event kind, or a gap or repeat in ``seq``.
    """
    found = [entry for entry in entries if entry.kind is EntryKind.EVENT]
    for expected, entry in enumerate(found, start=1):
        schema = entry.attrs.get(SCHEMA)
        if schema != EVENT_SCHEMA:
            raise ValueError(f"{entry.item_id}: event schema {schema!r}, expected {EVENT_SCHEMA!r}")
        EventKind(entry.step)
        seq = int(entry.attrs[SEQ])
        if seq != expected:
            raise ValueError(
                f"{entry.item_id}: event seq {seq} where {expected} was expected; "
                "two processes may be writing this run root"
            )
    return found


UNBUDGETED_TRIALS = frozenset({TrialKind.SOLVER, TrialKind.ADVERSARY})
"""Trial kinds whose rollout-model calls the validation policy bounds (``k``, ``adversary_k``) instead."""


def item_tokens_out(entries: Iterable[LedgerEntry]) -> int:
    """The output tokens the item's budget counts: ``LLM_CALL`` entries outside validation trials.

    A trial's calls are recorded under step ``<kind>/<trial>`` (``validate.solver.ValidationSite.call_ledger``);
    a build step's calls are recorded under the bare step name, which never contains ``/``, so a builder
    step named ``solver`` or ``adversary`` still counts.
    """
    return sum(
        entry.tokens_out or 0 for entry in entries if entry.kind is EntryKind.LLM_CALL and not is_trial_step(entry.step)
    )


def is_trial_step(step: str) -> bool:
    head, separator, _ = step.partition("/")
    return bool(separator) and head in UNBUDGETED_TRIALS


def split(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(LIST_SEPARATOR) if part)


def joined(values: Iterable[str]) -> str:
    return LIST_SEPARATOR.join(values)


def cause_counts(causes: Iterable[str]) -> str:
    """``causes`` as ``cause:count`` pairs in cause order, comma-separated."""
    return joined(f"{cause}{COUNT_SEPARATOR}{n}" for cause, n in sorted(Counter(causes).items()))


def build_host_failures(entries: Iterable[LedgerEntry]) -> Counter[InfrastructureCause]:
    """The item's ``BUILD_INFRASTRUCTURE`` events by cause, over every launch."""
    return Counter(
        InfrastructureCause(entry.attrs["cause"])
        for entry in events(entries)
        if entry.step == EventKind.BUILD_INFRASTRUCTURE
    )


def derive_idea_state(entries: Iterable[LedgerEntry]) -> IdeaState:
    """Fold an idea item's events. Raises ``ValueError`` on a proposal item's event or a bad log."""
    state = IdeaState(seq=0, items=None, reproposals=0, exhausted=False)
    for entry in events(entries):
        kind = EventKind(entry.step)
        if kind not in IDEA_EVENTS:
            raise ValueError(f"{entry.item_id}: {kind} is not an idea event")
        state = replace(state, seq=int(entry.attrs[SEQ]))
        if kind is EventKind.PROPOSED:
            items = split(entry.attrs["items"])
            state = replace(state, items=items) if items else replace(state, reproposals=state.reproposals + 1)
        elif kind is EventKind.IDEA_EXHAUSTED:
            state = replace(state, exhausted=True)
    return state


def derive_state(entries: Iterable[LedgerEntry]) -> ItemState:
    """Fold a proposal item's events into its ``ItemState``.

    Raises:
        ValueError: the log does not start with ``OPENED``, holds an idea event or an event after a
            final terminal, or fails the checks of ``events``.
    """
    found = events(entries)
    if not found or EventKind(found[0].step) is not EventKind.OPENED:
        raise ValueError("an item's event log starts with OPENED")
    opened = found[0]
    assert opened.input_hash is not None
    state = ItemState(
        phase=Phase.TRIAGE,
        round=0,
        seq=1,
        proposal_digest=opened.input_hash,
        triage=None,
        program_digest=None,
        task_digest=None,
        revision=RevisionKind.NONE,
        failure_file=None,
        repaired_task_digest=None,
        repair_round=None,
        invalidate=(),
        triage_repairs=0,
        build_revisions=0,
        repairs_used=0,
        retry_causes=(),
        build_host_failures=(),
        band_repairs={},
        solved=False,
        adversaries_run=False,
        not_before=None,
        closing=None,
        terminal=None,
    )
    for entry in found[1:]:
        if state.terminal in FINAL:
            raise ValueError(f"{entry.item_id}: event {entry.step} after terminal {state.terminal}")
        state = _apply(state, EventKind(entry.step), entry)
    return state


def _apply(state: ItemState, kind: EventKind, entry: LedgerEntry) -> ItemState:
    attrs = entry.attrs
    state = replace(state, seq=int(attrs[SEQ]))
    if kind is not EventKind.TERMINAL:
        state = replace(state, terminal=None)
    match kind:
        case EventKind.OPENED | EventKind.PROPOSED | EventKind.SLOT_FAILED | EventKind.IDEA_EXHAUSTED:
            raise ValueError(f"{entry.item_id}: unexpected {kind} in a proposal item's log")
        case EventKind.TRIAGED:
            decision = TriageDecision(attrs["decision"])
            state = replace(state, triage=decision)
            if decision is TriageDecision.ACCEPT:
                return replace(state, phase=Phase.BUILD)
            if decision is TriageDecision.REJECT:
                return _closing(state, Terminal.REJECTED, "triage rejected the proposal", RejectKind.TASK)
            return state
        case EventKind.TRIAGE_REPAIRED:
            assert entry.input_hash is not None
            return replace(state, proposal_digest=entry.input_hash, triage=None, triage_repairs=state.triage_repairs + 1)
        case EventKind.AUTHORED:
            return replace(state, program_digest=entry.input_hash, revision=RevisionKind.NONE, failure_file=None)
        case EventKind.BUILD_FAILED:
            return replace(
                state,
                program_digest=None,
                revision=RevisionKind.BUILD_FAILURE,
                failure_file=attrs["failure_file"],
                build_revisions=state.build_revisions + 1,
                build_host_failures=(),
                not_before=None,
            )
        case EventKind.BUILD_INFRASTRUCTURE:
            failures = (*state.build_host_failures, InfrastructureCause(attrs["cause"]))
            state = replace(state, build_host_failures=failures)
            if attrs["abandon"] == "true":
                reason = f"build host failures: {cause_counts(failures)}"
                return _closing(state, Terminal.ABANDONED, reason, None, failures)
            return replace(state, not_before=float(attrs["not_before"]))
        case EventKind.BUILT:
            staged = attrs["staged"] == "true"
            return replace(
                state,
                task_digest=entry.input_hash,
                phase=Phase.DECIDE if staged else Phase.CONTROLS,
                solved=False,
                adversaries_run=False,
                build_host_failures=(),
                not_before=None,
            )
        case EventKind.CONTROLS_REPLAYED:
            passed = attrs["passed"] == "true"
            return replace(state, phase=Phase.TRIALS if passed else Phase.DECIDE, not_before=None)
        case EventKind.SOLVED:
            return _trials(replace(state, solved=True))
        case EventKind.ADVERSARIES_RUN:
            return _trials(replace(state, adversaries_run=True))
        case EventKind.DECIDED:
            return _decided(state, entry)
        case EventKind.TERMINAL:
            return _terminal(state, Terminal(attrs["terminal"]))


def _closing(
    state: ItemState, terminal: Terminal, reason: str, kind: RejectKind | None, causes: tuple[str, ...] = ()
) -> ItemState:
    return replace(state, phase=Phase.DONE, closing=Closing(terminal, reason, kind, causes))


def _trials(state: ItemState) -> ItemState:
    return replace(state, phase=Phase.DECIDE) if state.solved and state.adversaries_run else state


def _decided(state: ItemState, entry: LedgerEntry) -> ItemState:
    attrs = entry.attrs
    decision = DecisionKind(attrs["decision"])
    if decision is DecisionKind.ACCEPT:
        band = BandOutcome(attrs["band"])
        reason = CALIBRATED if band is BandOutcome.IN_BAND else f"{OUTSIDE_BAND}: {band}"
        return _closing(state, Terminal.ACCEPTED, reason, None)
    if decision is DecisionKind.REJECT:
        return _closing(state, Terminal.REJECTED, attrs["reasons"], RejectKind(attrs["kind"]))
    if decision is DecisionKind.RETRY:
        causes = (*state.retry_causes, attrs["cause"])
        state = replace(state, retry_causes=causes, solved=False, adversaries_run=False)
        if attrs["abandon"] == "true":
            return _closing(state, Terminal.ABANDONED, attrs["cause"], None, causes)
        return replace(state, phase=Phase.CONTROLS, not_before=float(attrs["not_before"]))
    findings = frozenset(FindingKind(kind) for kind in split(attrs["findings"]))
    band_repairs = Counter(state.band_repairs)
    band_repairs.update(findings - DECISIVE)
    return replace(
        state,
        phase=Phase.BUILD,
        round=state.round + 1,
        program_digest=None,
        task_digest=None,
        revision=RevisionKind.REPAIR,
        failure_file=None,
        repaired_task_digest=entry.input_hash,
        repair_round=state.round,
        invalidate=split(attrs["invalidate"]),
        repairs_used=state.repairs_used + 1,
        band_repairs=dict(band_repairs),
        solved=False,
        adversaries_run=False,
    )


def _terminal(state: ItemState, terminal: Terminal) -> ItemState:
    if terminal is Terminal.FAILED:
        return replace(state, terminal=terminal)
    state = replace(state, terminal=terminal, closing=None)
    if terminal is Terminal.ABANDONED:
        # Abandoned before this round's draft was built, the item re-enters the build of the same program.
        resume = Phase.BUILD if state.task_digest is None else Phase.CONTROLS
        return replace(state, phase=resume, retry_causes=(), build_host_failures=(), not_before=None)
    return replace(state, phase=Phase.DONE)
