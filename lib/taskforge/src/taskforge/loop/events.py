# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The item event log: ``EntryKind.EVENT`` ledger rows, and item status as a pure fold over them.

Every stage boundary of an item appends one event through the run's ``Ledger``, into the item's own
JSONL file beside its LLM, step and trial spans. An event's ``step`` is its ``EventKind``,
``input_hash`` is the digest of the artifact it points at, and its payload is flat string ``attrs``.
Every event carries ``attrs["seq"]``, contiguous from 1 per item, and ``attrs["schema"] =
EVENT_SCHEMA``; every stage event also carries ``attrs["exit"]``, the ``Exit`` it leaves its stage by.
Large payloads stay in the item directory (proposals, programs, drafts, attempt files,
``calibration.json``, ``decision.json``) and an event names them by digest.

The item machine. A stage runs, records one event with its exit, and the transition table names the
next stage (``DEFAULT_TRANSITIONS``; a caller may pass its own table, which re-targets these edges)::

    stage     event         exit       next      bookkeeping
    --------  ------------  ---------  --------  ------------------------------------------------
    TRIAGE    TRIAGED       accept     AUTHOR
    TRIAGE    TRIAGED       reject     DONE      closing rejected (task)
    AUTHOR    AUTHORED      authored   BUILD     program_digest
    BUILD     BUILT         built      VALIDATE  task_digest; revision none
    BUILD     BUILD_FAILED  revise     AUTHOR    build_revisions + 1; revision build_failure
    BUILD     BUILD_FAILED  reject     DONE      build_revisions + 1; closing rejected (budget)
    VALIDATE  VALIDATED     validated  REVIEW
    REVIEW    DECIDED       accept     DONE      closing accepted (labelled with the band)
    REVIEW    DECIDED       reject     DONE      closing rejected (the decision's kind)
    REVIEW    DECIDED       retry      VALIDATE  retries + 1; not_before
    REVIEW    DECIDED       abandon    DONE      retries + 1; closing abandoned
    REVIEW    DECIDED       modify     AUTHOR    round + 1; task_digest cleared; revision modify
    DONE      TERMINAL      -          -         terminal = the closing's terminal
    any       TERMINAL      -          (stays)   terminal failed, when ``terminal`` is ``failed``

A budget verdict is recorded on the event that spends the budget (``BUILD_FAILED``'s exit is revise
or reject, ``DECIDED``'s retry or abandon), so the fold needs no policy.

The idea machine (item ``idea--<idea_id>``) records proposal batches::

    event           attrs                                effect
    --------------  -----------------------------------  -----------------------------------------
    SLOT_FAILED     slot, error                          none; the batch's PROPOSED follows
    PROPOSED        items, slots, failures, reproposal   items set, or reproposals + 1 when empty
    IDEA_EXHAUSTED  reproposals                          exhausted

``derive_state`` folds a proposal item's events into its ``ItemState`` with ``transition``;
``derive_idea_state`` folds an idea item's. Status is never stored. A gap or a repeat in ``seq``, the
only sign of two processes writing one run root, and an unknown schema raise ``ValueError``.

An item that ended ``ABANDONED`` (validation retries spent) or ``FAILED`` (an unhandled exception) is
not final: when it is run again, the next event clears ``terminal``; an abandoned item resumes at
``VALIDATE`` with a fresh retry count, a failed one at the stage it failed in. ``ACCEPTED`` and
``REJECTED`` are final.
"""

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType

from taskforge.builder.author import RevisionKind
from taskforge.ledger.records import EntryKind, Ledger, LedgerEntry
from taskforge.review.decision import BandOutcome, RejectKind

EVENT_SCHEMA = "2"
SEQ = "seq"
SCHEMA = "schema"
EXIT = "exit"
LIST_SEPARATOR = ","
CALIBRATED = "calibrated"
OUTSIDE_BAND = "accepted outside the band"
TRIAGE_REJECTED = "triage rejected the proposal"


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
    """input_hash: the proposal digest. attrs: exit (accept, reject)."""
    AUTHORED = "authored"
    """input_hash: the program digest. attrs: exit (authored), revision (the ``RevisionKind`` it answered)."""
    BUILD_FAILED = "build_failed"
    """input_hash: the program digest. attrs: exit (revise, reject), step, failure (truncated), failure_file."""
    BUILT = "built"
    """input_hash: the task digest. attrs: exit (built), program_digest, steps, hits."""
    VALIDATED = "validated"
    """input_hash: the task digest. attrs: exit (validated), graded, solved, timed_out, ungraded and
    findings (finding kinds, comma-separated)."""
    DECIDED = "decided"
    """input_hash: the task digest. attrs: exit (accept, reject, retry, abandon, modify), retries_used;
    band (a ``BandOutcome``), solved, graded and solve_rate (three decimals) for accept; kind (a
    ``RejectKind``) and reasons for reject; cause and count for retry and abandon, and not_before for
    retry; findings for modify."""
    TERMINAL = "terminal"
    """attrs: terminal (a ``Terminal``), reason, and kind (a ``RejectKind``) when rejected."""


class Stage(StrEnum):
    TRIAGE = "triage"
    AUTHOR = "author"
    BUILD = "build"
    VALIDATE = "validate"
    REVIEW = "review"
    DONE = "done"
    """A closing is recorded and the ``TERMINAL`` event is next."""


class Exit(StrEnum):
    """How a stage ended: ``attrs["exit"]`` on every stage event, the key into the transition table."""

    ACCEPT = "accept"
    REJECT = "reject"
    AUTHORED = "authored"
    BUILT = "built"
    REVISE = "revise"
    VALIDATED = "validated"
    RETRY = "retry"
    ABANDON = "abandon"
    MODIFY = "modify"


EXITS: Mapping[Stage, frozenset[Exit]] = {
    Stage.TRIAGE: frozenset({Exit.ACCEPT, Exit.REJECT}),
    Stage.AUTHOR: frozenset({Exit.AUTHORED}),
    Stage.BUILD: frozenset({Exit.BUILT, Exit.REVISE, Exit.REJECT}),
    Stage.VALIDATE: frozenset({Exit.VALIDATED}),
    Stage.REVIEW: frozenset({Exit.ACCEPT, Exit.REJECT, Exit.RETRY, Exit.ABANDON, Exit.MODIFY}),
}
"""Every exit of every stage; ``DONE`` has none (its ``TERMINAL`` event ends the item)."""

EVENT_STAGES: Mapping[EventKind, Stage] = {
    EventKind.TRIAGED: Stage.TRIAGE,
    EventKind.AUTHORED: Stage.AUTHOR,
    EventKind.BUILD_FAILED: Stage.BUILD,
    EventKind.BUILT: Stage.BUILD,
    EventKind.VALIDATED: Stage.VALIDATE,
    EventKind.DECIDED: Stage.REVIEW,
}
"""The stage each stage event is recorded by."""

type Transitions = Mapping[tuple[Stage, Exit], Stage]
"""A transition table: the next stage for each stage and exit."""

DEFAULT_TRANSITIONS: Transitions = MappingProxyType(
    {
        (Stage.TRIAGE, Exit.ACCEPT): Stage.AUTHOR,
        (Stage.TRIAGE, Exit.REJECT): Stage.DONE,
        (Stage.AUTHOR, Exit.AUTHORED): Stage.BUILD,
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
"""The default item machine; read-only, so a ``Loop`` may hold it as its default."""

CLOSING_EDGES = frozenset(
    {
        (Stage.TRIAGE, Exit.REJECT),
        (Stage.BUILD, Exit.REJECT),
        (Stage.REVIEW, Exit.ACCEPT),
        (Stage.REVIEW, Exit.REJECT),
        (Stage.REVIEW, Exit.ABANDON),
    }
)
"""The edges whose event records a ``Closing``; exactly these lead to ``DONE``."""


def check_transitions(table: Transitions) -> None:
    """Raise ``ValueError`` unless ``table`` maps every stage's exits, and only those, to stages.

    ``DONE`` is never a source; the closing edges (``CLOSING_EDGES``) and only they lead to ``DONE``,
    since the event that closes an item is the one that records why.
    """
    expected = {(stage, way) for stage, ways in EXITS.items() for way in ways}
    missing = sorted(f"{stage}/{way}" for stage, way in expected - table.keys())
    unknown = sorted(f"{stage}/{way}" for stage, way in table.keys() - expected)
    if missing or unknown:
        raise ValueError(f"a transition table needs exactly the stage exits: missing {missing}, unknown {unknown}")
    for edge, target in table.items():
        if not isinstance(target, Stage):
            raise ValueError(f"transition {edge[0]}/{edge[1]} leads to {target!r}, not a Stage")
        if (target is Stage.DONE) != (edge in CLOSING_EDGES):
            raise ValueError(f"transition {edge[0]}/{edge[1]} -> {target}: only a closing edge leads to done")


class ProposalOrigin(StrEnum):
    """Where an item's proposal came from: a ``PROPOSED`` batch, or the caller of ``run_item``."""

    GENERATED = "generated"
    SUPPLIED = "supplied"


IDEA_EVENTS = frozenset({EventKind.PROPOSED, EventKind.SLOT_FAILED, EventKind.IDEA_EXHAUSTED})


class Terminal(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    """Validation retries spent; re-entered at ``VALIDATE`` when the item is run again."""
    FAILED = "failed"
    """An unhandled exception; re-entered at its stage when the item is run again."""


FINAL = frozenset({Terminal.ACCEPTED, Terminal.REJECTED})


@dataclass(frozen=True)
class Closing:
    """The ``TERMINAL`` event a recorded closing edge leads to."""

    terminal: Terminal
    reason: str
    kind: RejectKind | None


@dataclass(frozen=True)
class ItemState:
    """A proposal item's status, derived from its events in append order; never stored.

    Attributes:
        stage: The stage to run next.
        round: The author, build and validation round; a ``modify`` exit starts the next one.
        seq: The last event's ``seq``; the next event is ``seq + 1``.
        proposal_digest: The opened proposal.
        program_digest: The program last authored.
        task_digest: The draft built in this round; ``None`` until it is built.
        revision: What the next author call revises; the build re-asks the author with it, so only
            ``BUILT`` clears it.
        build_revisions: Build failures so far, each owed one revision.
        retries: Review ``Retry`` decisions since the item was opened or last abandoned.
        not_before: Unix time before which a retried validation does not start.
        closing: The terminal event a closing edge leads to (stage ``DONE``).
        terminal: The item's terminal, until a later event re-enters it.
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
    not_before: float | None
    closing: Closing | None
    terminal: Terminal | None


@dataclass(frozen=True)
class IdeaState:
    """An idea item's status: its accepted batch, or how many batches came back empty."""

    seq: int
    items: tuple[str, ...] | None
    """The item ids of the batch that yielded proposals, in slot order; ``None`` until one did."""
    reproposals: int
    """Batches that yielded zero proposals."""
    exhausted: bool


@dataclass(frozen=True)
class Event:
    """One event before it is numbered and recorded: its kind, the digest it points at, its attrs."""

    kind: EventKind
    input_hash: str | None
    attrs: Mapping[str, str]


def record_event(
    ledger: Ledger,
    item_id: str,
    round: int,  # noqa: A002 - matches LedgerEntry.round
    seq: int,
    event: Event,
) -> LedgerEntry:
    """Write ``event`` as one ``EVENT`` entry with ``attrs["seq"] = seq`` and ``attrs["schema"] = EVENT_SCHEMA``.

    Returns the entry written.
    """
    if SEQ in event.attrs or SCHEMA in event.attrs:
        raise ValueError(f"event attrs may not set {SEQ!r} or {SCHEMA!r}")
    now = time.time()
    entry = LedgerEntry(
        item_id=item_id,
        round=round,
        step=event.kind.value,
        kind=EntryKind.EVENT,
        started=now,
        ended=now,
        input_hash=event.input_hash,
        attrs={**event.attrs, SEQ: str(seq), SCHEMA: EVENT_SCHEMA},
    )
    ledger.record(entry)
    return entry


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


def split(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(LIST_SEPARATOR) if part)


def joined(values: Iterable[str]) -> str:
    return LIST_SEPARATOR.join(values)


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


def opened(entry: LedgerEntry) -> ItemState:
    """The state an ``OPENED`` event starts a proposal item in: ``TRIAGE``, round 0."""
    if entry.kind is not EntryKind.EVENT or EventKind(entry.step) is not EventKind.OPENED:
        raise ValueError(f"{entry.item_id}: an item's event log starts with OPENED, not {entry.step}")
    assert entry.input_hash is not None, "OPENED points at its proposal"
    return ItemState(
        stage=Stage.TRIAGE,
        round=0,
        seq=int(entry.attrs[SEQ]),
        proposal_digest=entry.input_hash,
        program_digest=None,
        task_digest=None,
        revision=RevisionKind.NONE,
        build_revisions=0,
        retries=0,
        not_before=None,
        closing=None,
        terminal=None,
    )


def transition(state: ItemState, entry: LedgerEntry, table: Transitions = DEFAULT_TRANSITIONS) -> ItemState:
    """The state after ``entry``, the item's next event: pure, and the only place an item's state moves.

    A ``TERMINAL`` event ends the item at its closing (or marks it failed in place). Any other event
    must be its stage's (``EVENT_STAGES``); its exit picks the next stage from ``table``, and the
    event's (kind, exit) pair does the bookkeeping of the module docstring's table.

    Raises:
        ValueError: an event after a final terminal, out of ``seq`` order, out of its stage, with an
            exit its stage or kind does not have, or a ``TERMINAL`` that does not match the closing.
    """
    item = entry.item_id
    kind = EventKind(entry.step)
    attrs = entry.attrs
    if state.terminal in FINAL:
        raise ValueError(f"{item}: event {kind} after terminal {state.terminal}")
    seq = int(attrs[SEQ])
    if seq != state.seq + 1:
        raise ValueError(f"{item}: event seq {seq} where {state.seq + 1} was expected")
    state = replace(state, seq=seq, terminal=None)
    if kind is EventKind.TERMINAL:
        return _terminal(state, entry)
    stage = EVENT_STAGES.get(kind)
    if stage is None:
        raise ValueError(f"{item}: unexpected {kind} in a proposal item's log")
    if stage is not state.stage:
        raise ValueError(f"{item}: {kind} is recorded by stage {stage}, but the item is in stage {state.stage}")
    way = Exit(attrs[EXIT])
    try:
        target = table[(state.stage, way)]
    except KeyError:
        raise ValueError(f"{item}: {kind} exits {way}, which stage {state.stage} has no edge for") from None
    state = replace(state, stage=target)
    match (kind, way):
        case (EventKind.TRIAGED, Exit.ACCEPT) | (EventKind.VALIDATED, Exit.VALIDATED):
            return state
        case (EventKind.TRIAGED, Exit.REJECT):
            return replace(state, closing=Closing(Terminal.REJECTED, TRIAGE_REJECTED, RejectKind.TASK))
        case (EventKind.AUTHORED, Exit.AUTHORED):
            return replace(state, program_digest=entry.input_hash)
        case (EventKind.BUILT, Exit.BUILT):
            return replace(state, task_digest=entry.input_hash, revision=RevisionKind.NONE)
        case (EventKind.BUILD_FAILED, Exit.REVISE):
            return replace(state, build_revisions=state.build_revisions + 1, revision=RevisionKind.BUILD_FAILURE)
        case (EventKind.BUILD_FAILED, Exit.REJECT):
            failures = state.build_revisions + 1
            reason = f"no buildable program after {failures} build failures"
            closing = Closing(Terminal.REJECTED, reason, RejectKind.BUDGET)
            return replace(state, build_revisions=failures, revision=RevisionKind.BUILD_FAILURE, closing=closing)
        case (EventKind.DECIDED, Exit.ACCEPT):
            band = BandOutcome(attrs["band"])
            reason = CALIBRATED if band is BandOutcome.IN_BAND else f"{OUTSIDE_BAND}: {band}"
            return replace(state, closing=Closing(Terminal.ACCEPTED, reason, None))
        case (EventKind.DECIDED, Exit.REJECT):
            return replace(state, closing=Closing(Terminal.REJECTED, attrs["reasons"], RejectKind(attrs["kind"])))
        case (EventKind.DECIDED, Exit.RETRY):
            return replace(state, retries=state.retries + 1, not_before=float(attrs["not_before"]))
        case (EventKind.DECIDED, Exit.ABANDON):
            closing = Closing(Terminal.ABANDONED, attrs["cause"], None)
            return replace(state, retries=state.retries + 1, closing=closing)
        case (EventKind.DECIDED, Exit.MODIFY):
            return replace(state, round=state.round + 1, task_digest=None, revision=RevisionKind.MODIFY, not_before=None)
        case _:
            raise ValueError(f"{item}: {kind} cannot exit {way}")


def _terminal(state: ItemState, entry: LedgerEntry) -> ItemState:
    terminal = Terminal(entry.attrs["terminal"])
    if terminal is Terminal.FAILED:
        return replace(state, terminal=terminal)
    closing = state.closing
    if state.stage is not Stage.DONE or closing is None or closing.terminal is not terminal:
        raise ValueError(
            f"{entry.item_id}: TERMINAL {terminal} in stage {state.stage} with closing "
            f"{None if closing is None else closing.terminal}"
        )
    state = replace(state, terminal=terminal, closing=None)
    if terminal is Terminal.ABANDONED:
        return replace(state, stage=Stage.VALIDATE, retries=0, not_before=None)
    return state


def derive_state(entries: Iterable[LedgerEntry], table: Transitions = DEFAULT_TRANSITIONS) -> ItemState:
    """Fold a proposal item's events into its ``ItemState``: ``opened``, then ``transition`` per event.

    Raises:
        ValueError: the log does not start with ``OPENED``, or fails the checks of ``events`` or
            ``transition``.
    """
    found = events(entries)
    if not found:
        raise ValueError("an item's event log starts with OPENED")
    state = opened(found[0])
    for entry in found[1:]:
        state = transition(state, entry, table)
    return state
