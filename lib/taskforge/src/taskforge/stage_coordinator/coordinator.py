# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an item through its stages with the stage functions a caller passes.

A ``StageCoordinator`` holds the caller's ``StageFunctions``, the item budget, the run root, the
event store, the model with its sampling policy, and a transition table. ``run_item`` runs an
item to a terminal:

1. When the caller's log is empty and the event store holds the item's events, ``run_item``
   restores the log from the store.
2. It opens the item with an ``OPENED`` event, or checks the proposal and the budget against the
   digests in the existing ``OPENED`` event.
3. It folds the log with ``derive_state``. An accepted or rejected item returns its terminal here,
   before any stage runs or any file is written.
4. It runs the stage in the state, appends the stage's event and folds the event in, until the
   item reaches a terminal.
5. It saves the log to the event store, after every other file of the item.

Each stage runs through an adapter in this module. The adapter calls the caller's function, writes
the round's files, and returns the stage's event. A build raising ``BuildFailure`` or any other
error except ``ModelUnavailable`` costs a revision, recorded as ``BUILD_FAILED``. Any other
exception from a stage, ``ModelUnavailable`` included, ends the run with ``TERMINAL`` failed. The
coordinator then saves the log and re-raises, and the next ``run_item`` on the item resumes at that
stage, with the step store serving the steps already built.

The coordinator writes a program's manifest to the round directory but cannot read the program
back from it, because a program is code. The build stage therefore asks the author again, with the
revision kept in the state, and refuses a program whose digest differs from the ``AUTHORED`` event.
An author has to return the same program for the same proposal and revision.

The layout of the run root is in ``lib/taskforge/README.md``.
"""

import json
import traceback
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename

from taskforge.content_hash import digest, pretty_json
from taskforge.llm.client import ModelClient, ModelUnavailable
from taskforge.llm.policy import LLMPolicy
from taskforge.stage20_triage.verdict import TriageDecision
from taskforge.stage30_author.author import PROGRAM_MANIFEST_FILE, Revision, RevisionKind, program_manifest
from taskforge.stage40_build.context import BuildFailure
from taskforge.stage40_build.run import DRAFT_DIR, load_draft
from taskforge.stage40_build.step_store import CacheStatus, check_item_id
from taskforge.stage60_review.decision import Accept, Decision, Modify, Reject, Retry
from taskforge.stage_coordinator.event_log import ITEMS_DIR, ItemEventLog, ItemEventStore
from taskforge.stage_coordinator.item_machine import (
    DEFAULT_TRANSITIONS,
    EXIT,
    FINAL,
    REASON_ATTR,
    TERMINAL_ATTR,
    EventKind,
    Exit,
    ItemEvent,
    ItemState,
    Stage,
    Terminal,
    Transitions,
    check_transitions,
    derive_state,
    transition,
)
from taskforge.stage_coordinator.stage_functions import StageFunctions

STORE_DIR = "store"
ROUNDS_DIR = "rounds"
FAILURES_DIR = "build-failures"
VALIDATION_FILE = "validation.json"
DECISION_FILE = "decision.json"
BUDGET_DIGEST = "budget_digest"
TRIAGE_EXITS: Mapping[TriageDecision, Exit] = {TriageDecision.ACCEPT: Exit.ACCEPT, TriageDecision.REJECT: Exit.REJECT}
FAILURE_CHARS = 6000
"""The length of the traceback tail kept from a build failure for the author's revision."""
ATTR_CHARS = 500
"""The length of the failure text kept in a ``BUILD_FAILED`` event. The failure file keeps the rest."""


@dataclass(frozen=True)
class ItemBudget:
    """The revisions and retries each item may spend.

    Attributes:
        max_build_revisions: The build failures an item may revise after. The next failure rejects it.
        max_validation_retries: The review ``Retry`` decisions an item may follow before it is abandoned.
    """

    max_build_revisions: int
    max_validation_retries: int


@dataclass(frozen=True)
class StageCoordinator:
    """The stage functions, budget and storage shared by a run's items.

    Attributes:
        stages: The caller's function for each stage.
        budget: The budget of every item. ``OPENED`` records its digest.
        root: The run root holding ``items/`` and the step store in ``store/``.
        event_store: The store each item's events are saved to on reaching a terminal.
        model: The model build steps call.
        sampling: The sampling policy build steps pass to ``model``.
        model_digest: The model part of every step key, usually ``step_store.model_digest(model, sampling)``.
        transitions: The next stage for each stage and exit, checked by ``check_transitions``.
    """

    stages: StageFunctions
    budget: ItemBudget
    root: Path
    event_store: ItemEventStore
    model: ModelClient
    sampling: LLMPolicy
    model_digest: str
    transitions: Transitions = DEFAULT_TRANSITIONS

    def __post_init__(self) -> None:
        check_transitions(self.transitions)


@dataclass(frozen=True)
class _Item:
    coordinator: StageCoordinator
    item_id: str
    proposal: object
    log: ItemEventLog

    @property
    def directory(self) -> Path:
        return self.coordinator.root / ITEMS_DIR / self.item_id

    def round_dir(self, round: int) -> Path:  # noqa: A002 - the item's round
        return self.directory / ROUNDS_DIR / str(round)


async def run_item(coordinator: StageCoordinator, item_id: str, proposal: object, log: ItemEventLog) -> Terminal:
    """Run ``proposal`` as item ``item_id`` to a terminal, resuming from ``log`` or the event store.

    An item already ``ACCEPTED`` or ``REJECTED`` returns at once. An ``ABANDONED`` item re-enters
    validation with its retries reset, and a ``FAILED`` one resumes at the stage that raised.

    Raises:
        ValueError: ``item_id`` is unsafe as a directory name, the item was opened for another
            proposal or budget, the author returned a program other than the authored one, or the
            log is inconsistent.
        Exception: whatever a stage raised, after the item is saved as failed.
    """
    check_item_id(item_id)
    if not log.events():
        stored = coordinator.event_store.load(item_id)
        if stored is not None:
            log.restore(stored)
    item = _Item(coordinator, item_id, proposal, log)
    _open(item)
    state = derive_state(log.events(), coordinator.transitions)
    if state.terminal is not None and state.terminal in FINAL:
        return state.terminal
    try:
        state = await _advance(item, state)
    except Exception as error:
        attrs = {TERMINAL_ATTR: Terminal.FAILED.value, REASON_ATTR: f"{type(error).__name__}: {error}"}
        log.append(ItemEvent(EventKind.TERMINAL, log.events()[-1].round, None, attrs))
        coordinator.event_store.save(item_id, log.events())
        raise
    coordinator.event_store.save(item_id, log.events())
    assert state.terminal is not None
    return state.terminal


async def _advance(item: _Item, state: ItemState) -> ItemState:
    while True:
        event = await ADAPTERS[state.stage](item, state)
        state = transition(state, item.log.append(event), item.coordinator.transitions)
        if state.terminal is not None:
            return state


def _open(item: _Item) -> None:
    proposal_digest = digest(item.proposal)
    budget_digest = digest(item.coordinator.budget)
    events = item.log.events()
    if not events:
        item.log.append(ItemEvent(EventKind.OPENED, 0, proposal_digest, {BUDGET_DIGEST: budget_digest}))
        return
    first = events[0]
    if first.digest != proposal_digest:
        raise ValueError(f"{item.item_id} was opened for proposal {first.digest}, not {proposal_digest}")
    if first.attrs[BUDGET_DIGEST] != budget_digest:
        raise ValueError(f"{item.item_id} was opened under budget {first.attrs[BUDGET_DIGEST]}, not {budget_digest}")


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_rename(str(path)) as temp:
        Path(temp).write_bytes(content)


async def _triage(item: _Item, state: ItemState) -> ItemEvent:
    way = TRIAGE_EXITS[item.coordinator.stages.triage(item.proposal)]
    return ItemEvent(EventKind.TRIAGED, state.round, state.proposal_digest, {EXIT: way})


def _revision(item: _Item, state: ItemState) -> Revision | None:
    """Read what the round's author call revises: nothing, the last build failure, or review's findings."""
    match state.revision:
        case RevisionKind.NONE:
            return None
        case RevisionKind.BUILD_FAILURE:
            failure = item.round_dir(state.round) / FAILURES_DIR / f"{state.build_revisions}.txt"
            return Revision(kind=state.revision, detail=failure.read_text())
        case RevisionKind.MODIFY:
            path = item.round_dir(state.round - 1) / DECISION_FILE
            decision = _DECISION.validate_json(path.read_bytes())
            assert isinstance(decision, Modify), f"{path} holds {decision}, not a Modify"
            return Revision(kind=state.revision, detail="\n\n".join(decision.findings))


async def _author(item: _Item, state: ItemState) -> ItemEvent:
    """Ask the author for the round's program and write its manifest to ``program.json``."""
    program = item.coordinator.stages.author(item.proposal, _revision(item, state))
    _write(item.round_dir(state.round) / PROGRAM_MANIFEST_FILE, pretty_json(program_manifest(program)).encode())
    attrs: dict[str, str] = {EXIT: Exit.AUTHORED, "revision": state.revision}
    return ItemEvent(EventKind.AUTHORED, state.round, program.digest, attrs)


async def _build(item: _Item, state: ItemState) -> ItemEvent:
    """Build the authored program, recording a failure other than ``ModelUnavailable`` as a spent revision."""
    coordinator = item.coordinator
    program = coordinator.stages.author(item.proposal, _revision(item, state))
    if program.digest != state.program_digest:
        raise ValueError(
            f"{item.item_id}: the author now returns program {program.digest}, not the authored {state.program_digest}"
        )
    try:
        draft = await coordinator.stages.build(
            program,
            item.proposal,
            item.item_id,
            item.round_dir(state.round),
            str(coordinator.root / STORE_DIR),
            coordinator.model,
            coordinator.sampling,
            coordinator.model_digest,
            state.round,
        )
    except ModelUnavailable:
        raise
    except Exception as error:
        failures = state.build_revisions + 1
        relative = Path(ROUNDS_DIR) / str(state.round) / FAILURES_DIR / f"{failures}.txt"
        failure = "".join(traceback.format_exception(error))[-FAILURE_CHARS:]
        _write(item.directory / relative, failure.encode())
        attrs = {
            EXIT: Exit.REVISE if failures <= coordinator.budget.max_build_revisions else Exit.REJECT,
            "step": (error.step or "") if isinstance(error, BuildFailure) else "",
            "failure": f"{type(error).__name__}: {error}"[:ATTR_CHARS],
            "failure_file": str(relative),
        }
        return ItemEvent(EventKind.BUILD_FAILED, state.round, program.digest, attrs)
    steps = draft.provenance.steps
    hits = sum(outcome.status is CacheStatus.HIT for outcome in steps)
    attrs = {EXIT: Exit.BUILT, "program_digest": program.digest, "steps": str(len(steps)), "hits": str(hits)}
    return ItemEvent(EventKind.BUILT, state.round, digest(draft.task), attrs)


async def _validate(item: _Item, state: ItemState) -> ItemEvent:
    """Validate the round's draft and write the summary to ``validation.json``."""
    round_dir = item.round_dir(state.round)
    draft = load_draft(round_dir / DRAFT_DIR)
    if digest(draft.task) != state.task_digest:
        raise ValueError(f"{item.item_id}: the stored draft has digest {digest(draft.task)}, not {state.task_digest}")
    summary = await item.coordinator.stages.validate(draft)
    _write(round_dir / VALIDATION_FILE, pretty_json(summary).encode())
    attrs = {EXIT: Exit.VALIDATED, "summary_digest": digest(summary)}
    return ItemEvent(EventKind.VALIDATED, state.round, state.task_digest, attrs)


async def _review(item: _Item, state: ItemState) -> ItemEvent:
    """Decide on the summary read back from ``validation.json`` and write the decision to ``decision.json``."""
    round_dir = item.round_dir(state.round)
    decision = item.coordinator.stages.review(json.loads((round_dir / VALIDATION_FILE).read_bytes()))
    _write(round_dir / DECISION_FILE, _DECISION.dump_json(decision, indent=2))
    return ItemEvent(EventKind.DECIDED, state.round, state.task_digest, _decision_attrs(item, state, decision))


def _decision_attrs(item: _Item, state: ItemState, decision: Decision) -> dict[str, str]:
    """The ``DECIDED`` attrs of ``decision``. A ``Retry`` past ``max_validation_retries`` exits abandon."""
    used = {"retries_used": str(state.retries)}
    match decision:
        case Accept(band=band):
            return {EXIT: Exit.ACCEPT, "band": band, **used}
        case Reject(kind=kind, reasons=reasons):
            return {EXIT: Exit.REJECT, "kind": kind, "reasons": "; ".join(reasons), **used}
        case Retry(cause=cause, count=count):
            spent = state.retries >= item.coordinator.budget.max_validation_retries
            way = Exit.ABANDON if spent else Exit.RETRY
            return {EXIT: way, "cause": cause, "count": str(count), "retries_used": str(state.retries + 1)}
        case Modify(findings=findings):
            return {EXIT: Exit.MODIFY, "findings": "; ".join(findings), **used}


async def _close(item: _Item, state: ItemState) -> ItemEvent:
    closing = state.closing
    assert closing is not None, f"{item.item_id} is done without a closing"
    attrs = {TERMINAL_ATTR: closing.terminal.value, REASON_ATTR: closing.reason}
    if closing.kind is not None:
        attrs["kind"] = closing.kind.value
    return ItemEvent(EventKind.TERMINAL, state.round, None, attrs)


ADAPTERS: Mapping[Stage, Callable[[_Item, ItemState], Awaitable[ItemEvent]]] = {
    Stage.TRIAGE: _triage,
    Stage.AUTHOR: _author,
    Stage.BUILD: _build,
    Stage.VALIDATE: _validate,
    Stage.REVIEW: _review,
    Stage.DONE: _close,
}
"""The adapter running each stage and returning its event."""

_DECISION: TypeAdapter[Decision] = TypeAdapter(Decision)
