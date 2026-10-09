# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One unattended run in one asyncio process: every idea and every item concurrently, resumed from the logs.

``run_queue`` starts every idea as a task; as each idea's batch parses, and its item ids are checked
against every batch parsed before it (``check_item_ids``), it starts every item that is not finished as
a task. Concurrency is bounded by ``LoopServices.slots`` (the run's width), which
``loop.program.run_item`` acquires around each model- or sandbox-bound phase, so an item sleeping on a
retry backoff holds no slot. There is no request limiter: a phase fans out further (solver trials,
adversary roles, controls), and ``GlmClient`` already pools connections and holds on router drain. The
first throttle is added after a failure ``RunSummary`` shows.

An item's status is its event log (``loop.events.derive_state``). A launch on an existing run root
skips items whose log ends in ``TERMINAL(ACCEPTED)`` or ``TERMINAL(REJECTED)``, skips ``FAILED`` items
unless ``FailedItems.RETRY`` asks for them, and re-enters ``ABANDONED`` ones (validation retries ran
out on an infrastructure cause, or build retries on a machine host failure, which a later launch may
not hit). Inside an item ``run_item`` resumes
from the sub-phase its log names, and validation only from the trials not settled on disk. An idea
whose batch is in its log is not proposed again.

``RunSummary`` is where a run exports its accepted tasks: every ``ACCEPTED`` item is listed as an
``AcceptedTask`` with its draft, the synthesis pass rate (solved of ``k`` and the solve rate) of the
calibration summary it was accepted on, and the band outcome of the ``Accept`` decision: ``IN_BAND``, or
the band kind the policy's ``BandRules`` chose to accept outside the band. Every item whose final
decision carries a calibration summary lists its noted-tier adversary passes (``NotedPass``). Both are
read from the item directories, so a relaunch exports items an earlier launch finished.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from taskforge.builder.infrastructure import InfrastructureCause
from taskforge.builder.run import DRAFT_DIR, item_id_for
from taskforge.ledger.jsonl import JsonlLedger, ledger_files, read_entries
from taskforge.ledger.records import EntryKind, LedgerEntry
from taskforge.llm.client import GlmUnavailable
from taskforge.loop.events import (
    FINAL,
    EventKind,
    ProposalOrigin,
    Terminal,
    build_host_failures,
    derive_state,
    events,
)
from taskforge.loop.policy import LoopPolicy
from taskforge.loop.program import (
    ITEMS_DIR,
    LEDGER_DIR,
    ROUNDS_DIR,
    LoopServices,
    evidence_dir,
    idea_item_id,
    run_idea,
    run_item,
)
from taskforge.proposal.model import TaskProposal
from taskforge.review.decision import DECISION_FILE, Accept, BandOutcome, Reject, load_decision
from taskforge.validate.adversary import AdversaryRole
from taskforge.validate.calibration import CalibrationSummary, DefectTier
from taskforge.validate.outcome import Cause

logger = logging.getLogger(__name__)

CAUSES = frozenset(str(cause) for cause in Cause)


@dataclass(frozen=True)
class AcceptedTask:
    """The exported record of one ``ACCEPTED`` item.

    Attributes:
        round: The round whose task was accepted.
        task_digest: The accepted draft's task digest.
        draft: The accepted draft's directory, relative to the run root.
        solved: Solver trials that passed in the accepted round (the synthesis pass count).
        k: Solver trials the validation policy asks for.
        solve_rate: ``solved`` over the graded solver trials.
        band: The accept decision's band outcome: ``IN_BAND``, or the band kind accepted outside the band.
    """

    round: int
    task_digest: str
    draft: str
    solved: int
    k: int
    solve_rate: float
    band: BandOutcome


@dataclass(frozen=True)
class NotedPass:
    """One adversary trial the calibration tiered ``NOTED``: recorded, never blocking an accept."""

    role: AdversaryRole
    trial: int
    rule: str
    reason: str


class FailedItems(StrEnum):
    """What a launch does with items an earlier launch ended in ``TERMINAL(FAILED)``."""

    SKIP = "skip"
    RETRY = "retry"


@dataclass(frozen=True)
class RunSummary:
    """How every item of a run ended, and the infrastructure failures the run observed.

    Attributes:
        items: Item id to its terminal, for every item the launch reached, including skipped ones.
        failed: Item id, or ``idea--<idea_id>``, to the class of the exception that ended it in this launch.
        ungraded_causes: Ungraded trial attempts by cause over every TRIAL entry in the run's ledger, so a
            ``MODEL_UNAVAILABLE`` wave is observed rather than guessed.
        model_unavailable: ``GlmUnavailable`` raised outside trials (proposing, triage, authoring, building).
        build_infrastructure: Build host failures by cause over the ``BUILD_INFRASTRUCTURE`` events of every
            item that reached a terminal. ``run_item`` retries such a build up to
            ``LoopPolicy.max_build_retries`` times, then ends the item ``ABANDONED``, and the next launch
            re-enters its build. A cause in ``HOST_REJECTIONS`` is not retried: its one event ends the
            item ``ABANDONED`` at once.
        accepted: Item id to its ``AcceptedTask``, for every item that ended ``ACCEPTED``.
        noted: Item id to its noted-tier adversary passes, for every item whose final decision carries a
            calibration summary with at least one.
    """

    items: Mapping[str, Terminal]
    failed: Mapping[str, str]
    ungraded_causes: Counter[Cause]
    model_unavailable: int
    build_infrastructure: Counter[InfrastructureCause]
    accepted: Mapping[str, AcceptedTask]
    noted: Mapping[str, tuple[NotedPass, ...]]

    def summary_json(self) -> dict[str, object]:
        """The summary as ``summary.json`` holds it, with terminal counts first."""
        return {
            "terminals": {str(t): n for t, n in sorted(Counter(self.items.values()).items())},
            "items": {item: str(t) for item, t in sorted(self.items.items())},
            "failed": dict(sorted(self.failed.items())),
            "ungraded_causes": {str(c): n for c, n in sorted(self.ungraded_causes.items())},
            "model_unavailable": self.model_unavailable,
            "build_infrastructure": {str(c): n for c, n in sorted(self.build_infrastructure.items())},
            "accepted": {item: asdict(task) for item, task in sorted(self.accepted.items())},
            "noted": {item: [asdict(note) for note in notes] for item, notes in sorted(self.noted.items())},
        }


def item_terminal(ledger_dir: Path, item_id: str) -> Terminal | None:
    """The terminal the item's event log ends in, or None for an item without events or a terminal."""
    path = JsonlLedger(ledger_dir).path_for(item_id)
    if not path.exists():
        return None
    entries = list(read_entries(path))
    if not any(entry.kind is EntryKind.EVENT for entry in entries):
        return None
    return derive_state(entries).terminal


def enters(terminal: Terminal | None, failed: FailedItems) -> bool:
    """Whether a launch runs an item whose log ends in ``terminal``."""
    if terminal is Terminal.FAILED:
        return failed is FailedItems.RETRY
    return terminal not in FINAL


def ungraded_causes(ledger_dir: Path) -> Counter[Cause]:
    """Ungraded trial attempts by cause over every item log under ``ledger_dir``."""
    return Counter(
        Cause(entry.cause)
        for path in ledger_files(ledger_dir)
        for entry in read_entries(path)
        if entry.kind is EntryKind.TRIAL and entry.cause in CAUSES
    )


@dataclass(frozen=True)
class DecidedRound:
    """The round of an item's last ``DECIDED`` event, the calibration summary its decision carries, and the
    band outcome of an ``Accept`` (``IN_BAND`` for a ``Reject``, which is never exported)."""

    round: int
    summary: CalibrationSummary
    band: BandOutcome


def final_round(root: Path, item_id: str, entries: Sequence[LedgerEntry]) -> DecidedRound | None:
    """The item's last decision when it accepts, or rejects with a calibration summary; ``None`` for an item
    that never decided, or whose last decision repairs, retries, or rejects without a summary (a staged
    draft)."""
    decided = [entry for entry in events(entries) if entry.step == EventKind.DECIDED]
    if not decided:
        return None
    last = decided[-1]
    assert last.input_hash is not None
    evidence = evidence_dir(root / ITEMS_DIR / item_id, last.round, last.input_hash)
    match load_decision(evidence / DECISION_FILE):
        case Accept(summary=summary, band=band):
            return DecidedRound(last.round, summary, band)
        case Reject(summary=CalibrationSummary() as summary):
            return DecidedRound(last.round, summary, BandOutcome.IN_BAND)
        case _:
            return None


def accepted_task(
    item_id: str, round: int, summary: CalibrationSummary, band: BandOutcome  # noqa: A002
) -> AcceptedTask:
    """The exported record of an item accepted on ``summary`` in ``round`` with band outcome ``band``."""
    if summary.solve_rate is None:
        raise ValueError(f"{item_id}: accepted on a summary with no graded solver trial")
    return AcceptedTask(
        round=round,
        task_digest=summary.task_digest,
        draft=str(Path(ITEMS_DIR) / item_id / ROUNDS_DIR / str(round) / DRAFT_DIR),
        solved=summary.solver.solved,
        k=summary.k,
        solve_rate=summary.solve_rate,
        band=band,
    )


def noted_passes(summary: CalibrationSummary) -> tuple[NotedPass, ...]:
    return tuple(NotedPass(a.role, a.index, a.rule, a.reason) for a in summary.assessments if a.tier is DefectTier.NOTED)


@dataclass
class _Tally:
    items: dict[str, Terminal] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    model_unavailable: int = 0
    build_infrastructure: Counter[InfrastructureCause] = field(default_factory=Counter)
    accepted: dict[str, AcceptedTask] = field(default_factory=dict)
    noted: dict[str, tuple[NotedPass, ...]] = field(default_factory=dict)
    item_ideas: Mapping[str, str] = field(default_factory=dict)

    def record(self, root: Path, item_id: str, terminal: Terminal, entries: Sequence[LedgerEntry]) -> None:
        """Record a finished item's terminal, build host failures, accepted record and noted passes."""
        self.items[item_id] = terminal
        self.build_infrastructure += build_host_failures(entries)
        final = final_round(root, item_id, entries) if terminal in FINAL else None
        if final is None:
            return
        if terminal is Terminal.ACCEPTED:
            self.accepted[item_id] = accepted_task(item_id, final.round, final.summary, final.band)
        notes = noted_passes(final.summary)
        if notes:
            self.noted[item_id] = notes

    def failure(self, key: str, error: Exception) -> None:
        logger.error("%s failed", key, exc_info=error)
        self.failed[key] = type(error).__name__
        if isinstance(error, GlmUnavailable):
            self.model_unavailable += 1


async def _gather(coroutines: Sequence[Coroutine[Any, Any, None]]) -> None:
    """Run ``coroutines`` concurrently; each handles its own ``Exception``, so one never cancels another."""
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result


async def _item(
    proposal: TaskProposal, policy: LoopPolicy, services: LoopServices, failed: FailedItems, tally: _Tally
) -> None:
    item_id = item_id_for(proposal)
    ledger_dir = services.root / LEDGER_DIR
    try:
        terminal = item_terminal(ledger_dir, item_id)
        if enters(terminal, failed):
            terminal = await run_item(proposal, ProposalOrigin.GENERATED, policy, services)
        assert terminal is not None
        tally.record(services.root, item_id, terminal, list(read_entries(JsonlLedger(ledger_dir).path_for(item_id))))
    except Exception as error:
        tally.items[item_id] = Terminal.FAILED
        tally.failure(item_id, error)


async def _idea[IdeaT](
    idea_id: str, idea: IdeaT, policy: LoopPolicy, services: LoopServices[IdeaT], failed: FailedItems, tally: _Tally
) -> None:
    try:
        proposals = await run_idea(idea_id, idea, policy, services)
    except Exception as error:
        tally.failure(idea_item_id(idea_id), error)
        return
    tally.item_ideas = check_item_ids(tally.item_ideas, proposals, idea_id)
    await _gather([_item(proposal, policy, services, failed, tally) for proposal in proposals])


def check_item_ids(seen: Mapping[str, str], batch: Sequence[TaskProposal], idea_id: str) -> Mapping[str, str]:
    """``seen``, which maps item ids to the idea whose batch produced them, with ``batch``'s ids added for
    ``idea_id``.

    Raises:
        ValueError: an item id of ``batch`` (``item_id_for``) is in ``seen`` or repeats within ``batch``; the
            message names each such id and both ideas.
    """
    registry = dict(seen)
    duplicates = []
    for proposal in batch:
        item_id = item_id_for(proposal)
        if item_id in registry:
            duplicates.append(f"{item_id} from ideas {registry[item_id]} and {idea_id}")
        else:
            registry[item_id] = idea_id
    if duplicates:
        raise ValueError(f"proposals share item ids: {'; '.join(duplicates)}")
    return registry


async def run_queue[IdeaT](
    ideas: Mapping[str, IdeaT], policy: LoopPolicy, services: LoopServices[IdeaT], failed: FailedItems
) -> RunSummary:
    """Run every idea and every unfinished item of the run rooted at ``services.root`` to a terminal.

    An idea's or item's exception is recorded in ``RunSummary.failed`` and never cancels its siblings;
    cancelling the run cancels every item, and the next launch resumes them from their logs.

    Raises:
        ValueError: an idea's batch shares an item id with an earlier batch (``check_item_ids``); the run
            cancels every item still running, and none of the idea's items starts.
    """
    tally = _Tally()
    try:
        async with asyncio.TaskGroup() as group:
            for idea_id, idea in ideas.items():
                group.create_task(_idea(idea_id, idea, policy, services, failed, tally))
    except ExceptionGroup as failure:
        raise failure.exceptions[0] from failure
    return RunSummary(
        items=tally.items,
        failed=tally.failed,
        ungraded_causes=ungraded_causes(services.root / LEDGER_DIR),
        model_unavailable=tally.model_unavailable,
        build_infrastructure=tally.build_infrastructure,
        accepted=tally.accepted,
        noted=tally.noted,
    )
