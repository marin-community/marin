# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One unattended run in one asyncio process: every idea and every item concurrently, resumed from the logs.

``run_queue`` starts every idea as a task; as each idea's batch parses it starts every item that is not
finished as a task. Concurrency is bounded by ``LoopServices.slots`` (the run's width), which
``loop.program.run_item`` acquires around each model- or sandbox-bound phase, so an item sleeping on a
retry backoff holds no slot. There is no request limiter: a phase fans out further (solver trials,
adversary roles, controls), and ``GlmClient`` already pools connections and holds on router drain. The
first throttle is added after a failure ``RunSummary`` shows.

An item's status is its event log (``loop.events.derive_state``). A launch on an existing run root
skips items whose log ends in ``TERMINAL(ACCEPTED)`` or ``TERMINAL(REJECTED)``, skips ``FAILED`` items
unless ``FailedItems.RETRY`` asks for them, and re-enters ``ABANDONED`` ones (validation retries ran
out on an infrastructure cause, which a later launch may not hit). Inside an item ``run_item`` resumes
from the sub-phase its log names, and validation only from the trials not settled on disk. An idea
whose batch is in its log is not proposed again.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from taskforge.build.infrastructure import BuildInfrastructureFailure, InfrastructureCause
from taskforge.build.run import item_id_for
from taskforge.ledger.jsonl import JsonlLedger, ledger_files, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import GlmUnavailable
from taskforge.loop.events import FINAL, Terminal, derive_state
from taskforge.loop.policy import LoopPolicy
from taskforge.loop.program import LEDGER_DIR, LoopServices, idea_item_id, run_idea, run_item
from taskforge.proposal.model import TaskProposal
from taskforge.validate.outcome import Cause

logger = logging.getLogger(__name__)

CAUSES = frozenset(str(cause) for cause in Cause)


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
        build_infrastructure: ``BuildInfrastructureFailure`` raised by builds, by cause; the item ends
            ``FAILED`` without spending a revision and re-enters its build when a launch retries it.
    """

    items: Mapping[str, Terminal]
    failed: Mapping[str, str]
    ungraded_causes: Counter[Cause]
    model_unavailable: int
    build_infrastructure: Counter[InfrastructureCause]

    def summary_json(self) -> dict[str, object]:
        """The summary as ``summary.json`` holds it, with terminal counts first."""
        return {
            "terminals": {str(t): n for t, n in sorted(Counter(self.items.values()).items())},
            "items": {item: str(t) for item, t in sorted(self.items.items())},
            "failed": dict(sorted(self.failed.items())),
            "ungraded_causes": {str(c): n for c, n in sorted(self.ungraded_causes.items())},
            "model_unavailable": self.model_unavailable,
            "build_infrastructure": {str(c): n for c, n in sorted(self.build_infrastructure.items())},
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


@dataclass
class _Tally:
    items: dict[str, Terminal] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    model_unavailable: int = 0
    build_infrastructure: Counter[InfrastructureCause] = field(default_factory=Counter)

    def failure(self, key: str, error: Exception) -> None:
        logger.error("%s failed", key, exc_info=error)
        self.failed[key] = type(error).__name__
        if isinstance(error, GlmUnavailable):
            self.model_unavailable += 1
        if isinstance(error, BuildInfrastructureFailure):
            self.build_infrastructure[error.cause] += 1


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
    try:
        terminal = item_terminal(services.root / LEDGER_DIR, item_id)
        if not enters(terminal, failed):
            assert terminal is not None
            tally.items[item_id] = terminal
            return
        tally.items[item_id] = await run_item(proposal, policy, services)
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
    await _gather([_item(proposal, policy, services, failed, tally) for proposal in proposals])


async def run_queue[IdeaT](
    ideas: Mapping[str, IdeaT], policy: LoopPolicy, services: LoopServices[IdeaT], failed: FailedItems
) -> RunSummary:
    """Run every idea and every unfinished item of the run rooted at ``services.root`` to a terminal.

    An idea's or item's exception is recorded in ``RunSummary.failed`` and never cancels its siblings;
    cancelling the run cancels every item, and the next launch resumes them from their logs.
    """
    tally = _Tally()
    await _gather([_idea(idea_id, idea, policy, services, failed, tally) for idea_id, idea in ideas.items()])
    return RunSummary(
        items=tally.items,
        failed=tally.failed,
        ungraded_causes=ungraded_causes(services.root / LEDGER_DIR),
        model_unavailable=tally.model_unavailable,
        build_infrastructure=tally.build_infrastructure,
    )
