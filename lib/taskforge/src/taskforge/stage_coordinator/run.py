# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run proposals as they arrive, isolating each workpiece's failures from its siblings.

A workpiece whose run raises stays HELD at the stage that raised, as ``run_workpiece`` recorded it; the next run
replays its event log and resumes there, so this wrapper keeps no retry budget of its own. Admission errors stop
new work; already admitted workpieces finish before the error is raised. Resource pools, rather than this wrapper,
bound concurrency.
"""

import asyncio
import logging
from collections.abc import AsyncIterable

from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage_coordinator.coordinator import StageCoordinator, run_workpiece
from taskforge.stage_coordinator.workpiece_machine import Terminal

logger = logging.getLogger(__name__)


async def run_workpieces(
    coordinator: StageCoordinator, proposals: AsyncIterable[tuple[str, TaskProposal]]
) -> dict[str, Terminal]:
    """Run each proposal ``proposals`` yields as workpiece ``workpiece_id``, starting it as it arrives.

    Each workpiece resumes from the coordinator's event store. A workpiece whose run raised is ``HELD``; the error
    is logged, and a stage's error is also recorded in the workpiece's ``TERMINAL`` event.

    Returns:
        The terminal of every workpiece started, once ``proposals`` is exhausted and each has one.

    Raises:
        ValueError: ``proposals`` yielded a workpiece id a second time. The workpiece that id started
            runs to its terminal first, and no later proposal starts.
        Exception: whatever reading ``proposals`` raised, after the workpieces started run to their terminals.
    """
    runs: dict[str, asyncio.Task[Terminal]] = {}
    admission_error: Exception | None = None
    async with asyncio.TaskGroup() as group:
        try:
            async for workpiece_id, proposal in proposals:
                if workpiece_id in runs:
                    raise ValueError(f"workpiece {workpiece_id} arrived a second time")
                runs[workpiece_id] = group.create_task(_run_held(coordinator, workpiece_id, proposal), name=workpiece_id)
        except Exception as error:
            admission_error = error
    if admission_error is not None:
        raise admission_error
    return {workpiece_id: run.result() for workpiece_id, run in runs.items()}


async def _run_held(coordinator: StageCoordinator, workpiece_id: str, proposal: TaskProposal) -> Terminal:
    """Run the workpiece to a terminal; ``HELD`` on any error, which ``run_workpiece`` has recorded."""
    try:
        return await run_workpiece(coordinator, workpiece_id, proposal)
    except Exception:
        logger.warning("workpiece %s is held", workpiece_id, exc_info=True)
        return Terminal.HELD
