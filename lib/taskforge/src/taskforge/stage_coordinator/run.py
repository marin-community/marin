# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run many workpieces at once, each from the moment its proposal arrives, with failures kept to their workpiece.

``run_workpieces`` reads ``(workpiece_id, proposal)`` pairs from an async iterable and starts
``run_workpiece`` for each as it arrives, so a proposal still being written holds back no other
workpiece. It sets no width of its own: the stages' resource pools (build processes, model
connections, machines) bound how much runs at once.

Each workpiece runs inside a wrapper that records its terminal and never raises, so one
workpiece's failure cancels no sibling. When ``run_workpiece`` raises a transient infrastructure
failure (``transient_infrastructure_failure``), the wrapper waits out a backoff and calls it again
with the same log, which resumes the workpiece at the stage that raised; every attempt leaves its
``TERMINAL`` failed event in the workpiece's log. Once ``InfrastructureRetry.max_attempts`` attempts
have failed, or on any other error, the workpiece stays failed in place, as one ``run_workpiece``
call leaves it, and the run goes on.

``run_workpieces`` returns when the iterable is exhausted and every workpiece it started has a
terminal. An error while reading the iterable, or a workpiece id the iterable yields a second time,
stops admission: the workpieces already started run to their terminals, and ``run_workpieces``
then raises that error.
"""

import asyncio
import logging
from collections.abc import AsyncIterable
from dataclasses import dataclass

from rigging.timing import ExponentialBackoff

from taskforge.llm.client import ModelUnavailable
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage40_build_with_program.context import BuildInfrastructureFailure
from taskforge.stage40_build_with_program.infrastructure import HOST_REJECTIONS
from taskforge.stage_coordinator.coordinator import StageCoordinator, run_workpiece
from taskforge.stage_coordinator.event_log import InMemoryEventLog
from taskforge.stage_coordinator.workpiece_machine import Terminal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InfrastructureRetry:
    """How often, and how far apart, ``run_workpieces`` runs a workpiece that a transient infrastructure failure stopped.

    Attributes:
        max_attempts: The ``run_workpiece`` calls a workpiece may take, the first included. One never retries.
        backoff_initial: The wait, in seconds, before the first retry.
        backoff_maximum: The longest wait, in seconds, between two attempts.
        backoff_factor: The factor each wait grows by over the one before.
    """

    max_attempts: int
    backoff_initial: float
    backoff_maximum: float
    backoff_factor: float


def transient_infrastructure_failure(error: BaseException) -> bool:
    """Whether ``error`` is a host or model failure that a later attempt on the same host may not meet.

    These are the failures the coordinator already ends as the host's or the model's, never the
    task's or the program's:

    * ``ModelUnavailable``, the model unreachable past its client's retries and hold, from any stage;
    * ``BuildInfrastructureFailure``, the build host failing, from the build or an agentic author,
      except the causes in ``infrastructure.HOST_REJECTIONS``, which hold for as long as the host is
      unchanged.

    ``BuildExpired`` is not one: whether Taskforge or the program was slow is unknown until someone
    reads the record. A storage fault (an ``OSError`` outside the program's frames) is the step
    store's, and a missing or corrupt blob or memo stays so on every attempt.
    """
    if isinstance(error, ModelUnavailable):
        return True
    return isinstance(error, BuildInfrastructureFailure) and error.cause not in HOST_REJECTIONS


async def run_workpieces(
    coordinator: StageCoordinator,
    proposals: AsyncIterable[tuple[str, TaskProposal]],
    retry: InfrastructureRetry,
) -> dict[str, Terminal]:
    """Run each proposal ``proposals`` yields as workpiece ``workpiece_id``, starting it as it arrives.

    Each workpiece resumes from the coordinator's event store, as ``run_workpiece`` does with an
    empty log. A workpiece whose run raised is ``FAILED``; the error is logged, and a stage's error
    is also recorded in the workpiece's ``TERMINAL`` event.

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
                runs[workpiece_id] = group.create_task(
                    _run_retrying(coordinator, workpiece_id, proposal, retry), name=workpiece_id
                )
        except Exception as error:
            admission_error = error
    if admission_error is not None:
        raise admission_error
    return {workpiece_id: run.result() for workpiece_id, run in runs.items()}


async def _run_retrying(
    coordinator: StageCoordinator, workpiece_id: str, proposal: TaskProposal, retry: InfrastructureRetry
) -> Terminal:
    """Run the workpiece to a terminal, retrying transient infrastructure failures; ``FAILED`` on any other error."""
    log = InMemoryEventLog()
    backoff = ExponentialBackoff(
        initial=retry.backoff_initial, maximum=retry.backoff_maximum, factor=retry.backoff_factor
    )
    attempt = 1
    while True:
        try:
            return await run_workpiece(coordinator, workpiece_id, proposal, log)
        except Exception as error:
            if not transient_infrastructure_failure(error) or attempt >= retry.max_attempts:
                logger.warning("workpiece %s failed on attempt %d", workpiece_id, attempt, exc_info=True)
                return Terminal.FAILED
            delay = backoff.next_interval()
            logger.warning(
                "workpiece %s attempt %d/%d failed (%s: %s); retrying in %.1fs",
                workpiece_id,
                attempt,
                retry.max_attempts,
                type(error).__name__,
                error,
                delay,
            )
        await asyncio.sleep(delay)
        attempt += 1
