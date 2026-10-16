# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run many independent attempts through one engine with one model: a managed agent."""

import asyncio
import itertools
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    Grader,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)

from rolloutengine.contracts import RolloutContractError, RolloutData, RolloutInterrupted
from rolloutengine.engine import Model, RolloutEngine
from rolloutengine.spec import RolloutSpec

UNGRADED_REASON = "The agent task declares no grader"


@dataclass(frozen=True)
class AttemptOutcome:
    """The result of one spec: its run key, its record, and the error that interrupted or prevented it."""

    spec: RolloutSpec
    run_key: str
    rollout: RolloutData | None
    error: Exception | None = None


@dataclass(frozen=True)
class ManagedAgent:
    """Run independent attempts with one model, at most ``concurrency`` at a time.

    The engine holds the machine factories and the harnesses; each spec selects its harness, and
    ``model`` serves every attempt. One failed attempt never stops the others, but a
    ``RolloutContractError`` from a broken transport or harness stops the run.
    """

    engine: RolloutEngine
    model: Model
    concurrency: int

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"A managed agent needs a concurrency of at least 1, not {self.concurrency}")

    async def run(self, specs: Iterable[RolloutSpec]) -> AsyncIterator[AttemptOutcome]:
        """Yield one outcome per spec as attempts finish.

        Each attempt gets the run key ``<call id>/<task id>/<position in specs>``, so repeated tasks and
        concurrent calls stay apart.
        A new attempt starts after the caller takes an outcome, so a long stream is read as attempts finish
        and a slow caller holds back new attempts. Closing the iterator (``contextlib.aclosing``) cancels
        the running attempts and waits for their cleanup.
        """
        call_id = uuid.uuid4().hex[:8]
        pending = enumerate(specs)
        running: set[asyncio.Task[AttemptOutcome]] = set()

        def start_next() -> None:
            for index, spec in itertools.islice(pending, 1):
                running.add(asyncio.create_task(self._attempt(spec, f"{call_id}/{spec.task.id}/{index}")))

        try:
            for _ in range(self.concurrency):
                start_next()
            while running:
                finished, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in finished:
                    yield task.result()
                    start_next()
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    async def _attempt(self, spec: RolloutSpec, run_key: str) -> AttemptOutcome:
        try:
            return AttemptOutcome(spec, run_key, await self.engine.run(spec, self.model, run_key=run_key))
        except RolloutInterrupted as error:
            return AttemptOutcome(spec, run_key, error.rollout, error)
        except RolloutContractError:
            # A transport or harness that breaks the token contract breaks every attempt, so the run stops.
            raise
        except Exception as error:
            # A rejected spec ends this attempt only.
            return AttemptOutcome(spec, run_key, None, error)


def agent_task(
    task_id: str,
    instruction: str,
    *,
    source: Source,
    environment: EnvironmentRequirements = EnvironmentRequirements(),
    resources: ResourceGroups = ResourceGroups(),
    output_paths: tuple[str, ...] = (),
    grader: Grader | None = None,
) -> TaskSpec:
    """A task for free-form agentic work: one user instruction and a plain-text final answer.

    Without a grader, the task uses ``NoGrader`` and its attempts record ``Outcome.UNAVAILABLE``.
    """
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=environment,
        output_paths=output_paths,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason=UNGRADED_REASON) if grader is None else grader,
        source=source,
        resources=resources,
    )
