# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout iteration, cancellation, and model execution."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, replace

from shellbox.machine import Machine, MachineFactory
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.submission import Submission, conversation_messages

from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutData,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    TaskSession,
    Transition,
)
from rolloutengine.lowering import SHELLBOX_SESSION, validate_lowered_task
from rolloutengine.machines import _prepare_machine
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.task_session import _ShellboxTaskSession


def _empty_rollout(task: TaskSpec) -> RolloutData:
    return RolloutData(
        task.id,
        tuple(conversation_messages(task.context)),
        (),
        (),
        (),
        (),
        GradeResult(Outcome.UNAVAILABLE, None, "Execution has no final grade"),
        "error",
    )


class ShellboxRolloutEngine:
    """Generate exact-token rollouts from lowered single-stage tasks."""

    def __init__(
        self,
        model: Callable[[ModelRequest], Awaitable[ModelTurn]],
        factories: Mapping[str, MachineFactory],
        *,
        convention: Submission,
        sessions: Mapping[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] | None = None,
    ):
        self.model = model
        self.factories = factories
        self.convention = convention
        self.sessions = {} if sessions is None else sessions

    async def run(self, lowered: LoweredTaskSpec) -> RolloutData:
        """Run one attempt; release its resources outside the attempt deadline."""
        validate_lowered_task(lowered, factories=self.factories, sessions=self.sessions)
        deadline = asyncio.timeout(lowered.session.attempt_timeout)
        cleanup = _Cleanup(lowered.session.cleanup_timeout)
        operation = None
        cause = None
        async with AsyncExitStack() as resources:
            try:
                async with deadline:
                    record = await self._run_task(lowered, resources, cleanup, deadline)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                operation, cause = RolloutOperation.ATTEMPT, error
                record = _empty_rollout(lowered.task)
            except RolloutInterrupted as error:
                operation, cause = error.operation, error.__cause__
                record = error.rollout
        if cleanup.errors:
            record = replace(
                record,
                grade=replace(
                    record.grade,
                    diagnostics={
                        **record.grade.diagnostics,
                        "cleanup_errors": [asdict(error) for error in cleanup.errors],
                    },
                ),
                metrics={**record.metrics, "cleanup_error_count": float(len(cleanup.errors))},
            )
        if operation is not None:
            raise RolloutInterrupted(record, operation) from cause
        return record

    async def _run_task(
        self, lowered: LoweredTaskSpec, resources: AsyncExitStack, cleanup: _Cleanup, attempt: asyncio.Timeout
    ) -> RolloutData:
        task = lowered.task
        try:
            machine = await _prepare_machine(
                task.environment_requirements,
                lowered.runtime.task_machine,
                task.resources.all + task.resources.worker,
                self.factories,
                cleanup,
                resources,
            )
        except Exception as error:
            raise RolloutInterrupted(_empty_rollout(task), RolloutOperation.START) from error
        try:
            if lowered.session.task_session == SHELLBOX_SESSION:
                session = _ShellboxTaskSession(lowered, machine, self.convention, self.factories, cleanup, resources)
            else:
                session = self.sessions[lowered.session.task_session](lowered, machine)
        except Exception as error:
            raise RolloutInterrupted(_empty_rollout(task), RolloutOperation.PREPARE) from error
        resources.push_async_callback(cleanup.run, "session_close", session.close)
        return await self._run_session(lowered, session, attempt)

    async def _run_session(
        self, lowered: LoweredTaskSpec, session: TaskSession, attempt: asyncio.Timeout
    ) -> RolloutData:
        task = lowered.task
        limits = lowered.session
        completed = _empty_rollout(task)
        owner = asyncio.current_task()
        assert owner is not None
        cancellation_count = owner.cancelling()
        try:
            start = await session.prepare()
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.PREPARE) from error
        messages = list(start.messages)
        completed = replace(completed, messages=tuple(messages))
        prompt: tuple[int, ...] = ()
        tokens: tuple[int, ...] = ()
        masks: tuple[int, ...] = ()
        logprobs: tuple[float, ...] | None = ()
        steps: list[RolloutStep] = []
        assistant_index = None
        stop_reason = MAX_TURNS_STOP_REASON
        model_error = None
        deadline = asyncio.timeout(limits.total_turn_timeout)
        try:
            async with deadline:
                for index in range(limits.max_turns):
                    try:
                        async with asyncio.timeout(limits.model_turn_timeout):
                            turn = await self.model(
                                ModelRequest(tuple(messages), start.options, tokens, assistant_index)
                            )
                    except GenerationLimitReached as limit:
                        stop_reason = LENGTH_STOP_REASON
                        if not steps:
                            return replace(
                                completed,
                                prompt_token_ids=limit.prompt_token_ids,
                                grade=GradeResult(
                                    Outcome.UNAVAILABLE, None, "Generation limit reached before the first response"
                                ),
                                stop_reason=stop_reason,
                            )
                        messages = list(completed.messages)
                        break
                    except RolloutContractError:
                        raise
                    except Exception as error:
                        model_error = error
                        break
                    if turn.logprobs is not None and len(turn.logprobs) != len(turn.response_token_ids):
                        raise RolloutContractError("Model logprobs must align with response tokens")
                    if not turn.response_token_ids:
                        raise RolloutContractError("Model response contains no token evidence")
                    if assistant_index is None:
                        prompt = turn.prompt_token_ids
                        tokens = prompt
                    if turn.prompt_token_ids[: len(tokens)] != tokens:
                        raise RolloutContractError("Model transport changed the served token prefix")
                    observation_count = len(turn.prompt_token_ids) - len(tokens)
                    masks += (0,) * observation_count + (1,) * len(turn.response_token_ids)
                    if logprobs is not None:
                        logprobs = (
                            None if turn.logprobs is None else logprobs + (0.0,) * observation_count + turn.logprobs
                        )
                    tokens = turn.prompt_token_ids + turn.response_token_ids
                    assistant_index = len(messages)
                    messages.append(turn.message)
                    pending = RolloutStep(
                        turn,
                        Transition(done=True, metrics={"advance_incomplete": 1.0}),
                        len(tokens) - len(prompt) - 1,
                        tuple(messages),
                    )
                    # Token evidence belongs to the attempt even if a tool operation fails.
                    completed = RolloutData(
                        task.id,
                        tuple(messages),
                        prompt,
                        tokens[len(prompt) :],
                        masks,
                        logprobs,
                        completed.grade,
                        turn.stop_reason,
                        (*steps, pending),
                        pending.transition.metrics,
                    )
                    try:
                        async with asyncio.timeout(limits.tool_turn_timeout):
                            transition = await session.advance(turn)
                    except RolloutContractError:
                        raise
                    except Exception as error:
                        raise RolloutInterrupted(completed, RolloutOperation.ADVANCE) from error
                    for values in (transition.token_rewards, transition.token_credit):
                        if values is not None and len(values) != len(turn.response_token_ids):
                            raise RolloutContractError(
                                "Transition rewards and credit must align with model response tokens"
                            )
                    stop_reason = turn.stop_reason
                    if transition.reset_conversation is not None and index + 1 < limits.max_turns:
                        messages = list(transition.reset_conversation)
                        prompt = tokens = masks = ()
                        logprobs = ()
                        steps.clear()
                        completed = replace(
                            completed,
                            messages=tuple(messages),
                            prompt_token_ids=(),
                            response_token_ids=(),
                            loss_mask=(),
                            logprobs=(),
                            steps=(),
                            metrics={},
                        )
                        assistant_index = None
                        continue
                    steps.append(replace(pending, transition=transition))
                    completed = replace(
                        completed, steps=tuple(steps), stop_reason=stop_reason, metrics=transition.metrics
                    )
                    if transition.done or stop_reason == LENGTH_STOP_REASON:
                        break
                    if index + 1 == limits.max_turns:
                        stop_reason = MAX_TURNS_STOP_REASON
                        break
                    messages.extend(transition.observations)
        except TimeoutError:
            if not deadline.expired():
                raise
            stop_reason = TOTAL_TURN_TIMEOUT_STOP_REASON
            if not completed.steps:
                return replace(
                    completed,
                    grade=GradeResult(Outcome.UNAVAILABLE, None, "Turn deadline expired before the first response"),
                    stop_reason=stop_reason,
                )
            messages = list(completed.messages)
        except asyncio.CancelledError:
            if not attempt.expired() or owner.cancelling() > cancellation_count + 1:
                raise
            raise RolloutInterrupted(completed, RolloutOperation.ATTEMPT) from TimeoutError("Attempt deadline expired")
        if model_error is not None:
            if not completed.steps:
                raise RolloutInterrupted(completed, RolloutOperation.MODEL) from model_error
            messages = list(completed.messages)
        try:
            async with asyncio.timeout(limits.verifier_timeout):
                grade = await session.grade(tuple(messages))
        except asyncio.CancelledError:
            if not attempt.expired() or owner.cancelling() > cancellation_count + 1:
                raise
            raise RolloutInterrupted(completed, RolloutOperation.ATTEMPT) from TimeoutError("Attempt deadline expired")
        except TimeoutError as error:
            completed = replace(
                completed,
                grade=GradeResult(
                    Outcome.INFRA_ERROR, None, "Verifier deadline expired", failure=GradingFailure.TIMEOUT
                ),
            )
            raise RolloutInterrupted(completed, RolloutOperation.GRADE) from error
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.GRADE) from error
        completed = replace(completed, grade=grade, stop_reason=stop_reason)
        if model_error is not None:
            raise RolloutInterrupted(completed, RolloutOperation.MODEL) from model_error
        return completed
