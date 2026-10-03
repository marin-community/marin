# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout iteration, cancellation, and model execution."""

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import replace

from shellbox.machine import Machine, MachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.models import AnswerType, StageVerifierSpec, TaskSpec
from taskcompendium.submission import FinalAction, Submission, conversation_messages

from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutData,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    TaskSession,
)
from rolloutengine.grading import _combined_stage_grade, _remove_stage_grader, validate_task_verifiers
from rolloutengine.machines import _task_machine
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
    """Generate exact-token rollouts with one isolated machine per task."""

    def __init__(
        self,
        model: Callable[[ModelRequest], Awaitable[ModelTurn]],
        factories: Mapping[EnvironmentKind, MachineFactory],
        *,
        max_turns: int,
        command_timeout: float,
        convention: Submission,
        sessions: Mapping[str, Callable[[TaskSpec], TaskSession]] | None = None,
    ):
        if max_turns < 1 or command_timeout <= 0:
            raise ValueError("Rollout limits must be positive")
        self.model = model
        self.factories = factories
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.convention = convention
        self.sessions = {} if sessions is None else sessions

    async def run(self, task: TaskSpec) -> RolloutData:
        """Run one task and release its session and machine after failure."""
        validate_task_verifiers(task)
        deadline = asyncio.timeout(task.attempt_timeout)
        async with AsyncExitStack() as resources:
            try:
                async with deadline:
                    return await self._run_task(task, resources)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                raise RolloutInterrupted(_empty_rollout(task), RolloutOperation.ATTEMPT) from error

    async def _run_task(self, task: TaskSpec, resources: AsyncExitStack) -> RolloutData:
        convention = self.convention
        if task.answer_type == AnswerType.NATIVE_ACTION:
            convention = FinalAction(id="final-action")
        try:
            machine = await resources.enter_async_context(_task_machine(task.environment, self.factories))
        except Exception as error:
            raise RolloutInterrupted(_empty_rollout(task), RolloutOperation.START) from error
        if task.stages:
            assert machine is not None
            return await self._run_stages(task, machine, convention)
        if task.environment.interaction is None:
            session = _ShellboxTaskSession(task, machine, convention, self.command_timeout, self.factories)
        else:
            session = self.sessions[task.environment.interaction](task)
        resources.push_async_callback(session.close)
        return await self._run_session(task, session)

    async def _run_stages(self, task: TaskSpec, machine: Machine, convention: Submission) -> RolloutData:
        specification = StageVerifierSpec.model_validate_json(task.verifier.parameters_json)
        record = None
        grades = []
        stage_names = []
        last_graded_step = None
        interruption = None
        for stage in task.stages:
            phase = task.model_copy(
                update={
                    "context": stage.context or task.context,
                    "verifier": stage.verifier,
                    "stages": (),
                    "agent_timeout": task.agent_timeout if stage.agent_timeout is None else stage.agent_timeout,
                    "agent_user": task.agent_user if stage.agent_user is None else stage.agent_user,
                }
            )
            initial_steps = 0 if record is None else len(record.steps)
            initial_tokens = 0 if record is None else len(record.response_token_ids)
            session = _ShellboxTaskSession(phase, machine, convention, self.command_timeout, self.factories, stage)
            try:
                record = await self._run_session(phase, session, record)
            except RolloutInterrupted as error:
                record = error.rollout
                interruption = error
            finally:
                try:
                    await session.close()
                finally:
                    await _remove_stage_grader(stage, machine)
            grade = record.grade
            grades.append(grade)
            stage_names.append(stage.name)
            steps = tuple(
                (
                    replace(step, transition=replace(step.transition, grade=grade, reward=0.0))
                    if index >= initial_steps
                    else step
                )
                for index, step in enumerate(record.steps)
            )
            record = replace(record, steps=steps)
            if grade.status == Outcome.SKIPPED and interruption is None:
                continue
            if grade.status != Outcome.GRADED:
                record = replace(
                    record,
                    loss_mask=record.loss_mask[:initial_tokens] + (0,) * (len(record.loss_mask) - initial_tokens),
                )
                break
            last_graded_step = len(record.steps) - 1
            rewards = grade.diagnostics.get("rewards", {"reward": grade.reward})
            if any(rewards.get(key, -math.inf) < minimum for key, minimum in stage.minimum_rewards.items()):
                break
            if interruption is not None:
                break
        assert record is not None
        final = _combined_stage_grade(grades, specification.strategy)
        if last_graded_step is not None and final.status == Outcome.GRADED:
            steps = list(record.steps)
            last = steps[last_graded_step]
            steps[last_graded_step] = replace(last, transition=replace(last.transition, reward=final.reward))
            record = replace(record, steps=tuple(steps))
        record = replace(
            record,
            grade=replace(
                final,
                diagnostics={
                    **final.diagnostics,
                    "stages": [
                        {
                            "name": name,
                            "status": grade.status.value,
                            "reward": grade.reward,
                            "passed": grade.passed,
                            "error": grade.error,
                            "rewards": grade.diagnostics.get("rewards", {}),
                        }
                        for name, grade in zip(stage_names, grades, strict=True)
                    ],
                },
            ),
        )
        if interruption is not None:
            raise RolloutInterrupted(record, interruption.operation) from interruption.__cause__
        return record

    async def _run_session(self, task: TaskSpec, session: TaskSession, prefix: RolloutData | None = None) -> RolloutData:
        completed = _empty_rollout(task)
        if prefix is not None:
            completed = replace(prefix, grade=completed.grade)
        try:
            start = await session.prepare()
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.PREPARE) from error
        messages = list(start.messages)
        if prefix is not None:
            messages = [*prefix.messages, *prefix.steps[-1].transition.observations, *messages]
        if prefix is None:
            completed = replace(completed, messages=tuple(messages))
        prompt: tuple[int, ...] = ()
        tokens: tuple[int, ...] = ()
        masks: tuple[int, ...] = ()
        logprobs: tuple[float, ...] | None = ()
        steps: list[RolloutStep] = []
        assistant_index = None
        if prefix is not None:
            prompt = prefix.prompt_token_ids
            tokens = prefix.prompt_token_ids + prefix.response_token_ids
            masks = prefix.loss_mask
            logprobs = prefix.logprobs
            steps = list(prefix.steps)
            assistant_index = len(prefix.messages) - 1
        initial_step_count = len(steps)
        stop_reason = MAX_TURNS_STOP_REASON
        deadline = None if task.agent_timeout is None else asyncio.get_running_loop().time() + task.agent_timeout
        for index in range(self.max_turns):
            try:
                async with asyncio.timeout_at(deadline):
                    turn = await self.model(ModelRequest(tuple(messages), start.options, tokens, assistant_index))
            except GenerationLimitReached as limit:
                stop_reason = LENGTH_STOP_REASON
                if len(steps) == initial_step_count:
                    return replace(
                        completed,
                        prompt_token_ids=prompt if prefix is not None else limit.prompt_token_ids,
                        grade=GradeResult(
                            Outcome.UNAVAILABLE, None, "Generation limit reached before the first response"
                        ),
                        stop_reason=stop_reason,
                    )
                messages = list(steps[-1].messages)
                break
            except RolloutContractError:
                raise
            except Exception as error:
                if len(steps) > initial_step_count:
                    try:
                        grade = await session.grade(completed.messages)
                    except Exception as grading_error:
                        raise RolloutInterrupted(completed, RolloutOperation.GRADE) from grading_error
                    completed = replace(completed, grade=grade)
                raise RolloutInterrupted(completed, RolloutOperation.MODEL) from error
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
                logprobs = None if turn.logprobs is None else logprobs + (0.0,) * observation_count + turn.logprobs
            tokens = turn.prompt_token_ids + turn.response_token_ids
            assistant_index = len(messages)
            messages.append(turn.message)
            try:
                async with asyncio.timeout_at(deadline):
                    transition = await session.advance(turn)
            except RolloutContractError:
                raise
            except Exception as error:
                raise RolloutInterrupted(completed, RolloutOperation.ADVANCE) from error
            for values in (transition.token_rewards, transition.token_credit):
                if values is not None and len(values) != len(turn.response_token_ids):
                    raise RolloutContractError("Transition rewards and credit must align with model response tokens")
            stop_reason = turn.stop_reason
            if transition.reset_conversation is not None and index + 1 < self.max_turns:
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
            steps.append(RolloutStep(turn, transition, len(tokens) - len(prompt) - 1, tuple(messages)))
            completed = RolloutData(
                task.id,
                tuple(messages),
                prompt,
                tokens[len(prompt) :],
                masks,
                logprobs,
                completed.grade,
                stop_reason,
                tuple(steps),
                transition.metrics,
            )
            if transition.done or stop_reason == LENGTH_STOP_REASON:
                break
            if index + 1 == self.max_turns:
                stop_reason = MAX_TURNS_STOP_REASON
                break
            messages.extend(transition.observations)
        try:
            grade = await session.grade(tuple(messages))
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.GRADE) from error
        return replace(completed, grade=grade, stop_reason=stop_reason)
