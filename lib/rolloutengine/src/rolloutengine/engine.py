# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout iteration, cancellation, and model execution."""

import asyncio
import math
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, replace
from functools import partial

from shellbox.machine import Machine, MachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.execution import StageExecution, TaskExecution
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import StageVerifierSpec, TaskSpec, TaskStage
from taskcompendium.submission import SubmissionConvention, conversation_messages

from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import (
    AGENT_TIMEOUT_STOP_REASON,
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
    SuppliedState,
    TaskSession,
    Transition,
)
from rolloutengine.grading import _combined_stage_grade, _remove_stage_grader, _validate_task
from rolloutengine.machines import _install_files, _run_setup_commands, _task_machine
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


def _phase(task: TaskSpec, execution: TaskExecution, stage: TaskStage | None) -> tuple[TaskSpec, StageExecution]:
    """The task view and preparation settings for one stage, or for an unstaged task."""
    if stage is None:
        return task, StageExecution(agent_user=execution.agent_user)
    phase = task.model_copy(update={"context": stage.context or task.context, "verifier": stage.verifier, "stages": ()})
    stage_execution = execution.stages[stage.name]
    stage_execution = stage_execution.model_copy(
        update={
            "agent_timeout": stage_execution.agent_timeout or execution.agent_timeout,
            "agent_user": execution.agent_user if stage_execution.agent_user is None else stage_execution.agent_user,
        }
    )
    return phase, stage_execution


class ShellboxRolloutEngine:
    """Generate exact-token rollouts with one isolated machine per task."""

    def __init__(
        self,
        model: Callable[[ModelRequest], Awaitable[ModelTurn]],
        factories: Mapping[EnvironmentKind, MachineFactory],
        *,
        max_turns: int,
        command_timeout: float,
        cleanup_timeout: float,
        convention: SubmissionConvention,
        sessions: Mapping[str, Callable[[TaskSpec, Machine | None], TaskSession]] | None = None,
    ):
        if max_turns < 1 or command_timeout <= 0 or cleanup_timeout <= 0:
            raise ValueError("Rollout limits must be positive")
        self.model = model
        self.factories = factories
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.cleanup_timeout = cleanup_timeout
        self.convention = convention
        self.sessions = {} if sessions is None else sessions

    async def run(self, task: TaskSpec, *, execution: TaskExecution) -> RolloutData:
        """Run one task with bounded session and machine cleanup."""
        _validate_task(task, execution)
        return await self._attempt(task, execution, partial(self._run_task, task, execution))

    async def grade_state(
        self, task: TaskSpec, state: SuppliedState, *, execution: TaskExecution, stage: str | None = None
    ) -> GradeResult:
        """Grade a supplied final state with the task's verifier, without model inference.

        The engine creates and prepares the task machine as `run` does, installs `state.files`,
        runs `state.commands`, then grades `state.messages` through the rollout's grading path.
        For a staged task, `stage` names the graded stage: the preparation of that stage and of
        every earlier stage runs first, and the result is that stage's grade, not the aggregate.

        Raises:
            ValueError: The task needs an interaction session, `stage` does not name one of the
                task's stages, or a null environment receives files or commands.
            RolloutInterrupted: Execution failed, as in `run`, with an empty rollout record. The
                `state` operation means a supplied file or command failed.
        """
        _validate_task(task, execution)
        if task.environment.interaction is not None:
            raise ValueError("Supplied-state grading requires the Shellbox task session")
        names = [candidate.name for candidate in task.stages]
        if (stage is None) != (not names) or (stage is not None and stage not in names):
            raise ValueError(f"Stage {stage!r} does not select one of the task stages {names}")
        if task.environment.kind == EnvironmentKind.NULL and (state.files or state.commands):
            raise ValueError("A null environment cannot receive state files or commands")
        stages = () if stage is None else task.stages[: names.index(stage) + 1]
        record = await self._attempt(task, execution, partial(self._grade_state, task, execution, state, stages))
        return record.grade

    async def _attempt(
        self,
        task: TaskSpec,
        execution: TaskExecution,
        body: Callable[[AsyncExitStack, _Cleanup], Coroutine[None, None, RolloutData]],
    ) -> RolloutData:
        """Run `body` under the attempt deadline, release its resources, and report cleanup errors."""
        deadline = asyncio.timeout(execution.attempt_timeout)
        cleanup = _Cleanup(self.cleanup_timeout)
        operation = None
        cause = None
        async with AsyncExitStack() as resources:
            try:
                async with deadline:
                    record = await body(resources, cleanup)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                operation, cause = RolloutOperation.ATTEMPT, error
                record = _empty_rollout(task)
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

    async def _start_machine(self, task: TaskSpec, resources: AsyncExitStack, cleanup: _Cleanup) -> Machine | None:
        try:
            return await resources.enter_async_context(_task_machine(task.environment, self.factories, cleanup))
        except Exception as error:
            raise RolloutInterrupted(_empty_rollout(task), RolloutOperation.START) from error

    def _shellbox_session(
        self, phase: TaskSpec, execution: StageExecution, machine: Machine | None, cleanup: _Cleanup
    ) -> _ShellboxTaskSession:
        return _ShellboxTaskSession(
            phase, machine, self.convention, self.command_timeout, self.factories, cleanup, execution
        )

    async def _grade_state(
        self,
        task: TaskSpec,
        execution: TaskExecution,
        state: SuppliedState,
        stages: tuple[TaskStage, ...],
        resources: AsyncExitStack,
        cleanup: _Cleanup,
    ) -> RolloutData:
        empty = _empty_rollout(task)
        machine = await self._start_machine(task, resources, cleanup)
        phases = [_phase(task, execution, stage) for stage in stages] or [_phase(task, execution, None)]
        for phase, stage_execution in phases:
            session = self._shellbox_session(phase, stage_execution, machine, cleanup)
            resources.push_async_callback(cleanup.run, "session_close", session.close)
            try:
                await session.prepare()
            except Exception as error:
                raise RolloutInterrupted(empty, RolloutOperation.PREPARE) from error
        if machine is not None:
            try:
                await _install_files(machine, state.files)
                await _run_setup_commands(machine, state.commands, "Supplied state command")
            except Exception as error:
                raise RolloutInterrupted(empty, RolloutOperation.STATE) from error
        try:
            grade = await session.grade(state.messages)
        except Exception as error:
            raise RolloutInterrupted(empty, RolloutOperation.GRADE) from error
        return replace(empty, messages=state.messages, grade=grade)

    async def _run_task(
        self, task: TaskSpec, execution: TaskExecution, resources: AsyncExitStack, cleanup: _Cleanup
    ) -> RolloutData:
        machine = await self._start_machine(task, resources, cleanup)
        if task.stages:
            assert machine is not None
            return await self._run_stages(task, execution, machine, cleanup)
        if task.environment.interaction is None:
            phase, stage_execution = _phase(task, execution, None)
            session = self._shellbox_session(phase, stage_execution, machine, cleanup)
        else:
            session = self.sessions[task.environment.interaction](task, machine)
        resources.push_async_callback(cleanup.run, "session_close", session.close)
        return await self._run_session(task, session, agent_timeout=execution.agent_timeout)

    async def _run_stages(
        self,
        task: TaskSpec,
        execution: TaskExecution,
        machine: Machine,
        cleanup: _Cleanup,
    ) -> RolloutData:
        specification = StageVerifierSpec.model_validate_json(task.verifier.parameters_json)
        record: RolloutData | None = None
        grades = []
        stage_names = []
        last_graded_step = None
        operation = None
        cause = None
        for stage_index, stage in enumerate(task.stages):
            phase, stage_execution = _phase(task, execution, stage)
            initial_steps = 0 if record is None else len(record.steps)
            initial_tokens = 0 if record is None else len(record.response_token_ids)
            session = self._shellbox_session(phase, stage_execution, machine, cleanup)
            try:
                record = await self._run_session(phase, session, record, agent_timeout=stage_execution.agent_timeout)
            except RolloutInterrupted as error:
                record = error.rollout
                operation, cause = error.operation, error.__cause__
            finally:
                try:
                    await cleanup.run("session_close", session.close)
                finally:
                    removal_error = await cleanup.run(
                        "stage_grader_remove", partial(_remove_stage_grader, stage, machine)
                    )
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
            if grade.status != Outcome.SKIPPED:
                if grade.status != Outcome.GRADED:
                    record = replace(
                        record,
                        loss_mask=record.loss_mask[:initial_tokens] + (0,) * (len(record.loss_mask) - initial_tokens),
                    )
                    break
                last_graded_step = len(record.steps) - 1
                assert grade.reward is not None
                rewards = grade.rewards or {"reward": grade.reward}
                if any(rewards.get(key, -math.inf) < minimum for key, minimum in stage.minimum_rewards.items()):
                    break
            if operation is not None:
                break
            if record.stop_reason == AGENT_TIMEOUT_STOP_REASON:
                break
            if removal_error is not None and stage_index + 1 < len(task.stages):
                operation, cause = RolloutOperation.CLEANUP, removal_error
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
                            "rewards": grade.rewards,
                        }
                        for name, grade in zip(stage_names, grades, strict=True)
                    ],
                },
            ),
        )
        if operation is not None:
            raise RolloutInterrupted(record, operation) from cause
        return record

    async def _run_session(
        self, task: TaskSpec, session: TaskSession, prefix: RolloutData | None = None, *, agent_timeout: float | None
    ) -> RolloutData:
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
        model_error = None
        pending_turn = None
        deadline = asyncio.timeout(agent_timeout)
        try:
            async with deadline:
                for index in range(self.max_turns):
                    try:
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
                    pending_turn = turn
                    try:
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
                        pending_turn = None
                        continue
                    steps.append(RolloutStep(turn, transition, len(tokens) - len(prompt) - 1, tuple(messages)))
                    pending_turn = None
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
        except TimeoutError:
            if not deadline.expired():
                raise
            stop_reason = AGENT_TIMEOUT_STOP_REASON
            if pending_turn is not None:
                pending_step = RolloutStep(
                    pending_turn,
                    Transition(done=True, metrics={"advance_incomplete": 1.0}),
                    len(tokens) - len(prompt) - 1,
                    tuple(messages),
                )
                steps.append(pending_step)
                completed = replace(
                    completed,
                    messages=tuple(messages),
                    prompt_token_ids=prompt,
                    response_token_ids=tokens[len(prompt) :],
                    loss_mask=masks,
                    logprobs=logprobs,
                    steps=tuple(steps),
                    metrics=pending_step.transition.metrics,
                )
            if len(steps) == initial_step_count:
                return replace(
                    completed,
                    grade=GradeResult(Outcome.UNAVAILABLE, None, "Agent deadline expired before the first response"),
                    stop_reason=stop_reason,
                )
            messages = list(completed.messages)
        if model_error is not None:
            if len(steps) == initial_step_count:
                raise RolloutInterrupted(completed, RolloutOperation.MODEL) from model_error
            messages = list(completed.messages)
        try:
            grade = await session.grade(tuple(messages))
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.GRADE) from error
        completed = replace(completed, grade=grade, stop_reason=stop_reason)
        if model_error is not None:
            raise RolloutInterrupted(completed, RolloutOperation.MODEL) from model_error
        return completed
