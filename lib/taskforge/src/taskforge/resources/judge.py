# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer VerifyIt's LLM judge with recorded inference effects."""

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from dataclasses import dataclass, replace
from functools import partial

from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from verifyit.modes.grade_judge import JudgeCompletion, JudgeRequest

from taskforge.effects.context import Context
from taskforge.llm.policy import LLMPolicy, ReasoningEffort
from taskforge.model.llm import Completion, Message, Role
from taskforge.resources.inference import InferenceClient, InferenceRequest, complete
from taskforge.resources.rollouts import Engine


class _EffectFailure(BaseException):
    """Cross VerifyIt, TaskCompendium and the rollout engine without their handlers reclassifying an effect failure."""

    def __init__(self, error: Exception) -> None:
        self.error = error


@dataclass(frozen=True, slots=True)
class JudgeEngine:
    """Bind the judge role's inference effect as each judged spec's VerifyIt judge callback."""

    engine: Engine
    ctx: Context
    client: InferenceClient

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        connection = spec.judge_connection
        if connection is None:
            return await self.engine.run_task(spec, model, run_id=run_id)
        loop = asyncio.get_running_loop()
        sample = 0
        pending: Future[Completion] | None = None
        closed = False

        async def infer(request: JudgeRequest, sample: int) -> Completion:
            messages = tuple(Message(Role(message.role), message.content) for message in request.messages)
            effort = ReasoningEffort(request.reasoning_effort or ReasoningEffort.HIGH)
            sampling = LLMPolicy(request.max_output_tokens, effort, temperature=0.0, max_continuations=0)
            async with self.ctx.scope("judge"):
                self.ctx.handle(InferenceRequest, partial(complete, notify=self.ctx.notify), self.client)
                return await self.ctx.emit(InferenceRequest(messages, (), sampling, sample=sample))

        def judge(request: JudgeRequest) -> JudgeCompletion:
            # VerifyIt calls this from its grading thread; the owning loop alone executes and records effects.
            nonlocal sample, pending
            if closed:
                raise asyncio.CancelledError
            pending = asyncio.run_coroutine_threadsafe(infer(request, sample), loop)
            sample += 1
            if closed:
                pending.cancel()
            try:
                result = pending.result()
            except FutureCancelledError as error:
                raise asyncio.CancelledError from error
            except Exception as error:
                raise _EffectFailure(error) from error
            return JudgeCompletion(result.content, result.finish_reason.value, result.usage.completion_tokens)

        spec = spec.model_copy(update={"judge_connection": replace(connection, callback=judge)})
        try:
            return await self.engine.run_task(spec, model, run_id=run_id)
        except _EffectFailure as error:
            raise error.error from error.error.__cause__
        finally:
            closed = True
            if pending is not None:
                pending.cancel()
