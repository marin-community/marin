# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer VerifyIt's LLM judge with recorded inference effects."""

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from dataclasses import dataclass, replace

from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from verifyit.modes.grade_judge import JudgeCompletion, JudgeRequest

from taskforge.effects.context import Context
from taskforge.effects.issues import ContextExceeded
from taskforge.llm.policy import LLMPolicy, ReasoningEffort
from taskforge.model.llm import Completion, FinishReason, Message, Role
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
        # The owning loop alone reads and writes these, so a sample either registers before closure or never starts.
        closed = False
        active: asyncio.Task[Completion] | None = None

        async def execute(event: InferenceRequest, client: InferenceClient) -> Completion:
            completion = await complete(event, client, notify=self.ctx.notify)
            if closed:  # A completion a provider returns after the grade closed is discarded, never resolved.
                raise asyncio.CancelledError
            return completion

        async def infer(request: JudgeRequest, sample: int) -> Completion:
            nonlocal active
            if closed:
                raise asyncio.CancelledError
            active = asyncio.current_task()
            messages = tuple(Message(Role(message.role), message.content) for message in request.messages)
            effort = ReasoningEffort(request.reasoning_effort or ReasoningEffort.HIGH)
            sampling = LLMPolicy(request.max_output_tokens, effort, temperature=0.0, max_continuations=0)
            async with self.ctx.scope("judge"):
                self.ctx.handle(InferenceRequest, execute, self.client)
                result = await self.ctx.emit(InferenceRequest(messages, (), sampling, sample=sample))
            if result.finish_reason is FinishReason.CONTEXT_EXCEEDED:
                raise ContextExceeded(result.content)
            return result

        def judge(request: JudgeRequest) -> JudgeCompletion:
            # VerifyIt calls this from its grading thread; the owning loop alone executes and records effects.
            nonlocal sample
            pending = asyncio.run_coroutine_threadsafe(infer(request, sample), loop)
            sample += 1
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
            # A cancelled or expired grade may leave its grading thread unwinding; it starts and records no sample.
            closed = True
            if active is not None and not active.done():
                active.cancel()
                await asyncio.wait({active})
