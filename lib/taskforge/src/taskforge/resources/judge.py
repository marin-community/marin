# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run VerifyIt's host judge transport through recorded inference effects."""

import asyncio
import json
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from dataclasses import dataclass, replace
from functools import partial

import httpx
from pydantic import TypeAdapter
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec

from taskforge.effects.context import Context
from taskforge.llm.policy import LLMPolicy
from taskforge.model.llm import FinishReason, Message, Role
from taskforge.resources.inference import InferenceClient, InferenceRequest, complete
from taskforge.resources.rollouts import Engine


class _EffectFailure(BaseException):
    """Cross the SDK thread without its transport retries reclassifying an effect failure."""

    def __init__(self, error: Exception) -> None:
        self.error = error


@dataclass(frozen=True)
class JudgeEngine:
    engine: Engine
    ctx: Context
    client: InferenceClient

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        connection = spec.judge_connection
        if connection is None:
            return await self.engine.run_task(spec, model, run_id=run_id)
        loop = asyncio.get_running_loop()
        sample = 0
        pending: Future[dict] | None = None
        closed = False

        async def request(payload: dict, responses: bool) -> dict:
            nonlocal sample
            wire = payload["input"] if responses else payload["messages"]
            if isinstance(wire, str):
                wire = [{"role": "user", "content": wire}]
            messages = tuple(Message(Role(message["role"]), message["content"]) for message in wire)
            sampling = TypeAdapter(LLMPolicy).validate_python(
                {
                    "max_tokens": payload["max_output_tokens" if responses else "max_completion_tokens"],
                    "temperature": payload.get("temperature", 0.0),
                    "reasoning_effort": (
                        payload.get("reasoning_effort") or payload.get("reasoning", {}).get("effort", "high")
                    ),
                    "max_continuations": 0,
                }
            )
            event = InferenceRequest(messages, (), sampling, sample=sample)
            sample += 1
            async with self.ctx.scope("judge"):
                self.ctx.resource(self.client, self.client.fingerprint)
                self.ctx.handle(InferenceRequest, partial(complete, notify=self.ctx.emit), uses=(self.client,))
                result = await self.ctx.emit(event)
            common = {"id": "forge-judge", "model": payload["model"], "created": 0}
            if responses:
                return {
                    **common,
                    "object": "response",
                    "status": "completed" if result.finish_reason is FinishReason.STOP else "incomplete",
                    "incomplete_details": (
                        None if result.finish_reason is FinishReason.STOP else {"reason": "max_output_tokens"}
                    ),
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": result.content}],
                        }
                    ],
                }
            return {
                **common,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": result.finish_reason.value,
                        "message": {"role": "assistant", "content": result.content},
                    }
                ],
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.prompt_tokens + result.usage.completion_tokens,
                },
            }

        def transport(wire: httpx.Request) -> httpx.Response:
            nonlocal pending
            if closed:
                raise asyncio.CancelledError
            # VerifyIt runs in asyncio.to_thread; the owning loop alone executes/logs effects.
            pending = asyncio.run_coroutine_threadsafe(
                request(json.loads(wire.content), wire.url.path.endswith("/responses")), loop
            )
            if closed:
                pending.cancel()
            try:
                return httpx.Response(200, json=pending.result())
            except FutureCancelledError as error:
                # Do not let the SDK retry a cancelled effect in the worker thread.
                raise asyncio.CancelledError from error
            except Exception as error:
                raise _EffectFailure(error) from error

        spec = spec.model_copy(
            update={"judge_connection": replace(connection, transport=httpx.MockTransport(transport))}
        )
        try:
            return await self.engine.run_task(spec, model, run_id=run_id)
        except _EffectFailure as error:
            raise error.error from error.error.__cause__
        finally:
            closed = True
            if pending is not None:
                pending.cancel()
