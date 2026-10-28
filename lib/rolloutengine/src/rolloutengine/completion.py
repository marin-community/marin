# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A raw-text transport for source benchmarks that do not use chat templates."""

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from openai.types.completion import Completion

from rolloutengine.models import ModelRequest, ModelResponseError, ModelTurn, TokenUsage

RESERVED_OPTIONS = frozenset({"model", "prompt", "stream", "n"})


@dataclass(frozen=True)
class CompletionModel:
    client: AsyncOpenAI
    model: str

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if request.continuation is not None:
            raise ValueError("Raw completions do not support a continued rollout")
        (message,) = request.messages
        options: dict[str, Any] = dict(request.options)
        if RESERVED_OPTIONS & (options.keys() | options.get("extra_body", {}).keys()):
            raise ValueError("The completion transport owns model, prompt, stream and n")
        response = await self.client.completions.create(
            model=self.model,
            prompt=list(request.prompt_token_ids) if request.prompt_token_ids is not None else message["content"],
            stream=False,
            **options,
        )
        assert isinstance(response, Completion)
        if len(response.choices) != 1:
            raise ModelResponseError(
                "The completion endpoint must return one choice", response=response.model_dump(mode="json")
            )
        (choice,) = response.choices
        if response.usage is None:
            raise ModelResponseError(
                "The completion endpoint reported no token usage", response=response.model_dump(mode="json")
            )
        return ModelTurn(
            message={"role": "assistant", "content": choice.text},
            prompt_token_ids=(),
            response_token_ids=(),
            logprobs=None,
            stop_reason=choice.finish_reason,
            text=choice.text,
            metadata={"response_id": response.id},
            usage=TokenUsage(response.usage.prompt_tokens, response.usage.completion_tokens),
        )
