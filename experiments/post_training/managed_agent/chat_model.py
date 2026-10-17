# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A RolloutEngine model over an OpenAI-compatible chat endpoint, such as the GLM-5.3 service."""

from dataclasses import dataclass
from typing import Any

import openai
from rolloutengine.contracts import ModelRequest, ModelTurn, TokenUsage

CACHE_KEY_OPTION = "prompt_cache_key"
"""A request option that keeps one agent session on one server, so that each turn reuses its KV cache."""


def _assistant_message(served: dict[str, Any]) -> dict[str, Any]:
    """The served message with only the fields that a later request sends back.

    A server can add fields, such as an empty ``tool_calls`` list, that it then rejects in a request. The reasoning
    stays, so that the model keeps its thinking from turn to turn.
    """
    message: dict[str, Any] = {"role": "assistant", "content": served.get("content") or ""}
    if served.get("tool_calls"):
        message["tool_calls"] = served["tool_calls"]
    if served.get("reasoning_content"):
        message["reasoning_content"] = served["reasoning_content"]
    return message


@dataclass(frozen=True)
class ChatModel:
    """Serve each ``ModelRequest`` with one chat completion, for a ``TraceMode.MESSAGES`` rollout.

    The endpoint renders the messages itself, so the turn has no token IDs. It reports ``usage``, which the token
    budget counts. The client is async, so the recorder can cancel a request at the model-turn deadline.
    """

    client: openai.AsyncOpenAI
    model: str
    max_tokens: int
    reasoning_effort: str

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if request.continuation is not None or request.prompt_token_ids is not None:
            raise ValueError("A chat endpoint serves messages only, so the rollout must use TraceMode.MESSAGES")
        options: dict[str, Any] = dict(request.options)
        extra_body: dict[str, Any] = {"chat_template_kwargs": {"reasoning_effort": self.reasoning_effort}}
        if CACHE_KEY_OPTION in options:
            extra_body[CACHE_KEY_OPTION] = options.pop(CACHE_KEY_OPTION)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=list(request.messages),
            max_tokens=self.max_tokens,
            extra_body=extra_body,
            **options,
        )
        choice = response.choices[0]
        message = _assistant_message(choice.message.model_dump(exclude_none=True))
        usage = response.usage
        return ModelTurn(
            message,
            prompt_token_ids=(),
            response_token_ids=(),
            logprobs=None,
            stop_reason=choice.finish_reason or "stop",
            text=message.get("content") or "",
            usage=None if usage is None else TokenUsage(usage.prompt_tokens, usage.completion_tokens),
        )
