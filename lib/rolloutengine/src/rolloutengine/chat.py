# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A rollout model over an OpenAI-compatible chat endpoint, such as a vLLM, SGLang, or GLM server."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import openai
from openai.types.chat import ChatCompletion

from rolloutengine.models import ModelRequest, ModelTurn, TokenUsage

RESERVED_OPTIONS = frozenset({"model", "messages", "stream", "n"})
"""Request fields that the adapter sets itself. A prompt option or ``extra_body`` key with one of these names fails."""


@dataclass(frozen=True)
class ChatModel:
    """Serve each ``ModelRequest`` with one chat completion, for a ``TraceMode.MESSAGES`` rollout.

    The endpoint renders the messages itself, so a turn has no token IDs, and the token budget counts the
    ``usage`` that the endpoint reports. The prompt options go into the request: they can set ``max_tokens``
    and other sampling parameters, and their ``extra_body`` keys replace the keys of ``extra_body``. A failed
    request, or a response with no usage, is a model failure of the attempt. The client is async, so the
    session can cancel a request at its deadline. The caller owns the client and closes it.
    """

    client: openai.AsyncOpenAI
    model: str
    max_tokens: int | None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    context_window: int | None = None

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if request.continuation is not None or request.prompt_token_ids is not None:
            raise ValueError("A chat endpoint serves messages only, so the rollout must use TraceMode.MESSAGES")
        options = dict(request.options)
        extra_body = {**self.extra_body, **options.pop("extra_body", {})}
        reserved = sorted(RESERVED_OPTIONS.intersection(options.keys() | extra_body.keys()))
        if reserved:
            raise ValueError(f"The chat model sets these request fields itself: {reserved}")
        arguments: dict[str, Any] = {
            **({} if self.max_tokens is None else {"max_tokens": self.max_tokens}),
            **options,
            "model": self.model,
            "messages": list(request.messages),
            "extra_body": extra_body,
        }
        response = await self.client.chat.completions.create(stream=False, **arguments)
        assert isinstance(response, ChatCompletion)  # The adapter rejects `stream`, so the response is one completion.
        usage = response.usage
        if usage is None:
            raise ValueError("The chat endpoint reported no token usage")
        choice = response.choices[0]
        served = choice.message.model_dump(exclude_none=True)
        # The next request sends the message back, so it keeps only the fields that a chat server accepts.
        message: dict[str, Any] = {"role": "assistant", "content": served.get("content") or ""}
        if served.get("tool_calls"):
            message["tool_calls"] = served["tool_calls"]
        if served.get("reasoning_content"):
            message["reasoning_content"] = served["reasoning_content"]
        metadata: dict[str, Any] = {"response_id": response.id}
        if served.get("reasoning"):
            metadata["reasoning"] = served["reasoning"]
        if served.get("refusal"):
            metadata["refusal"] = served["refusal"]
        stop_reason = choice.finish_reason
        if (
            stop_reason == "tool_calls"
            and self.context_window is not None
            and usage.prompt_tokens + usage.completion_tokens >= self.context_window
        ):
            stop_reason = "length"
        return ModelTurn(
            message,
            prompt_token_ids=(),
            response_token_ids=(),
            logprobs=None,
            stop_reason=stop_reason,
            text=message["content"],
            metadata=metadata,
            usage=TokenUsage(usage.prompt_tokens, usage.completion_tokens),
        )
