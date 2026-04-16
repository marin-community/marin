# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AsyncOpenAI-compatible adapter built on top of Marin inference contexts."""

import asyncio
from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletion

_SUPPORTED_EXTRA_BODY_KEYS = frozenset({"top_k", "return_tokens_as_token_ids"})
_SUPPORTED_TOP_LOGPROBS = {None, 1}


def _supports_logprobs(value: object) -> bool:
    return value is None or value is True or value == 1


def _normalize_messages(messages: object) -> list[dict[str, object]]:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list of chat message dicts")

    normalized_messages: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"messages[{index}] must be a mapping")
        if not isinstance(message.get("role"), str):
            raise TypeError(f"messages[{index}] is missing a string role")
        if not isinstance(message.get("content"), str):
            raise TypeError(f"messages[{index}] is missing a string content")
        normalized_messages.append(dict(message))

    return normalized_messages


def _normalize_extra_body(extra_body: object) -> dict[str, object]:
    if extra_body is None:
        return {}
    if not isinstance(extra_body, Mapping):
        raise TypeError("extra_body must be a mapping when provided")

    unsupported_keys = sorted(set(extra_body) - _SUPPORTED_EXTRA_BODY_KEYS)
    if unsupported_keys:
        raise NotImplementedError(f"Unsupported OpenAI compatibility extra_body keys: {', '.join(unsupported_keys)}")

    return dict(extra_body)


def _extract_top_k(top_k: object, extra_body: dict[str, object]) -> int | None:
    body_top_k = extra_body.get("top_k")
    if top_k is not None and body_top_k is not None and top_k != body_top_k:
        raise ValueError("top_k and extra_body['top_k'] must match when both are provided")
    if body_top_k is not None and not isinstance(body_top_k, int):
        raise TypeError("extra_body['top_k'] must be an int")
    if top_k is not None and not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    return top_k if top_k is not None else body_top_k


def _validate_generation_kwargs(
    *,
    tools: object,
    tool_choice: object,
    logprobs: object,
    top_logprobs: object,
    timeout: object,
    kwargs: dict[str, object],
) -> None:
    if tools not in (None, []):
        raise NotImplementedError("Tool-enabled verifier environments are not supported yet")
    if tool_choice is not None:
        raise NotImplementedError("tool_choice is not supported by Marin's OpenAI compatibility adapter")
    if not _supports_logprobs(logprobs):
        raise NotImplementedError("Only logprobs=True is supported by Marin's OpenAI compatibility adapter")
    if top_logprobs not in _SUPPORTED_TOP_LOGPROBS:
        raise NotImplementedError("Only top_logprobs=1 is supported by Marin's OpenAI compatibility adapter")
    if timeout is not None and not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be numeric when provided")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise NotImplementedError(f"Unsupported OpenAI compatibility kwargs: {unsupported}")


class _CompatChatCompletions:
    def __init__(self, ctx: Any):
        self._ctx = ctx

    async def create(
        self,
        *,
        messages: object,
        model: str,
        temperature: float | None = None,
        n: int = 1,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: list[str] | None = None,
        top_k: int | None = None,
        logprobs: bool | int | None = None,
        top_logprobs: int | None = None,
        extra_body: object = None,
        timeout: float | int | None = None,
        tools: object = None,
        tool_choice: object = None,
        **kwargs: object,
    ) -> ChatCompletion:
        del model

        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")

        _validate_generation_kwargs(
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            timeout=timeout,
            kwargs=kwargs,
        )

        if max_tokens is not None and max_completion_tokens is not None and max_tokens != max_completion_tokens:
            raise ValueError("max_tokens and max_completion_tokens must match when both are provided")

        normalized_messages = _normalize_messages(messages)
        normalized_extra_body = _normalize_extra_body(extra_body)
        resolved_top_k = _extract_top_k(top_k, normalized_extra_body)
        resolved_max_tokens = max_tokens if max_tokens is not None else max_completion_tokens
        resolved_temperature = (
            temperature
            if temperature is not None
            else getattr(getattr(self._ctx, "sampling_config", None), "temperature", 1.0)
        )

        completions = await asyncio.to_thread(
            self._ctx.batch_completions,
            prompts=[normalized_messages],
            temperature=resolved_temperature,
            n=n,
            max_tokens=resolved_max_tokens,
            top_k=resolved_top_k,
            stop=stop,
            system_prompt=None,
        )

        if len(completions) != 1:
            raise ValueError(f"Expected exactly one completion for one prompt, got {len(completions)}")

        return completions[0]


class _CompatChatNamespace:
    def __init__(self, ctx: Any):
        self.completions = _CompatChatCompletions(ctx)


class _CompatCompletionsNamespace:
    async def create(self, **_kwargs: object) -> ChatCompletion:
        raise NotImplementedError("Completion-format requests are not supported by Marin's OpenAI compatibility adapter")


class OpenAICompatClient:
    """AsyncOpenAI-compatible client for verifier environments."""

    def __init__(self, ctx: Any):
        self.chat = _CompatChatNamespace(ctx)
        self.completions = _CompatCompletionsNamespace()

    async def close(self) -> None:
        return None
