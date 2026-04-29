# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI
from openai.types.chat import ChatCompletion


@dataclass(frozen=True)
class ChatCompletionRequest:
    """OpenAI-compatible chat completion request parameters."""

    messages: tuple[dict[str, str], ...]
    num_completions: int
    temperature: float
    top_p: float = 1.0
    max_tokens: int | None = None
    seed: int | None = None
    logprobs: bool = False

    def __post_init__(self) -> None:
        if self.num_completions <= 0:
            raise ValueError("num_completions must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when set")


class CompletionProvider(Protocol):
    """Protocol for chat completion backends used by inference clients."""

    def complete_messages(self, request: ChatCompletionRequest) -> ChatCompletion:
        """Return an OpenAI-compatible chat completion response."""


class OpenAIChatCompletionProvider:
    """Minimal synchronous OpenAI-compatible completion provider."""

    def __init__(
        self,
        *,
        server_url: str,
        model: str,
        api_key: str = "marin-tts",
        timeout: float | None = None,
        extra_request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = OpenAI(base_url=server_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._extra_request_kwargs = dict(extra_request_kwargs or {})

    def complete_messages(self, request: ChatCompletionRequest) -> ChatCompletion:
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(request.messages),
            "n": request.num_completions,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "logprobs": request.logprobs,
            **self._extra_request_kwargs,
        }
        if request.max_tokens is not None:
            request_kwargs["max_tokens"] = request.max_tokens
        if request.seed is not None:
            request_kwargs["seed"] = request.seed

        return self._client.chat.completions.create(**request_kwargs)
