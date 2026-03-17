# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified LLM client supporting both litellm (API models) and vLLM (local models).

This module provides a single `llm_chat` function that dispatches to litellm for API models
(e.g. "openai/gpt-4.1") or to a vLLM instance for local model checkpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import litellm

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


@dataclass
class LLMResponse:
    """Normalized response from any LLM backend."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


def llm_chat(
    model_id: str,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    n: int = 1,
    num_retries: int = 10,
) -> list[LLMResponse]:
    """Call an LLM via litellm.

    Args:
        model_id: Model identifier (e.g. "openai/gpt-4.1", "anthropic/claude-sonnet-4-20250514").
        messages: List of message dicts with "role" and "content" keys.
        system_prompt: Optional system prompt prepended to messages.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        n: Number of completions to generate.
        num_retries: Number of retries on failure.

    Returns:
        List of LLMResponse objects (length = n).
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    response = litellm.completion(
        model=model_id,
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        num_retries=num_retries,
    )

    results = []
    for choice in response.choices:
        results.append(
            LLMResponse(
                content=choice.message.content or "",
                model=response.model or model_id,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
            )
        )
    return results


def llm_chat_single(
    model_id: str,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    num_retries: int = 10,
) -> LLMResponse:
    """Convenience wrapper that returns a single response."""
    responses = llm_chat(
        model_id=model_id,
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        num_retries=num_retries,
    )
    return responses[0]
