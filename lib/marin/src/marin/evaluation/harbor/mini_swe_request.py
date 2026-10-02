# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for Mini-SWE context using the serving model's rendered prompt."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ContextLimits:
    max_context_tokens: int
    max_input_tokens: int
    max_output_tokens: int


class ContextBudgetExhausted(ValueError):
    """The next Mini-SWE request cannot fit within its context budget."""


def context_max_tokens(
    client: httpx.Client,
    api_base: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    request_kwargs: Mapping[str, Any],
    limits: ContextLimits,
) -> int:
    """Return the reply budget after counting the fully rendered chat prompt."""
    extra_body = request_kwargs.get("extra_body", {})
    body = {
        "model": model,
        "messages": list(messages),
        "tools": list(tools),
        "add_generation_prompt": True,
        "chat_template_kwargs": extra_body.get("chat_template_kwargs", {}),
    }
    response = client.post(api_base.rstrip("/").removesuffix("/v1") + "/tokenize", json=body)
    response.raise_for_status()
    tokenization = response.json()
    prompt_tokens = tokenization["count"]
    server_context = tokenization["max_model_len"]
    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        raise ValueError("The model tokenizer returned an invalid token count")
    if not isinstance(server_context, int) or server_context < 1:
        raise ValueError("The model tokenizer returned an invalid context window")
    context_limit = min(limits.max_context_tokens, server_context)
    if prompt_tokens > limits.max_input_tokens or prompt_tokens >= context_limit:
        raise ContextBudgetExhausted(
            f"Prompt has {prompt_tokens} tokens; input limit {limits.max_input_tokens}, context limit {context_limit}"
        )
    return min(
        request_kwargs.get("max_tokens", limits.max_output_tokens),
        limits.max_output_tokens,
        context_limit - prompt_tokens,
    )
