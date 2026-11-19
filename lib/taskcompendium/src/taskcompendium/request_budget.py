# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint request limits and exact, model-specific prompt budget evidence."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class RequestBudget:
    context_tokens: int
    input_tokens: int
    output_tokens: int

    def output_capacity(self, prompt_tokens: int) -> int:
        """Return the largest output allowed by the prompt and endpoint limits."""
        if min(self.context_tokens, self.input_tokens, self.output_tokens) <= 0:
            raise ValueError("Native endpoint limits must be positive")
        if not 0 < prompt_tokens <= self.input_tokens:
            raise ValueError("Exact prompt token count exceeds the supplied input budget")
        capacity = min(self.output_tokens, self.context_tokens - prompt_tokens)
        if capacity <= 0:
            raise ValueError("Exact prompt leaves no output capacity")
        return capacity


class TokenizationIdentity(BaseModel):
    """The immutable tokenizer, template and template arguments used for counting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokenizer: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_arguments: dict[str, Any]


class PromptBudget(BaseModel):
    """Supplied evidence for one unchanged system/user prompt on one backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    backend_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenization: TokenizationIdentity
    endpoint_limit_source: str = Field(min_length=1)
    limits: RequestBudget
    prompt_tokens: int = Field(gt=0)

    def require_match(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        endpoint: str,
        backend_identity_sha256: str,
        tokenization: TokenizationIdentity,
        prompt_tokens: int,
    ) -> int:
        """Validate the actual request identity and count before any inference."""
        actual = (
            tool_free_prompt_sha256(messages),
            model,
            endpoint.rstrip("/"),
            backend_identity_sha256,
            tokenization,
            prompt_tokens,
        )
        expected = (
            self.prompt_sha256,
            self.model,
            self.endpoint.rstrip("/"),
            self.backend_identity_sha256,
            self.tokenization,
            self.prompt_tokens,
        )
        if actual != expected:
            raise ValueError("Supplied prompt budget does not match the actual prompt, tokenizer or backend")
        if self.limits.input_tokens != self.limits.context_tokens - 1:
            raise ValueError("Tool-free prompt budgets must retain the full native input window")
        return self.limits.output_capacity(prompt_tokens)


def tool_free_prompt_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    """Identify the source's exact two-message, text-only completion envelope."""
    if len(messages) != 2 or [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("Tool-free completion requires exactly one system and one user message")
    if any(set(message) != {"role", "content"} or not isinstance(message["content"], str) for message in messages):
        raise ValueError("Tool-free completion permits only the source's text content")
    return hashlib.sha256((messages[0]["content"] + "\n" + messages[1]["content"]).encode()).hexdigest()
