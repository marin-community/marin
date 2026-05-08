# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight OpenAI-compatible protocol models shared by servers and tests."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message in the conversation."""

    role: Literal["system", "user", "assistant", "tool", "function", "developer"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, object] | None = None


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions endpoint."""

    model: str
    messages: list[ChatMessage]
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool = Field(default=False, description="Whether to include logprobs in the response")
    top_logprobs: int | None = None
    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, object] | None = None
    seed: int | None = None
    service_tier: str | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, object] | None = None
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: float | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: str | dict[str, object] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None


class CompletionRequest(BaseModel):
    """Request model for text completions endpoint."""

    model: str
    prompt: str | list[str]
    best_of: int | None = None
    echo: bool | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: int | None = None
    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    n: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, object] | None = None
    suffix: str | None = None
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: float | None = None
    user: str | None = None


class TokensRequest(BaseModel):
    """Request tokens from the given prompts after system prompt injection and encoding."""

    model: str = "marin-default"
    message_list: list[list[ChatMessage]]


class TokenList(BaseModel):
    """List of token IDs."""

    tokens: list[int]


class TokensResponse(BaseModel):
    """Response containing tokenized prompts."""

    results: list[TokenList]
