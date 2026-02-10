# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = Field(default=None, alias="tool_call_id")


class DolmaConversationOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    source: str
    messages: list[OpenAIChatMessage]
    added: str
    created: str
    metadata: dict[str, Any]
