# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
