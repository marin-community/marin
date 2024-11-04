from typing import Any

from pydantic import BaseModel


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class DolmaConversationOutput(BaseModel):
    id: str
    source: str
    messages: list[OpenAIChatMessage]
    added: str
    created: str
    metadata: dict[str, Any]
