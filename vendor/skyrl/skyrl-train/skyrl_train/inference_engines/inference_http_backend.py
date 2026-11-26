"""Shared protocol for the OpenAI-compatible inference HTTP bridge."""

from typing import Any, Protocol


class InferenceHTTPBackend(Protocol):
    """Engine operations consumed by the HTTP endpoint and continuation bridge."""

    model_name: str
    max_model_len: int | None

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...

    async def chat_completion_stream(self, request_payload: dict[str, Any]): ...

    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...

    async def tokenize(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...
