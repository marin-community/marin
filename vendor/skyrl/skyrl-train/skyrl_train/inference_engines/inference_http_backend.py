"""Shared protocol for the OpenAI-compatible inference HTTP bridge."""

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import uuid4

from starlette.datastructures import Headers, State


class OpenAIRequest:
    """Case-insensitive headers and response state for an in-process HTTP handler."""

    def __init__(self, headers: Mapping[str, str]):
        values = Headers(headers)
        self.headers = Headers({**values, "x-request-id": values.get("x-request-id") or uuid4().hex})
        self.state = State()


class InferenceHTTPBackend(Protocol):
    """Engine operations consumed by the HTTP endpoint and continuation bridge."""

    model_name: str
    max_model_len: int | None

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...

    async def chat_completion_stream(self, request_payload: dict[str, Any]): ...

    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...

    async def tokenize(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...
