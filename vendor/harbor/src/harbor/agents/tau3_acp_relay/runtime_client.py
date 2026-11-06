"""Typed MCP operations for the benchmark-owned Tau3 runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from mcp import ClientSession

from .relay import RuntimeStatus, RuntimeTurn, Tau3RuntimeInfrastructureError


@contextmanager
def runtime_http_errors() -> Iterator[None]:
    """Classify HTTP failures raised by MCP's transport task group."""
    try:
        yield
    except* httpx.HTTPError as exc:
        raise Tau3RuntimeInfrastructureError(
            "Tau3 runtime HTTP transport failed"
        ) from exc


class MCPRuntimeClient:
    """Typed wrapper around Harbor's tau3-runtime MCP lifecycle tools."""

    def __init__(self, session: ClientSession, timeout_sec: float) -> None:
        self._session = session
        self._timeout_sec = timeout_sec

    async def configure(
        self, *, seed: int | None, max_steps: int, max_errors: int
    ) -> RuntimeStatus:
        payload = await self._call_json(
            "configure_run",
            {"seed": seed, "max_steps": max_steps, "max_errors": max_errors},
        )
        return parse_status(payload)

    async def start(self) -> tuple[str, RuntimeStatus]:
        observation = await self._call_text("start_conversation")
        status = await self.status()
        return observation, status

    async def status(self) -> RuntimeStatus:
        return parse_status(await self._call_json("get_runtime_status"))

    async def submit_message(self, message: str) -> RuntimeTurn:
        payload = await self._call_json(
            "submit_assistant_message", {"message": message}
        )
        return RuntimeTurn(
            observation=payload.get("observation"),
            status=parse_status(payload),
        )

    async def record_termination(self, reason: str) -> RuntimeStatus:
        payload = await self._call_json("record_termination", {"reason": reason})
        return parse_status(payload)

    async def _call_json(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        text = await self._call_text(name, arguments)
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise Tau3RuntimeInfrastructureError(
                f"{name} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise Tau3RuntimeInfrastructureError(
                f"{name} returned {type(payload).__name__}, not an object"
            )
        return payload

    async def _call_text(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        try:
            async with asyncio.timeout(self._timeout_sec):
                result = await self._session.call_tool(name, arguments or {})
        except Exception as exc:
            raise Tau3RuntimeInfrastructureError(
                f"tau3 runtime tool {name} request failed"
            ) from exc
        payload = result.model_dump(by_alias=True)
        if payload["isError"]:
            raise Tau3RuntimeInfrastructureError(
                f"tau3 runtime tool {name} failed: {result}"
            )
        parts = [item["text"] for item in payload["content"] if item["type"] == "text"]
        if not parts:
            raise Tau3RuntimeInfrastructureError(
                f"tau3 runtime tool {name} returned no text"
            )
        return "\n".join(parts)


def parse_status(payload: dict[str, Any]) -> RuntimeStatus:
    """Parse runtime status fields without accepting malformed values."""
    try:
        reason = payload["termination_reason"]
        if reason is not None and not isinstance(reason, str):
            raise ValueError("termination_reason must be a string or null")
        return RuntimeStatus(
            termination_reason=reason,
            step_count=int(payload["step_count"]),
            num_errors=int(payload["num_errors"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Tau3RuntimeInfrastructureError("Malformed tau3 runtime status") from exc
