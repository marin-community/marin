# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Relay task-local model requests to a fixed, host-authenticated endpoint."""

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from enum import StrEnum

import httpx

from harbor.llms.vllm_tokenization import tokenize_url
from harbor.host_capabilities import (
    CANDIDATE_MODEL_RELAY_PORT,
    MODEL_RELAY_PORT,
    model_api_url,
)
from harbor.models.agent.model_api import ModelAPI
from harbor.trial.errors import RuntimeServiceError
from shellbox.http_service import MAX_BODY, HttpRequest, HttpResponse, serve_http
from shellbox.machine import Machine

SUPPORTED_MODEL_APIS = frozenset({ModelAPI.OPENAI_CHAT})


class JudgeAccess(StrEnum):
    VERIFIER = "verifier"
    RUNTIME = "runtime"


@asynccontextmanager
async def verifier_model_access(
    machine: Machine,
    judge: "ModelRelay | None",
    access: JudgeAccess,
    service_tasks: asyncio.TaskGroup,
) -> AsyncIterator[None]:
    """Grant verifier-only model access for the grading phase."""
    async with AsyncExitStack() as stack:
        if access is JudgeAccess.VERIFIER:
            await start_model_relays(
                machine, stack, service_tasks, judge=judge, candidate=None
            )
        yield


async def start_model_relays(
    machine: Machine,
    stack: AsyncExitStack,
    service_tasks: asyncio.TaskGroup,
    *,
    judge: "ModelRelay | None",
    candidate: "ModelRelay | None",
) -> None:
    """Enter role relays whose contexts and service tasks belong to the caller."""
    for relay, port in (
        (judge, MODEL_RELAY_PORT),
        (candidate, CANDIDATE_MODEL_RELAY_PORT),
    ):
        if relay is not None:
            try:
                await stack.enter_async_context(
                    serve_http(
                        machine, relay.handle, port=port, task_group=service_tasks
                    )
                )
            except Exception as exc:
                raise RuntimeServiceError("Model relay failed to start") from exc


class ModelRelay:
    """Relay bounded chat, text Responses and tokenization requests using host credentials.

    Streaming chat responses are buffered before SSE encoding.
    """

    def __init__(self, endpoint: str, model: str, *, api_key: str):
        self.endpoint = endpoint
        self.model = model.removeprefix("together_ai/")
        self._api_key = api_key

    @classmethod
    def from_endpoint(
        cls,
        endpoint: str | None,
        model: str | None,
        *,
        authentication: Mapping[str, str],
    ) -> "ModelRelay | None":
        """Resolve an optional relay before allocating its guest environment."""
        if endpoint is None and model is None:
            return None
        if endpoint is None or model is None:
            raise ValueError("Model relay requires a host endpoint and model")
        return cls(endpoint, model, api_key=authentication.get(endpoint, "EMPTY"))

    async def handle(self, request: HttpRequest) -> HttpResponse:
        if request.method != "POST" or request.path not in {
            "/v1/chat/completions",
            "/v1/responses",
            "/tokenize",
        }:
            return HttpResponse(404, b'{"error":"unsupported model route"}')
        if len(request.body) > MAX_BODY:
            return HttpResponse(413, b'{"error":"model request exceeded limit"}')
        try:
            payload = json.loads(request.body)
        except (ValueError, UnicodeDecodeError):
            return HttpResponse(400, b'{"error":"invalid JSON"}')
        if not isinstance(payload, dict):
            return HttpResponse(400, b'{"error":"JSON object required"}')
        if request.path == "/v1/responses":
            if payload.keys() - {
                "model",
                "input",
                "max_output_tokens",
                "reasoning",
                "stream",
            }:
                return HttpResponse(400, b'{"error":"unsupported response options"}')
            if payload.get("stream") or not isinstance(payload.get("input"), str):
                return HttpResponse(
                    400, b'{"error":"non-streaming text responses required"}'
                )
        streaming = (
            request.path == "/v1/chat/completions" and payload.get("stream") is True
        )
        if request.path == "/v1/chat/completions":
            payload["stream"] = False
        else:
            payload.pop("stream", None)
        payload.pop("stream_options", None)
        payload["model"] = self.model
        if len(json.dumps(payload).encode()) > MAX_BODY:
            return HttpResponse(413, b'{"error":"model request exceeded limit"}')
        if request.path == "/tokenize":
            url = tokenize_url(self.endpoint)
        else:
            route = (
                "/responses" if request.path == "/v1/responses" else "/chat/completions"
            )
            url = self.endpoint.rstrip("/") + route
        try:
            async with httpx.AsyncClient(follow_redirects=False, timeout=300) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ) as response:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > MAX_BODY:
                            return HttpResponse(
                                502, b'{"error":"model response exceeded limit"}'
                            )
                        body.extend(chunk)
                    if not response.is_success:
                        return _upstream_error(response.status_code, bytes(body))
                    if streaming:
                        try:
                            events = _completion_events(json.loads(body))
                        except (ValueError, KeyError, TypeError, AttributeError):
                            return HttpResponse(
                                502, b'{"error":"invalid model response"}'
                            )
                        if len(events) > MAX_BODY:
                            return HttpResponse(
                                502, b'{"error":"model response exceeded limit"}'
                            )
                        return HttpResponse(200, events, "text/event-stream")
                    return HttpResponse(response.status_code, bytes(body))
        except httpx.HTTPError:
            return HttpResponse(502, b'{"error":"model request failed"}')


def task_model_relays(
    *,
    model_endpoint: str | None,
    model_name: str | None,
    candidate_model_endpoint: str | None,
    candidate_model_name: str | None,
    authentication: Mapping[str, str],
    judge_access: JudgeAccess,
    environments: Iterable[Mapping[str, str]],
) -> tuple[ModelRelay | None, ModelRelay | None]:
    """Resolve role relays and require runtime authority for task model callers."""
    judge = ModelRelay.from_endpoint(
        model_endpoint, model_name, authentication=authentication
    )
    candidate = ModelRelay.from_endpoint(
        candidate_model_endpoint, candidate_model_name, authentication=authentication
    )
    endpoints = tuple(model_api_url(values) for values in environments)
    if any(endpoint is not None for endpoint in endpoints):
        if judge is None:
            raise ValueError(
                "Task model calls require an explicit host judge endpoint and model"
            )
        if judge_access is not JudgeAccess.RUNTIME:
            raise ValueError("Task model calls require explicit judge_access='runtime'")
    return judge, candidate


def _upstream_error(status: int, body: bytes) -> HttpResponse:
    """Preserve SDK failure classification without exposing upstream capabilities."""
    if status in (401, 403):
        return HttpResponse(
            status,
            b'{"error":{"type":"authentication_error","code":"authentication_error",'
            b'"message":"Model authentication required"}}',
        )
    if status in (400, 413, 422):
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and (
            error.get("code") == "context_length_exceeded"
            or "maximum context length" in str(error.get("message", "")).lower()
        ):
            return HttpResponse(
                400,
                b'{"error":{"type":"invalid_request_error","code":"context_length_exceeded",'
                b'"message":"Maximum context length exceeded"}}',
            )
    return HttpResponse(502, b'{"error":"model request failed"}')


def _completion_events(completion: dict) -> bytes:
    """Encode a buffered completion in the OpenAI SDK's SSE wire format."""
    base = {
        key: completion[key] for key in ("id", "created", "model") if key in completion
    }
    base["object"] = "chat.completion.chunk"
    choices = []
    finished = []
    for index, choice in enumerate(completion["choices"]):
        delta = dict(choice["message"])
        if delta.get("tool_calls"):
            delta["tool_calls"] = [
                {**tool, "index": tool_index}
                for tool_index, tool in enumerate(delta["tool_calls"])
            ]
        choices.append(
            {"index": choice.get("index", index), "delta": delta, "finish_reason": None}
        )
        finished.append(
            {
                "index": choice.get("index", index),
                "delta": {},
                "finish_reason": choice["finish_reason"],
            }
        )
    events = [{**base, "choices": choices}, {**base, "choices": finished}]
    if "usage" in completion:
        events.append({**base, "choices": [], "usage": completion["usage"]})
    return (
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        + b"data: [DONE]\n\n"
    )
