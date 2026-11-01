# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only model capabilities shared by Shellbox environment adapters."""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

import httpx

from harbor.llms.vllm_tokenization import tokenize_url
from shellbox.http_service import MAX_BODY, HttpRequest, HttpResponse

MODEL_RELAY_PORT = 8787
CANDIDATE_MODEL_RELAY_PORT = 8788
_HOST_MODEL_AUTH: ContextVar[Mapping[str, str]] = ContextVar(
    "harbor_host_model_auth", default={}
)


@contextmanager
def host_model_authentication(values: Mapping[str, str]) -> Iterator[None]:
    """Grant upstream authentication to host handlers without serializing it."""
    token = _HOST_MODEL_AUTH.set(dict(values))
    try:
        yield
    finally:
        _HOST_MODEL_AUTH.reset(token)


class ModelRelay:
    """A fixed OpenAI-compatible endpoint and model, authenticated only on the host."""

    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model.removeprefix("together_ai/")
        self._api_key = _HOST_MODEL_AUTH.get().get(endpoint, "EMPTY")

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
