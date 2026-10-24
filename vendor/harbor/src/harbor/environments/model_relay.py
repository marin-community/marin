# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only model capabilities shared by Shellbox environment adapters."""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

import httpx

from shellbox.http_service import MAX_BODY, HttpRequest, HttpResponse

MODEL_RELAY_PORT = 8787
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
        if request.method != "POST" or request.path != "/v1/chat/completions":
            return HttpResponse(404, b'{"error":"unsupported model route"}')
        try:
            payload = json.loads(request.body)
        except (ValueError, UnicodeDecodeError):
            return HttpResponse(400, b'{"error":"invalid JSON"}')
        if not isinstance(payload, dict) or payload.get("stream"):
            return HttpResponse(
                400, b'{"error":"non-streaming chat completions required"}'
            )
        payload["model"] = self.model
        try:
            async with httpx.AsyncClient(follow_redirects=False, timeout=300) as client:
                async with client.stream(
                    "POST",
                    self.endpoint.rstrip("/") + "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ) as response:
                    if not response.is_success:
                        return HttpResponse(502, b'{"error":"model request failed"}')
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > MAX_BODY:
                            return HttpResponse(
                                502, b'{"error":"model response exceeded limit"}'
                            )
                        body.extend(chunk)
                    return HttpResponse(response.status_code, bytes(body))
        except httpx.HTTPError:
            return HttpResponse(502, b'{"error":"model request failed"}')
