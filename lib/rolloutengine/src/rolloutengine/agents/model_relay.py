# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve an installed agent's OpenAI-compatible model requests from an ``AgentHarness`` completion.

Adapted from Harbor's ``ModelRelay`` (``harbor.environments.model_relay``, Apache-2.0; see
``rolloutengine.agents.pi``). Harbor relays the agent's requests to a host-authenticated endpoint. Here each chat
request becomes one ``Prompt`` for the harness's ``complete``, so the rollout records every turn, and the policy
model, with its credentials, stays on the host. ``serve_model_relay`` serves the relay at a loopback port inside
the task machine through Shellbox's HTTP service, which carries requests over the machine's process pipes: the
machine needs no route to the host, and the host none to the machine.

The relay answers ``POST /v1/chat/completions`` and vLLM's ``POST /tokenize``. A streaming chat request gets the
complete answer as one buffered server-sent event stream, as Harbor's relay sends it.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from shellbox.http_service import MAX_BODY, HttpRequest, HttpResponse, serve_http
from shellbox.machine import Machine

from rolloutengine.chat import CONTEXT_LENGTH_CODE, CONTEXT_LENGTH_PHRASES
from rolloutengine.harness.external import AgentCompletion
from rolloutengine.models import ContextWindowExceeded, ModelTurn, Prompt, RolloutEnded

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
TOKENIZE_PATH = "/tokenize"
TRANSPORT_FIELDS = frozenset({"model", "messages", "stream", "stream_options", "n"})
"""Request fields that the relay handles itself. The policy model sets the model and the messages."""
CHAT_PARAMETERS = CompletionCreateParamsBase.__required_keys__ | CompletionCreateParamsBase.__optional_keys__
"""OpenAI chat parameters. Other request fields, such as ``chat_template_kwargs``, go into ``extra_body``."""
CONTEXT_LENGTH_RESPONSE = HttpResponse(
    400,
    b'{"error":{"type":"invalid_request_error","code":"context_length_exceeded",'
    b'"message":"Maximum context length exceeded"}}',
)
AUTHENTICATION_RESPONSE_BODY = (
    b'{"error":{"type":"authentication_error","code":"authentication_error","message":"Model authentication required"}}'
)
FAILED_RESPONSE = HttpResponse(502, b'{"error":"model request failed"}')
ENDED_RESPONSE = HttpResponse(503, b'{"error":"the rollout ended"}')


class PromptTokenizer(Protocol):
    """Answer a vLLM ``/tokenize`` request for the policy model, with its credentials on the host."""

    async def __call__(self, payload: dict[str, Any]) -> Any: ...


class PromptTokenizationError(RuntimeError):
    """An unsuccessful tokenizer response from a direct model client."""

    def __init__(self, status_code: int | None):
        self.status_code = status_code
        super().__init__(f"Policy tokenization failed (status {status_code})")


def tokenize_url(api_base: str) -> str:
    """Return vLLM's server-root tokenization endpoint for an OpenAI API base."""
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return f"{base}/tokenize"


@dataclass(frozen=True)
class VllmTokenizer:
    """Tokenize with the vLLM server behind ``client``, the client of the policy's ``ChatModel``, as ``model``."""

    client: openai.AsyncOpenAI
    model: str

    async def __call__(self, payload: dict[str, Any]) -> Any:
        url = tokenize_url(str(self.client.base_url))
        return await self.client.post(url, cast_to=object, body={**payload, "model": self.model})


def chat_prompt(payload: dict[str, Any]) -> Prompt:
    """The ``Prompt`` of a chat request: its messages, its OpenAI parameters, and its other fields as ``extra_body``."""
    options = {key: value for key, value in payload.items() if key not in TRANSPORT_FIELDS}
    extra_body = {key: options.pop(key) for key in list(options) if key not in CHAT_PARAMETERS}
    if extra_body:
        options["extra_body"] = extra_body
    return Prompt(tuple(payload["messages"]), options)


def chat_completion(turn: ModelTurn, model: str) -> dict[str, Any]:
    """The OpenAI chat completion of ``turn``, with the reasoning and refusal that ``ChatModel`` keeps as metadata."""
    message = dict(turn.message)
    for key in ("reasoning", "refusal"):
        if key in turn.metadata and key not in message:
            message[key] = turn.metadata[key]
    if turn.usage is not None:
        prompt_tokens, completion_tokens = turn.usage.prompt_tokens, turn.usage.completion_tokens
    else:
        prompt_tokens, completion_tokens = len(turn.prompt_token_ids), len(turn.response_token_ids)
    return {
        "id": turn.metadata.get("response_id", f"chatcmpl-{uuid.uuid4().hex}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": turn.stop_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def completion_events(completion: dict[str, Any]) -> bytes:
    """Encode a buffered completion in the OpenAI SDK's SSE wire format."""
    base = {key: completion[key] for key in ("id", "created", "model") if key in completion}
    base["object"] = "chat.completion.chunk"
    choices = []
    finished = []
    for index, choice in enumerate(completion["choices"]):
        delta = dict(choice["message"])
        if delta.get("tool_calls"):
            delta["tool_calls"] = [{**tool, "index": tool_index} for tool_index, tool in enumerate(delta["tool_calls"])]
        choices.append({"index": choice.get("index", index), "delta": delta, "finish_reason": None})
        finished.append({"index": choice.get("index", index), "delta": {}, "finish_reason": choice["finish_reason"]})
    events = [{**base, "choices": choices}, {**base, "choices": finished}]
    if "usage" in completion:
        events.append({**base, "choices": [], "usage": completion["usage"]})
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"


def upstream_error(status: int, body: bytes) -> HttpResponse:
    """Preserve SDK failure classification without exposing upstream capabilities."""
    if status in (401, 403):
        return HttpResponse(status, AUTHENTICATION_RESPONSE_BODY)
    if status in (400, 413, 422):
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        message = str(error.get("message", "")).lower() if isinstance(error, dict) else ""
        if isinstance(error, dict) and (
            error.get("code") == CONTEXT_LENGTH_CODE or any(phrase in message for phrase in CONTEXT_LENGTH_PHRASES)
        ):
            return CONTEXT_LENGTH_RESPONSE
    return FAILED_RESPONSE


@dataclass
class ModelRelay:
    """Answer an agent's chat requests with ``complete`` and its tokenization requests with ``tokenize``.

    A context rejection answers with an OpenAI ``context_length_exceeded`` error, so the agent can compact and
    retry. A request after the end of the attempt waits for the cleanup, then gets a 503 response. An
    authentication failure of ``tokenize`` answers as one; any other failure answers with a 502 response, and
    ``upstream_failure`` keeps the first such failure.
    """

    complete: AgentCompletion
    tokenize: PromptTokenizer
    upstream_failure: Exception | None = field(default=None, init=False)

    async def handle(self, request: HttpRequest) -> HttpResponse:
        if request.method != "POST" or request.path not in {CHAT_COMPLETIONS_PATH, TOKENIZE_PATH}:
            return HttpResponse(404, b'{"error":"unsupported model route"}')
        if len(request.body) > MAX_BODY:
            return HttpResponse(413, b'{"error":"model request exceeded limit"}')
        try:
            payload = json.loads(request.body)
        except (ValueError, UnicodeDecodeError):
            return HttpResponse(400, b'{"error":"invalid JSON"}')
        if not isinstance(payload, dict):
            return HttpResponse(400, b'{"error":"JSON object required"}')
        if request.path == TOKENIZE_PATH:
            return await self._tokenize(payload)
        if payload.get("n", 1) != 1:
            return HttpResponse(400, b'{"error":"one completion per request is required"}')
        try:
            turn = await self.complete(chat_prompt(payload))
        except ContextWindowExceeded:
            return CONTEXT_LENGTH_RESPONSE
        except RolloutEnded:
            return ENDED_RESPONSE
        completion = chat_completion(turn, str(payload.get("model", "")))
        if payload.get("stream") is True:
            return HttpResponse(200, completion_events(completion), "text/event-stream")
        return HttpResponse(200, json.dumps(completion).encode())

    async def _tokenize(self, payload: dict[str, Any]) -> HttpResponse:
        payload.pop("stream", None)
        payload.pop("stream_options", None)
        try:
            tokens = await self.tokenize(payload)
        except openai.APIStatusError as error:
            response = upstream_error(error.status_code, error.response.content)
            if response is FAILED_RESPONSE:
                self._record_failure(error)
            return response
        except PromptTokenizationError as error:
            response = upstream_error(error.status_code or 500, b"{}")
            if response is FAILED_RESPONSE:
                self._record_failure(error)
            return response
        except openai.APIError as error:
            self._record_failure(error)
            return FAILED_RESPONSE
        return HttpResponse(200, json.dumps(tokens).encode())

    def _record_failure(self, error: Exception) -> None:
        if self.upstream_failure is None:
            self.upstream_failure = error


@asynccontextmanager
async def serve_model_relay(machine: Machine, relay: ModelRelay, *, port: int) -> AsyncIterator[str]:
    """Serve ``relay`` at ``http://127.0.0.1:{port}`` inside ``machine`` until the context exits.

    The machine must support persistent process sessions and run ``python3``. A failure of the relay's transport
    fails the context's body.

    Raises:
        UnsupportedMachineSpec: The machine has no persistent process sessions.
    """
    try:
        async with asyncio.TaskGroup() as services:
            async with serve_http(machine, relay.handle, port=port, task_group=services) as url:
                yield url
    except BaseExceptionGroup as group:
        # The task group wraps the one failure of the body or of the relay's transport; raise it as it is.
        if len(group.exceptions) == 1:
            raise group.exceptions[0]  # noqa: B904
        raise
