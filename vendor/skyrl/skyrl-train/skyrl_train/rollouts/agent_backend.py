"""Serve installed-agent requests through the shared rollout engine."""

import json
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol
from uuid import uuid4

from rolloutengine.models import LENGTH_STOP_REASON, ModelTurn, Prompt


TRIAL_HEADER = "x-ot-trial-id"
TRANSPORT_FIELDS = frozenset(
    {"model", "messages", "stream", "stream_options", "n", "user", "return_token_ids", "logprobs", "top_logprobs"}
)


class TokenDecoder(Protocol):
    def decode(self, token_ids: list[int]) -> str: ...


class AgentGenerationStopped(Exception):
    """The engine ended generation; the agent must proceed to verification."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Generation stopped: {reason}")


class AgentInferenceBackend:
    """Route each correlated HTTP request to its waiting agent session.

    Register this backend with the existing inference HTTP server. Tokenization
    uses the serving backend; generation must pass through a registered session.
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int,
        tokenize: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        decoder: TokenDecoder,
        *,
        model_revision: str | None = None,
        source_kind: str | None = None,
        model_endpoint: str | None = None,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.source_kind = source_kind
        self.model_endpoint = model_endpoint
        self.max_model_len = max_model_len
        self.tokenize = tokenize
        self.decoder = decoder
        self.sessions: dict[str, Callable[[Prompt], Awaitable[ModelTurn]]] = {}

    @contextmanager
    def session(self, trial_id: str, complete: Callable[[Prompt], Awaitable[ModelTurn]]) -> Iterator[None]:
        if trial_id in self.sessions:
            raise ValueError(f"Agent session already registered: {trial_id}")
        self.sessions[trial_id] = complete
        try:
            yield
        finally:
            del self.sessions[trial_id]

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        trial_id = request_payload["headers"].get(TRIAL_HEADER)
        if trial_id not in self.sessions:
            raise ValueError("No active agent session matches the request correlation ID")
        body = request_payload["json"]
        if body.get("n", 1) != 1:
            raise ValueError("An agent session requires exactly one completion per request")
        options = {key: value for key, value in body.items() if key not in TRANSPORT_FIELDS}
        try:
            turn = await self.sessions[trial_id](Prompt(tuple(body["messages"]), options))
        except AgentGenerationStopped as error:
            return {
                "error": {
                    "message": "Maximum context length reached" if error.reason == LENGTH_STOP_REASON else str(error),
                    "type": "context_length_exceeded" if error.reason == LENGTH_STOP_REASON else "generation_stopped",
                    "code": 400,
                }
            }
        finish_reason = (
            "length"
            if turn.stop_reason == LENGTH_STOP_REASON
            else "tool_calls"
            if turn.message.get("tool_calls")
            else "stop"
        )
        logprobs = None
        if body.get("logprobs") is True and turn.logprobs is not None:
            logprobs = {
                "content": [
                    {
                        "token": self.decoder.decode([token]),
                        "logprob": probability,
                        "bytes": None,
                        "top_logprobs": [],
                    }
                    for token, probability in zip(turn.response_token_ids, turn.logprobs, strict=True)
                ]
            }
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "prompt_token_ids": list(turn.prompt_token_ids),
            "choices": [
                {
                    "index": 0,
                    "message": turn.message,
                    "finish_reason": finish_reason,
                    "logprobs": logprobs,
                    "token_ids": list(turn.response_token_ids),
                }
            ],
            "usage": {
                "prompt_tokens": len(turn.prompt_token_ids),
                "completion_tokens": len(turn.response_token_ids),
                "total_tokens": len(turn.prompt_token_ids) + len(turn.response_token_ids),
            },
        }

    async def chat_completion_stream(self, request_payload: dict[str, Any]):
        response = await self.chat_completion(request_payload)
        if "error" in response:
            yield f"data: {json.dumps(response)}\n\n"
            yield "data: [DONE]\n\n"
            return
        choice = response["choices"][0]
        delta = dict(choice["message"])
        if "tool_calls" in delta:
            delta["tool_calls"] = [dict(call, index=index) for index, call in enumerate(delta["tool_calls"])]
        chunk = {
            **response,
            "object": "chat.completion.chunk",
            "choices": [{key: value for key, value in choice.items() if key != "message"} | {"delta": delta}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("Installed-agent sessions require chat-completion requests")
