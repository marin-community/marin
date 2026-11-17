# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible proxy that records every chat completion a Harbor agent makes.

Point the agent's API base at the app from ``build_proxy`` (serve it with uvicorn). The proxy forwards
each request to a teacher backend, or, with a ``RelayPolicy``, routes it between a student and the
teacher with ``relay.route``. For every chat completion it:

- adds the backend's headers, so credentials stay in the proxy process and never reach the agent or
  its sandbox. Read them from the environment; do not pass them on a command line;
- appends one record to a JSONL ``RequestLog``: trial id, author, relay trigger, request settings, the
  conversation the request continues and the messages it adds, usage, latency and the full reply,
  reasoning included. ``rebuild_traces.py`` turns this log into one row per conversation.

Records are grouped by the per-trial correlation header (``TRIAL_HEADER``) that the agents in
``harbor_agents/trial_agents.py`` send. A request continues the trial's longest conversation whose
messages are a prefix of its own. Otherwise it starts a new conversation. Relay state is per trial,
so with a ``RelayPolicy`` a request without the header is refused.

The proxy also serves vLLM's ``/tokenize`` (``count`` and ``max_model_len``), which Harbor's
mini-swe-agent context limiter calls and many hosted APIs lack. Each backend counts the request it
would receive, overrides applied, and the count is the largest over the backends, so the context
limit holds for whichever model answers. Streaming is not supported.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from transformers import PreTrainedTokenizerBase

from experiments.post_training.sft_shootout.relay import Author, TrialState, Trigger, is_completion_claim, route

logger = logging.getLogger(__name__)

# Harbor's CORRELATION_ID_HEADER (harbor.literal.native_api).
TRIAL_HEADER = "x-ot-trial-id"
UNLOGGED_REQUEST_KEYS = frozenset({"messages", "tools"})
# Status codes an Iris capability URL returns while its endpoint is unregistered or its cluster is unreachable.
GATEWAY_STATUSES = frozenset({404, 502, 503, 504})
RETRY_INTERVAL = 30
LOGGED_ERROR_CHARS = 2000


@dataclass(frozen=True)
class Backend:
    """An OpenAI-compatible chat completions endpoint.

    ``retry_window`` is how long refused connections and gateway errors are retried, for a backend that
    can move while the proxy runs (an Iris job that is preempted and rescheduled). ``overrides`` are
    request fields set on every request to this backend, over the agent's (e.g. a reasoning effort).
    """

    base_url: str
    model: str
    tokenizer: PreTrainedTokenizerBase
    headers: dict[str, str]
    retry_window: float = 0.0
    overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RelayPolicy:
    student: Backend
    budget_tokens: int


class RequestLog:
    """Append-only JSONL log shared by the proxy's request handlers."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        with self._lock, self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def template_message(message: dict) -> dict:
    """Parse tool-call arguments from JSON strings to dicts, as vLLM does before rendering."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    parsed = []
    for call in calls:
        function = call.get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function = {**function, "arguments": json.loads(arguments)}
            except json.JSONDecodeError:
                pass  # vLLM renders unparseable arguments as the raw string too
        parsed.append({**call, "function": function})
    return {**message, "tool_calls": parsed}


def backend_request(backend: Backend, body: dict) -> dict:
    """The request ``backend`` receives: the agent's body under the backend's overrides and model."""
    return {**body, **backend.overrides, "model": backend.model}


def count_tokens(tokenizer: PreTrainedTokenizerBase, body: dict) -> int:
    rendered = tokenizer.apply_chat_template(
        [template_message(m) for m in body["messages"]],
        tools=body.get("tools") or None,
        add_generation_prompt=body.get("add_generation_prompt", True),
        tokenize=False,
        **(body.get("chat_template_kwargs") or {}),
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def message_key(message: dict) -> str:
    """Identity of a message for prefix matching: role, content, tool calls and tool call id."""
    calls = [(c.get("function") or {}) for c in message.get("tool_calls") or []]
    fields = {
        "role": message.get("role"),
        "content": message.get("content"),
        "tool_calls": [[c.get("name"), c.get("arguments")] for c in calls],
        "tool_call_id": message.get("tool_call_id"),
    }
    return hashlib.sha1(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def continued_conversation(conversations: list[list[str]], keys: list[str]) -> int | None:
    """Index of the longest conversation whose message keys are a prefix of ``keys``."""
    best = None
    for index, known in enumerate(conversations):
        if len(known) <= len(keys) and keys[: len(known)] == known:
            if best is None or len(known) > len(conversations[best]):
                best = index
    return best


def build_proxy(
    teacher: Backend,
    max_model_len: int,
    log: RequestLog,
    relay: RelayPolicy | None,
    client: httpx.AsyncClient,
) -> Starlette:
    """Return the proxy app. ``max_model_len`` is the window ``/tokenize`` reports to agents."""
    backends = {Author.TEACHER: teacher, **({Author.STUDENT: relay.student} if relay else {})}
    states: dict[str, TrialState] = defaultdict(TrialState)
    conversations: dict[str, list[list[str]]] = defaultdict(list)

    async def counts(body: dict) -> dict[Author, int]:
        values = await asyncio.gather(
            *(
                asyncio.to_thread(count_tokens, backend.tokenizer, backend_request(backend, body))
                for backend in backends.values()
            )
        )
        return dict(zip(backends, values, strict=True))

    async def call(backend: Backend, body: dict) -> tuple[httpx.Response, float]:
        started = time.monotonic()
        while True:
            retry = time.monotonic() - started < backend.retry_window
            try:
                response = await client.post(
                    f"{backend.base_url}/chat/completions", json=backend_request(backend, body), headers=backend.headers
                )
                if response.status_code not in GATEWAY_STATUSES or not retry:
                    return response, time.monotonic() - started
                logger.warning("Backend model %s answered %d; retrying", backend.model, response.status_code)
            except httpx.ConnectError:
                if not retry:
                    raise
                logger.warning("Backend model %s refused the connection; retrying", backend.model)
            await asyncio.sleep(RETRY_INTERVAL)

    def turn_record(author: Author, trigger: Trigger | None, response: httpx.Response, latency: float) -> dict:
        backend = backends[author]
        record: dict = {
            "author": author.value,
            "model": backend.model,
            "trigger": trigger,
            "status": response.status_code,
            "latency": latency,
        }
        if backend.overrides:
            record["overrides"] = backend.overrides
        if response.status_code != 200:
            record["error"] = response.text[:LOGGED_ERROR_CHARS]
            return record
        payload = response.json()
        choices = payload.get("choices", [])
        record.update(
            usage=payload.get("usage"),
            finish_reasons=[c.get("finish_reason") for c in choices],
            response_messages=[c.get("message") for c in choices],
            response_id=payload.get("id"),
            served_model=payload.get("model"),
        )
        return record

    async def tokenize(request: Request) -> Response:
        tokens = await counts(await request.json())
        return JSONResponse({"count": max(tokens.values()), "max_model_len": max_model_len})

    async def models(_request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": teacher.model, "object": "model"}]})

    def track_conversation(trial: str, messages: list[dict]) -> tuple[int, int]:
        """Record ``messages`` in the trial's conversations; return its index and the first new message."""
        keys = [message_key(m) for m in messages]
        known = conversations[trial]
        index = continued_conversation(known, keys) if trial else None
        if index is None:
            known.append(keys)
            return len(known) - 1, 0
        first = len(known[index])
        known[index] = keys
        return index, first

    async def chat(request: Request) -> Response:
        body = await request.json()
        if body.get("stream"):
            return JSONResponse({"error": "streaming is not supported by this proxy"}, status_code=400)
        trial = request.headers.get(TRIAL_HEADER) or ""
        if relay is not None and not trial:
            logger.warning("refused a relay request without the %s header", TRIAL_HEADER)
            return JSONResponse({"error": f"relay routing needs the {TRIAL_HEADER} header"}, status_code=400)
        messages = body.get("messages", [])
        index, first = track_conversation(trial, messages)
        record: dict = {
            "ts": time.time(),
            "trial": trial or None,
            "settings": {k: v for k, v in body.items() if k not in UNLOGGED_REQUEST_KEYS},
            "conversation": index,
            "n_messages": len(messages),
            "first_message": first,
            "messages": messages[first:],
            "tools": body.get("tools") if first == 0 else None,
        }
        author, trigger = Author.TEACHER, None
        try:
            tokens = await counts(body)
            record["prompt_tokens_local"] = {a.value: value for a, value in tokens.items()}
            if relay is not None:
                author, trigger = route(states[trial], messages, tokens[Author.STUDENT], relay.budget_tokens)
            response, latency = await call(backends[author], body)
            turn = turn_record(author, trigger, response, latency)

            if author is Author.STUDENT and response.status_code == 200:
                if is_completion_claim(turn["response_messages"][0] or {}):
                    states[trial].takeover = Trigger.COMPLETION_CLAIM
                    record["withheld_student_turn"] = turn
                    author, trigger = Author.TEACHER, Trigger.COMPLETION_CLAIM
                    response, latency = await call(teacher, body)
                    turn = turn_record(author, trigger, response, latency)
        except Exception as error:  # log it: a bare 500 tells the agent nothing and leaves no trace of the cause
            logger.exception("proxy failure for trial %s", trial)
            log.write(
                {
                    **record,
                    "author": author.value,
                    "trigger": trigger,
                    "status": 502,
                    "latency": None,
                    "error": repr(error)[:LOGGED_ERROR_CHARS],
                }
            )
            return JSONResponse({"error": f"proxy failure: {type(error).__name__}"}, status_code=502)
        log.write({**record, **turn})
        return Response(response.content, response.status_code, media_type=response.headers.get("content-type"))

    return Starlette(
        routes=[
            Route("/tokenize", tokenize, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat, methods=["POST"]),
        ]
    )
