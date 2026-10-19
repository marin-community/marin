# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible proxy between Harbor agents and the models of a trace generation run.

Plain runs have one backend, GLM 5.3 (the teacher). Relay runs add the student (v0) and route each
request with ``relay.route``. For every chat completion the proxy:

- adds the backend's credentials (the GLM token never reaches an agent or sandbox);
- writes one ``requests.jsonl`` record: trial id (``x-ot-trial-id``), author, relay trigger, the
  request settings exactly as sent, the conversation it continues and the messages it adds, usage,
  latency and the full reply (reasoning included). This log is the canonical trace; see
  ``rebuild_traces.py``.

A trial can hold several conversations: Terminus-2's context summarization sends summarizer calls
and then restarts from a shorter history. A request continues the trial's conversation whose
messages are a prefix of its own (the longest such); otherwise it starts a new one.

It also serves vLLM's ``/tokenize`` (``count``, ``max_model_len``), which Harbor's mini-swe-agent
context limiter needs and the GLM relay does not provide. In relay runs the count is the larger of
the GLM and v0 tokenizations, so the window holds for whichever model answers.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.post_training.sft_shootout.relay import Author, TrialState, Trigger, is_completion_claim, route

logger = logging.getLogger(__name__)

TRIAL_HEADER = "x-ot-trial-id"
UNLOGGED_REQUEST_KEYS = frozenset({"messages", "tools"})
# How long a request to a backend that can move keeps retrying before the proxy gives up on it: a preempted v0 is
# rescheduled by Iris, and its capability URL works again once the new job registers.
MOVE_RETRY_WINDOW = 1200
MOVE_RETRY_INTERVAL = 30
# What an Iris capability URL answers while its endpoint is unregistered or its cluster unreachable.
GATEWAY_STATUSES = frozenset({404, 502, 503, 504})


@dataclass(frozen=True)
class Backend:
    author: Author
    base_url: str
    model: str
    tokenizer: PreTrainedTokenizerBase
    headers: dict[str, str]
    # Iris may move it while the proxy runs: refused connections and gateway errors are retried for MOVE_RETRY_WINDOW.
    can_move: bool = False


@dataclass(frozen=True)
class RelayPolicy:
    student: Backend
    budget_tokens: int


class RequestLog:
    """Append-only JSONL log shared by the proxy's request handlers."""

    def __init__(self, path: Path):
        self._file = path.open("a")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()


def load_tokenizer(name: str, revision: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(name, revision=revision)


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
                pass
        parsed.append({**call, "function": function})
    return {**message, "tool_calls": parsed}


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
    relay: RelayPolicy | None = None,
    client: httpx.AsyncClient | None = None,
) -> Starlette:
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=30))
    states: dict[str, TrialState] = defaultdict(TrialState)
    conversations: dict[str, list[list[str]]] = defaultdict(list)

    async def counts(body: dict) -> dict[Author, int]:
        backends = [teacher] + ([relay.student] if relay else [])
        values = await asyncio.gather(*(asyncio.to_thread(count_tokens, b.tokenizer, body) for b in backends))
        return {b.author: value for b, value in zip(backends, values, strict=True)}

    async def call(backend: Backend, body: dict) -> tuple[httpx.Response, float]:
        started = time.monotonic()
        while True:
            retry = backend.can_move and time.monotonic() - started <= MOVE_RETRY_WINDOW
            try:
                response = await client.post(
                    f"{backend.base_url}/chat/completions",
                    json={**body, "model": backend.model},
                    headers=backend.headers,
                )
                if response.status_code not in GATEWAY_STATUSES or not retry:
                    return response, time.monotonic() - started
                logger.warning("%s answered %d; retrying while it moves", backend.author.value, response.status_code)
            except httpx.ConnectError:
                if not retry:
                    raise
                logger.warning("%s refused the connection; retrying while it moves", backend.author.value)
            await asyncio.sleep(MOVE_RETRY_INTERVAL)

    def turn_record(backend: Backend, trigger: Trigger | None, response: httpx.Response, latency: float) -> dict:
        record: dict = {
            "author": backend.author.value,
            "trigger": trigger,
            "status": response.status_code,
            "latency_s": latency,
        }
        if response.status_code != 200:
            record["error"] = response.text[:2000]
            return record
        payload = response.json()
        record.update(
            usage=payload.get("usage"),
            finish_reasons=[c.get("finish_reason") for c in payload.get("choices", [])],
            response_messages=[c.get("message") for c in payload.get("choices", [])],
            response_id=payload.get("id"),
            served_model=payload.get("model"),
            headers={k: v for k, v in response.headers.items() if k.startswith("x-glm")},
        )
        return record

    async def tokenize(request: Request) -> Response:
        tokens = await counts(await request.json())
        return JSONResponse({"count": max(tokens.values()), "max_model_len": max_model_len})

    async def models(request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": teacher.model, "object": "model"}]})

    async def chat(request: Request) -> Response:
        body = await request.json()
        if body.get("stream"):
            return JSONResponse({"error": "streaming is not supported by this proxy"}, status_code=400)
        trial = request.headers.get(TRIAL_HEADER)
        messages = body.get("messages", [])
        keys = [message_key(m) for m in messages]
        known = conversations[trial or ""]
        index = continued_conversation(known, keys) if trial else None
        first = len(known[index]) if index is not None else 0
        if index is None:
            known.append(keys)
            index = len(known) - 1
        else:
            known[index] = keys
        record: dict = {
            "ts": time.time(),
            "trial": trial,
            "settings": {k: v for k, v in body.items() if k not in UNLOGGED_REQUEST_KEYS},
            "conversation": index,
            "n_messages": len(messages),
            "first_message": first,
            "messages": messages[first:],
            "tools": body.get("tools") if first == 0 else None,
        }
        backend, trigger = teacher, None
        try:
            tokens = await counts(body)
            record["prompt_tokens_local"] = {author.value: value for author, value in tokens.items()}

            if relay is not None:
                state = states[trial or ""]
                author, trigger = route(state, messages, tokens[Author.STUDENT], relay.budget_tokens)
                backend = relay.student if author is Author.STUDENT else teacher
            response, latency = await call(backend, body)
            turn = turn_record(backend, trigger, response, latency)

            if relay is not None and backend is relay.student and response.status_code == 200:
                reply = turn["response_messages"][0] or {}
                if is_completion_claim(reply):
                    states[trial or ""].takeover = Trigger.COMPLETION_CLAIM
                    record["withheld_student_turn"] = turn
                    response, latency = await call(teacher, body)
                    turn = turn_record(teacher, Trigger.COMPLETION_CLAIM, response, latency)
        except Exception as error:  # log it: a bare 500 tells the agent nothing and leaves no trace of the cause
            logger.exception("proxy failure for trial %s", trial)
            log.write(
                {
                    **record,
                    "author": backend.author.value,
                    "trigger": trigger,
                    "status": 502,
                    "latency_s": None,
                    "error": repr(error)[:2000],
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
