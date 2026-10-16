# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible proxy between Harbor agents and the models of a #207 generation run.

Plain runs have one backend, GLM 5.3 (the teacher). Relay runs add the student (v0) and route each
request with ``relay.route``. For every chat completion the proxy:

- adds the backend's credentials (the GLM token never reaches an agent or sandbox);
- writes one ``requests.jsonl`` record: trial id (``x-ot-trial-id``), author, relay trigger, the
  request settings exactly as sent, the messages new since the trial's previous request, usage,
  latency and the full reply (reasoning included). This log is the canonical trace; see
  ``rebuild_traces.py``.

It also serves vLLM's ``/tokenize`` (``count``, ``max_model_len``), which Harbor's mini-swe-agent
context limiter needs and the GLM relay does not provide. In relay runs the count is the larger of
the GLM and v0 tokenizations, so the window holds for whichever model answers.
"""

import asyncio
import json
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

TRIAL_HEADER = "x-ot-trial-id"
UNLOGGED_REQUEST_KEYS = frozenset({"messages", "tools"})


@dataclass(frozen=True)
class Backend:
    author: Author
    base_url: str
    model: str
    tokenizer: PreTrainedTokenizerBase
    headers: dict[str, str]


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


def build_proxy(
    teacher: Backend,
    max_model_len: int,
    log: RequestLog,
    relay: RelayPolicy | None = None,
    client: httpx.AsyncClient | None = None,
) -> Starlette:
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=30))
    states: dict[str, TrialState] = defaultdict(TrialState)
    seen_messages: dict[str, int] = defaultdict(int)

    async def counts(body: dict) -> dict[Author, int]:
        backends = [teacher] + ([relay.student] if relay else [])
        values = await asyncio.gather(*(asyncio.to_thread(count_tokens, b.tokenizer, body) for b in backends))
        return {b.author: value for b, value in zip(backends, values, strict=True)}

    async def call(backend: Backend, body: dict) -> tuple[httpx.Response, float]:
        started = time.monotonic()
        response = await client.post(
            f"{backend.base_url}/chat/completions", json={**body, "model": backend.model}, headers=backend.headers
        )
        return response, time.monotonic() - started

    def turn_record(backend: Backend, trigger: Trigger | None, response: httpx.Response, latency: float) -> dict:
        record: dict = {"author": backend.author.value, "trigger": trigger, "status": response.status_code, "latency_s": latency}
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
        record: dict = {
            "ts": time.time(),
            "trial": trial,
            "settings": {k: v for k, v in body.items() if k not in UNLOGGED_REQUEST_KEYS},
            "n_messages": len(messages),
            "first_message": seen_messages[trial] if trial else 0,
            "messages": messages[seen_messages[trial] :] if trial else messages,
            "tools": body.get("tools") if not trial or seen_messages[trial] == 0 else None,
        }
        if trial:
            seen_messages[trial] = len(messages)
        tokens = await counts(body)
        record["prompt_tokens_local"] = {author.value: value for author, value in tokens.items()}

        backend, trigger = teacher, None
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
        log.write({**record, **turn})
        return Response(response.content, response.status_code, media_type=response.headers.get("content-type"))

    return Starlette(
        routes=[
            Route("/tokenize", tokenize, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat, methods=["POST"]),
        ]
    )
