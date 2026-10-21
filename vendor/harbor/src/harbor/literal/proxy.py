"""Recording proxy for literal token-level trace capture.

The proxy forwards OpenAI-compatible chat/completions requests to an
upstream vLLM server while injecting the parameters that make the server
return literal token IDs and logprobs. Each forwarded request/response
pair is appended to a JSONL log so the literal data can be correlated
with agent trajectories after the run.

Two pure helpers (``inject_literal_params`` and
``extract_literal_from_response``) carry the testable logic; the
FastAPI app in :class:`RecordProxy` is a thin transport wrapper around
them.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from litellm import stream_chunk_builder
from litellm.types.utils import ModelResponse, ModelResponseStream

from harbor.literal.native_api import CORRELATION_ID_HEADER, NativeAPILimits
from harbor.literal.native_routes import register_native_routes

logger = logging.getLogger(__name__)

# SSE framing: each event line is ``data: <payload>``; the stream terminates
# with a ``data: [DONE]`` sentinel that carries no JSON.
_SSE_DATA_PREFIX = "data:"
_SSE_DONE_SENTINEL = "[DONE]"

# vLLM exposes ``return_token_ids`` as a top-level field on the chat
# completion request body; the completion token IDs come back in
# ``choice.provider_specific_fields.token_ids`` and the full prompt
# token IDs in the top-level ``prompt_token_ids`` response field.
_RETURN_TOKEN_IDS_KEY = "return_token_ids"

# Per-trial correlation id request header. A CLI agent (opencode) run behind this
# proxy for an RL rollout stamps a unique id into every completion request (via
# its provider ``options.headers``); the proxy records it on each JSONL entry so a
# single shared controller-level log can be split back into per-trial rollouts by
# id — the concurrency-safe alternative to content correlation (immune to GRPO
# identical-seed ambiguity). Absent (datagen / non-RL) → recorded as ``None`` and
# ignored, so those paths are unaffected.
TASK_AGENT_HEADER = "x-ot-task-agent"
TOKENIZE_ENDPOINT = "/tokenize"
LITERAL_LOG_FILENAME = "literal.jsonl"


def inject_literal_params(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *body* enriched to request literal token data.

    Sets ``logprobs=True`` and ``top_logprobs=1`` so the upstream returns
    per-token logprobs for the sampled sequence, and ``return_token_ids=True``
    so vLLM echoes both prompt and completion token IDs.

    Existing values are preserved — the proxy only fills in defaults when
    the caller did not set them, so an agent that already requests logprobs
    is not downgraded.
    """
    enriched = dict(body)
    enriched.setdefault("logprobs", True)
    # ``top_logprobs`` is only meaningful when ``logprobs`` is true; vLLM
    # requires it to be a non-negative integer. Default to 1 (the sampled
    # token's logprob is always included regardless) unless the caller asked
    # for more. NB: ``top_logprobs=0`` crashes the vLLM v0.20.0 tpu-inference
    # build with an IndexError in ``_create_chat_logprobs`` (empty top-logprobs
    # list), 500-ing every literal-capture request; ``1`` dodges that while the
    # literal capture only ever needs the sampled token's logprob.
    if enriched.get("logprobs") and "top_logprobs" not in enriched:
        enriched["top_logprobs"] = 1
    enriched.setdefault(_RETURN_TOKEN_IDS_KEY, True)
    return enriched


def _extract_completion_token_ids(choice: dict[str, Any]) -> list[int] | None:
    """Pull completion token IDs from a vLLM choice object.

    vLLM places them under ``provider_specific_fields.token_ids``; some
    builds surface them directly as ``token_ids`` on the choice.
    """
    psf = choice.get("provider_specific_fields")
    if isinstance(psf, dict):
        ids = psf.get("token_ids")
        if isinstance(ids, list):
            return ids
    ids = choice.get("token_ids")
    if isinstance(ids, list):
        return ids
    return None


def _extract_logprobs(choice: dict[str, Any]) -> list[float] | None:
    """Pull per-token logprobs from a vLLM choice object."""
    lp = choice.get("logprobs")
    if not isinstance(lp, dict):
        return None
    token_logprobs = lp.get("token_logprobs")
    if isinstance(token_logprobs, list) and all(
        isinstance(value, int | float) for value in token_logprobs
    ):
        return [float(value) for value in token_logprobs]
    content = lp.get("content")
    if not isinstance(content, list):
        return None
    out: list[float] = []
    for entry in content:
        if isinstance(entry, dict) and "logprob" in entry:
            out.append(entry["logprob"])
    return out if out else None


def extract_literal_from_response(resp_json: dict[str, Any]) -> dict[str, Any]:
    """Extract literal token data from a vLLM chat or text completion response.

    Returns a dict with keys ``prompt_token_ids``, ``completion_token_ids``,
    ``logprobs`` (any may be ``None`` if the upstream did not populate them),
    plus ``model`` and ``response_id`` for correlation.
    """
    choices = resp_json.get("choices") or []
    first = choices[0] if choices else {}
    completion_ids = _extract_completion_token_ids(first) if first else None
    logprobs = _extract_logprobs(first) if first else None
    prompt_ids = resp_json.get("prompt_token_ids")
    if not isinstance(prompt_ids, list):
        prompt_ids = first.get("prompt_token_ids")
    if not isinstance(prompt_ids, list):
        prompt_ids = None
    return {
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "logprobs": logprobs,
        "model": resp_json.get("model"),
        "response_id": resp_json.get("id"),
        "assistant_message": first.get("message"),
    }


def _new_stream_accumulator() -> dict[str, Any]:
    """Fresh state for accumulating literal token data across SSE chunks."""
    return {
        "prompt_token_ids": None,
        "completion_token_ids": [],
        "logprobs": [],
        "model": None,
        "response_id": None,
        "chunks": [],
    }


def _chunk_completion_token_ids(choice: dict[str, Any]) -> list[int] | None:
    """Pull the completion token IDs carried on a single streaming choice.

    vLLM's ``return_token_ids`` surfaces per-chunk completion token IDs under the
    delta's ``provider_specific_fields.token_ids`` (some builds place them on the
    choice directly). Any of the known locations is accepted.
    """
    delta = choice.get("delta")
    if isinstance(delta, dict):
        psf = delta.get("provider_specific_fields")
        if isinstance(psf, dict) and isinstance(psf.get("token_ids"), list):
            return psf["token_ids"]
        if isinstance(delta.get("token_ids"), list):
            return delta["token_ids"]
    return _extract_completion_token_ids(choice)


def accumulate_stream_chunk(state: dict[str, Any], chunk: dict[str, Any]) -> None:
    """Fold one parsed SSE chunk into the running literal accumulator *in place*.

    Concatenates per-chunk completion token IDs and logprobs (in arrival order),
    captures the first prompt_token_ids seen, and tracks model / response id so
    the finalized record matches the non-streaming :func:`extract_literal_from_response`
    shape.
    """
    if chunk.get("choices"):
        state["chunks"].append(
            ModelResponseStream(
                id=chunk.get("id"),
                model=chunk.get("model"),
                choices=[
                    {
                        key: value
                        for key, value in choice.items()
                        if key in ("index", "delta", "finish_reason")
                    }
                    for choice in chunk["choices"]
                ],
            )
        )
    if state["prompt_token_ids"] is None:
        prompt_ids = chunk.get("prompt_token_ids")
        if isinstance(prompt_ids, list):
            state["prompt_token_ids"] = prompt_ids

    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        if state["prompt_token_ids"] is None and isinstance(
            choice.get("prompt_token_ids"), list
        ):
            state["prompt_token_ids"] = choice["prompt_token_ids"]
        ids = _chunk_completion_token_ids(choice)
        if ids:
            state["completion_token_ids"].extend(ids)
        lp = _extract_logprobs(choice)
        if lp:
            state["logprobs"].extend(lp)

    if chunk.get("model"):
        state["model"] = chunk["model"]
    if chunk.get("id"):
        state["response_id"] = chunk["id"]


def finalize_stream_literal(state: dict[str, Any]) -> dict[str, Any]:
    """Convert a stream accumulator into a literal record dict.

    Empty accumulations collapse to ``None`` so the record mirrors the
    non-streaming path (where absent fields are ``None``, not ``[]``).
    """
    response = stream_chunk_builder(state["chunks"]) if state["chunks"] else None
    message = (
        response.choices[0].message.model_dump(exclude_none=True)
        if isinstance(response, ModelResponse) and response.choices
        else None
    )
    return {
        "prompt_token_ids": state["prompt_token_ids"],
        "completion_token_ids": state["completion_token_ids"] or None,
        "logprobs": state["logprobs"] or None,
        "model": state["model"],
        "response_id": state["response_id"],
        "assistant_message": message,
    }


class RecordProxy:
    """A recording reverse-proxy in front of an OpenAI-compatible endpoint.

    Forwards requests to *upstream_base_url*, injecting literal-token
    request params, and appends a JSONL record of each exchange to
    *log_path*. The records are designed to be merged back into an
    ATIF trajectory's ``step.metrics`` after a run.

    The FastAPI/Starlette app is built lazily via :meth:`app` so importing
    this module never pulls in the server stack.
    """

    def __init__(
        self,
        upstream_base_url: str,
        log_path: str | Path,
        *,
        timeout: float = 600.0,
    ) -> None:
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.log_path = Path(log_path)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Release upstream connections after the proxy stops accepting requests."""
        if self._client is not None:
            await self._client.aclose()

    def _write_entry(
        self,
        *,
        request_body: dict[str, Any],
        literal: dict[str, Any] | None,
        status_code: int,
        elapsed_ms: float,
        trial_id: str | None = None,
    ) -> None:
        """Append one exchange record to the literal JSONL log.

        ``trial_id`` is the per-trial correlation id from the
        :data:`CORRELATION_ID_HEADER` request header (``None`` when absent, e.g.
        datagen / non-RL). Always written so a reader can split a shared log by
        trial; readers that don't correlate by id simply ignore the field.
        """
        entry = {
            "timestamp": time.time(),
            "status_code": status_code,
            "elapsed_ms": elapsed_ms,
            "trial_id": trial_id,
            "request": request_body,
            "literal": literal,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _record(
        self,
        *,
        request_body: dict[str, Any],
        response_json: dict[str, Any] | None,
        status_code: int,
        elapsed_ms: float,
        trial_id: str | None = None,
    ) -> None:
        literal = (
            extract_literal_from_response(response_json)
            if response_json is not None
            else None
        )
        self._write_entry(
            request_body=request_body,
            literal=literal,
            status_code=status_code,
            elapsed_ms=elapsed_ms,
            trial_id=trial_id,
        )

    async def forward(
        self,
        path: str,
        body: dict[str, Any],
        method: str = "POST",
        *,
        trial_id: str | None = None,
        task_agent: str | None = None,
    ) -> tuple[int, dict[str, Any] | None, float]:
        """Inject, forward, and record one request.

        Returns ``(status_code, response_json_or_None, elapsed_ms)``.
        Non-JSON upstream responses yield ``response_json=None`` but are
        still recorded (with ``literal=None``) so the log is complete.
        ``trial_id`` (from :data:`CORRELATION_ID_HEADER`) is stamped on the record
        and forwarded upstream under the same header.
        """
        enriched = body if path == TOKENIZE_ENDPOINT else inject_literal_params(body)
        url = f"{self.upstream_base_url}{path}"
        client = await self._get_client()
        start = time.monotonic()
        headers = {CORRELATION_ID_HEADER: trial_id} if trial_id is not None else {}
        if task_agent is not None:
            headers[TASK_AGENT_HEADER] = task_agent
        resp = await client.request(method, url, json=enriched, headers=headers)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        try:
            resp_json: dict[str, Any] | None = resp.json()
        except Exception:
            resp_json = None
        if path != TOKENIZE_ENDPOINT:
            self._record(
                request_body=enriched,
                response_json=resp_json,
                status_code=resp.status_code,
                elapsed_ms=elapsed_ms,
                trial_id=trial_id,
            )
        return resp.status_code, resp_json, elapsed_ms

    async def open_upstream_stream(
        self,
        path: str,
        body: dict[str, Any],
        method: str = "POST",
        *,
        trial_id: str | None = None,
        task_agent: str | None = None,
    ) -> tuple[dict[str, Any], httpx.Response]:
        """Inject params and open a streaming upstream request.

        Returns ``(enriched_request_body, streaming_response)``. The caller MUST
        drain the response via :meth:`relay_and_record` (which closes it). The
        request keeps ``stream=True`` so vLLM emits an ``text/event-stream`` with
        per-chunk token IDs / logprobs. When provided, ``trial_id`` is forwarded
        upstream as :data:`CORRELATION_ID_HEADER`.
        """
        enriched = inject_literal_params(body)
        url = f"{self.upstream_base_url}{path}"
        client = await self._get_client()
        headers = {CORRELATION_ID_HEADER: trial_id} if trial_id is not None else {}
        if task_agent is not None:
            headers[TASK_AGENT_HEADER] = task_agent
        request = client.build_request(method, url, json=enriched, headers=headers)
        resp = await client.send(request, stream=True)
        return enriched, resp

    async def relay_and_record(
        self,
        enriched: dict[str, Any],
        resp: httpx.Response,
        *,
        trial_id: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Relay an upstream SSE stream byte-for-byte while capturing literal data.

        Yields the upstream's ``text/event-stream`` bytes unchanged (so the
        client's OpenAI-compatible SSE parser sees a byte-correct stream, tool-call
        deltas included) and, in parallel, parses each ``data:`` chunk to
        accumulate completion token IDs / logprobs. On stream end the accumulated
        literal record is appended to the JSONL log, matching the non-streaming
        path. A non-200 upstream is relayed verbatim and recorded with
        ``literal=None``.
        """
        start = time.monotonic()
        state = _new_stream_accumulator()
        ok = resp.status_code == 200
        # Buffer at the byte level (not str): a multi-byte UTF-8 char can be split
        # across two byte-chunks, and per-chunk decoding would corrupt it. SSE
        # lines end at a single-byte '\n', so each complete line is valid UTF-8.
        buffer = b""
        recorded = False
        try:
            async for chunk in resp.aiter_bytes():
                if ok:
                    buffer += chunk
                    buffer = self._consume_sse_buffer(buffer, state)
                    if state.get("stream_done") and not recorded:
                        self._write_entry(
                            request_body=enriched,
                            literal=finalize_stream_literal(state),
                            status_code=resp.status_code,
                            elapsed_ms=(time.monotonic() - start) * 1000.0,
                            trial_id=trial_id,
                        )
                        recorded = True
                yield chunk
        finally:
            await resp.aclose()
        if recorded:
            return
        elapsed_ms = (time.monotonic() - start) * 1000.0
        literal = finalize_stream_literal(state) if ok else None
        self._write_entry(
            request_body=enriched,
            literal=literal,
            status_code=resp.status_code,
            elapsed_ms=elapsed_ms,
            trial_id=trial_id,
        )

    @staticmethod
    def _consume_sse_buffer(buffer: bytes, state: dict[str, Any]) -> bytes:
        """Parse complete SSE lines out of *buffer*, folding them into *state*.

        Splits on the single-byte ``\\n`` delimiter and decodes each complete
        line (valid UTF-8). Returns the unconsumed remainder (a partial trailing
        line) so the next byte chunk can complete it.
        """
        while b"\n" in buffer:
            raw_line, buffer = buffer.split(b"\n", 1)
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith(_SSE_DATA_PREFIX):
                continue
            payload = line[len(_SSE_DATA_PREFIX) :].strip()
            if payload == _SSE_DONE_SENTINEL:
                state["stream_done"] = True
                continue
            if not payload:
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            accumulate_stream_chunk(state, chunk)
        return buffer

    def app(
        self, native_limits: NativeAPILimits | None = None
    ):  # pragma: no cover - exercised via integration tests
        """Build and return the FastAPI app (server stack imported lazily)."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse

        api = FastAPI(title="harbor-literal-proxy")

        async def _proxy(request: Request):
            try:
                body = await request.json()
            except Exception:
                body = {}
            path = request.url.path
            # Per-trial correlation id (RL rollout): the agent stamps it into every
            # completion request header so a shared log splits back per trial.
            trial_id = request.headers.get(CORRELATION_ID_HEADER)
            task_agent = request.headers.get(TASK_AGENT_HEADER)
            # Streaming clients (opencode always sends stream=true) get a tee'd
            # SSE relay: the native text/event-stream is passed through unchanged
            # while the proxy accumulates the literal record from each chunk.
            if body.get("stream"):
                enriched, resp = await self.open_upstream_stream(
                    path, body, request.method, trial_id=trial_id, task_agent=task_agent
                )
                return StreamingResponse(
                    self.relay_and_record(enriched, resp, trial_id=trial_id),
                    status_code=resp.status_code,
                    media_type="text/event-stream",
                )
            status, resp_json, _ = await self.forward(
                path, body, request.method, trial_id=trial_id, task_agent=task_agent
            )
            if resp_json is None:
                return JSONResponse(
                    {"error": "upstream returned non-JSON"}, status_code=502
                )
            return JSONResponse(resp_json, status_code=status)

        # Cover both chat and text completions; vLLM serves both under /v1.
        # Use Starlette's add_route (not FastAPI's add_api_route): this module sets
        # ``from __future__ import annotations`` and imports ``Request`` LOCALLY here,
        # so FastAPI's parameter introspection cannot resolve the stringized
        # ``request: Request`` annotation from the module globals and would 422 every
        # request (treating ``request`` as a required query field). add_route passes
        # the Request positionally with no introspection, so the proxy forwards cleanly.
        api.add_route("/v1/chat/completions", _proxy, methods=["POST"])
        api.add_route("/v1/completions", _proxy, methods=["POST"])
        api.add_route(TOKENIZE_ENDPOINT, _proxy, methods=["POST"])
        if native_limits is not None:
            register_native_routes(api, self, native_limits)
        return api
