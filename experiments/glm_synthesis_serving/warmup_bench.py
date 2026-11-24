# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Warm-run benchmark for the GLM synthesis pool at the tool-conversion request shape.

The request body mirrors the conversion driver (tool_normalization/model.py): streaming
chat completions with ``cache_salt``, ``reasoning_effort`` via ``chat_template_kwargs``,
``tool_choice: none``, and a strict ``json_schema`` response format. Prompt lengths are
drawn from the measured distribution of 120 real conversion receipts (ortet batches,
November 2026): p10 13.5k / p50 35.8k / p90 72.9k / p99 114.2k tokens. Payloads are
synthetic filler at the measured token sizes — never real trace text.

Run inside an Iris job on the pool cluster (or anywhere the endpoint is reachable)::

    uv run python -m experiments.glm_synthesis_serving.warmup_bench \
        --base-url http://...:port/v1 --concurrency 24 --duration 600

The report prints aggregate output tokens/s/GPU (the brief's headline metric), prompt
tokens/s/GPU, and latency percentiles. Compare runs by holding the engine config fixed
and changing one variable (e.g. MTP on/off via launch_pool --mtp).
"""

import argparse
import asyncio
import json
import logging
import random
import time
from itertools import pairwise

import httpx
from iris.client.client import iris_ctx

from experiments.glm_synthesis_serving import pool

logger = logging.getLogger(__name__)

# Prompt-token distribution from 120 real conversion receipts (empirical percentiles;
# the sampler interpolates between them).
_PROMPT_TOKEN_PCTS = (
    (0, 13_540),
    (25, 23_622),
    (50, 35_798),
    (75, 50_168),
    (90, 72_874),
    (99, 114_199),
    (100, 114_247),
)
# Real completions run p50 832 / p90 2.8k tokens against a 32k reserve; the bench caps
# generation so one stage finishes quickly while exercising the same decode path.
_BENCH_MAX_TOKENS = 2_048
_SYSTEM_PROMPT_TOKENS = 2_000
_CHARS_PER_TOKEN = 4.0
_REQUEST_TIMEOUT_SECONDS = 1800.0
# A modest strict schema: exercises guided decoding without pinning the bench to the
# conversion schema's exact contents.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "argument": {"type": "string"}},
                "required": ["name", "argument"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["actions", "notes"],
    "additionalProperties": False,
}


def _sample_prompt_tokens(rng: random.Random) -> int:
    roll = rng.random() * 100
    for (low_pct, low_tokens), (high_pct, high_tokens) in pairwise(_PROMPT_TOKEN_PCTS):
        if roll <= high_pct:
            fraction = (roll - low_pct) / (high_pct - low_pct) if high_pct > low_pct else 0.0
            return int(low_tokens + fraction * (high_tokens - low_tokens))
    return _PROMPT_TOKEN_PCTS[-1][1]


def _request_body(rng: random.Random, cache_salt: str, reasoning_effort: str, request_index: int) -> dict:
    prompt_tokens = _sample_prompt_tokens(rng)
    filler_tokens = max(64, prompt_tokens - _SYSTEM_PROMPT_TOKENS)
    # Production payloads are unique per row while the system/harness prefix repeats:
    # keep a shared head and pepper a nonce through the tail so prefix caching sees
    # realistic hit rates.
    unit = "tool trace line with json payloads and shell commands "
    needed_chars = int(filler_tokens * _CHARS_PER_TOKEN)
    lines = []
    produced = 0
    while produced < needed_chars:
        chunk = f"row-{request_index}-{rng.randrange(1 << 40):010x} " + unit
        lines.append(chunk)
        produced += len(chunk)
    filler = "".join(lines)[:needed_chars]
    return {
        "model": pool.SERVED_MODEL_NAME,
        "cache_salt": cache_salt,
        "messages": [
            {"role": "system", "content": "You convert tool-harness traces into native tool calls. " * 60},
            {"role": "user", "content": filler},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "thinking_token_budget": 8192,
        "max_tokens": _BENCH_MAX_TOKENS,
        "stream": True,
        "tool_choice": "none",
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort, "clear_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "conversion", "strict": True, "schema": _RESPONSE_SCHEMA},
        },
    }


class RequestStats:
    """Aggregate counters and latency samples for one bench stage."""

    def __init__(self) -> None:
        self.ttft: list[float] = []
        self.itl: list[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = 0
        self.requests = 0
        self.successes = 0

    def summary(self, elapsed_seconds: float, gpus_per_engine: int) -> dict:
        def pct(values: list[float], quantile: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]

        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "elapsed": elapsed_seconds,
            "responses_per_hour": self.successes * 3600 / elapsed_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "output_tok_per_s_per_gpu": (self.completion_tokens / elapsed_seconds) / gpus_per_engine,
            "prompt_tok_per_s_per_gpu": (self.prompt_tokens / elapsed_seconds) / gpus_per_engine,
            "ttft_p50_s": pct(self.ttft, 0.5),
            "ttft_p95_s": pct(self.ttft, 0.95),
            "itl_p50_s": pct(self.itl, 0.5),
            "itl_p95_s": pct(self.itl, 0.95),
        }


async def _one_request(
    client: httpx.AsyncClient, base_url: str, body: dict, started: float, stats: RequestStats
) -> None:
    first_token_at = None
    last_token_at = None
    usage = None
    done = False
    stats.requests += 1
    try:
        async with client.stream(
            "POST", base_url.rstrip("/") + "/chat/completions", json=body, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if not (delta.get("content") or delta.get("reasoning_content")):
                        continue
                    now = time.monotonic()
                    if first_token_at is None:
                        first_token_at = now
                    else:
                        stats.itl.append(now - last_token_at)
                    last_token_at = now
                if event.get("usage"):
                    usage = event["usage"]
        if usage and done and first_token_at is not None:
            stats.successes += 1
            stats.ttft.append(first_token_at - started)
            stats.prompt_tokens += usage.get("prompt_tokens") or 0
            stats.completion_tokens += usage.get("completion_tokens") or 0
        else:
            stats.failures += 1
    except Exception as error:
        stats.failures += 1
        logger.warning("request failed: %s", type(error).__name__)


async def run_load(
    base_url: str,
    concurrency: int,
    duration_seconds: float,
    reasoning_effort: str,
    api_key: str | None = None,
) -> RequestStats:
    """Drive `concurrency` conversion-shaped streaming requests for `duration_seconds`."""
    stats = RequestStats()
    rng = random.Random(1234)
    cache_salt = f"bench-{int(time.time())}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)
    stop_at = time.monotonic() + duration_seconds
    in_flight: set[asyncio.Task] = set()
    request_index = 0

    async with httpx.AsyncClient(limits=limits, headers=headers) as client:
        while time.monotonic() < stop_at or in_flight:
            while len(in_flight) < concurrency and time.monotonic() < stop_at:
                body = _request_body(rng, cache_salt, reasoning_effort, request_index)
                request_index += 1
                task = asyncio.create_task(_one_request(client, base_url, body, time.monotonic(), stats))
                in_flight.add(task)
            if not in_flight:
                break
            done, in_flight = await asyncio.wait(in_flight, timeout=5.0, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is not None:
                    logger.warning("bench task error: %s", task.exception())
    return stats


def normalize_base_url(url: str) -> str:
    """The driver's convention: the OpenAI base URL always ends in /v1."""
    url = url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def resolve_base_url(arg: str | None) -> str:
    if arg:
        return normalize_base_url(arg)
    client = iris_ctx().client
    if client is None:
        raise SystemExit("--base-url is required outside an Iris job")
    if not client.list_endpoint_instances(pool.ENDPOINT_NAME):
        raise SystemExit(f"pool endpoint {pool.ENDPOINT_NAME} is not registered")
    return normalize_base_url(client.resolve_endpoint(pool.ENDPOINT_NAME))


def resolve_api_key(arg: str | None, base_url: str) -> str | None:
    """The --api-key value, or a freshly minted endpoint capability JWT for 'mint'.

    'mint' binds the token to the endpoint the base URL targets, parsed from the
    controller proxy path (the proxy enforces token↔endpoint binding, so a token
    for a different endpoint is 403).
    """
    if arg != "mint":
        return arg
    client = iris_ctx().client
    if client is None:
        raise SystemExit("--api-key mint requires an Iris job")
    if "/proxy/" in base_url:
        after = base_url.split("/proxy/", 1)[1].split("/")[0]
        endpoint = "/" + after.replace(".", "/")
    else:
        endpoint = pool.ENDPOINT_NAME
    return client.mint_endpoint_token(endpoint).token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None, help="OpenAI base URL (default: resolve the pool endpoint).")
    parser.add_argument(
        "--concurrency", type=int, default=24, help="Concurrent requests (pool steady state: 24/engine)."
    )
    parser.add_argument("--duration", type=float, default=600.0, help="Load duration in seconds.")
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "high", "max"])
    parser.add_argument("--gpus-per-engine", type=int, default=pool.GPUS_PER_ENGINE)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Authorization: Bearer value for every request (e.g. the endpoint's capability JWT).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[warmup-bench] %(message)s")
    base_url = resolve_base_url(args.base_url)
    api_key = resolve_api_key(args.api_key, base_url)
    logger.info(
        "bench %s concurrency=%d duration=%.0fs effort=%s auth=%s",
        base_url,
        args.concurrency,
        args.duration,
        args.reasoning_effort,
        "bearer-jwt" if api_key else "none",
    )
    started = time.monotonic()
    stats = asyncio.run(run_load(base_url, args.concurrency, args.duration, args.reasoning_effort, api_key))
    elapsed = time.monotonic() - started
    summary = stats.summary(elapsed, args.gpus_per_engine)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
