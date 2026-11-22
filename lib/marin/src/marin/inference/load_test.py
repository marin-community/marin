# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream a reproducible closed-loop workload and measure the serving window."""

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean

import httpx
import numpy as np
from rigging.telemetry.prometheus import PrometheusScraper

RETAINED_METRICS = frozenset(
    {
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:kv_cache_config_info",
        "vllm:num_preemptions_total",
    }
)


@dataclass(frozen=True)
class LoadRequest:
    identity: str
    path: str
    body: dict


@dataclass(frozen=True)
class RequestMeasurement:
    identity: str
    request_sha256: str
    started: float
    first_token: float | None
    last_token: float | None
    ended: float
    prompt_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    error: str | None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token is None else self.first_token - self.started

    @property
    def tpot(self) -> float | None:
        if self.first_token is None or self.last_token is None or self.output_tokens is None or self.output_tokens < 2:
            return None
        return (self.last_token - self.first_token) / (self.output_tokens - 1)


@dataclass(frozen=True)
class LoadConfig:
    concurrency: int
    warmup_seconds: float
    measurement_seconds: float
    request_timeout_seconds: float
    gpu_count: int
    seed: int
    metrics_interval_seconds: float = 2.0


def request_digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _has_output(choice: dict) -> bool:
    if choice.get("text"):
        return True
    delta = choice.get("delta", {})
    return any(delta.get(key) for key in ("content", "reasoning_content", "reasoning", "tool_calls"))


async def measure_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    request: LoadRequest,
    *,
    seed: int,
    clock: Callable[[], float] = time.perf_counter,
    on_event: Callable[[dict], None] | None = None,
) -> RequestMeasurement:
    """Measure one streaming request without treating SSE chunks as single tokens."""
    body = {
        **request.body,
        "model": model,
        "n": 1,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = clock()
    first = last = None
    prompt_tokens = output_tokens = None
    finish_reason = error = None
    done = False
    try:
        async with client.stream("POST", f"{base_url.rstrip('/')}/{request.path.lstrip('/')}", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise ValueError(f"Serving stream error: {event['error']}")
                for choice in event.get("choices", []):
                    if _has_output(choice):
                        last = clock()
                        if first is None:
                            first = last
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                if on_event is not None:
                    on_event(event)
                usage = event.get("usage")
                if usage is not None:
                    prompt_tokens, output_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        if not done or prompt_tokens is None or output_tokens is None:
            raise ValueError("Incomplete stream: final usage and [DONE] are required")
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        # Failed attempts remain explicit rows and invalidate the aggregate success flag.
        error = f"{type(exc).__name__}: {exc}"
    return RequestMeasurement(
        request.identity,
        request_digest(body),
        started,
        first,
        last,
        clock(),
        prompt_tokens,
        output_tokens,
        finish_reason,
        error,
    )


def _distribution(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    return {"mean": mean(values), **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)}}


def summarize_requests(rows: Sequence[RequestMeasurement], start: float, end: float) -> dict:
    """Summarize requests admitted in the window, keeping drain outside the steady-window rate."""
    selected = [row for row in rows if start <= row.started < end]
    overlapping = [row for row in rows if row.started < end and row.ended >= start]
    successful = [row for row in selected if row.error is None]
    completed = [row for row in successful if row.ended <= end]
    elapsed = max([end, *(row.ended for row in selected)]) - start
    tpots = [row.tpot for row in successful if row.tpot is not None]
    ttfts = [row.ttft for row in successful if row.ttft is not None]
    return {
        "admitted": len(selected),
        "succeeded": len(successful),
        "failed": len(selected) - len(successful),
        "overlapping_succeeded": sum(row.error is None for row in overlapping),
        "overlapping_failed": sum(row.error is not None for row in overlapping),
        "completed_in_window": len(completed),
        "drain_seconds": elapsed - (end - start),
        "request_output_tokens_per_second_including_drain": sum(row.output_tokens or 0 for row in successful) / elapsed,
        "ttft_seconds": _distribution(ttfts),
        "tpot_seconds": _distribution(tpots),
        "request_decode_tokens_per_second": _distribution([1 / value for value in tpots if value > 0]),
        "request_prompt_tokens": _distribution(
            [float(row.prompt_tokens) for row in successful if row.prompt_tokens is not None]
        ),
    }


def _counter_delta(before: list[dict], after: list[dict], name: str) -> float | None:
    def series(samples):
        return {tuple(sorted(row["labels"].items())): row["value"] for row in samples if row["name"] == name}

    initial, final = series(before), series(after)
    if not initial or set(initial) != set(final):
        return None
    deltas = [final[key] - initial[key] for key in initial]
    return sum(deltas) if all(value >= 0 for value in deltas) else None


def summarize_metrics(samples: list[dict], gpu_count: int) -> dict:
    """Use server counters for steady-window rates; missing/reset counters stay unavailable."""
    first, last = samples[0], samples[-1]
    elapsed = last["time"] - first["time"]
    rates = {}
    for label, name in (("output", "vllm:generation_tokens_total"), ("input", "vllm:prompt_tokens_total")):
        delta = _counter_delta(first["series"], last["series"], name)
        rate = None if delta is None or elapsed <= 0 else delta / elapsed
        rates[f"{label}_tokens_per_second"] = rate
        rates[f"{label}_tokens_per_second_per_gpu"] = None if rate is None else rate / gpu_count
    gauges = {}
    for name in (
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    ):
        values = [row["value"] for sample in samples for row in sample["series"] if row["name"] == name]
        gauges[name] = (
            None if not values else {"max_rank_value": max(values), "rank_sample_distribution": _distribution(values)}
        )
    wait_sum = _counter_delta(first["series"], last["series"], "vllm:request_queue_time_seconds_sum")
    wait_count = _counter_delta(first["series"], last["series"], "vllm:request_queue_time_seconds_count")
    prefix_queries = _counter_delta(first["series"], last["series"], "vllm:prefix_cache_queries_total")
    prefix_hits = _counter_delta(first["series"], last["series"], "vllm:prefix_cache_hits_total")
    return {
        "counter_window_seconds": elapsed,
        **rates,
        "gauges": gauges,
        "prefix_hit_fraction": None if prefix_hits is None or not prefix_queries else prefix_hits / prefix_queries,
        "kv_cache_configuration": [row for row in first["series"] if row["name"] == "vllm:kv_cache_config_info"],
        "mean_queue_seconds": None if wait_sum is None or not wait_count else wait_sum / wait_count,
        "preemptions": _counter_delta(first["series"], last["series"], "vllm:num_preemptions_total"),
    }


async def run_load(
    base_url: str,
    metrics_url: str,
    model: str,
    requests: Sequence[LoadRequest],
    config: LoadConfig,
    *,
    headers: dict[str, str] | None = None,
) -> dict:
    """Keep clients active through warmup and measurement; retain request and metric evidence."""
    if (
        not requests
        or config.concurrency < 1
        or min(config.measurement_seconds, config.gpu_count, config.metrics_interval_seconds) <= 0
        or config.warmup_seconds < 0
    ):
        raise ValueError("A load run requires requests, positive load/window/GPU counts and nonnegative warmup")
    order = list(requests)
    random.Random(config.seed).shuffle(order)
    rows: list[RequestMeasurement] = []
    samples: list[dict] = []
    failed = asyncio.Event()
    scraper = PrometheusScraper(metrics_url)
    start = time.perf_counter()
    measurement_start = start + config.warmup_seconds
    deadline = measurement_start + config.measurement_seconds

    async def observe() -> None:
        await asyncio.sleep(config.warmup_seconds)
        while True:
            families = await asyncio.to_thread(scraper.scrape)
            samples.append(
                {
                    "time": time.perf_counter(),
                    "series": [
                        {"name": sample.name, "labels": sample.labels, "value": float(sample.value)}
                        for family in families
                        for sample in family.samples
                        if sample.name in RETAINED_METRICS
                    ],
                }
            )
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            await asyncio.sleep(min(config.metrics_interval_seconds, remaining))

    async with httpx.AsyncClient(
        timeout=config.request_timeout_seconds,
        headers=headers,
        limits=httpx.Limits(max_connections=config.concurrency, max_keepalive_connections=config.concurrency),
    ) as client:

        async def drive(rank: int) -> None:
            index = rank
            while time.perf_counter() < deadline and not failed.is_set():
                request = order[index % len(order)]
                row = await measure_request(client, base_url, model, request, seed=config.seed + index)
                rows.append(row)
                if row.error is not None:
                    failed.set()
                index += config.concurrency

        await asyncio.gather(observe(), *(drive(rank) for rank in range(config.concurrency)))
    server = summarize_metrics(samples, config.gpu_count)
    summary = summarize_requests(rows, measurement_start, deadline)
    return {
        "config": asdict(config),
        "model": model,
        "request_inventory_sha256": request_digest(
            {str(index): {"id": item.identity, "path": item.path, "body": item.body} for index, item in enumerate(order)}
        ),
        "summary": summary,
        "server": server,
        "valid": (
            summary["overlapping_failed"] == 0
            and server["output_tokens_per_second_per_gpu"] is not None
            and summary["overlapping_succeeded"] > 0
        ),
        "requests": [asdict(row) for row in rows],
        "metrics": samples,
        "notes": [
            "Input counter rate includes prefix-cache hits and is not isolated prefill compute throughput.",
            "SSE chunks may contain multiple tokens; TPOT uses final usage counts and output event times.",
            "Client metrics select requests admitted during measurement; rates include their drain.",
            "Server counter window defines the steady throughput; gauge summaries retain rank labels in raw samples.",
        ],
    }
