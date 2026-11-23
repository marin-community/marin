# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure prefetched Mochi streams without request-wave admission bias."""

import asyncio
import random
import time
from dataclasses import asdict, dataclass, field, replace
from statistics import median
from typing import Protocol

import httpx
from marin.evaluation.inference_metrics import InferenceMetricWindow
from marin.inference.load_test import (
    RETAINED_METRICS,
    LoadRequest,
    RequestMeasurement,
    _distribution,
    measure_request,
    request_digest,
    summarize_metrics,
)
from rigging.telemetry.prometheus import PrometheusScraper

from experiments.post_training.long_loop.context_cohort import ResponseCapture

STREAM_ID_PROBE_TOKENS = 8


class DecodeEndpoint(Protocol):
    """OpenAI serving endpoint and its Prometheus metrics."""

    @property
    def server_url(self) -> str: ...

    @property
    def metrics_url(self) -> str: ...


@dataclass(frozen=True)
class DecodeWindowConfig:
    data_parallel_size: int
    running_per_engine: int
    gpu_count: int
    seed: int
    max_new_tokens: int
    first_token_timeout: float
    settle: float
    measurement: float
    metrics_interval: float


@dataclass
class DecodeTrace:
    request: LoadRequest
    rank: int
    seed: int
    started: float = 0
    capture: ResponseCapture = field(default_factory=ResponseCapture)
    emissions: list[tuple[float, int]] = field(default_factory=list)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    completed: RequestMeasurement | None = None

    def add(self, event: dict) -> None:
        before = len(self.capture.token_ids)
        self.capture.add(event)
        emitted = len(self.capture.token_ids) - before
        if emitted:
            self.emissions.append((time.perf_counter(), emitted))
            self.ready.set()


def request_window(trace: DecodeTrace, start: float, end: float) -> dict:
    """Count exact token arrivals in one common measurement interval."""
    events = [(timestamp, count) for timestamp, count in trace.emissions if start <= timestamp < end]
    count = sum(count for _, count in events)
    if not events or count < 1 or end <= start:
        raise ValueError("A ready stream did not emit exact token IDs during the measurement window")
    if trace.emissions[0][0] >= start:
        raise ValueError("A stream was not prefetched before the measurement window")
    return {
        "identity": trace.request.identity,
        "requested_data_parallel_rank": trace.rank,
        "seed": trace.seed,
        "input_body_sha256": request_digest(trace.request.body),
        "initial_cohort_ttft_seconds": trace.emissions[0][0] - trace.started,
        "window_start": start,
        "window_end": end,
        "window_first_emission": events[0][0],
        "window_last_emission": events[-1][0],
        "window_tokens": count,
        "window_chunks": len(events),
        "decode_tokens_per_second": count / (end - start),
        "tpot_seconds": (end - start) / count,
        "tokens_before_window": sum(count for timestamp, count in trace.emissions if timestamp < start),
        "intentionally_censored": trace.completed is None,
    }


async def steady_decode_window(
    environment: DecodeEndpoint, requests: list[LoadRequest], config: DecodeWindowConfig, *, speculative: bool
) -> dict:
    """Measure exact token arrivals after all rank-pinned streams have prefetched."""
    if (
        not requests
        or min(config.data_parallel_size, config.running_per_engine, config.gpu_count, config.max_new_tokens) < 1
    ):
        raise ValueError("A stream window requires a positive cohort geometry and token limit")
    if min(config.first_token_timeout, config.measurement, config.metrics_interval) <= 0 or config.settle < 0:
        raise ValueError("A stream window requires positive deadlines and a nonnegative settle period")
    concurrency = config.data_parallel_size * config.running_per_engine
    order = list(requests)
    random.Random(config.seed).shuffle(order)
    traces = [
        DecodeTrace(
            replace(
                order[slot % len(order)],
                identity=f"{order[slot % len(order)].identity}/rank{rank}/slot{slot}",
                body={
                    **order[slot % len(order)].body,
                    "max_tokens": config.max_new_tokens,
                    "return_token_ids": True,
                    "return_tokens_as_token_ids": True,
                },
            ),
            rank,
            config.seed + slot * config.data_parallel_size + rank,
        )
        for slot in range(config.running_per_engine)
        for rank in range(config.data_parallel_size)
    ]
    samples = []
    scraper = PrometheusScraper(environment.metrics_url)

    def snapshot() -> dict:
        families = scraper.scrape()
        return {
            "time": time.perf_counter(),
            "series": [
                {"name": sample.name, "labels": sample.labels, "value": float(sample.value)}
                for family in families
                for sample in family.samples
                if sample.name in RETAINED_METRICS
            ],
        }

    async with httpx.AsyncClient(
        timeout=config.first_token_timeout,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    ) as client:
        probe = DecodeTrace(
            replace(
                traces[0].request,
                body={**traces[0].request.body, "max_tokens": STREAM_ID_PROBE_TOKENS, "ignore_eos": True},
            ),
            0,
            config.seed,
        )
        probe.started = time.perf_counter()
        qualification = await measure_request(
            client,
            environment.server_url,
            "mochi",
            probe.request,
            seed=probe.seed,
            on_event=probe.add,
            data_parallel_rank=0,
        )
        if (
            qualification.error is not None
            or len(probe.capture.token_ids) != qualification.output_tokens
            or len(probe.emissions) < 2
        ):
            raise ValueError("Exact token IDs were not progressively streamed and reconciled with final usage")

        async def drive(trace: DecodeTrace) -> None:
            trace.started = time.perf_counter()
            trace.completed = await measure_request(
                client,
                environment.server_url,
                "mochi",
                trace.request,
                seed=trace.seed,
                on_event=trace.add,
                data_parallel_rank=trace.rank,
            )

        tasks = [asyncio.create_task(drive(trace)) for trace in traces]
        all_ready = asyncio.gather(*(trace.ready.wait() for trace in traces))
        try:
            done, _ = await asyncio.wait(
                [all_ready, *tasks], timeout=config.first_token_timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if all_ready not in done or any(task.done() for task in tasks):
                raise ValueError("A stream failed or completed before the entire cohort was prefetched")
            await asyncio.sleep(config.settle)
            if any(task.done() for task in tasks):
                raise ValueError("A stream failed or completed during settling")
            # Retain the preemption baseline before starting the counter window.
            samples.append(await asyncio.to_thread(snapshot))
            counter_window = await asyncio.to_thread(InferenceMetricWindow.start, environment, speculative=speculative)
            started = time.perf_counter()
            ended = started + config.measurement
            while True:
                samples.append(await asyncio.to_thread(snapshot))
                remaining = ended - time.perf_counter()
                if remaining <= 0:
                    break
                done, _ = await asyncio.wait(
                    tasks, timeout=min(config.metrics_interval, remaining), return_when=asyncio.FIRST_COMPLETED
                )
                if done:
                    raise ValueError("A stream completed before the fixed measurement window ended")
            counters = (await asyncio.to_thread(counter_window.finish)).model_dump(mode="json")
            if any(trace.completed is not None and trace.completed.ended < ended for trace in traces):
                raise ValueError("A stream completed before the fixed measurement window ended")
            if any(trace.completed is not None and trace.completed.error is not None for trace in traces):
                raise ValueError("A stream failed before cohort cancellation")
            rows = [request_window(trace, started, ended) for trace in traces]
            server = summarize_metrics(samples, config.gpu_count)
            engine_samples: dict[tuple, list[float]] = {}
            occupancy_samples = []
            for sample in samples:
                occupancy = {}
                for row in sample["series"]:
                    if row["name"] == "vllm:num_requests_running":
                        labels = tuple(sorted(row["labels"].items()))
                        engine_samples.setdefault(labels, []).append(row["value"])
                        occupancy[labels] = row["value"]
                occupancy_samples.append(occupancy)
            engine_medians = [
                {"labels": dict(labels), "median": median(values)} for labels, values in engine_samples.items()
            ]
            achieved = len(engine_samples) == config.data_parallel_size and all(
                occupancy.keys() == engine_samples.keys()
                and all(count == config.running_per_engine for count in occupancy.values())
                for occupancy in occupancy_samples
            )
            acceptance = counters["speculative_decoding"]
            proposals_measured = not speculative or (acceptance is not None and acceptance["drafts"] > 0)
            result = {
                "config": asdict(config),
                "protocol": "prefetched_exact_token_stream_window",
                "qualification": {
                    "measurement": asdict(qualification),
                    "token_ids": probe.capture.token_ids,
                    "positive_chunks": len(probe.emissions),
                },
                "request_inventory_sha256": request_digest(
                    {
                        str(index): {"identity": request.identity, "path": request.path, "body": request.body}
                        for index, request in enumerate(order)
                    }
                ),
                "occupancy_achieved": achieved,
                "observed_engine_medians": engine_medians,
                "valid": (
                    achieved
                    and proposals_measured
                    and server["preemptions"] == 0
                    and server["output_tokens_per_second"] is not None
                ),
                "summary": {
                    "request_decode_tokens_per_second": _distribution([row["decode_tokens_per_second"] for row in rows]),
                    "tpot_seconds": _distribution([row["tpot_seconds"] for row in rows]),
                    "ttft_seconds": _distribution([row["initial_cohort_ttft_seconds"] for row in rows]),
                    "intentionally_censored_requests": sum(row["intentionally_censored"] for row in rows),
                },
                "server": server,
                "counter_window": counters,
                "requests": rows,
                "metrics": samples,
                "notes": [
                    "Every stream emits a first exact token before measurement and remains live for the full window.",
                    "Decode rate counts exact token IDs in a common interval, including multi-token chunks.",
                    "Streams are censored after measurement; completed-response usage is unavailable for those streams.",
                    "An eight-token completed probe reconciles progressive token IDs with final usage.",
                    "TTFT describes initial cohort admission; the window has no new prefill traffic.",
                    "Workload sampling and EOS settings are preserved; early completion invalidates the cohort.",
                ],
            }
        finally:
            all_ready.cancel()
            for task in tasks:
                task.cancel()
            outcomes = await asyncio.gather(all_ready, *tasks, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                    raise outcome
    return result
