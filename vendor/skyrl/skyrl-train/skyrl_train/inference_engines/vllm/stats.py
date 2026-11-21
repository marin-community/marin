"""Typed boundary between inference metric producers and trainer observability sinks."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


HTTP_BRIDGE_OUTCOME_METRIC = "request_outcome"
HTTP_BRIDGE_SUMMARY_BOUNDS = {
    "event_loop_lag_seconds": (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    "response_bytes": (1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576),
    "json_serialization_seconds": (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
}
HTTP_BRIDGE_HISTOGRAM_BOUNDS = {
    **HTTP_BRIDGE_SUMMARY_BOUNDS,
    HTTP_BRIDGE_OUTCOME_METRIC: (),  # The exported _count is the outcome counter.
}
HTTP_BRIDGE_SUMMARY_METRIC_NAMES = tuple(HTTP_BRIDGE_SUMMARY_BOUNDS)
VLLM_NUM_ENGINES_METRIC = "vllm/num_engines"
VLLM_FINISH_REASONS = ("stop", "length", "abort", "error", "repetition")
VLLM_HISTOGRAM_UNITS = {
    "request_queue_time_seconds": "s",
    "request_prefill_time_seconds": "s",
    "request_decode_time_seconds": "s",
    "e2e_request_latency_seconds": "s",
    "time_to_first_token_seconds": "s",
    "request_generation_tokens": "{token}",
    "iteration_tokens_total": "{token}",
    "request_time_per_output_token_seconds": "s",
}


class IntervalReadMode(StrEnum):
    PEEK = "peek"
    RESET = "reset"


@dataclass(frozen=True)
class VLLMHistogramSnapshot:
    name: str
    buckets: tuple[tuple[float, float], ...]
    count: float
    total: float
    unit: str
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VLLMCurrentStats:
    running_requests: int = 0
    waiting_capacity: int = 0
    waiting_deferred: int = 0
    kv_cache_usage: float = 0.0


@dataclass(frozen=True)
class VLLMCumulativeStats:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0
    preemptions: int = 0
    spec_decode_drafts: int = 0
    spec_decode_draft_tokens: int = 0
    spec_decode_accepted_tokens: int = 0
    finished_by_reason: Mapping[str, int] = field(default_factory=lambda: {reason: 0 for reason in VLLM_FINISH_REASONS})


@dataclass(frozen=True)
class VLLMIntervalStats:
    peak_prompt_throughput: float = 0.0
    peak_generation_throughput: float = 0.0
    peak_running_reqs: int = 0
    peak_waiting_reqs: int = 0
    peak_gpu_cache_usage_perc: float = 0.0
    peak_prefix_cache_hit_rate: float = 0.0
    median_prompt_throughput: float = 0.0
    median_generation_throughput: float = 0.0
    median_running_reqs: float = 0.0
    median_waiting_reqs: float = 0.0
    median_gpu_cache_usage_perc: float = 0.0
    median_prefix_cache_hit_rate: float = 0.0
    mean_prompt_throughput: float = 0.0
    mean_generation_throughput: float = 0.0
    latency_prefill_mean: float = 0.0
    latency_prefill_median: float = 0.0
    latency_prefill_p90: float = 0.0
    latency_decode_mean: float = 0.0
    latency_decode_median: float = 0.0
    latency_decode_p90: float = 0.0
    latency_e2e_mean: float = 0.0
    latency_e2e_median: float = 0.0
    latency_e2e_p90: float = 0.0
    latency_queued_mean: float = 0.0
    latency_queued_median: float = 0.0
    latency_queued_p90: float = 0.0
    latency_ttft_mean: float = 0.0
    latency_ttft_median: float = 0.0
    latency_ttft_p90: float = 0.0
    finished_requests: int = 0
    preempted_reqs: int = 0
    samples: int = 0
    active_samples: int = 0


@dataclass(frozen=True)
class VLLMEngineStatsSnapshot:
    engine_id: str
    timestamp: float
    current: VLLMCurrentStats
    cumulative: VLLMCumulativeStats
    interval: VLLMIntervalStats
    attributes: Mapping[str, str] = field(default_factory=dict)
    histograms: tuple[VLLMHistogramSnapshot, ...] = ()


@dataclass(frozen=True)
class InferenceStatsSnapshot:
    engines: tuple[VLLMEngineStatsSnapshot, ...]
    http_bridge: HTTPBridgeStatsSnapshot | None = None


@dataclass(frozen=True)
class DistributionSummary:
    count: int = 0
    mean: float = 0.0
    p95: float = 0.0
    maximum: float = 0.0


@dataclass(frozen=True)
class HTTPBridgeStatsSnapshot:
    event_loop_lag_seconds: DistributionSummary = field(default_factory=DistributionSummary)
    response_bytes: DistributionSummary = field(default_factory=DistributionSummary)
    json_serialization_seconds: DistributionSummary = field(default_factory=DistributionSummary)
    histograms: tuple[VLLMHistogramSnapshot, ...] = ()


@dataclass
class _IntervalDistributionAccumulator:
    """Exact count/mean/max with a bounded sample for interval p95."""

    sample_limit: int = 4_096
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        if len(self.samples) < self.sample_limit:
            self.samples.append(value)
        else:
            self.samples[(self.count - 1) % self.sample_limit] = value

    def snapshot(self) -> DistributionSummary:
        if not self.count:
            return DistributionSummary()
        ordered = sorted(self.samples)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return DistributionSummary(self.count, self.total / self.count, ordered[p95_index], self.maximum)


class HTTPBridgeStatsAccumulator:
    """Thread-safe bridge observations with PEEK and RESET interval reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._interval = {name: _IntervalDistributionAccumulator() for name in HTTP_BRIDGE_SUMMARY_METRIC_NAMES}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], HistogramAccumulator] = {}

    def observe(self, name: str, value: float, *, attributes: Mapping[str, str] | None = None) -> None:
        if name not in HTTP_BRIDGE_SUMMARY_METRIC_NAMES:
            raise ValueError(f"unknown HTTP bridge metric: {name}")
        labels = tuple(sorted((attributes or {}).items()))
        with self._lock:
            self._interval[name].observe(value)
            histogram = self._histograms.setdefault(
                (name, labels), HistogramAccumulator(HTTP_BRIDGE_HISTOGRAM_BOUNDS[name])
            )
            histogram.observe(value)

    def record_request_outcome(self, endpoint: str, reason: str) -> None:
        """Count an ASGI request outcome with its endpoint and observed close reason."""
        labels = (("endpoint", endpoint), ("reason", reason))
        with self._lock:
            histogram = self._histograms.setdefault(
                (HTTP_BRIDGE_OUTCOME_METRIC, labels),
                HistogramAccumulator(HTTP_BRIDGE_HISTOGRAM_BOUNDS[HTTP_BRIDGE_OUTCOME_METRIC]),
            )
            histogram.observe(1.0)

    def snapshot(self, read_mode: IntervalReadMode) -> HTTPBridgeStatsSnapshot:
        with self._lock:
            summaries = {name: accumulator.snapshot() for name, accumulator in self._interval.items()}
            histograms = tuple(
                histogram.snapshot(
                    name,
                    "By" if name == "response_bytes" else "count" if name == HTTP_BRIDGE_OUTCOME_METRIC else "s",
                    dict(labels),
                )
                for (name, labels), histogram in self._histograms.items()
            )
            if read_mode is IntervalReadMode.RESET:
                self._interval = {name: _IntervalDistributionAccumulator() for name in HTTP_BRIDGE_SUMMARY_METRIC_NAMES}
        return HTTPBridgeStatsSnapshot(histograms=histograms, **summaries)


@dataclass(frozen=True)
class VLLMNativeStatsSnapshot:
    current: VLLMCurrentStats
    cumulative: VLLMCumulativeStats
    histograms: tuple[VLLMHistogramSnapshot, ...]


@dataclass
class HistogramAccumulator:
    """Process-local cumulative histogram for HTTP bridge observations."""

    bounds: tuple[float, ...]
    count: int = 0
    total: float = 0.0
    bucket_counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.bucket_counts:
            self.bucket_counts = [0] * (len(self.bounds) + 1)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for index, bound in enumerate((*self.bounds, math.inf)):
            if value <= bound:
                self.bucket_counts[index] += 1

    def snapshot(self, name: str, unit: str, attributes: Mapping[str, str]) -> VLLMHistogramSnapshot:
        return VLLMHistogramSnapshot(
            name=name,
            buckets=tuple(zip((*self.bounds, math.inf), self.bucket_counts, strict=True)),
            count=self.count,
            total=self.total,
            unit=unit,
            attributes=attributes,
        )


def snapshot_vllm_prometheus_metrics(metrics: Sequence[Any], engine_index: str) -> VLLMNativeStatsSnapshot:
    """Translate one engine's built-in vLLM Prometheus snapshot to the wire type."""
    values: dict[str, float] = {}
    finished = {reason: 0 for reason in VLLM_FINISH_REASONS}
    histograms: list[VLLMHistogramSnapshot] = []

    for metric in metrics:
        raw_name = str(getattr(metric, "name", ""))
        labels = getattr(metric, "labels", {})
        if not raw_name.startswith("vllm:") or labels.get("engine") != engine_index:
            continue
        name = raw_name.removeprefix("vllm:")
        if name == "request_success":
            reason = labels.get("finished_reason")
            if reason in finished:
                finished[reason] = int(metric.value)
        elif name in VLLM_HISTOGRAM_UNITS and hasattr(metric, "buckets"):
            attributes = {"engine_index": engine_index}
            if model_name := labels.get("model_name"):
                attributes["model_name"] = str(model_name)
            histograms.append(
                VLLMHistogramSnapshot(
                    name=name,
                    buckets=tuple(
                        sorted(
                            (
                                (math.inf if bound == "+Inf" else float(bound), float(count))
                                for bound, count in metric.buckets.items()
                            ),
                            key=lambda item: item[0],
                        )
                    ),
                    count=float(metric.count),
                    total=float(metric.sum),
                    unit=VLLM_HISTOGRAM_UNITS[name],
                    attributes=attributes,
                )
            )
        elif hasattr(metric, "value"):
            key = f"{name}:{labels.get('reason')}" if name == "num_requests_waiting_by_reason" else name
            values[key] = float(metric.value)

    return VLLMNativeStatsSnapshot(
        current=VLLMCurrentStats(
            running_requests=int(values.get("num_requests_running", 0)),
            waiting_capacity=int(values.get("num_requests_waiting_by_reason:capacity", 0)),
            waiting_deferred=int(values.get("num_requests_waiting_by_reason:deferred", 0)),
            kv_cache_usage=values.get("kv_cache_usage_perc", 0.0),
        ),
        cumulative=VLLMCumulativeStats(
            prompt_tokens=int(values.get("prompt_tokens", 0)),
            generation_tokens=int(values.get("generation_tokens", 0)),
            prefix_cache_hits=int(values.get("prefix_cache_hits", 0)),
            prefix_cache_queries=int(values.get("prefix_cache_queries", 0)),
            preemptions=int(values.get("num_preemptions", 0)),
            spec_decode_drafts=int(values.get("spec_decode_num_drafts", 0)),
            spec_decode_draft_tokens=int(values.get("spec_decode_num_draft_tokens", 0)),
            spec_decode_accepted_tokens=int(values.get("spec_decode_num_accepted_tokens", 0)),
            finished_by_reason=finished,
        ),
        histograms=tuple(histograms),
    )
