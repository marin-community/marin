"""Sink adapters for the canonical inference-service snapshot."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

from loguru import logger

from skyrl_train.inference_engines.vllm.stats import (
    HTTP_BRIDGE_SUMMARY_METRIC_NAMES,
    VLLM_NUM_ENGINES_METRIC,
    InferenceStatsSnapshot,
    VLLMEngineStatsSnapshot,
    VLLMHistogramSnapshot,
)
from skyrl_train.telemetry import TelemetryConfig


VLLM_MAX_RECORDS_PER_ENGINE = 512
PUBLICATION_LOSS_METRIC = "metric_publication_dropped_records"


class InferenceMetricsSink(Protocol):
    """An interchangeable destination for one callback-owned snapshot."""

    def publish(self, snapshot: InferenceStatsSnapshot, step: int) -> None: ...


def configured_inference_sinks() -> tuple[InferenceMetricsSink, ...]:
    """Return the Finelog sink when telemetry is configured and Rigging is installed."""
    if not TelemetryConfig.from_environment().endpoint:
        return ()
    try:
        return (FinelogInferenceMetricsSink(),)
    except ImportError as error:
        if error.name != "rigging":
            raise
        logger.info("Rigging is unavailable; inference Finelog metrics are disabled")
        return ()


def trainer_metrics(snapshot: InferenceStatsSnapshot) -> dict[str, float]:
    """Project engine and HTTP bridge observations into the tracker's flat scalar contract."""
    engines = snapshot.engines
    if not engines:
        metrics = {}
    else:
        metrics = _engine_trainer_metrics(engines)
    if snapshot.http_bridge is not None:
        for name in HTTP_BRIDGE_SUMMARY_METRIC_NAMES:
            summary = getattr(snapshot.http_bridge, name)
            for statistic in ("count", "mean", "p95", "maximum"):
                metrics[f"inference_bridge/{name}/{statistic}"] = float(getattr(summary, statistic))
    return metrics


def _engine_trainer_metrics(engines: tuple[VLLMEngineStatsSnapshot, ...]) -> dict[str, float]:
    intervals = [engine.interval for engine in engines]
    count = len(intervals)
    finished = sum(item.finished_requests for item in intervals)
    cumulative = [engine.cumulative for engine in engines]
    spec_drafts = sum(item.spec_decode_drafts for item in cumulative)
    spec_draft_tokens = sum(item.spec_decode_draft_tokens for item in cumulative)
    spec_accepted_tokens = sum(item.spec_decode_accepted_tokens for item in cumulative)

    def average(name: str) -> float:
        return sum(float(getattr(item, name)) for item in intervals) / count

    def weighted(name: str) -> float:
        if not finished:
            return 0.0
        return sum(float(getattr(item, name)) * item.finished_requests for item in intervals) / finished

    return {
        VLLM_NUM_ENGINES_METRIC: float(count),
        "vllm/peak_running_reqs": float(sum(item.peak_running_reqs for item in intervals)),
        "vllm/peak_waiting_reqs": float(sum(item.peak_waiting_reqs for item in intervals)),
        "vllm/peak_prompt_throughput": average("peak_prompt_throughput"),
        "vllm/peak_generation_throughput": average("peak_generation_throughput"),
        "vllm/peak_gpu_cache_usage_perc": average("peak_gpu_cache_usage_perc"),
        "vllm/peak_prefix_cache_hit_rate": average("peak_prefix_cache_hit_rate"),
        "vllm/median_running_reqs": average("median_running_reqs"),
        "vllm/median_waiting_reqs": average("median_waiting_reqs"),
        "vllm/median_prompt_throughput": average("median_prompt_throughput"),
        "vllm/median_generation_throughput": average("median_generation_throughput"),
        "vllm/median_gpu_cache_usage_perc": average("median_gpu_cache_usage_perc"),
        "vllm/median_prefix_cache_hit_rate": average("median_prefix_cache_hit_rate"),
        "vllm/latency_prefill_mean": weighted("latency_prefill_mean"),
        "vllm/latency_prefill_p90": max(item.latency_prefill_p90 for item in intervals),
        "vllm/latency_decode_mean": weighted("latency_decode_mean"),
        "vllm/latency_decode_p90": max(item.latency_decode_p90 for item in intervals),
        "vllm/latency_e2e_mean": weighted("latency_e2e_mean"),
        "vllm/latency_e2e_p90": max(item.latency_e2e_p90 for item in intervals),
        "vllm/latency_queued_mean": weighted("latency_queued_mean"),
        "vllm/latency_queued_p90": max(item.latency_queued_p90 for item in intervals),
        "vllm/latency_ttft_mean": weighted("latency_ttft_mean"),
        "vllm/latency_ttft_p90": max(item.latency_ttft_p90 for item in intervals),
        "vllm/total_finished_requests": float(finished),
        "vllm/total_preempted_reqs": float(sum(item.preempted_reqs for item in intervals)),
        "vllm/total_samples": float(sum(item.samples for item in intervals)),
        "vllm/total_active_samples": float(sum(item.active_samples for item in intervals)),
        "vllm/spec_decode_drafts_total": float(spec_drafts),
        "vllm/spec_decode_draft_tokens_total": float(spec_draft_tokens),
        "vllm/spec_decode_accepted_tokens_total": float(spec_accepted_tokens),
        "vllm/spec_decode_acceptance_rate": (
            float(spec_accepted_tokens) / spec_draft_tokens if spec_draft_tokens else 0.0
        ),
        "vllm/spec_decode_mean_acceptance_length": (
            1.0 + float(spec_accepted_tokens) / spec_drafts if spec_drafts else 1.0
        ),
    }


def format_console_summary(metrics: Mapping[str, float], step: int) -> str:
    """Format the compact console view without coupling collection to a logger."""
    return (
        f"vLLM Stats (step {step}): engines={metrics[VLLM_NUM_ENGINES_METRIC]:.0f}, "
        f"running={metrics['vllm/median_running_reqs']:.1f}/{metrics['vllm/peak_running_reqs']:.0f}, "
        f"waiting={metrics['vllm/median_waiting_reqs']:.1f}/{metrics['vllm/peak_waiting_reqs']:.0f}, "
        f"generation={metrics['vllm/median_generation_throughput']:.1f}/"
        f"{metrics['vllm/peak_generation_throughput']:.1f} tok/s, "
        f"kv_cache={metrics['vllm/median_gpu_cache_usage_perc']:.1f}/"
        f"{metrics['vllm/peak_gpu_cache_usage_perc']:.1f}%, "
        f"e2e={metrics['vllm/latency_e2e_mean']:.3f}s"
    )


class FinelogInferenceMetricsSink:
    """Convert the neutral snapshot to Rigging records at the publishing edge."""

    def __init__(self) -> None:
        from rigging.telemetry.metrics import MetricSnapshotPublisher  # noqa: PLC0415

        self._publisher = MetricSnapshotPublisher(
            max_records=VLLM_MAX_RECORDS_PER_ENGINE,
            attributes={"metric_source": "vllm"},
        )
        self._bridge_publisher = MetricSnapshotPublisher(
            max_records=512, attributes={"metric_source": "inference_http_bridge"}
        )

    def publish(self, snapshot: InferenceStatsSnapshot, step: int) -> None:
        from rigging import telemetry  # noqa: PLC0415
        from rigging.telemetry.metrics import MetricSnapshot  # noqa: PLC0415

        sample_limit_dropped = 0
        telemetry_lost = 0
        for engine in snapshot.engines:
            records = []
            cumulative_base = {**engine.attributes, "engine": engine.engine_id}
            current_base = {**cumulative_base, "step": str(step)}
            current = engine.current
            for name, value, unit, attributes in (
                ("num_requests_running", current.running_requests, "{request}", {}),
                (
                    "num_requests_waiting",
                    current.waiting_capacity + current.waiting_deferred,
                    "{request}",
                    {},
                ),
                (
                    "num_requests_waiting_by_reason",
                    current.waiting_capacity,
                    "{request}",
                    {"reason": "capacity"},
                ),
                (
                    "num_requests_waiting_by_reason",
                    current.waiting_deferred,
                    "{request}",
                    {"reason": "deferred"},
                ),
                ("kv_cache_usage_perc", current.kv_cache_usage, "1", {}),
            ):
                records.append(
                    MetricSnapshot(
                        name=name,
                        value=value,
                        unit=unit,
                        attributes={**current_base, **attributes},
                        source_kind="gauge",
                        source_temporality=telemetry.CURRENT_SNAPSHOT,
                    )
                )
            cumulative = engine.cumulative
            counters = (
                ("num_preemptions_total", cumulative.preemptions, "{request}", {}),
                ("prefix_cache_hits_total", cumulative.prefix_cache_hits, "{token}", {}),
                ("prefix_cache_queries_total", cumulative.prefix_cache_queries, "{token}", {}),
                ("generation_tokens_total", cumulative.generation_tokens, "{token}", {}),
                ("prompt_tokens_total", cumulative.prompt_tokens, "{token}", {}),
                ("spec_decode_num_drafts_total", cumulative.spec_decode_drafts, "{draft}", {}),
                (
                    "spec_decode_num_draft_tokens_total",
                    cumulative.spec_decode_draft_tokens,
                    "{token}",
                    {},
                ),
                (
                    "spec_decode_num_accepted_tokens_total",
                    cumulative.spec_decode_accepted_tokens,
                    "{token}",
                    {},
                ),
                *(
                    ("request_success_total", value, "{request}", {"finished_reason": reason})
                    for reason, value in cumulative.finished_by_reason.items()
                ),
            )
            for name, value, unit, attributes in counters:
                records.append(
                    MetricSnapshot(
                        name=name,
                        value=value,
                        unit=unit,
                        attributes={**cumulative_base, **attributes},
                        source_kind="counter",
                        source_temporality=telemetry.CUMULATIVE_SNAPSHOT,
                    )
                )
            for histogram in engine.histograms:
                records.extend(
                    _histogram_records(histogram, cumulative_base, MetricSnapshot, telemetry.CUMULATIVE_SNAPSHOT)
                )
            if len(records) > VLLM_MAX_RECORDS_PER_ENGINE:
                sample_limit_dropped += len(records)
                logger.warning(
                    "Rejected oversized vLLM metric batch for engine {}: {} records exceeds {}",
                    engine.engine_id,
                    len(records),
                    VLLM_MAX_RECORDS_PER_ENGINE,
                )
            elif not _metric_batch_is_valid(records):
                telemetry_lost += len(records)
                logger.warning(
                    "Rejected invalid vLLM metric batch for engine {} before publication: {} records",
                    engine.engine_id,
                    len(records),
                )
            elif records:
                result = self._publisher.publish(records)
                if result.configured:
                    sample_limit_dropped += result.sample_limit_dropped_records
                    telemetry_lost += result.telemetry_lost_records
                if result.sample_limit_dropped_records or result.telemetry_lost_records:
                    logger.warning(
                        "vLLM metric publication lost records: sample_limit={}, telemetry_queue={}",
                        result.sample_limit_dropped_records,
                        result.telemetry_lost_records,
                    )
        _record_publication_health(telemetry, "vllm", sample_limit_dropped, telemetry_lost)
        if snapshot.http_bridge is not None:
            records = []
            for histogram in snapshot.http_bridge.histograms:
                records.extend(
                    _histogram_records(
                        histogram,
                        {},
                        MetricSnapshot,
                        telemetry.CUMULATIVE_SNAPSHOT,
                    )
                )
            if records:
                result = self._bridge_publisher.publish(records)
                if result.sample_limit_dropped_records or result.telemetry_lost_records:
                    logger.warning(
                        "HTTP bridge metric publication lost records: sample_limit={}, telemetry_queue={}",
                        result.sample_limit_dropped_records,
                        result.telemetry_lost_records,
                    )
                _record_publication_health(
                    telemetry,
                    "inference_http_bridge",
                    result.sample_limit_dropped_records if result.configured else 0,
                    result.telemetry_lost_records if result.configured else 0,
                )


class _MetricRecord(Protocol):
    name: str
    value: float
    unit: str
    attributes: Mapping[str, str]
    source_kind: str
    source_temporality: str


class _MetricRecordFactory(Protocol):
    def __call__(
        self,
        *,
        name: str,
        value: float,
        unit: str,
        attributes: Mapping[str, str],
        source_kind: str,
        source_temporality: str,
    ) -> _MetricRecord: ...


class _Gauge(Protocol):
    def set(self, value: float, *, attributes: Mapping[str, str]) -> None: ...


class _Telemetry(Protocol):
    def gauge(self, name: str, *, unit: str) -> _Gauge: ...


def _metric_batch_is_valid(records: list[_MetricRecord]) -> bool:
    """Mirror Rigging's per-record validation before admitting an engine batch."""
    from rigging.telemetry import serialization  # noqa: PLC0415

    try:
        for record in records:
            serialization.validate_string(record.name, "name")
            if record.unit:
                serialization.validate_string(record.unit, "unit")
            if not math.isfinite(float(record.value)):
                raise ValueError("metric value must be finite")
            serialization.validate_attributes(
                {
                    **record.attributes,
                    "metric_source": "vllm",
                    "source_kind": record.source_kind,
                    "source_temporality": record.source_temporality,
                }
            )
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _histogram_records(
    histogram: VLLMHistogramSnapshot,
    base: Mapping[str, str],
    metric_snapshot_type: _MetricRecordFactory,
    cumulative: str,
) -> list[_MetricRecord]:
    attributes = {**histogram.attributes, **base}
    records = [
        metric_snapshot_type(
            name=f"{histogram.name}_bucket",
            value=value,
            unit=histogram.unit,
            attributes={**attributes, "le": "+Inf" if math.isinf(bound) else str(bound)},
            source_kind="histogram",
            source_temporality=cumulative,
        )
        for bound, value in histogram.buckets
    ]
    for suffix, value in (("count", histogram.count), ("sum", histogram.total)):
        records.append(
            metric_snapshot_type(
                name=f"{histogram.name}_{suffix}",
                value=value,
                unit=histogram.unit,
                attributes=attributes,
                source_kind="histogram",
                source_temporality=cumulative,
            )
        )
    return records


def _record_publication_health(
    telemetry: _Telemetry, metric_source: str, sample_limit: int, telemetry_loss: int
) -> None:
    """Publish current loss state; this best-effort signal cannot attest to its own delivery."""
    gauge = telemetry.gauge(PUBLICATION_LOSS_METRIC, unit="{record}")
    for reason, value in (("sample_limit", sample_limit), ("telemetry_loss", telemetry_loss)):
        gauge.set(value, attributes={"metric_source": metric_source, "drop_reason": reason})
