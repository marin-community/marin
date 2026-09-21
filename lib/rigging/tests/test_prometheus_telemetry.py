# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator, Sequence

import pytest
import requests
from prometheus_client.core import Metric as PrometheusMetric
from rigging import telemetry
from rigging.telemetry import metrics
from rigging.telemetry.prometheus import (
    DEFAULT_MAX_SCRAPE_BYTES,
    PrometheusCollector,
    PrometheusProcessor,
    PrometheusScrapeError,
    PrometheusScraper,
    prefixed_metric_snapshots,
)
from rigging.testing import RecordingTelemetryTransport

_SCRAPE = """
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="test"} 42
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="test"} 3
# TYPE vllm:request_duration_seconds summary
vllm:request_duration_seconds{quantile="0.5"} 1.5
vllm:request_duration_seconds_sum 9
vllm:request_duration_seconds_count 4
# TYPE process_cpu_seconds counter
process_cpu_seconds_total 9
"""


class _PrometheusResponse:
    status_code = 200
    encoding = "utf-8"

    def __init__(self, body: str, *, content_length: int | None = None) -> None:
        self._body = body.encode()
        self.headers = {"content-length": str(len(self._body) if content_length is None else content_length)}
        self.body_read = False

    def __enter__(self) -> "_PrometheusResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        self.body_read = True
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    telemetry.shutdown(0.01)
    yield
    telemetry.shutdown(0.1)


def _transport(monkeypatch: pytest.MonkeyPatch) -> RecordingTelemetryTransport:
    transport = RecordingTelemetryTransport()
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    telemetry.configure(
        endpoint="http://finelog/v1/telemetry",
        service="vllm",
        attributes={"job_id": "/serve"},
    )
    return transport


def _collector(
    processor: PrometheusProcessor,
    *,
    max_records: int = 128,
) -> PrometheusCollector:
    return PrometheusCollector(
        metric_source="vllm",
        scraper=PrometheusScraper("http://vllm/metrics"),
        processor=processor,
        publisher=metrics.MetricSnapshotPublisher(
            max_records=max_records,
            attributes={"metric_source": "vllm"},
        ),
    )


def test_prometheus_pipeline_preserves_prefixed_snapshot_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    oversized_labels = ",".join(f'label_{index}="x"' for index in range(62))
    scrape = _SCRAPE + f"\n# TYPE vllm:oversized gauge\nvllm:oversized{{{oversized_labels}}} 1\n"
    monkeypatch.setattr(
        "rigging.telemetry.prometheus.requests.get", lambda *_args, **_kwargs: _PrometheusResponse(scrape)
    )

    _collector(lambda families: prefixed_metric_snapshots(families, metric_prefix="vllm:")).poll_once()

    generation = transport.record("generation_tokens_total", {"model_name": "test"})
    assert generation["value"] == 42
    assert generation["attributes"] == {
        "metric_source": "vllm",
        "model_name": "test",
        "source_kind": "counter",
        "source_temporality": "cumulative_snapshot",
    }
    assert (
        transport.record("num_requests_running", {"model_name": "test"})["attributes"]["source_temporality"]
        == "current_snapshot"
    )
    assert (
        transport.record("request_duration_seconds", {"quantile": "0.5"})["attributes"]["source_temporality"]
        == "current_snapshot"
    )
    assert (
        transport.record("request_duration_seconds_count", {})["attributes"]["source_temporality"]
        == "cumulative_snapshot"
    )
    transport.wait_for(5)
    assert not [record for record in transport.records if record["name"] == "process_cpu_seconds_total"]
    assert not [record for record in transport.records if record["name"] == "oversized"]
    assert telemetry.runtime_status().lost_records == 1
    assert transport.record("prometheus_source_available", {"metric_source": "vllm"})["value"] == 1


def test_metric_snapshot_publisher_caps_processor_output(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    monkeypatch.setattr(
        "rigging.telemetry.prometheus.requests.get", lambda *_args, **_kwargs: _PrometheusResponse(_SCRAPE)
    )

    def processor(_families: tuple[PrometheusMetric, ...]) -> Sequence[metrics.MetricSnapshot]:
        return tuple(
            metrics.MetricSnapshot(
                name="bounded_metric",
                value=index,
                unit="1",
                attributes={"index": str(index)},
                source_kind="gauge",
                source_temporality=telemetry.CURRENT_SNAPSHOT,
            )
            for index in range(4)
        )

    _collector(processor, max_records=2).poll_once()

    assert transport.record("prometheus_enqueued_samples", {"metric_source": "vllm"})["value"] == 2
    assert (
        transport.record(
            "prometheus_dropped_samples",
            {"metric_source": "vllm", "drop_reason": "sample_limit"},
        )["value"]
        == 2
    )
    transport.wait_for(2)
    bounded_records = (record for record in transport.records if record["name"] == "bounded_metric")
    indices = sorted(record["attributes"]["index"] for record in bounded_records)
    assert indices == ["0", "1"]


def test_cumulative_histogram_bundle_is_one_exact_event_record(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    count = (1 << 53) + 1
    histogram = metrics.CumulativeHistogramSnapshot(
        name="request_queue_time_seconds",
        finite_bounds=(0.01, 0.1),
        cumulative_counts=(1, 2, count),
        count=count,
        total=1.2345678901234567,
        unit="s",
        attributes={"model_name": "m"},
    )
    snapshot = metrics.CumulativeHistogramBundleSnapshot(
        name="vllm_histogram_bundle",
        histograms=(histogram,),
        attributes={"engine": "physical-a", "engine_index": "0"},
        timestamp_ms=1_789_776_000_123,
        sample_sequence=7,
    )

    publisher = metrics.CumulativeHistogramBundleSnapshotPublisher(
        max_records=8,
        attributes={"metric_source": "vllm"},
    )
    result = publisher.publish((snapshot,))

    assert result.enqueued_records == 1
    record = transport.record("vllm_histogram_bundle", {"engine": "physical-a"})
    assert record["kind"] == "event"
    assert record["timestamp_ms"] == 1_789_776_000_123
    assert record["attributes"] == {
        "engine": "physical-a",
        "engine_index": "0",
        "histogram_encoding": metrics.CUMULATIVE_HISTOGRAM_BUNDLE_ENCODING,
        "metric_source": "vllm",
        "source_kind": "histogram_bundle",
        "source_temporality": "cumulative_snapshot",
    }
    histogram_body = record["body"]["histograms"]["request_queue_time_seconds"]
    assert record["body"]["encoding"] == metrics.CUMULATIVE_HISTOGRAM_BUNDLE_ENCODING
    assert record["body"]["sample_sequence"] == 7
    assert len(histogram_body["schema"]) == 64
    assert len(histogram_body["series"]) == 64
    assert histogram_body == {
        "attributes": {"model_name": "m"},
        "count": count,
        "cumulative_counts": [1, 2, count],
        "delta_valid": False,
        "encoding": metrics.CUMULATIVE_HISTOGRAM_ENCODING,
        "finite_bounds": [0.01, 0.1],
        "schema": histogram_body["schema"],
        "series": histogram_body["series"],
        "sum": 1.2345678901234567,
        "unit": "s",
    }

    next_snapshot = metrics.CumulativeHistogramBundleSnapshot(
        name=snapshot.name,
        histograms=(
            metrics.CumulativeHistogramSnapshot(
                name=histogram.name,
                finite_bounds=histogram.finite_bounds,
                cumulative_counts=(2, 4, count + 3),
                count=count + 3,
                total=1.7345678901234567,
                unit=histogram.unit,
                attributes=histogram.attributes,
            ),
        ),
        attributes={
            **snapshot.attributes,
            "histogram_collection_timestamp_ms": str(snapshot.timestamp_ms + 5_000),
            "histogram_publication_id": "physical-a:next:8",
            "histogram_sample_sequence": "8",
        },
        timestamp_ms=snapshot.timestamp_ms + 5_000,
        sample_sequence=8,
    )
    assert publisher.publish((next_snapshot,)).enqueued_records == 1
    records = transport.wait_for(2)
    delta = records[-1]["body"]["histograms"]["request_queue_time_seconds"]
    assert delta["delta_valid"] is True
    assert delta["delta_cumulative_counts"] == [1, 2, 3]
    assert delta["delta_count"] == 3
    assert delta["delta_sum"] == 0.5
    assert delta["delta_from_sample_sequence"] == 7
    assert delta["delta_from_timestamp_ms"] == snapshot.timestamp_ms
    assert records[-1]["body"]["delta_component_bounds_pipe"] == "0.01|0.1|+Inf|_|_"
    assert records[-1]["body"]["delta_component_kinds_pipe"] == "bucket|bucket|bucket|count|sum"
    assert records[-1]["body"]["delta_component_values_pipe"] == "1|2|3|3|0.5"
    assert records[-1]["body"]["delta_family_indexes_pipe"] == "0|0|0|0|0"
    assert records[-1]["body"]["delta_family_names_pipe"] == "request_queue_time_seconds"


def test_cumulative_histogram_series_is_canonical_and_label_sensitive() -> None:
    assert metrics.cumulative_histogram_series({"engine": "a", "model": "m"}) == metrics.cumulative_histogram_series(
        {"model": "m", "engine": "a"}
    )
    assert metrics.cumulative_histogram_series({"engine": "a"}) != metrics.cumulative_histogram_series({"engine": "b"})


def test_cumulative_histogram_delta_skips_reset_then_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    publisher = metrics.CumulativeHistogramBundleSnapshotPublisher(max_records=8)

    def snapshot(sequence: int, count: int, total: float) -> metrics.CumulativeHistogramBundleSnapshot:
        return metrics.CumulativeHistogramBundleSnapshot(
            name="vllm_histogram_bundle",
            histograms=(
                metrics.CumulativeHistogramSnapshot(
                    name="latency_seconds",
                    finite_bounds=(0.1, 1.0),
                    cumulative_counts=(count, count, count),
                    count=count,
                    total=total,
                    unit="s",
                    attributes={"engine": "a"},
                ),
            ),
            attributes={"engine": "a"},
            timestamp_ms=sequence * 5_000,
            sample_sequence=sequence,
        )

    assert publisher.publish((snapshot(1, 10, 1.0),)).enqueued_records == 1
    assert publisher.publish((snapshot(2, 2, 0.2),)).enqueued_records == 1
    assert publisher.publish((snapshot(3, 3, 0.3),)).enqueued_records == 1

    records = transport.wait_for(3)
    bodies = [record["body"] for record in records]
    assert [body["histograms"]["latency_seconds"]["delta_valid"] for body in bodies] == [False, False, True]
    assert bodies[-1]["delta_component_values_pipe"] == "1|1|1|1|0.09999999999999998"


def test_cumulative_histogram_delta_ignores_out_of_order_baseline_and_allows_same_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport(monkeypatch)
    publisher = metrics.CumulativeHistogramBundleSnapshotPublisher(max_records=8)

    def snapshot(sequence: int, timestamp_ms: int, count: int) -> metrics.CumulativeHistogramBundleSnapshot:
        return metrics.CumulativeHistogramBundleSnapshot(
            name="vllm_histogram_bundle",
            histograms=(
                metrics.CumulativeHistogramSnapshot(
                    name="latency_seconds",
                    finite_bounds=(0.1,),
                    cumulative_counts=(count, count),
                    count=count,
                    total=count / 10,
                    unit="s",
                ),
            ),
            attributes={"engine": "a"},
            timestamp_ms=timestamp_ms,
            sample_sequence=sequence,
        )

    assert publisher.publish((snapshot(1, 5_000, 10),)).enqueued_records == 1
    assert publisher.publish((snapshot(3, 15_000, 30),)).enqueued_records == 1
    assert publisher.publish((snapshot(2, 10_000, 20),)).enqueued_records == 1
    assert publisher.publish((snapshot(4, 15_000, 40),)).enqueued_records == 1

    records = transport.wait_for(4)
    bodies = [record["body"]["histograms"]["latency_seconds"] for record in records]
    assert [body["delta_valid"] for body in bodies] == [False, True, False, True]
    assert bodies[-1]["delta_from_sample_sequence"] == 3
    assert bodies[-1]["delta_from_timestamp_ms"] == 15_000
    assert bodies[-1]["delta_cumulative_counts"] == [10, 10]


def test_cumulative_histogram_schema_change_starts_a_new_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    publisher = metrics.CumulativeHistogramBundleSnapshotPublisher(max_records=8)

    def snapshot(sequence: int, bound: float, count: int) -> metrics.CumulativeHistogramBundleSnapshot:
        return metrics.CumulativeHistogramBundleSnapshot(
            name="vllm_histogram_bundle",
            histograms=(
                metrics.CumulativeHistogramSnapshot(
                    name="latency_seconds",
                    finite_bounds=(bound,),
                    cumulative_counts=(count, count),
                    count=count,
                    total=count / 10,
                    unit="s",
                ),
            ),
            attributes={"engine": "a"},
            timestamp_ms=sequence * 5_000,
            sample_sequence=sequence,
        )

    for item in (
        snapshot(1, 0.1, 10),
        snapshot(2, 0.2, 20),
        snapshot(3, 0.1, 30),
        snapshot(4, 0.1, 40),
    ):
        assert publisher.publish((item,)).enqueued_records == 1

    records = transport.wait_for(4)
    bodies = [record["body"]["histograms"]["latency_seconds"] for record in records]
    assert [body["delta_valid"] for body in bodies] == [False, False, False, True]
    assert bodies[-1]["delta_from_sample_sequence"] == 3
    assert bodies[-1]["delta_cumulative_counts"] == [10, 10]


def test_cumulative_histogram_bundle_rejects_query_index_larger_than_finelog_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport(monkeypatch)
    publisher = metrics.CumulativeHistogramBundleSnapshotPublisher(max_records=8)
    bounds = tuple(float(index) for index in range(1, 128))

    def snapshot(sequence: int, count: int) -> metrics.CumulativeHistogramBundleSnapshot:
        histograms = tuple(
            metrics.CumulativeHistogramSnapshot(
                name=f"latency_{index}_seconds",
                finite_bounds=bounds,
                cumulative_counts=(count,) * 128,
                count=count,
                total=float(count),
                unit="s",
            )
            for index in range(64)
        )
        return metrics.CumulativeHistogramBundleSnapshot(
            name="vllm_histogram_bundle",
            histograms=histograms,
            attributes={"engine": "a"},
            timestamp_ms=sequence * 5_000,
            sample_sequence=sequence,
        )

    assert publisher.publish((snapshot(1, 1),)).enqueued_records == 1
    result = publisher.publish((snapshot(2, 2),))

    assert result.enqueued_records == 0
    assert result.telemetry_lost_records == 1
    assert telemetry.runtime_status().lost_records == 1
    transport.wait_for(1)
    assert len(transport.records) == 1


def test_invalid_cumulative_histogram_bundle_is_lost_as_one_record(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    publisher = metrics.CumulativeHistogramBundleSnapshotPublisher(max_records=8)
    common_histogram = {
        "name": "request_queue_time_seconds",
        "finite_bounds": (0.01, 0.1),
        "count": 2,
        "total": 0.2,
        "unit": "s",
    }
    invalid_histogram = metrics.CumulativeHistogramSnapshot(cumulative_counts=(1, 2), **common_histogram)
    valid_histogram = metrics.CumulativeHistogramSnapshot(cumulative_counts=(1, 2, 2), **common_histogram)
    common_bundle = {
        "name": "vllm_histogram_bundle",
        "attributes": {"engine": "physical-a"},
        "timestamp_ms": 1_000,
        "sample_sequence": 1,
    }
    invalid = metrics.CumulativeHistogramBundleSnapshot(histograms=(invalid_histogram,), **common_bundle)
    valid = metrics.CumulativeHistogramBundleSnapshot(histograms=(valid_histogram,), **common_bundle)

    result = publisher.publish((invalid, valid))

    assert result.enqueued_records == 1
    assert result.telemetry_lost_records == 1
    transport.wait_for(1)
    assert len(transport.records) == 1
    assert transport.records[0]["body"]["histograms"]["request_queue_time_seconds"]["cumulative_counts"] == [
        1,
        2,
        2,
    ]


def test_cumulative_histogram_bundle_rejects_negative_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    publisher = metrics.CumulativeHistogramBundleSnapshotPublisher(max_records=8)
    histogram = metrics.CumulativeHistogramSnapshot(
        name="request_queue_time_seconds",
        finite_bounds=(0.01,),
        cumulative_counts=(0, 1),
        count=1,
        total=-0.1,
        unit="s",
    )
    bundle = metrics.CumulativeHistogramBundleSnapshot(
        name="vllm_histogram_bundle",
        histograms=(histogram,),
        attributes={"engine": "physical-a"},
        timestamp_ms=1_000,
        sample_sequence=1,
    )

    result = publisher.publish((bundle,))

    assert result.enqueued_records == 0
    assert result.telemetry_lost_records == 1
    assert transport.records == []


def test_processor_failure_does_not_hide_successful_scrape(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    monkeypatch.setattr(
        "rigging.telemetry.prometheus.requests.get", lambda *_args, **_kwargs: _PrometheusResponse(_SCRAPE)
    )

    def processor(_families: tuple[PrometheusMetric, ...]) -> Sequence[metrics.MetricSnapshot]:
        raise RuntimeError("policy failed")

    _collector(processor).poll_once()

    assert transport.record("prometheus_source_available", {"metric_source": "vllm"})["value"] == 1
    assert (
        transport.record(
            "prometheus_stage_failures",
            {"metric_source": "vllm", "stage": "process"},
        )["value"]
        == 1
    )
    assert not [record for record in transport.records if record["name"] == "generation_tokens_total"]


def test_scrape_failure_is_reported_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)

    def unavailable(*_args, **_kwargs):
        raise requests.ConnectionError("unavailable")

    monkeypatch.setattr("rigging.telemetry.prometheus.requests.get", unavailable)
    _collector(lambda families: prefixed_metric_snapshots(families, metric_prefix="vllm:")).poll_once()

    assert transport.record("prometheus_source_available", {"metric_source": "vllm"})["value"] == 0
    assert (
        transport.record(
            "prometheus_stage_failures",
            {"metric_source": "vllm", "stage": "scrape"},
        )["value"]
        == 1
    )


def test_health_reporting_failure_does_not_stop_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    responses: Iterator[Exception | _PrometheusResponse] = iter(
        [requests.ConnectionError("unavailable"), _PrometheusResponse(_SCRAPE)]
    )

    def scrape(*_args: object, **_kwargs: object) -> _PrometheusResponse:
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    def unavailable_health() -> None:
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr("rigging.telemetry.prometheus.requests.get", scrape)
    monkeypatch.setattr(telemetry, "record_runtime_health", unavailable_health)
    collector = _collector(lambda families: prefixed_metric_snapshots(families, metric_prefix="vllm:"))

    collector.poll_once()
    collector.poll_once()

    assert transport.record("generation_tokens_total", {"model_name": "test"})["value"] == 42


def test_scraper_rejects_oversized_response_before_reading_body(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _PrometheusResponse(_SCRAPE, content_length=DEFAULT_MAX_SCRAPE_BYTES + 1)
    monkeypatch.setattr("rigging.telemetry.prometheus.requests.get", lambda *_args, **_kwargs: response)

    with pytest.raises(PrometheusScrapeError):
        PrometheusScraper("http://vllm/metrics").scrape()

    assert not response.body_read
