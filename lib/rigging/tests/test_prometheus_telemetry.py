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
    telemetry.configure(endpoint="http://finelog/v1/telemetry", service="vllm", attributes={"job_id": "/serve"})
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


def test_scraper_propagates_transport_error_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise requests.ConnectionError("unavailable")

    monkeypatch.setattr("rigging.telemetry.prometheus.requests.get", unavailable)

    with pytest.raises(requests.ConnectionError):
        PrometheusScraper("http://vllm/metrics").scrape()


def test_scraper_rejects_oversized_response_before_reading_body(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _PrometheusResponse(_SCRAPE, content_length=DEFAULT_MAX_SCRAPE_BYTES + 1)
    monkeypatch.setattr("rigging.telemetry.prometheus.requests.get", lambda *_args, **_kwargs: response)

    with pytest.raises(PrometheusScrapeError):
        PrometheusScraper("http://vllm/metrics").scrape()

    assert not response.body_read
