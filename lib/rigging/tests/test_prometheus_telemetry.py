# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from rigging import telemetry
from rigging.telemetry.prometheus import PrometheusForwarder
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


def _forwarder(body: str | None) -> PrometheusForwarder:
    return PrometheusForwarder(
        "http://vllm/metrics",
        metric_prefix="vllm:",
        metric_source="vllm",
        fetch=lambda _url: body,
    )


def test_filtered_snapshots_preserve_prometheus_temporality(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    labels = ",".join(f'label_{index}="x"' for index in range(62))
    scrape = _SCRAPE + f"\n# TYPE vllm:oversized gauge\nvllm:oversized{{{labels}}} 1\n"

    _forwarder(scrape).poll_once()
    transport.record("request_duration_seconds_sum", {})

    by_name = {record["name"]: record for record in transport.records}
    assert by_name["generation_tokens_total"]["attributes"] == {
        "metric_source": "vllm",
        "model_name": "test",
        "source_kind": "counter",
        "source_temporality": "cumulative_snapshot",
    }
    assert by_name["num_requests_running"]["attributes"]["source_temporality"] == "current_snapshot"
    assert by_name["request_duration_seconds"]["attributes"]["source_temporality"] == "current_snapshot"
    assert by_name["request_duration_seconds_count"]["attributes"]["source_temporality"] == "cumulative_snapshot"
    assert by_name["request_duration_seconds_sum"]["attributes"]["source_temporality"] == "cumulative_snapshot"
    assert "process_cpu_seconds_total" not in by_name
    assert "oversized" not in by_name
    assert telemetry.runtime_status().lost_records == 1


def test_failed_scrape_skips_source_metrics_but_reports_exporter_health(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)

    _forwarder(None).poll_once()
    health = transport.record("queue_depth", {"queue_kind": "telemetry_export"})

    assert health["attributes"]["source_temporality"] == "current_snapshot"
    assert all(record["attributes"].get("metric_source") != "vllm" for record in transport.records)
