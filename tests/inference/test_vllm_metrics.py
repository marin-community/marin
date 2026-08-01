# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.inference.vllm_metrics import VllmMetricsForwarder, parse_vllm_families, publish_vllm_families
from rigging import telemetry
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


def test_vllm_cumulative_snapshots_use_direct_gauge_wire_format(monkeypatch):
    telemetry.shutdown(0)
    transport = RecordingTelemetryTransport()
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    telemetry.configure(endpoint="http://finelog/v1/telemetry", service="vllm", attributes={"job_id": "/serve"})

    publish_vllm_families(parse_vllm_families(_SCRAPE))
    with transport.condition:
        assert transport.condition.wait_for(lambda: len(transport.records) >= 5, timeout=5)
    telemetry.shutdown(1)

    by_name = {record["name"]: record for record in transport.records}
    tokens = by_name["generation_tokens_total"]
    assert tokens["kind"] == "gauge"
    assert tokens["value"] == 42
    assert tokens["attributes"] == {
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


def test_failed_vllm_scrape_skips_source_metrics_but_reports_exporter_health(monkeypatch):
    emitted = []
    health = []
    monkeypatch.setattr("marin.inference.vllm_metrics.publish_vllm_families", lambda families: emitted.append(families))
    monkeypatch.setattr(telemetry, "record_runtime_health", lambda: health.append(True))
    VllmMetricsForwarder("http://vllm/metrics", fetch=lambda _url: None).poll_once()
    assert emitted == []
    assert health == [True]
