# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for forwarding a native vLLM server's /metrics to telltale.

The forwarder polls the subprocess's Prometheus ``/metrics``, keeps only vLLM's own
``vllm:`` families, and mirrors them through a telltale collector so the existing
telltale->finelog forwarder persists them. These exercise the parsing/filtering, the
poller lifecycle, and the collision-avoidance that filtering buys.
"""

import time

import pytest
from marin.inference.vllm_metrics import (
    _PROXY,
    VLLM_METRICS_SOURCE,
    VllmMetricsForwarder,
    _VllmMetricsProxy,
    parse_vllm_families,
    start_vllm_metrics_forwarding,
)
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from prometheus_client.parser import text_string_to_metric_families

# A representative vLLM /metrics body: a gauge, two counters, a histogram — all vllm:-prefixed —
# plus the stdlib process/python collectors vLLM's registry also exposes and the parent already has.
_VLLM_METRICS_BODY = """# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 3.0
# HELP vllm:prompt_tokens_total Prompt tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m"} 1234.0
# HELP vllm:generation_tokens_total Generation tokens produced.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="m"} 567.0
# HELP vllm:time_to_first_token_seconds Time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 5.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 8.0
vllm:time_to_first_token_seconds_sum{model_name="m"} 1.2
vllm:time_to_first_token_seconds_count{model_name="m"} 8.0
# HELP process_cpu_seconds_total Total user and system CPU time.
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 5.0
# HELP python_gc_objects_collected_total Objects collected during gc.
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 42.0
"""


def test_parse_vllm_families_keeps_only_vllm_prefixed_and_preserves_types():
    families = {f.name: f.type for f in parse_vllm_families(_VLLM_METRICS_BODY)}

    # The stdlib process/python collectors — which the parent registry already holds — are dropped.
    assert not any(name.startswith(("process_", "python_")) for name in families)
    # vLLM's own families survive with their metric types intact (histogram, not flattened to gauges).
    assert families == {
        "vllm:num_requests_running": "gauge",
        "vllm:prompt_tokens": "counter",
        "vllm:generation_tokens": "counter",
        "vllm:time_to_first_token_seconds": "histogram",
    }


def test_proxy_collect_returns_latest_snapshot():
    proxy = _VllmMetricsProxy()
    assert list(proxy.collect()) == []

    proxy.set_latest(parse_vllm_families(_VLLM_METRICS_BODY))
    names = {s.name for family in proxy.collect() for s in family.samples}

    assert "vllm:num_requests_running" in names
    assert "vllm:time_to_first_token_seconds_bucket" in names  # histogram buckets are preserved


def test_poll_once_populates_proxy_from_fetch():
    proxy = _VllmMetricsProxy()
    forwarder = VllmMetricsForwarder("http://server/metrics", proxy, fetch=lambda _url: _VLLM_METRICS_BODY)

    forwarder.poll_once()

    assert {f.name for f in proxy.collect()} == {
        "vllm:num_requests_running",
        "vllm:prompt_tokens",
        "vllm:generation_tokens",
        "vllm:time_to_first_token_seconds",
    }


def test_poll_once_keeps_last_snapshot_when_scrape_fails():
    proxy = _VllmMetricsProxy()
    bodies = [_VLLM_METRICS_BODY, None]  # first scrape succeeds, second misses
    forwarder = VllmMetricsForwarder("http://server/metrics", proxy, fetch=lambda _url: bodies.pop(0))

    forwarder.poll_once()
    forwarder.poll_once()  # a miss must not blank the series

    assert {f.name for f in proxy.collect()} != set()
    assert "vllm:num_requests_running" in {f.name for f in proxy.collect()}


def test_forwarder_thread_polls_then_stops():
    proxy = _VllmMetricsProxy()
    scrapes = []

    def _fetch(url: str) -> str:
        scrapes.append(url)
        return _VLLM_METRICS_BODY

    # A long interval keeps the loop to its one immediate scrape; stop() joins the thread.
    forwarder = VllmMetricsForwarder("http://server/metrics", proxy, fetch=_fetch, interval=1000.0)
    forwarder.start()

    deadline = time.monotonic() + 5
    while not scrapes:
        if time.monotonic() > deadline:
            raise AssertionError("poller never scraped")
        time.sleep(0.01)

    forwarder.stop(timeout=5)
    assert not forwarder._thread.is_alive()
    assert proxy.collect()  # the immediate scrape populated the proxy


def test_filtered_mirror_does_not_collide_with_a_parent_process_collector():
    # auto_describe=True mirrors telltale's default registry, which already exposes
    # process_cpu_seconds_total via its process collector and checks names at register time.
    registry = CollectorRegistry(auto_describe=True)
    Gauge("process_cpu_seconds_total", "already held by the parent", registry=registry)

    # Re-emitting the raw body (which carries process_cpu_seconds_total) would collide...
    with pytest.raises(ValueError, match="Duplicated timeseries"):
        colliding = _VllmMetricsProxy()
        colliding.set_latest(text_string_to_metric_families(_VLLM_METRICS_BODY))
        registry.register(colliding)

    # ...but the filtered mirror only emits vllm: families, so it registers cleanly.
    proxy = _VllmMetricsProxy()
    proxy.set_latest(parse_vllm_families(_VLLM_METRICS_BODY))
    registry.register(proxy)

    body = generate_latest(registry).decode()
    assert "vllm:num_requests_running" in body


@pytest.fixture
def reset_global_telltale():
    """Restore telltale's process-global labels and clear the shared proxy after the case."""
    from rigging import telltale

    saved = telltale.get_global_labels()
    yield
    telltale._global_labels.clear()
    telltale._global_labels.update(saved)
    _PROXY.set_latest([])


def test_start_vllm_metrics_forwarding_wires_source_and_proxy(monkeypatch, reset_global_telltale):
    from rigging import telltale

    class _FakeResponse:
        status_code = 200
        text = _VLLM_METRICS_BODY

    monkeypatch.setattr("marin.inference.vllm_metrics.requests.get", lambda *a, **k: _FakeResponse())

    forwarder = start_vllm_metrics_forwarding("http://server/metrics", interval=1000.0)
    try:
        deadline = time.monotonic() + 5
        while not _PROXY.collect():
            if time.monotonic() > deadline:
                raise AssertionError("proxy never populated")
            time.sleep(0.01)

        # The process's forwarded rows are stamped as vLLM's, and the mirrored families are present.
        assert telltale.get_global_labels()["source"] == VLLM_METRICS_SOURCE
        assert "vllm:num_requests_running" in {f.name for f in _PROXY.collect()}
    finally:
        forwarder.stop(timeout=5)
