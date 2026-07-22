# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import itertools
from datetime import datetime

import pytest
from rigging import telltale
from rigging.server_auth import RequestAuthPolicy, RouteAuthMiddleware
from starlette.applications import Starlette
from starlette.testclient import TestClient

_names = itertools.count()
_TS = datetime(2026, 7, 17, 12, 0, 0)


@pytest.fixture
def name() -> str:
    """A metric name unique to each test.

    telltale registers into the process-global prometheus registry by design, so
    tests cannot share names without colliding through it.
    """
    return f"telltale_test_{next(_names)}"


@pytest.fixture
def client() -> TestClient:
    return TestClient(Starlette(routes=telltale.routes()))


@pytest.fixture
def clean_global_labels():
    """Restore the process-global label set so cases don't leak into each other."""
    saved = telltale.get_global_labels()
    telltale._global_labels.clear()
    yield
    telltale._global_labels.clear()
    telltale._global_labels.update(saved)


@pytest.fixture
def reset_forwarding():
    """Clear the module-level forwarder so each case starts one cleanly."""
    telltale.stop_forwarding()
    yield
    telltale.stop_forwarding()


class _RecordingSink:
    """A telltale MetricSink that records the batches it is handed."""

    def __init__(self) -> None:
        self.batches: list[list[telltale.TelltaleMetric]] = []
        self.closed = False

    def write(self, rows) -> None:
        self.batches.append(list(rows))

    def close(self) -> None:
        self.closed = True


def _one(name: str, rows: list[telltale.TelltaleMetric]) -> telltale.TelltaleMetric:
    matching = [r for r in rows if r.name == name]
    assert len(matching) == 1, f"expected one {name!r} row, got {len(matching)}"
    return matching[0]


def test_counter_is_get_or_create(name):
    telltale.counter(name, "d", ["route"]).labels("/a").inc(2)
    telltale.counter(name, "d", ["route"]).labels("/a").inc()

    assert telltale.counter(name, "d", ["route"]).labels("/a")._value.get() == 3


def test_reregistering_with_a_different_type_raises(name):
    telltale.counter(name, "d")

    with pytest.raises(ValueError, match="already registered as Counter"):
        telltale.gauge(name, "d")


def test_publish_gauge_sets_a_gauge_named_after_an_arbitrary_key(name):
    telltale.publish_gauge(f"{name}/records in", 12, "d")

    assert telltale.gauge(f"{name}_records_in", "d")._value.get() == 12


def test_publish_gauge_drops_a_key_whose_name_is_taken_by_another_type(name):
    """A key it cannot publish must not raise: exposition never breaks its caller."""
    telltale.counter(name, "d")

    telltale.publish_gauge(name, 1, "d")


def test_reregistering_with_different_labels_raises(name):
    telltale.counter(name, "d", ["route"])

    with pytest.raises(ValueError, match="already registered with labels"):
        telltale.counter(name, "d", ["route", "method"])


def test_metrics_route_exposes_counter_in_prometheus_format(name, client):
    telltale.counter(name, "d", ["route"]).labels("/a").inc(4)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert f'{name}_total{{route="/a"}} 4.0' in response.text


def test_metrics_route_flattens_histograms_into_le_buckets(name, client):
    telltale.histogram(name, "d").observe(0.5)

    body = client.get("/metrics").text

    assert f'{name}_bucket{{le="+Inf"}} 1.0' in body
    assert f"{name}_sum 0.5" in body
    assert f"{name}_count 1.0" in body


def test_samples_reports_family_type_for_each_sample(name):
    telltale.gauge(name, "d").set(7)

    matching = [fs for fs in telltale.samples() if fs.family == name]

    assert [(fs.kind, fs.sample.value) for fs in matching] == [("gauge", 7.0)]


def test_samples_flattens_a_histogram_into_le_tagged_buckets(name):
    telltale.histogram(name, "d", []).observe(0.5)

    buckets = [fs for fs in telltale.samples() if fs.sample.name == f"{name}_bucket"]

    assert all(fs.kind == "histogram" for fs in buckets)
    assert "+Inf" in {fs.sample.labels["le"] for fs in buckets}


@pytest.mark.parametrize(
    ("raw", "prefix", "expected"),
    [
        ("zephyr/records_in", "", "zephyr_records_in"),
        ("train/loss", "levanter", "levanter_train_loss"),
        ("already_legal", "", "already_legal"),
        ("throughput/tokens per second", "levanter", "levanter_throughput_tokens_per_second"),
    ],
)
def test_metric_name_sanitizes_keys(raw, prefix, expected):
    assert telltale.metric_name(raw, prefix=prefix) == expected


def test_index_renders_status_and_metric_values(name, client):
    telltale.gauge(name, "d").set(12)
    telltale.set_status("shard 3/10 done")

    body = client.get("/").text

    assert "shard 3/10 done" in body
    assert name in body
    assert "12.0" in body


def test_index_escapes_status_html(client):
    telltale.set_status("<script>alert(1)</script>")

    body = client.get("/").text

    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_index_links_are_relative_so_they_survive_a_proxy_prefix(client):
    body = client.get("/").text

    assert 'href="metrics"' in body
    assert 'href="/metrics"' not in body


def test_routes_are_reachable_under_a_fail_closed_auth_middleware():
    """RouteAuthMiddleware denies unannotated routes, so the @public marks matter."""
    app = Starlette(routes=telltale.routes())
    wrapped = RouteAuthMiddleware(app, RequestAuthPolicy.enforcing(verifier=None))

    client = TestClient(wrapped)

    assert client.get("/metrics").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_global_labels_merge_and_a_later_key_overrides(clean_global_labels):
    telltale.set_global_labels(run="r1", source="levanter")
    telltale.set_global_labels(run="r2")

    assert telltale.get_global_labels() == {"run": "r2", "source": "levanter"}


def test_global_labels_coerce_to_str_and_snapshot_is_a_copy(clean_global_labels):
    telltale.set_global_labels(process_index=3)

    snapshot = telltale.get_global_labels()
    assert snapshot == {"process_index": "3"}

    snapshot["process_index"] = "mutated"
    assert telltale.get_global_labels() == {"process_index": "3"}


def test_status_survives_a_scrape_and_is_replaced_not_appended(client):
    telltale.set_status("first")
    telltale.set_status("second")

    body = client.get("/").text

    assert "second" in body
    assert "first" not in body


# --- forwarding ------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_prefix", "expected"),
    [
        ("levanter_", "levanter"),
        ("zephyr_", "zephyr"),
        ("iris_", "iris"),
        ("process_", "process"),
        ("vllm:", "process"),
    ],
)
def test_scrape_source_falls_back_to_metric_name_prefix(name, clean_global_labels, metric_prefix, expected):
    metric_name = f"{metric_prefix}{name}"
    telltale.gauge(metric_name, "d").set(1)

    row = _one(metric_name, telltale.scrape_metrics(telltale.MetricIdentity(), _TS))

    assert row.source == expected


def test_scrape_flattens_identity_and_run_source_into_columns(name, clean_global_labels):
    telltale.gauge(name, "d").set(2.0)
    telltale.set_global_labels(run="r1", source="levanter", cluster="us-central2")

    identity = telltale.MetricIdentity(job_id="/a/b", task_index=3, attempt=1)
    row = _one(name, telltale.scrape_metrics(identity, _TS))

    assert row.value == 2.0
    assert row.kind == "gauge"
    assert row.source == "levanter"
    assert row.run == "r1"
    assert row.job_id == "/a/b" and row.task_index == 3 and row.attempt == 1
    assert row.labels == {"cluster": "us-central2"}


def test_scrape_identity_is_authoritative_over_a_colliding_metric_label(name):
    # A metric carrying its own `worker` label cannot spoof the job identity.
    telltale.counter(name, "d", ["worker"]).labels("evil").inc()

    row = _one(f"{name}_total", telltale.scrape_metrics(telltale.MetricIdentity(worker="real"), _TS))

    assert row.worker == "real"
    assert row.labels["worker"] == "evil"  # the raw label survives; the column is authoritative


def test_scrape_keeps_metric_source_and_run_labels(name, clean_global_labels):
    telltale.counter(name, "d", ["source", "run"]).labels("local_cache_hit", "leg-3").inc(9)
    telltale.set_global_labels(source="vllm", run="serve-1")

    row = _one(f"{name}_total", telltale.scrape_metrics(telltale.MetricIdentity(), _TS))

    assert row.source == "vllm" and row.run == "serve-1"
    assert row.labels == {"source": "local_cache_hit", "run": "leg-3"}


def test_scrape_keeps_sibling_series_distinct_by_metric_source(name, clean_global_labels):
    counter = telltale.counter(name, "d", ["source"])
    counter.labels("local_compute").inc(9)
    counter.labels("local_cache_hit").inc(4)
    telltale.set_global_labels(source="vllm")

    rows = [r for r in telltale.scrape_metrics(telltale.MetricIdentity(), _TS) if r.name == f"{name}_total"]

    assert {r.labels["source"]: r.value for r in rows} == {"local_compute": 9.0, "local_cache_hit": 4.0}
    assert all(r.source == "vllm" for r in rows)


def test_scrape_ignores_metric_source_and_run_without_globals(clean_global_labels):
    telltale.counter("levanter_test_own_identity", "d", ["source", "run"]).labels("evil", "spoofed").inc()

    row = _one("levanter_test_own_identity_total", telltale.scrape_metrics(telltale.MetricIdentity(), _TS))

    assert row.source == "levanter"
    assert row.run is None
    assert row.labels == {"source": "evil", "run": "spoofed"}


def test_scrape_metric_label_wins_over_global_label(name, clean_global_labels):
    telltale.counter(name, "d", ["cluster"]).labels("us-east-02a").inc()
    telltale.set_global_labels(cluster="us-central2")

    row = _one(f"{name}_total", telltale.scrape_metrics(telltale.MetricIdentity(), _TS))

    assert row.labels == {"cluster": "us-east-02a"}


def test_scrape_drops_created_and_keeps_histogram_le_in_the_map(name):
    telltale.counter(name, "d").inc()
    telltale.histogram(f"{name}_h", "d").observe(0.5)

    rows = telltale.scrape_metrics(telltale.MetricIdentity(), _TS)
    names = {r.name for r in rows}

    assert f"{name}_total" in names
    assert f"{name}_created" not in names
    buckets = [r for r in rows if r.name == f"{name}_h_bucket"]
    assert buckets and "+Inf" in {r.labels["le"] for r in buckets}


def test_start_forwarding_pushes_to_the_sink_and_is_idempotent(name, reset_forwarding, clean_global_labels):
    telltale.gauge(name, "d").set(9.0)
    telltale.set_global_labels(source="levanter")
    sink = _RecordingSink()

    # A long interval keeps the daemon thread from scraping on its own; stop
    # forces the final scrape + close deterministically.
    assert telltale.start_forwarding(sink, identity=telltale.MetricIdentity(job_id="/a/b"), interval=1000.0) is True
    # Second call while running is a no-op.
    assert telltale.start_forwarding(_RecordingSink(), interval=1000.0) is False

    telltale.stop_forwarding()

    assert sink.closed
    row = _one(name, [r for batch in sink.batches for r in batch])
    assert row.value == 9.0 and row.source == "levanter" and row.job_id == "/a/b"
