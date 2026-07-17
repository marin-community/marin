# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
from rigging import telltale
from rigging.server_auth import RequestAuthPolicy, RouteAuthMiddleware
from starlette.applications import Starlette
from starlette.testclient import TestClient

_names = itertools.count()


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


def test_counter_is_get_or_create(name):
    telltale.counter(name, "d", ["route"]).labels("/a").inc(2)
    telltale.counter(name, "d", ["route"]).labels("/a").inc()

    assert telltale.counter(name, "d", ["route"]).labels("/a")._value.get() == 3


def test_reregistering_with_a_different_type_raises(name):
    telltale.counter(name, "d")

    with pytest.raises(ValueError, match="already registered as Counter"):
        telltale.gauge(name, "d")


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

    matching = [(family, kind, s) for family, kind, s in telltale.samples() if family == name]

    assert [(kind, s.value) for _, kind, s in matching] == [("gauge", 7.0)]


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
