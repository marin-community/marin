# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose native Iris RPC metrics through the process Telltale registry.

Counters reset when the native proxy restarts. Rate queries must discard
negative deltas across that reset boundary. Processes hosting multiple local
controllers expose the sum of their native proxies because Prometheus metric
families are process-global.
"""

import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

from prometheus_client.core import Metric
from prometheus_client.registry import Collector
from rigging import telltale

from iris.cluster.controller.native_proxy import NativeProxy

IN_FLIGHT_METRIC_NAME = "iris_rpc_in_flight"


@dataclass(frozen=True)
class _RpcMetricSeries:
    service: str
    method: str
    upstream: str
    requests: int
    responses: dict[str, int]
    in_flight: int
    latency_buckets: list[tuple[str, int]]
    latency_count: int
    latency_sum_seconds: float


@dataclass(frozen=True)
class _RpcMetricSeriesKey:
    service: str
    method: str
    upstream: str


@dataclass
class _AggregatedRpcMetricSeries:
    requests: int = 0
    responses: dict[str, int] = field(default_factory=dict)
    in_flight: int = 0
    latency_buckets: dict[str, int] = field(default_factory=dict)
    latency_count: int = 0
    latency_sum_seconds: float = 0.0

    def add(self, series: _RpcMetricSeries) -> None:
        _merge_shared_counters(self, series)


PROXY_IN_FLIGHT_METRIC_NAME = "iris_proxy_in_flight"


@dataclass(frozen=True)
class _ProxyMetricSeries:
    endpoint: str
    method: str
    route_kind: str
    requests: int
    responses: dict[str, int]
    in_flight: int
    latency_buckets: list[tuple[str, int]]
    latency_count: int
    latency_sum_seconds: float
    request_bytes: int
    response_bytes: int


@dataclass(frozen=True)
class _ProxyMetricSeriesKey:
    endpoint: str
    method: str
    route_kind: str


@dataclass
class _AggregatedProxyMetricSeries:
    requests: int = 0
    responses: dict[str, int] = field(default_factory=dict)
    in_flight: int = 0
    latency_buckets: dict[str, int] = field(default_factory=dict)
    latency_count: int = 0
    latency_sum_seconds: float = 0.0
    request_bytes: int = 0
    response_bytes: int = 0

    def add(self, series: _ProxyMetricSeries) -> None:
        _merge_shared_counters(self, series)
        self.request_bytes += series.request_bytes
        self.response_bytes += series.response_bytes


def _merge_shared_counters(
    agg: _AggregatedRpcMetricSeries | _AggregatedProxyMetricSeries,
    series: _RpcMetricSeries | _ProxyMetricSeries,
) -> None:
    """Fold the counters shared by the RPC and proxy families into ``agg``.

    Proxy-only byte counters are merged by the caller.
    """
    agg.requests += series.requests
    agg.in_flight += series.in_flight
    agg.latency_count += series.latency_count
    agg.latency_sum_seconds += series.latency_sum_seconds
    for status, count in series.responses.items():
        agg.responses[status] = agg.responses.get(status, 0) + count
    for bound, count in series.latency_buckets:
        agg.latency_buckets[bound] = agg.latency_buckets.get(bound, 0) + count


class NativeProxyMetricsCollector(Collector):
    """Collect native RPC counters, gauges, and histograms for Telltale."""

    def __init__(self) -> None:
        self._proxies: list[NativeProxy] = []
        self._lock = threading.Lock()

    def attach(self, proxy: NativeProxy) -> None:
        with self._lock:
            if any(current is proxy for current in self._proxies):
                return
            self._proxies.append(proxy)

    def detach(self, proxy: NativeProxy) -> None:
        with self._lock:
            self._proxies = [current for current in self._proxies if current is not proxy]

    def collect(self) -> Iterable[Metric]:
        with self._lock:
            rpc_snapshots: list[dict] = [json.loads(proxy.rpc_metrics_json) for proxy in self._proxies]
            proxy_snapshots: list[dict] = [json.loads(proxy.proxy_metrics_json) for proxy in self._proxies]
        return (*self._collect_rpc(rpc_snapshots), *self._collect_proxy(proxy_snapshots))

    def _collect_rpc(self, snapshots: list[dict]) -> tuple[Metric, ...]:
        """Connect-RPC counters keyed by service/method/upstream (`iris_rpc_*`)."""
        aggregated: dict[_RpcMetricSeriesKey, _AggregatedRpcMetricSeries] = {}
        for snapshot in snapshots:
            for raw_series in snapshot["series"]:
                series = _RpcMetricSeries(**raw_series)
                key = _RpcMetricSeriesKey(series.service, series.method, series.upstream)
                aggregated.setdefault(key, _AggregatedRpcMetricSeries()).add(series)

        requests = Metric("iris_rpc_requests", "Iris RPC requests handled by the native proxy", "counter")
        responses = Metric("iris_rpc_responses", "Iris RPC responses returned by the native proxy", "counter")
        in_flight = Metric(IN_FLIGHT_METRIC_NAME, "Iris RPC requests currently handled by the native proxy", "gauge")
        duration = Metric("iris_rpc_duration_seconds", "Iris RPC native-proxy latency", "histogram")
        for key, series in aggregated.items():
            labels = {
                "service": key.service,
                "method": key.method,
                "upstream": key.upstream,
            }
            requests.add_sample("iris_rpc_requests_total", labels=labels, value=series.requests)
            in_flight.add_sample(IN_FLIGHT_METRIC_NAME, labels=labels, value=series.in_flight)
            for status, count in series.responses.items():
                responses.add_sample("iris_rpc_responses_total", labels={**labels, "status": status}, value=count)
            for bound, count in series.latency_buckets.items():
                duration.add_sample("iris_rpc_duration_seconds_bucket", labels={**labels, "le": bound}, value=count)
            duration.add_sample("iris_rpc_duration_seconds_sum", labels=labels, value=series.latency_sum_seconds)
            duration.add_sample("iris_rpc_duration_seconds_count", labels=labels, value=series.latency_count)
        return (requests, responses, in_flight, duration)

    def _collect_proxy(self, snapshots: list[dict]) -> tuple[Metric, ...]:
        """Proxy transport load, keyed by endpoint/method/route_kind (`iris_proxy_*`).

        Distinct from `iris_rpc_*` — a proxied Connect call appears in both and the
        two must not be summed. Each snapshot carries an exact ``aggregate`` (emitted
        as ``scope=total``) plus the bounded per-endpoint ``series`` (``scope=endpoint``).
        """
        total = _AggregatedProxyMetricSeries()
        by_endpoint: dict[_ProxyMetricSeriesKey, _AggregatedProxyMetricSeries] = {}
        for snapshot in snapshots:
            total.add(_ProxyMetricSeries(**snapshot["aggregate"]))
            for raw_series in snapshot["series"]:
                series = _ProxyMetricSeries(**raw_series)
                key = _ProxyMetricSeriesKey(series.endpoint, series.method, series.route_kind)
                by_endpoint.setdefault(key, _AggregatedProxyMetricSeries()).add(series)

        requests = Metric("iris_proxy_requests", "Requests the native proxy forwarded upstream", "counter")
        responses = Metric(
            "iris_proxy_responses", "Responses the native proxy returned for forwarded requests", "counter"
        )
        in_flight = Metric(PROXY_IN_FLIGHT_METRIC_NAME, "Requests the native proxy is currently forwarding", "gauge")
        duration = Metric("iris_proxy_duration_seconds", "Native-proxy forwarding latency", "histogram")
        request_bytes = Metric(
            "iris_proxy_request_bytes", "Request body bytes the native proxy read from clients", "counter"
        )
        response_bytes = Metric(
            "iris_proxy_response_bytes", "Response body bytes the native proxy delivered to clients", "counter"
        )

        def emit(labels: dict[str, str], agg: _AggregatedProxyMetricSeries) -> None:
            requests.add_sample("iris_proxy_requests_total", labels=labels, value=agg.requests)
            in_flight.add_sample(PROXY_IN_FLIGHT_METRIC_NAME, labels=labels, value=agg.in_flight)
            request_bytes.add_sample("iris_proxy_request_bytes_total", labels=labels, value=agg.request_bytes)
            response_bytes.add_sample("iris_proxy_response_bytes_total", labels=labels, value=agg.response_bytes)
            for status, count in agg.responses.items():
                responses.add_sample("iris_proxy_responses_total", labels={**labels, "status": status}, value=count)
            for bound, count in agg.latency_buckets.items():
                duration.add_sample("iris_proxy_duration_seconds_bucket", labels={**labels, "le": bound}, value=count)
            duration.add_sample("iris_proxy_duration_seconds_sum", labels=labels, value=agg.latency_sum_seconds)
            duration.add_sample("iris_proxy_duration_seconds_count", labels=labels, value=agg.latency_count)

        # Every sample carries the same label keys so the family stays well-formed;
        # the aggregate leaves endpoint/method/route_kind empty under scope=total.
        emit({"scope": "total", "endpoint": "", "method": "", "route_kind": ""}, total)
        for key, agg in by_endpoint.items():
            emit(
                {"scope": "endpoint", "endpoint": key.endpoint, "method": key.method, "route_kind": key.route_kind},
                agg,
            )
        return (requests, responses, in_flight, duration, request_bytes, response_bytes)


_COLLECTOR = NativeProxyMetricsCollector()


def install_native_proxy_metrics(proxy: NativeProxy) -> NativeProxyMetricsCollector:
    """Add ``proxy`` to the process's Iris RPC Telltale series."""
    _COLLECTOR.attach(proxy)
    telltale.register_collector(_COLLECTOR)
    telltale.set_global_labels(source="iris")
    return _COLLECTOR


def uninstall_native_proxy_metrics(proxy: NativeProxy) -> None:
    """Stop exposing a native proxy after its controller shuts down."""
    _COLLECTOR.detach(proxy)
