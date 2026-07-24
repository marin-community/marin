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
        self.requests += series.requests
        self.in_flight += series.in_flight
        self.latency_count += series.latency_count
        self.latency_sum_seconds += series.latency_sum_seconds
        for status, count in series.responses.items():
            self.responses[status] = self.responses.get(status, 0) + count
        for bound, count in series.latency_buckets:
            self.latency_buckets[bound] = self.latency_buckets.get(bound, 0) + count


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
            snapshots: list[dict] = [json.loads(proxy.rpc_metrics_json) for proxy in self._proxies]
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
