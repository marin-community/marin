# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose native Iris RPC metrics through the controller Telltale registry.

Counters reset when the native proxy restarts. Rate queries must discard
negative deltas across that reset boundary.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass

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


class NativeProxyMetricsCollector(Collector):
    """A Telltale bridge that never owns or mutates native metric values."""

    def __init__(self, proxy: NativeProxy) -> None:
        self._proxy = proxy

    def collect(self) -> Iterable[Metric]:
        snapshot: dict = json.loads(self._proxy.rpc_metrics_json)
        requests = Metric("iris_rpc_requests", "Iris RPC requests handled by the native proxy", "counter")
        responses = Metric("iris_rpc_responses", "Iris RPC responses returned by the native proxy", "counter")
        in_flight = Metric(IN_FLIGHT_METRIC_NAME, "Iris RPC requests currently handled by the native proxy", "gauge")
        duration = Metric("iris_rpc_duration_seconds", "Iris RPC native-proxy latency", "histogram")
        for raw_series in snapshot["series"]:
            series = _RpcMetricSeries(**raw_series)
            labels = {
                "service": series.service,
                "method": series.method,
                "upstream": series.upstream,
            }
            requests.add_sample("iris_rpc_requests_total", labels=labels, value=series.requests)
            in_flight.add_sample(IN_FLIGHT_METRIC_NAME, labels=labels, value=series.in_flight)
            for status, count in series.responses.items():
                responses.add_sample("iris_rpc_responses_total", labels={**labels, "status": status}, value=count)
            for bound, count in series.latency_buckets:
                duration.add_sample("iris_rpc_duration_seconds_bucket", labels={**labels, "le": bound}, value=count)
            duration.add_sample("iris_rpc_duration_seconds_sum", labels=labels, value=series.latency_sum_seconds)
            duration.add_sample("iris_rpc_duration_seconds_count", labels=labels, value=series.latency_count)
        return (requests, responses, in_flight, duration)


def install_native_proxy_metrics(proxy: NativeProxy) -> NativeProxyMetricsCollector:
    """Make ``proxy`` the source for the process's Iris RPC Telltale series."""
    collector = NativeProxyMetricsCollector(proxy)
    telltale.register_collector(collector)
    telltale.set_global_labels(source="iris")
    return collector
