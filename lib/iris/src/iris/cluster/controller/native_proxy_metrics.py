# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose Rust-owned Iris RPC metrics through the controller Telltale registry.

The native proxy owns all state. This collector only snapshots its Prometheus
shape so Telltale can forward it to finelog alongside the controller's other
measurements. Counter resets therefore coincide with native-proxy restarts;
readers must compute rates between consecutive samples and discard negative
deltas.
"""

import json
from collections.abc import Iterable
from typing import Protocol

from prometheus_client.core import Metric
from prometheus_client.registry import Collector
from rigging import telltale


class _NativeProxy(Protocol):
    @property
    def rpc_metrics_json(self) -> str: ...


class NativeProxyMetricsCollector(Collector):
    """A Telltale bridge that never owns or mutates native metric values."""

    def __init__(self, proxy: _NativeProxy) -> None:
        self._proxy = proxy

    def collect(self) -> Iterable[Metric]:
        snapshot = json.loads(self._proxy.rpc_metrics_json)
        requests = Metric("iris_rpc_requests", "Iris RPC requests handled by the native proxy", "counter")
        responses = Metric("iris_rpc_responses", "Iris RPC responses returned by the native proxy", "counter")
        in_flight = Metric("iris_rpc_in_flight", "Iris RPC requests currently handled by the native proxy", "gauge")
        duration = Metric("iris_rpc_duration_seconds", "Iris RPC native-proxy latency", "histogram")
        for series in snapshot["series"]:
            labels = {
                "service": series["service"],
                "method": series["method"],
                "upstream": series["upstream"],
            }
            requests.add_sample("iris_rpc_requests_total", labels=labels, value=series["requests"])
            in_flight.add_sample("iris_rpc_in_flight", labels=labels, value=series["in_flight"])
            for status, count in series["responses"].items():
                responses.add_sample("iris_rpc_responses_total", labels={**labels, "status": status}, value=count)
            for bound, count in series["latency_buckets"]:
                duration.add_sample("iris_rpc_duration_seconds_bucket", labels={**labels, "le": bound}, value=count)
            duration.add_sample("iris_rpc_duration_seconds_sum", labels=labels, value=series["latency_sum_seconds"])
            duration.add_sample("iris_rpc_duration_seconds_count", labels=labels, value=series["latency_count"])
        return (requests, responses, in_flight, duration)


def install_native_proxy_metrics(proxy: _NativeProxy) -> NativeProxyMetricsCollector:
    """Make ``proxy`` the source for the process's Iris RPC Telltale series."""
    collector = NativeProxyMetricsCollector(proxy)
    telltale.register_collector(collector)
    telltale.set_global_labels(source="iris")
    return collector


def uninstall_native_proxy_metrics(collector: NativeProxyMetricsCollector) -> None:
    """Remove the collector before its native proxy shuts down."""
    telltale.unregister_collector(collector)
