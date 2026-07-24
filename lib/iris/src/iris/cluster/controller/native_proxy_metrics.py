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

    def __init__(self) -> None:
        self._proxy: _NativeProxy | None = None

    def set_proxy(self, proxy: _NativeProxy | None) -> None:
        self._proxy = proxy

    def clear_proxy(self, proxy: _NativeProxy) -> None:
        if self._proxy is proxy:
            self._proxy = None

    def collect(self) -> Iterable[Metric]:
        proxy = self._proxy
        if proxy is None:
            return ()
        payload = getattr(proxy, "rpc_metrics_json", None)
        if payload is None:
            return ()
        snapshot = json.loads(payload)
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
            duration.add_sample("iris_rpc_duration_seconds_count", labels=labels, value=series["requests"])
        return (requests, responses, in_flight, duration)


_COLLECTOR = NativeProxyMetricsCollector()


def install_native_proxy_metrics(proxy: _NativeProxy) -> None:
    """Make ``proxy`` the source for the process's Iris RPC Telltale series."""
    _COLLECTOR.set_proxy(proxy)
    telltale.register_collector(_COLLECTOR)
    telltale.set_global_labels(source="iris")


def clear_native_proxy_metrics(proxy: _NativeProxy) -> None:
    """Stop exposing a proxy that has been shut down."""
    _COLLECTOR.clear_proxy(proxy)
