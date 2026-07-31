# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export native Iris RPC and proxy snapshots through direct telemetry.

Counters reset when the native proxy restarts. Rate queries must discard
negative deltas across that reset boundary. Processes hosting multiple local
controllers export the sum of their native proxies as one process snapshot.
"""

import json
import logging
import threading
from dataclasses import dataclass, field

from rigging import telemetry

from iris.cluster.controller.native_proxy import NativeProxy

IN_FLIGHT_METRIC_NAME = "rpc_in_flight"
DEFAULT_POLL_INTERVAL = 15.0

logger = logging.getLogger(__name__)


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


PROXY_IN_FLIGHT_METRIC_NAME = "proxy_in_flight"


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


def _publish(name: str, value: float, attributes: dict[str, str], source_kind: str) -> None:
    temporality = "current_snapshot" if source_kind == "gauge" else "cumulative_snapshot"
    telemetry.gauge(name).set(
        value,
        attributes={**attributes, "source_kind": source_kind, "source_temporality": temporality},
    )


class NativeProxyTelemetry:
    """Aggregate local native proxies and publish source-typed snapshots."""

    def __init__(self, *, interval: float = DEFAULT_POLL_INTERVAL) -> None:
        self._proxies: list[NativeProxy] = []
        self._lock = threading.Lock()
        self._interval = interval
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def attach(self, proxy: NativeProxy) -> None:
        with self._lock:
            if any(current is proxy for current in self._proxies):
                return
            self._proxies.append(proxy)
            if self._thread is None:
                stop = threading.Event()
                self._stop = stop
                self._thread = threading.Thread(
                    target=self._run,
                    args=(stop,),
                    name="iris-native-telemetry",
                    daemon=True,
                )
                self._thread.start()

    def detach(self, proxy: NativeProxy) -> None:
        with self._lock:
            self._proxies = [current for current in self._proxies if current is not proxy]
            if self._proxies or self._thread is None:
                return
            thread, self._thread = self._thread, None
            stop, self._stop = self._stop, None
            assert stop is not None
            stop.set()
        thread.join(timeout=5.0)

    def publish_once(self) -> None:
        with self._lock:
            rpc_snapshots: list[dict] = [json.loads(proxy.rpc_metrics_json) for proxy in self._proxies]
            proxy_snapshots: list[dict] = [json.loads(proxy.proxy_metrics_json) for proxy in self._proxies]
        self._publish_rpc(rpc_snapshots)
        self._publish_proxy(proxy_snapshots)

    def _run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.publish_once()
            except Exception:
                # Native metrics remain best-effort and must not affect controller service.
                try:
                    logger.warning("could not read native proxy telemetry snapshot", exc_info=True)
                except Exception:
                    pass
            if stop.wait(self._interval):
                return

    def _publish_rpc(self, snapshots: list[dict]) -> None:
        """Publish Connect-RPC snapshots keyed by service/method/upstream."""
        aggregated: dict[_RpcMetricSeriesKey, _AggregatedRpcMetricSeries] = {}
        for snapshot in snapshots:
            for raw_series in snapshot["series"]:
                series = _RpcMetricSeries(**raw_series)
                key = _RpcMetricSeriesKey(series.service, series.method, series.upstream)
                aggregated.setdefault(key, _AggregatedRpcMetricSeries()).add(series)

        for key, series in aggregated.items():
            labels = {
                "service": key.service,
                "method": key.method,
                "upstream": key.upstream,
            }
            _publish("rpc_requests_total", series.requests, labels, "counter")
            _publish(IN_FLIGHT_METRIC_NAME, series.in_flight, labels, "gauge")
            for status, count in series.responses.items():
                _publish("rpc_responses_total", count, {**labels, "status": status}, "counter")
            for bound, count in series.latency_buckets.items():
                _publish("rpc_duration_seconds_bucket", count, {**labels, "le": bound}, "histogram")
            _publish("rpc_duration_seconds_sum", series.latency_sum_seconds, labels, "histogram")
            _publish("rpc_duration_seconds_count", series.latency_count, labels, "histogram")

    def _publish_proxy(self, snapshots: list[dict]) -> None:
        """Publish proxy transport load keyed by endpoint/method/route_kind.

        Distinct from `iris_rpc_*` — a proxied Connect call appears in both and the
        two must not be summed. Each snapshot carries an exact ``aggregate`` (emitted
        as ``scope=total``) plus the bounded per-endpoint ``series`` (``scope=endpoint``).

        Emits nothing when no proxy provided a snapshot.
        """
        if not snapshots:
            return
        total = _AggregatedProxyMetricSeries()
        by_endpoint: dict[_ProxyMetricSeriesKey, _AggregatedProxyMetricSeries] = {}
        for snapshot in snapshots:
            total.add(_ProxyMetricSeries(**snapshot["aggregate"]))
            for raw_series in snapshot["series"]:
                series = _ProxyMetricSeries(**raw_series)
                key = _ProxyMetricSeriesKey(series.endpoint, series.method, series.route_kind)
                by_endpoint.setdefault(key, _AggregatedProxyMetricSeries()).add(series)

        def emit(labels: dict[str, str], agg: _AggregatedProxyMetricSeries) -> None:
            _publish("proxy_requests_total", agg.requests, labels, "counter")
            _publish(PROXY_IN_FLIGHT_METRIC_NAME, agg.in_flight, labels, "gauge")
            _publish("proxy_request_bytes_total", agg.request_bytes, labels, "counter")
            _publish("proxy_response_bytes_total", agg.response_bytes, labels, "counter")
            for status, count in agg.responses.items():
                _publish("proxy_responses_total", count, {**labels, "status": status}, "counter")
            for bound, count in agg.latency_buckets.items():
                _publish("proxy_duration_seconds_bucket", count, {**labels, "le": bound}, "histogram")
            _publish("proxy_duration_seconds_sum", agg.latency_sum_seconds, labels, "histogram")
            _publish("proxy_duration_seconds_count", agg.latency_count, labels, "histogram")

        emit({"scope": "total"}, total)
        for key, agg in by_endpoint.items():
            emit(
                {"scope": "endpoint", "endpoint": key.endpoint, "method": key.method, "route_kind": key.route_kind},
                agg,
            )


_PUBLISHER = NativeProxyTelemetry()


def install_native_proxy_metrics(proxy: NativeProxy) -> NativeProxyTelemetry:
    """Add ``proxy`` to the process's Iris controller telemetry snapshot."""
    _PUBLISHER.attach(proxy)
    return _PUBLISHER


def uninstall_native_proxy_metrics(proxy: NativeProxy) -> None:
    """Stop exposing a native proxy after its controller shuts down."""
    _PUBLISHER.detach(proxy)
