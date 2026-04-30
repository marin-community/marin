# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``iris.rpc.stats`` and the StatsService wrapper.

These cover the shapes and invariants that the dashboard relies on:
histogram buckets add up, slow samples are bounded, discovery is throttled,
and the service returns a populated snapshot.
"""

from unittest.mock import Mock

from iris.rpc import job_pb2, stats_pb2
from iris.rpc.stats import (
    BUCKET_UPPER_BOUNDS_MS,
    RpcStatsCollector,
)
from iris.rpc.stats_service import RpcStatsService


def _request() -> job_pb2.GetProcessStatusRequest:
    return job_pb2.GetProcessStatusRequest(max_log_lines=5, log_substring="boom")


def _ctx() -> Mock:
    ctx = Mock()
    ctx.request_headers.return_value = {"user-agent": "iris-cli/1.2.3", "x-forwarded-for": "10.0.0.7"}
    return ctx


def test_collector_records_counts_and_histogram():
    collector = RpcStatsCollector(slow_threshold_ms=1000)

    for d in (0.5, 3, 12, 70, 900, 1500, 200000):
        collector.record(method="ListJobs", duration_ms=d, request=_request(), ctx=_ctx())

    snap = collector.snapshot_proto()
    assert len(snap.methods) == 1
    m = snap.methods[0]
    assert m.method == "ListJobs"
    assert m.count == 7
    assert m.error_count == 0
    # Buckets echoed for the UI and sum to count.
    assert list(m.bucket_upper_bounds_ms) == list(BUCKET_UPPER_BOUNDS_MS)
    assert sum(m.bucket_counts) == 7
    # 0.5ms lands in the first bucket (≤1ms); 200000ms is above the largest
    # finite bound (~52s) and falls into the +inf sentinel bucket.
    assert m.bucket_counts[0] == 1
    assert m.bucket_counts[-1] == 1
    # p99 picks up the large tail; p50 stays small.
    assert m.p99_ms >= m.p95_ms >= m.p50_ms
    assert m.max_duration_ms == 200000


def test_collector_captures_slow_samples_and_respects_bound():
    collector = RpcStatsCollector(slow_threshold_ms=1000, slow_samples_per_method=3, discovery_samples_per_method=0)

    for i in range(5):
        collector.record(method="ListJobs", duration_ms=2000 + i, request=_request(), ctx=_ctx())

    snap = collector.snapshot_proto()
    assert len(snap.slow_samples) == 3
    # deque drops oldest first → last three are preserved in order.
    durations = [s.duration_ms for s in snap.slow_samples]
    assert durations == [2002, 2003, 2004]
    sample = snap.slow_samples[0]
    assert sample.method == "ListJobs"
    assert sample.peer == "10.0.0.7"
    assert sample.user_agent == "iris-cli/1.2.3"
    assert "max_log_lines" in sample.request_preview


def test_collector_records_errors_and_fast_calls_stay_out_of_slow_ring():
    collector = RpcStatsCollector(slow_threshold_ms=1000, discovery_samples_per_method=0)

    collector.record(method="LaunchJob", duration_ms=5, request=_request(), ctx=_ctx())
    collector.record(
        method="LaunchJob",
        duration_ms=10,
        request=_request(),
        ctx=_ctx(),
        error_code="INTERNAL",
        error_message="boom",
    )

    snap = collector.snapshot_proto()
    (m,) = snap.methods
    assert m.count == 2
    assert m.error_count == 1
    # Fast success is not captured in slow; fast failure IS (errors always sampled).
    assert len(snap.slow_samples) == 1
    assert snap.slow_samples[0].error_code == "INTERNAL"
    assert snap.slow_samples[0].error_message == "boom"


def test_discovery_samples_throttled_per_method():
    collector = RpcStatsCollector(
        slow_threshold_ms=1000,
        slow_samples_per_method=0,
        discovery_samples_per_method=10,
        discovery_interval=3600,
    )

    for _ in range(5):
        collector.record(method="ListJobs", duration_ms=1, request=_request(), ctx=_ctx())
    collector.record(method="LaunchJob", duration_ms=1, request=_request(), ctx=_ctx())

    snap = collector.snapshot_proto()
    # First call per method captures one sample; the throttle gate blocks the
    # remaining four ListJobs calls within the 1-hour window.
    methods = sorted(s.method for s in snap.discovery_samples)
    assert methods == ["LaunchJob", "ListJobs"]


def test_discovery_samples_capture_after_interval_elapses():
    collector = RpcStatsCollector(
        slow_threshold_ms=1000,
        slow_samples_per_method=0,
        discovery_samples_per_method=10,
        discovery_interval=0,  # never throttle
    )

    for _ in range(3):
        collector.record(method="ListJobs", duration_ms=1, request=_request(), ctx=_ctx())

    snap = collector.snapshot_proto()
    assert len(snap.discovery_samples) == 3


def test_stats_service_returns_snapshot():
    collector = RpcStatsCollector(slow_threshold_ms=1000)
    collector.record(method="ListJobs", duration_ms=42, request=_request(), ctx=_ctx())

    service = RpcStatsService(collector)
    resp = service.get_rpc_stats(stats_pb2.GetRpcStatsRequest(), ctx=Mock())

    assert isinstance(resp, stats_pb2.GetRpcStatsResponse)
    assert len(resp.methods) == 1
    assert resp.methods[0].method == "ListJobs"
    assert resp.collector_started_at.epoch_ms > 0
