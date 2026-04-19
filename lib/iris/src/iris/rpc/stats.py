# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process RPC statistics collector.

Tracks per-method call counts, a fixed-bucket latency histogram, and two
bounded ring buffers of sampled calls:
- ``slow_samples``: last N calls whose duration exceeded the slow threshold.
- ``discovery_samples``: at most one call per method per interval regardless
  of latency, so operators can see what a typical request looks like.

Everything lives in memory on the process that owns the collector; stats
reset when that process restarts. Designed to be cheap on the hot path:
the per-call recording is O(log buckets) + a couple of deque pushes under
a single lock.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from collections.abc import Mapping

from connectrpc.request import RequestContext
from google.protobuf.json_format import MessageToJson
from google.protobuf.message import Message

from iris.rpc import stats_pb2, time_pb2
from iris.rpc.auth import get_verified_identity

logger = logging.getLogger(__name__)

# Bucket upper bounds in milliseconds. A trailing 0 sentinel means "+inf".
# Kept static so the dashboard can render consistent histograms across
# restarts without needing to read them back.
BUCKET_UPPER_BOUNDS_MS: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 0)

# Default sample ring sizes. Tuned so the whole structure stays <1 MB even
# with many methods; request previews are capped separately below.
DEFAULT_SLOW_SAMPLES = 50
DEFAULT_DISCOVERY_SAMPLES = 20
DEFAULT_DISCOVERY_INTERVAL_S = 30.0
DEFAULT_REQUEST_PREVIEW_BYTES = 1024


@dataclass(frozen=True, slots=True)
class CallSample:
    """One recorded RPC invocation."""

    method: str
    timestamp_ms: int
    duration_ms: float
    peer: str = ""
    user_agent: str = ""
    caller: str = ""
    error_code: str = ""
    error_message: str = ""
    request_preview: str = ""


@dataclass(slots=True)
class _MethodState:
    count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    last_call_ms: int = 0
    # Parallel array to BUCKET_UPPER_BOUNDS_MS.
    bucket_counts: list[int] = field(default_factory=lambda: [0] * len(BUCKET_UPPER_BOUNDS_MS))
    last_discovery_ms: int = 0


class RpcStatsCollector:
    """Thread-safe RPC call aggregator.

    The collector exposes two entry points: ``record()`` from the
    interceptor hot path, and ``snapshot_proto()`` for consumers. All
    mutations hold a single lock; contention is expected to be tiny
    (recording is a handful of arithmetic ops).
    """

    def __init__(
        self,
        *,
        slow_threshold_ms: float,
        slow_samples: int = DEFAULT_SLOW_SAMPLES,
        discovery_samples: int = DEFAULT_DISCOVERY_SAMPLES,
        discovery_interval_s: float = DEFAULT_DISCOVERY_INTERVAL_S,
        request_preview_bytes: int = DEFAULT_REQUEST_PREVIEW_BYTES,
    ):
        self._slow_threshold_ms = slow_threshold_ms
        self._discovery_interval_ms = int(discovery_interval_s * 1000)
        self._request_preview_bytes = request_preview_bytes
        self._lock = threading.Lock()
        self._methods: dict[str, _MethodState] = {}
        self._slow: deque[CallSample] = deque(maxlen=slow_samples)
        self._discovery: deque[CallSample] = deque(maxlen=discovery_samples)
        self._started_at_ms = int(time.time() * 1000)

    # -- Hot path ------------------------------------------------------

    def record(
        self,
        *,
        method: str,
        duration_ms: float,
        request: Message | None = None,
        ctx: RequestContext | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        """Record a single RPC call. Safe to call from any thread."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            state = self._methods.get(method)
            if state is None:
                state = _MethodState()
                self._methods[method] = state
            state.count += 1
            if error_code:
                state.error_count += 1
            state.total_duration_ms += duration_ms
            if duration_ms > state.max_duration_ms:
                state.max_duration_ms = duration_ms
            state.last_call_ms = now_ms
            _bump_bucket(state.bucket_counts, duration_ms)

            is_slow = duration_ms >= self._slow_threshold_ms or bool(error_code)
            is_discovery = (now_ms - state.last_discovery_ms) >= self._discovery_interval_ms
            if not (is_slow or is_discovery):
                return
            sample = self._build_sample(
                method=method,
                timestamp_ms=now_ms,
                duration_ms=duration_ms,
                request=request,
                ctx=ctx,
                error_code=error_code,
                error_message=error_message,
            )
            if is_slow:
                self._slow.append(sample)
            if is_discovery:
                self._discovery.append(sample)
                state.last_discovery_ms = now_ms

    def _build_sample(
        self,
        *,
        method: str,
        timestamp_ms: int,
        duration_ms: float,
        request: Message | None,
        ctx: RequestContext | None,
        error_code: str,
        error_message: str,
    ) -> CallSample:
        peer, user_agent = _extract_call_metadata(ctx)
        identity = get_verified_identity()
        caller = identity.user_id if identity is not None else ""
        preview = _render_preview(request, self._request_preview_bytes)
        return CallSample(
            method=method,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            peer=peer,
            user_agent=user_agent,
            caller=caller,
            error_code=error_code,
            error_message=_truncate(error_message, 512),
            request_preview=preview,
        )

    # -- Readout -------------------------------------------------------

    def snapshot_proto(self) -> stats_pb2.GetRpcStatsResponse:
        """Return a protobuf snapshot of current stats."""
        with self._lock:
            methods = [_method_to_proto(name, state) for name, state in self._methods.items()]
            slow = [_sample_to_proto(s) for s in self._slow]
            discovery = [_sample_to_proto(s) for s in self._discovery]
            started = time_pb2.Timestamp(epoch_ms=self._started_at_ms)
        methods.sort(key=lambda m: m.method)
        return stats_pb2.GetRpcStatsResponse(
            methods=methods,
            slow_samples=slow,
            discovery_samples=discovery,
            collector_started_at=started,
        )


def _bump_bucket(counts: list[int], duration_ms: float) -> None:
    for i, upper in enumerate(BUCKET_UPPER_BOUNDS_MS):
        if upper == 0 or duration_ms <= upper:
            counts[i] += 1
            return


def _percentile_ms(counts: list[int], pct: float) -> float:
    """Estimate a percentile from bucket counts via linear interpolation.

    The sentinel +inf bucket returns its lower bound (last finite upper).
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    target = pct / 100.0 * total
    cumulative = 0
    lower = 0.0
    for i, upper in enumerate(BUCKET_UPPER_BOUNDS_MS):
        prev_cum = cumulative
        cumulative += counts[i]
        if cumulative >= target:
            if upper == 0:
                # +inf bucket: report the lower bound, we can't do better.
                return lower
            in_bucket = counts[i]
            if in_bucket == 0:
                return float(upper)
            frac = (target - prev_cum) / in_bucket
            return lower + frac * (upper - lower)
        lower = float(upper) if upper != 0 else lower
    return lower


def _method_to_proto(name: str, state: _MethodState) -> stats_pb2.RpcMethodStats:
    return stats_pb2.RpcMethodStats(
        method=name,
        count=state.count,
        error_count=state.error_count,
        total_duration_ms=state.total_duration_ms,
        max_duration_ms=state.max_duration_ms,
        p50_ms=_percentile_ms(state.bucket_counts, 50),
        p95_ms=_percentile_ms(state.bucket_counts, 95),
        p99_ms=_percentile_ms(state.bucket_counts, 99),
        bucket_upper_bounds_ms=list(BUCKET_UPPER_BOUNDS_MS),
        bucket_counts=list(state.bucket_counts),
        last_call=time_pb2.Timestamp(epoch_ms=state.last_call_ms),
    )


def _sample_to_proto(sample: CallSample) -> stats_pb2.RpcCallSample:
    return stats_pb2.RpcCallSample(
        method=sample.method,
        timestamp=time_pb2.Timestamp(epoch_ms=sample.timestamp_ms),
        duration_ms=sample.duration_ms,
        peer=sample.peer,
        user_agent=sample.user_agent,
        caller=sample.caller,
        error_code=sample.error_code,
        error_message=sample.error_message,
        request_preview=sample.request_preview,
    )


def _extract_call_metadata(ctx: RequestContext | None) -> tuple[str, str]:
    if ctx is None:
        return ("", "")
    try:
        headers: Mapping[str, str] = ctx.request_headers()
    except Exception:
        return ("", "")
    user_agent = headers.get("user-agent", "") or headers.get("grpc-user-agent", "")
    # x-forwarded-for may be a comma-separated chain; take the first hop.
    forwarded = headers.get("x-forwarded-for", "")
    peer = forwarded.split(",", 1)[0].strip() if forwarded else headers.get("x-real-ip", "")
    return (peer, user_agent)


def _render_preview(request: Message | None, max_bytes: int) -> str:
    if request is None:
        return ""
    try:
        rendered = MessageToJson(request, preserving_proto_field_name=True, indent=None)
    except Exception:
        return ""
    return _truncate(rendered, max_bytes)


def _truncate(text: str, max_bytes: int) -> str:
    if len(text) <= max_bytes:
        return text
    return text[:max_bytes] + "…"
