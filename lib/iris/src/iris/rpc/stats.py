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
from collections.abc import Mapping
from dataclasses import dataclass

from connectrpc.request import RequestContext
from google.protobuf.json_format import MessageToJson
from google.protobuf.message import Message

from iris.cluster.redaction import redact_json_preview
from iris.rpc import stats_pb2, time_pb2
from iris.rpc.auth import get_verified_identity

logger = logging.getLogger(__name__)


def _make_bucket_bounds() -> tuple[int, ...]:
    """Log-scale bucket upper bounds in ms with a +inf sentinel (trailing 0).

    Buckets sit at 2**(k/3) — three buckets per octave, base ≈ 1.26 — from
    1 ms up to ~60 s. That is an order of magnitude tighter than the legacy
    1/2/5/10/... scale, so percentile estimates stay tight without bloating
    storage (roughly 30 finite buckets). Rounded to integers and deduped.
    """
    bounds: list[int] = []
    for k in range(0, 50):
        b = round(2.0 ** (k / 3.0))
        if b > 60000:
            break
        if not bounds or b > bounds[-1]:
            bounds.append(b)
    bounds.append(0)  # +inf sentinel
    return tuple(bounds)


# A trailing 0 sentinel in the tuple means "+inf". Kept static so the
# dashboard can render consistent histograms across restarts without
# needing to read them back.
BUCKET_UPPER_BOUNDS_MS: tuple[int, ...] = _make_bucket_bounds()

# Default sample ring sizes. Tuned so the whole structure stays <1 MB even
# with many methods; request previews are capped separately below.
DEFAULT_SLOW_SAMPLES = 50
DEFAULT_DISCOVERY_SAMPLES = 20
DEFAULT_DISCOVERY_INTERVAL = 30.0
DEFAULT_REQUEST_PREVIEW_BYTES = 1024


@dataclass(frozen=True, slots=True)
class _CallMetadata:
    """Caller-identifying fields lifted from the RPC headers."""

    peer: str = ""
    user_agent: str = ""


class RpcStatsCollector:
    """Thread-safe RPC call aggregator.

    The collector exposes two entry points: ``record()`` from the
    interceptor hot path, and ``snapshot_proto()`` for consumers. All
    mutations hold a single lock; contention is expected to be tiny
    (recording is a handful of arithmetic ops).

    Internal state is held as ``stats_pb2.RpcMethodStats`` and
    ``stats_pb2.RpcCallSample`` protos directly, mutated in place.
    Percentiles and the echoed bucket-bounds array are filled in at
    snapshot time, not on the hot path.
    """

    def __init__(
        self,
        *,
        slow_threshold_ms: float,
        slow_samples: int = DEFAULT_SLOW_SAMPLES,
        discovery_samples: int = DEFAULT_DISCOVERY_SAMPLES,
        discovery_interval: float = DEFAULT_DISCOVERY_INTERVAL,
        request_preview_bytes: int = DEFAULT_REQUEST_PREVIEW_BYTES,
    ):
        self._slow_threshold_ms = slow_threshold_ms
        self._discovery_interval_ms = int(discovery_interval * 1000)
        self._request_preview_bytes = request_preview_bytes
        self._lock = threading.Lock()
        self._methods: dict[str, stats_pb2.RpcMethodStats] = {}
        self._last_discovery_ms: dict[str, int] = {}
        self._slow: deque[stats_pb2.RpcCallSample] = deque(maxlen=slow_samples)
        self._discovery: deque[stats_pb2.RpcCallSample] = deque(maxlen=discovery_samples)
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
                state = stats_pb2.RpcMethodStats(method=method)
                state.bucket_counts.extend([0] * len(BUCKET_UPPER_BOUNDS_MS))
                self._methods[method] = state
            state.count += 1
            if error_code:
                state.error_count += 1
            state.total_duration_ms += duration_ms
            if duration_ms > state.max_duration_ms:
                state.max_duration_ms = duration_ms
            state.last_call.epoch_ms = now_ms
            _bump_bucket(state.bucket_counts, duration_ms)

            last_discovery_ms = self._last_discovery_ms.get(method, 0)
            is_slow = duration_ms >= self._slow_threshold_ms or bool(error_code)
            is_discovery = (now_ms - last_discovery_ms) >= self._discovery_interval_ms
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
                self._last_discovery_ms[method] = now_ms

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
    ) -> stats_pb2.RpcCallSample:
        meta = _extract_call_metadata(ctx)
        identity = get_verified_identity()
        caller = identity.user_id if identity is not None else ""
        preview = _render_preview(request, self._request_preview_bytes)
        return stats_pb2.RpcCallSample(
            method=method,
            timestamp=time_pb2.Timestamp(epoch_ms=timestamp_ms),
            duration_ms=duration_ms,
            peer=meta.peer,
            user_agent=meta.user_agent,
            caller=caller,
            error_code=error_code,
            error_message=_truncate(error_message, 512),
            request_preview=preview,
        )

    # -- Readout -------------------------------------------------------

    def snapshot_proto(self) -> stats_pb2.GetRpcStatsResponse:
        """Return a protobuf snapshot of current stats."""
        response = stats_pb2.GetRpcStatsResponse(
            collector_started_at=time_pb2.Timestamp(epoch_ms=self._started_at_ms),
        )
        with self._lock:
            for state in self._methods.values():
                m = response.methods.add()
                m.CopyFrom(state)
                m.p50_ms = _percentile_ms(state.bucket_counts, 50)
                m.p95_ms = _percentile_ms(state.bucket_counts, 95)
                m.p99_ms = _percentile_ms(state.bucket_counts, 99)
                m.bucket_upper_bounds_ms.extend(BUCKET_UPPER_BOUNDS_MS)
            response.slow_samples.extend(self._slow)
            response.discovery_samples.extend(self._discovery)
        response.methods.sort(key=lambda m: m.method)
        return response


def _bump_bucket(counts, duration_ms: float) -> None:
    for i, upper in enumerate(BUCKET_UPPER_BOUNDS_MS):
        if upper == 0 or duration_ms <= upper:
            counts[i] += 1
            return


def _percentile_ms(counts, pct: float) -> float:
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


def _extract_call_metadata(ctx: RequestContext | None) -> _CallMetadata:
    if ctx is None:
        return _CallMetadata()
    try:
        headers: Mapping[str, str] = ctx.request_headers()
    except Exception:
        # Stats hot path must not crash the RPC; log so silent breakage
        # is still visible in debug-level server logs.
        logger.debug("Failed to read request headers for stats", exc_info=True)
        return _CallMetadata()
    user_agent = headers.get("user-agent", "") or headers.get("grpc-user-agent", "")
    # x-forwarded-for may be a comma-separated chain; take the first hop.
    forwarded = headers.get("x-forwarded-for", "")
    peer = forwarded.split(",", 1)[0].strip() if forwarded else headers.get("x-real-ip", "")
    return _CallMetadata(peer=peer, user_agent=user_agent)


def _render_preview(request: Message | None, max_bytes: int) -> str:
    if request is None:
        return ""
    try:
        rendered = MessageToJson(request, preserving_proto_field_name=True, indent=None)
    except Exception:
        logger.debug("Failed to render request preview for %s", type(request).__name__, exc_info=True)
        return ""
    rendered = redact_json_preview(rendered)
    return _truncate(rendered, max_bytes)


def _truncate(text: str, max_bytes: int) -> str:
    if len(text) <= max_bytes:
        return text
    return text[:max_bytes] + "…"
