# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trips a single LogEntry through Finelog: push, then fetch_logs to
verify the row is readable. Push-only would miss "writes accepted but
indexer wedged" failure modes."""

from __future__ import annotations

import logging
import time
import uuid

from finelog.client.log_client import LogClient
from finelog.rpc import logging_pb2

from probes.probe import ErrorClass, ProbeOutcome, ProbeResult

logger = logging.getLogger(__name__)

_PROBE_SOURCE = "/canary/finelog-write-probe"
_PROBE_NAMESPACE = "marin.canary.probe"


class FinelogWrite:
    """Push a unique-nonce LogEntry, then fetch_logs by source and require
    the nonce to be present in the response."""

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._client = LogClient.connect(endpoint)

    def run(self, deadline_seconds: float) -> ProbeResult:
        nonce = uuid.uuid4().hex
        ts_ms = int(time.time() * 1000)

        push_start = time.monotonic()
        try:
            entry = logging_pb2.LogEntry(
                timestamp=logging_pb2.Timestamp(epoch_ms=ts_ms),
                source=_PROBE_SOURCE,
                data=nonce,
                level=logging_pb2.LOG_LEVEL_INFO,
                key=nonce,
            )
            self._client.write_batch(key=_PROBE_NAMESPACE, messages=[entry])
        except TimeoutError as exc:
            return ProbeResult(
                outcome=ProbeOutcome.TIMEOUT,
                error_class=ErrorClass.TIMEOUT,
                error_detail=f"push timeout: {exc}",
            )
        except Exception as exc:
            return ProbeResult.remote_error(_classify_http(exc), f"push: {type(exc).__name__}: {exc}")
        push_latency_ms = int((time.monotonic() - push_start) * 1000)

        # LogClient's read timeout is set at construction; if we're already
        # past the deadline, the daemon will still record TIMEOUT after the
        # fact via its hard-deadline enforcement.
        readback_start = time.monotonic()
        try:
            request = logging_pb2.FetchLogsRequest(
                source=_PROBE_SOURCE,
                since_ms=ts_ms - 1_000,
                max_lines=64,
            )
            response = self._client.fetch_logs(request)
        except TimeoutError as exc:
            return ProbeResult(
                outcome=ProbeOutcome.TIMEOUT,
                error_class=ErrorClass.TIMEOUT,
                error_detail=f"readback timeout: {exc}",
                extras={"push_latency_ms": push_latency_ms},
            )
        except Exception as exc:
            return ProbeResult.remote_error(
                _classify_http(exc),
                f"readback: {type(exc).__name__}: {exc}",
                extras={"push_latency_ms": push_latency_ms},
            )
        readback_latency_ms = int((time.monotonic() - readback_start) * 1000)

        extras: dict[str, str | int | float] = {
            "push_latency_ms": push_latency_ms,
            "readback_latency_ms": readback_latency_ms,
        }

        if any(e.data == nonce for e in response.entries):
            return ProbeResult.success(extras=extras)

        return ProbeResult.remote_error(
            ErrorClass.READBACK_MISMATCH,
            f"nonce {nonce} not found in {len(response.entries)} returned entries",
            extras=extras,
        )

    def close(self) -> None:
        self._client.close()


def _classify_http(exc: Exception) -> ErrorClass:
    name = type(exc).__name__
    if name == "ConnectError":
        return ErrorClass.HTTP_ERROR
    if isinstance(exc, ConnectionError):
        return ErrorClass.CONNECT_ERROR
    return ErrorClass.HTTP_ERROR
