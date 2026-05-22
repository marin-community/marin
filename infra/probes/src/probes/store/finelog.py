# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-effort secondary Finelog writer.

ProbeSamples are mirrored to Finelog under ``marin.canary`` / ``marin.canary.meta``
so existing query tools can read them uniformly. The write is best-effort:
failures are recorded as a *synthetic* ProbeSample (probe_kind="finelog-push")
in the local SQLite store, so a Finelog outage is observable from the same
dataset that everything else lands in.

Without that synthetic recording, a Finelog outage would only appear as the
absence of rows in Finelog — which by definition cannot be seen.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from finelog.client.log_client import LogClient
from finelog.rpc import logging_pb2

from probes.probe import ErrorClass, ProbeOutcome, ProbeSample
from probes.store.sqlite import SqliteSampleStore

logger = logging.getLogger(__name__)


# Namespaces a probe sample can land in. We use "source" (per-row) to scope to
# probe instance; the key is the namespace passed to write_batch.
DEFAULT_PROBE_NAMESPACE = "marin.canary"
DEFAULT_META_NAMESPACE = "marin.canary.meta"


@dataclass(frozen=True)
class FinelogStoreConfig:
    endpoint: str
    probe_namespace: str = DEFAULT_PROBE_NAMESPACE
    meta_namespace: str = DEFAULT_META_NAMESPACE
    push_timeout_ms: int = 10_000


class FinelogSampleStore:
    """Mirrors ProbeSamples to Finelog. Push failures land in SQLite as
    synthetic ``finelog-push`` samples; success path emits nothing extra.

    Construction never fails on Finelog being down — the LogClient itself
    only establishes connections lazily on first write.
    """

    def __init__(self, config: FinelogStoreConfig, fallback_store: SqliteSampleStore):
        self._config = config
        self._fallback = fallback_store
        self._client = LogClient.connect(config.endpoint, timeout_ms=config.push_timeout_ms)

    def write(self, sample: ProbeSample, *, is_meta: bool = False) -> None:
        """Push one sample to Finelog. Best-effort; records a synthetic
        failure sample to SQLite (and logs) on any exception."""
        entry = _sample_to_log_entry(sample)
        key = self._config.meta_namespace if is_meta else self._config.probe_namespace
        try:
            self._client.write_batch(key=key, messages=[entry])
        except Exception as exc:
            self._record_push_failure(sample, exc)

    def _record_push_failure(self, original: ProbeSample, exc: Exception) -> None:
        error_class = _classify(exc)
        synthetic = replace(
            original,
            timestamp=datetime.now(UTC),
            probe_name=f"finelog-push/{original.probe_name}",
            probe_kind="finelog-push",
            location=None,
            outcome=ProbeOutcome.REMOTE_ERROR if error_class is not ErrorClass.TIMEOUT else ProbeOutcome.TIMEOUT,
            latency_ms=0,
            error_class=error_class,
            error_detail=f"{type(exc).__name__}: {exc}",
            target_id=None,
            extras_json=json.dumps({"original_probe": original.probe_name}),
        )
        try:
            self._fallback.write(synthetic)
        except Exception:
            logger.exception("failed to record Finelog push failure to SQLite for %s", original.probe_name)
        logger.warning("Finelog push failed for %s: %s", original.probe_name, exc)

    def close(self) -> None:
        self._client.close()


def _sample_to_log_entry(sample: ProbeSample) -> logging_pb2.LogEntry:
    payload = {
        "timestamp": sample.timestamp.isoformat(),
        "probe_name": sample.probe_name,
        "probe_kind": sample.probe_kind,
        "location": sample.location,
        "outcome": sample.outcome.value,
        "latency_ms": sample.latency_ms,
        "error_class": sample.error_class.value if sample.error_class is not None else None,
        "error_detail": sample.error_detail,
        "target_id": sample.target_id,
        "extras": json.loads(sample.extras_json),
        "daemon_instance": sample.daemon_instance,
    }
    level = logging_pb2.LOG_LEVEL_INFO if sample.outcome is ProbeOutcome.SUCCESS else logging_pb2.LOG_LEVEL_ERROR
    return logging_pb2.LogEntry(
        timestamp=logging_pb2.Timestamp(epoch_ms=int(sample.timestamp.timestamp() * 1000)),
        source=f"/canary/{sample.probe_name}",
        data=json.dumps(payload),
        attempt_id=0,
        level=level,
        key=sample.daemon_instance,
    )


def _classify(exc: Exception) -> ErrorClass:
    name = type(exc).__name__
    if "Timeout" in name or isinstance(exc, TimeoutError):
        return ErrorClass.TIMEOUT
    if isinstance(exc, ConnectionError):
        return ErrorClass.CONNECT_ERROR
    # ConnectError comes from the connect-python RPC layer.
    if name == "ConnectError":
        return ErrorClass.RPC_ERROR
    return ErrorClass.HTTP_ERROR
