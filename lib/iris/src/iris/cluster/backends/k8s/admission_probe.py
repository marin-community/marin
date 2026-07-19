# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``iris.admission_probe`` finelog namespace: periodic dry-run canary pod applies.

A ``dryRun=All`` pod create traverses the cluster's full admission chain —
mutating and validating webhooks, quota, policy — without persisting anything.
Probing it from the controller (which already holds pod-create RBAC) turns a
fail-closed admission webhook, webhook TLS breakage, or an admission timeout
into ``failed`` rows within a probe interval, even though no task pod ever comes
into existence on such a cluster. Alert on failed rows, or on silence: no rows
means the controller (or this emitter) is down.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from finelog.client.log_client import Table
from rigging.timing import Timestamp

from iris.cluster.platforms.k8s.service import K8sService
from iris.cluster.platforms.k8s.types import KubectlError
from iris.cluster.worker.stats import stats_timestamp

logger = logging.getLogger(__name__)

# finelog namespace for ``IrisAdmissionProbe`` rows.
ADMISSION_PROBE_NAMESPACE = "iris.admission_probe"

# Probe cadence. One dry-run apply per minute is invisible load on the API
# server and bounds detection latency for an admission outage to ~a minute.
DEFAULT_ADMISSION_PROBE_INTERVAL = 60.0

# Webhook denial bodies quote the offending manifest and can run long; keep
# enough to identify the webhook and its verdict without bloating every row.
PROBE_MESSAGE_MAX_LEN = 500

# The canary pod name is stable: a dry-run create never persists the object, so
# the name cannot collide — and a fixed name makes the probe easy to filter in
# API-server audit logs.
CANARY_POD_NAME = "iris-admission-probe"

# Never pulled — admission is evaluated on the manifest alone.
_CANARY_IMAGE = "registry.k8s.io/pause:3.9"


class ProbeOutcome(StrEnum):
    """How one dry-run canary apply ended."""

    OK = "ok"
    FAILED = "failed"


@dataclass
class IrisAdmissionProbe:
    """One dry-run canary apply outcome. Doubles as the finelog table schema.

    ``outcome`` and ``error_class`` are stored as strings (finelog columns are
    primitive); ``outcome`` always holds a :class:`ProbeOutcome` value and
    ``error_class`` a :func:`classify_probe_failure` bucket (``""`` on ok).
    """

    # Alert queries scan for failures in a window; clustering parquet by outcome
    # lets row-group min/max skip the all-ok bulk of the table.
    key_column: ClassVar[str] = "outcome"

    outcome: str
    ts: datetime
    namespace: str
    error_class: str
    latency_ms: int
    message: str  # truncated failure detail, "" on ok


def canary_pod_manifest(namespace: str) -> dict:
    """A minimal pod manifest that exercises admission without ever running.

    Deliberately not labeled as an Iris-managed task pod and not queued through
    Kueue: the probe measures whether the admission chain accepts a plain pod
    CREATE, not whether quota or scheduling would place it.
    """
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": CANARY_POD_NAME,
            "namespace": namespace,
            "labels": {"app": "iris-admission-probe"},
        },
        "spec": {
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "canary",
                    "image": _CANARY_IMAGE,
                    "resources": {
                        "requests": {"cpu": "1m", "memory": "1Mi"},
                        "limits": {"cpu": "1m", "memory": "1Mi"},
                    },
                }
            ],
        },
    }


def classify_probe_failure(exc: Exception) -> str:
    """Bucket a dry-run apply failure for alert labels.

    ``webhook`` is the fail-closed-admission-webhook incident class (the API
    server names the webhook in its error body). The rest split API verdicts
    (HTTP status, with ``forbidden`` called out for RBAC regressions) from
    transport failures (``timeout``/``unreachable``).
    """
    message = str(exc).lower()
    if "webhook" in message:
        return "webhook"
    if isinstance(exc, KubectlError) and exc.status is not None:
        if exc.status == 403:
            return "forbidden"
        return f"http_{exc.status}"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "connection" in message or "refused" in message or "unreachable" in message:
        return "unreachable"
    return "error"


class AdmissionProber:
    """Background thread that dry-run-applies a canary pod and emits ``iris.admission_probe`` rows.

    Runs off the reconcile path on its own thread, one probe per
    ``poll_interval``. A failed probe (or a failed row write) logs and the next
    interval retries; nothing propagates to the backend.
    """

    def __init__(
        self,
        kubectl: K8sService,
        table: Table,
        *,
        poll_interval: float = DEFAULT_ADMISSION_PROBE_INTERVAL,
    ) -> None:
        self._kubectl = kubectl
        self._table = table
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="admission-prober")
        self._thread.start()

    def _run(self) -> None:
        # Wait first: the first probe lands one interval after the backend comes
        # up, and a test-sized interval keeps the thread quiet while probe_once
        # is driven directly.
        while not self._stop.wait(timeout=self._poll_interval):
            try:
                self.probe_once()
            except Exception:
                logger.warning("admission probe cycle failed", exc_info=True)

    def probe_once(self) -> None:
        manifest = canary_pod_manifest(self._kubectl.namespace)
        start = Timestamp.now()
        error: Exception | None = None
        try:
            self._kubectl.dry_run_create(manifest)
        except Exception as e:
            error = e
        latency_ms = max(0, Timestamp.now().epoch_ms() - start.epoch_ms())
        if error is None:
            row = IrisAdmissionProbe(
                outcome=ProbeOutcome.OK,
                ts=stats_timestamp(),
                namespace=self._kubectl.namespace,
                error_class="",
                latency_ms=latency_ms,
                message="",
            )
        else:
            logger.warning("admission probe dry-run apply failed: %s", error)
            row = IrisAdmissionProbe(
                outcome=ProbeOutcome.FAILED,
                ts=stats_timestamp(),
                namespace=self._kubectl.namespace,
                error_class=classify_probe_failure(error),
                latency_ms=latency_ms,
                message=str(error)[:PROBE_MESSAGE_MAX_LEN],
            )
        self._table.write([row])

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
