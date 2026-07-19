# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live Iris job status and finelog logs for a run's child jobs, read over Direct VPC egress.

evaldash runs on Cloud Run with Direct VPC egress (PRIVATE_RANGES_ONLY) into the hai-gcp-models
default VPC, the same network as the Iris controller and finelog hub VMs. Both are reached by internal
IP resolved from a GCE instance filter (cached with a TTL) and queried with no token: the Iris
controller runs null-auth and finelog's cidr auth admits the RFC1918 ranges, so a caller on the VPC is
trusted.

Outside the VPC (local dev) discovery or the RPC fails fast under a short timeout, and every method
returns a degrade payload carrying ``reachable=False`` and an ``error`` string rather than raising, so
the dashboard shows "unreachable" instead of a 500. This is the one place exceptions are deliberately
swallowed into data — reachability is the signal the UI renders.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import httpx
from discovery import resolve_internal_ip

logger = logging.getLogger(__name__)

PROJECT = "hai-gcp-models"
ZONE = "us-central1-a"

# The marin production Iris controller and finelog hub, mirroring infra/grafana's config.
CONTROLLER_FILTER = "labels.iris-marin-controller=true AND status=RUNNING"
CONTROLLER_PORT = 10000
FINELOG_FILTER = "name = finelog-marin"
FINELOG_PORT = 10001

CONTROLLER_RPC_BASE = "iris.cluster.ControllerService"
FINELOG_RPC_BASE = "finelog.logging.LogService"

IP_CACHE_TTL = 300.0
# Short so a caller outside the VPC degrades quickly rather than blocking the request.
HTTP_TIMEOUT = 4.0


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:400]


def _short_state(raw: str | int | None) -> str:
    """``JOB_STATE_RUNNING`` / ``TASK_STATE_WORKER_FAILED`` -> ``running`` / ``worker_failed``."""
    if not raw:
        return "unspecified"
    if isinstance(raw, int):
        return str(raw)
    tail = raw.rsplit("_STATE_", 1)[-1] if "_STATE_" in raw else raw
    return tail.lower()


_LOG_LEVELS = {1: "debug", 2: "info", 3: "warning", 4: "error", 5: "critical"}


def _short_level(raw: str | int | None) -> str | None:
    if raw is None or raw == "" or raw == 0:
        return None
    if isinstance(raw, int):
        return _LOG_LEVELS.get(raw, str(raw))
    return raw.removeprefix("LOG_LEVEL_").lower()


def _ts_ms(obj: dict | None) -> int | None:
    """Milliseconds from an iris/finelog ``Timestamp`` message (``epochMs``/``epoch_ms``), or None."""
    if not obj:
        return None
    value = obj.get("epochMs", obj.get("epoch_ms"))
    return int(value) if value is not None else None


def _job(job: dict | None) -> dict | None:
    if not job:
        return None
    return {
        "state": _short_state(job.get("state")),
        "error": job.get("error") or None,
        "exit_code": int(job.get("exitCode") or 0),
        "started_at_ms": _ts_ms(job.get("startedAt")),
        "finished_at_ms": _ts_ms(job.get("finishedAt")),
        "name": job.get("name") or None,
        "status_message": job.get("statusMessage") or None,
    }


def _attempt(attempt: dict) -> dict:
    return {
        "attempt_id": int(attempt.get("attemptId") or 0),
        "state": _short_state(attempt.get("state")),
        "worker_id": attempt.get("workerId") or None,
        "exit_code": int(attempt.get("exitCode") or 0),
        "error": attempt.get("error") or None,
        "started_at_ms": _ts_ms(attempt.get("startedAt")),
        "finished_at_ms": _ts_ms(attempt.get("finishedAt")),
        "is_worker_failure": bool(attempt.get("isWorkerFailure") or False),
        "attempt_uid": attempt.get("attemptUid") or None,
    }


def _task(task: dict) -> dict:
    task_id = task.get("taskId") or ""
    return {
        "task_id": task_id,
        "task_index": task_id.rsplit("/", 1)[-1] if task_id else "",
        "state": _short_state(task.get("state")),
        "worker_id": task.get("workerId") or None,
        "exit_code": int(task.get("exitCode") or 0),
        "error": task.get("error") or None,
        "started_at_ms": _ts_ms(task.get("startedAt")),
        "finished_at_ms": _ts_ms(task.get("finishedAt")),
        "current_attempt_id": int(task.get("currentAttemptId") or 0),
        "attempts": [_attempt(attempt) for attempt in task.get("attempts") or []],
    }


def _log_entry(entry: dict) -> dict:
    return {
        "timestamp_ms": _ts_ms(entry.get("timestamp")),
        "source": entry.get("source") or None,
        "data": entry.get("data") or "",
        "attempt_id": entry.get("attemptId", entry.get("attempt_id")),
        "level": _short_level(entry.get("level")),
        "key": entry.get("key") or None,
    }


@dataclass
class _CachedIp:
    ip: str
    expires_at: float


class ClusterGateway:
    """Resolves the controller and finelog VMs by internal IP and queries them with plain JSON POSTs.

    IPs are cached for ``ip_ttl`` and re-resolved after a transport error, so a rebuilt VM is picked up
    without a restart. One ``httpx.Client`` is shared across calls; close it on shutdown.
    """

    def __init__(self, *, timeout: float = HTTP_TIMEOUT, ip_ttl: float = IP_CACHE_TTL) -> None:
        self._timeout = timeout
        self._ip_ttl = ip_ttl
        self._client = httpx.Client(timeout=timeout, headers={"content-type": "application/json"})
        self._lock = threading.Lock()
        self._ips: dict[str, _CachedIp] = {}

    def close(self) -> None:
        self._client.close()

    def _resolve(self, instance_filter: str, port: int) -> str:
        now = time.monotonic()
        with self._lock:
            cached = self._ips.get(instance_filter)
            if cached is not None and cached.expires_at > now:
                return f"http://{cached.ip}:{port}"
        ip = resolve_internal_ip(PROJECT, ZONE, instance_filter, timeout=self._timeout)
        with self._lock:
            self._ips[instance_filter] = _CachedIp(ip, now + self._ip_ttl)
        logger.info("resolved %s to %s", instance_filter, ip)
        return f"http://{ip}:{port}"

    def _invalidate(self, instance_filter: str) -> None:
        with self._lock:
            self._ips.pop(instance_filter, None)

    def _post(self, instance_filter: str, port: int, rpc_base: str, method: str, body: dict) -> dict:
        """POST a Connect RPC as JSON, re-resolving the VM IP once on a transport error."""
        for attempt in (1, 2):
            base = self._resolve(instance_filter, port)
            try:
                response = self._client.post(f"{base}/{rpc_base}/{method}", json=body)
            except httpx.TransportError:
                self._invalidate(instance_filter)
                if attempt == 2:
                    raise
                continue
            if response.status_code != 200:
                raise httpx.HTTPError(f"{method} returned {response.status_code}: {response.text[:200]}")
            return response.json()
        raise AssertionError("unreachable")

    def job_status(self, job_path: str) -> dict:
        """Job state plus per-task attempt detail for one iris job, or an unreachable degrade payload."""
        try:
            status = self._post(
                CONTROLLER_FILTER, CONTROLLER_PORT, CONTROLLER_RPC_BASE, "GetJobStatus", {"job_id": job_path}
            )
            tasks = self._post(
                CONTROLLER_FILTER, CONTROLLER_PORT, CONTROLLER_RPC_BASE, "ListTasks", {"job_id": job_path}
            )
        except Exception as exc:
            logger.info("iris controller unreachable for %s: %s", job_path, exc)
            return {
                "reachable": False,
                "error": f"iris controller unreachable — {_describe(exc)}",
                "job": None,
                "tasks": [],
            }
        return {
            "reachable": True,
            "error": None,
            "job": _job(status.get("job")),
            "tasks": [_task(task) for task in tasks.get("tasks") or []],
        }

    def fetch_logs(self, job_path: str, *, max_lines: int, substring: str | None) -> dict:
        """The last ``max_lines`` finelog lines across every task of ``job_path`` (prefix match), or an
        unreachable degrade payload."""
        source = f"{job_path.rstrip('/')}/"
        # tail is the proto's direction bool: return the LAST max_lines entries.
        body: dict = {"source": source, "match_scope": "MATCH_SCOPE_PREFIX", "max_lines": max_lines, "tail": True}
        if substring:
            body["substring"] = substring
        try:
            response = self._post(FINELOG_FILTER, FINELOG_PORT, FINELOG_RPC_BASE, "FetchLogs", body)
        except Exception as exc:
            logger.info("finelog unreachable for %s: %s", source, exc)
            return {
                "reachable": False,
                "error": f"finelog unreachable — {_describe(exc)}",
                "source": source,
                "entries": [],
            }
        return {
            "reachable": True,
            "error": None,
            "source": source,
            "entries": [_log_entry(entry) for entry in response.get("entries") or []],
        }
