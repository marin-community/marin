# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The live Iris controller data the dashboard reads, as flat JSON rows.

The bridge owns each query and its shape behind a fixed endpoint — the dashboard
never sends raw SQL. ExecuteRawQuery is admin-only and works only because the marin
controller runs null-auth (callers resolve to admin); ListWorkers is the only way to
count live workers since worker liveness is in-memory, not in the controller's SQL.

The controller is reached by internal IP over Direct VPC egress, resolved from a GCE
label and refreshed after a connection failure so a rebuilt controller is picked up
without a restart. Calls carry no token (null-auth admits the VPC as admin).
"""

import json
import logging
import time

import httpx
from config import CONTROLLER_PORT, ClusterTarget
from discovery import resolve_internal_ip
from errors import UpstreamError

logger = logging.getLogger(__name__)

_RPC_BASE = "iris.cluster.ControllerService"
_LIST_WORKERS_PAGE = 1000
_MAX_WORKER_PAGES = 50

# Job states, tracking lib/iris/src/iris/rpc/job.proto. Unknown ints render as state_N.
JOB_STATE_NAMES: dict[int, str] = {
    0: "unspecified",
    1: "pending",
    2: "building",
    3: "running",
    4: "succeeded",
    5: "failed",
    6: "killed",
    7: "worker_failed",
    8: "unschedulable",
}
_IN_FLIGHT_STATES = (1, 2, 3)

# Root jobs only. In-flight states are current; terminal states are counted over the
# trailing 24h by finish time. strftime keeps the window controller-side.
_JOBS_SQL = """
SELECT state, COUNT(*) AS n
FROM jobs
WHERE parent_job_id IS NULL
  AND (
    state IN (1, 2, 3)
    OR (state IN (4, 5, 6, 7, 8)
        AND finished_at_ms > (strftime('%s', 'now') * 1000 - 86400000))
  )
GROUP BY state
"""


def _state_name(state: int) -> str:
    return JOB_STATE_NAMES.get(state, f"state_{state}")


class IrisSource:
    """A query handle for one cluster's Iris controller, addressed by internal IP."""

    def __init__(self, target: ClusterTarget, *, timeout: float) -> None:
        self._target = target
        self._client = httpx.Client(timeout=timeout, headers={"content-type": "application/json"})
        self._base_url: str | None = None

    @property
    def target(self) -> ClusterTarget:
        return self._target

    def _resolve(self) -> str:
        ip = resolve_internal_ip(self._target.project, self._target.zone, self._target.controller_filter)
        base = f"http://{ip}:{CONTROLLER_PORT}"
        logger.info("resolved iris controller for %s to %s", self._target.name, base)
        self._base_url = base
        return base

    def _base(self) -> str:
        return self._base_url or self._resolve()

    def _post_rpc(self, method: str, body: dict) -> dict:
        """POST a Connect RPC, re-resolving the controller IP once on a transport error."""
        for attempt in (1, 2):
            base = self._base()
            try:
                response = self._client.post(f"{base}/{_RPC_BASE}/{method}", json=body)
            except httpx.TransportError as err:
                self._base_url = None  # force re-resolve on the retry
                if attempt == 2:
                    raise UpstreamError("iris", f"{method}: controller unreachable ({err})", status_code=504) from err
                continue
            if response.status_code != 200:
                raise UpstreamError("iris", f"{method}: controller returned {response.status_code}", status_code=502)
            return response.json()
        raise AssertionError("unreachable")

    def jobs(self) -> list[dict]:
        """Root-job counts by state: in-flight now plus terminal states over the last 24h."""
        result = self._post_rpc("ExecuteRawQuery", {"sql": _JOBS_SQL})
        rows = []
        for raw in result.get("rows", []):
            state, count = json.loads(raw)
            bucket = "inflight" if state in _IN_FLIGHT_STATES else "last24h"
            rows.append({"bucket": bucket, "state": _state_name(state), "count": count})
        return rows

    def workers(self) -> list[dict]:
        """Healthy worker counts and resource totals per region (empty region -> 'unknown')."""
        regions: dict[str, dict[str, float]] = {}
        offset = 0
        for _ in range(_MAX_WORKER_PAGES):
            page = self._post_rpc("ListWorkers", {"query": {"offset": offset, "limit": _LIST_WORKERS_PAGE}})
            workers = page.get("workers", [])
            for worker in workers:
                if worker.get("healthy") is not True:
                    continue
                metadata = worker.get("metadata") or {}
                attributes = metadata.get("attributes") or {}
                region = (attributes.get("region") or {}).get("stringValue") or "unknown"
                device = metadata.get("device") or {}
                bucket = regions.setdefault(
                    region, {"healthy": 0, "cpu_millicores": 0, "memory_bytes": 0, "tpu_chips": 0}
                )
                bucket["healthy"] += 1
                bucket["cpu_millicores"] += int(metadata.get("cpuCount") or 0) * 1000
                bucket["memory_bytes"] += int(metadata.get("memoryBytes") or 0)
                bucket["tpu_chips"] += int((device.get("tpu") or {}).get("count") or 0)
            if not page.get("hasMore") or not workers:
                break
            offset += len(workers)
        return [{"region": region, **totals} for region, totals in sorted(regions.items())]

    def health(self) -> list[dict]:
        """One row: whether the controller /health answers, and the round-trip latency.

        Returns 200 with reachable=false on failure — reachability is the signal, so the
        panel threshold renders it, rather than the endpoint erroring.
        """
        base = self._base()
        started = time.monotonic()
        try:
            response = self._client.get(f"{base}/health")
            latency_ms = round((time.monotonic() - started) * 1000)
            reachable = response.status_code == 200
            return [{"reachable": reachable, "latency_ms": latency_ms if reachable else None}]
        except httpx.TransportError as err:
            self._base_url = None
            return [{"reachable": False, "latency_ms": None, "error": str(err)}]

    def raw_query(self, sql: str) -> list[dict]:
        """Run an ad-hoc SELECT via ExecuteRawQuery, zipping columns to values."""
        result = self._post_rpc("ExecuteRawQuery", {"sql": sql})
        names = [column["name"] for column in result.get("columns", [])]
        return [dict(zip(names, json.loads(raw), strict=False)) for raw in result.get("rows", [])]
