# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Query worker health from the finelog stats service via the controller proxy.

Workers emit heartbeat rows into the ``iris.worker`` stats namespace. The
controller's endpoint proxy at ``/proxy/system.log-server`` forwards requests
to the co-hosted finelog log-server, so callers only need the controller URL.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.ipc as paipc
from connectrpc.errors import ConnectError
from finelog.errors import StatsError
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

logger = logging.getLogger(__name__)

# How far back the latest-row-per-worker query looks. Workers heartbeat at
# ~5s; 5 minutes gives plenty of room for one missed heartbeat without
# losing the worker from the pane.
_LOOKBACK_MINUTES = 5

# DuckDB's now() returns a TIMESTAMPTZ; the stored ts column is tz-naive
# TIMESTAMP populated from a UTC-normalized datetime. Comparing the two
# directly silently uses the session timezone, which causes a non-UTC host
# to filter with the wrong window. Pin the comparison to UTC explicitly.
LATEST_WORKER_ROW_SQL = """
SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY worker_id ORDER BY ts DESC) AS _rn
    FROM "iris.worker"
    WHERE ts > (now() AT TIME ZONE 'UTC')::TIMESTAMP - INTERVAL '{minutes} minutes'
) ranked
WHERE _rn = 1
ORDER BY worker_id
""".strip()

# Endpoint name for the log-server as registered in the controller endpoint store.
_LOG_SERVER_ENDPOINT = "system.log-server"


@dataclass
class WorkerHealth:
    worker_id: str
    healthy: bool
    status_message: str
    address: str = ""


def _age_message(ts: object) -> str:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = int((datetime.now(tz=timezone.utc) - ts).total_seconds())
        return f"Unhealthy (last seen {age_s}s ago)"
    return "Unhealthy"


def rows_to_worker_health(rows: list[dict]) -> list[WorkerHealth]:
    """Map latest-row-per-worker query result rows to WorkerHealth."""
    out: list[WorkerHealth] = []
    for row in rows:
        healthy = bool(row.get("healthy"))
        ts = row.get("ts")
        status_message = "" if healthy else _age_message(ts)
        out.append(
            WorkerHealth(
                worker_id=str(row.get("worker_id") or ""),
                healthy=healthy,
                status_message=status_message,
                address=str(row.get("address") or ""),
            )
        )
    return out


def query_workers(controller_url: str) -> list[WorkerHealth]:
    """Query worker health from the stats service via the controller proxy.

    Soft-fails to an empty list on transport error; schema-mapping bugs
    (KeyError / TypeError) propagate so callers can see and fix them.
    """
    proxy_url = f"{controller_url.rstrip('/')}/proxy/{_LOG_SERVER_ENDPOINT}"
    try:
        sql = LATEST_WORKER_ROW_SQL.format(minutes=_LOOKBACK_MINUTES)
        client = StatsServiceClientSync(address=proxy_url)
        response = client.query(stats_pb2.QueryRequest(sql=sql))
        table: pa.Table = paipc.open_stream(pa.BufferReader(bytes(response.arrow_ipc))).read_all()
        return rows_to_worker_health(table.to_pylist())
    except (StatsError, ConnectError, ConnectionError, OSError, TimeoutError) as exc:
        logger.warning("worker stats query failed: %s: %s", type(exc).__name__, exc)
        return []
