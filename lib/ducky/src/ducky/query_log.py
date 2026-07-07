# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist every submitted query to a finelog namespace (``ducky.query``).

ducky records one row per submitted query — the SQL text plus its terminal
outcome and cost (rows, bytes, wall-clock, cache hit) — to finelog under its own
namespace. The result is a durable, queryable log of what ran through ducky, so
the query mix can be reviewed later for hot paths and optimization opportunities.
The namespace is itself queryable from the dashboard via the pre-baked
``finelog."ducky.query"`` view (see ``catalog.py``).

Writes are fire-and-forget: finelog's ``Table`` buffers rows on a background
flush thread and swallows transport failures, so a finelog outage never blocks
or fails a user's query. Recording is best-effort by design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from finelog.client import LogClient, StoragePolicy, Table
from iris.client.client import IrisContext
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from rigging.timing import Timestamp

logger = logging.getLogger(__name__)

QUERY_LOG_NAMESPACE = "ducky.query"

# ducky's query log is an audit / optimization record, so it is kept far longer
# than the transient iris.* stats namespaces: ~180 days, capped at 2 GiB (rows are
# small — SQL text plus a handful of scalars, so this holds many millions of them).
# Both caps are explicit so retention is deterministic regardless of the cluster-wide
# compaction defaults.
QUERY_LOG_STORAGE_POLICY = StoragePolicy(
    max_bytes=2 * 1024 * 1024 * 1024,
    max_age_seconds=180 * 86400,
)


@dataclass
class QueryLogRow:
    """One row per submitted query: the SQL and its terminal outcome/cost.

    ``ts`` is when the query reached a terminal state (done/error); it keys the
    finelog segments so time-range analytics prune to a few row groups. Cost
    columns are ``None`` for a failed query; ``error`` is ``None`` for a
    successful one. ``cached`` marks a submission served from ducky's in-memory
    result cache (no DuckDB execution).
    """

    key_column: ClassVar[str] = "ts"

    ts: datetime  # tz-naive UTC (see now_utc)
    query_id: str
    sql: str
    status: str
    cached: bool
    elapsed_ms: int | None = None
    total_rows: int | None = None
    result_bytes: int | None = None
    result_path: str | None = None
    error: str | None = None


class QueryLog:
    """Records submitted queries to the ``ducky.query`` finelog namespace.

    Wraps a finelog ``Table`` whose background flush thread does the network I/O;
    :meth:`record` only enqueues a row and never raises, so it is safe to call on
    the query manager's hot path.
    """

    def __init__(self, log_client: LogClient, table: Table) -> None:
        self._log_client = log_client
        self._table = table

    @classmethod
    def connect(cls, ctx: IrisContext) -> QueryLog:
        """Open a finelog client against the cluster log server and bind the table.

        Resolves the finelog endpoint through ``ctx.client`` (an in-cluster iris
        client that returns the log server's direct address, bypassing the
        controller proxy). The namespace is registered lazily by the Table's
        flush thread on its first write, so this never blocks on the server.
        """
        if ctx.client is None:
            raise RuntimeError("QueryLog requires an in-cluster iris client to resolve the finelog endpoint")
        log_client = LogClient.connect(LOG_SERVER_ENDPOINT_NAME, resolver=ctx.client.resolve_endpoint)
        table = log_client.get_table(QUERY_LOG_NAMESPACE, QueryLogRow, storage_policy=QUERY_LOG_STORAGE_POLICY)
        return cls(log_client, table)

    def record(self, row: QueryLogRow) -> None:
        """Enqueue one query record.

        Best-effort telemetry on the query hot path: a failure here (e.g. the
        table closed mid-shutdown) must never surface to the caller running the
        query, so it is logged and swallowed rather than raised.
        """
        try:
            self._table.write([row])
        except Exception:
            logger.warning("failed to enqueue query-log row for %s", row.query_id, exc_info=True)

    def close(self) -> None:
        """Drain the table's flush thread and close the finelog client."""
        self._log_client.close()


def now_utc() -> datetime:
    """Current tz-naive UTC datetime for the ``ts`` segment key (matches iris stats)."""
    return Timestamp.now().as_naive_utc()
