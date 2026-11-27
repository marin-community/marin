#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that one cluster's controller rows reached Finelog, for each recent UTC day.

Compares the controller DB (read with the admin, SELECT-only ``iris query`` RPC)
with ``iris.controller_db__jobs``, ``__job_config``, ``__tasks`` and
``__task_attempts`` in Finelog, as described in
``lib/iris/docs/observability-rows.md#finding-lost-rows``. Exits 1 if any
controller key, or any finish, is missing from Finelog.

Usage:
    uv run python scripts/ops/observability_rows_check.py \\
        --cluster marin --finelog marin --start-date 2026-11-12

``--finelog`` names the Finelog that holds the cluster's rows: the hub
(``marin``) for ``marin`` and the CoreWeave clusters, whose Finelogs forward
there; a cluster whose Finelog does not forward, such as ``marin-dev``, names
its own. ``marin-dev``'s Finelog has no ``client_url``, so it is reached over a
``gcloud compute ssh`` tunnel and needs an operator with SSH access to the VM.
``--start-date`` is the first full UTC day after the cluster's
controller started writing these rows.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, date, datetime

from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client
from iris.cli.connect import connect_controller, rpc_client
from iris.cluster.stats.observability_rows_check import (
    MAX_CHECK_DAYS,
    SETTLE_MARGIN,
    DayWindow,
    day_coverage,
    days_to_check,
    format_report,
)
from iris.rpc import query_pb2

# cw-rno2a, the busiest cluster in 2026-11, had about 25k attempts a day.
FINELOG_MAX_ROWS = 1_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", required=True, help="Iris cluster config name, e.g. marin or cw-rno2a")
    parser.add_argument("--finelog", required=True, help="Finelog config name that holds the cluster's rows")
    parser.add_argument("--start-date", required=True, type=date.fromisoformat, help="First UTC day to check")
    parser.add_argument("--days", type=int, default=5, help=f"Full UTC days to check, 1 to {MAX_CHECK_DAYS} (default 5)")
    parser.add_argument("--max-keys", type=int, default=20, help="Keys to list per failing line (default 20)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    now = datetime.now(UTC)
    days = days_to_check(now, start=args.start_date, days=args.days)
    if not days:
        print(f"cluster {args.cluster}: no full UTC day since {args.start_date} yet")
        return 0

    coverage = []
    with (
        connect_controller(cluster_name=args.cluster) as endpoint,
        rpc_client(endpoint.url, endpoint.credentials) as controller,
        open_client(load_finelog_config(args.finelog), args.finelog) as finelog,
    ):

        def controller_query(sql: str) -> list[list[object]]:
            response = controller.execute_raw_query(query_pb2.RawQueryRequest(sql=sql))
            return [json.loads(row) for row in response.rows]

        def finelog_query(sql: str) -> list[tuple[object, ...]]:
            table = finelog.query(sql, max_rows=FINELOG_MAX_ROWS)
            return list(zip(*(column.to_pylist() for column in table.columns), strict=True))

        for day in days:
            window = DayWindow(day=day, settled_before=now - SETTLE_MARGIN, until=now)
            coverage.extend(day_coverage(controller_query, finelog_query, args.cluster, window))

    print(format_report(args.cluster, coverage, max_keys=args.max_keys))
    return 0 if all(c.ok for c in coverage) else 1


if __name__ == "__main__":
    sys.exit(main())
