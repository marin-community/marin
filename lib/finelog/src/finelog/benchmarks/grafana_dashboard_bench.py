# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Finelog queries from one or more checked-in Grafana dashboards.

The server runs in shadow mode: it serves reads from ``--log-dir`` and refuses a
`gs://`/`s3://` archive at startup, so it cannot compact, evict, or drop what it
was pointed at. Pass each dashboard variable as ``--variable NAME=VALUE``;
repeat the option name to model a Grafana multi-value selection. The JSON result
records latency, result rows and bytes, digest, and ``EXPLAIN ANALYZE`` scan
metrics for every rendered SQL target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from finelog.benchmarks.grafana_dashboard_corpus import DashboardQuery, load_dashboard_corpus
from finelog.benchmarks.layout_candidates import table_digest
from finelog.benchmarks.query_measurement import (
    explain_metrics,
    local_server,
    nonnegative_int,
    positive_int,
    query_table,
    server_info,
    stats_client,
)
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync


def _measure_query(
    client: StatsServiceClientSync,
    query: DashboardQuery,
    *,
    warmup: int,
    iterations: int,
    include_explain: bool,
) -> dict[str, object]:
    for _ in range(warmup):
        query_table(client, query.sql)
    timings = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = query_table(client, query.sql)
        timings.append((time.perf_counter() - started) * 1_000)
    assert result is not None
    payload: dict[str, object] = {
        "name": query.name,
        "panel_title": query.panel_title,
        "target": query.target,
        "sql": query.sql,
        "samples_ms": timings,
        "p50_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "rows": result.num_rows,
        "result_bytes": result.nbytes,
        "result_digest": table_digest(result),
    }
    if include_explain:
        payload["explain"] = asdict(explain_metrics(client, query.sql))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", action="append", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--interval-ms", type=positive_int, default=60_000)
    parser.add_argument(
        "--variable",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="fixed Grafana variable value; repeat a name for a multi-value variable",
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--query-timeout-ms", type=nonnegative_int, default=0)
    parser.add_argument("--explain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _variables_by_name(raw_variables: Sequence[str]) -> dict[str, tuple[str, ...]]:
    values: defaultdict[str, list[str]] = defaultdict(list)
    for raw in raw_variables:
        name, separator, value = raw.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"invalid dashboard variable {raw!r}; expected NAME=VALUE")
        values[name].append(value)
    return {name: tuple(items) for name, items in values.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    variables = _variables_by_name(args.variable)
    corpora = [
        (
            dashboard,
            load_dashboard_corpus(
                dashboard,
                start_ms=args.start_ms,
                end_ms=args.end_ms,
                interval_ms=args.interval_ms,
                variables=variables,
            ),
        )
        for dashboard in args.dashboard
    ]
    with local_server(
        args.server_binary,
        args.log_dir,
        query_timeout_ms=args.query_timeout_ms,
        extra_args=("--mode", "shadow"),
    ) as address:
        client = stats_client(address)
        dashboards = []
        for dashboard, corpus in corpora:
            queries = [
                _measure_query(
                    client,
                    query,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    include_explain=args.explain,
                )
                for query in corpus.queries
            ]
            dashboards.append(
                {
                    "path": str(dashboard),
                    "sha256": hashlib.sha256(dashboard.read_bytes()).hexdigest(),
                    "uid": corpus.uid,
                    "title": corpus.title,
                    "refresh": corpus.refresh,
                    "queries": queries,
                }
            )
        payload = {
            "dashboards": dashboards,
            "window": {
                "start_ms": args.start_ms,
                "end_ms": args.end_ms,
                "interval_ms": args.interval_ms,
                "variables": variables,
            },
            "server_binary": str(args.server_binary),
            "query_timeout_ms": args.query_timeout_ms,
            "server": server_info(address),
        }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dashboards": len(dashboards),
                "queries": sum(len(dashboard["queries"]) for dashboard in dashboards),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
