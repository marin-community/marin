# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark every Finelog query in a checked-in Grafana dashboard.

The server runs in shadow mode: it serves reads from ``--log-dir`` and refuses a
`gs://`/`s3://` archive at startup, so it cannot compact, evict, or drop what it
was pointed at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
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
        "result_digest": table_digest(result),
    }
    if include_explain:
        payload["explain"] = asdict(explain_metrics(client, query.sql))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--interval-ms", type=positive_int, default=60_000)
    parser.add_argument("--cluster", action="append", required=True)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--query-timeout-ms", type=nonnegative_int, default=0)
    parser.add_argument("--explain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = load_dashboard_corpus(
        args.dashboard,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        interval_ms=args.interval_ms,
        clusters=tuple(args.cluster),
    )
    dashboard_sha256 = hashlib.sha256(args.dashboard.read_bytes()).hexdigest()
    with local_server(
        args.server_binary,
        args.log_dir,
        query_timeout_ms=args.query_timeout_ms,
        extra_args=("--mode", "shadow"),
    ) as address:
        client = stats_client(address)
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
        payload = {
            "dashboard": {
                "path": str(args.dashboard),
                "sha256": dashboard_sha256,
                "uid": corpus.uid,
                "title": corpus.title,
                "refresh": corpus.refresh,
            },
            "window": {
                "start_ms": args.start_ms,
                "end_ms": args.end_ms,
                "interval_ms": args.interval_ms,
                "clusters": args.cluster,
            },
            "server_binary": str(args.server_binary),
            "query_timeout_ms": args.query_timeout_ms,
            "server": server_info(address),
            "queries": queries,
        }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "queries": len(queries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
