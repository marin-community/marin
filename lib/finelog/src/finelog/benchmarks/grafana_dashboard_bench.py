# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Finelog queries from Grafana dashboards or fixed SQL exports.

Benchmark-owned servers run in shadow mode: they serve reads from ``--log-dir``
and refuse a `gs://`/`s3://` archive at startup, preventing compaction, eviction,
and deletion. ``--address`` measures an existing server without changing its
mode or background work. Pass dashboard variables as ``--variable NAME=VALUE``;
repeat the option name to model a Grafana multi-value selection. The JSON result
records latency, result rows and bytes, digest, and ``EXPLAIN ANALYZE`` scan
metrics for every rendered SQL target.

Bridge-generated dataset queries are not embedded in dashboard JSON. Export
their source queries as a JSON array of ``{"name": ..., "sql": ...}`` objects
and pass ``--query-file``. These queries execute verbatim; freeze their time
bounds in the SQL before comparing servers. ``--start-ms``, ``--end-ms``,
``--interval-ms``, and ``--variable`` apply only to dashboard rendering. Results
record each workload's rendered window, or null for literal query files.
EXPLAIN wall times include RPC overhead and queueing, so they are not pure
planner CPU time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

from finelog.benchmarks.grafana_dashboard_corpus import DashboardCorpus, DashboardQuery, load_dashboard_corpus
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


@dataclass(frozen=True)
class BenchmarkWorkload:
    path: Path
    corpus: DashboardCorpus
    window: dict[str, object] | None


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
        started = time.perf_counter()
        plan = query_table(client, "EXPLAIN " + query.sql)
        payload["explain_wall_ms"] = (time.perf_counter() - started) * 1_000
        payload["plan"] = plan.to_pylist()
        started = time.perf_counter()
        payload["explain"] = asdict(explain_metrics(client, query.sql))
        payload["explain_analyze_wall_ms"] = (time.perf_counter() - started) * 1_000
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", action="append", type=Path, default=[])
    parser.add_argument(
        "--query-file",
        action="append",
        type=Path,
        default=[],
        help='fixed bridge SQL: a JSON array of {"name": "...", "sql": "..."} objects',
    )
    parser.add_argument("--address", help="measure an existing server, for example an SSH tunnel to the hub")
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--start-ms", type=int, help="dashboard rendering start; required with --dashboard")
    parser.add_argument("--end-ms", type=int, help="dashboard rendering end; required with --dashboard")
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


def _query_file_corpus(path: Path) -> DashboardCorpus:
    contents = path.read_bytes()
    queries = tuple(DashboardQuery(item["name"], item["name"], "A", item["sql"]) for item in json.loads(contents))
    if not queries or len({query.name for query in queries}) != len(queries):
        raise ValueError(f"query file must contain uniquely named queries: {path}")
    return DashboardCorpus(path.stem, path.stem, "", hashlib.sha256(contents).hexdigest(), queries)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.dashboard and not args.query_file:
        parser.error("provide --dashboard or --query-file")
    if args.dashboard and (args.start_ms is None or args.end_ms is None):
        parser.error("--dashboard requires --start-ms and --end-ms")
    if args.address:
        if args.server_binary or args.log_dir:
            parser.error("--address cannot be combined with --server-binary or --log-dir")
        if args.query_timeout_ms:
            parser.error("--query-timeout-ms configures only a benchmark-owned server")
    elif not args.server_binary or not args.log_dir:
        parser.error("provide --address or both --server-binary and --log-dir")
    variables = _variables_by_name(args.variable)
    rendered_window = {
        "start_ms": args.start_ms,
        "end_ms": args.end_ms,
        "interval_ms": args.interval_ms,
        "variables": variables,
    }
    workloads = [
        BenchmarkWorkload(
            dashboard,
            load_dashboard_corpus(
                dashboard,
                start_ms=args.start_ms,
                end_ms=args.end_ms,
                interval_ms=args.interval_ms,
                variables=variables,
            ),
            rendered_window,
        )
        for dashboard in args.dashboard
    ]
    workloads.extend(BenchmarkWorkload(path, _query_file_corpus(path), None) for path in args.query_file)
    server = (
        nullcontext(args.address)
        if args.address
        else local_server(
            args.server_binary.resolve(),
            args.log_dir.resolve(),
            query_timeout_ms=args.query_timeout_ms,
            extra_args=("--mode", "shadow"),
        )
    )
    with server as address:
        client = stats_client(address)
        results = []
        for workload in workloads:
            queries = [
                _measure_query(
                    client,
                    query,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    include_explain=args.explain,
                )
                for query in workload.corpus.queries
            ]
            results.append(
                {
                    "path": str(workload.path),
                    "sha256": workload.corpus.sha256,
                    "uid": workload.corpus.uid,
                    "title": workload.corpus.title,
                    "refresh": workload.corpus.refresh,
                    "window": workload.window,
                    "queries": queries,
                }
            )
        payload = {
            "workloads": results,
            "server_binary": str(args.server_binary) if args.server_binary else None,
            "address": address,
            "query_timeout_ms": args.query_timeout_ms,
            "server": server_info(address),
        }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "workloads": len(results),
                "queries": sum(len(workload["queries"]) for workload in results),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
