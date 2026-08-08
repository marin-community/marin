# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the operator query corpus for the `log` namespace.

Runs the real ``finelog-server`` binary, so it measures the deployed planner,
segment indexes, and parquet layout.

- ``generate`` builds a deterministic corpus into a fresh work directory (see
  :mod:`finelog.benchmarks.log_workload_corpus`), compacts it, and waits for
  index backfill.
- ``measure`` reuses a log directory that already holds segments, which is how a
  change is checked against copied production shards.

Both modes use the ``log`` schema the server registers for itself, so the same
corpus under two binaries measures a schema change.

The log directory must be a disposable copy: starting Finelog activates
compaction, layout rewrites, and index backfill.

Run from ``lib/finelog``:

    uv run --group dev python -m finelog.benchmarks.log_query_bench generate \
        --server-binary rust/target/release/finelog-server \
        --rows 8000000 --work-dir /tmp/finelog-log-bench \
        --output /tmp/finelog-log-bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import httpx

from finelog.benchmarks.layout_candidates import table_digest
from finelog.benchmarks.log_workload_corpus import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_ITERATIONS,
    DEFAULT_SEGMENTS,
    DEFAULT_WARMUP_ITERATIONS,
    LOG_NAMESPACE,
    TARGET_JOB,
    LogDatasetSpec,
    Workload,
    build_workloads,
    dataset_facts,
    generate_batches,
    segment_row_ranges,
)
from finelog.benchmarks.query_measurement import (
    explain_metrics,
    local_server,
    maintain,
    nonnegative_int,
    positive_int,
    query_table,
    server_info,
    stats_client,
    write_batch,
)
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

# A tick backfills a bounded number of index bundles, so a corpus needs several
# to reach full coverage. This bounds a stuck backfill instead of looping.
MAX_MAINTENANCE_TICKS = 200


def _segments(address: str, namespace: str) -> list[dict[str, object]]:
    response = httpx.get(
        f"{address}/api/segments",
        params={"namespace": namespace, "physical": "true"},
        timeout=300,
    )
    response.raise_for_status()
    return cast(list[dict[str, object]], response.json()["segments"])


def _index_bundle(segment: dict[str, object]) -> dict[str, object] | None:
    physical = cast(dict[str, object] | None, segment.get("physical")) or {}
    return cast(dict[str, object] | None, physical.get("indexBundle"))


def _segment_facts(address: str, namespace: str) -> dict[str, object]:
    """Per-segment physical layout and index-bundle coverage after maintenance."""
    segments = _segments(address, namespace)
    rows = [
        {
            "path": segment["path"],
            "level": segment["level"],
            "rows": segment["rowCount"],
            "parquet_bytes": segment["byteSize"],
            "row_groups": (cast(dict, segment.get("physical")) or {}).get("rowGroups"),
            "bundle_bytes": (_index_bundle(segment) or {}).get("bytes"),
            "sections": {
                cast(str, section["id"]): section["payloadBytes"]
                for section in cast(list[dict], (_index_bundle(segment) or {}).get("sections") or [])
            },
        }
        for segment in segments
    ]
    return {
        "count": len(rows),
        "rows": sum(cast(int, row["rows"]) for row in rows),
        "parquet_bytes": sum(cast(int, row["parquet_bytes"]) for row in rows),
        "bundle_bytes": sum(cast(int, row["bundle_bytes"]) or 0 for row in rows),
        "indexed": sum(1 for row in rows if row["bundle_bytes"]),
        "segments": rows,
    }


def _required_sections(client: StatsServiceClientSync, namespace: str) -> set[str]:
    """The `.fidx` section ids the registered schema's index policy implies.

    A bundle built under a superseded policy still sits on disk, so "has a
    bundle" is not "is indexed for this schema".
    """
    schema = client.get_table_schema(stats_pb2.GetTableSchemaRequest(namespace=namespace)).schema
    return {f"trigram:{column.name}" for column in schema.columns if column.index.trigram}


def _indexed_segments(address: str, namespace: str, required: set[str]) -> tuple[int, int]:
    """`(segments carrying every required section, total segments)`."""
    segments = _segments(address, namespace)
    covered = sum(
        1
        for segment in segments
        if required
        <= {
            cast(str, section["id"])
            for section in cast(list[dict], (_index_bundle(segment) or {}).get("sections") or [])
        }
    )
    return covered, len(segments)


def _drive_maintenance(address: str, client: StatsServiceClientSync) -> dict[str, object]:
    """Backfill until every segment carries the schema's index sections.

    The server backfills a few bundles per tick, so the harness drives ticks
    explicitly rather than waiting on the background schedule. The reported
    duration is also what an index-policy change costs a live namespace.
    """
    required = _required_sections(client, LOG_NAMESPACE)
    started = time.perf_counter()
    covered, total = 0, 0
    for tick in range(MAX_MAINTENANCE_TICKS):
        maintain(address, LOG_NAMESPACE, force_compact_l0=False)
        covered, total = _indexed_segments(address, LOG_NAMESPACE, required)
        if total and covered == total:
            return {
                "ticks": tick + 1,
                "seconds": time.perf_counter() - started,
                "required_sections": sorted(required),
                "indexed_segments": covered,
            }
    raise TimeoutError(f"index backfill did not converge: {covered}/{total} segments carry {sorted(required)}")


def _latest_epoch_ms(client: StatsServiceClientSync) -> int:
    """The namespace's newest `epoch_ms`, which anchors the recent-window shape.

    `measure` has no corpus dimensions to derive a cutoff from, so both modes
    read it from the data.
    """
    latest = query_table(client, f'SELECT max(epoch_ms) AS latest FROM "{LOG_NAMESPACE}"')
    if not latest.num_rows or latest.column("latest")[0].as_py() is None:
        raise SystemExit(f"namespace {LOG_NAMESPACE!r} holds no rows to measure")
    return int(cast(int, latest.column("latest")[0].as_py()))


def _load_corpus(address: str, client: StatsServiceClientSync, spec: LogDatasetSpec) -> dict[str, object]:
    """Write the corpus one key band at a time, compacting each into a segment.

    Finelog merges on its own byte thresholds, so loading everything and
    compacting once yields a single file. Compacting per band gives the store the
    segment count the spec asks for, built by the real compactor.
    """
    started = time.perf_counter()
    written = 0
    for start, stop in segment_row_ranges(spec):
        written += sum(write_batch(client, LOG_NAMESPACE, batch) for batch in generate_batches(spec, start, stop))
        maintain(address, LOG_NAMESPACE, force_compact_l0=True)
    return {"rows_written": written, "seconds": time.perf_counter() - started}


def _measure_workload(
    client: StatsServiceClientSync,
    workload: Workload,
    *,
    warmup: int,
    iterations: int,
    include_explain: bool,
) -> dict[str, object]:
    for _ in range(warmup):
        query_table(client, workload.sql)
    timings = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = query_table(client, workload.sql)
        timings.append((time.perf_counter() - started) * 1_000)
    assert result is not None
    payload: dict[str, object] = {
        "name": workload.name.value,
        "sql": workload.sql,
        "scoped_by_key_substring": workload.scoped_by_key_substring,
        "samples_ms": timings,
        "p50_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "rows": result.num_rows,
        "result_digest": table_digest(result),
    }
    if include_explain:
        payload["explain"] = asdict(explain_metrics(client, workload.sql))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("generate", "generate a log corpus into a fresh work directory, then measure it"),
        ("measure", "measure an existing log directory (e.g. copied production shards)"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--server-binary", type=Path, required=True)
        sub.add_argument("--warmup", type=nonnegative_int, default=DEFAULT_WARMUP_ITERATIONS)
        sub.add_argument("--iterations", type=positive_int, default=DEFAULT_ITERATIONS)
        sub.add_argument("--query-timeout-ms", type=nonnegative_int, default=0)
        sub.add_argument("--explain", action=argparse.BooleanOptionalAction, default=True)
        sub.add_argument("--output", type=Path, required=True)

    generate = subparsers.choices["generate"]
    generate.add_argument("--rows", type=positive_int, required=True)
    generate.add_argument("--segments", type=positive_int, default=DEFAULT_SEGMENTS)
    generate.add_argument("--batch-rows", type=positive_int, default=DEFAULT_BATCH_ROWS)
    generate.add_argument("--work-dir", type=Path, required=True)

    measure = subparsers.choices["measure"]
    measure.add_argument("--log-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generating = args.command == "generate"
    if generating:
        spec = LogDatasetSpec(rows=args.rows, segments=args.segments, batch_rows=args.batch_rows)
        log_dir = cast(Path, args.work_dir)
        if any(log_dir.iterdir()) if log_dir.is_dir() else log_dir.exists():
            raise SystemExit(f"--work-dir must be empty or absent: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        spec = LogDatasetSpec(rows=1)
        log_dir = cast(Path, args.log_dir)

    with local_server(
        args.server_binary,
        log_dir,
        query_timeout_ms=args.query_timeout_ms,
        extra_args=("--debug-admin",),
    ) as address:
        client = stats_client(address)
        preparation: dict[str, object] = {}
        if generating:
            preparation["load"] = _load_corpus(address, client, spec)
        preparation["maintenance"] = _drive_maintenance(address, client)
        latest_ms = _latest_epoch_ms(client)
        workloads = build_workloads(latest_ms)

        payload = {
            "command": args.command,
            "server_binary": str(args.server_binary),
            "log_dir": str(log_dir),
            "query_timeout_ms": args.query_timeout_ms,
            "dataset": dataset_facts(spec) if generating else {"target_job": TARGET_JOB},
            "latest_epoch_ms": latest_ms,
            "preparation": preparation,
            "storage": _segment_facts(address, LOG_NAMESPACE),
            "server": server_info(address),
            "queries": [
                _measure_workload(
                    client,
                    workload,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    include_explain=args.explain,
                )
                for workload in workloads
            ],
        }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "queries": len(workloads),
                "p50_ms": {
                    cast(str, query["name"]): round(cast(float, query["p50_ms"]), 1)
                    for query in cast(list[dict[str, object]], payload["queries"])
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
