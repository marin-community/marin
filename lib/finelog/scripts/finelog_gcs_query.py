#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb>=1.0",
#     "gcsfs>=2024.0",
# ]
# ///
"""Query archived finelog parquet directly from GCS via DuckDB.

Use when an iris task's log entries have been evicted from the live
finelog server (``iris job logs`` returns empty) but still exist in the
durable GCS archive at ``<remote_log_dir>/log/seg_L*.parquet``. The live
FetchLogs path only scans local segments (``_local_segments`` deque); evicted
segments are flipped to ``REMOTE`` and dropped from the deque. The bytes
remain in GCS — this script reads them directly.

Marin cluster default archive: ``gs://marin-us-central2/finelog/marin/log``
(see ``lib/finelog/config/marin.yaml``).

Schema (from ``lib/finelog/src/finelog/store/log_namespace.py:69``):
    (seq, key, source, data, epoch_ms, level)

Authentication: uses ``gcsfs`` with Application Default Credentials
(``gcloud auth application-default login``).

Examples:
    # All attempts of a task, last 14 days
    scripts/finelog_gcs_query.py \
        '/ryan/my-job/1:' --since-seconds 1209600

    # Specific attempt, last 24h, only error-level rows
    scripts/finelog_gcs_query.py \
        '/ryan/my-job/1:3' --since-seconds 86400 --min-level WARNING
"""

import argparse
import sys
import time

import duckdb
import gcsfs

# finelog.types.LogLevel — keep in sync with proto enum.
_LEVELS: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "key_prefix",
        help="Log-store key prefix, e.g. '/ryan/job-id/1:' for all attempts of task 1.",
    )
    parser.add_argument(
        "--archive",
        default="gs://marin-us-central2/finelog/marin/log",
        help="GCS path to the finelog log-namespace directory.",
    )
    parser.add_argument("--since-ms", type=int, default=None, help="Min epoch_ms (inclusive).")
    parser.add_argument(
        "--since-seconds",
        type=int,
        default=None,
        help="Min age in seconds relative to now (alternative to --since-ms).",
    )
    parser.add_argument("--until-ms", type=int, default=None, help="Max epoch_ms (inclusive).")
    parser.add_argument("--max-lines", type=int, default=0, help="0 = unlimited.")
    parser.add_argument("--substring", default="", help="Optional substring filter on the data column.")
    parser.add_argument(
        "--min-level",
        choices=sorted(_LEVELS.keys()),
        default=None,
        help="Drop entries below this finelog level.",
    )
    parser.add_argument("--head", action="store_true", help="Print earliest lines (default); pairs with --max-lines.")
    parser.add_argument("--tail", action="store_true", help="Print latest lines; pairs with --max-lines.")
    args = parser.parse_args()

    if args.since_ms is not None and args.since_seconds is not None:
        parser.error("specify at most one of --since-ms / --since-seconds")
    if args.head and args.tail:
        parser.error("--head and --tail are mutually exclusive")

    now_ms = int(time.time() * 1000)
    since_ms = (
        args.since_ms
        if args.since_ms is not None
        else (now_ms - args.since_seconds * 1000 if args.since_seconds is not None else 0)
    )
    until_ms = args.until_ms if args.until_ms is not None else now_ms

    con = duckdb.connect()
    con.register_filesystem(gcsfs.GCSFileSystem())

    glob = f"{args.archive.rstrip('/')}/seg_L*.parquet"

    where = ["key LIKE ? || '%'", "epoch_ms BETWEEN ? AND ?"]
    params: list[object] = [args.key_prefix, since_ms, until_ms]
    if args.substring:
        where.append("data LIKE '%' || ? || '%'")
        params.append(args.substring)
    if args.min_level is not None:
        where.append("level >= ?")
        params.append(_LEVELS[args.min_level])

    order = "ORDER BY epoch_ms DESC" if args.tail else "ORDER BY epoch_ms"
    limit = f"LIMIT {args.max_lines}" if args.max_lines > 0 else ""

    sql = f"""
        SELECT epoch_ms, key, source, level, data
        FROM read_parquet('{glob}')
        WHERE {' AND '.join(where)}
        {order}
        {limit}
    """

    sys.stderr.write(
        f"finelog_gcs_query: archive={args.archive}\n"
        f"  key LIKE {args.key_prefix!r}%\n"
        f"  epoch_ms in [{since_ms}, {until_ms}]\n"
    )
    started = time.monotonic()
    rows = con.execute(sql, params).fetchall()
    elapsed = time.monotonic() - started
    sys.stderr.write(f"  fetched {len(rows)} rows in {elapsed:.1f}s\n")

    # When tailing, sort back to chronological for display.
    if args.tail:
        rows = sorted(rows, key=lambda r: r[0])
    for epoch_ms, key, source, level, data in rows:
        print(f"[{epoch_ms}] {key} {source} L{level} | {data.rstrip()}")


if __name__ == "__main__":
    main()
