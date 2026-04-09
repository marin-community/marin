#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze cross-region GCS operations from archived Iris task logs.

This script:
- locates archived log parquet segments under ``<remote_state_dir>/logs``
- downloads the recent segments needed for a time window
- downloads the latest controller checkpoint unless a checkpoint directory is supplied
- joins task log entries against task attempt / worker region metadata
- reports cross-region ``gs://`` path mentions

The default mode analyzes the last 24 hours.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb
import fsspec

from iris.cluster.config import IrisConfig
from iris.cluster.controller.checkpoint import _find_latest_checkpoint_dir, download_checkpoint_to_local
from rigging.filesystem import get_bucket_location, region_from_prefix

GS_RE = re.compile(r"gs://[^\s'\"`]+")


@dataclass(frozen=True)
class TimeWindow:
    start: dt.datetime
    end: dt.datetime

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        return int(self.end.timestamp() * 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="lib/iris/examples/marin.yaml",
        help="Iris cluster config to use. Defaults to lib/iris/examples/marin.yaml.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory for downloaded parquet segments, checkpoint DB, and reports. Defaults to a temp dir.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Rolling analysis window in hours. Defaults to 24.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Window end in ISO8601 UTC, e.g. 2026-04-09T21:38:40Z. Defaults to now UTC.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional remote checkpoint directory to use instead of the latest checkpoint.",
    )
    parser.add_argument(
        "--download-lookback-hours",
        type=float,
        default=12.0,
        help="Extra lookback on log object mtimes when choosing parquet segments. Defaults to 12.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Path to write the JSON summary. Defaults to <outdir>/cross_region_ops_summary.json.",
    )
    parser.add_argument(
        "--appendix-csv",
        default=None,
        help="Path to write the CSV appendix. Defaults to <outdir>/cross_region_ops_appendix.csv.",
    )
    return parser.parse_args()


def parse_window(args: argparse.Namespace) -> TimeWindow:
    if args.end_time is None:
        end = dt.datetime.now(dt.timezone.utc)
    else:
        text = args.end_time.replace("Z", "+00:00")
        end = dt.datetime.fromisoformat(text)
        if end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)
        else:
            end = end.astimezone(dt.timezone.utc)
    start = end - dt.timedelta(hours=args.hours)
    return TimeWindow(start=start, end=end)


def default_outdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="iris_cross_region_ops_"))


def normalize_region(region: str | None) -> str | None:
    if region is None:
        return None
    region = region.lower()
    if region == "eu-west4":
        return "europe-west4"
    return region


def bucket_region(bucket_name: str, cache: dict[str, str | None]) -> str | None:
    bucket_name = bucket_name.rstrip(":")
    if bucket_name in cache:
        return cache[bucket_name]

    direct = normalize_region(region_from_prefix(f"gs://{bucket_name}"))
    if direct is not None:
        cache[bucket_name] = direct
        return direct

    try:
        cache[bucket_name] = normalize_region(get_bucket_location(bucket_name))
    except Exception:
        cache[bucket_name] = None
    return cache[bucket_name]


def classify_op(line: str) -> str:
    lower = line.lower()
    if any(
        token in lower
        for token in (
            "saving ",
            "saved ",
            "copying ",
            "finished copying",
            "writing ",
            "flushing ",
            "-> gs://",
        )
    ):
        return "write"
    if any(
        token in lower
        for token in (
            "loading ",
            "attempting to load",
            "reading ",
            "read ",
            "download",
            "opening ",
            "exists",
            "cache ledger",
            "initializing a v1 llm engine",
            "runai streamer",
        )
    ):
        return "read"
    return "other"


def choose_log_objects(
    remote_logs_dir: str,
    window: TimeWindow,
    lookback_hours: float,
) -> list[dict]:
    fs, path = fsspec.core.url_to_fs(remote_logs_dir)
    entries = sorted(fs.ls(path, detail=True), key=lambda e: e["mtime"])
    mtime_cutoff = window.start - dt.timedelta(hours=lookback_hours)

    before_window: dict | None = None
    chosen: list[dict] = []
    for entry in entries:
        entry_mtime = entry["mtime"].astimezone(dt.timezone.utc)
        if entry_mtime < mtime_cutoff:
            before_window = entry
            continue
        chosen.append(entry)

    if before_window is not None:
        chosen.insert(0, before_window)

    return chosen


def download_log_objects(remote_logs_dir: str, entries: list[dict], target_dir: Path) -> list[Path]:
    fs, _ = fsspec.core.url_to_fs(remote_logs_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    local_paths: list[Path] = []
    for entry in entries:
        name = entry["name"]
        local_path = target_dir / Path(name).name
        if not local_path.exists():
            with fs.open(name, "rb") as src, local_path.open("wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        local_paths.append(local_path)
    return local_paths


def download_checkpoint(config: IrisConfig, outdir: Path, checkpoint_dir: str | None) -> Path:
    checkpoint_outdir = outdir / "checkpoint"
    checkpoint_outdir.mkdir(parents=True, exist_ok=True)
    chosen = checkpoint_dir or _find_latest_checkpoint_dir(config.proto.storage.remote_state_dir)
    if chosen is None:
        raise RuntimeError(f"No checkpoint found under {config.proto.storage.remote_state_dir}")
    ok = download_checkpoint_to_local(config.proto.storage.remote_state_dir, checkpoint_outdir, chosen)
    if not ok:
        raise RuntimeError(f"Failed to download checkpoint from {chosen}")
    db_path = checkpoint_outdir / "controller.sqlite3"
    if not db_path.exists():
        raise RuntimeError(f"Controller DB missing at {db_path}")
    return db_path


def load_attempt_regions(db_path: Path) -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    conn = sqlite3.connect(db_path)
    attempt_region: dict[tuple[str, int], str] = {}
    current_region: dict[str, str] = {}

    for task_id, attempt_id, region in conn.execute(
        """
        SELECT ta.task_id, ta.attempt_id, wa.str_value
        FROM task_attempts ta
        JOIN worker_attributes wa
          ON wa.worker_id = ta.worker_id
         AND wa.key = 'region'
        """
    ):
        attempt_region[(task_id, attempt_id)] = region

    for task_id, region in conn.execute(
        """
        SELECT t.task_id, wa.str_value
        FROM tasks t
        JOIN worker_attributes wa
          ON wa.worker_id = t.current_worker_id
         AND wa.key = 'region'
        WHERE t.current_worker_id IS NOT NULL
        """
    ):
        current_region[task_id] = region

    return attempt_region, current_region


def analyze(
    parquet_glob: str,
    db_path: Path,
    window: TimeWindow,
) -> dict:
    attempt_region, current_region = load_attempt_regions(db_path)
    bucket_cache: dict[str, str | None] = {}

    total_lines = 0
    total_path_mentions = 0
    matched_lines = 0
    matched_path_mentions = 0
    cross_region_lines = 0
    cross_region_path_mentions = 0

    op_type_counts = Counter()
    cross_region_op_type_counts = Counter()
    pair_counts = Counter()
    bucket_counts = Counter()
    job_counts = Counter()
    task_counts = Counter()
    unmatched_tasks = Counter()
    unknown_buckets = Counter()
    samples: list[dict] = []

    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT key, epoch_ms, data
        FROM read_parquet('{parquet_glob}')
        WHERE epoch_ms >= ? AND epoch_ms < ?
          AND key LIKE '/%:%'
          AND data LIKE '%gs://%'
        ORDER BY epoch_ms DESC
        """,
        [window.start_ms, window.end_ms],
    ).fetchall()

    for key, epoch_ms, data in rows:
        total_lines += 1
        task_id, attempt_str = key.rsplit(":", 1)
        attempt_id = int(attempt_str)
        paths = [m.group(0).rstrip(".,:);]}") for m in GS_RE.finditer(data)]
        if not paths:
            continue
        total_path_mentions += len(paths)

        worker_region = attempt_region.get((task_id, attempt_id)) or current_region.get(task_id)
        if worker_region is None:
            unmatched_tasks[task_id] += 1
            continue

        matched_lines += 1
        matched_path_mentions += len(paths)
        op_type = classify_op(data)
        op_type_counts[op_type] += len(paths)

        line_cross_region = False
        for path in paths:
            bucket_name = path.split("/", 3)[2]
            region = bucket_region(bucket_name, bucket_cache)
            if region is None:
                unknown_buckets[bucket_name] += 1
                continue
            if region != worker_region:
                line_cross_region = True
                cross_region_path_mentions += 1
                cross_region_op_type_counts[op_type] += 1
                pair_counts[f"{worker_region}->{region}"] += 1
                bucket_counts[bucket_name.rstrip(":")] += 1
                job_counts[task_id.rsplit("/", 1)[0]] += 1
                task_counts[task_id] += 1
                if len(samples) < 50:
                    samples.append(
                        {
                            "timestamp_ms": epoch_ms,
                            "task_id": task_id,
                            "attempt_id": attempt_id,
                            "worker_region": worker_region,
                            "bucket_region": region,
                            "bucket": bucket_name.rstrip(":"),
                            "path": path,
                            "op_type": op_type,
                            "data": data,
                        }
                    )
        if line_cross_region:
            cross_region_lines += 1

    return {
        "window_start_ms": window.start_ms,
        "window_end_ms": window.end_ms,
        "total_gs_log_lines": total_lines,
        "total_gs_path_mentions": total_path_mentions,
        "matched_gs_log_lines": matched_lines,
        "matched_gs_path_mentions": matched_path_mentions,
        "cross_region_log_lines": cross_region_lines,
        "cross_region_path_mentions": cross_region_path_mentions,
        "all_op_type_counts": dict(op_type_counts),
        "cross_region_op_type_counts": dict(cross_region_op_type_counts),
        "cross_region_region_pairs": dict(pair_counts.most_common()),
        "cross_region_buckets": dict(bucket_counts.most_common(50)),
        "cross_region_jobs": dict(job_counts.most_common(50)),
        "cross_region_tasks": dict(task_counts.most_common(50)),
        "unmatched_tasks_top25": dict(unmatched_tasks.most_common(25)),
        "unknown_buckets_top25": dict(unknown_buckets.most_common(25)),
        "samples": samples,
    }


def write_csv(summary: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "key", "count"])
        for key, value in summary["cross_region_region_pairs"].items():
            writer.writerow(["region_pair", key, value])
        for key, value in summary["cross_region_buckets"].items():
            writer.writerow(["bucket", key, value])
        for key, value in summary["cross_region_jobs"].items():
            writer.writerow(["job", key, value])
        for key, value in summary["cross_region_tasks"].items():
            writer.writerow(["task", key, value])


def main() -> None:
    args = parse_args()
    window = parse_window(args)
    outdir = Path(args.outdir) if args.outdir is not None else default_outdir()
    outdir.mkdir(parents=True, exist_ok=True)

    config = IrisConfig.load(args.config)
    remote_logs_dir = f"{config.proto.storage.remote_state_dir.rstrip('/')}/logs"

    log_entries = choose_log_objects(remote_logs_dir, window, args.download_lookback_hours)
    local_logs_dir = outdir / "logs"
    local_logs = download_log_objects(remote_logs_dir, log_entries, local_logs_dir)
    db_path = download_checkpoint(config, outdir, args.checkpoint_dir)

    parquet_glob = str(local_logs_dir / "*.parquet")
    summary = analyze(parquet_glob, db_path, window)
    summary["local_log_files"] = [str(path) for path in local_logs]
    summary["local_db_path"] = str(db_path)
    summary["remote_logs_dir"] = remote_logs_dir

    summary_json = Path(args.summary_json) if args.summary_json else outdir / "cross_region_ops_summary.json"
    appendix_csv = Path(args.appendix_csv) if args.appendix_csv else outdir / "cross_region_ops_appendix.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_csv(summary, appendix_csv)

    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "appendix_csv": str(appendix_csv),
                "local_db_path": str(db_path),
                "local_log_files": [str(path) for path in local_logs],
                "cross_region_log_lines": summary["cross_region_log_lines"],
                "cross_region_path_mentions": summary["cross_region_path_mentions"],
                "top_pairs": list(summary["cross_region_region_pairs"].items())[:10],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
