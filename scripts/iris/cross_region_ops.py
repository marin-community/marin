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

import csv
import datetime as dt
import json
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import click
import duckdb
import fsspec

from iris.cluster.config import IrisConfig
from iris.cluster.controller.checkpoint import _find_latest_checkpoint_dir, download_checkpoint_to_local
from rigging.filesystem import get_bucket_location, region_from_prefix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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


def parse_window(end_time: str | None, hours: float) -> TimeWindow:
    if end_time is None:
        end = dt.datetime.now(dt.timezone.utc)
    else:
        text = end_time.replace("Z", "+00:00")
        end = dt.datetime.fromisoformat(text)
        if end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)
        else:
            end = end.astimezone(dt.timezone.utc)
    start = end - dt.timedelta(hours=hours)
    return TimeWindow(start=start, end=end)


DEFAULT_OUTDIR = Path("/tmp/cross-region")


def default_outdir() -> Path:
    DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTDIR


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
    logging.info(f"Listing log objects in {remote_logs_dir}")
    fs, path = fsspec.core.url_to_fs(remote_logs_dir)
    entries = sorted(fs.ls(path, detail=True), key=lambda e: e["mtime"])
    logging.info(f"Found {len(entries)} total log objects")
    mtime_cutoff = window.start - dt.timedelta(hours=lookback_hours)
    logging.info(f"Analysis window: {window.start.isoformat()} to {window.end.isoformat()}")
    logging.info(f"Lookback: {lookback_hours} hours, so filtering for objects modified after {mtime_cutoff.isoformat()}")

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
        before_mtime = before_window["mtime"].astimezone(dt.timezone.utc)
        logging.info(f"Included 1 file before cutoff (mtime: {before_mtime.isoformat()})")

    if chosen:
        oldest = chosen[0]["mtime"].astimezone(dt.timezone.utc)
        newest = chosen[-1]["mtime"].astimezone(dt.timezone.utc)
        logging.info(f"Selected {len(chosen)} files (mtime range: {oldest.isoformat()} to {newest.isoformat()})")
        logging.info(f"File names: {[Path(e['name']).name for e in chosen]}")
    return chosen


def download_log_objects(remote_logs_dir: str, entries: list[dict], target_dir: Path) -> list[Path]:
    fs, _ = fsspec.core.url_to_fs(remote_logs_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading {len(entries)} parquet files to {target_dir}")

    local_paths: list[Path] = []
    for i, entry in enumerate(entries, 1):
        name = entry["name"]
        remote_size = entry.get("size")
        local_path = target_dir / Path(name).name
        cached_ok = local_path.exists() and remote_size is not None and local_path.stat().st_size == remote_size
        if cached_ok:
            logging.info(f"  [{i}/{len(entries)}] {Path(name).name} already cached")
        else:
            if local_path.exists():
                logging.info(
                    f"  [{i}/{len(entries)}] {Path(name).name} cached but size mismatch "
                    f"(local={local_path.stat().st_size}, remote={remote_size}); re-downloading"
                )
            else:
                logging.info(f"  [{i}/{len(entries)}] Downloading {Path(name).name}")
            tmp_path = local_path.with_suffix(local_path.suffix + ".part")
            with fs.open(name, "rb") as src, tmp_path.open("wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            tmp_path.replace(local_path)
        local_paths.append(local_path)
    logging.info(f"Finished downloading {len(local_paths)} parquet files")
    return local_paths


def validate_parquet_files(paths: list[Path]) -> list[Path]:
    """Return parquets whose footer duckdb can read; log and drop the rest.

    The most recent segment is frequently still being written upstream, so the
    GCS read races the writer and we get a file without a parquet footer.
    """
    con = duckdb.connect()
    good: list[Path] = []
    for path in paths:
        try:
            con.execute(f"SELECT 1 FROM read_parquet('{path}') LIMIT 0")
            good.append(path)
        except duckdb.Error as exc:
            logging.warning(f"Skipping unreadable parquet {path.name}: {exc}")
            # Remove so a later resume re-downloads it fresh.
            path.unlink(missing_ok=True)
    return good


def download_checkpoint(config: IrisConfig, outdir: Path, checkpoint_dir: str | None) -> Path:
    chosen = checkpoint_dir or _find_latest_checkpoint_dir(config.proto.storage.remote_state_dir)
    if chosen is None:
        raise RuntimeError(f"No checkpoint found under {config.proto.storage.remote_state_dir}")
    # Key the local cache on the remote checkpoint id so reruns that pick up
    # a newer checkpoint don't collide with a stale one.
    checkpoint_id = chosen.rstrip("/").rsplit("/", 1)[-1]
    checkpoint_outdir = outdir / "checkpoint" / checkpoint_id
    checkpoint_outdir.mkdir(parents=True, exist_ok=True)
    db_path = checkpoint_outdir / "controller.sqlite3"
    if db_path.exists():
        logging.info(f"Checkpoint {checkpoint_id} already cached at {db_path}")
        return db_path
    logging.info(f"Downloading checkpoint {chosen}")
    ok = download_checkpoint_to_local(config.proto.storage.remote_state_dir, checkpoint_outdir, chosen)
    if not ok:
        raise RuntimeError(f"Failed to download checkpoint from {chosen}")
    if not db_path.exists():
        raise RuntimeError(f"Controller DB missing at {db_path}")
    logging.info(f"Checkpoint downloaded to {db_path}")
    return db_path


def load_attempt_regions(db_path: Path) -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    logging.info(f"Loading task attempt regions from {db_path}")
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

    logging.info(f"Loaded {len(attempt_region)} attempt regions and {len(current_region)} current regions")
    return attempt_region, current_region


def analyze(
    parquet_paths: list[Path],
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

    logging.info(f"Querying {len(parquet_paths)} parquet files for logs with gs:// paths")
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT key, epoch_ms, data
        FROM read_parquet(?)
        WHERE epoch_ms >= ? AND epoch_ms < ?
          AND key LIKE '/%:%'
          AND data LIKE '%gs://%'
        ORDER BY epoch_ms DESC
        """,
        [[str(p) for p in parquet_paths], window.start_ms, window.end_ms],
    ).fetchall()
    logging.info(f"Found {len(rows)} log lines with gs:// paths in time window")

    for i, (key, epoch_ms, data) in enumerate(rows, 1):
        if i % 10000 == 0:
            logging.info(f"  Processed {i}/{len(rows)} log lines...")

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

    logging.info("Analysis complete:")
    logging.info(f"  Total log lines with gs:// paths: {total_lines}")
    logging.info(f"  Matched to task regions: {matched_lines} ({100*matched_lines//total_lines if total_lines else 0}%)")
    logging.info(f"  Cross-region log lines: {cross_region_lines}")
    logging.info(f"  Cross-region path mentions: {cross_region_path_mentions}")
    logging.info(f"  Region pairs found: {len(pair_counts)}")
    logging.info(f"  Top region pair: {pair_counts.most_common(1)[0] if pair_counts else 'N/A'}")
    logging.info(f"  Unmatched tasks: {len(unmatched_tasks)}")
    logging.info(f"  Unknown buckets: {len(unknown_buckets)}")

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


def _fmt_ms(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).isoformat()


def _md_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_(none)_\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def write_markdown(summary: dict, path: Path) -> None:
    start = _fmt_ms(summary["window_start_ms"])
    end = _fmt_ms(summary["window_end_ms"])
    total = summary["total_gs_log_lines"]
    matched = summary["matched_gs_log_lines"]
    match_pct = (100 * matched // total) if total else 0

    lines: list[str] = []
    lines.append(f"# Cross-region GCS operations — {start} to {end}\n")
    lines.append("## Totals\n")
    lines.append(
        _md_table(
            ["Metric", "Value"],
            [
                ["Log lines with gs:// paths", total],
                ["Path mentions", summary["total_gs_path_mentions"]],
                ["Lines matched to worker region", f"{matched} ({match_pct}%)"],
                ["Cross-region lines", summary["cross_region_log_lines"]],
                ["Cross-region path mentions", summary["cross_region_path_mentions"]],
            ],
        )
    )

    lines.append("\n## Op type mix (path mentions)\n")
    lines.append(
        _md_table(
            ["Op", "All", "Cross-region"],
            [
                [op, summary["all_op_type_counts"].get(op, 0), summary["cross_region_op_type_counts"].get(op, 0)]
                for op in ("read", "write", "other")
            ],
        )
    )

    lines.append("\n## Top region pairs (source → destination)\n")
    lines.append(
        _md_table(
            ["Pair", "Mentions"],
            [[k, v] for k, v in list(summary["cross_region_region_pairs"].items())[:25]],
        )
    )

    lines.append("\n## Top cross-region buckets\n")
    lines.append(
        _md_table(
            ["Bucket", "Mentions"],
            [[k, v] for k, v in list(summary["cross_region_buckets"].items())[:25]],
        )
    )

    lines.append("\n## Top cross-region jobs\n")
    lines.append(
        _md_table(
            ["Job", "Mentions"],
            [[k, v] for k, v in list(summary["cross_region_jobs"].items())[:25]],
        )
    )

    lines.append("\n## Top cross-region tasks\n")
    lines.append(
        _md_table(
            ["Task", "Mentions"],
            [[k, v] for k, v in list(summary["cross_region_tasks"].items())[:25]],
        )
    )

    if summary.get("unknown_buckets_top25"):
        lines.append("\n## Buckets with unknown region\n")
        lines.append(
            _md_table(
                ["Bucket", "Mentions"],
                [[k, v] for k, v in summary["unknown_buckets_top25"].items()],
            )
        )

    samples = summary.get("samples") or []
    if samples:
        lines.append("\n## Sample log lines (up to 20)\n")
        for s in samples[:20]:
            ts = _fmt_ms(s["timestamp_ms"])
            lines.append(
                f"- `{ts}` `{s['worker_region']} → {s['bucket_region']}` "
                f"`{s['op_type']}` `{s['task_id']}#{s['attempt_id']}` `{s['path']}`"
            )
        lines.append("")

    path.write_text("\n".join(lines))


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


@click.command(help=__doc__)
@click.option(
    "--config",
    default="lib/iris/examples/marin.yaml",
    help="Iris cluster config to use. Defaults to lib/iris/examples/marin.yaml.",
)
@click.option(
    "--outdir",
    default=None,
    help="Directory for downloaded parquet segments, checkpoint DB, and reports. Defaults to a temp dir.",
)
@click.option(
    "--hours",
    type=float,
    default=24.0,
    help="Rolling analysis window in hours. Defaults to 24.",
)
@click.option(
    "--end-time",
    default=None,
    help="Window end in ISO8601 UTC, e.g. 2026-04-09T21:38:40Z. Defaults to now UTC.",
)
@click.option(
    "--checkpoint-dir",
    default=None,
    help="Optional remote checkpoint directory to use instead of the latest checkpoint.",
)
@click.option(
    "--download-lookback-hours",
    type=float,
    default=12.0,
    help="Extra lookback on log object mtimes when choosing parquet segments. Defaults to 12.",
)
@click.option(
    "--summary-json",
    default=None,
    help="Path to write the JSON summary. Defaults to <outdir>/cross_region_ops_summary.json.",
)
@click.option(
    "--appendix-csv",
    default=None,
    help="Path to write the CSV appendix. Defaults to <outdir>/cross_region_ops_appendix.csv.",
)
@click.option(
    "--report-md",
    default=None,
    help="Path to write the Markdown report. Defaults to <outdir>/cross_region_ops_report.md.",
)
def main(
    config: str,
    outdir: str | None,
    hours: float,
    end_time: str | None,
    checkpoint_dir: str | None,
    download_lookback_hours: float,
    summary_json: str | None,
    appendix_csv: str | None,
    report_md: str | None,
) -> None:
    window = parse_window(end_time, hours)
    out_path = Path(outdir) if outdir is not None else default_outdir()
    out_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting cross-region analysis for {window.start.isoformat()} to {window.end.isoformat()}")
    logging.info(f"Output directory: {out_path}")

    cfg = IrisConfig.load(config)
    remote_logs_dir = f"{cfg.proto.storage.remote_state_dir.rstrip('/')}/logs"

    log_entries = choose_log_objects(remote_logs_dir, window, download_lookback_hours)
    local_logs_dir = out_path / "logs"
    local_logs = download_log_objects(remote_logs_dir, log_entries, local_logs_dir)
    local_logs = validate_parquet_files(local_logs)
    db_path = download_checkpoint(cfg, out_path, checkpoint_dir)

    summary = analyze(local_logs, db_path, window)
    summary["local_log_files"] = [str(path) for path in local_logs]
    summary["local_db_path"] = str(db_path)
    summary["remote_logs_dir"] = remote_logs_dir

    summary_json_path = Path(summary_json) if summary_json else out_path / "cross_region_ops_summary.json"
    appendix_csv_path = Path(appendix_csv) if appendix_csv else out_path / "cross_region_ops_appendix.csv"
    report_md_path = Path(report_md) if report_md else out_path / "cross_region_ops_report.md"
    logging.info(f"Writing summary to {summary_json_path}")
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_csv(summary, appendix_csv_path)
    logging.info(f"Writing appendix CSV to {appendix_csv_path}")
    write_markdown(summary, report_md_path)
    logging.info(f"Writing markdown report to {report_md_path}")

    gist_desc = f"Iris cross-region GCS ops {_fmt_ms(summary['window_start_ms'])} to {_fmt_ms(summary['window_end_ms'])}"
    logging.info("To publish this report as a private gist, run:")
    logging.info(f"  gh gist create --desc {gist_desc!r} {report_md_path} {appendix_csv_path}")

    logging.info("Analysis complete!")
    click.echo(
        json.dumps(
            {
                "summary_json": str(summary_json_path),
                "appendix_csv": str(appendix_csv_path),
                "report_md": str(report_md_path),
                "gist_command": f"gh gist create --desc {gist_desc!r} {report_md_path} {appendix_csv_path}",
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
