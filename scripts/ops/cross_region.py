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
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Extensions that typically carry bulk bytes. A single cross-region read of one
# of these is orders of magnitude more expensive than a small JSON/config file.
LARGE_EXTS = frozenset(
    {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".parquet",
        ".arrow",
        ".tfrecord",
        ".tfrecords",
        ".npz",
        ".npy",
        ".tar",
        ".tgz",
        ".zip",
        ".msgpack",
        ".h5",
        ".gguf",
        ".bag",
    }
)
# Compressed record streams — individually moderate but usually many per job.
MEDIUM_EXTS = frozenset({".jsonl.gz", ".json.gz", ".jsonl.zst", ".jsonl.xz", ".jsonl", ".csv.gz"})
SMALL_EXTS = frozenset({".json", ".yaml", ".yml", ".txt", ".log", ".md", ".toml", ".csv", ".cfg", ".ini"})


def _extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    # Two-suffix forms (.jsonl.gz etc.) checked before the single-suffix fallback.
    for ext in MEDIUM_EXTS:
        if name.endswith(ext):
            return ext
    if "." not in name:
        return ""
    tail = name.rsplit(".", 1)[-1]
    # A real extension is short and alphanumeric — reject directory names like
    # "grug-train-9.00e-18-truncated-1_16-382154" that happen to contain a dot.
    if 1 <= len(tail) <= 12 and tail.isalnum():
        return "." + tail
    return ""


def classify_size_tier(path: str) -> str:
    ext = _extension(path)
    if ext in LARGE_EXTS:
        return "large"
    if ext in MEDIUM_EXTS:
        return "medium"
    if ext in SMALL_EXTS:
        return "small"
    lower = path.lower()
    # Path patterns that override a missing/ambiguous extension.
    if "compilation-cache" in lower or "cache_ledger" in lower:
        return "small"
    if "/checkpoints/" in lower or "/hf/step-" in lower or "model-" in lower or "optimizer" in lower:
        return "large"
    if "/documents/" in lower or "/tokenized/" in lower or "/data/" in lower:
        return "medium"
    return "unknown"


def extract_user(task_id: str) -> str:
    # task ids are formatted like "/<user>/...".
    parts = task_id.split("/", 2)
    return parts[1] if len(parts) > 1 and parts[1] else "<unknown>"


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


def _download_one(fs, entry: dict, target_dir: Path, index: int, total: int) -> Path:
    name = entry["name"]
    remote_size = entry.get("size")
    local_path = target_dir / Path(name).name
    cached_ok = local_path.exists() and remote_size is not None and local_path.stat().st_size == remote_size
    if cached_ok:
        logging.info(f"  [{index}/{total}] {Path(name).name} already cached")
        return local_path
    if local_path.exists():
        logging.info(
            f"  [{index}/{total}] {Path(name).name} cached but size mismatch "
            f"(local={local_path.stat().st_size}, remote={remote_size}); re-downloading"
        )
    else:
        logging.info(f"  [{index}/{total}] Downloading {Path(name).name}")
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    with fs.open(name, "rb") as src, tmp_path.open("wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    tmp_path.replace(local_path)
    return local_path


def download_log_objects(remote_logs_dir: str, entries: list[dict], target_dir: Path) -> list[Path]:
    fs, _ = fsspec.core.url_to_fs(remote_logs_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading {len(entries)} parquet files to {target_dir}")

    index_map = {id(entry): i for i, entry in enumerate(entries, 1)}
    total = len(entries)
    results: dict[int, Path] = {}

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(_download_one, fs, entry, target_dir, index_map[id(entry)], total): entry for entry in entries
        }
        for fut in as_completed(futures):
            entry = futures[fut]
            results[index_map[id(entry)]] = fut.result()

    local_paths = [results[i] for i in range(1, total + 1)]
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


@dataclass(frozen=True)
class AttemptWindow:
    """Time bounds and worker region for a single task attempt.

    `region` is ``None`` when the attempt's worker row was purged from the
    controller DB (ON DELETE CASCADE on `worker_attributes`). We still keep the
    window so we can bucket matching log lines as "attempt matched, region
    unknown" instead of silently fabricating a region from the current worker.
    """

    started_at_ms: int
    finished_at_ms: int
    region: str | None


# Grace margin (milliseconds) added to an attempt's upper bound when matching
# log lines. Checkpoint writes flush asynchronously and their log lines can
# land a few seconds after `finished_at_ms`; a wider grace would start to
# overlap with a subsequent resubmission of the same (task_id, attempt_id).
ATTEMPT_WINDOW_GRACE_MS = 30_000


def load_attempt_windows(db_path: Path) -> dict[tuple[str, int], AttemptWindow]:
    """Load (started_at_ms, finished_at_ms, region) for every attempt.

    We only return attempts that actually started (``started_at_ms IS NOT NULL``).
    In-flight attempts get ``finished_at_ms`` set to "now" so log lines from the
    currently-running worker still match.
    """
    logging.info(f"Loading task attempt windows from {db_path}")
    conn = sqlite3.connect(db_path)
    now_ms = int(time.time() * 1000)
    attempts: dict[tuple[str, int], AttemptWindow] = {}

    for task_id, attempt_id, started_at_ms, finished_at_ms, region in conn.execute(
        """
        SELECT ta.task_id,
               ta.attempt_id,
               ta.started_at_ms,
               COALESCE(ta.finished_at_ms, :now_ms) AS finished_at_ms,
               wa.str_value AS region
          FROM task_attempts ta
     LEFT JOIN worker_attributes wa
            ON wa.worker_id = ta.worker_id
           AND wa.key = 'region'
         WHERE ta.started_at_ms IS NOT NULL
        """,
        {"now_ms": now_ms},
    ):
        attempts[(task_id, attempt_id)] = AttemptWindow(
            started_at_ms=int(started_at_ms),
            finished_at_ms=int(finished_at_ms),
            region=region,
        )

    missing_region = sum(1 for a in attempts.values() if a.region is None)
    logging.info(f"Loaded {len(attempts)} attempt windows ({missing_region} with unknown worker region)")
    return attempts


def analyze(
    parquet_paths: list[Path],
    db_path: Path,
    window: TimeWindow,
) -> dict:
    attempts = load_attempt_windows(db_path)
    bucket_cache: dict[str, str | None] = {}

    total_lines = 0
    total_path_mentions = 0
    matched_lines = 0
    matched_path_mentions = 0
    cross_region_lines = 0
    cross_region_path_mentions = 0
    # Coverage counters for the time-window join. See the docstring on
    # `AttemptWindow` for why these exist — we no longer silently fall back to
    # the current worker's region, so we surface the drops here instead.
    lines_no_attempt_match = 0  # (task_id, attempt_id) not in DB at all
    lines_outside_attempt_window = 0  # attempt exists but epoch_ms outside [start, finish+grace]
    lines_matched_attempt_worker_unknown = 0  # attempt matched but worker_attributes was cascaded away

    op_type_counts = Counter()
    cross_region_op_type_counts = Counter()
    pair_counts = Counter()
    bucket_counts = Counter()
    job_counts = Counter()
    task_counts = Counter()
    user_counts = Counter()
    size_tier_counts = Counter()
    cross_region_size_tier_counts = Counter()
    cross_region_extension_counts = Counter()
    unmatched_tasks = Counter()
    unknown_buckets = Counter()
    samples: list[dict] = []
    # Per-user sample buffer: keep a small, diverse slice for each user so a
    # single loud user can't crowd everyone else out of the report.
    user_samples: dict[str, list[dict]] = {}
    # Dedup identical (task, attempt, path, ms) rows — the same log line
    # frequently appears twice in the parquet stream.
    seen_sample_keys: set[tuple] = set()
    # Collect a wider pool per user so we can rank by tier priority at the end
    # (large ≫ medium ≫ small ≫ unknown) rather than just taking whatever
    # arrived first in reverse-chron order.
    per_user_sample_cap = 10
    per_user_sample_pool = 40
    tier_priority = {"large": 0, "medium": 1, "unknown": 2, "small": 3}

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

        # Attempt match must be both (a) present in the DB and (b) contain
        # `epoch_ms` within its execution window. `attempt_id` is re-used across
        # resubmissions of a task with the same name, so a raw (task_id,
        # attempt_id) hit is not enough — without the time bound we'd flag log
        # lines from a prior, deleted attempt against the current attempt's
        # worker region (the bug this script's rewrite fixes).
        attempt = attempts.get((task_id, attempt_id))
        if attempt is None:
            unmatched_tasks[task_id] += 1
            lines_no_attempt_match += 1
            continue
        if not (attempt.started_at_ms <= epoch_ms <= attempt.finished_at_ms + ATTEMPT_WINDOW_GRACE_MS):
            lines_outside_attempt_window += 1
            continue
        if attempt.region is None:
            # Attempt matched, but its worker row was cascaded away before this
            # report ran, so we don't know what region actually executed it.
            # Counting it as cross-region from *any* fallback would re-introduce
            # the old bug; surface it instead.
            lines_matched_attempt_worker_unknown += 1
            continue
        worker_region = attempt.region

        matched_lines += 1
        matched_path_mentions += len(paths)
        op_type = classify_op(data)
        op_type_counts[op_type] += len(paths)

        user = extract_user(task_id)
        line_cross_region = False
        for path in paths:
            parts = path.split("/", 3)
            bucket_name = parts[2] if len(parts) > 2 else ""
            if not bucket_name:
                unknown_buckets["<empty>"] += 1
                continue
            size_tier = classify_size_tier(path)
            size_tier_counts[size_tier] += 1
            region = bucket_region(bucket_name, bucket_cache)
            if region is None:
                unknown_buckets[bucket_name] += 1
                continue
            if region != worker_region:
                line_cross_region = True
                cross_region_path_mentions += 1
                cross_region_op_type_counts[op_type] += 1
                cross_region_size_tier_counts[size_tier] += 1
                cross_region_extension_counts[_extension(path) or "<no-ext>"] += 1
                pair_counts[f"{worker_region}->{region}"] += 1
                bucket_counts[bucket_name.rstrip(":")] += 1
                job_counts[task_id.rsplit("/", 1)[0]] += 1
                task_counts[task_id] += 1
                user_counts[user] += 1

                sample_key = (task_id, attempt_id, path, epoch_ms)
                if sample_key in seen_sample_keys:
                    continue
                seen_sample_keys.add(sample_key)
                sample = {
                    "timestamp_ms": epoch_ms,
                    "user": user,
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "worker_region": worker_region,
                    "bucket_region": region,
                    "bucket": bucket_name.rstrip(":"),
                    "path": path,
                    "op_type": op_type,
                    "size_tier": size_tier,
                    "data": data,
                }
                if len(samples) < 50:
                    samples.append(sample)
                bucket = user_samples.setdefault(user, [])
                if len(bucket) < per_user_sample_pool:
                    bucket.append(sample)
        if line_cross_region:
            cross_region_lines += 1

    for u, pool in user_samples.items():
        pool.sort(key=lambda s: (tier_priority.get(s["size_tier"], 99), -s["timestamp_ms"]))
        user_samples[u] = pool[:per_user_sample_cap]

    logging.info("Analysis complete:")
    logging.info(f"  Total log lines with gs:// paths: {total_lines}")
    logging.info(f"  Matched to task regions: {matched_lines} ({100*matched_lines//total_lines if total_lines else 0}%)")
    logging.info(f"  Lines with no attempt in DB: {lines_no_attempt_match}")
    logging.info(f"  Lines outside attempt window (likely prior attempt): {lines_outside_attempt_window}")
    logging.info(f"  Lines with attempt matched but worker region unknown: {lines_matched_attempt_worker_unknown}")
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
        "lines_no_attempt_match": lines_no_attempt_match,
        "lines_outside_attempt_window": lines_outside_attempt_window,
        "lines_matched_attempt_worker_unknown": lines_matched_attempt_worker_unknown,
        "cross_region_log_lines": cross_region_lines,
        "cross_region_path_mentions": cross_region_path_mentions,
        "all_op_type_counts": dict(op_type_counts),
        "cross_region_op_type_counts": dict(cross_region_op_type_counts),
        "all_size_tier_counts": dict(size_tier_counts),
        "cross_region_size_tier_counts": dict(cross_region_size_tier_counts),
        "cross_region_extensions": dict(cross_region_extension_counts.most_common(30)),
        "cross_region_region_pairs": dict(pair_counts.most_common()),
        "cross_region_buckets": dict(bucket_counts.most_common(50)),
        "cross_region_jobs": dict(job_counts.most_common(50)),
        "cross_region_tasks": dict(task_counts.most_common(50)),
        "cross_region_users": dict(user_counts.most_common()),
        "unmatched_tasks_top25": dict(unmatched_tasks.most_common(25)),
        "unknown_buckets_top25": dict(unknown_buckets.most_common(25)),
        "samples": samples,
        "samples_by_user": user_samples,
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
                ["Lines with no attempt in DB", summary.get("lines_no_attempt_match", 0)],
                [
                    "Lines outside attempt window (prior attempt)",
                    summary.get("lines_outside_attempt_window", 0),
                ],
                [
                    "Lines matched but worker region unknown",
                    summary.get("lines_matched_attempt_worker_unknown", 0),
                ],
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

    lines.append("\n## Size tier mix (path mentions)\n")
    lines.append(
        "_Heuristic: `large` = model/checkpoint shards (safetensors/bin/parquet/…); "
        "`medium` = compressed record streams (jsonl.gz, …); `small` = config/log/metadata; "
        "path patterns like `checkpoints/` or `compilation-cache` override when no extension is present._\n"
    )
    lines.append(
        _md_table(
            ["Tier", "All", "Cross-region"],
            [
                [
                    tier,
                    summary["all_size_tier_counts"].get(tier, 0),
                    summary["cross_region_size_tier_counts"].get(tier, 0),
                ]
                for tier in ("large", "medium", "small", "unknown")
            ],
        )
    )

    lines.append("\n## Top cross-region extensions\n")
    lines.append(
        _md_table(
            ["Extension", "Mentions"],
            [[k, v] for k, v in list(summary["cross_region_extensions"].items())[:20]],
        )
    )

    lines.append("\n## Cross-region by user\n")
    lines.append(
        _md_table(
            ["User", "Mentions"],
            [[k, v] for k, v in list(summary["cross_region_users"].items())[:25]],
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

    samples_by_user = summary.get("samples_by_user") or {}
    if samples_by_user:
        lines.append("\n## Sample log lines by user\n")
        # Order users by how many cross-region mentions they racked up so the
        # worst offenders show up first.
        user_order = [u for u, _ in summary.get("cross_region_users", {}).items() if u in samples_by_user]
        for u in samples_by_user:
            if u not in user_order:
                user_order.append(u)
        for user in user_order:
            user_samples = samples_by_user[user]
            if not user_samples:
                continue
            lines.append(f"### `{user}` ({summary['cross_region_users'].get(user, len(user_samples))} mentions)\n")
            rows = []
            for s in user_samples:
                ts = _fmt_ms(s["timestamp_ms"]).split("+")[0]
                # Bold the cross-region path so it's obvious what the wrong-region ref is.
                # Collapse pipes and newlines so markdown table rendering survives log output.
                data = s["data"].replace(s["path"], f"**{s['path']}**", 1)
                data = data.replace("|", "\\|").replace("\n", " ")
                rows.append(
                    [
                        ts,
                        f"`{s['worker_region']} → {s['bucket_region']}`",
                        f"`{s['op_type']}/{s['size_tier']}`",
                        f"`{s['task_id']}` (attempt {s['attempt_id']})",
                        data,
                    ]
                )
            lines.append(_md_table(["Time", "Regions", "Op/Tier", "Task", "Log"], rows))
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
        for key, value in summary["cross_region_users"].items():
            writer.writerow(["user", key, value])
        for key, value in summary["cross_region_size_tier_counts"].items():
            writer.writerow(["size_tier", key, value])
        for key, value in summary["cross_region_extensions"].items():
            writer.writerow(["extension", key, value])


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
