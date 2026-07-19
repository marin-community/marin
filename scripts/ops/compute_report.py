#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weekly Iris compute-accounting report.

Reads durable finelog namespaces and emits, per ISO week, with no controller
change:

- TPU chip-hours by user, split preemptible vs reserved (``iris.task``)
- chip-hours by capacity type and generation (``iris.task``)
- preemption events by pool/zone (``iris.provisioning``)
- MFU per job (``telltale.levanter_throughput_mfu``)

Placement (capacity type, generation, slice size, zone) is parsed from the
``iris.task`` ``worker_id`` string (e.g.
``marin-tpu-v5p-preemptible-32-us-east5-a-...-worker-1``), which is durable in
finelog — so a preempted attempt's placement is not lost the way the controller
DB's worker row is. The per-host chip count is a slice's chips divided by its
live hosts.

Publishes the markdown as a secret gist and posts a compact summary to Discord,
following ``scripts/ops/egress_report.py``. ``--dry-run`` prints the markdown and
posts nothing.

Design: ``.agents/projects/iris_compute_accounting/``. Attributing chip-hours to
the specific attempts that ended in preemption (waste vs. consumption) needs a
per-attempt terminal cause, which iris.task lacks; that is the remaining
follow-up.

Compute (the ``*_by_*`` functions) is separated from I/O so it is unit-tested by
registering synthetic DuckDB tables named ``telltale`` / ``provisioning`` /
``task``.
"""

import datetime as dt
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
import duckdb
from finelog.deploy.config import load_finelog_config
from iris.cluster.config import load_config
from rigging.filesystem.storage_path import prefix_join

from scripts.ops.cross_region import (
    TimeWindow,
    choose_log_objects,
    download_log_objects,
    validate_parquet_files,
)
from scripts.ops.discord import post

# Namespaces the report reads. iris.task is high volume (multi-GB/week); the
# others are small. Each maps to (duckdb view name, empty-view DDL) — the empty
# DDL declares the columns the rollups reference so a window with no segments
# yields empty results instead of a missing-column error.
_VIEW_FOR = {
    "telltale": (
        "telltale",
        "SELECT CAST(NULL AS VARCHAR) AS name, CAST(NULL AS DOUBLE) AS value, "
        "CAST(NULL AS TIMESTAMPTZ) AS ts, CAST(NULL AS VARCHAR) AS job_id, "
        "CAST(NULL AS VARCHAR) AS run WHERE false",
    ),
    "iris.provisioning": (
        "provisioning",
        "SELECT CAST(NULL AS VARCHAR) AS outcome, CAST(NULL AS TIMESTAMPTZ) AS ts, "
        "CAST(NULL AS VARCHAR) AS accelerator_variant, CAST(NULL AS VARCHAR) AS zone WHERE false",
    ),
    "iris.task": (
        "task",
        "SELECT CAST(NULL AS VARCHAR) AS task_id, CAST(NULL AS BIGINT) AS attempt_id, "
        "CAST(NULL AS VARCHAR) AS worker_id, CAST(NULL AS TIMESTAMPTZ) AS ts WHERE false",
    ),
}
NAMESPACES = tuple(_VIEW_FOR)

# Extra mtime slack when selecting segments: a finelog segment's mtime is its
# ship/compaction time, so any row with ts in the window lives in a segment
# whose mtime is >= window.start. The slack absorbs compaction lag.
SEGMENT_LOOKBACK_HOURS = 72.0

MFU_METRIC = "levanter_throughput_mfu"

# Max rows shown per table in the rendered report (users, pools, jobs).
TOP_ROWS = 20

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobMfu:
    job_id: str
    run: str
    mean_mfu: float
    steps: int


@dataclass(frozen=True)
class PoolPreemptions:
    accelerator_variant: str
    zone: str
    preemptions: int


@dataclass(frozen=True)
class UserChipHours:
    user: str
    preemptible: float
    reserved: float

    @property
    def total(self) -> float:
        return self.preemptible + self.reserved


@dataclass(frozen=True)
class CapacityGenChipHours:
    capacity: str  # "preemptible" | "reserved"
    generation: str  # "v4" | "v5p" | "v5e" | "v6e" | ...
    chip_hours: float


@dataclass(frozen=True)
class WeeklyReport:
    iso_week: str
    window: TimeWindow
    chip_hours: list[UserChipHours]
    by_capacity_gen: list[CapacityGenChipHours]
    chip_hour_coverage: float
    mfu: list[JobMfu]
    preemptions: list[PoolPreemptions]

    @property
    def total_preemptible(self) -> float:
        return sum(u.preemptible for u in self.chip_hours)

    @property
    def total_reserved(self) -> float:
        return sum(u.reserved for u in self.chip_hours)

    @property
    def total_preemptions(self) -> int:
        return sum(p.preemptions for p in self.preemptions)


# --------------------------------------------------------------------------- #
# ISO-week windowing
# --------------------------------------------------------------------------- #
def iso_week_window(iso_week: str) -> TimeWindow:
    """[Monday 00:00, next Monday 00:00) UTC for an ISO week like ``2026-W29``."""
    year_str, week_str = iso_week.upper().split("-W")
    monday = dt.date.fromisocalendar(int(year_str), int(week_str), 1)
    start = dt.datetime.combine(monday, dt.time.min, tzinfo=dt.UTC)
    return TimeWindow(start=start, end=start + dt.timedelta(days=7))


def previous_iso_week(today: dt.date) -> str:
    """The ISO week (``YYYY-Www``) of the Monday before ``today``'s week."""
    this_monday = today - dt.timedelta(days=today.weekday())
    prev = this_monday - dt.timedelta(days=7)
    y, w, _ = prev.isocalendar()
    return f"{y}-W{w:02d}"


# --------------------------------------------------------------------------- #
# Compute — pure DuckDB over views telltale / provisioning / task
# --------------------------------------------------------------------------- #
def mfu_per_job(con: duckdb.DuckDBPyConnection, window: TimeWindow) -> list[JobMfu]:
    """Time-averaged MFU per iris job.

    Every worker of a multi-host job forwards the same value, so collapse
    replicas per step (``any_value`` grouped by job/run/ts) before averaging.
    ``process_index`` is null in the data and is not used.
    """
    rows = con.execute(
        """
        WITH per_step AS (
          SELECT job_id, run, ts, any_value(value) AS mfu
          FROM telltale
          WHERE name = ? AND ts >= ? AND ts < ?
          GROUP BY job_id, run, ts)
        SELECT job_id, run, round(avg(mfu), 1), count(*)
        FROM per_step GROUP BY job_id, run ORDER BY count(*) DESC
        """,
        [MFU_METRIC, window.start, window.end],
    ).fetchall()
    return [JobMfu(j, r, m, n) for j, r, m, n in rows]


def preemptions_by_pool(con: duckdb.DuckDBPyConnection, window: TimeWindow) -> list[PoolPreemptions]:
    """Count of ``preempted`` provisioning outcomes per accelerator/zone.

    Cluster/pool grain with no user key; this is the preemption event rate.
    """
    rows = con.execute(
        """
        SELECT accelerator_variant, zone, count(*)
        FROM provisioning
        WHERE outcome = 'preempted' AND ts >= ? AND ts < ?
        GROUP BY 1, 2 ORDER BY count(*) DESC
        """,
        [window.start, window.end],
    ).fetchall()
    return [PoolPreemptions(v, z, n) for v, z, n in rows]


# A marin TPU worker_id encodes placement: marin-tpu-<gen>-<capacity>-<chips>-<zone>-...-worker-<n>.
# capacity, generation, and slice chip count are parsed from it; the per-host
# chip count is the slice's chips divided by its live hosts (SPMD, so every host
# of a slice emits iris.task rows). ondemand is folded into reserved.
_PARSE_WORKER = (
    "regexp_extract(worker_id, '^marin-tpu-([a-z0-9]+)-(reserved|preemptible|ondemand)-([0-9]+)-', "
    "['generation', 'cap', 'chips'])"
)

# CTE that resolves per (task, attempt, worker): user, capacity, generation, and
# chip-hours = host active-seconds * (slice chips / live hosts in slice) / 3600.
# Bind the window twice (chips CTE + the caller's SELECT reuse the same view).
_CHIP_HOURS_CTE = f"""
WITH parsed AS (
  SELECT split_part(task_id, '/', 2) AS user,
         {_PARSE_WORKER} AS p,
         regexp_replace(worker_id, '-worker-[0-9]+$', '') AS slice_id,
         worker_id, ts
  FROM task
  WHERE worker_id LIKE 'marin-tpu-%' AND ts >= ? AND ts < ?),
host AS (
  SELECT user,
         CASE WHEN p.cap = 'preemptible' THEN 'preemptible' ELSE 'reserved' END AS capacity,
         p.generation AS generation, slice_id, CAST(p.chips AS BIGINT) AS slice_chips, worker_id,
         date_diff('second', min(ts), max(ts)) AS active_seconds
  FROM parsed WHERE p.cap <> '' GROUP BY 1, 2, 3, 4, 5, 6),
chip AS (
  SELECT user, capacity, generation,
         active_seconds * slice_chips / count(*) OVER (PARTITION BY slice_id) / 3600.0 AS chip_hours
  FROM host)
"""


def chip_hours_by_user(con: duckdb.DuckDBPyConnection, window: TimeWindow) -> list[UserChipHours]:
    """TPU chip-hours per user, split preemptible vs reserved, from iris.task.

    Placement comes from the worker_id string (see ``_CHIP_HOURS_CTE``); no
    controller state is consulted, so a preempted attempt's placement is not lost.
    """
    rows = con.execute(
        _CHIP_HOURS_CTE
        + """
        SELECT user,
               coalesce(sum(chip_hours) FILTER (WHERE capacity = 'preemptible'), 0),
               coalesce(sum(chip_hours) FILTER (WHERE capacity = 'reserved'), 0)
        FROM chip GROUP BY user ORDER BY sum(chip_hours) DESC
        """,
        [window.start, window.end],
    ).fetchall()
    return [UserChipHours(u, round(p, 1), round(r, 1)) for u, p, r in rows]


def chip_hours_by_capacity_gen(con: duckdb.DuckDBPyConnection, window: TimeWindow) -> list[CapacityGenChipHours]:
    """TPU chip-hours by capacity type and generation."""
    rows = con.execute(
        _CHIP_HOURS_CTE
        + """
        SELECT capacity, generation, round(sum(chip_hours), 1)
        FROM chip GROUP BY 1, 2 ORDER BY sum(chip_hours) DESC
        """,
        [window.start, window.end],
    ).fetchall()
    return [CapacityGenChipHours(c, g, h) for c, g, h in rows]


def chip_hour_coverage(con: duckdb.DuckDBPyConnection, window: TimeWindow) -> float:
    """Share of active worker-seconds on a parseable marin-tpu worker (0..1).

    The rest (GPU/CoreWeave, CPU, non-standard names) has no chip-hour attribution
    yet; reporting coverage keeps the chip-hour totals honest.
    """
    row = con.execute(
        """
        WITH attempt AS (
          SELECT worker_id, task_id, attempt_id, date_diff('second', min(ts), max(ts)) AS active_seconds
          FROM task WHERE worker_id IS NOT NULL AND ts >= ? AND ts < ?
          GROUP BY 1, 2, 3)
        SELECT coalesce(sum(active_seconds) FILTER (WHERE worker_id LIKE 'marin-tpu-%'), 0),
               coalesce(sum(active_seconds), 0)
        FROM attempt
        """,
        [window.start, window.end],
    ).fetchone()
    tpu_seconds, total_seconds = row if row else (0, 0)
    return tpu_seconds / total_seconds if total_seconds else 0.0


def build_report(con: duckdb.DuckDBPyConnection, iso_week: str) -> WeeklyReport:
    window = iso_week_window(iso_week)
    return WeeklyReport(
        iso_week=iso_week,
        window=window,
        chip_hours=chip_hours_by_user(con, window),
        by_capacity_gen=chip_hours_by_capacity_gen(con, window),
        chip_hour_coverage=chip_hour_coverage(con, window),
        mfu=mfu_per_job(con, window),
        preemptions=preemptions_by_pool(con, window),
    )


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(none)_\n"
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}\n"


def render_markdown(report: WeeklyReport) -> str:
    out = [
        f"# Iris compute — {report.iso_week}",
        f"_{report.window.start:%Y-%m-%d} → {report.window.end:%Y-%m-%d} UTC. "
        "TPU chip-hours from iris.task worker placement; "
        f"{report.chip_hour_coverage:.0%} of active worker-time is on parseable TPU workers._",
        "",
        "## Chip-hours by user",
        f"Total: {report.total_preemptible:,.0f} preemptible + {report.total_reserved:,.0f} reserved chip-hours.",
        "",
        _table(
            ["user", "preemptible", "reserved", "total"],
            [
                [u.user, f"{u.preemptible:,.0f}", f"{u.reserved:,.0f}", f"{u.total:,.0f}"]
                for u in report.chip_hours[:TOP_ROWS]
            ],
        ),
        "## Chip-hours by capacity and generation",
        _table(
            ["capacity", "generation", "chip-hours"],
            [[c.capacity, c.generation, f"{c.chip_hours:,.0f}"] for c in report.by_capacity_gen],
        ),
        "## Preemption events by pool",
        f"Total preemptions this week: {report.total_preemptions}.",
        "",
        _table(
            ["accelerator", "zone", "preemptions"],
            [[p.accelerator_variant, p.zone, str(p.preemptions)] for p in report.preemptions[:TOP_ROWS]],
        ),
        "## MFU per job (Levanter training)",
        _table(
            ["job", "run", "mean MFU %", "steps"],
            [[m.job_id, m.run, f"{m.mean_mfu:.1f}", str(m.steps)] for m in report.mfu[:TOP_ROWS]],
        ),
    ]
    return "\n".join(out)


def compose_discord_summary(report: WeeklyReport, gist_url: str) -> str:
    top = report.chip_hours[0] if report.chip_hours else None
    top_str = f"{top.user} ({top.total:,.0f} chip-h)" if top else "—"
    return (
        f"**Iris compute — {report.iso_week}**\n"
        f"{report.total_preemptible:,.0f} preemptible + {report.total_reserved:,.0f} reserved chip-hours. "
        f"Top user: {top_str}. Preemption events: {report.total_preemptions}.\n"
        f"Full report: {gist_url}"
    )


# --------------------------------------------------------------------------- #
# I/O — fetch finelog segments, publish
# --------------------------------------------------------------------------- #
def load_finelog_views(
    con: duckdb.DuckDBPyConnection,
    remote_log_dir: str,
    window: TimeWindow,
    cache_dir: Path,
) -> None:
    """Fetch each namespace's segments in the window and register a DuckDB view.

    View names are the namespace with dots dropped: ``iris.provisioning`` →
    ``provisioning``, ``iris.task`` → ``task``, ``telltale`` → ``telltale``. When a
    namespace has no segments in the window (e.g. telltale before it existed), an
    empty view with the columns the rollups reference is registered so the
    queries return nothing instead of failing on a missing column.
    """
    for ns in NAMESPACES:
        view, empty_ddl = _VIEW_FOR[ns]
        ns_dir = prefix_join(remote_log_dir, ns)
        entries = choose_log_objects(ns_dir, window, SEGMENT_LOOKBACK_HOURS)
        paths = download_log_objects(ns_dir, entries, cache_dir / ns)
        good = validate_parquet_files(paths)
        if not good:
            con.execute(f"CREATE TABLE {view} AS {empty_ddl}")
            log.warning("no readable segments for %s in window", ns)
            continue
        path_list = ", ".join(f"'{p}'" for p in good)
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet([{path_list}], union_by_name=true)")


def create_gist(markdown: str, description: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(markdown)
        tmp = f.name
    url = subprocess.check_output(["gh", "gist", "create", "--desc", description, tmp], text=True).strip()
    Path(tmp).unlink(missing_ok=True)
    return url


@click.command()
@click.option("--config", default="lib/iris/config/marin.yaml", help="Iris cluster config (for finelog remote_log_dir).")
@click.option("--iso-week", default=None, help="ISO week YYYY-Www; default is the previous complete week.")
@click.option("--channel", default="internal-discuss", help="Discord channel for the summary.")
@click.option("--dry-run/--no-dry-run", default=True, help="Print markdown and post nothing (default).")
def main(config: str, iso_week: str | None, channel: str, dry_run: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    iso_week = iso_week or previous_iso_week(dt.datetime.now(dt.UTC).date())
    window = iso_week_window(iso_week)

    cfg = load_config(config)
    finelog_cfg = load_finelog_config(cfg.finelog.config)
    if not finelog_cfg.remote_log_dir:
        raise click.ClickException(f"finelog config {cfg.finelog.config!r} has no remote_log_dir.")

    con = duckdb.connect()
    with tempfile.TemporaryDirectory(prefix="compute_report_") as tmp:
        load_finelog_views(con, finelog_cfg.remote_log_dir, window, Path(tmp))
        report = build_report(con, iso_week)

    markdown = render_markdown(report)
    if dry_run:
        click.echo(markdown)
        return

    gist_url = create_gist(markdown, f"Iris compute — {iso_week}")
    post(channel, compose_discord_summary(report, gist_url))
    log.info("posted %s summary; gist %s", iso_week, gist_url)


if __name__ == "__main__":
    main()
