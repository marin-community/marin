#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weekly Iris compute-accounting report.

Reads durable finelog namespaces and emits, per ISO week, with no controller
change:

- TPU chip-hours by user, split preemptible vs reserved (``iris.task``)
- top jobs by chip-hours, with per-job attempt (churn) count (``iris.task``)
- waste: chip-hours on attempts a later attempt superseded — an upper bound on
  preempt→re-place churn (``iris.task``)
- chip-hours by region and by capacity type / generation (``iris.task``)
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
class RegionChipHours:
    region: str  # "us-east5" | "us-central1" | "europe-west" | ...
    preemptible: float
    reserved: float

    @property
    def total(self) -> float:
        return self.preemptible + self.reserved


@dataclass(frozen=True)
class JobChipHours:
    job: str  # "/user/jobname"
    user: str
    generation: str
    chip_hours: float
    attempts: int  # distinct (task, attempt) pairs — a churn signal


@dataclass(frozen=True)
class RetriedWaste:
    """Chip-hours on attempts that a later attempt of the same task superseded.

    An upper bound on preemption/failure churn: an attempt followed by a higher
    attempt_id did not complete, so its compute was redone. The final attempt of
    each task is excluded (it completed, or is the last one seen in the window).
    """

    capacity: str  # "preemptible" | "reserved"
    retried: float
    total: float

    @property
    def pct(self) -> float:
        return 100.0 * self.retried / self.total if self.total else 0.0


@dataclass(frozen=True)
class WeeklyReport:
    iso_week: str
    window: TimeWindow
    chip_hours: list[UserChipHours]
    by_capacity_gen: list[CapacityGenChipHours]
    by_region: list[RegionChipHours]
    top_jobs: list[JobChipHours]
    retried_waste: list[RetriedWaste]
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
    def total_chip_hours(self) -> float:
        return self.total_preemptible + self.total_reserved

    @property
    def total_retried(self) -> float:
        return sum(w.retried for w in self.retried_waste)

    @property
    def retried_pct(self) -> float:
        return 100.0 * self.total_retried / self.total_chip_hours if self.total_chip_hours else 0.0

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


# A marin TPU worker_id encodes placement:
# marin-tpu-<gen>-<capacity>-<chips>-<zone>-<yyyymmdd>-<hhmm>-<hash>-worker-<n>.
# capacity, generation, slice chip count, and zone are parsed from it; the
# per-host chip count is the slice's chips divided by its live hosts (SPMD, so
# every host of a slice emits iris.task rows). ondemand is folded into reserved.
# `zn` avoids the reserved word `zone`.
_PARSE_WORKER = (
    "regexp_extract(worker_id, "
    "'^marin-tpu-([a-z0-9]+)-(reserved|preemptible|ondemand)-([0-9]+)-([a-z0-9-]+?)-[0-9]{8}-', "
    "['generation', 'cap', 'chips', 'zn'])"
)

# Full SELECT producing one row per (task, attempt, host): user, job (/user/job),
# capacity, generation, region, and chip-hours = host active-seconds *
# (slice chips / live hosts in slice) / 3600. ``materialize_chip_hours`` runs it
# once into a temp table the rollups read, so the iris.task scan and the
# worker_id regex parse happen once, not once per rollup.
_CHIP_HOURS_SELECT = f"""
WITH parsed AS (
  SELECT split_part(task_id, '/', 2) AS user,
         '/' || split_part(task_id, '/', 2) || '/' || split_part(task_id, '/', 3) AS job,
         task_id, attempt_id,
         {_PARSE_WORKER} AS p,
         regexp_replace(worker_id, '-worker-[0-9]+$', '') AS slice_id,
         worker_id, ts
  FROM task
  WHERE worker_id LIKE 'marin-tpu-%' AND ts >= ? AND ts < ?),
host AS (
  SELECT user, job, task_id, attempt_id,
         CASE WHEN p.cap = 'preemptible' THEN 'preemptible' ELSE 'reserved' END AS capacity,
         p.generation AS generation,
         regexp_extract(p.zn, '^([a-z]+-[a-z0-9]+)', 1) AS region,
         slice_id, CAST(p.chips AS BIGINT) AS slice_chips, worker_id,
         date_diff('second', min(ts), max(ts)) AS active_seconds
  FROM parsed WHERE p.cap <> '' GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
SELECT user, job, task_id, attempt_id, capacity, generation, region,
       active_seconds * slice_chips / count(*) OVER (PARTITION BY slice_id) / 3600.0 AS chip_hours
FROM host
"""


def materialize_chip_hours(con: duckdb.DuckDBPyConnection, window: TimeWindow) -> None:
    """Parse iris.task worker placement into a temp ``chip`` table for the window.

    One row per (task, attempt, host) with user/job/capacity/generation/region
    and chip-hours. Every chip-hour rollup reads this table, so the full
    iris.task scan and the worker_id regex parse run once. Placement comes from
    the worker_id string, so a preempted attempt's placement is not lost.
    """
    con.execute("DROP TABLE IF EXISTS chip")
    con.execute(f"CREATE TEMP TABLE chip AS {_CHIP_HOURS_SELECT}", [window.start, window.end])


def chip_hours_by_user(con: duckdb.DuckDBPyConnection) -> list[UserChipHours]:
    """TPU chip-hours per user, split preemptible vs reserved (reads ``chip``)."""
    rows = con.execute(
        """
        SELECT user,
               coalesce(sum(chip_hours) FILTER (WHERE capacity = 'preemptible'), 0),
               coalesce(sum(chip_hours) FILTER (WHERE capacity = 'reserved'), 0)
        FROM chip GROUP BY user ORDER BY sum(chip_hours) DESC
        """
    ).fetchall()
    return [UserChipHours(u, round(p), round(r)) for u, p, r in rows]


def chip_hours_by_capacity_gen(con: duckdb.DuckDBPyConnection) -> list[CapacityGenChipHours]:
    """TPU chip-hours by capacity type and generation (reads ``chip``)."""
    rows = con.execute(
        """
        SELECT capacity, generation, round(sum(chip_hours))
        FROM chip GROUP BY 1, 2 ORDER BY sum(chip_hours) DESC
        """
    ).fetchall()
    return [CapacityGenChipHours(c, g, h) for c, g, h in rows]


def chip_hours_by_region(con: duckdb.DuckDBPyConnection) -> list[RegionChipHours]:
    """TPU chip-hours by region, split preemptible vs reserved (reads ``chip``).

    Region is parsed from the worker_id zone (``us-east5-a`` -> ``us-east5``). Some
    pool names truncate the zone to varying lengths (``europe-wes`` /
    ``europe-west`` / ``europe-west4``); merge each region into the longest region
    it is a prefix of so one region is not split across rows.
    """
    rows = con.execute(
        """
        SELECT coalesce(nullif(region, ''), 'unknown'),
               coalesce(sum(chip_hours) FILTER (WHERE capacity = 'preemptible'), 0),
               coalesce(sum(chip_hours) FILTER (WHERE capacity = 'reserved'), 0)
        FROM chip GROUP BY 1
        """
    ).fetchall()
    names = [reg for reg, _, _ in rows]
    longest_first = sorted(names, key=len, reverse=True)
    canonical = {reg: next(c for c in longest_first if c.startswith(reg)) for reg in names}
    merged: dict[str, list[float]] = {}
    for reg, preempt, reserved in rows:
        acc = merged.setdefault(canonical[reg], [0.0, 0.0])
        acc[0] += preempt
        acc[1] += reserved
    out = [RegionChipHours(reg, round(p), round(r)) for reg, (p, r) in merged.items()]
    return sorted(out, key=lambda r: r.total, reverse=True)


def top_jobs(con: duckdb.DuckDBPyConnection, limit: int = TOP_ROWS) -> list[JobChipHours]:
    """Top jobs by chip-hours (``/user/jobname`` grain), with attempt count.

    ``attempts`` counts distinct (task, attempt) pairs under the job — a rough
    churn signal; a job re-placed many times shows more attempts than tasks.
    Reads ``chip``.
    """
    rows = con.execute(
        """
        SELECT job, any_value(user), mode(generation), round(sum(chip_hours)),
               count(DISTINCT task_id || '#' || attempt_id)
        FROM chip GROUP BY job ORDER BY sum(chip_hours) DESC LIMIT ?
        """,
        [limit],
    ).fetchall()
    return [JobChipHours(j, u, g, h, n) for j, u, g, h, n in rows]


def retried_attempt_waste(con: duckdb.DuckDBPyConnection) -> list[RetriedWaste]:
    """Chip-hours on superseded attempts, split by capacity — a waste upper bound.

    For each task the final (max) attempt is treated as productive; every earlier
    attempt's chip-hours are counted as redone work. This is the preemption/
    failure-churn overhead derivable from finelog alone (iris.task carries no
    terminal cause), so it over-counts intentional restarts and under-counts a
    final attempt that itself never completed. Reads ``chip``.
    """
    rows = con.execute(
        """
        WITH task_final AS (SELECT task_id, max(attempt_id) AS final_attempt FROM chip GROUP BY task_id)
        SELECT c.capacity,
               round(coalesce(sum(c.chip_hours) FILTER (WHERE c.attempt_id < f.final_attempt), 0)),
               round(sum(c.chip_hours))
        FROM chip c JOIN task_final f USING (task_id)
        GROUP BY c.capacity ORDER BY c.capacity
        """
    ).fetchall()
    return [RetriedWaste(cap, retried, total) for cap, retried, total in rows]


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
    materialize_chip_hours(con, window)
    return WeeklyReport(
        iso_week=iso_week,
        window=window,
        chip_hours=chip_hours_by_user(con),
        by_capacity_gen=chip_hours_by_capacity_gen(con),
        by_region=chip_hours_by_region(con),
        top_jobs=top_jobs(con),
        retried_waste=retried_attempt_waste(con),
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
        "## Headline",
        f"- **{report.total_chip_hours:,.0f} TPU chip-hours** — "
        f"{report.total_preemptible:,.0f} preemptible + {report.total_reserved:,.0f} reserved.",
        f"- **{report.total_retried:,.0f} chip-hours ({report.retried_pct:.0f}%) on attempts that did not complete** — "
        "redone after preemption or failure (upper bound; iris.task has no terminal cause).",
        f"- **{report.total_preemptions} preemption events** across pools.",
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
        "## Top jobs by chip-hours",
        _table(
            ["job", "user", "generation", "chip-hours", "attempts"],
            [[j.job, j.user, j.generation, f"{j.chip_hours:,.0f}", str(j.attempts)] for j in report.top_jobs],
        ),
        "## Waste: chip-hours redone after preemption or failure",
        "Compute on attempts a later attempt superseded — an upper bound on preempt→re-place churn.",
        "",
        _table(
            ["capacity", "redone chip-hours", "total chip-hours", "share"],
            [[w.capacity, f"{w.retried:,.0f}", f"{w.total:,.0f}", f"{w.pct:.0f}%"] for w in report.retried_waste],
        ),
        "## Chip-hours by region",
        _table(
            ["region", "preemptible", "reserved", "total"],
            [[r.region, f"{r.preemptible:,.0f}", f"{r.reserved:,.0f}", f"{r.total:,.0f}"] for r in report.by_region],
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
    top_job = report.top_jobs[0] if report.top_jobs else None
    job_str = f"{top_job.job} ({top_job.chip_hours:,.0f} chip-h)" if top_job else "—"
    return (
        f"**Iris compute — {report.iso_week}**\n"
        f"{report.total_chip_hours:,.0f} TPU chip-hours "
        f"({report.total_preemptible:,.0f} preemptible + {report.total_reserved:,.0f} reserved). "
        f"{report.total_retried:,.0f} ({report.retried_pct:.0f}%) redone after preemption/failure.\n"
        f"Top user: {top_str}. Top job: {job_str}. Preemption events: {report.total_preemptions}.\n"
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
