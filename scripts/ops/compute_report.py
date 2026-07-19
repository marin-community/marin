#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weekly Iris compute-accounting report.

Reads durable finelog namespaces and emits, per ISO week, three tables that run
today with no controller change:

- MFU per job (``telltale.levanter_throughput_mfu``)
- preemption events by pool/zone (``iris.provisioning``)
- active host-hours by user (``iris.task``; ``task_id`` carries the user)

Publishes the markdown as a secret gist and posts a compact summary to Discord,
following ``scripts/ops/egress_report.py``. ``--dry-run`` prints the markdown and
posts nothing.

Design: ``.agents/projects/iris_compute_accounting/``. Chip-hours and the
preemptible split need the ``iris.accounting`` namespace (Phase 2); host-hours is
the finelog-only stand-in until then.

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

from scripts.ops.cross_region import (
    TimeWindow,
    choose_log_objects,
    download_log_objects,
    validate_parquet_files,
)
from scripts.ops.discord import post

# Namespaces the Phase-1 report reads. iris.task is high volume (multi-GB/week);
# the others are small.
NAMESPACES = ("telltale", "iris.provisioning", "iris.task")

# Extra mtime slack when selecting segments: a finelog segment's mtime is its
# ship/compaction time, so any row with ts in the window lives in a segment
# whose mtime is >= window.start. The slack absorbs compaction lag.
SEGMENT_LOOKBACK_HOURS = 72.0

# OS usernames that are shared launch accounts rather than a person. Host-hours
# under these are not attributable in Phase 1.
SHARED_USERS = frozenset({"root", "ubuntu", "app", "runner", "local_admin"})

MFU_METRIC = "levanter_throughput_mfu"

TOP_JOBS = 20

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
class UserHostHours:
    user: str
    host_hours: float
    tasks: int


@dataclass(frozen=True)
class WeeklyReport:
    iso_week: str
    window: TimeWindow
    mfu: list[JobMfu]
    preemptions: list[PoolPreemptions]
    host_hours: list[UserHostHours]


# --------------------------------------------------------------------------- #
# ISO-week windowing
# --------------------------------------------------------------------------- #
def iso_week_window(iso_week: str) -> TimeWindow:
    """[Monday 00:00, next Monday 00:00) UTC for an ISO week like ``2026-W29``."""
    year_s, week_s = iso_week.upper().split("-W")
    monday = dt.date.fromisocalendar(int(year_s), int(week_s), 1)
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


def host_hours_by_user(con: duckdb.DuckDBPyConnection, window: TimeWindow) -> list[UserHostHours]:
    """Active host-hours per user from iris.task.

    ``task_id`` is ``/user/job/...``; active time per (task, attempt, worker) is
    its ``ts`` span. This is host-hours, not chip-hours (no device count in
    iris.task) — Phase 2's iris.accounting supplies chip-hours.
    """
    rows = con.execute(
        """
        WITH attempt AS (
          SELECT split_part(task_id, '/', 2) AS user, task_id, attempt_id, worker_id,
                 date_diff('second', min(ts), max(ts)) AS active_s
          FROM task
          WHERE ts >= ? AND ts < ?
          GROUP BY 1, 2, 3, 4)
        SELECT user, round(sum(active_s) / 3600.0, 1), count(DISTINCT task_id)
        FROM attempt GROUP BY user ORDER BY sum(active_s) DESC
        """,
        [window.start, window.end],
    ).fetchall()
    return [UserHostHours(u, h, t) for u, h, t in rows]


def attribution_coverage(host_hours: list[UserHostHours]) -> float:
    """Share of host-hours attributed to a non-shared username (0..1).

    The Phase-1 signal for whether per-user attribution is worth building on.
    """
    total = sum(u.host_hours for u in host_hours)
    if total <= 0:
        return 0.0
    attributed = sum(u.host_hours for u in host_hours if u.user not in SHARED_USERS)
    return attributed / total


def build_report(con: duckdb.DuckDBPyConnection, iso_week: str) -> WeeklyReport:
    window = iso_week_window(iso_week)
    return WeeklyReport(
        iso_week=iso_week,
        window=window,
        mfu=mfu_per_job(con, window),
        preemptions=preemptions_by_pool(con, window),
        host_hours=host_hours_by_user(con, window),
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
    cov = attribution_coverage(report.host_hours)
    total_preempt = sum(p.preemptions for p in report.preemptions)
    out = [
        f"# Iris compute — {report.iso_week}",
        f"_{report.window.start:%Y-%m-%d} → {report.window.end:%Y-%m-%d} UTC. "
        "Host-hours are a finelog-only stand-in for chip-hours until iris.accounting (Phase 2)._",
        "",
        "## Host-hours by user",
        f"Attribution coverage (non-shared usernames): {cov:.0%}.",
        "",
        _table(
            ["user", "host-hours", "tasks"],
            [[u.user, f"{u.host_hours:.1f}", str(u.tasks)] for u in report.host_hours[:TOP_JOBS]],
        ),
        "## Preemption events by pool",
        f"Total preemptions this week: {total_preempt}.",
        "",
        _table(
            ["accelerator", "zone", "preemptions"],
            [[p.accelerator_variant, p.zone, str(p.preemptions)] for p in report.preemptions[:TOP_JOBS]],
        ),
        "## MFU per job (Levanter training)",
        _table(
            ["job", "run", "mean MFU %", "steps"],
            [[m.job_id, m.run, f"{m.mean_mfu:.1f}", str(m.steps)] for m in report.mfu[:TOP_JOBS]],
        ),
    ]
    return "\n".join(out)


def compose_discord_summary(report: WeeklyReport, gist_url: str) -> str:
    cov = attribution_coverage(report.host_hours)
    total_preempt = sum(p.preemptions for p in report.preemptions)
    top = report.host_hours[0] if report.host_hours else None
    top_s = f"{top.user} ({top.host_hours:.0f}h)" if top else "—"
    return (
        f"**Iris compute — {report.iso_week}**\n"
        f"Top user by host-hours: {top_s}. "
        f"Preemption events: {total_preempt}. "
        f"Attribution coverage: {cov:.0%}.\n"
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
    ``provisioning``, ``iris.task`` → ``task``, ``telltale`` → ``telltale``.
    """
    view_for = {"telltale": "telltale", "iris.provisioning": "provisioning", "iris.task": "task"}
    for ns in NAMESPACES:
        ns_dir = f"{remote_log_dir.rstrip('/')}/{ns}"
        entries = choose_log_objects(ns_dir, window, SEGMENT_LOOKBACK_HOURS)
        paths = download_log_objects(ns_dir, entries, cache_dir / ns)
        good = validate_parquet_files(paths)
        view = view_for[ns]
        if not good:
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM (SELECT NULL) WHERE false")
            log.warning("no readable segments for %s in window", ns)
            continue
        path_list = ", ".join(f"'{p}'" for p in good)
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet([{path_list}], union_by_name=true)")


def create_gist(markdown: str, description: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(markdown)
        tmp = f.name
    url = subprocess.check_output(
        ["gh", "gist", "create", "--desc", description, tmp], text=True
    ).strip()
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
