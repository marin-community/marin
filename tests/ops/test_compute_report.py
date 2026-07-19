# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the compute-accounting rollups.

The rollup functions take a DuckDB connection with views named ``telltale`` /
``provisioning`` / ``task``; these tests register synthetic tables under those
names and assert the aggregation, so no finelog or network access is needed.
"""

import datetime as dt

import duckdb
import pytest

from scripts.ops.compute_report import (
    _VIEW_FOR,
    build_report,
    chip_hour_coverage,
    chip_hours_by_capacity_gen,
    chip_hours_by_region,
    chip_hours_by_user,
    iso_week_window,
    materialize_chip_hours,
    mfu_per_job,
    preemptions_by_pool,
    previous_iso_week,
    render_markdown,
    retried_attempt_waste,
    top_jobs,
)

WEEK = "2026-W29"  # Mon 2026-07-13 .. Mon 2026-07-20 UTC


def _ts(day: int, hour: int = 12) -> dt.datetime:
    return dt.datetime(2026, 7, day, hour, tzinfo=dt.UTC)


def _wid(gen: str, cap: str, chips: int, host: int, tag: str = "aaaa1111", zone: str = "us-east5-a") -> str:
    return f"marin-tpu-{gen}-{cap}-{chips}-{zone}-20260101-0000-{tag}-worker-{host}"


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE telltale(name VARCHAR, value DOUBLE, ts TIMESTAMPTZ, "
        "job_id VARCHAR, run VARCHAR, worker VARCHAR)"
    )
    c.execute("CREATE TABLE provisioning(outcome VARCHAR, ts TIMESTAMPTZ, accelerator_variant VARCHAR, zone VARCHAR)")
    c.execute("CREATE TABLE task(task_id VARCHAR, attempt_id BIGINT, worker_id VARCHAR, ts TIMESTAMPTZ)")
    return c


def _run_hour(con, task_id: str, worker_id: str, day: int = 14, attempt: int = 0):
    # One attempt active for exactly one hour on one host (two samples, 1h apart).
    con.execute("INSERT INTO task VALUES (?, ?, ?, ?)", [task_id, attempt, worker_id, _ts(day, 10)])
    con.execute("INSERT INTO task VALUES (?, ?, ?, ?)", [task_id, attempt, worker_id, _ts(day, 11)])


def _chip(con):
    """Materialize the chip table for WEEK; chip-hour rollups read it."""
    materialize_chip_hours(con, iso_week_window(WEEK))
    return con


def test_iso_week_window():
    w = iso_week_window(WEEK)
    assert w.start == dt.datetime(2026, 7, 13, tzinfo=dt.UTC)
    assert w.end == dt.datetime(2026, 7, 20, tzinfo=dt.UTC)


def test_previous_iso_week():
    # A Thursday in 2026-W30 -> previous complete week is W29.
    assert previous_iso_week(dt.date(2026, 7, 23)) == "2026-W29"


def test_mfu_collapses_worker_replicas(con):
    # One job, two workers forwarding the SAME value each of 3 steps: mean is the
    # value, and step count is 3 (not 6) — replicas collapsed per step.
    for step in range(3):
        for worker in ("w0", "w1"):
            con.execute(
                "INSERT INTO telltale VALUES ('levanter_throughput_mfu', 40.0, ?, '/u/job', 'run-a', ?)",
                [_ts(14, 10 + step), worker],
            )
    # A row outside the window is ignored.
    con.execute("INSERT INTO telltale VALUES ('levanter_throughput_mfu', 99.0, ?, '/u/job', 'run-a', 'w0')", [_ts(1)])
    (res,) = mfu_per_job(con, iso_week_window(WEEK))
    assert res.job_id == "/u/job" and res.run == "run-a"
    assert res.mean_mfu == 40.0
    assert res.steps == 3


def test_preemptions_by_pool(con):
    rows = [
        ("preempted", _ts(14), "v5p-8", "us-east5-a"),
        ("preempted", _ts(15), "v5p-8", "us-east5-a"),
        ("preempted", _ts(15), "v6e-4", "us-east1-d"),
        ("ready", _ts(15), "v5p-8", "us-east5-a"),  # non-preempt ignored
        ("preempted", _ts(1), "v5p-8", "us-east5-a"),  # out of window
    ]
    for r in rows:
        con.execute("INSERT INTO provisioning VALUES (?, ?, ?, ?)", list(r))
    out = preemptions_by_pool(con, iso_week_window(WEEK))
    assert (out[0].accelerator_variant, out[0].zone, out[0].preemptions) == ("v5p-8", "us-east5-a", 2)
    counts = {(r.accelerator_variant, r.zone): r.preemptions for r in out}
    assert counts == {("v5p-8", "us-east5-a"): 2, ("v6e-4", "us-east1-d"): 1}


def test_chip_hours_split_and_per_host_division(con):
    # alice: a v5p preemptible-8 slice, single host, 1h -> 8 preemptible chip-hours.
    _run_hour(con, "/alice/job/t0", _wid("v5p", "preemptible", 8, 0, "a1"))
    # bob: a v4 reserved-16 slice over TWO hosts, each 1h. slice=16 chips / 2 hosts
    # -> 8 chips/host, 1h each -> 16 reserved chip-hours. ondemand folds to reserved.
    _run_hour(con, "/bob/job/t0", _wid("v4", "reserved", 16, 0, "b1"))
    _run_hour(con, "/bob/job/t0", _wid("v4", "reserved", 16, 1, "b1"))
    by_user = {u.user: u for u in chip_hours_by_user(_chip(con))}
    assert by_user["alice"].preemptible == 8.0
    assert by_user["alice"].reserved == 0.0
    assert by_user["bob"].reserved == 16.0
    assert by_user["bob"].preemptible == 0.0

    by_pool = {(c.capacity, c.generation): c.chip_hours for c in chip_hours_by_capacity_gen(con)}
    assert by_pool == {("preemptible", "v5p"): 8.0, ("reserved", "v4"): 16.0}


def test_ondemand_counts_as_reserved(con):
    _run_hour(con, "/carol/job/t0", _wid("v6e", "ondemand", 4, 0, "c1"))
    (u,) = chip_hours_by_user(_chip(con))
    assert u.reserved == 4.0 and u.preemptible == 0.0


def test_chip_hours_by_region_parses_and_merges_truncated_zone(con):
    _run_hour(con, "/alice/job/t0", _wid("v5p", "preemptible", 8, 0, "a1", zone="us-east5-a"))
    _run_hour(con, "/bob/job/t0", _wid("v4", "reserved", 8, 0, "b1", zone="us-central1-b"))
    # Two europe rows whose worker_id zone was truncated to different lengths;
    # both are really europe-west4 and must merge into one row.
    _run_hour(con, "/carol/job/t0", _wid("v6e", "preemptible", 8, 0, "c1", zone="europe-west4-a"))
    _run_hour(con, "/dave/job/t0", _wid("v5e", "preemptible", 8, 0, "d1", zone="europe-wes"))
    by_region = {r.region: r for r in chip_hours_by_region(_chip(con))}
    assert by_region["us-east5"].preemptible == 8.0 and by_region["us-east5"].reserved == 0.0
    assert by_region["us-central1"].reserved == 8.0
    assert by_region["europe-west4"].preemptible == 16.0  # truncated + full merged
    assert "europe-wes" not in by_region and "europe-west" not in by_region


def test_top_jobs_group_by_job_and_count_attempts(con):
    # One job, two tasks; task t1 was retried (attempt 0 then 1) -> 3 attempts, 24 chip-h.
    _run_hour(con, "/alice/train/t0", _wid("v5p", "preemptible", 8, 0, "a1"))
    _run_hour(con, "/alice/train/t1", _wid("v5p", "preemptible", 8, 0, "a2"), attempt=0)
    _run_hour(con, "/alice/train/t1", _wid("v5p", "preemptible", 8, 0, "a3"), attempt=1)
    # A smaller job by another user.
    _run_hour(con, "/bob/eval/t0", _wid("v4", "reserved", 4, 0, "b1"))
    jobs = top_jobs(_chip(con))
    assert jobs[0].job == "/alice/train" and jobs[0].user == "alice"
    assert jobs[0].chip_hours == 24.0 and jobs[0].attempts == 3
    assert jobs[0].generation == "v5p"
    assert jobs[1].job == "/bob/eval" and jobs[1].chip_hours == 4.0


def test_retried_attempt_waste_counts_superseded_attempts(con):
    # task t0: attempt 0 (superseded) + attempt 1 (final). 8 preemptible chip-h each.
    _run_hour(con, "/alice/job/t0", _wid("v5p", "preemptible", 8, 0, "a1"), attempt=0)
    _run_hour(con, "/alice/job/t0", _wid("v5p", "preemptible", 8, 0, "a2"), attempt=1)
    # task t1: single reserved attempt, never superseded.
    _run_hour(con, "/bob/job/t1", _wid("v4", "reserved", 8, 0, "b1"), attempt=0)
    waste = {w.capacity: w for w in retried_attempt_waste(_chip(con))}
    assert waste["preemptible"].retried == 8.0 and waste["preemptible"].total == 16.0
    assert waste["preemptible"].pct == 50.0
    assert waste["reserved"].retried == 0.0 and waste["reserved"].total == 8.0


def test_chip_hour_coverage_excludes_non_tpu(con):
    _run_hour(con, "/alice/job/t0", _wid("v5p", "preemptible", 8, 0))  # 1h tpu
    _run_hour(con, "/dave/job/t0", "coreweave-h100-node-7")  # 1h non-parseable
    assert chip_hour_coverage(con, iso_week_window(WEEK)) == pytest.approx(0.5)


def test_empty_namespaces_do_not_crash():
    # The empty-view DDLs (used when a namespace has no segments in the window)
    # must declare the columns the rollups reference so queries return empty.
    c = duckdb.connect()
    for _ns, (view, empty_ddl) in _VIEW_FOR.items():
        c.execute(f"CREATE TABLE {view} AS {empty_ddl}")
    report = build_report(c, WEEK)
    assert report.chip_hours == []
    assert report.by_capacity_gen == []
    assert report.by_region == []
    assert report.top_jobs == []
    assert report.retried_waste == []
    assert report.mfu == []
    assert report.preemptions == []
    assert report.chip_hour_coverage == 0.0
    assert report.total_chip_hours == 0.0
    assert report.retried_pct == 0.0
    assert "# Iris compute — 2026-W29" in render_markdown(report)


def test_render_markdown_has_sections(con):
    _run_hour(con, "/alice/job/t0", _wid("v5p", "preemptible", 8, 0))
    md = render_markdown(build_report(con, WEEK))
    assert "# Iris compute — 2026-W29" in md
    assert "## Headline" in md
    assert "Chip-hours by user" in md
    assert "Top jobs by chip-hours" in md
    assert "Waste: chip-hours redone after preemption or failure" in md
    assert "Chip-hours by region" in md
    assert "Chip-hours by capacity and generation" in md
    assert "Preemption events by pool" in md
    assert "MFU per job" in md
    assert "alice" in md
