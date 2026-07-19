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
    build_report,
    chip_hour_coverage,
    chip_hours_by_capacity_gen,
    chip_hours_by_user,
    iso_week_window,
    mfu_per_job,
    preemptions_by_pool,
    previous_iso_week,
    render_markdown,
)

WEEK = "2026-W29"  # Mon 2026-07-13 .. Mon 2026-07-20 UTC


def _ts(day: int, hour: int = 12) -> dt.datetime:
    return dt.datetime(2026, 7, day, hour, tzinfo=dt.UTC)


def _wid(gen: str, cap: str, chips: int, host: int, tag: str = "aaaa1111") -> str:
    return f"marin-tpu-{gen}-{cap}-{chips}-us-east5-a-20260101-0000-{tag}-worker-{host}"


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


def _run_hour(con, task_id: str, worker_id: str, day: int = 14):
    # One attempt active for exactly one hour on one host (two samples, 1h apart).
    con.execute("INSERT INTO task VALUES (?, 0, ?, ?)", [task_id, worker_id, _ts(day, 10)])
    con.execute("INSERT INTO task VALUES (?, 0, ?, ?)", [task_id, worker_id, _ts(day, 11)])


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
    by_user = {u.user: u for u in chip_hours_by_user(con, iso_week_window(WEEK))}
    assert by_user["alice"].preemptible == 8.0
    assert by_user["alice"].reserved == 0.0
    assert by_user["bob"].reserved == 16.0
    assert by_user["bob"].preemptible == 0.0

    by_pool = {(c.capacity, c.generation): c.chip_hours for c in chip_hours_by_capacity_gen(con, iso_week_window(WEEK))}
    assert by_pool == {("preemptible", "v5p"): 8.0, ("reserved", "v4"): 16.0}


def test_ondemand_counts_as_reserved(con):
    _run_hour(con, "/carol/job/t0", _wid("v6e", "ondemand", 4, 0, "c1"))
    (u,) = chip_hours_by_user(con, iso_week_window(WEEK))
    assert u.reserved == 4.0 and u.preemptible == 0.0


def test_chip_hour_coverage_excludes_non_tpu(con):
    _run_hour(con, "/alice/job/t0", _wid("v5p", "preemptible", 8, 0))  # 1h tpu
    _run_hour(con, "/dave/job/t0", "coreweave-h100-node-7")  # 1h non-parseable
    assert chip_hour_coverage(con, iso_week_window(WEEK)) == pytest.approx(0.5)


def test_render_markdown_has_sections(con):
    _run_hour(con, "/alice/job/t0", _wid("v5p", "preemptible", 8, 0))
    md = render_markdown(build_report(con, WEEK))
    assert "# Iris compute — 2026-W29" in md
    assert "Chip-hours by user" in md
    assert "Chip-hours by capacity and generation" in md
    assert "Preemption events by pool" in md
    assert "MFU per job" in md
    assert "alice" in md
