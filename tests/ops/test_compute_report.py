# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase-1 compute-accounting rollups.

The rollup functions take a DuckDB connection with views named ``telltale`` /
``provisioning`` / ``task``; these tests register synthetic tables under those
names and assert the aggregation, so no finelog or network access is needed.
"""

import datetime as dt

import duckdb
import pytest

from scripts.ops.compute_report import (
    UserHostHours,
    attribution_coverage,
    build_report,
    host_hours_by_user,
    iso_week_window,
    mfu_per_job,
    preemptions_by_pool,
    previous_iso_week,
    render_markdown,
)

WEEK = "2026-W29"  # Mon 2026-07-13 .. Mon 2026-07-20 UTC


def _ts(day: int, hour: int = 12) -> dt.datetime:
    return dt.datetime(2026, 7, day, hour, tzinfo=dt.UTC)


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


def test_iso_week_window():
    w = iso_week_window(WEEK)
    assert w.start == dt.datetime(2026, 7, 13, tzinfo=dt.UTC)
    assert w.end == dt.datetime(2026, 7, 20, tzinfo=dt.UTC)


def test_previous_iso_week():
    # A Thursday in 2026-W30 -> previous complete week is W29.
    assert previous_iso_week(dt.date(2026, 7, 23)) == "2026-W29"


def test_mfu_collapses_worker_replicas(con):
    # One job, two workers forwarding the SAME value each of 3 steps: mean is the
    # value, not double-counted, and step count is 3 (not 6).
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


def test_host_hours_by_user_uses_task_path(con):
    # /alice ran one attempt for 2h on one worker; /bob 1h. task_id path -> user.
    con.execute("INSERT INTO task VALUES ('/alice/job/t0', 0, 'w0', ?)", [_ts(14, 10)])
    con.execute("INSERT INTO task VALUES ('/alice/job/t0', 0, 'w0', ?)", [_ts(14, 12)])
    con.execute("INSERT INTO task VALUES ('/bob/job/t0', 0, 'w0', ?)", [_ts(15, 9)])
    con.execute("INSERT INTO task VALUES ('/bob/job/t0', 0, 'w0', ?)", [_ts(15, 10)])
    out = host_hours_by_user(con, iso_week_window(WEEK))
    by_user = {u.user: u.host_hours for u in out}
    assert by_user == {"alice": 2.0, "bob": 1.0}


def test_attribution_coverage_excludes_shared_users():
    hh = [UserHostHours("alice", 8.0, 1), UserHostHours("root", 2.0, 1)]
    assert attribution_coverage(hh) == pytest.approx(0.8)
    assert attribution_coverage([]) == 0.0


def test_render_markdown_has_sections(con):
    con.execute("INSERT INTO task VALUES ('/alice/job/t0', 0, 'w0', ?)", [_ts(14, 10)])
    con.execute("INSERT INTO task VALUES ('/alice/job/t0', 0, 'w0', ?)", [_ts(14, 12)])
    md = render_markdown(build_report(con, WEEK))
    assert "# Iris compute — 2026-W29" in md
    assert "Host-hours by user" in md
    assert "Preemption events by pool" in md
    assert "MFU per job" in md
    assert "alice" in md
