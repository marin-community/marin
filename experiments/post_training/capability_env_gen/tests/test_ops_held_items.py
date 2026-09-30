"""Held items are not finished items (integration review 2026-09-29).

A conveyor hold -- a step's ``retryable: False`` (harness hold, disabled queue, spent
publisher retries) or a transient controller exception on a waiting/fresh item -- ends
the item for the current job only: its conveyor row is ``terminal`` but its state is
preserved with a ``wait_hold``, and only a relaunch re-enters it.  The ops tools must
count such rows as non-terminal ("held") so the supervisor relaunches their base.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conveyor as cv
import shard_supervisor as sv
from test_ops_conveyor import (
    NOW,
    FakeRun,
    FakeStore,
    epoch,
    job,
    live_payload,
    model_for,
    row,
)


def _hold(state, kind="image_capture"):
    return {"kind": kind, "state": state, "attempts": 1, "held_at": epoch(NOW - timedelta(hours=1)),
            "blocked_until": "job_relaunch"}


def _held_row(item, state, kind="image_capture"):
    value = row(item, state, "terminal", failure_stage=kind)
    value["wait_hold"] = _hold(state, kind)
    return value


ROWS = {
    "cap.a:1": row("a", "quality_accepted", "terminal"),
    "cap.f:1": row("f", "failed", "terminal", failure_stage="task_bundle", issue="invalid task bundle: x"),
    # retryable: False capture harness hold, state preserved
    "cap.h:1": _held_row("h", "pending_image_capture"),
    # a transient controller exception on a fresh item: state absent, kept absent
    "cap.x:1": _held_row("x", None, kind="controller_exception"),
    # a hold that no longer applies (the state moved on): terminal as usual
    "cap.s:1": {**row("s", "failed", "terminal", failure_stage="image_capture"),
                "wait_hold": _hold("pending_image_capture")},
}


def _model(jobs):
    store = FakeStore([], live={"shard-090": live_payload(ROWS)})
    return cv.build_model(cv.collect(store, jobs=jobs, fetch_job_list=False), now=NOW)


def test_held_rows_count_as_non_terminal_and_the_supervisor_relaunches_their_base():
    jobs = [job("shard-090", "v1", state="failed", submitted=NOW - timedelta(hours=5))]
    model = _model(jobs)
    records = {r["item"]: r for r in model["items"]}
    assert records["a"]["disposition"] == "accepted"
    assert records["f"]["disposition"] == "rejected"
    assert records["s"]["disposition"] == "rejected"
    assert records["h"]["disposition"] == "held" and records["h"]["resume"] == "progress"
    assert "wait_hold image_capture until job_relaunch" in records["h"]["resume_reason"]
    assert records["x"]["disposition"] == "held"
    base = model["bases"]["shard-090"]
    assert base["counts"]["held"] == 2 and base["non_terminal"] == 2
    assert base["all_terminal"] is False
    spec = {"active": True, **sv.sizing_for("shard-090", set(), None), "note": ""}
    decisions = {d.base: d for d in sv.decide(model, jobs, {"bases": {"shard-090": spec}}, [], now=NOW)}
    assert decisions["shard-090"].action == "LAUNCH", decisions["shard-090"].reason


def test_without_held_rows_the_same_base_is_all_terminal():
    rows = {key: value for key, value in ROWS.items() if key in ("cap.a:1", "cap.f:1", "cap.s:1")}
    store = FakeStore([], live={"shard-091": live_payload(rows)})
    jobs = [job("shard-091", "v1", state="failed", submitted=NOW - timedelta(hours=5))]
    model = cv.build_model(cv.collect(store, jobs=jobs, fetch_job_list=False), now=NOW)
    assert model["bases"]["shard-091"]["all_terminal"] is True
    spec = {"active": True, **sv.sizing_for("shard-091", set(), None), "note": ""}
    decision = sv.decide(model, jobs, {"bases": {"shard-091": spec}}, [], now=NOW)[0]
    assert decision.action == "SKIP" and "all items terminal" in decision.reason


def test_manifest_status_with_a_hold_in_force_is_held():
    run = FakeRun("shard-092")
    run.item("h", {"state": "pending_image_publication", "wait_hold": _hold("pending_image_publication",
                                                                           "image_publication")})
    run.item("g", {"state": "failed", "issues": ["x"], "terminal_disposition": "rejected",
                   "wait_hold": _hold("pending_image_capture")})
    run.put("run.json", {"accepted_count": 2})
    model = model_for([run], [job("shard-092", "c6", state="killed")])
    records = {r["item"]: r for r in model["items"]}
    assert records["h"]["disposition"] == "held"
    assert records["g"]["disposition"] == "rejected"
    assert cv.held_in_force({"state": "quality_accepted", "wait_hold": _hold("quality_accepted")}) is None
