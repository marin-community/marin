"""ops/conveyor.py: manifest indexing, stage derivation (today's and tomorrow's status shapes),
PARKED/STUCK/STALE detection, throughput, reason classes and the edge-triggered watcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
import conveyor as cv  # noqa: E402

NOW = datetime(2026, 9, 29, 20, 0, tzinfo=UTC)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeRun:
    """One run base: a manifest plus content-addressed objects with LastModified times."""

    def __init__(self, base: str, created: datetime = NOW - timedelta(minutes=3)):
        self.base = base
        self.created = created
        self.files: dict[str, str] = {}
        self.objects: dict[str, tuple[bytes, datetime]] = {}

    def put(self, rel: str, value, written: datetime = NOW - timedelta(hours=1)) -> str:
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        digest = sha(data)
        self.files[rel] = digest
        self.objects.setdefault(digest, (data, written))
        return digest

    def item(self, name: str, status: dict | None = None, *, written=None, started=None, sessions=None, active_op=None):
        self.put(f"items/{name}/contract/accepted.json", {"proposal": name}, started or NOW - timedelta(days=1))
        if status is not None:
            status = {"key": name, **status}  # real statuses carry their key, so contents differ per item
            self.put(f"items/{name}/status.json", status, written or NOW - timedelta(hours=1))
        for session, (state, when) in (sessions or {}).items():
            self.put(f"items/{name}/sessions/{session}/status.json", {"session": session, "status": state}, when)
        if active_op is not None:
            self.put(f"items/{name}/controller/active-operation.json", active_op, NOW - timedelta(hours=1))

    def manifest(self, compact: bool = False) -> bytes:
        body = {"created_utc": iso(self.created), "files": self.files, "final": False, "omitted": {}, "snapshot_id": "x"}
        return json.dumps(body, separators=(",", ":")).encode() if compact else json.dumps(body, indent=2).encode()


class FakeStore:
    workers = 4
    manifest_workers = 2

    def __init__(self, runs: list[FakeRun], missing: tuple[str, ...] = (), live: dict | None = None):
        self.runs = {r.base: r for r in runs}
        self.missing = set(missing)
        self.live = dict(live or {})  # base -> conveyor.json payload
        self.stats = Counter()
        self.manifest_reads: list[str] = []

    def live_view(self, base):
        if base not in self.live:
            return None
        return {"etag": "e", "last_modified": None, "conveyor": self.live[base], "report": {"state": "needs_continuation"}}

    def manifest_head(self, base):
        run = self.runs.get(base)
        return {"etag": "m", "last_modified": iso(run.created)} if run else None

    def list_bases(self):
        bases = sorted(set(self.runs) | self.missing | set(self.live))
        if not bases:
            raise cv.ReadError("empty listing")
        return bases

    def manifest_index(self, base):
        self.manifest_reads.append(base)
        if base in self.missing:
            return None
        return cv.index_manifest(self.runs[base].manifest())

    def get_object(self, base, digest, want_data=True):
        data, written = self.runs[base].objects[digest]
        return {"data": json.loads(data) if want_data else None, "last_modified": iso(written), "has_data": want_data}

    def raw_manifest(self, base):
        return self.runs[base].manifest()

    def get_bytes(self, base, digest):
        return self.runs[base].objects[digest][0]


def job(base: str, suffix: str, state: str = "running", submitted: datetime = NOW - timedelta(hours=10)) -> dict:
    return {"name": f"/muchanem/cap-construct-003-{base}-{suffix}", "state": state, "submitted": submitted.isoformat(),
            "reason": "", "base": base, "suffix": suffix}


def model_for(runs, jobs, *, now=NOW, history=None, thresholds=None, missing=()):
    store = FakeStore(runs, missing)
    collected = cv.collect(store, jobs=jobs, fetch_job_list=False)
    return cv.build_model(collected, now=now, history=history, thresholds=thresholds)


def submission(run, name, started):
    run.put("submission.json", {"run_name": name, "concurrency": 6, "started_utc": iso(started)})
    run.put("run.json", {"accepted_count": len({k.split("/")[1] for k in run.files if k.startswith("items/")})})


# ---------------------------------------------------------------------------------- indexing


@pytest.mark.parametrize("compact", [False, True])
def test_index_manifest_extracts_only_conveyor_members(compact):
    run = FakeRun("shard-001")
    run.item("a", {"state": "failed", "issues": ["x"]},
             sessions={"s1": ("complete", NOW), "s2": ("in_progress", NOW)},
             active_op={"state": "active", "attempt": 2, "operation": "construction_repair"})
    run.put("items/a/harbor/task.toml", b"toml")
    run.put("items/a/workspace/nested/status.json", {"not": "an item status"})
    run.put("items/task/evidence/x.json", {"stray": True})
    run.put("repair-budget/a/attempt-2/reservation.json", {"r": 2})
    run.put("repair-budget/a/attempt-1/reservation.json", {"r": 1})
    run.put("quality/a/attempt-3/result.json", {"q": 3})
    run.put("quality/a/attempt-3/input/workspace/foo", b"q")
    run.put("validated/a/task.toml", b"v")
    run.put("run.json", {"accepted_count": 3})
    run.put("submission.json", {"run_name": "cap-construct-003-shard-001-c6"})
    idx = cv.index_manifest(run.manifest(compact=compact))
    assert set(idx["items"]) == {"a"}, "the stray items/task dir has no contract and no status"
    a = idx["items"]["a"]
    assert a["status"] == run.files["items/a/status.json"]
    assert a["active_op"] and a["contract"] and a["lowered"] and a["validated"]
    assert set(a["sessions"]) == {"s1", "s2"}
    assert a["repair_reserved"] == 2 and a["quality_attempts"] == 3
    assert set(idx["run_files"]) == {"run.json", "submission.json"}
    assert idx["created_utc"] == iso(run.created) and idx["final"] is False and idx["snapshot_id"] == "x"


# ---------------------------------------------------------------------------------- stages


TRACEBACK = ("runtime controls failed: Traceback (most recent call last):\n  File \"/app/x.py\", line 1, in f\n"
             "    raise RuntimeError(\nRuntimeError: authored reference positive-golden failed in Harbor "
             "(VerifierTimeoutError); inspect runtime-trials/oracle-positive-golden/result.json")


@pytest.mark.parametrize("status, stage, resume", [
    (None, "build", "progress"),
    ({"state": "pending_build", "issues": ["one or more declared build sessions need continuation"]}, "build", "progress"),
    ({"state": "pending_build", "issues": ["missing final bundle files: workspace/task/x"]}, "bundle", "progress"),
    ({"state": "pending_judge_policy"}, "judge_policy", "progress"),
    ({"state": "pending_image_review", "custom_images": {"state": "pending_review", "reason": "image_plan_review_incomplete"}}, "image_review", "progress"),
    ({"state": "pending_image_capture", "custom_images": {"state": "pending_capture", "reason": "capture_command_failed"}}, "image_capture", "progress"),
    ({"state": "pending_image_infrastructure", "custom_images": {"state": "pending_infrastructure"}}, "image_capture", "progress"),
    ({"state": "pending_image_publication", "custom_images": {"state": "pending_publication"}}, "image_publication", "progress"),
    ({"state": "pending_image_cold_pull", "custom_images": {"state": "pending_cold_pull"}}, "image_cold_pull", "progress"),
    ({"state": "pending_image_migration", "custom_images": {"state": "pending_migration"}}, "image_migration", "progress"),
    ({"state": "pending_build_acceptance", "repair_budget": {"used": 1, "max": 4}}, "build_acceptance", "repair"),
    ({"state": "lowered"}, "lowering", "progress"),
    ({"state": "pending_judge_calibration"}, "judge_calibration", "frozen"),
    ({"state": "pending_judge_calibration", "judge_calibration_failure": {"artifact": "x"}}, "judge_calibration", "maybe"),
    ({"state": "failed", "issues": [TRACEBACK], "runtime_evidence": "x"}, "runtime_controls", "frozen"),
    ({"state": "failed", "issues": ["gold: reward is below reward_min"], "repair_budget": {"used": 1, "max": 4}}, "runtime_controls", "repair"),
    ({"state": "failed", "issues": ["invalid task bundle: control c has invalid reward_min"], "repair_budget": {"used": 1, "max": 4}}, "bundle", "repair"),
    ({"state": "failed", "issues": ["invalid task bundle: x"], "custom_images": {"state": "repairable", "reason": "image_sensitive_paths_present"}}, "image_capture", "repair"),
    ({"state": "failed", "issues": ["TaskCompendium validation/lowering failed: y"]}, "lowering", "repair"),
    ({"state": "failed", "issues": ["invalid task bundle: x"], "repair_budget": {"used": 4, "max": 4}, "terminal_disposition": "rejected"}, "bundle", "terminal"),
    ({"state": "failed", "issues": ["repair agent did not finish"], "quality_review": {"state": "repair"}}, "quality_review", "frozen"),
    ({"state": "pending_solver_adjudication", "issues": ["gold: reward is below reward_min"]}, "adjudication", "frozen"),
    ({"state": "pending_adversary_retry"}, "adjudication", "frozen"),
    ({"state": "pending_attack_adjudication"}, "adjudication", "maybe"),
    ({"state": "validated"}, "repeated_diagnostics", "progress"),
    ({"state": "pending_repeated_diagnostics"}, "repeated_diagnostics", "frozen"),
    ({"state": "pending_quality_review", "quality_review": {"state": "repair"}, "repair_budget": {"used": 0, "max": 4}}, "quality_review", "repair"),
    ({"state": "pending_quality_review", "issues": ["semantic review is bound to another snapshot"]}, "quality_review", "frozen"),
    ({"state": "pending_readmission"}, "readmission", "progress"),
    ({"state": "quality_accepted"}, "accepted", "terminal"),
    ({"state": "pending_runtime_infrastructure", "failure_stage": "runtime_infrastructure"}, "runtime_controls", "progress"),
])
def test_stage_and_resume_class_for_todays_statuses(status, stage, resume):
    assert cv.state_stage(status) == stage
    assert cv.resume_class(status)[0] == resume


def test_unknown_state_is_flagged_not_guessed():
    run = FakeRun("shard-002")
    run.item("odd", {"state": "pending_teleport"})
    submission(run, "cap-construct-003-shard-002-c6", NOW - timedelta(hours=2))
    model = model_for([run], [job("shard-002", "c6")])
    rec = next(r for r in model["items"] if r["item"] == "odd")
    assert rec["stage"] == "unknown" and rec["disposition"] == "unknown"
    assert model["anomaly_counts"]["UNKNOWN_STATE"] == 1


def test_tomorrows_status_shape_takes_precedence():
    status = {
        "state": "failed", "issues": ["something new: exploded"], "failure_stage": "judge_calibration",
        "state_since": iso(NOW - timedelta(hours=3)), "updated_at": iso(NOW - timedelta(minutes=5)),
        "transitions": [{"state": "pending_build", "at": iso(NOW - timedelta(hours=9)), "reason": "start"},
                        {"state": "failed", "at": iso(NOW - timedelta(hours=3)), "reason": "calibration"}],
        "wait": {"kind": "provider", "attempts": 3, "first_seen": iso(NOW - timedelta(hours=3)),
                 "next_attempt_at": iso(NOW + timedelta(minutes=20)), "deadline": iso(NOW + timedelta(hours=6))},
        "activity": {"step": "judge_calibration", "since": iso(NOW - timedelta(minutes=30))},
        "repair_budget": {"used": 2, "max": 4},
    }
    rec = cv.classify_item("hc1", "x", {"status": status, "status_written": iso(NOW - timedelta(hours=1))},
                           live=True, job_start=NOW - timedelta(hours=5))
    assert rec["failure_stage"] == "judge_calibration"
    assert rec["since"] == iso(NOW - timedelta(hours=3)) and rec["since_source"] == "state_since"
    assert rec["activity"] == "judge_calibration" and rec["activity_since"] == iso(NOW - timedelta(minutes=30))
    assert rec["wait_attempts"] == 3 and rec["next_attempt_at"] == iso(NOW + timedelta(minutes=20))
    assert rec["repairs_used"] == 2 and rec["repairs_max"] == 4
    assert rec["pass_state"] == "touched"


def test_transitions_drive_throughput_when_present():
    items = [
        {"transitions": [{"state": "quality_accepted", "at": iso(NOW - timedelta(minutes=30))}], "disposition": "accepted", "since": None},
        {"transitions": [{"state": "quality_accepted", "at": iso(NOW - timedelta(hours=3))}], "disposition": "accepted", "since": None},
        {"transitions": [{"state": "failed", "at": iso(NOW - timedelta(minutes=10))}], "disposition": "held", "since": None},
    ]
    tp = cv.compute_throughput(items, now=NOW, history=None)
    assert tp["source"] == "transitions"
    assert (tp["accepted_1h"], tp["accepted_6h"], tp["failed_1h"]) == (1, 2, 1)


# ---------------------------------------------------------------------------------- anomalies


def test_parked_when_nonterminal_items_and_no_live_job():
    parked = FakeRun("shard-010")
    parked.item("a", {"state": "pending_image_capture", "custom_images": {"state": "pending_capture"}})
    parked.item("b", {"state": "quality_accepted"})
    parked.put("run.json", {"accepted_count": 5})
    done = FakeRun("shard-011")
    done.item("c", {"state": "quality_accepted"})
    done.put("run.json", {"accepted_count": 1})
    model = model_for([parked, done], [job("shard-099", "c6")])
    kinds = {(a["kind"], a["run"]) for a in model["anomalies"]}
    assert ("PARKED", "shard-010") in kinds and ("PARKED", "shard-011") not in kinds
    base = model["bases"]["shard-010"]
    assert base["queued"] == 3 and base["non_terminal"] == 4 and base["counts"]["parked"] == 1
    assert model["bases"]["shard-011"]["all_terminal"] is True


def test_unknown_job_list_never_claims_parked():
    run = FakeRun("shard-012")
    run.item("a", {"state": "pending_build"})
    store = FakeStore([run])
    collected = cv.collect(store, jobs=None, fetch_job_list=False)
    model = cv.build_model(collected, now=NOW)
    assert not model["jobs_known"] and "PARKED" not in model["anomaly_counts"]


def test_stuck_only_for_items_the_current_pass_touched():
    start = NOW - timedelta(hours=10)
    run = FakeRun("shard-020")
    # repair started this pass, 7h ago -> STUCK (repair threshold 5h)
    run.item("repairing", {"state": "failed", "issues": ["invalid task bundle: x"]}, written=start - timedelta(hours=5),
             active_op={"state": "active", "operation": "construction_repair", "attempt": 2,
                        "started_at": iso(NOW - timedelta(hours=7))})
    # repair op left over from a killed job (before this pass) -> pending, not stuck
    run.item("stale-op", {"state": "failed", "issues": ["invalid task bundle: x"]}, written=start - timedelta(hours=5),
             active_op={"state": "active", "operation": "construction_repair", "attempt": 1,
                        "started_at": iso(start - timedelta(hours=2))})
    # returned this pass 8h ago at a progressable state -> waiting for the next pass, not stuck
    run.item("returned", {"state": "pending_image_publication", "custom_images": {"state": "pending_publication"}},
             written=start + timedelta(hours=2))
    # first attempt whose builder session last moved 13h... no: 1h ago -> in progress, fine
    run.item("building", None, sessions={"s1": ("complete", start + timedelta(hours=1)), "s2": ("in_progress", NOW - timedelta(hours=1))})
    submission(run, "cap-construct-003-shard-020-v1", start)
    model = model_for([run], [job("shard-020", "v1", submitted=start - timedelta(minutes=2))])
    recs = {r["item"]: r for r in model["items"]}
    assert "STUCK" in recs["repairing"]["anomalies"]
    assert recs["stale-op"]["pass_state"] == "pending" and "STUCK" not in recs["stale-op"]["anomalies"]
    assert recs["returned"]["disposition"] == "waiting" and "STUCK" not in recs["returned"]["anomalies"]
    assert recs["building"]["activity"].startswith("build s2") and recs["building"]["pass_state"] == "touched"
    assert "STUCK" not in recs["building"]["anomalies"]
    assert model["anomaly_counts"]["STUCK"] == 1
    # a tighter threshold makes the builder stuck too
    tight = model_for([run], [job("shard-020", "v1", submitted=start - timedelta(minutes=2))], thresholds={"build": 0.5})
    assert tight["anomaly_counts"]["STUCK"] == 2


def test_stale_snapshot_and_no_snapshot():
    stale = FakeRun("shard-030", created=NOW - timedelta(minutes=45))
    stale.item("a", {"state": "pending_build"})
    fresh_job = FakeRun("shard-031", created=NOW - timedelta(minutes=45))
    fresh_job.item("a", {"state": "pending_build"})
    jobs = [job("shard-030", "c6", submitted=NOW - timedelta(hours=3)),
            job("shard-031", "v2", submitted=NOW - timedelta(minutes=5)),
            job("shard-032", "v1", submitted=NOW - timedelta(hours=1))]
    model = model_for([stale, fresh_job], jobs, missing=("shard-032",))
    kinds = {(a["kind"], a["run"]) for a in model["anomalies"]}
    assert ("STALE_SNAPSHOT", "shard-030") in kinds
    assert ("STALE_SNAPSHOT", "shard-031") not in kinds, "a freshly (re)submitted job is still restoring"
    assert ("NO_SNAPSHOT", "shard-032") in kinds


def test_stalled_live_run_without_any_change():
    start = NOW - timedelta(hours=6)
    run = FakeRun("hc9")
    run.item("a", {"state": "pending_build"}, written=start - timedelta(hours=3))
    submission(run, "cap-construct-003-hc9-s1", start)
    model = model_for([run], [job("hc9", "s1", submitted=start)])
    assert model["anomaly_counts"].get("STALLED") == 1


# ---------------------------------------------------------------------------------- reasons


def test_reason_normalisation_collapses_names_paths_and_numbers():
    assert cv.issue_head(TRACEBACK).startswith("runtime controls failed: RuntimeError: authored reference")
    cls = cv.normalize_reason(TRACEBACK)
    assert cls == cv.normalize_reason(TRACEBACK.replace("positive-golden", "pos-dry-run"))
    assert cls.startswith("runtime: RuntimeError: authored reference failed in Harbor (VerifierTimeoutError)")
    assert cv.normalize_reason("gold-a: reward is below reward_min") == cv.normalize_reason("pc1: reward is below reward_min")
    a = cv.normalize_reason("invalid task bundle: missing file /tmp/a/b/c.json at line 12")
    b = cv.normalize_reason("invalid task bundle: missing file /tmp/x/y/z.json at line 99")
    assert a == b and a.startswith("bundle: ")


# ---------------------------------------------------------------------------------- throughput/history


def test_history_seeds_from_status_write_time_then_diffs(tmp_path):
    run = FakeRun("shard-040")
    run.item("new", {"state": "quality_accepted"}, written=NOW - timedelta(minutes=20))
    run.item("resumed", {"state": "quality_accepted", "acceptance_resumed": {"source": "status"}}, written=NOW - timedelta(minutes=10))
    run.item("old", {"state": "quality_accepted"}, written=NOW - timedelta(hours=20))
    run.item("later", {"state": "pending_quality_review"}, written=NOW - timedelta(hours=2))
    run.put("run.json", {"accepted_count": 4})
    jobs = [job("shard-040", "c6")]
    history = cv.History(tmp_path)
    model = model_for([run], jobs, history=history)
    assert model["throughput"]["accepted_1h"] == 1, "resumed and old acceptances are not throughput"
    # next cycle: 'later' is accepted
    run.item("later", {"state": "quality_accepted"}, written=NOW + timedelta(minutes=4))
    history2 = cv.History(tmp_path)
    model2 = model_for([run], jobs, now=NOW + timedelta(minutes=5), history=history2)
    assert model2["throughput"]["accepted_1h"] == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sorted(e["key"] for e in events if e["kind"] == "accepted") == ["shard-040/later", "shard-040/new"]


# ---------------------------------------------------------------------------------- jobs


def test_job_name_parsing():
    assert cv.parse_job_name("/muchanem/cap-construct-003-shard-000-c6") == ("shard-000", "c6")
    assert cv.parse_job_name("/muchanem/cap-construct-003-shard-070-c30") == ("shard-070", "c30")
    assert cv.parse_job_name("/muchanem/cap-construct-003-hc2-s1") == ("hc2", "s1")
    assert cv.parse_job_name("/muchanem/cap-construct-003-hc4") == ("hc4", None)
    assert cv.parse_job_name("/muchanem/cap-construct-003-shard-089") == ("shard-089", None)
    text = ("JOB ID  STATE  SUBMITTED  REASON\n"
            "/muchanem/cap-construct-003-shard-053-h1   running  2026-09-29T19:44:35.088000+00:00\n"
            "/muchanem/cap-construct-003-shard-070-k2   killed   2026-09-29T18:56:42.166000+00:00  Terminated by user\n")
    jobs = cv.parse_job_list(text)
    assert [j["base"] for j in jobs] == ["shard-053", "shard-070"]
    assert jobs[1]["reason"] == "Terminated by user"
    assert set(cv.live_jobs_by_base(jobs)) == {"shard-053"}


def test_fetch_jobs_fails_loud_on_empty_or_truncated(monkeypatch):
    class Proc:
        def __init__(self, out):
            self.returncode, self.stdout, self.stderr = 0, out, ""

    monkeypatch.setattr(cv.subprocess, "run", lambda *a, **k: Proc("JOB ID STATE SUBMITTED\n"))
    with pytest.raises(cv.ReadError, match="zero jobs"):
        cv.fetch_jobs()
    rows = "".join(f"/muchanem/cap-construct-003-shard-{i:03d}-c6 running 2026-09-29T00:00:00+00:00\n" for i in range(3))
    monkeypatch.setattr(cv.subprocess, "run", lambda *a, **k: Proc(rows))
    with pytest.raises(cv.ReadError, match="truncated"):
        cv.fetch_jobs(limit=3)


def test_connect_without_keys_fails_loud(monkeypatch):
    monkeypatch.delenv("CW_KEY_ID", raising=False)
    monkeypatch.delenv("CW_KEY_SECRET", raising=False)
    with pytest.raises(cv.ReadError, match="not exported"):
        cv.S3Store().connect()


def test_empty_listing_fails_loud():
    with pytest.raises(cv.ReadError):
        cv.collect(FakeStore([]), jobs=[], fetch_job_list=False)


# ---------------------------------------------------------------------------------- watcher


def test_watcher_is_edge_triggered():
    run = FakeRun("shard-050")
    run.item("a", {"state": "pending_build"})
    run.put("run.json", {"accepted_count": 1})
    live = [job("shard-050", "c6")]
    watcher = cv.Watcher(stage_step=1, zero_minutes=60, status_every_s=900)
    first = watcher.diff(model_for([run], live), NOW, 1.0, wall=0)
    assert any(line.startswith("STATUS") for line in first)
    assert any(line.startswith("THROUGHPUT_ZERO") for line in first), "low-side alarm fires first"
    assert not any(line.startswith("PARKED") for line in first)
    second = watcher.diff(model_for([run], []), NOW, 1.0, wall=10)
    assert [line.split()[0] for line in second if line.startswith("PARKED")] == ["PARKED"]
    assert not any(line.startswith(("STATUS", "THROUGHPUT_ZERO")) for line in second), "no repeats inside the heartbeat period"
    third = watcher.diff(model_for([run], []), NOW, 1.0, wall=20)
    assert third == []
    run.item("a", {"state": "quality_accepted"}, written=NOW - timedelta(minutes=1))
    fourth = watcher.diff(model_for([run], live), NOW, 1.0, wall=30)
    kinds = {line.split()[0] for line in fourth}
    assert {"UNPARKED", "ACCEPTED", "STAGE"} <= kinds
    run.item("a", {"state": "failed", "issues": ["runtime controls failed: boom"]}, written=NOW)
    fifth = watcher.diff(model_for([run], live), NOW, 1.0, wall=40)
    lost = [line for line in fifth if line.startswith("UNACCEPTED")]
    assert lost and "shard-050/a" in lost[0] and "failed" in lost[0]


def test_history_records_lost_acceptances(tmp_path):
    run = FakeRun("shard-041")
    run.item("x", {"state": "quality_accepted"}, written=NOW - timedelta(hours=2))
    run.put("run.json", {"accepted_count": 1})
    jobs = [job("shard-041", "c6")]
    model_for([run], jobs, history=cv.History(tmp_path))
    run.item("x", {"state": "failed", "issues": ["runtime controls failed: y"]}, written=NOW + timedelta(minutes=2))
    model = model_for([run], jobs, now=NOW + timedelta(minutes=5), history=cv.History(tmp_path))
    assert model["throughput"]["unaccepted_1h"] == 1


def test_watch_loop_reports_errors_and_gaps_never_silence():
    ticks = iter([0, 5, 400, 400, 405, 900])  # second cycle starts 400s after the first (interval 100)
    lines: list[str] = []
    calls = {"n": 0}

    def collect_fn(store, bases=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise cv.ReadError("listing failed")
        run = FakeRun("shard-060")
        run.item("a", {"state": "pending_build"})
        return cv.collect(FakeStore([run]), jobs=[job("shard-060", "c6")], fetch_job_list=False)

    args = argparse.Namespace(stage_step=5, zero_minutes=60, status_every=900, no_state=True, interval=100,
                              key_refresh_min=50, run_list=None, verbose=False, max_cycles=2, manifest_every=6)
    cv.watch(args, {}, store=object(), collect_fn=collect_fn, emit=lines.append,
             clock=lambda: next(ticks), sleep=lambda s: None)
    kinds = [line.split()[0] for line in lines]
    assert kinds[0] == "ERROR" and "GAP" in kinds and "STATUS" in kinds


# ---------------------------------------------------------------------------------- _live/conveyor.json


def epoch(dt: datetime) -> float:
    return dt.timestamp()


def live_payload(rows: dict, *, updated=NOW - timedelta(minutes=2), started=NOW - timedelta(hours=4), concurrency=6):
    return {"schema_version": "capability-conveyor-v1", "started_at": epoch(started), "concurrency": concurrency,
            "updated_at": epoch(updated), "counts": {}, "items": rows}


def row(item, state, klass, *, issue=None, failure_stage=None, activity=None, wait=None, since=NOW - timedelta(hours=1)):
    return {"item": item, "state": state, "state_since": epoch(since) if state else None, "class": klass,
            "activity": activity, "wait": wait, "failure_stage": failure_stage, "issue": issue, "calls": 1,
            "updated_at": epoch(since) if state else None}


LIVE_ROWS = {
    "cap.q:1": row("q", None, "queued"),
    "cap.r:1": row("r", None, "running", activity={"step": "builder:s2", "since": epoch(NOW - timedelta(minutes=30))}),
    "cap.w:1": row("w", "pending_image_publication", "waiting",
                   wait={"kind": "image_publication", "attempts": 4, "next_attempt_at": epoch(NOW + timedelta(minutes=2))},
                   activity={"step": "waiting", "kind": "image_publication", "since": epoch(NOW - timedelta(minutes=5))}),
    "cap.a:1": row("a", "quality_accepted", "terminal"),
    "cap.f:1": row("f", "failed", "terminal", failure_stage="task_bundle",
                   issue="invalid task bundle: control neg has invalid reward_min"),
    "cap.s:1": row("s", "failed", "running", issue="gold: reward is below reward_min",
                   activity={"step": "repair", "since": epoch(NOW - timedelta(hours=9))}),
}


def test_live_view_replaces_the_manifest():
    store = FakeStore([], live={"shard-070": live_payload(LIVE_ROWS)})
    jobs = [job("shard-070", "v1", submitted=NOW - timedelta(hours=4, minutes=1))]
    model = cv.build_model(cv.collect(store, jobs=jobs, fetch_job_list=False), now=NOW)
    assert store.manifest_reads == [], "a base with a current _live view never touches its manifest"
    base = model["bases"]["shard-070"]
    assert base["source"] == "live" and base["report_state"] == "needs_continuation"
    assert base["accepted_count"] == 6 and base["queued"] == 1 and base["started"] == 5
    assert base["job_start"] == iso(NOW - timedelta(hours=4))
    assert base["submission"]["run_name"] == "cap-construct-003-shard-070-v1" and base["submission"]["concurrency"] == 6
    recs = {r["item"]: r for r in model["items"]}
    assert recs["r"]["disposition"] == "active" and recs["r"]["pass_state"] == "touched"
    assert recs["r"]["stage"] == "build" and recs["r"]["activity"] == "builder:s2"
    assert recs["w"]["disposition"] == "waiting" and recs["w"]["wait_attempts"] == 4 and recs["w"]["stage"] == "image_publication"
    assert recs["a"]["disposition"] == "accepted"
    assert recs["f"]["disposition"] == "rejected" and recs["f"]["failure_stage"] == "bundle" and recs["f"]["failure_stage_raw"] == "task_bundle"
    assert recs["f"]["reason_class"] == "bundle: control has invalid reward_min"
    assert "STUCK" in recs["s"]["anomalies"], "running for 9h on one step"
    assert model["failure_histogram"]["bundle"] == {"bundle: control has invalid reward_min": 1}


def test_live_view_without_a_job_is_parked_and_stale_views_fall_back():
    run = FakeRun("shard-071", created=NOW - timedelta(minutes=5))
    run.item("old", {"state": "pending_build"})
    run.put("run.json", {"accepted_count": 3})
    # _live written 3h before the manifest: an old-code job ran afterwards -> read the manifest.
    stale = live_payload(LIVE_ROWS, updated=NOW - timedelta(hours=3))
    store = FakeStore([run], live={"shard-071": stale, "shard-072": live_payload(LIVE_ROWS)})
    model = cv.build_model(cv.collect(store, jobs=[job("shard-099", "c6")], fetch_job_list=False), now=NOW)
    assert store.manifest_reads == ["shard-071"]
    assert model["bases"]["shard-071"]["source"] == "manifest"
    assert "newer than the _live view" in model["bases"]["shard-071"]["live_fallback"]
    assert model["bases"]["shard-072"]["source"] == "live"
    parked = {a["run"] for a in model["anomalies"] if a["kind"] == "PARKED"}
    assert parked == {"shard-071", "shard-072"}
    # a live job submitted after the _live view was written: that job has not reported yet
    store2 = FakeStore([run], live={"shard-071": live_payload(LIVE_ROWS)})
    cv.collect(store2, jobs=[job("shard-071", "v2", submitted=NOW)], fetch_job_list=False)
    assert store2.manifest_reads == ["shard-071"]


def test_cheap_mode_reuses_manifest_snapshots_but_always_reads_live_views():
    run = FakeRun("shard-080", created=NOW - timedelta(minutes=10))
    run.item("x", {"state": "pending_build"})
    live = {"shard-081": live_payload(LIVE_ROWS)}
    jobs = [job("shard-080", "c6", submitted=NOW - timedelta(hours=5)), job("shard-081", "v1", submitted=NOW - timedelta(hours=5))]
    store = FakeStore([run], live=live)
    first = cv.collect(store, jobs=jobs, fetch_job_list=False)
    first.collected_at = NOW  # the fake clock
    assert store.manifest_reads == ["shard-080"]
    store.live["shard-081"] = live_payload({**LIVE_ROWS, "cap.new:1": row("new", "quality_accepted", "terminal")})
    second = cv.collect(store, jobs=jobs, fetch_job_list=False, previous=first, refresh_manifests=False)
    assert store.manifest_reads == ["shard-080"], "no manifest read on a cheap cycle"
    assert second.snapshots["shard-080"]["source"] == "reused"
    assert "new" in second.snapshots["shard-081"]["items"]
    assert second.timings["manifest_reused"] == 1 and second.timings["live_views"] == 1
    # the reused snapshot is judged as of when it was read: no STALE flapping 30 minutes later
    later = cv.build_model(second, now=NOW + timedelta(minutes=30))
    stale = {a["run"] for a in later["anomalies"] if a["kind"] == "STALE_SNAPSHOT"}
    assert "shard-080" not in stale and "shard-081" in stale, "live views are judged against now"
    cv.collect(store, jobs=jobs, fetch_job_list=False, previous=second, refresh_manifests=True)
    assert store.manifest_reads == ["shard-080", "shard-080"]


def test_watch_refreshes_manifests_every_nth_cycle():
    seen = []

    def collect_fn(store, bases=None, previous=None, refresh_manifests=True):
        seen.append(refresh_manifests)
        return cv.collect(FakeStore([], live={"shard-090": live_payload(LIVE_ROWS)}), jobs=[job("shard-090", "v1")],
                          fetch_job_list=False)

    ticks = iter(range(0, 10_000, 10))
    args = argparse.Namespace(stage_step=5, zero_minutes=60, status_every=900, no_state=True, interval=100,
                              key_refresh_min=50, run_list=None, verbose=False, max_cycles=7, manifest_every=3)
    cv.watch(args, {}, store=object(), collect_fn=collect_fn, emit=lambda line: None,
             clock=lambda: next(ticks), sleep=lambda s: None)
    assert seen == [True, False, False, True, False, False, True]
