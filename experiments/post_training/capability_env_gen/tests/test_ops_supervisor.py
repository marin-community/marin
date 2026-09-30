"""ops/shard_supervisor.py: desired-state init, reconcile decisions, rate limit, suffixes, launch shape."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shard_supervisor as sv  # noqa: E402
from test_ops_conveyor import NOW, FakeRun, iso, job, model_for, submission  # noqa: E402

PILOTS = {"shard-068", "shard-070"}


def spec(base, active=True, conc=None):
    s = sv.sizing_for(base, PILOTS, conc)
    return {"active": active, **s, "note": ""}


def desired(**bases):
    return {"bases": bases}


def parked_run(base, *, accepted_count=10, states=("pending_image_capture",)):
    run = FakeRun(base)
    for i, state in enumerate(states):
        run.item(f"{base}-i{i}", {"state": state, "custom_images": {"state": "pending_capture"}} if state.startswith("pending_image")
                 else {"state": state})
    run.put("run.json", {"accepted_count": accepted_count})
    run.put("submission.json", {"run_name": f"cap-construct-003-{base}-c6", "concurrency": 6, "started_utc": iso(NOW - timedelta(days=1))})
    return run


def killed(base, suffix, when=NOW - timedelta(hours=3)):
    return job(base, suffix, state="killed", submitted=when)


# ---------------------------------------------------------------------------------- decisions


def test_active_parked_base_launches_under_next_free_suffix():
    runs = [parked_run("shard-001")]
    jobs = [killed("shard-001", "c6"), killed("shard-001", "v1"), killed("shard-001", "v3"), job("shard-099", "c6")]
    model = model_for(runs, jobs)
    d = {x.base: x for x in sv.decide(model, jobs, desired(**{"shard-001": spec("shard-001")}), [], now=NOW)}
    assert d["shard-001"].action == "LAUNCH"
    assert d["shard-001"].run_name == "cap-construct-003-shard-001-v4"
    assert "no live job" in d["shard-001"].reason and "queued 9" in d["shard-001"].reason


def test_suffix_skips_names_in_the_ledger_too():
    jobs = [killed("shard-002", "v1")]
    ledger = [{"base": "shard-002", "run_name": "cap-construct-003-shard-002-v2"}]
    assert sv.next_suffix("shard-002", jobs, ledger) == "v3"
    assert sv.next_suffix("shard-003", [], []) == "v1"


def test_live_active_base_is_left_alone():
    run = parked_run("shard-004")
    jobs = [job("shard-004", "c6")]
    model = model_for([run], jobs)
    d = {x.base: x for x in sv.decide(model, jobs, desired(**{"shard-004": spec("shard-004")}), [], now=NOW)}
    assert d["shard-004"].action == "OK" and "c6" in d["shard-004"].reason


def test_all_terminal_base_is_never_launched():
    run = FakeRun("shard-005")
    run.item("a", {"state": "quality_accepted"})
    run.item("b", {"state": "failed", "issues": ["x"], "terminal_disposition": "rejected"})
    run.put("run.json", {"accepted_count": 2})
    jobs = [killed("shard-005", "c6")]
    model = model_for([run], jobs)
    d = {x.base: x for x in sv.decide(model, jobs, desired(**{"shard-005": spec("shard-005")}), [], now=NOW)}
    assert d["shard-005"].action == "SKIP" and "all items terminal" in d["shard-005"].reason


def test_queued_items_keep_a_base_non_terminal():
    run = FakeRun("shard-006")
    run.item("a", {"state": "quality_accepted"})
    run.put("run.json", {"accepted_count": 5})
    jobs = [killed("shard-006", "c6")]
    model = model_for([run], jobs)
    d = {x.base: x for x in sv.decide(model, jobs, desired(**{"shard-006": spec("shard-006")}), [], now=NOW)}
    assert d["shard-006"].action == "LAUNCH"


def test_inactive_live_base_is_reported_and_cancelled_only_when_enforced():
    run = parked_run("shard-070")
    jobs = [job("shard-070", "k2")]
    model = model_for([run], jobs)
    want = desired(**{"shard-070": spec("shard-070", active=False)})
    d = {x.base: x for x in sv.decide(model, jobs, want, [], now=NOW)}
    assert d["shard-070"].action == "REPORT" and "--enforce-cuts" in d["shard-070"].reason
    d = {x.base: x for x in sv.decide(model, jobs, want, [], now=NOW, opts=sv.Options(enforce_cuts=True))}
    assert d["shard-070"].action == "CANCEL" and d["shard-070"].jobs == ["cap-construct-003-shard-070-k2"]


def test_inactive_parked_base_stays_parked():
    run = parked_run("shard-068")
    jobs = [killed("shard-068", "k1")]
    model = model_for([run], jobs)
    d = {x.base: x for x in sv.decide(model, jobs, desired(**{"shard-068": spec("shard-068", active=False)}), [], now=NOW)}
    assert d["shard-068"].action == "OK"


def test_rate_limit_defers_the_overflow_and_prioritises_healthcare():
    bases = ["shard-010", "shard-011", "shard-012", "shard-013", "hc3"]
    runs = [parked_run(b) for b in bases]
    jobs = [killed(b, "c6") for b in bases]
    model = model_for(runs, jobs)
    ledger = [{"at": iso(NOW - timedelta(minutes=4)), "base": "shard-050", "action": "launch", "result": "submitted",
               "run_name": "cap-construct-003-shard-050-v1"}]
    jobs.append(job("shard-050", "v1", submitted=NOW - timedelta(minutes=3)))
    decisions = sv.decide(model, jobs, desired(**{b: spec(b) for b in bases}), ledger, now=NOW,
                          opts=sv.Options(max_launches=3, window_min=10))
    launched = [d.base for d in decisions if d.action == "LAUNCH"]
    deferred = [d.base for d in decisions if d.action == "DEFER"]
    assert len(launched) == 2 and "hc3" in launched, "one of three slots is used by the ledger; hc first"
    assert len(deferred) == 3 and all("rate limit" in d.reason for d in decisions if d.action == "DEFER")


def test_no_progress_guard_stops_a_relaunch_loop():
    run = parked_run("shard-020")
    jobs = [job("shard-020", "v1", state="failed", submitted=NOW - timedelta(hours=1))]
    model = model_for([run], jobs)
    fp = model["bases"]["shard-020"]["fingerprint"]
    ledger = [{"at": iso(NOW - timedelta(hours=1)), "base": "shard-020", "action": "launch", "result": "submitted",
               "run_name": "cap-construct-003-shard-020-v1", "fingerprint": fp, "queued": model["bases"]["shard-020"]["queued"]}]
    want = desired(**{"shard-020": spec("shard-020")})
    d = {x.base: x for x in sv.decide(model, jobs, want, ledger, now=NOW)}
    assert d["shard-020"].action == "SKIP" and "NO_PROGRESS" in d["shard-020"].reason
    d = {x.base: x for x in sv.decide(model, jobs, want, ledger, now=NOW, opts=sv.Options(force_bases=("shard-020",)))}
    assert d["shard-020"].action == "LAUNCH" and d["shard-020"].run_name.endswith("-v2")


def test_recent_launch_not_yet_visible_is_in_flight():
    run = parked_run("shard-021")
    jobs = [killed("shard-021", "c6")]
    model = model_for([run], jobs)
    ledger = [{"at": iso(NOW - timedelta(minutes=3)), "base": "shard-021", "action": "launch", "result": "submitted",
               "run_name": "cap-construct-003-shard-021-v1"}]
    d = {x.base: x for x in sv.decide(model, jobs, desired(**{"shard-021": spec("shard-021")}), ledger, now=NOW)}
    assert d["shard-021"].action == "SKIP" and "IN_FLIGHT" in d["shard-021"].reason


def test_launch_that_never_appears_is_reported():
    ledger = [{"at": iso(NOW - timedelta(minutes=40)), "base": "shard-022", "action": "launch", "result": "submitted",
               "run_name": "cap-construct-003-shard-022-v1"}]
    decisions = sv.decide({"bases": {}}, [job("shard-099", "c6")], desired(), ledger, now=NOW)
    assert any(d.action == "REPORT" and "LAUNCH_NOT_VISIBLE" in d.reason for d in decisions)


def test_concurrency_mismatch_reported_then_relaunched_when_enforced():
    start = NOW - timedelta(hours=2)
    run = FakeRun("shard-068")
    run.item("a", {"state": "pending_build"})
    submission(run, "cap-construct-003-shard-068-c6", start)
    jobs = [job("shard-068", "c6", submitted=start)]
    model = model_for([run], jobs)
    want = desired(**{"shard-068": spec("shard-068")})  # pilot: desired conc 30
    d = {x.base: x for x in sv.decide(model, jobs, want, [], now=NOW)}
    assert d["shard-068"].action == "REPORT" and "CONC_MISMATCH" in d["shard-068"].reason
    d = {x.base: x for x in sv.decide(model, jobs, want, [], now=NOW, opts=sv.Options(enforce_concurrency=True))}
    assert d["shard-068"].action == "RELAUNCH" and d["shard-068"].jobs == ["cap-construct-003-shard-068-c6"]


def test_no_snapshot_base_is_not_resumed():
    jobs = [killed("shard-030", "c6")]
    d = {x.base: x for x in sv.decide({"bases": {}}, jobs, desired(**{"shard-030": spec("shard-030")}), [], now=NOW)}
    assert d["shard-030"].action == "SKIP" and "NO_SNAPSHOT" in d["shard-030"].reason


def test_duplicate_live_jobs_are_reported():
    run = parked_run("shard-031")
    jobs = [job("shard-031", "c6"), job("shard-031", "v1")]
    model = model_for([run], jobs)
    d = {x.base: x for x in sv.decide(model, jobs, desired(**{"shard-031": spec("shard-031")}), [], now=NOW)}
    assert d["shard-031"].action == "REPORT" and "DUPLICATE_LIVE" in d["shard-031"].reason


# ---------------------------------------------------------------------------------- launch shape


def test_launch_script_matches_the_relaunch_scripts():
    shard = sv.launch_script("shard-000", spec("shard-000"), "cap-construct-003-shard-000-v1", Path("/tmp/l.log"))
    assert "cd /Users/k3sc0re/openathena/capability_env_gen\n" in shard
    assert ("export MARIN=/Users/k3sc0re/openathena/marin-construct CAPABILITY_MAX_REPAIR_ROUNDS=4 "
            "CAPABILITY_JUDGE_CALIBRATION_CONCURRENCY=8 CPU=6 MEM=96g DISK=150GB MAX_RETRIES=3 "
            "OMP_MODELS_SOURCE=$PWD/$N/omp-arms/models.armB.yml") in shard
    assert ". $N/silo_env.sh; unset TARGET_CLUSTER RELAY_JOB CAPABILITY_RELAY_ROUTE_JOB" in shard
    assert ("bash scripts/submit.sh --stage synthesize --source build/construct-003/construct-src-003/shard-000.json "
            "--out runs/catalog-full-construct-003/shard-000 --run-name cap-construct-003-shard-000-v1 --resume "
            "--sandbox silo --tier bulk --concurrency 6 > /tmp/l.log 2>&1") in shard
    pilot = sv.launch_script("shard-070", spec("shard-070"), "cap-construct-003-shard-070-v1", Path("/tmp/l.log"))
    assert "CPU=16 MEM=192g DISK=200GB" in pilot and "--concurrency 30" in pilot
    hc = sv.launch_script("hc3", spec("hc3"), "cap-construct-003-hc3-v1", Path("/tmp/l.log"))
    assert ("--source build/construct-003/construct-src-seeded/hc3 --out runs/catalog-full-construct-003/hc3 "
            "--run-name cap-construct-003-hc3-v1 --resume --sandbox silo --tier bulk --concurrency 30 "
            "-- --acceptance-seed acceptance-seeds.json") in hc


def test_execute_launch_clears_stale_lock_and_retries_staging_in_use(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "LAUNCH_DIR", tmp_path / "launch")
    lock = tmp_path / ".submit.lock"
    lock.mkdir()
    monkeypatch.setattr(sv, "LOCK_DIR", lock)
    outcomes = iter(["error: submission staging is in use: lock", "Job submitted: /muchanem/x"])
    calls = []

    class Proc:
        def __init__(self, rc):
            self.returncode, self.stdout, self.stderr = rc, "", ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "pgrep":
            return Proc(1)  # no submit.sh running -> lock is stale
        log = Path(cmd[1]).read_text().split("> ")[-1].split(" 2>&1")[0].strip("'")
        Path(log).write_text(next(outcomes))
        return Proc(0)

    monkeypatch.setattr(sv.subprocess, "run", fake_run)
    d = sv.Decision("shard-001", "LAUNCH", "why", run_name="cap-construct-003-shard-001-v1", spec=spec("shard-001"))
    res = sv.execute_launch(d, lambda m: None, retry_sleep=0)
    assert res["result"] == "submitted"
    assert not lock.exists()
    assert calls.count("bash") == 2


def test_dry_run_render_shows_command_and_reason():
    runs = [parked_run("shard-040")]
    jobs = [killed("shard-040", "c6")]
    model = model_for(runs, jobs)
    want = desired(**{"shard-040": spec("shard-040")})
    decisions = sv.decide(model, jobs, want, [], now=NOW)
    text = sv.render(decisions, want, apply=False, opts=sv.Options(), now=NOW)
    assert "mode=DRY-RUN" in text and "LAUNCH   shard-040" in text
    assert "why: active, no live job" in text and "bash scripts/submit.sh --stage synthesize" in text


# ---------------------------------------------------------------------------------- init


def test_parse_base_list_accepts_the_deferred_formats(tmp_path):
    f = tmp_path / "deferred.txt"
    f.write_text("shard-082 (pilot, conc30; cut 18:12)\n070\n\n# comment\nhc3 hold\nnonsense line\n")
    assert sv.parse_base_list(f) == {"shard-082": "(pilot, conc30; cut 18:12)", "shard-070": "deferred.txt", "hc3": "hold"}


def test_load_pilots_reads_the_healer_list(tmp_path):
    f = tmp_path / "sync_healer.sh"
    f.write_text('MAX=3\nPILOT=" 068 070 074 "\n')
    assert sv.load_pilots(f) == {"shard-068", "shard-070", "shard-074"}


def test_init_desired_snapshots_what_runs_now():
    jobs = [
        job("shard-000", "c6"),                                          # live
        job("shard-068", "k1"),                                          # live pilot but deferred
        killed("shard-070", "k2") | {"reason": "Terminated by user"},    # deliberate cut, deferred
        killed("shard-076", "k1") | {"reason": "Terminated by user"},    # deliberate cut, not listed
        job("shard-077", "c6", state="failed") | {"reason": "max_task_failures"},  # died on its own
        job("hc2", "s1"),
    ]
    out = sv.init_desired(jobs, pilots=PILOTS | {"shard-076"}, deferred={"shard-070": "cut", "shard-068": "cut"},
                          quarantine={"shard-091": "q"}, hold={}, bases=["hc2", "shard-000", "shard-068", "shard-070",
                                                                       "shard-076", "shard-077", "shard-091", "shard-092"])
    b = out["bases"]
    assert b["shard-000"]["active"] and b["shard-000"]["concurrency"] == 6 and b["shard-000"]["cpu"] == 6
    assert b["shard-068"]["active"] and "also listed in stage2-deferred" in b["shard-068"]["note"]
    assert b["shard-068"]["concurrency"] == 30 and b["shard-068"]["mem"] == "192g"
    assert not b["shard-070"]["active"] and not b["shard-076"]["active"] and not b["shard-091"]["active"]
    assert b["shard-077"]["active"] and "not a cut" in b["shard-077"]["note"]
    assert b["shard-092"]["active"] and "never launched" in b["shard-092"]["note"]
    assert b["hc2"]["source"] == "build/construct-003/construct-src-seeded/hc2"
    assert b["hc2"]["extra_args"] == ["--", "--acceptance-seed", "acceptance-seeds.json"] and b["hc2"]["concurrency"] == 30


def test_load_desired_validates(tmp_path):
    f = tmp_path / "desired.json"
    f.write_text('{"bases": {"shard-000": {"active": true, "concurrency": 0, "cpu": 6, "mem": "96g", "disk": "150GB", "source": "x"}}}')
    with pytest.raises(SystemExit, match="positive integer"):
        sv.load_desired(f)


def test_supervisor_reads_live_views_for_concurrency():
    from test_ops_conveyor import LIVE_ROWS, FakeStore, live_payload

    jobs = [job("shard-070", "v1", submitted=NOW - timedelta(hours=4, minutes=1))]
    store = FakeStore([], live={"shard-070": live_payload(LIVE_ROWS, concurrency=6)})
    model = sv.cv.build_model(sv.cv.collect(store, jobs=jobs, fetch_job_list=False), now=NOW)
    d = {x.base: x for x in sv.decide(model, jobs, desired(**{"shard-070": spec("shard-070")}), [], now=NOW)}
    assert d["shard-070"].action == "REPORT" and "CONC_MISMATCH" in d["shard-070"].reason
    parked = sv.decide(model, [], desired(**{"shard-070": spec("shard-070")}), [], now=NOW)
    assert {x.base: x for x in parked}["shard-070"].action == "LAUNCH"


def test_controller_rollout_lifts_the_no_progress_guard_once():
    """Held items only move when their base relaunches on the new controller."""
    run = parked_run("shard-020")
    jobs = [job("shard-020", "v1", state="succeeded", submitted=NOW - timedelta(hours=1))]
    model = model_for([run], jobs)
    base = model["bases"]["shard-020"]
    ledger = [{"at": iso(NOW - timedelta(hours=1)), "base": "shard-020", "action": "launch", "result": "submitted",
               "run_name": "cap-construct-003-shard-020-v1", "fingerprint": base["fingerprint"],
               "queued": base["queued"], "controller": "old"}]
    want = desired(**{"shard-020": spec("shard-020")})
    same = {x.base: x for x in sv.decide(model, jobs, want, ledger, now=NOW, controller="old")}
    assert same["shard-020"].action == "SKIP" and "NO_PROGRESS" in same["shard-020"].reason
    rolled = {x.base: x for x in sv.decide(model, jobs, want, ledger, now=NOW, controller="new")}
    assert rolled["shard-020"].action == "LAUNCH" and rolled["shard-020"].controller == "new"


def test_controller_revision_hashes_the_live_controller(tmp_path):
    (tmp_path / "capability_pipeline").mkdir()
    (tmp_path / "capability_pipeline/synthesis.py").write_text("A = 1\n")
    first = sv.controller_revision(tmp_path)
    (tmp_path / "capability_pipeline/synthesis.py").write_text("A = 2\n")
    assert first and sv.controller_revision(tmp_path) != first
    assert sv.controller_revision(tmp_path / "missing") is None


# ---------------------------------------------------------------------------------- rollout (--upgrade-before)

OLD = NOW - timedelta(hours=10)


def upgrade_fleet():
    runs, jobs = [], []

    def add(base, statuses, *, submitted=OLD, live=True):
        run = FakeRun(base)
        for i, status in enumerate(statuses):
            run.item(f"{base}-i{i}", status)
        run.put("run.json", {"accepted_count": len(statuses)})
        runs.append(run)
        jobs.append(job(base, "c6", submitted=submitted) if live else killed(base, "c6"))

    capture = {"state": "pending_image_capture", "custom_images": {"state": "pending_capture"}}
    held = {"state": "failed", "issues": ["runtime controls failed: Traceback (most recent call last):\n"
                                          "RuntimeError: authored reference g failed in Harbor (VerifierTimeoutError)"]}
    add("hc2", [{"state": "pending_image_publication", "custom_images": {"state": "pending_publication"}}])
    add("shard-010", [capture, capture, capture])
    add("shard-011", [held, held])
    add("shard-012", [{"state": "pending_build"}])
    add("shard-013", [{"state": "pending_build"}], submitted=NOW - timedelta(minutes=20))   # after the cutoff
    add("shard-015", [{"state": "quality_accepted"}])                                      # all terminal
    add("shard-016", [{"state": "pending_build"}], live=False)                             # parked
    return runs, jobs


def test_upgrade_plan_scope_and_order():
    runs, jobs = upgrade_fleet()
    model = model_for(runs, jobs)
    want = desired(**{r.base: spec(r.base) for r in runs})
    opts = sv.Options(upgrade_before=NOW - timedelta(minutes=30), max_launches=20, max_upgrades_per_cycle=2)
    decisions = sv.decide(model, jobs, want, [], now=NOW, opts=opts)
    d = {x.base: x for x in decisions}
    plan = [x for x in decisions if x.position is not None]
    assert [x.base for x in plan] == ["hc2", "shard-010", "shard-011", "shard-012"]
    assert [x.action for x in plan] == ["UPGRADE", "UPGRADE", "DEFER", "DEFER"]
    assert "upgrade cap 2/cycle" in plan[2].reason
    assert d["shard-010"].replaces == "cap-construct-003-shard-010-c6" and d["shard-010"].run_name.endswith("-v1")
    assert d["shard-010"].unblock == {"total": 3, "capture": 3}
    assert d["shard-011"].unblock == {"total": 2, "runtime_infra": 2, "held": 2}
    assert d["hc2"].spec["concurrency"] == 30 and d["hc2"].spec["extra_args"][-1] == "acceptance-seeds.json"
    assert d["shard-013"].action == "OK", "a job submitted after the cutoff is untouched"
    assert d["shard-015"].action == "OK" and "terminal" in d["shard-015"].reason
    assert d["shard-016"].action == "LAUNCH", "no live job: the normal LAUNCH path"
    text = sv.render(decisions, want, apply=False, opts=opts, now=NOW)
    assert text.index("#1   UPGRADE hc2") < text.index("#2   UPGRADE shard-010") < text.index("#3   later   shard-011")
    assert "cancel cap-construct-003-shard-010-c6 -> cap-construct-003-shard-010-v1" in text


def test_upgrade_shares_the_launch_budget_and_guards_the_healer_and_replacements():
    runs, jobs = upgrade_fleet()
    jobs.append(job("shard-020", "c6", submitted=NOW - timedelta(minutes=4)))          # mid-heal?
    heal = FakeRun("shard-020")
    heal.item("x", {"state": "pending_build"})
    heal.put("run.json", {"accepted_count": 1})
    runs.append(heal)
    model = model_for(runs, jobs)
    want = desired(**{r.base: spec(r.base) for r in runs})
    ledger = [
        {"at": iso(NOW - timedelta(minutes=5)), "base": "shard-099", "action": "launch", "result": "submitted", "run_name": "x-v1"},
        {"at": iso(NOW - timedelta(hours=2)), "base": "shard-012", "action": "upgrade", "result": "submitted",
         "run_name": "cap-construct-003-shard-012-c6"},  # the live job is itself a replacement
    ]
    opts = sv.Options(upgrade_before=NOW + timedelta(hours=1), max_launches=3, window_min=10)
    d = {x.base: x for x in sv.decide(model, jobs, want, ledger, now=NOW, opts=opts)}
    assert d["shard-016"].action == "LAUNCH", "parked bases take the budget first"
    assert d["hc2"].action == "UPGRADE" and d["shard-010"].action == "DEFER" and "rate limit" in d["shard-010"].reason
    assert d["shard-020"].action == "SKIP" and "UPGRADE_HEAL_GUARD" in d["shard-020"].reason
    assert d["shard-012"].action == "OK" and "replacement" in d["shard-012"].reason
    assert d["shard-013"].action == "DEFER" and d["shard-013"].position is not None, "before a future cutoff it qualifies"


def test_execute_upgrade_waits_for_the_old_job_to_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    states = iter(["running", "unknown: list failed", "killed"])
    launched, clock = [], iter(range(0, 1000, 10))
    d = sv.Decision("shard-010", "UPGRADE", "why", run_name="cap-construct-003-shard-010-v1", spec=spec("shard-010"),
                    replaces="cap-construct-003-shard-010-c6")
    res = sv.execute_upgrade(d, lambda m: None, cancel=lambda job: {"result": "cancelled", "detail": ""},
                             preflight=lambda dec, log: {"result": "ok"},
                             launch=lambda dec, log: launched.append(dec.run_name) or {"result": "submitted", "detail": ""},
                             state_of=lambda job: next(states), settle_s=180, sleep=lambda s: None, clock=lambda: next(clock))
    assert res["result"] == "submitted" and res["old_state"] == "killed" and launched == ["cap-construct-003-shard-010-v1"]
    assert "upgrade-cancel" in (tmp_path / "ledger.jsonl").read_text()


def test_execute_upgrade_never_submits_while_the_old_job_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    launched, clock = [], iter(range(0, 10_000, 30))
    d = sv.Decision("shard-011", "UPGRADE", "why", run_name="cap-construct-003-shard-011-v1", spec=spec("shard-011"),
                    replaces="cap-construct-003-shard-011-c6")
    res = sv.execute_upgrade(d, lambda m: None, cancel=lambda job: {"result": "cancelled", "detail": ""},
                             preflight=lambda dec, log: {"result": "ok"},
                             launch=lambda dec, log: launched.append(1) or {"result": "submitted"},
                             state_of=lambda job: "running", settle_s=180, sleep=lambda s: None, clock=lambda: next(clock))
    assert res["result"] == "cancel_not_settled" and launched == []
    failed = sv.execute_upgrade(d, lambda m: None, cancel=lambda job: {"result": "failed", "detail": "boom"},
                                preflight=lambda dec, log: {"result": "ok"},
                                launch=lambda dec, log: launched.append(1), state_of=lambda job: "killed")
    assert failed["result"] == "cancel_failed" and launched == []


def test_execute_upgrade_preflights_before_cancelling(tmp_path, monkeypatch):
    # 2026-09-30: the health preflight failed only AFTER the old job was cancelled, leaving
    # three bases with no job.  A failing replacement preflight must cancel nothing.
    monkeypatch.setattr(sv, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    cancelled, launched = [], []
    d = sv.Decision("shard-012", "UPGRADE", "why", run_name="cap-construct-003-shard-012-v2", spec=spec("shard-012"),
                    replaces="cap-construct-003-shard-012-v1")
    res = sv.execute_upgrade(d, lambda m: None,
                             preflight=lambda dec, log: {"result": "preflight_failed", "detail": "health preflight failed"},
                             cancel=lambda job: cancelled.append(job) or {"result": "cancelled"},
                             launch=lambda dec, log: launched.append(1), state_of=lambda job: "killed")
    assert res["result"] == "preflight_failed" and "nothing cancelled" in res["detail"]
    assert cancelled == [] and launched == []


def test_dry_run_flag_precedes_synthesis_arguments():
    hc = sv.sizing_for("hc1", set())
    cmd = sv.submit_command("hc1", hc, "cap-construct-003-hc1-v2", dry_run=True)
    assert cmd.index("--dry-run") < cmd.index("--"), "after '--' it would reach synthesis, not submit.sh"
    assert "--dry-run" not in sv.submit_command("hc1", hc, "cap-construct-003-hc1-v2")


def test_a_past_cutoff_upgrades_replacements_from_an_earlier_wave():
    runs, jobs = upgrade_fleet()
    model = model_for(runs, jobs)
    want = desired(**{r.base: spec(r.base) for r in runs})
    ledger = [{"at": iso(NOW - timedelta(hours=2)), "base": "shard-012", "action": "upgrade", "result": "submitted",
               "run_name": "cap-construct-003-shard-012-c6"}]
    opts = sv.Options(upgrade_before=NOW - timedelta(minutes=30), max_launches=20, max_upgrades_per_cycle=20)
    d = {x.base: x for x in sv.decide(model, jobs, want, ledger, now=NOW, opts=opts)}
    assert d["shard-012"].action == "UPGRADE", "an earlier wave's replacement predates a new past cutoff: old code"
