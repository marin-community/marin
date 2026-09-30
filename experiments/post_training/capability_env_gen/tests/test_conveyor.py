"""The synthesis conveyor: every item moves forward or ends terminal in one job."""

import json
import re
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_synthesis import FakeAgent, FakeToolchain, accepted

from capability_pipeline import conveyor, synthesis
from capability_pipeline.conveyor import (
    ACTIVE,
    REPAIRABLE_STATES,
    STATE_TABLE,
    TERMINAL,
    WAITING,
    ConveyorConfig,
    ConveyorEntry,
    ConveyorScheduler,
    classify_result,
    mark_activity,
    stamp_status,
    write_status,
)
from capability_pipeline.inference import digest
from capability_pipeline.synthesis import (
    SynthesisError,
    _fresh_construction_repair_allowed,
    _safe_name,
    _synthesize_attempt,
    synthesize_one,
)

PROJECT = Path(__file__).resolve().parents[1]


def _config(**env):
    """Fast budgets for tests; any knob can be overridden by env-style name."""
    values = {
        "CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS": "0.05",
        **{f"CAPABILITY_WAIT_{kind.upper()}_BACKOFF_SECONDS": "0.01" for kind in conveyor.DEFAULT_WAIT_BUDGETS},
    }
    values.update({key: str(value) for key, value in env.items()})
    return ConveyorConfig.from_env(values)


def _never_repair(result, entry):
    return classify_result(result, repair_actionable=lambda value: False)


class Script:
    """An item call that returns scripted results and records timing."""

    def __init__(self, root, name, results, *, delay=0.0, active=None):
        self.item_root = root / "items" / name
        self.results = list(results)
        self.delay = delay
        self.calls = []
        self.active = active

    def __call__(self):
        if self.active is not None:
            self.active.enter()
        try:
            self.calls.append(time.monotonic())
            if self.delay:
                time.sleep(self.delay)
            value = self.results.pop(0) if len(self.results) > 1 else self.results[0]
            if isinstance(value, BaseException):
                raise value
            result = {"item_root": str(self.item_root), "sessions": [], "issues": [], **value}
            write_status(self.item_root, result)
            return result
        finally:
            if self.active is not None:
                self.active.leave()


class Active:
    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def enter(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def leave(self):
        with self.lock:
            self.now -= 1


def _entry(script, key):
    return ConveyorEntry(key=key, item_root=script.item_root, proposal_hash="0" * 64, call=script)


def _run(root, scripts, *, concurrency=1, config=None, classify=_never_repair, retry=lambda error: True):
    entries = [_entry(script, f"cap.test:{index}") for index, script in enumerate(scripts, 1)]
    return ConveyorScheduler(
        root, entries, concurrency, classify=classify, retry_exception=retry, config=config or _config()
    ).run()


def _status(script):
    return json.loads((script.item_root / "status.json").read_text())


# -- classification --------------------------------------------------------


def _returned_states():
    """Every state literal _synthesize_attempt / synthesize_one can return."""
    source = (PROJECT / "capability_pipeline/synthesis.py").read_text()
    body = source[source.index("def _synthesize_attempt("):source.index("def _repair_feedback(")]
    body += source[source.index("def synthesize_one("):source.index("def _conveyor_classification(")]
    states = set(re.findall(r'state="([a-z_]+)"', body))
    states |= set(re.findall(r'"state": "([a-z_]+)"', body))
    states |= set(re.findall(r'result\["state"\] = "([a-z_]+)"', body))
    # Internal checkpoints overwritten before any return; the image prefix.
    states = {state for state in states if not state.endswith("_")} - {"lowered", "validated"}
    # The image controller and the step modules whose results it returns as-is.
    for name in ("image_pipeline.py", "image_capture_control.py", "publication_exchange.py"):
        path = PROJECT / "capability_pipeline" / name
        if not path.is_file():
            continue
        for image_state in set(re.findall(r'"state": "([a-z_]+)"', path.read_text())):
            if image_state.startswith("pending_"):
                states.add("pending_image_" + image_state.removeprefix("pending_"))
    return states


def test_every_returned_state_is_classified():
    states = _returned_states()
    assert {"quality_accepted", "failed", "pending_readmission", "pending_image_publication"} <= states
    assert states - set(STATE_TABLE) == set()


def test_repair_set_is_unchanged_and_owned_by_the_table():
    assert REPAIRABLE_STATES == {
        "failed",
        "pending_build_acceptance",
        "pending_judge_calibration",
        "pending_solver_adjudication",
        "pending_attack_adjudication",
        "runtime_controls_passed_pending_adversary",
        "pending_quality_review",
        "pending_repeated_diagnostics",
    }
    assert set(conveyor.DEFAULT_WAIT_BUDGETS) >= {
        stage for klass, stage in STATE_TABLE.values() if klass == WAITING and stage != "builder"
    } | {"builder_process", "builder_continuation", "controller_exception"}


@pytest.mark.parametrize(
    ("result", "actionable", "expected"),
    [
        ({"state": "quality_accepted"}, False, (TERMINAL, "accepted")),
        ({"state": "pending_readmission"}, False, (TERMINAL, "readmission")),
        ({"state": "pending_image_capture"}, False, (WAITING, "image_capture")),
        ({"state": "failed", "issues": ["runtime controls failed: x"]}, False, (TERMINAL, "runtime_controls")),
        ({"state": "failed", "issues": ["invalid task bundle: x"], "repair_budget": {"exhausted": False}}, True, (ACTIVE, "task_bundle")),
        (
            {"state": "failed", "issues": ["invalid task bundle: x"], "repair_budget": {"exhausted": True}, "terminal_disposition": "rejected"},
            True,
            (TERMINAL, "task_bundle"),
        ),
        ({"state": "pending_quality_review", "repair_budget": {"exhausted": True}}, True, (TERMINAL, "quality_review")),
        (
            {"state": "pending_build", "sessions": [{"session": "s1", "status": "continuation_required", "continuation_reason": "agent_process_failed"}]},
            False,
            (WAITING, "builder_process"),
        ),
        (
            {"state": "pending_build", "sessions": [{"session": "s1", "status": "continuation_required", "continuation_reason": "no_workspace_progress"}, {"session": "s2", "status": "blocked_dependency"}]},
            False,
            (WAITING, "builder_continuation"),
        ),
        (
            {"state": "pending_build", "sessions": [{"session": "s1", "status": "complete"}], "issues": ["missing final bundle files: workspace/task/controls.json"]},
            False,
            (TERMINAL, "build"),
        ),
    ],
)
def test_classification(result, actionable, expected):
    classification = classify_result(result, repair_actionable=lambda value: actionable)
    assert (classification.klass, classification.stage) == expected


def test_holds_are_terminal_with_their_reason():
    classification = classify_result({"state": "pending_solver_adjudication"}, repair_actionable=lambda value: False)
    assert classification.klass == TERMINAL
    assert classification.reason == "operational_hold:pending_solver_adjudication"


# -- scheduler -------------------------------------------------------------


def test_waiting_item_is_retried_after_backoff_without_holding_a_slot(tmp_path):
    active = Active()
    waiting = Script(
        tmp_path,
        "a",
        [{"state": "pending_image_publication", "custom_images": {"state": "pending_publication"}}, {"state": "quality_accepted"}],
        active=active,
    )
    busy = Script(tmp_path, "b", [{"state": "quality_accepted"}], delay=0.1, active=active)
    config = _config(CAPABILITY_WAIT_IMAGE_PUBLICATION_BACKOFF_SECONDS=0.3)
    outcome = _run(tmp_path, [waiting, busy], concurrency=1, config=config)
    assert [result["state"] for result in outcome.results] == ["quality_accepted", "quality_accepted"]
    assert active.peak == 1
    assert len(waiting.calls) == 2 and len(busy.calls) == 1
    # The only slot served the other item while the waiting one was parked.
    assert waiting.calls[0] < busy.calls[0] < waiting.calls[1]
    assert waiting.calls[1] - waiting.calls[0] >= 0.29
    status = _status(waiting)
    assert status["state"] == "quality_accepted"
    assert "wait" not in status
    assert [row["state"] for row in status["transitions"]] == ["pending_image_publication", "quality_accepted"]
    board = json.loads((tmp_path / "conveyor.json").read_text())
    assert board["schema_version"] == "capability-conveyor-v1"
    assert board["counts"] == {"terminal": 2}
    assert board["items"]["cap.test:1"]["state"] == "quality_accepted"


def test_due_retries_run_before_fresh_items(tmp_path):
    order = []

    def recorded(name, results, delay=0.0):
        script = Script(tmp_path, name, results, delay=delay)

        def call():
            order.append(name)
            return script()

        return ConveyorEntry(f"cap.test:{name}", script.item_root, "0" * 64, call)

    entries = [
        recorded("poll", [{"state": "pending_image_publication"}, {"state": "quality_accepted"}]),
        recorded("slow", [{"state": "quality_accepted"}], delay=0.2),
        recorded("late", [{"state": "quality_accepted"}]),
    ]
    ConveyorScheduler(
        tmp_path, entries, 1, classify=_never_repair, retry_exception=lambda error: True, config=_config()
    ).run()
    # The due poll jumps ahead of the fresh item: waits never starve behind new work.
    assert order == ["poll", "slow", "poll", "late"]


def test_a_due_retry_with_every_slot_busy_does_not_spin(tmp_path):
    ticks = []

    def clock():
        ticks.append(1)
        return time.time()

    waiting = Script(tmp_path, "waiting", [{"state": "pending_image_publication"}, {"state": "quality_accepted"}])
    busy = Script(tmp_path, "busy", [{"state": "quality_accepted"}], delay=0.4)
    entries = [_entry(waiting, "cap.test:1"), _entry(busy, "cap.test:2")]
    ConveyorScheduler(
        tmp_path, entries, 1, classify=_never_repair, retry_exception=lambda error: True,
        config=_config(CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS=1), clock=clock,
    ).run()
    assert len(waiting.calls) == 2
    assert len(ticks) < 100  # a spin would read the clock thousands of times


def test_wait_budget_exhaustion_is_terminal_failed(tmp_path):
    stuck = Script(tmp_path, "stuck", [{"state": "pending_image_capture", "issues": ["capture_command_failed"]}])
    outcome = _run(tmp_path, [stuck], config=_config(CAPABILITY_WAIT_IMAGE_CAPTURE_ATTEMPTS=3))
    result = outcome.results[0]
    assert len(stuck.calls) == 3
    assert result["state"] == "failed"
    assert result["issues"] == ["wait_budget_exhausted:pending_image_capture", "capture_command_failed"]
    assert result["failure_stage"] == "image_capture"
    assert result["wait_exhausted"]["cause"] == "attempts"
    assert result["wait_exhausted"]["attempts"] == 3
    assert result["conveyor"]["class"] == "terminal"
    assert _status(stuck) == result
    # A resumed item stays terminal: no repair path, no re-entry.
    assert not _fresh_construction_repair_allowed(result)
    assert classify_result(result, repair_actionable=lambda value: False).klass == TERMINAL


def test_wait_deadline_exhaustion(tmp_path):
    stuck = Script(tmp_path, "stuck", [{"state": "pending_image_publication"}])
    config = _config(
        CAPABILITY_WAIT_IMAGE_PUBLICATION_SECONDS=0.1,
        CAPABILITY_WAIT_IMAGE_PUBLICATION_BACKOFF_SECONDS=0.04,
    )
    result = _run(tmp_path, [stuck], config=config).results[0]
    assert result["state"] == "failed"
    assert result["wait_exhausted"]["cause"] == "deadline"
    assert 2 <= len(stuck.calls) <= 5


def test_relaunch_carries_the_wait_budget(tmp_path):
    item_root = tmp_path / "items" / "carried"
    now = time.time()
    write_status(
        item_root,
        {
            "state": "pending_image_capture",
            "issues": [],
            "wait": {"kind": "image_capture", "attempts": 5, "first_seen": now - 60},
        },
        intermediate=False,
    )
    carried = Script(tmp_path, "carried", [{"state": "pending_image_capture"}])
    result = _run(tmp_path, [carried], config=_config(CAPABILITY_WAIT_IMAGE_CAPTURE_ATTEMPTS=6)).results[0]
    assert len(carried.calls) == 1
    assert result["state"] == "failed"
    assert result["wait_exhausted"]["first_seen"] == pytest.approx(now - 60)


def test_image_hints_are_honoured(tmp_path):
    item_root = tmp_path / "items" / "hinted"
    calls, observed = [], {}

    def call():
        calls.append(time.monotonic())
        if len(calls) == 1:
            images = {"state": "pending_publication", "backoff_seconds": 0.2, "packet_sha": "abc", "attempts": 4}
        else:
            observed["wait"] = json.loads((item_root / "status.json").read_text())["wait"]
            images = {"state": "pending_publication", "retryable": False, "reason": "publisher rejected packet"}
        return write_status(item_root, {"state": "pending_image_publication", "custom_images": images, "item_root": str(item_root)})

    outcome = ConveyorScheduler(
        tmp_path, [ConveyorEntry("cap.test:1", item_root, "0" * 64, call)], 1,
        classify=_never_repair, retry_exception=lambda error: True, config=_config(),
    ).run()
    result = outcome.results[0]
    assert calls[1] - calls[0] >= 0.19
    assert observed["wait"]["attempts"] == 4
    assert observed["wait"]["packet_sha"] == "abc"
    assert observed["wait"]["backoff_seconds"] == 0.2
    # retryable: False is a hold for this job that keeps the state, so a
    # relaunch re-enters it (the capture harness contract: blocked_until job_relaunch).
    assert len(calls) == 2
    assert result["state"] == "pending_image_publication"
    assert result["failure_stage"] == "image_publication"
    assert result["conveyor"] == {"class": "terminal", "reason": "not_retryable:publisher rejected packet", "calls": 2}
    assert result["wait_hold"]["blocked_until"] == "job_relaunch"
    assert result["wait_hold"]["attempts"] == 5
    assert "wait" not in result
    assert conveyor.summarize([result])["wait_hold_items"] == 1


def test_a_step_that_owns_its_attempt_budget_is_not_double_capped(tmp_path):
    item_root = tmp_path / "items" / "multi-role"
    calls = []

    def call():
        calls.append(1)
        n = len(calls)
        if n <= 4:  # two roles, each counting its own attempts 1..2 of 6
            images = {"state": "pending_capture", "retryable": True, "attempts": (n - 1) % 2 + 1, "max_attempts": 6, "backoff_seconds": 0.01}
            return write_status(item_root, {"state": "pending_image_capture", "custom_images": images, "item_root": str(item_root)})
        return write_status(item_root, {"state": "quality_accepted", "item_root": str(item_root)})

    outcome = ConveyorScheduler(
        tmp_path, [ConveyorEntry("cap.test:1", item_root, "0" * 64, call)], 1,
        classify=_never_repair, retry_exception=lambda error: True,
        config=_config(CAPABILITY_WAIT_IMAGE_CAPTURE_ATTEMPTS=2),
    ).run()
    assert len(calls) == 5
    assert outcome.results[0]["state"] == "quality_accepted"


def test_capture_controller_failed_terminal_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "capability_pipeline.image_pipeline.process_image_construction",
        lambda **kwargs: {
            "state": "failed_terminal", "failure_stage": "image_capture", "retryable": False,
            "reason": "image_capture_failed: snapshot:SiloError", "attempts": 6, "max_attempts": 6,
        },
    )
    result = _synthesize_attempt(accepted(), tmp_path, FakeAgent(), FakeToolchain(), None, 30)
    assert result["state"] == "failed"
    assert result["failure_stage"] == "image_capture"
    assert result["issues"] == ["image_capture failed terminally: image_capture_failed: snapshot:SiloError"]


def test_unclassified_state_fails_closed_loudly(tmp_path, capsys):
    odd = Script(tmp_path, "odd", [{"state": "pending_image_teleport", "issues": ["new"]}])
    result = _run(tmp_path, [odd]).results[0]
    assert len(odd.calls) == 1
    assert result["state"] == "failed"
    assert result["issues"] == ["unclassified_state:pending_image_teleport", "new"]
    assert result["failure_stage"] == "unclassified_state"
    assert result["unclassified_state"] == "pending_image_teleport"
    assert '"event":"unclassified_state"' in capsys.readouterr().err


def test_active_repair_reentry_is_immediate_and_capped(tmp_path):
    failing = {"state": "failed", "issues": ["invalid task bundle: x"], "repair_budget": {"exhausted": False}}
    repaired = Script(tmp_path, "repaired", [failing, {"state": "quality_accepted"}])
    looping = Script(tmp_path, "looping", [failing])

    def actionable(result, entry):
        return classify_result(result, repair_actionable=lambda value: True)

    outcome = _run(
        tmp_path,
        [repaired, looping],
        classify=actionable,
        config=_config(CAPABILITY_CONVEYOR_MAX_ACTIVE_REENTRIES=2),
    )
    assert outcome.results[0]["state"] == "quality_accepted"
    assert len(looping.calls) == 3
    assert outcome.results[1]["state"] == "failed"
    assert outcome.results[1]["conveyor"]["reason"] == "active_reentry_cap_exhausted"


def test_exceptions_retry_then_end_terminal_preserving_resting_status(tmp_path):
    flaky = Script(tmp_path, "flaky", [RuntimeError("relay reset"), {"state": "quality_accepted"}])
    broken = Script(tmp_path, "broken", [RuntimeError("disk gone")])
    rejected = Script(tmp_path, "rejected", [SynthesisError("archived evidence changed")])
    write_status(rejected.item_root, {"state": "failed", "issues": ["runtime controls failed: x"]})
    write_status(broken.item_root, {"state": "pending_image_capture", "issues": ["capture_command_failed"]})
    outcome = _run(
        tmp_path,
        [flaky, broken, rejected],
        concurrency=2,
        config=_config(CAPABILITY_WAIT_CONTROLLER_EXCEPTION_ATTEMPTS=2),
        retry=lambda error: not isinstance(error, SynthesisError),
    )
    flaky_result, broken_result, rejected_result = outcome.results
    assert flaky_result["state"] == "quality_accepted" and len(flaky.calls) == 2
    assert len(broken.calls) == 2
    # A transient fault on a waiting item is a hold: the state is kept for a relaunch.
    assert broken_result["state"] == "pending_image_capture"
    assert broken_result["issues"] == ["capture_command_failed"]
    assert broken_result["failure_stage"] == "controller_exception"
    assert broken_result["wait_hold"]["state"] == "pending_image_capture"
    assert broken_result["wait_hold"]["reason"] == "controller_exception:RuntimeError: disk gone"
    assert broken_result["wait_hold"]["blocked_until"] == "job_relaunch"
    assert broken_result["conveyor"] == {"class": "terminal", "reason": "controller_exception_hold:RuntimeError", "calls": 2}
    assert outcome.summary["controller_exception_holds"] == 1
    assert len(rejected.calls) == 1
    assert rejected_result["state"] == "failed"
    assert rejected_result["issues"] == ["runtime controls failed: x"]
    assert rejected_result["controller_exception"]["error_type"] == "SynthesisError"
    assert set(outcome.failures) == {"cap.test:2", "cap.test:3"}
    assert _status(rejected) == rejected_result


def test_accepted_item_is_never_rerun(tmp_path, monkeypatch):
    item = accepted()
    key = f"{item['proposal']['capability_id']}:{item['proposal']['slot']}"
    item_root = tmp_path / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"
    resumed = []

    def resume(item, key, root, item_root, prior, seeds):
        resumed.append(key)
        result = {"key": key, "proposal_hash": item["proposal_hash"], "state": "quality_accepted", "item_root": str(item_root), "issues": []}
        conveyor.atomic_json(item_root / "status.json", result)
        return result

    def rebuild(*args, **kwargs):
        pytest.fail("an accepted item must never re-enter construction")

    monkeypatch.setattr("capability_pipeline.acceptance.resume_accepted", resume)
    monkeypatch.setattr(synthesis, "_synthesize_attempt", rebuild)
    entry = ConveyorEntry(
        key,
        item_root,
        item["proposal_hash"],
        lambda: synthesize_one(item, tmp_path, FakeAgent(), None, None, 30),
    )
    outcome = ConveyorScheduler(
        tmp_path, [entry], 1, classify=synthesis._conveyor_classification, retry_exception=lambda error: True, config=_config()
    ).run()
    assert resumed == [key]
    result = outcome.results[0]
    assert result["state"] == "quality_accepted"
    assert "failure_stage" not in result
    assert result["conveyor"]["class"] == "terminal"


# -- status stamping -------------------------------------------------------


def test_stamping_preserves_state_since_and_carries_transitions_across_resume(tmp_path):
    item_root = tmp_path / "run-1" / "items" / "x"
    write_status(item_root, {"state": "pending_build", "issues": ["one or more declared build sessions need continuation"]}, now=100.0)
    first = write_status(item_root, {"state": "pending_build", "issues": []}, now=150.0)
    assert first["state_since"] == 100.0 and first["updated_at"] == 150.0
    assert first["transitions"] == [
        {"state": "pending_build", "at": 100.0, "reason": "one or more declared build sessions need continuation"}
    ]
    moved = write_status(item_root, {"state": "pending_image_capture", "custom_images": {"reason": "capture_command_failed"}}, now=200.0)
    assert moved["state_since"] == 200.0
    assert moved["transitions"][-1] == {"state": "pending_image_capture", "at": 200.0, "reason": "capture_command_failed"}
    # A relaunch restores the tree under a new root; a fresh attempt result has no history.
    relaunched = tmp_path / "run-2" / "items" / "x"
    shutil.copytree(item_root, relaunched)
    resumed = write_status(relaunched, {"state": "pending_image_capture", "issues": []}, now=300.0)
    assert resumed["state_since"] == 200.0
    assert [row["state"] for row in resumed["transitions"]] == ["pending_build", "pending_image_capture"]
    done = write_status(relaunched, {"state": "quality_accepted", "issues": []}, now=400.0)
    assert [row["at"] for row in done["transitions"]] == [100.0, 200.0, 400.0]


def test_stamping_edge_cases(tmp_path):
    # Legacy status (no history) is seeded on its first stamped write.
    item_root = tmp_path / "items" / "legacy"
    conveyor.atomic_json(item_root / "status.json", {"state": "failed", "issues": ["runtime controls failed: x"]})
    legacy = write_status(item_root, {"state": "failed", "issues": ["runtime controls failed: x"]}, now=10.0)
    assert legacy["transitions"] == [{"state": "failed", "at": 10.0, "reason": "runtime controls failed: x"}]
    # An archive that moved status.json: the result's own history is used.
    archived = {"state": "pending_adversary_retry", "transitions": [{"state": "pending_adversary_retry", "at": 5.0, "reason": None}], "state_since": 5.0}
    stamped = stamp_status(dict(archived), None, now=20.0)
    assert stamped["state_since"] == 5.0 and len(stamped["transitions"]) == 1
    # Intermediate writes keep the wait while the state holds, drop it on a change.
    waiting = tmp_path / "items" / "waiting"
    write_status(waiting, {"state": "pending_image_capture", "wait": {"kind": "image_capture", "attempts": 2}, "conveyor": {"class": "waiting"}}, intermediate=False, now=1.0)
    same = write_status(waiting, {"state": "pending_image_capture", "conveyor": {"class": "terminal"}}, now=2.0)
    assert same["wait"]["attempts"] == 2 and "conveyor" not in same
    changed = write_status(waiting, {"state": "failed", "wait": {"stale": True}}, now=3.0)
    assert "wait" not in changed
    # History is bounded.
    bounded = tmp_path / "items" / "bounded"
    for index in range(conveyor.TRANSITION_LIMIT + 20):
        write_status(bounded, {"state": f"s{index % 2}"}, now=float(index))
    assert len(json.loads((bounded / "status.json").read_text())["transitions"]) == conveyor.TRANSITION_LIMIT


def test_activity_markers_only_inside_a_conveyor(tmp_path):
    item_root = tmp_path / "items" / "a"
    item_root.mkdir(parents=True)
    assert mark_activity(item_root, "builder_session", session="s1") is None
    assert not (item_root / "activity.json").exists()
    seen = {}

    def call():
        mark_activity(item_root, "quality_review", attempt=1)
        seen["activity"] = json.loads((item_root / "activity.json").read_text())
        seen["board"] = json.loads((tmp_path / "conveyor.json").read_text())["items"]["cap.test:1"]
        return write_status(item_root, {"state": "quality_accepted", "item_root": str(item_root)})

    ConveyorScheduler(
        tmp_path, [ConveyorEntry("cap.test:1", item_root, "0" * 64, call)], 1,
        classify=_never_repair, retry_exception=lambda error: True, config=_config(),
    ).run()
    assert seen["activity"]["step"] == "quality_review" and seen["activity"]["attempt"] == 1
    assert seen["board"]["activity"]["step"] == "quality_review"
    assert seen["board"]["class"] == "running"
    assert json.loads((item_root / "activity.json").read_text())["step"] == "terminal"


# -- synthesis integration ------------------------------------------------


def test_failed_terminal_image_step_maps_to_terminal_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "capability_pipeline.image_pipeline.process_image_construction",
        lambda **kwargs: {"state": "failed_terminal", "failure_stage": "capture", "reason": "silo cannot snapshot this image"},
    )
    result = _synthesize_attempt(accepted(), tmp_path, FakeAgent(), FakeToolchain(), None, 30)
    assert result["state"] == "failed"
    assert result["failure_stage"] == "image_capture"
    assert result["issues"] == ["image_capture failed terminally: silo cannot snapshot this image"]
    assert not _fresh_construction_repair_allowed(result)
    classification = classify_result(result, repair_actionable=lambda value: False)
    assert (classification.klass, classification.stage) == (TERMINAL, "image_capture")


def test_image_wait_reentry_does_not_rerun_completed_stages(tmp_path, monkeypatch):
    monkeypatch.delenv("GLM_BASE_URL", raising=False)
    image = "registry.example/task@sha256:" + "a" * 64
    image_calls = []
    lowered = []

    def images(**kwargs):
        image_calls.append(kwargs["item_root"])
        if len(image_calls) == 1:
            return {"state": "pending_publication", "reason": "isolated publisher receipt absent"}
        specification_path = kwargs["item_root"] / "workspace/task/specification.json"
        specification = json.loads(specification_path.read_text())
        specification["requirements"] = {"state": {"image": image}}
        specification_path.write_text(json.dumps(specification))
        return {"state": "ready", "reason": "reviewed_images_migrated"}

    class CountingToolchain(FakeToolchain):
        def validate_and_lower(self, bundle, harbor, timeout):
            lowered.append(bundle)
            return super().validate_and_lower(bundle, harbor, timeout)

    monkeypatch.setattr("capability_pipeline.image_pipeline.process_image_construction", images)
    agent = FakeAgent()
    item = accepted()
    key = f"{item['proposal']['capability_id']}:{item['proposal']['slot']}"
    item_root = tmp_path / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"
    entry = ConveyorEntry(
        key, item_root, item["proposal_hash"],
        lambda: synthesize_one(item, tmp_path, agent, CountingToolchain(), None, 30),
    )
    outcome = ConveyorScheduler(
        tmp_path, [entry], 1, classify=synthesis._conveyor_classification,
        retry_exception=lambda error: True, config=_config(),
    ).run()
    result = outcome.results[0]
    # Builder sessions ran once; the re-entry went straight to the image step.
    assert agent.calls == [("s1", 0), ("s2", 0)]
    assert len(image_calls) == 2
    assert len(lowered) == 1
    assert result["state"] == "controls_passed_pending_rollout"
    assert result["failure_stage"] == "configuration"
    assert [row["state"] for row in result["transitions"]] == [
        "pending_image_publication",
        "controls_passed_pending_rollout",
    ]
    assert json.loads((item_root / "status.json").read_text()) == result


def test_resume_holds_readmission_and_reopens_exhausted_waits_only_on_request(tmp_path, monkeypatch):
    item = accepted()
    key = f"{item['proposal']['capability_id']}:{item['proposal']['slot']}"
    item_root = tmp_path / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"
    attempts = []

    def attempt(*args, **kwargs):
        attempts.append(1)
        return {"state": "pending_image_publication", "issues": [], "item_root": str(item_root)}

    monkeypatch.setattr(synthesis, "_synthesize_attempt", attempt)
    write_status(item_root, {"key": key, "proposal_hash": item["proposal_hash"], "state": "pending_readmission", "issues": ["x"], "item_root": str(item_root)})
    assert synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)["state"] == "pending_readmission"
    exhausted = {
        "key": key, "proposal_hash": item["proposal_hash"], "state": "failed", "item_root": str(item_root),
        "issues": ["wait_budget_exhausted:pending_image_publication"],
        "wait_exhausted": {"kind": "image_publication", "attempts": 200},
    }
    write_status(item_root, dict(exhausted))
    assert synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)["state"] == "failed"
    assert attempts == []
    reopened = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30, retry_exhausted_waits=True)
    assert reopened["state"] == "pending_image_publication"
    assert attempts == [1]


def test_quality_review_writes_a_post_review_gate_receipt(tmp_path, monkeypatch):
    from capability_pipeline.acceptance import GATE_RECEIPT

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "0")
    runner = tmp_path / "runner.py"
    runner.write_text("#!/usr/bin/env python3\n")
    runner.chmod(0o755)
    cases = [
        ("positive", "known_correct", "independent_solver", {"status": "graded", "reward": 1.0}),
        ("malformed", "empty_or_malformed", "authored_adversarial_control", {"status": "extraction_error", "reward": None}),
        ("plausible-wrong", "plausible_wrong", "authored_adversarial_control", {"status": "graded", "reward": 0.0}),
        ("shortcut", "task_specific_shortcut", "authored_adversarial_control", {"status": "graded", "reward": 0.0}),
    ]
    monkeypatch.setattr(
        "capability_pipeline.synthesis._external_controls",
        lambda *args, **kwargs: {
            "cases": [
                {"id": case, "source_author": "builder", "category": category, "control_type": kind, "result": graded}
                for case, category, kind, graded in cases
            ]
        },
    )
    monkeypatch.setattr("capability_pipeline.synthesis._attestation_issues", lambda *args: [])

    def review(item_root, review_root, agent):
        review_root.mkdir(parents=True, exist_ok=True)
        result = {"schema_version": "capability-quality-result-v1", "snapshot_hash": "s", "state": "repair"}
        (review_root / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("capability_pipeline.quality.run_review", review)
    monkeypatch.setattr(
        "capability_pipeline.synthesis._repeated_quality_diagnostics",
        lambda *args: {"state": "pending", "reviewable": True, "extra_files": {}},
    )
    result = synthesize_one(accepted(), tmp_path / "run", FakeAgent(), FakeToolchain(), runner, 30)
    assert result["state"] == "pending_quality_review"
    gate = json.loads((Path(result["quality_review"]["artifact"]).parent / GATE_RECEIPT).read_text())
    assert gate["repeated_diagnostics_state"] == "pending"
    assert gate["reviewable"] is True


def _synthesize_args(tmp_path, accepted_path, **overrides):
    values = {
        "limit": None,
        "accepted": str(accepted_path),
        "out": str(tmp_path / "out"),
        "concurrency": 2,
        "taskcompendium_source": None,
        "omp": str(Path(shutil.which("true") or "/usr/bin/true")),
        "runtime_runner": None,
        "daytona_tools": None,
        "research_overlay": None,
        "model": "glm-orion/glm-5.3",
        "session_time": 60,
        "max_continuations": 1,
        "tier": "bulk",
        "validation_timeout": 30,
        "acceptance_seed": None,
        "retry_infrastructure": False,
        "retry_adversary": False,
        "infrastructure_health_receipt": None,
        "retry_exhausted_waits": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _two_items(tmp_path):
    first = accepted()
    second = accepted()
    second["proposal"]["slot"] = 2
    second["proposal_hash"] = digest(second["proposal"])
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps([first, second]))
    return path


@pytest.mark.parametrize("toolchain_fails", [False, True])
def test_job_exits_zero_once_every_item_is_terminal(tmp_path, monkeypatch, toolchain_fails):
    for name in ("CAPABILITY_DAYTONA_TOOLS", "CAPABILITY_OMP_CONFIG", "RESEARCH_OVERLAY"):
        monkeypatch.delenv(name, raising=False)
    for kind in conveyor.DEFAULT_WAIT_BUDGETS:
        monkeypatch.setenv(f"CAPABILITY_WAIT_{kind.upper()}_BACKOFF_SECONDS", "0.01")
    monkeypatch.setenv("CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS", "0.05")

    def resolve(*args):
        if toolchain_fails:
            raise SynthesisError("pinned toolchain unavailable")
        return object()

    monkeypatch.setattr(synthesis.OfficialToolchain, "resolve", staticmethod(resolve))
    calls = {}

    seen_seeds = {}

    def fake_one(item, root, *args):
        slot = item["proposal"]["slot"]
        calls[slot] = calls.get(slot, 0) + 1
        key = f"{item['proposal']['capability_id']}:{slot}"
        seen_seeds[key] = [seed["key"] for seed in args[8]]
        item_root = root / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"
        base = {"key": key, "proposal_hash": item["proposal_hash"], "item_root": str(item_root), "sessions": []}
        if slot == 1 and calls[slot] == 1:
            result = {**base, "state": "pending_image_publication", "issues": ["pending_publication"]}
        elif slot == 1:
            result = {**base, "state": "quality_accepted", "issues": [], "runtime_validated": True}
        else:
            result = {**base, "state": "failed", "issues": ["runtime controls failed: provider timeout"]}
        return write_status(item_root, result)

    monkeypatch.setattr(synthesis, "synthesize_one", fake_one)
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps([{"key": f"cap.test:{slot}", "state": "quality_accepted"} for slot in (1, 2)]))
    args = _synthesize_args(tmp_path, _two_items(tmp_path), acceptance_seed=str(seeds))
    exit_code = synthesis.synthesize(args)
    # Each item keeps its own seeds (bound at definition time, not the last item's).
    assert seen_seeds == {"cap.test:1": ["cap.test:1"], "cap.test:2": ["cap.test:2"]}
    out = Path(args.out)
    report = json.loads((out / "report.json").read_text())
    tasks = json.loads((out / "tasks.json").read_text())
    assert calls == {1: 2, 2: 1}
    assert exit_code == (2 if toolchain_fails else 0)
    assert report["exit_code"] == exit_code
    assert report["all_terminal"] is True
    assert report["state"] == "needs_continuation"  # unchanged meaning: not every item accepted
    assert report["states"] == {"quality_accepted": 1, "failed": 1}
    assert report["terminal_failures_by_stage"] == {"runtime_controls": 1}
    assert report["completed_items"] == len(tasks) == 2
    for row in tasks:  # coverage requires tasks.json rows to equal status.json
        assert json.loads((Path(row["item_root"]) / "status.json").read_text()) == row
    board = json.loads((out / "conveyor.json").read_text())
    assert board["counts"] == {"terminal": 2}
    run = json.loads((out / "run.json").read_text())
    assert run["conveyor"]["wait_budgets"]["image_publication"]["max_attempts"] == 200


def test_invalid_wait_budget_env_is_rejected_before_work(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPABILITY_WAIT_IMAGE_CAPTURE_ATTEMPTS", "zero")
    with pytest.raises(SynthesisError, match="CAPABILITY_WAIT_IMAGE_CAPTURE_ATTEMPTS"):
        synthesis.synthesize(_synthesize_args(tmp_path, _two_items(tmp_path)))


def test_board_rows_carry_a_one_line_issue_head(tmp_path):
    board = conveyor.ConveyorBoard(tmp_path, [("cap.a:1", "a"), ("cap.b:1", "b"), ("cap.c:1", "c")],
                                   concurrency=1, config=_config())
    traceback = (
        "runtime controls failed: Using CPython 3.12.13\nTraceback (most recent call last):\n"
        '  File "/app/x.py", line 1, in f\nRuntimeError: authored reference gold failed in Harbor (VerifierTimeoutError)'
    )
    board.observe_status("a", {"state": "failed", "issues": [traceback, "second"]})
    board.observe_status("b", {"state": "pending_quality_review", "issues": ["x" * 500]})
    board.observe_status("c", {"state": "quality_accepted", "issues": []})
    items = json.loads((tmp_path / "conveyor.json").read_text())["items"]
    assert items["cap.a:1"]["issue"] == (
        "runtime controls failed: RuntimeError: authored reference gold failed in Harbor (VerifierTimeoutError)"
    )
    assert items["cap.b:1"]["issue"] == "x" * conveyor.ISSUE_LIMIT
    assert items["cap.c:1"]["issue"] is None
    assert conveyor.issue_head({"issues": ["gold: reward is below reward_min"]}) == "gold: reward is below reward_min"
