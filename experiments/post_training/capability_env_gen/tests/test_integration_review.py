"""Integration-review findings (2026-09-29), turned into regression tests.

The review drove ``synthesize()`` end to end with items in every waiting state
and found: transient controller exceptions permanently failing fresh/waiting
items, builder-repairable publisher rejections never reaching repair, and a
relaunch after downtime exhausting a wait on its first poll.  Each test here
asserts the fixed behaviour.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_legacy_acceptance_gate as tla
from test_conveyor import _config, _synthesize_args, _two_items
from test_legacy_acceptance_gate import stub_toolchain  # noqa: F401 - fixture
from test_publication_exchange import _world
from test_synthesis import FakeAgent, FakeToolchain, accepted

from capability_pipeline import conveyor, synthesis
from capability_pipeline import image_capture_control as icc
from capability_pipeline import image_pipeline as pipeline
from capability_pipeline import publication_exchange as exchange
from capability_pipeline.conveyor import ConveyorEntry, ConveyorScheduler, write_status
from capability_pipeline.inference import digest
from capability_pipeline.synthesis import _safe_name

PINNED = "registry.example/task@sha256:" + "a" * 64


class Agent(FakeAgent):
    max_stagnant_attempts = 1
    model = "glm-orion/glm-5.3"


def _item(slot):
    item = accepted()
    item["proposal"]["slot"] = slot
    item["proposal_hash"] = digest(item["proposal"])
    return item


def _root_of(root, item):
    key = f"cap.test:{item['proposal']['slot']}"
    return root / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"


def _pin(item_root):
    spec_path = Path(item_root) / "workspace/task/specification.json"
    spec = json.loads(spec_path.read_text())
    spec["requirements"] = {"state": {"image": PINNED}}
    spec_path.write_text(json.dumps(spec))
    return {"state": "ready", "reason": "reviewed_images_migrated"}


def _synthesis_env(monkeypatch):
    for name in ("CAPABILITY_DAYTONA_TOOLS", "CAPABILITY_OMP_CONFIG", "RESEARCH_OVERLAY", "GLM_BASE_URL",
                 "CAPABILITY_MAX_REPAIR_ROUNDS"):
        monkeypatch.delenv(name, raising=False)
    for kind in conveyor.DEFAULT_WAIT_BUDGETS:
        monkeypatch.setenv(f"CAPABILITY_WAIT_{kind.upper()}_BACKOFF_SECONDS", "0.01")
    monkeypatch.setenv("CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS", "0.05")


# -- end to end: every waiting state, incl. the builder-repairable publisher rejection ----------


def test_e2e_every_waiting_state_and_publisher_rejection_reaches_repair(
    tmp_path, monkeypatch, stub_toolchain  # noqa: F811
):
    _synthesis_env(monkeypatch)
    monkeypatch.setattr(exchange, "_wait_backoff", lambda *a: 0)

    out = tmp_path / "out"
    items = {slot: _item(slot) for slot in range(1, 8)}
    roots = {slot: _root_of(out, item) for slot, item in items.items()}
    by_root = {str(root): slot for slot, root in roots.items()}

    # Real publication worlds (queue + publisher service + import).
    real_construct = pipeline.process_image_construction
    (tmp_path / "w4").mkdir()
    (tmp_path / "w5").mkdir()
    published = _world(tmp_path / "w4", monkeypatch)
    rejected = _world(tmp_path / "w5", monkeypatch, payload=b"private")
    worlds = {4: published, 5: rejected}
    prepared = {
        str(world.item): {"attempt": str(world.attempt), "plan_path": str(world.plan), "workspace": str(world.frozen)}
        for world in worlds.values()
    }
    monkeypatch.setattr(pipeline, "prepare_construction_capture", lambda item_root, *_: prepared[str(item_root)])

    # Legacy accepted-then-demoted item at slot 6 (old review, no receipt).
    legacy_item = items[6]
    monkeypatch.setattr(tla, "_item", lambda env, ver: legacy_item)
    monkeypatch.setattr(tla, "_name", lambda: roots[6].name)
    legacy = tla.Item(out)
    legacy.repeated("attempt-1")
    legacy.grading("attempt-1")
    legacy.reset()
    legacy.review("attempt-1")
    (legacy.path / "status.json").write_text(json.dumps({
        "key": "cap.test:6", "proposal_hash": legacy_item["proposal_hash"], "state": "failed",
        "issues": ["KC2: reward is below reward_min"], "item_root": str(legacy.path)}))

    calls = {slot: 0 for slot in items}
    repairs = []

    def images(**kwargs):
        slot = by_root[str(kwargs["item_root"])]
        calls[slot] += 1
        n = calls[slot]
        if slot == 1:  # capture transient, then captured
            if n == 1:
                record = {"reason": "capture_infrastructure_error", "failure_class": "transient", "attempt": 1}
                value = icc._retry_or_terminal("candidate", record, [record], 1, 6, None, None)
                value["backoff_seconds"] = 0
                return value
            return _pin(kwargs["item_root"])
        if slot == 2:  # capture budget exhausted
            record = {"reason": "capture_infrastructure_error", "stage": "sandbox_create",
                      "error_type": "SiloRateLimitError", "attempt": 6}
            return icc._terminal("candidate", record, 6, 6)
        if slot == 3:  # image-plan review incomplete, then approved
            if n == 1:
                value = pipeline._review_retry("image_plan_review_pending", Path("/x/attempt-a"), [])
                value["backoff_seconds"] = 0
                return value
            return _pin(kwargs["item_root"])
        if slot in worlds:  # real exchange against the local queue + publisher service
            world = worlds[slot]
            result = real_construct(
                item_root=world.item, capture_tools=world.tools, scripts_root=world.scripts,
                agent=SimpleNamespace(model="glm-5.3"), builder_session_ids={"builder-1"},
                command_runner=world.runner, review_base=world.review_base, publication_queue=world.queue)
            if result["state"] == "pending_publication" and result.get("retryable"):
                world.service().serve(poll_seconds=0, drain=True)
            if result["state"] == "ready":
                return _pin(kwargs["item_root"])
            return result
        if slot == 7:  # capture harness hold
            record = {"reason": "capture_transport_failed", "failure_class": "harness", "attempt": 1}
            return icc._harness_hold("candidate", record, 1, 6)
        raise AssertionError(slot)

    monkeypatch.setattr(pipeline, "process_image_construction", images)

    def run_repair(item_root, repair_root, agent, feedback, source=None):
        repairs.append((Path(item_root).name, json.dumps(feedback, default=str)))
        repair_root.mkdir(parents=True, exist_ok=True)
        value = {"state": "blocked", "issues": ["no-op repair"]}
        (repair_root / "result.json").write_text(json.dumps(value))
        return value

    monkeypatch.setattr("capability_pipeline.repair.run_repair", run_repair)

    class UniqueToolchain(FakeToolchain):
        def validate_and_lower(self, bundle, harbor, timeout):
            value = super().validate_and_lower(bundle, harbor, timeout)
            value["id"] = "task-" + Path(bundle).parents[1].name
            return value

    monkeypatch.setattr(synthesis.OfficialToolchain, "resolve", staticmethod(lambda *a: UniqueToolchain()))
    monkeypatch.setattr(synthesis, "OMPAgent", lambda *a, **k: Agent())

    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps([items[slot] for slot in sorted(items)]))
    exit_code = synthesis.synthesize(_synthesize_args(tmp_path, accepted_path, concurrency=3))

    report = json.loads((out / "report.json").read_text())
    statuses = {slot: json.loads((roots[slot] / "status.json").read_text()) for slot in items}
    assert exit_code == 0 and report["all_terminal"] is True
    assert report["controller_exception_holds"] == 0
    for slot in (1, 3, 4):  # progressed past the image step to the next (configuration) hold
        assert statuses[slot]["state"] == "controls_passed_pending_rollout", (slot, statuses[slot]["state"])
    assert statuses[2]["state"] == "failed" and statuses[2]["failure_stage"] == "image_capture"
    assert statuses[6]["state"] == "quality_accepted" and calls[6] == 0
    assert statuses[7]["state"] == "pending_image_capture"
    assert statuses[7]["wait_hold"]["blocked_until"] == "job_relaunch"
    # Finding 2: the builder-repairable publisher rejection is repair input.
    slot5 = statuses[5]
    assert slot5["custom_images"]["builder_repairable"] is True
    assert slot5["custom_images"]["rejection_class"] == "rootfs_review_failed"
    slot5_repairs = [feedback for name, feedback in repairs if name == roots[5].name]
    assert slot5_repairs, "the rejection never reached builder repair"
    assert "rootfs review rejected the captured image" in slot5_repairs[0]
    assert slot5["repairs"] and slot5["repairs"][0]["round"] == 1
    assert [name for name, _ in repairs] == [roots[5].name]  # no other item was sent to repair


def test_builder_repairable_rejection_maps_to_invalid_task_bundle(tmp_path, monkeypatch):
    rejection = {"state": "failed_terminal", "failure_stage": "image_publication", "retryable": False,
                 "reason": "publisher_rejected: rootfs_review_failed: private content",
                 "rejection_class": "rootfs_review_failed", "builder_repairable": True,
                 "issues": ["The trusted publisher's rootfs review rejected the captured image (x)."]}
    monkeypatch.setattr("capability_pipeline.image_pipeline.process_image_construction", lambda **kw: rejection)
    result = synthesis._synthesize_attempt(accepted(), tmp_path, FakeAgent(), FakeToolchain(), None, 30)
    assert result["state"] == "failed"
    assert result["issues"] == ["invalid task bundle: The trusted publisher's rootfs review rejected the captured image (x)."]
    assert synthesis._fresh_construction_repair_allowed(result)
    # A non-repairable rejection stays a terminal image failure.
    monkeypatch.setattr("capability_pipeline.image_pipeline.process_image_construction",
                        lambda **kw: {**rejection, "builder_repairable": False, "issues": ["registry refused"]})
    terminal = synthesis._synthesize_attempt(accepted(), tmp_path / "b", FakeAgent(), FakeToolchain(), None, 30)
    assert terminal["issues"][0].startswith("image_publication failed terminally")
    assert not synthesis._fresh_construction_repair_allowed(terminal)


def test_image_controller_disk_error_is_transient_not_repair_input(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("capability_pipeline.image_pipeline.process_image_construction", broken)
    result = synthesis._synthesize_attempt(accepted(), tmp_path, FakeAgent(), FakeToolchain(), None, 30)
    assert result["state"] == "pending_image_infrastructure"
    assert result["custom_images"]["reason"] == "image_controller_io_error"
    assert not any(str(issue).startswith("invalid task bundle") for issue in result["issues"])
    assert not synthesis._fresh_construction_repair_allowed(result)


# -- finding 1: transient controller exceptions hold, never permanently fail -------------------


def test_transient_controller_exception_holds_a_fresh_item_and_a_relaunch_reenters(tmp_path, monkeypatch):
    """Reviewer repro: the staging leaf vanished (FileNotFoundError from synthesize_one for every
    item).  Before the fix a FRESH item was written as state=failed and never re-entered."""
    item = _item(1)
    item_root = _root_of(tmp_path, item)
    key = "cap.test:1"

    def broken():
        raise FileNotFoundError("/app/capability-pipeline-staging/submissions/x/docs/task_contract.md")

    fault = ConveyorScheduler(tmp_path, [ConveyorEntry(key, item_root, item["proposal_hash"], broken)], 1,
                              classify=synthesis._conveyor_classification,
                              retry_exception=lambda e: not isinstance(e, synthesis.SynthesisError),
                              config=_config()).run()
    status = json.loads((item_root / "status.json").read_text())
    assert status["state"] is None  # absent before: kept absent, not rewritten to failed
    assert status["wait_hold"]["kind"] == "controller_exception"
    assert status["wait_hold"]["blocked_until"] == "job_relaunch"
    assert status["issues"] == [
        "controller_exception:FileNotFoundError: /app/capability-pipeline-staging/submissions/x/docs/task_contract.md"]
    assert status["conveyor"]["reason"] == "controller_exception_hold:FileNotFoundError"
    assert "None" not in status["conveyor"]["reason"]
    assert fault.summary["controller_exception_holds"] == 1
    assert conveyor.summarize(fault.results)["wait_hold_items"] == 1

    attempts = []
    monkeypatch.setattr(synthesis, "_synthesize_attempt",
                        lambda *a, **k: attempts.append(1) or {"state": "quality_accepted"})
    healthy = ConveyorScheduler(tmp_path, [ConveyorEntry(
        key, item_root, item["proposal_hash"],
        lambda: synthesis.synthesize_one(item, tmp_path, Agent(), None, None, 30))], 1,
        classify=synthesis._conveyor_classification, retry_exception=lambda e: True, config=_config()).run()
    assert attempts == [1]
    assert healthy.results[0]["state"] == "quality_accepted"
    assert "wait_hold" not in healthy.results[0]


def test_transient_exception_on_a_waiting_item_keeps_its_wait_budget(tmp_path):
    item_root = tmp_path / "items" / "w"
    now = time.time()
    write_status(item_root, {"state": "pending_image_publication", "issues": ["awaiting_publisher"],
                             "wait": {"kind": "image_publication", "attempts": 7, "first_seen": now - 600,
                                      "last_seen": now - 60}}, intermediate=False)

    def broken():
        raise ConnectionResetError("object store reset")

    ConveyorScheduler(tmp_path, [ConveyorEntry("cap.w:1", item_root, "0" * 64, broken)], 1,
                      classify=synthesis._conveyor_classification, retry_exception=lambda e: True,
                      config=_config()).run()
    status = json.loads((item_root / "status.json").read_text())
    assert status["state"] == "pending_image_publication"
    assert status["wait_hold"]["state"] == "pending_image_publication"
    assert status["wait"]["attempts"] == 7  # carried to the relaunch, not restarted


def test_deterministic_controller_exception_on_a_fresh_item_fails_with_a_named_reason(tmp_path):
    item_root = tmp_path / "items" / "d"

    def broken():
        raise synthesis.SynthesisError("composite verifier requires proposal verification=judge")

    out = ConveyorScheduler(tmp_path, [ConveyorEntry("cap.d:1", item_root, "0" * 64, broken)], 1,
                            classify=synthesis._conveyor_classification,
                            retry_exception=lambda e: not isinstance(e, synthesis.SynthesisError),
                            config=_config()).run()
    result = out.results[0]
    assert result["state"] == "failed"
    assert result["conveyor"]["reason"] == "controller_exception:SynthesisError"
    assert result["issues"][0].startswith("controller_exception:SynthesisError: composite verifier")
    assert out.summary.get("controller_exception_holds", 0) == 0


def test_job_exits_two_when_a_controller_exception_held_an_item(tmp_path, monkeypatch):
    _synthesis_env(monkeypatch)
    monkeypatch.setattr(synthesis.OfficialToolchain, "resolve", staticmethod(lambda *a: object()))

    def fake_one(item, root, *args):
        slot = item["proposal"]["slot"]
        key = f"{item['proposal']['capability_id']}:{slot}"
        item_root = root / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"
        if slot == 1:
            raise OSError("staging leaf vanished")
        return write_status(item_root, {"key": key, "proposal_hash": item["proposal_hash"],
                                        "item_root": str(item_root), "state": "quality_accepted", "issues": []})

    monkeypatch.setattr(synthesis, "synthesize_one", fake_one)
    args = _synthesize_args(tmp_path, _two_items(tmp_path))
    assert synthesis.synthesize(args) == 2
    report = json.loads((Path(args.out) / "report.json").read_text())
    assert report["all_terminal"] is True and report["exit_code"] == 2
    assert report["controller_exception_holds"] == 1 and report["wait_hold_items"] == 1


@pytest.mark.parametrize("value", ["two", "9", "-1"])
def test_invalid_max_repair_rounds_is_rejected_before_any_item_runs(tmp_path, monkeypatch, value):
    _synthesis_env(monkeypatch)
    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", value)
    monkeypatch.setattr(synthesis, "synthesize_one", lambda *a: pytest.fail("no item may run"))
    with pytest.raises(synthesis.SynthesisError, match="CAPABILITY_MAX_REPAIR_ROUNDS"):
        synthesis.synthesize(_synthesize_args(tmp_path, _two_items(tmp_path)))
    assert not (tmp_path / "out" / "conveyor.json").exists()


# -- finding 4: relaunch downtime and a live publisher's backlog are not charged ---------------


def _publication(images_reason, *, age=None):
    images = {"state": "pending_publication", "retryable": True, "reason": images_reason, "backoff_seconds": 0}
    if age is not None:
        images["publisher_heartbeat_age_seconds"] = age
    return {"state": "pending_image_publication", "issues": [images_reason], "custom_images": images}


def _classify(result, entry):
    return conveyor.classify_result(result, repair_actionable=lambda value: False)


def test_relaunch_after_downtime_does_not_exhaust_a_wait_on_its_first_poll(tmp_path):
    """Reviewer repro: first_seen carried across a 6.5 h outage made the first poll of the new
    job terminal (window 6 h) although the publisher was merely slow."""
    item_root = tmp_path / "items" / "a"
    now = time.time()
    write_status(item_root, {"state": "pending_image_publication", "issues": ["awaiting_publisher"],
                             "wait": {"kind": "image_publication", "attempts": 3, "first_seen": now - 7 * 3600,
                                      "last_seen": now - 6.5 * 3600, "deadline": now - 3600}},
                 intermediate=False)
    seen = []

    def poll():
        seen.append(json.loads((item_root / "status.json").read_text()).get("wait"))
        if len(seen) == 1:
            return write_status(item_root, _publication("awaiting_publisher_heartbeat_stale"))
        return write_status(item_root, {"state": "controls_passed_pending_rollout", "issues": []})

    out = ConveyorScheduler(tmp_path, [ConveyorEntry("cap.a:1", item_root, "0" * 64, poll)], 1,
                            classify=_classify, retry_exception=lambda e: True, config=_config()).run()
    assert out.results[0]["state"] == "controls_passed_pending_rollout" and len(seen) == 2
    wait = seen[1]
    assert wait["attempts"] == 4  # attempts carry over
    assert wait["downtime_seconds"] == pytest.approx(6.5 * 3600, abs=60)
    assert wait["deadline"] == pytest.approx(now - 7 * 3600 + 6 * 3600 + 6.5 * 3600, abs=60)


def test_a_carried_wait_without_last_seen_restarts_its_window(tmp_path):
    item_root = tmp_path / "items" / "b"
    now = time.time()
    write_status(item_root, {"state": "pending_image_publication", "issues": ["awaiting_publisher"],
                             "wait": {"kind": "image_publication", "attempts": 3, "first_seen": now - 7 * 3600,
                                      "deadline": now - 3600}}, intermediate=False)
    calls = []

    def poll():
        calls.append(1)
        if len(calls) == 1:
            return write_status(item_root, _publication("awaiting_publisher_heartbeat_stale"))
        return write_status(item_root, {"state": "controls_passed_pending_rollout", "issues": []})

    out = ConveyorScheduler(tmp_path, [ConveyorEntry("cap.b:1", item_root, "0" * 64, poll)], 1,
                            classify=_classify, retry_exception=lambda e: True, config=_config()).run()
    assert out.results[0]["state"] == "controls_passed_pending_rollout" and len(calls) == 2


class _Clock:
    def __init__(self, start=1_900_000_000.0):
        self.now = start

    def __call__(self):
        return self.now


@pytest.mark.parametrize("reason,age,polls", [
    ("awaiting_publisher", 30, 49),  # uncharged up to the 48 h cap (window 2 h + 46 h of backlog)
    ("awaiting_publisher_heartbeat_stale", 900, 3),  # a dead publisher is charged: window 2 h
    ("awaiting_publisher", None, 3),  # no heartbeat reading: no positive evidence, charged
])
def test_waiting_behind_a_live_publisher_is_not_charged_to_the_window(tmp_path, reason, age, polls):
    clock = _Clock()
    item_root = tmp_path / "items" / "p"
    calls = []

    def poll():
        clock.now += 3600  # every poll lands an hour after the previous one
        calls.append(clock.now)
        return write_status(item_root, _publication(reason, age=age), now=clock.now)

    config = _config(CAPABILITY_WAIT_IMAGE_PUBLICATION_SECONDS=7200, CAPABILITY_WAIT_IMAGE_PUBLICATION_ATTEMPTS=500)
    out = ConveyorScheduler(tmp_path, [ConveyorEntry("cap.p:1", item_root, "0" * 64, poll)], 1,
                            classify=_classify, retry_exception=lambda e: True, config=config, clock=clock).run()
    result = out.results[0]
    assert result["state"] == "failed" and result["wait_exhausted"]["cause"] == "deadline"
    assert len(calls) == polls
    assert conveyor.PUBLISHER_BACKLOG_CAP_SECONDS == 48 * 3600


def test_attempts_are_still_charged_behind_a_live_publisher(tmp_path):
    clock = _Clock()
    item_root = tmp_path / "items" / "q"
    calls = []

    def poll():
        clock.now += 60
        calls.append(1)
        return write_status(item_root, _publication("awaiting_publisher", age=10), now=clock.now)

    config = _config(CAPABILITY_WAIT_IMAGE_PUBLICATION_ATTEMPTS=5)
    result = ConveyorScheduler(tmp_path, [ConveyorEntry("cap.q:1", item_root, "0" * 64, poll)], 1,
                               classify=_classify, retry_exception=lambda e: True, config=config,
                               clock=clock).run().results[0]
    assert result["wait_exhausted"]["cause"] == "attempts" and len(calls) == 5


# -- nit: the capture window fits one step timeout plus the backoffs ---------------------------


def test_step_timeout_widens_the_window_of_a_step_budgeted_wait(tmp_path):
    item_root = tmp_path / "items" / "c"
    seen = []

    def poll():
        seen.append(json.loads((item_root / "status.json").read_text()).get("wait") if seen else None)
        if len(seen) == 1:
            return write_status(item_root, {"state": "pending_image_capture", "issues": ["capture_command_failed"],
                                            "custom_images": {"state": "pending_capture", "retryable": True,
                                                              "attempts": 1, "max_attempts": 6,
                                                              "backoff_seconds": 900,
                                                              "step_timeout_seconds": 18_000}})
        return write_status(item_root, {"state": "controls_passed_pending_rollout", "issues": []})

    config = _config(CAPABILITY_WAIT_IMAGE_CAPTURE_BACKOFF_SECONDS=900)
    clock = _Clock()

    def advancing():
        clock.now += 1000  # the backoff elapses between scheduler looks
        return clock.now

    ConveyorScheduler(tmp_path, [ConveyorEntry("cap.c:1", item_root, "0" * 64, poll)], 1,
                      classify=_classify, retry_exception=lambda e: True, config=config, clock=advancing).run()
    wait = seen[1]
    assert wait["window_seconds"] == 18_000 + 5 * 900  # > the 4 h default window
    assert wait["deadline"] - wait["first_seen"] == pytest.approx(18_000 + 5 * 900)
    assert wait["step_timeout_seconds"] == 18_000


def test_capture_and_cold_pull_results_report_the_command_timeout(monkeypatch):
    monkeypatch.setenv(pipeline.COMMAND_TIMEOUT_ENV, "7200")
    assert pipeline._command_timeout() == 7200.0
    hint = conveyor._image_hints({"custom_images": {"state": "pending_capture", "step_timeout_seconds": 7200.0}})
    assert hint["step_timeout_seconds"] == 7200.0


def test_review_transport_failures_use_their_own_larger_budget(tmp_path):
    item_root = tmp_path / "items" / "r"
    calls = []

    def poll():
        calls.append(1)
        return write_status(item_root, {
            "state": "pending_image_review", "issues": ["image_review_transport_failed"],
            "custom_images": {"state": "pending_review", "reason": "image_review_transport_failed",
                              "retryable": True, "failure_class": "transport", "transport_attempts": len(calls)}})

    config = _config(CAPABILITY_WAIT_IMAGE_REVIEW_TRANSPORT_ATTEMPTS=5)
    result = ConveyorScheduler(tmp_path, [ConveyorEntry("cap.r:1", item_root, "0" * 64, poll)], 1,
                               classify=_classify, retry_exception=lambda e: True, config=config).run().results[0]
    assert len(calls) == 5  # the image_review budget (3) does not apply
    assert result["state"] == "failed" and result["failure_stage"] == "image_review_transport"
    assert result["wait_exhausted"]["kind"] == "image_review_transport"
    assert conveyor.DEFAULT_WAIT_BUDGETS["image_review_transport"] == (21_600, 12, 900)


def test_transient_exception_on_a_repairable_item_holds_it_for_the_relaunch(tmp_path):
    """A transient fault mid-repair keeps the repairable state (and its issues) with a hold,
    so a relaunch continues the repair loop instead of the item reading as finished."""
    item_root = tmp_path / "items" / "r"
    write_status(item_root, {"state": "failed", "issues": ["invalid task bundle: grader crashes"]},
                 intermediate=False)

    def broken():
        raise OSError(5, "Input/output error")

    out = ConveyorScheduler(tmp_path, [ConveyorEntry("cap.r:1", item_root, "0" * 64, broken)], 1,
                            classify=synthesis._conveyor_classification, retry_exception=lambda e: True,
                            config=_config()).run()
    status = json.loads((item_root / "status.json").read_text())
    assert status["state"] == "failed" and status["issues"] == ["invalid task bundle: grader crashes"]
    assert status["wait_hold"]["state"] == "failed"
    assert out.summary["controller_exception_holds"] == 1
    # quality_accepted is terminal: annotated, never held
    accepted_root = tmp_path / "items" / "a"
    write_status(accepted_root, {"state": "quality_accepted", "issues": []}, intermediate=False)
    ConveyorScheduler(tmp_path, [ConveyorEntry("cap.a:1", accepted_root, "0" * 64, broken)], 1,
                      classify=synthesis._conveyor_classification, retry_exception=lambda e: True,
                      config=_config()).run()
    accepted_status = json.loads((accepted_root / "status.json").read_text())
    assert accepted_status["state"] == "quality_accepted" and "wait_hold" not in accepted_status
