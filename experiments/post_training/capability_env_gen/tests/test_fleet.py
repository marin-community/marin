import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline import fleet


@pytest.fixture
def plan_root(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, "_controller", lambda: {"controller": "fixed"})
    monkeypatch.setattr(fleet, "LEDGER_ROOT", tmp_path / "launch-ledger")
    root = tmp_path / "fleet"
    fleet.create_plan(Path(__file__).resolve().parents[1] / "data/new-catalog-cohort-001.json",
                      root, "runs/fleet-test")
    return root


def test_partitions_all_capabilities_and_keeps_hundreds_wide_capacity(plan_root):
    plan = fleet.validate_plan(plan_root)
    assert plan["capabilities"] == 45 and plan["slots"] == 450
    assert len(plan["shards"]) == 4
    assert sum(row["concurrency"] for row in plan["shards"]) == 256
    assert [len(row["capability_ids"]) for row in plan["shards"]] == [12, 11, 11, 11]


def test_repeated_submit_does_not_start_duplicate_workers(plan_root):
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"submitted", stderr=b"")

    assert fleet.submit_plan(plan_root, launch=launch)["submitted"] == 4
    assert fleet.submit_plan(plan_root, launch=launch)["submitted"] == 4
    assert len(calls) == 4
    assert len({kwargs["env"]["CAPABILITY_SUBMISSION_LOCK"] for _, kwargs in calls}) == 4
    assert len({command[-1] for command, _ in calls}) == 4
    assert all(kwargs["env"]["MEM"] == "128g" for _, kwargs in calls)
    assert all(kwargs["env"]["CAPABILITY_REQUIRE_EMPTY_DESTINATION"] == "1" for _, kwargs in calls)


def test_ambiguous_submission_is_not_replayed(plan_root):
    (plan_root / "shard-000.started.json").write_text("{}")
    result = fleet.submit_plan(plan_root, launch=lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout=b"", stderr=b""))
    assert result["submitted"] == 3
    assert result["receipts"][0]["state"] == "pending_submission_observation"


def test_changed_shard_or_controller_prevents_launch(plan_root, monkeypatch):
    shard = plan_root / "shard-000.json"
    data = json.loads(shard.read_text())
    data["capabilities"].pop()
    shard.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="shard changed"):
        fleet.validate_plan(plan_root)
    monkeypatch.setattr(fleet, "_controller", lambda: {"controller": "changed"})
    with pytest.raises(ValueError, match="controller changed"):
        fleet.validate_plan(plan_root)


def test_changed_plan_cannot_reuse_durable_destinations(plan_root):
    original = fleet.validate_plan(plan_root)
    other = plan_root.parent / "other"
    changed = fleet.create_plan(plan_root / "input-pilot.json", other, "runs/fleet-test", memory="256g")
    fleet.validate_plan(other)
    assert not {s["output"] for s in original["shards"]}.intersection(s["output"] for s in changed["shards"])


def test_drift_after_initial_validation_prevents_handoff(plan_root, monkeypatch):
    original = fleet.validate_plan
    calls = 0

    def validate(root):
        nonlocal calls
        value = original(root)
        calls += 1
        if calls == 1:
            with (root / "shard-000.json").open("a") as stream:
                stream.write("\n")
        return value

    monkeypatch.setattr(fleet, "validate_plan", validate)
    with pytest.raises(ValueError, match="shard changed"):
        fleet.submit_plan(plan_root, launch=lambda *_a, **_k: pytest.fail("changed input launched"))


def test_identical_plan_in_second_directory_uses_shared_launch_ledger(plan_root):
    other = plan_root.parent / "second-plan"
    fleet.create_plan(plan_root / "input-pilot.json", other, "runs/fleet-test")
    assert fleet.validate_plan(other) == fleet.validate_plan(plan_root)
    calls = []

    def launch(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    assert fleet.submit_plan(plan_root, launch=launch)["submitted"] == 4
    assert fleet.submit_plan(other, launch=launch)["submitted"] == 4
    assert len(calls) == 4


def test_remote_fleet_destination_must_be_empty():
    from scripts.check_fleet_destination import require_empty

    require_empty(SimpleNamespace(exists=lambda path: False), "bucket/fresh")
    with pytest.raises(ValueError, match="already exists"):
        require_empty(SimpleNamespace(exists=lambda path: True), "bucket/old/")


def test_stage_and_tier_are_frozen_into_the_submitted_command(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, "_controller", lambda: {"controller": "fixed"})
    monkeypatch.setattr(fleet, "LEDGER_ROOT", tmp_path / "launch-ledger")
    root = tmp_path / "propose-fleet"
    fleet.create_plan(Path(__file__).resolve().parents[1] / "data/new-catalog-cohort-001.json",
                      root, "runs/fleet-test", stage="propose", tier="bulk")
    calls = []
    fleet.submit_plan(root, launch=lambda command, **kwargs: calls.append(command)
                      or SimpleNamespace(returncode=0, stdout=b"", stderr=b""))
    assert calls and all(c[c.index("--stage") + 1] == "propose" for c in calls)
    assert all(c[c.index("--tier") + 1] == "bulk" for c in calls)


def test_stage_and_tier_change_the_fleet_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, "_controller", lambda: {"controller": "fixed"})
    pilot = Path(__file__).resolve().parents[1] / "data/new-catalog-cohort-001.json"
    a = fleet.create_plan(pilot, tmp_path / "a", "runs/x")
    b = fleet.create_plan(pilot, tmp_path / "b", "runs/x", stage="propose", tier="bulk")
    assert a["fleet_id"] != b["fleet_id"]
    with pytest.raises(ValueError, match="stage must be one of"):
        fleet.create_plan(pilot, tmp_path / "c", "runs/x", stage="synthesize")


def test_local_submission_parallelism_is_bounded_but_every_shard_submits(plan_root):
    import threading
    import time

    live = peak = 0
    lock = threading.Lock()

    def launch(command, **kwargs):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    report = fleet.submit_plan(plan_root, launch=launch, parallel=2)
    assert report["submitted"] == 4 and peak <= 2


def test_submission_does_not_rerun_the_full_semantic_validation_per_shard(plan_root, monkeypatch):
    calls = 0
    original = fleet.validate_plan

    def counting(root):
        nonlocal calls
        calls += 1
        return original(root)

    monkeypatch.setattr(fleet, "validate_plan", counting)
    fleet.submit_plan(plan_root, launch=lambda *_a, **_k: SimpleNamespace(
        returncode=0, stdout=b"", stderr=b""))
    assert calls == 1
