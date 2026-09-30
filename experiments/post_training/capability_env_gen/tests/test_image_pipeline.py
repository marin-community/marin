import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline import image_pipeline as pipeline


def _fixture(tmp_path, monkeypatch):
    item = tmp_path / "items/task-1"
    task = item / "workspace/task"
    task.mkdir(parents=True)
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": "authored-snapshot"}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": "authored-snapshot"}}))
    attempt = item / "diagnostics/image-capture/attempt-source"
    frozen = attempt / "input/workspace"
    frozen.mkdir(parents=True)
    plan = attempt / "input/plan.json"
    plan.write_text(json.dumps({"images": [{"role": "candidate", "source_snapshot": {
        "name": "authored-snapshot", "id": "snapshot-id", "ref": "snapshot-ref"}}]}))
    tools = tmp_path / "tools/capture-tools"
    tools.mkdir(parents=True)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(pipeline, "prepare_construction_capture", lambda *_: {
        "attempt": str(attempt), "plan_path": str(plan), "workspace": str(frozen)})
    monkeypatch.setattr(pipeline, "validate_review", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "_review_matches_task", lambda *a: None)
    return item, attempt, plan, frozen, tools, scripts


def _args(item, tools, scripts, runner):
    return {"item_root": item, "capture_tools": tools, "scripts_root": scripts,
            "agent": SimpleNamespace(model="glm-5.3"), "builder_session_ids": {"builder"},
            "command_runner": runner}


def test_pinned_images_require_no_review_or_commands(tmp_path):
    item = tmp_path / "item"
    task = item / "workspace/task"
    task.mkdir(parents=True)
    image = "python@sha256:" + "a" * 64
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": image}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": image}}))
    result = pipeline.process_image_construction(**_args(
        item, tmp_path, tmp_path, lambda _: pytest.fail("no command expected")))
    assert result["state"] == "ready"
    assert result["reason"] == "pinned_public_images"


def test_migrated_digest_resume_requires_unchanged_output(tmp_path):
    item = tmp_path / "item"
    task = item / "workspace/task"
    task.mkdir(parents=True)
    image = "python@sha256:" + "a" * 64
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": image}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": image}}))
    migration = item / "diagnostics/image-capture/attempt-one/migration"
    migration.mkdir(parents=True)
    receipt = {"state": "applied", "roles": {"candidate": {"image": image}},
               "documents": {name: {"after_sha256": pipeline._sha(task / name)}
                             for name in ("specification.json", "binding.json")}}
    (migration / "migration.json").write_text(json.dumps(receipt))
    args = _args(item, tmp_path, tmp_path, lambda _: pytest.fail("no command expected"))
    assert pipeline.process_image_construction(**args)["reason"] == "reviewed_images_migrated"
    (task / "binding.json").write_text(json.dumps({"environment": {"image": "python@sha256:" + "b" * 64}}))
    assert pipeline.process_image_construction(**args)["state"] == "pending_migration"


def test_migrated_judge_calibration_receipt_requires_updated_fixture_bytes(tmp_path):
    item = tmp_path / "item"
    task = item / "workspace/task"
    task.mkdir(parents=True)
    image = "python@sha256:" + "a" * 64
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": image}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": image}}))
    calibration = task / "judge-calibration.json"
    calibration.write_text(json.dumps({
        "schema_version": "taskcompendium-judge-calibration-v1",
        "specification_sha256": pipeline._sha(task / "specification.json"),
    }))
    migration = item / "diagnostics/image-capture/attempt-one/migration"
    migration.mkdir(parents=True)
    receipt = {
        "state": "applied",
        "roles": {"candidate": {"image": image}},
        "documents": {
            name: {"after_sha256": pipeline._sha(task / name)}
            for name in ("specification.json", "binding.json", "judge-calibration.json")
        },
    }
    (migration / "migration.json").write_text(json.dumps(receipt))
    args = _args(item, tmp_path, tmp_path, lambda _: pytest.fail("no command expected"))
    assert pipeline.process_image_construction(**args)["state"] == "ready"
    calibration.write_text(calibration.read_text() + "\n")
    assert pipeline.process_image_construction(**args)["state"] == "pending_migration"


def test_migrated_resume_selects_matching_receipt_not_last_name(tmp_path):
    item = tmp_path / "item"
    task = item / "workspace/task"
    task.mkdir(parents=True)
    image = "python@sha256:" + "a" * 64
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": image}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": image}}))
    good = item / "diagnostics/image-capture/attempt-a/migration"
    stale = item / "diagnostics/image-capture/attempt-z/migration"
    good.mkdir(parents=True)
    stale.mkdir(parents=True)
    receipt = {"state": "applied", "roles": {"candidate": {"image": image}},
               "documents": {name: {"after_sha256": pipeline._sha(task / name)}
                             for name in ("specification.json", "binding.json")}}
    (good / "migration.json").write_text(json.dumps(receipt))
    receipt["documents"]["binding.json"]["after_sha256"] = "0" * 64
    (stale / "migration.json").write_text(json.dumps(receipt))
    result = pipeline.process_image_construction(**_args(
        item, tmp_path, tmp_path, lambda _: pytest.fail("no command expected")))
    assert result["state"] == "ready"
    assert result["migration_path"] == str(good / "migration.json")


def test_invalid_authored_document_is_repairable(tmp_path):
    item = tmp_path / "item"
    task = item / "workspace/task"
    task.mkdir(parents=True)
    (task / "specification.json").write_text("{")
    (task / "binding.json").write_text("{}")
    result = pipeline.process_image_construction(**_args(
        item, tmp_path, tmp_path, lambda _: pytest.fail("no command expected")))
    assert result["state"] == "repairable"
    assert result["issues"]


def test_review_repair_stops_before_capture(tmp_path, monkeypatch):
    item, _, _, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "run_review", lambda **kw: {
        "state": "repair", "issues": ["Source closure incomplete"]})
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no command expected")))
    assert result["state"] == "repairable"
    assert result["issues"] == ["Source closure incomplete"]


def test_invalid_builder_request_returns_actionable_repair(tmp_path, monkeypatch):
    item, _, _, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    def invalid(*_):
        raise ValueError("capture request roles differ from custom task pointers")
    monkeypatch.setattr(pipeline, "prepare_construction_capture", invalid)
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no command expected")))
    assert result["state"] == "repairable"
    assert result["issues"] == ["capture request roles differ from custom task pointers"]


def test_capture_once_then_explicit_pending_publication(tmp_path, monkeypatch):
    item, attempt, plan, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    def review(**kw):
        root = kw["review_root"]
        root.mkdir(parents=True)
        (root / "approval.json").write_text("{}")
        return {"state": "approve"}
    monkeypatch.setattr(pipeline, "run_review", review)
    calls = []
    def runner(command):
        calls.append(command)
        assert Path(command[0]).name == "capture_generic_task_image.py"
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({
            "schema_version": "capability-rootfs-capture-v1",
            "state": "captured_pending_privacy_and_publication",
            "role": "candidate", "plan_sha256": pipeline._sha(plan),
            "source_snapshot": {"name": "authored-snapshot", "id": "snapshot-id", "ref": "snapshot-ref",
                                "cpu": 2, "mem": 4, "disk": 10, "state": "SnapshotState.ACTIVE"},
            "cleanup": {"absence_verified": True},
        }))
        return SimpleNamespace(returncode=0)
    first = pipeline.process_image_construction(**_args(item, tools, scripts, runner))
    second = pipeline.process_image_construction(**_args(item, tools, scripts, runner))
    assert first["state"] == second["state"] == "pending_publication"
    assert first["plan_path"] == str(plan)
    assert first["capture_paths"]["candidate"] == str(attempt / "capture-candidate.json")
    assert len(calls) == 1


def test_existing_publication_advances_to_cold_pull_and_migration(tmp_path, monkeypatch):
    item, attempt, plan, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    review = item.parent / "image-reviews" / item.name / attempt.name
    review.mkdir(parents=True)
    (review / "approval.json").write_text("{}")
    (attempt / "capture-candidate.json").write_text(json.dumps({
        "schema_version": "capability-rootfs-capture-v1",
        "state": "captured_pending_privacy_and_publication", "role": "candidate",
        "plan_sha256": pipeline._sha(plan),
        "source_snapshot": {"name": "authored-snapshot", "id": "snapshot-id", "ref": "snapshot-ref"},
        "cleanup": {"absence_verified": True},
    }))
    (attempt / "publication-candidate.json").write_text("{}")
    def runner(command):
        assert Path(command[0]).name == "probe_generic_task_image.py"
        Path(command[command.index("--output") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline, "migrate_image_pointers", lambda **kw: {
        "roles": {"candidate": {"image": "registry/task@sha256:" + "a" * 64}}})
    result = pipeline.process_image_construction(**_args(item, tools, scripts, runner))
    assert result["state"] == "ready"
    assert result["reason"] == "reviewed_images_migrated"


def test_capture_privacy_failure_returns_builder_repair_even_on_nonzero_exit(tmp_path, monkeypatch):
    item, _, plan, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    def review(**kw):
        kw["review_root"].mkdir(parents=True)
        (kw["review_root"] / "approval.json").write_text("{}")
        return {"state": "approve"}
    monkeypatch.setattr(pipeline, "run_review", review)
    def runner(command):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({
            "schema_version": "capability-rootfs-capture-v1",
            "state": "privacy_review_required", "role": "candidate",
            "plan_sha256": pipeline._sha(plan),
            "source_snapshot": {"name": "authored-snapshot", "id": "snapshot-id", "ref": "snapshot-ref"},
            "cleanup": {"absence_verified": True},
        }))
        return SimpleNamespace(returncode=1)
    result = pipeline.process_image_construction(**_args(item, tools, scripts, runner))
    assert result["state"] == "repairable"
    assert result["reason"] == "image_sensitive_paths_present"
    assert "updated image-capture-request.json" in result["issues"][0]


def _published_fixture(tmp_path, monkeypatch):
    item, attempt, plan, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    review = item.parent / "image-reviews" / item.name / attempt.name
    review.mkdir(parents=True)
    (review / "approval.json").write_text("{}")
    (attempt / "capture-candidate.json").write_text(json.dumps({
        "schema_version": "capability-rootfs-capture-v1",
        "state": "captured_pending_privacy_and_publication", "role": "candidate",
        "plan_sha256": pipeline._sha(plan),
        "source_snapshot": {"name": "authored-snapshot", "id": "snapshot-id", "ref": "snapshot-ref"},
        "cleanup": {"absence_verified": True},
    }))
    (attempt / "publication-candidate.json").write_text("{}")
    return item, attempt, tools, scripts


def _cold_runner(calls, state="passed_pending_task_gates"):
    def runner(command):
        assert Path(command[0]).name == "probe_generic_task_image.py"
        output = Path(command[command.index("--output") + 1])
        assert not output.exists()  # the probe refuses an existing receipt
        calls.append(output)
        output.write_text(json.dumps({"state": state, "error_type": None if state.startswith("passed") else "TransportError"}))
        return SimpleNamespace(returncode=0 if state.startswith("passed") else 1)
    return runner


def test_failed_cold_pull_receipt_is_retired_and_rerun(tmp_path, monkeypatch):
    # A failed cold boot still writes its receipt (state "pending").  It used to
    # be skipped as done, so migration rejected it on every pass until the
    # image_migration wait budget ran out.
    item, attempt, tools, scripts = _published_fixture(tmp_path, monkeypatch)
    migrated = []
    monkeypatch.setattr(pipeline, "migrate_image_pointers", lambda **kw: migrated.append(kw) or {
        "roles": {"candidate": {"image": "registry/task@sha256:" + "a" * 64}}})
    calls = []
    first = pipeline.process_image_construction(**_args(item, tools, scripts, _cold_runner(calls, "pending")))
    assert first["state"] == "pending_cold_pull"
    assert first["reason"] == "cold_pull_command_failed"
    assert not migrated
    result = pipeline.process_image_construction(**_args(item, tools, scripts, _cold_runner(calls)))
    assert result["state"] == "ready"
    assert len(calls) == 2 and len(migrated) == 1
    retired = attempt / "cold-pull-candidate.failed-1.json"
    assert json.loads(retired.read_text())["state"] == "pending"
    assert json.loads((attempt / "cold-pull-candidate.json").read_text())["state"] == "passed_pending_task_gates"


def test_passed_cold_pull_receipt_is_not_rerun(tmp_path, monkeypatch):
    item, attempt, tools, scripts = _published_fixture(tmp_path, monkeypatch)
    (attempt / "cold-pull-candidate.json").write_text(json.dumps({"state": "passed_pending_task_gates"}))
    monkeypatch.setattr(pipeline, "migrate_image_pointers", lambda **kw: {
        "roles": {"candidate": {"image": "registry/task@sha256:" + "a" * 64}}})
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no cold pull expected")))
    assert result["state"] == "ready"


def test_cold_pull_retirements_are_bounded(tmp_path, monkeypatch):
    item, attempt, tools, scripts = _published_fixture(tmp_path, monkeypatch)
    for index in range(1, pipeline.MAX_COLD_PULL_RETIREMENTS + 1):
        (attempt / f"cold-pull-candidate.failed-{index}.json").write_text(json.dumps({"state": "pending"}))
    (attempt / "cold-pull-candidate.json").write_text("not json")
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no cold pull expected")))
    assert result["state"] == "failed_terminal"
    assert result["failure_stage"] == "image_cold_pull"
    assert result["attempts"] == pipeline.MAX_COLD_PULL_RETIREMENTS + 1
    assert (attempt / "cold-pull-candidate.json").exists()


def test_migrated_item_with_builder_request_resumes_ready(tmp_path):
    # After migration the builder's image-capture-request.json still names the
    # replaced pointer; the applied receipt excuses it instead of turning a
    # migrated item into invalid_authored_image_documents.
    item = tmp_path / "item"
    task = item / "workspace/task"
    task.mkdir(parents=True)
    authored = "envgen.daytona/snap/dockerfile@sha256:" + "d" * 64
    image = "registry.example/task@sha256:" + "a" * 64
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": image}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": image}}))
    (task / "image-capture-request.json").write_text(json.dumps({
        "images": [{"role": "candidate", "authored_image_pointer": authored}]}))
    migration = item / "diagnostics/image-capture/attempt-one/migration"
    migration.mkdir(parents=True)
    receipt = {"state": "applied", "roles": {"candidate": {"image": image}},
               "changes": [{"role": "candidate", "pointer": "binding.environment.image", "image": authored,
                            "published_image": image}],
               "documents": {name: {"after_sha256": pipeline._sha(task / name)}
                             for name in ("specification.json", "binding.json")}}
    (migration / "migration.json").write_text(json.dumps(receipt))
    args = _args(item, tmp_path, tmp_path, lambda _: pytest.fail("no command expected"))
    result = pipeline.process_image_construction(**args)
    assert result["state"] == "ready"
    assert result["reason"] == "reviewed_images_migrated"
    receipt["state"] = "prepared"
    (migration / "migration.json").write_text(json.dumps(receipt))
    assert pipeline.process_image_construction(**args)["state"] == "repairable"
