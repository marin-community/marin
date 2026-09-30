import hashlib
import json

import pytest
from test_generic_image_publication import _artifacts, _sha

from capability_pipeline.generic_image_cold_pull import _recipe
from capability_pipeline.generic_image_migration import migrate_image_pointers
from capability_pipeline.inference import digest


def _bind_review_task(approval, task):
    review = approval.parent
    manifest_path = review / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in (
        "specification.json",
        "binding.json",
        "composite-verifier.json",
        "judge-calibration.json",
    ):
        path = task / name
        if path.is_file():
            relative = "workspace/task/" + name
            target = review / "input" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            manifest["files"][relative] = _sha(path)
    manifest.pop("snapshot_hash")
    manifest["snapshot_hash"] = digest(manifest)
    manifest_path.write_text(json.dumps(manifest))
    raw = review / "review-raw.json"
    raw_decision = json.loads(raw.read_text())
    raw_decision["snapshot_hash"] = manifest["snapshot_hash"]
    raw.write_text(json.dumps(raw_decision))
    approval_document = json.loads(approval.read_text())
    approval_document["snapshot_hash"] = manifest["snapshot_hash"]
    approval_document["input_manifest_sha256"] = _sha(manifest_path)
    approval_document["raw_sha256"] = _sha(raw)
    approval.write_text(json.dumps(approval_document))


def _evidence(
    tmp_path, *, role="candidate", composite_canonical=False, judge_calibration=False,
    authored_pointer=None,
):
    workspace, tools, plan_path, approval, _, _ = _artifacts(
        tmp_path, role=role, authored_pointer=authored_pointer)
    plan = json.loads(plan_path.read_text())
    image = plan["images"][0]
    task = workspace / "task"
    task.mkdir()
    authored = image["authored_image_pointer"]
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": authored if role == "candidate" else None}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": authored if role == "candidate" else None}, "other": "untouched"}))
    if role == "private_verifier":
        (task / "composite-verifier.json").write_text(json.dumps({
            "specification_sha256": _sha(task / "specification.json"),
            "steps": [{"machine_checks": [{"image": authored}]}],
        }))
    elif composite_canonical:
        (task / "composite-verifier.json").write_text(json.dumps({
            "specification_sha256": _sha(task / "specification.json"),
            "steps": [{"machine_checks": [{"image": "public@sha256:" + "b" * 64}]}],
        }))
    if judge_calibration:
        (task / "judge-calibration.json").write_text(json.dumps({
            "schema_version": "taskcompendium-judge-calibration-v1",
            "specification_sha256": _sha(task / "specification.json"),
            "cases": [{"id": "fixture", "unchanged": True}],
            "fixture_note": "preserve every nonbinding field",
        }))
    _bind_review_task(approval, task)
    reference = "registry.example/" + image["repository"] + "@sha256:" + "a" * 64
    publication = tmp_path / "publication.json"
    publication.write_text(json.dumps({
        "schema_version": "capability-task-image-publication-v1",
        "state": "published_pending_cold_pull", "role": role, "plan_sha256": _sha(plan_path),
        "review_sha256": _sha(approval),
        "rootfs_review": {"state": "passed", "required_file_hashes": image["required_ready_hashes"]},
        "publication": {"state": "integrity_verified", "image": reference,
                        "manifest_digest": "sha256:" + "a" * 64},
    }))
    cold = tmp_path / "cold.json"
    cold.write_text(json.dumps({
        "schema_version": "capability-generic-image-cold-pull-v1",
        "state": "passed_pending_task_gates", "role": role, "image": reference,
        "plan_sha256": _sha(plan_path), "approval_sha256": _sha(approval),
        "publication_sha256": _sha(publication),
        "reconstruction_recipe": _recipe(reference, image["image_config"]),
        "snapshot": {"snapshot_name": "cap-cold-" + hashlib.sha256(_recipe(reference, image["image_config"]).encode()).hexdigest()[:24],
                     "dockerfile_sha256": hashlib.sha256(_recipe(reference, image["image_config"]).encode()).hexdigest(),
                     "evidence": "provider-build-info-exact-match",
                     "id": "snapshot-id", "ref": "snapshot-ref"},
        "sandboxes": [{"sandbox_id": f"sandbox-{i}", "network_block_all": True,
                       "ready": True, "required_file_hashes": image["required_ready_hashes"],
                       "sensitive_paths_checked": ["/run/secrets"]} for i in range(2)],
        "cleanup": [{"sandbox_id": f"sandbox-{i}", "state": "not_found",
                     "observations": [{"state": "not_found"}]} for i in range(2)],
    }))
    args = {"task": task, "plan_path": plan_path, "frozen_workspace": workspace,
            "capture_tools": tools, "approval_path": approval,
            "builder_session_ids": {"builder"}, "publication_paths": {role: publication},
            "cold_pull_paths": {role: cold}, "output": tmp_path / "migration"}
    return args, authored, reference


def test_migration_changes_only_exact_pointers_and_retains_bytes(tmp_path):
    args, authored, reference = _evidence(tmp_path)
    task = args["task"]
    before = {name: (task / name).read_bytes() for name in ("specification.json", "binding.json")}
    receipt = migrate_image_pointers(**args)
    assert receipt["state"] == "applied"
    assert len(receipt["changes"]) == 2
    assert json.loads((task / "specification.json").read_text())["requirements"]["state"]["image"] == reference
    binding = json.loads((task / "binding.json").read_text())
    assert binding == {"environment": {"image": reference}, "other": "untouched"}
    assert (args["output"] / "originals/specification.json").read_bytes() == before["specification.json"]
    assert receipt["documents"]["specification.json"]["before_sha256"] == hashlib.sha256(before["specification.json"]).hexdigest()
    with pytest.raises(ValueError, match="already exists"):
        migrate_image_pointers(**args)
    assert authored != reference


def test_migration_rejects_changed_authored_pointer(tmp_path):
    args, _, _ = _evidence(tmp_path)
    task = args["task"]
    specification = json.loads((task / "specification.json").read_text())
    specification["requirements"]["state"]["image"] = "unreviewed-snapshot"
    (task / "specification.json").write_text(json.dumps(specification))
    with pytest.raises(ValueError, match="current task document bytes"):
        migrate_image_pointers(**args)
    assert not args["output"].exists()


def test_migration_rejects_post_review_semantic_change_with_same_image(tmp_path):
    args, _, _ = _evidence(tmp_path)
    task = args["task"]
    specification = json.loads((task / "specification.json").read_text())
    specification["instructions"] = "Changed after image review"
    (task / "specification.json").write_text(json.dumps(specification))
    with pytest.raises(ValueError, match="current task document bytes"):
        migrate_image_pointers(**args)
    assert not args["output"].exists()


def test_migration_rejects_new_composite_after_review(tmp_path):
    args, _, _ = _evidence(tmp_path)
    (args["task"] / "composite-verifier.json").write_text("{}")
    with pytest.raises(ValueError, match="presence"):
        migrate_image_pointers(**args)


def test_migration_rejects_cold_pull_without_absent_sandbox(tmp_path):
    args, _, _ = _evidence(tmp_path)
    cold = args["cold_pull_paths"]["candidate"]
    value = json.loads(cold.read_text())
    value["cleanup"][1]["state"] = "pending"
    cold.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cleanup"):
        migrate_image_pointers(**args)
    assert not args["output"].exists()


def test_migration_updates_composite_specification_binding(tmp_path):
    args, _, reference = _evidence(tmp_path, composite_canonical=True)
    task = args["task"]
    migrate_image_pointers(**args)
    config = json.loads((task / "composite-verifier.json").read_text())
    assert config["specification_sha256"] == _sha(task / "specification.json")
    assert config["steps"][0]["machine_checks"][0]["image"] != reference


def test_private_role_migrates_composite_only(tmp_path):
    args, authored, reference = _evidence(tmp_path, role="private_verifier")
    task = args["task"]
    spec_before = (task / "specification.json").read_bytes()
    binding_before = (task / "binding.json").read_bytes()
    migrate_image_pointers(**args)
    config = json.loads((task / "composite-verifier.json").read_text())
    assert config["steps"][0]["machine_checks"][0]["image"] == reference
    assert config["specification_sha256"] == _sha(task / "specification.json")
    assert (task / "specification.json").read_bytes() == spec_before
    assert (task / "binding.json").read_bytes() == binding_before
    assert authored != reference


def test_migration_updates_only_judge_calibration_specification_binding(tmp_path):
    args, _, _ = _evidence(tmp_path, judge_calibration=True)
    task = args["task"]
    before = json.loads((task / "judge-calibration.json").read_text())
    receipt = migrate_image_pointers(**args)
    after = json.loads((task / "judge-calibration.json").read_text())
    assert after["specification_sha256"] == _sha(task / "specification.json")
    assert {key: value for key, value in after.items() if key != "specification_sha256"} == {
        key: value for key, value in before.items() if key != "specification_sha256"
    }
    document = receipt["documents"]["judge-calibration.json"]
    assert document["before_sha256"] == _sha(args["output"] / "originals/judge-calibration.json")
    assert document["after_sha256"] == _sha(task / "judge-calibration.json")


def test_migration_rejects_unreviewed_judge_calibration(tmp_path):
    args, _, _ = _evidence(tmp_path)
    task = args["task"]
    (task / "judge-calibration.json").write_text(json.dumps({
        "schema_version": "taskcompendium-judge-calibration-v1",
        "specification_sha256": _sha(task / "specification.json"),
    }))
    with pytest.raises(ValueError, match="presence"):
        migrate_image_pointers(**args)


@pytest.mark.parametrize("authored", [
    # The builder's Daytona recipe fingerprint: digest-shaped, not a pullable OCI digest.
    "envgen.daytona/d02q4-pg16-verifier-supervisor/dockerfile@sha256:" + "5" * 64,
    # A digest-shaped pointer the builder explicitly requested capture for.
    "python:3.11-slim@sha256:" + "1" * 64,
])
def test_migration_replaces_digest_shaped_reviewed_pointer(tmp_path, authored):
    # request_needed sends these pointers down the capture path; migration used
    # to classify them as already pinned, find no custom pointer and raise
    # "current authored pointers differ from frozen reviewed plan" on every pass.
    args, planned, reference = _evidence(tmp_path, authored_pointer=authored)
    assert planned == authored
    receipt = migrate_image_pointers(**args)
    assert receipt["state"] == "applied"
    assert [change["image"] for change in receipt["changes"]] == [authored, authored]
    task = args["task"]
    assert json.loads((task / "specification.json").read_text())["requirements"]["state"]["image"] == reference
    assert json.loads((task / "binding.json").read_text())["environment"]["image"] == reference


def test_migration_leaves_unrequested_pinned_pointer_alone(tmp_path):
    args, _, _ = _evidence(tmp_path, composite_canonical=True)
    migrate_image_pointers(**args)
    config = json.loads((args["task"] / "composite-verifier.json").read_text())
    assert config["steps"][0]["machine_checks"][0]["image"] == "public@sha256:" + "b" * 64
