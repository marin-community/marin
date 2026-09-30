from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from capability_pipeline.image_migration import (
    ImageMigrationError,
    prepare_image_migration_revalidation,
)
from capability_pipeline.image_pointer_migration import (
    produce_pointer_migration_bundle,
    validate_pointer_migration_bundle,
)

CANDIDATE = "registry.example/tasks/candidate@sha256:" + "1" * 64
VERIFIER = "registry.example/tasks/verifier@sha256:" + "2" * 64
OLD_CANDIDATE = "sha256:" + "3" * 64
OLD_VERIFIER = "sha256:" + "4" * 64
ITEM = "c32.geometry_topology_repair-7-75adb802f475"
LEGACY_VALIDATION = """    for image in (verifier_image, harbor_image):
        digest = image.removeprefix("sha256:")
        if not (image.startswith("sha256:") or "@sha256:" in image) \\
                or len(digest) != 64:
            raise SystemExit("image references must be immutable digests")
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _evidence(root: Path, role: str, image: str) -> tuple[Path, Path]:
    plan_sha256 = "a" * 64
    producer_role = "private_verifier" if role == "verifier" else role
    required_files = {"/opt/ready": "b" * 64}
    publication = root / f"{role}-publication.json"
    _dump(
        publication,
        {
            "schema_version": "capability-task-image-publication-v1",
            "state": "published_pending_cold_pull",
            "role": producer_role,
            "plan_sha256": plan_sha256,
            "rootfs_review": {
                "state": "passed",
                "required_file_hashes": required_files,
            },
            "publication": {
                "schema_version": "capability-oci-transport-v1",
                "state": "integrity_verified",
                "image": image,
                "manifest_digest": image.rpartition("@")[2],
            },
        },
    )
    cold = root / f"{role}-cold.json"
    recipe_sha256 = hashlib.sha256(f"FROM {image}\n".encode()).hexdigest()
    sandboxes = [
        {
            "sandbox_id": f"{role}-sandbox-{index}",
            "network_block_all": True,
            "required_file_hashes": required_files,
            "registry_reachability": {"reachable": False},
            "database": {"public_tables": ["fixture"]},
            "database_and_workspace_pristine": True,
            **(
                {"database_and_workspace_mutated": True} if index == 0 else {}
            ),
        }
        for index in range(2)
    ]
    _dump(
        cold,
        {
            "schema_version": "capability-image-cold-pull-v1",
            "state": "passed",
            "role": producer_role,
            "image": image,
            "plan_sha256": plan_sha256,
            "publication_sha256": _sha(publication),
            "snapshot": {
                "snapshot_name": f"cap-cold-{recipe_sha256[:24]}",
                "dockerfile_sha256": recipe_sha256,
                "id": f"{role}-snapshot",
                "ref": f"{role}-ref",
            },
            "sandboxes": sandboxes,
            "cleanup": [
                {"sandbox_id": sandbox["sandbox_id"], "verified_absent": True}
                for sandbox in sandboxes
            ],
        },
    )
    return publication, cold


def _terminal_source(tmp_path: Path) -> tuple[Path, dict[str, tuple[Path, Path]]]:
    source = tmp_path / "terminal"
    item = source / "items" / ITEM
    accepted = {
        "proposal_hash": "75adb802f475" + "0" * 52,
        "proposal": {"builder_plan": [{"session": "s1"}]},
    }
    _dump(item / "contract/accepted.json", accepted)
    _dump(
        item / "status.json",
        {
            "key": "c32.geometry_topology_repair:7",
            "proposal_hash": accepted["proposal_hash"],
            "state": "failed",
            "sessions": [{"session": "s1", "status": "complete"}],
            "repairs": [{"round": 1}, {"round": 2}],
        },
    )
    _dump(item / "sessions/s1/status.json", {"session": "s1", "status": "complete"})
    _dump(
        item / "workspace/handoffs/s1.json",
        {
            "session": "s1",
            "status": "complete",
            "artifacts": ["grader.py"],
            "checks": [{"command": "true", "exit_code": 0}],
        },
    )
    (item / "workspace/grader.py").write_text("# unchanged grader\n")
    specification = {
        "requirements": {"state": {"image": OLD_CANDIDATE}},
        "steps": [{"verifier": {"runtime": {"image": OLD_VERIFIER}}}],
    }
    binding = {"environment": {"image": OLD_CANDIDATE}}
    manifest = {
        "specification_sha256": "placeholder",
        "binding": binding,
        "verifier_runtimes": [{"image": OLD_VERIFIER}],
    }
    for relative in (
        "harbor/specification.json",
        "workspace/task/specification.json",
        "workspace/task/harbor/specification.json",
    ):
        _dump(item / relative, specification)
    spec_sha = _sha(item / "harbor/specification.json")
    manifest["specification_sha256"] = spec_sha
    for relative in (
        "harbor/binding.json",
        "workspace/task/binding.json",
        "workspace/task/harbor/binding.json",
    ):
        _dump(item / relative, binding)
    for relative in (
        "harbor/manifest.json",
        "workspace/task/harbor/manifest.json",
    ):
        _dump(item / relative, manifest)
    (item / "workspace/task/build_task.py").write_text(
        "def main():\n"
        '    verifier_image = os.environ["VERIFIER_IMAGE"]\n'
        '    harbor_image = os.environ["HARBOR_IMAGE"]\n' + LEGACY_VALIDATION
    )
    for directory in ("repairs", "repair-history"):
        for attempt in (1, 2):
            _dump(
                source / directory / ITEM / f"attempt-{attempt}/status.json",
                {"round": attempt},
            )
    _dump(source / "report.json", {"state": "needs_continuation"})
    files = {
        str(path.relative_to(source)): _sha(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    _dump(
        source / "pull-manifest.json",
        {
            "schema_version": "capability-snapshot-pull-v1",
            "complete_manifest": True,
            "complete_snapshot": True,
            "remote_final": True,
            "remote_snapshot_id": "fixture-terminal",
            "files": files,
        },
    )
    _dump(source / "snapshot-capture.json", {"snapshot_id": "fixture-terminal"})
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    return source, {
        "candidate": _evidence(evidence_root, "candidate", CANDIDATE),
        "verifier": _evidence(evidence_root, "verifier", VERIFIER),
    }


def _bundle(tmp_path: Path) -> Path:
    source, evidence = _terminal_source(tmp_path)
    return produce_pointer_migration_bundle(
        source,
        tmp_path / "bundle",
        candidate_image=CANDIDATE,
        verifier_image=VERIFIER,
        evidence=evidence,
    )


def _validate(bundle: Path):
    return validate_pointer_migration_bundle(
        bundle,
        expected_images={"candidate": CANDIDATE, "verifier": VERIFIER},
        expected_manifest_sha256=_sha(bundle / "manifest.json"),
        expected_receipt_sha256=_sha(bundle / "migration-receipt.json"),
    )


def _rehash_manifest(bundle: Path) -> None:
    path = bundle / "manifest.json"
    value = json.loads(path.read_text())
    value["files"] = {
        str(member.relative_to(bundle)): _sha(member)
        for member in sorted(bundle.rglob("*"))
        if member.is_file() and member != path
    }
    _dump(path, value)


def test_two_pointer_bundle_preserves_history_and_is_revalidation_ready(tmp_path):
    bundle = _bundle(tmp_path)
    plan = _validate(bundle)

    assert [change.role for change in plan.pointer_changes] == [
        "candidate",
        "verifier",
    ]
    assert plan.repair_attempts == (1, 2)
    assert len(plan.changed_paths) == 9
    specification = json.loads(
        (bundle / "restore-seed/items" / ITEM / "harbor/specification.json").read_text()
    )
    assert specification["requirements"]["state"]["image"] == CANDIDATE
    assert specification["steps"][0]["verifier"]["runtime"]["image"] == VERIFIER
    assert (
        "removeprefix"
        not in (
            bundle / "restore-seed/items" / ITEM / "workspace/task/build_task.py"
        ).read_text()
    )

    prepared = prepare_image_migration_revalidation(plan, tmp_path / "result")
    operation = json.loads(prepared.operation_path.read_text())
    assert [change["role"] for change in operation["image_pointer_changes"]] == [
        "candidate",
        "verifier",
    ]
    assert "old_image" not in operation


def test_publication_and_cold_pull_are_both_required(tmp_path):
    source, evidence = _terminal_source(tmp_path)
    cold = evidence["candidate"][1]
    value = json.loads(cold.read_text())
    value["sandboxes"][0]["network_block_all"] = False
    _dump(cold, value)

    with pytest.raises(ImageMigrationError, match="cold-pull"):
        produce_pointer_migration_bundle(
            source,
            tmp_path / "bundle",
            candidate_image=CANDIDATE,
            verifier_image=VERIFIER,
            evidence=evidence,
        )


def test_undeclared_grader_change_is_rejected_even_if_bundle_is_rehashed(tmp_path):
    bundle = _bundle(tmp_path)
    grader = bundle / "restore-seed/items" / ITEM / "workspace/grader.py"
    grader.write_text("# semantic change\n")
    _rehash_manifest(bundle)

    with pytest.raises(ImageMigrationError, match="undeclared source change"):
        _validate(bundle)


def test_generator_change_is_exactly_the_registry_validation_fix(tmp_path):
    bundle = _bundle(tmp_path)
    generator = bundle / "restore-seed/items" / ITEM / "workspace/task/build_task.py"
    generator.write_text(generator.read_text() + "\n# unrelated change\n")
    receipt_path = bundle / "migration-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    relative = f"items/{ITEM}/workspace/task/build_task.py"
    next(record for record in receipt["changed_files"] if record["path"] == relative)[
        "new_sha256"
    ] = _sha(generator)
    _dump(receipt_path, receipt)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["migration_receipt_sha256"] = _sha(receipt_path)
    _dump(bundle / "manifest.json", manifest)
    _rehash_manifest(bundle)

    with pytest.raises(ImageMigrationError, match="generator changed beyond"):
        _validate(bundle)
