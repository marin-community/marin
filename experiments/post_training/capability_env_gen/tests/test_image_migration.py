from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from capability_pipeline.image_migration import (
    ImageMigrationError,
    prepare_image_migration_revalidation,
    run_prepared_image_migration,
    validate_image_migration_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "c05-portable-runtime-006"
IMAGE = (
    "docker.io/library/python:3.12-slim@"
    "sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
)
ITEM = "c05.analysis.protocol_trace-10-dda40ab415c3"
MANIFEST_SHA = "6fa887f054f6aae1ec89d26f142d3e9742481e314cc5221d9e3680daa815c175"
RECEIPT_SHA = "8dbaae73d28bae73b4e7a1303d63422243c985ae90d5b37dfd16f406fb34d2c8"

PULL_SPEC = importlib.util.spec_from_file_location(
    "image_migration_pull_snapshot", ROOT / "scripts" / "pull_snapshot.py"
)
assert PULL_SPEC and PULL_SPEC.loader
PULL_SNAPSHOT = importlib.util.module_from_spec(PULL_SPEC)
PULL_SPEC.loader.exec_module(PULL_SNAPSHOT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _copy_bundle(tmp_path: Path) -> Path:
    target = tmp_path / "bundle"
    shutil.copytree(SOURCE, target)
    return target


def _rehash_bundle(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["migration_receipt_sha256"] = _sha256(bundle / "migration-receipt.json")
    manifest["structural_validation_sha256"] = _sha256(
        bundle / "structural-validation.json"
    )
    manifest["files"] = {
        str(path.relative_to(bundle)): _sha256(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    _dump(manifest_path, manifest)


def _validate(bundle: Path):
    return validate_image_migration_bundle(
        bundle,
        expected_image=IMAGE,
        expected_manifest_sha256=_sha256(bundle / "manifest.json"),
        expected_receipt_sha256=_sha256(bundle / "migration-receipt.json"),
    )


def test_valid_image_only_migration_is_admitted() -> None:
    assert _sha256(SOURCE / "manifest.json") == MANIFEST_SHA
    assert _sha256(SOURCE / "migration-receipt.json") == RECEIPT_SHA
    plan = validate_image_migration_bundle(
        SOURCE,
        expected_image=IMAGE,
        expected_manifest_sha256=MANIFEST_SHA,
        expected_receipt_sha256=RECEIPT_SHA,
    )

    assert plan.item_name == ITEM
    assert plan.repair_attempts == (1, 2)
    assert plan.new_specification_sha256 == (
        "053263c42a415a25fd18c7bdacc4b3f78358793cf31e3ab1f52e2a255f283ef7"
    )
    assert len(plan.changed_paths) == 6


def test_reviewed_bundle_hash_mismatch_is_rejected() -> None:
    with pytest.raises(
        ImageMigrationError, match="differs from the reviewed migration"
    ):
        validate_image_migration_bundle(
            SOURCE,
            expected_image=IMAGE,
            expected_manifest_sha256="0" * 64,
            expected_receipt_sha256=RECEIPT_SHA,
        )


def test_unchanged_grader_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = _copy_bundle(tmp_path)
    grader = bundle / "restore-seed" / "items" / ITEM / "workspace" / "grader.py"
    grader.write_text(grader.read_text() + "\n# undeclared change\n")
    _rehash_bundle(bundle)

    with pytest.raises(
        ImageMigrationError, match="undeclared terminal artifact change"
    ):
        _validate(bundle)


def test_extra_taskspec_semantic_change_is_rejected(tmp_path: Path) -> None:
    bundle = _copy_bundle(tmp_path)
    item = bundle / "restore-seed" / "items" / ITEM
    specification_paths = (
        item / "harbor/specification.json",
        item / "workspace/task/specification.json",
        item / "workspace/task/harbor/specification.json",
    )
    for path in specification_paths:
        value = json.loads(path.read_text())
        value["difficulty"] = value["difficulty"] + 1
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    new_specification_sha = _sha256(specification_paths[0])
    manifest_paths = (
        item / "harbor/manifest.json",
        item / "workspace/task/harbor/manifest.json",
    )
    for path in manifest_paths:
        value = json.loads(path.read_text())
        value["specification_sha256"] = new_specification_sha
        _dump(path, value)

    receipt_path = bundle / "migration-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["task_specification"]["new_sha256"] = new_specification_sha
    hashes = {
        str(path.relative_to(bundle)): _sha256(path)
        for path in (*specification_paths, *manifest_paths)
    }
    for record in receipt["changed_files"]:
        if record["path"] in hashes:
            record["new_sha256"] = hashes[record["path"]]
    _dump(receipt_path, receipt)
    _rehash_bundle(bundle)

    with pytest.raises(ImageMigrationError, match="extra semantic change"):
        _validate(bundle)


@pytest.mark.parametrize("reset_kind", ["missing_history", "status_budget_reset"])
def test_missing_history_or_budget_reset_is_rejected(
    tmp_path: Path, reset_kind: str
) -> None:
    bundle = _copy_bundle(tmp_path)
    restore = bundle / "restore-seed"
    if reset_kind == "missing_history":
        shutil.rmtree(restore / "repairs" / ITEM / "attempt-2")
        shutil.rmtree(restore / "repair-history" / ITEM / "attempt-2")
    else:
        status_path = restore / "items" / ITEM / "status.json"
        status = json.loads(status_path.read_text())
        status["repairs"] = []
        _dump(status_path, status)
    _rehash_bundle(bundle)

    with pytest.raises(ImageMigrationError):
        _validate(bundle)


def test_prepared_run_archives_prior_gates_and_calls_once(tmp_path: Path) -> None:
    plan = _validate(SOURCE)
    prepared = prepare_image_migration_revalidation(plan, tmp_path / "result")
    prepared_status = json.loads((prepared.item_root / "status.json").read_text())
    archived_status = json.loads((prepared.history_root / "status.json").read_text())
    assert prepared_status["state"] == "pending_image_migration_revalidation"
    assert archived_status["state"] == "failed"
    migration_status = prepared_status["image_migration_revalidation"]
    assert migration_status["operation_state"] == "prepared"
    assert migration_status["prior_state"] == "failed"
    assert migration_status["prior_status_artifact"].endswith(
        "/revalidation-1/status.json"
    )
    assert migration_status["prior_status_sha256"] == _sha256(
        prepared.history_root / "status.json"
    )
    output_files = {
        str(path.relative_to(prepared.root)): _sha256(path)
        for path in prepared.root.rglob("*")
        if path.is_file()
    }
    assert (
        PULL_SNAPSHOT.validate_members(
            {"snapshot_id": "prepared-result", "files": output_files}
        )
        == output_files
    )
    assert "pull-manifest.json" not in output_files
    assert "snapshot-capture.json" not in output_files
    calls = []

    def attempt(item, root):
        calls.append((item, root))
        assert not (root / "items" / ITEM / "harbor").exists()
        active = json.loads((root / "items" / ITEM / "status.json").read_text())
        assert active["state"] == "image_migration_revalidation_running"
        assert active["image_migration_revalidation"]["operation_state"] == "running"
        return {
            "key": "c05.analysis.protocol_trace:10",
            "proposal_hash": item["proposal_hash"],
            "sessions": [],
            "state": "pending_runtime",
            "issues": ["test boundary"],
            "item_root": str(root / "items" / ITEM),
        }

    result = run_prepared_image_migration(prepared, attempt)

    assert len(calls) == 1
    assert result["state"] == "pending_runtime"
    assert result["image_migration_revalidation"]["semantic_repair_performed"] is False
    assert (prepared.history_root / "status.json").is_file()
    assert (prepared.history_root / "harbor").is_dir()
    assert sorted(
        path.name
        for path in (prepared.root / "repairs" / ITEM).iterdir()
        if path.is_dir()
    ) == ["attempt-1", "attempt-2"]
    operation = json.loads(prepared.operation_path.read_text())
    assert operation["state"] == "completed"
    assert operation["result"]["state"] == "pending_runtime"


def test_prepared_run_closes_operation_on_failure(tmp_path: Path) -> None:
    prepared = prepare_image_migration_revalidation(
        _validate(SOURCE), tmp_path / "result"
    )

    def fail(_item, _root):
        raise RuntimeError("maintained gate failed to start")

    with pytest.raises(RuntimeError, match="failed to start"):
        run_prepared_image_migration(prepared, fail)

    operation = json.loads(prepared.operation_path.read_text())
    assert operation["state"] == "error"
    assert operation["result"]["exception_type"] == "RuntimeError"
    assert operation["completed_at"] is not None
    status = json.loads((prepared.item_root / "status.json").read_text())
    assert status["state"] == "image_migration_revalidation_error"
    assert status["image_migration_revalidation"]["operation_state"] == "error"
    assert status["issues"] == ["maintained gates raised RuntimeError"]


def test_maintained_gate_cannot_change_grader(tmp_path: Path) -> None:
    prepared = prepare_image_migration_revalidation(
        _validate(SOURCE), tmp_path / "result"
    )

    def mutate(item, root):
        grader = root / "items" / ITEM / "workspace/grader.py"
        grader.write_text(grader.read_text() + "\n# forbidden\n")
        return {
            "key": "c05.analysis.protocol_trace:10",
            "proposal_hash": item["proposal_hash"],
            "sessions": [],
            "state": "failed",
            "issues": ["test"],
            "item_root": str(root / "items" / ITEM),
        }

    with pytest.raises(ImageMigrationError, match="protected task artifact"):
        run_prepared_image_migration(prepared, mutate)

    operation = json.loads(prepared.operation_path.read_text())
    assert operation["state"] == "error"
