import importlib.util
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "restore_bundle", ROOT / "scripts/restore_bundle.py"
)
restore_bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore_bundle)


def bundle(
    root: Path, schema_version: str = "capability-portable-runtime-bundle-v1"
) -> tuple[str, str]:
    (root / "restore-seed/items/x").mkdir(parents=True)
    (root / "accepted.json").write_text("[]\n")
    (root / "migration-receipt.json").write_text('{"schema_version":"receipt"}\n')
    (root / "restore-seed/items/x/.controller.lock").write_text("")
    executable = root / "restore-seed/items/x/check.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    files = {
        path.relative_to(root).as_posix(): restore_bundle.sha256(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "files": files,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return (
        restore_bundle.sha256((root / "manifest.json").read_bytes()),
        restore_bundle.sha256((root / "migration-receipt.json").read_bytes()),
    )


@pytest.mark.parametrize(
    "schema_version",
    [
        "capability-portable-runtime-bundle-v1",
        "capability-portable-runtime-bundle-v2",
    ],
)
def test_round_trip_is_deterministic_and_preserves_hidden_members(
    tmp_path, schema_version
):
    source = tmp_path / "source"
    manifest_sha, receipt_sha = bundle(source, schema_version)
    archives = [tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"]
    transports = [tmp_path / "one.json", tmp_path / "two.json"]
    for archive, transport in zip(archives, transports, strict=True):
        restore_bundle.build_transport(
            source,
            archive,
            transport,
            reviewed_manifest_sha256=manifest_sha,
            migration_receipt_sha256=receipt_sha,
        )
    assert archives[0].read_bytes() == archives[1].read_bytes()
    destination = tmp_path / "restored"
    result = restore_bundle.extract_transport(
        archives[0],
        transports[0],
        destination,
        expected_manifest_sha256=manifest_sha,
        expected_receipt_sha256=receipt_sha,
    )
    assert result["member_count"] == 5
    assert (destination / "restore-seed/items/x/.controller.lock").is_file()
    assert (
        destination / "restore-seed/items/x/check.sh"
    ).stat().st_mode & 0o777 == 0o755
    assert (
        restore_bundle.bundle_members(destination).keys()
        == restore_bundle.bundle_members(source).keys()
    )


def test_restore_rejects_changed_archive_and_reviewed_hash(tmp_path):
    source = tmp_path / "source"
    manifest_sha, receipt_sha = bundle(source)
    archive, transport = tmp_path / "bundle.tar.gz", tmp_path / "transport.json"
    restore_bundle.build_transport(
        source,
        archive,
        transport,
        reviewed_manifest_sha256=manifest_sha,
        migration_receipt_sha256=receipt_sha,
    )
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive fingerprint"):
        restore_bundle.extract_transport(
            archive,
            transport,
            tmp_path / "out",
            expected_manifest_sha256=manifest_sha,
            expected_receipt_sha256=receipt_sha,
        )
    with pytest.raises(ValueError, match="does not match reviewed"):
        restore_bundle.extract_transport(
            tmp_path / "missing.tar.gz",
            transport,
            tmp_path / "other",
            expected_manifest_sha256="0" * 64,
            expected_receipt_sha256=receipt_sha,
        )


def test_restore_rejects_unsafe_or_duplicate_tar_members(tmp_path):
    source = tmp_path / "source"
    manifest_sha, receipt_sha = bundle(source)
    archive, transport = tmp_path / "bundle.tar.gz", tmp_path / "transport.json"
    restore_bundle.build_transport(
        source,
        archive,
        transport,
        reviewed_manifest_sha256=manifest_sha,
        migration_receipt_sha256=receipt_sha,
    )
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    bad = payload.getvalue()
    archive.write_bytes(bad)
    document = json.loads(transport.read_text())
    document["archive_sha256"] = restore_bundle.sha256(bad)
    document["member_count"] = 1
    transport.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unsafe member"):
        restore_bundle.extract_transport(
            archive,
            transport,
            tmp_path / "out",
            expected_manifest_sha256=manifest_sha,
            expected_receipt_sha256=receipt_sha,
        )


def test_evaluation_transport_round_trip_binds_plan_and_full_inventory(tmp_path):
    source = tmp_path / "evaluation"
    (source / "package").mkdir(parents=True)
    (source / "bundle").mkdir()
    (source / "package/manifest.json").write_text("{}\n")
    (source / "bundle/specification.json").write_text("{}\n")
    (source / "bundle/controls.json").write_text("{}\n")
    plan = {
        "inputs": {
            "package": {"path": "package", "sha256": "x"},
            "bundle": {"path": "bundle", "sha256": "x"},
            "controls": {"path": "bundle/controls.json", "sha256": "x"},
        }
    }
    (source / "plan.json").write_text(json.dumps(plan))
    files = {
        path.relative_to(source).as_posix(): restore_bundle.sha256(path.read_bytes())
        for path in source.rglob("*")
        if path.is_file()
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "capability-runtime-evaluation-bundle-v1",
                "files": files,
                "plan_sha256": restore_bundle.sha256(
                    (source / "plan.json").read_bytes()
                ),
            }
        )
    )
    manifest_sha = restore_bundle.sha256((source / "manifest.json").read_bytes())
    plan_sha = restore_bundle.sha256((source / "plan.json").read_bytes())
    archive, transport = (
        tmp_path / "evaluation.tar.gz",
        tmp_path / "evaluation.transport.json",
    )
    restore_bundle.build_evaluation_transport(
        source, archive, transport, plan_sha256=plan_sha
    )
    destination = tmp_path / "restored"
    restore_bundle.extract_evaluation_transport(
        archive,
        transport,
        destination,
        expected_manifest_sha256=manifest_sha,
        expected_plan_sha256=plan_sha,
    )
    assert (
        restore_bundle.evaluation_members(destination).keys()
        == restore_bundle.evaluation_members(source).keys()
    )
    with pytest.raises(ValueError, match="destination must be new"):
        restore_bundle.extract_evaluation_transport(
            archive,
            transport,
            destination,
            expected_manifest_sha256=manifest_sha,
            expected_plan_sha256=plan_sha,
        )


def test_evaluation_transport_preserves_hash_bound_empty_directories(tmp_path):
    source = tmp_path / "evaluation"
    (source / "bundle/controls/malformed_missing/out").mkdir(parents=True)
    (source / "package").mkdir()
    (source / "package/manifest.json").write_text("{}\n")
    (source / "bundle/controls.json").write_text("{}\n")
    plan = {"inputs": {"controls": {"path": "bundle/controls.json", "sha256": "x"}}}
    (source / "plan.json").write_text(json.dumps(plan))
    files = {
        path.relative_to(source).as_posix(): restore_bundle.sha256(path.read_bytes())
        for path in source.rglob("*")
        if path.is_file()
    }
    directories = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_dir()
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "capability-runtime-evaluation-bundle-v1",
                "files": files,
                "directories": directories,
                "plan_sha256": restore_bundle.sha256(
                    (source / "plan.json").read_bytes()
                ),
            }
        )
    )
    manifest_sha = restore_bundle.sha256((source / "manifest.json").read_bytes())
    plan_sha = restore_bundle.sha256((source / "plan.json").read_bytes())
    archive, transport = tmp_path / "evaluation.tar.gz", tmp_path / "transport.json"
    packed = restore_bundle.build_evaluation_transport(
        source, archive, transport, plan_sha256=plan_sha
    )
    destination = tmp_path / "restored"
    restore_bundle.extract_evaluation_transport(
        archive,
        transport,
        destination,
        expected_manifest_sha256=manifest_sha,
        expected_plan_sha256=plan_sha,
    )
    assert packed["member_count"] == len(files) + len(directories) + 1
    assert (destination / "bundle/controls/malformed_missing/out").is_dir()
    assert restore_bundle.evaluation_members(destination).keys() == (
        restore_bundle.evaluation_members(source).keys()
    )


def test_evaluation_directory_manifest_requires_exact_safe_inventory(tmp_path):
    source = tmp_path / "evaluation"
    (source / "bundle/empty").mkdir(parents=True)
    (source / "plan.json").write_text(
        json.dumps({"inputs": {"plan": {"path": "plan.json"}}})
    )
    files = {"plan.json": restore_bundle.sha256((source / "plan.json").read_bytes())}
    manifest = {
        "schema_version": "capability-runtime-evaluation-bundle-v1",
        "files": files,
        "directories": ["bundle"],
        "plan_sha256": files["plan.json"],
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="directory set differs"):
        restore_bundle.evaluation_members(source)
    manifest["directories"] = ["bundle", "bundle/empty", "../unsafe"]
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe member"):
        restore_bundle.evaluation_members(source)


def test_evaluation_bundle_rejects_symlinked_parent_and_undeclared_symlink(tmp_path):
    source = tmp_path / "evaluation"
    (source / "real-package").mkdir(parents=True)
    (source / "real-package/input.txt").write_text("input")
    (source / "bundle").mkdir()
    (source / "bundle/specification.json").write_text("{}")
    (source / "plan.json").write_text(
        json.dumps({"inputs": {"package": {"path": "package"}}})
    )
    (source / "package").symlink_to(source / "real-package", target_is_directory=True)
    files = {
        "real-package/input.txt": restore_bundle.sha256(b"input"),
        "bundle/specification.json": restore_bundle.sha256(b"{}"),
        "plan.json": restore_bundle.sha256((source / "plan.json").read_bytes()),
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "capability-runtime-evaluation-bundle-v1",
                "files": files,
                "plan_sha256": files["plan.json"],
            }
        )
    )
    with pytest.raises(ValueError, match="symlink"):
        restore_bundle.evaluation_members(source)


@pytest.mark.parametrize("empty_directory", [False, True])
def test_regrade_transport_round_trip_preserves_external_plan_and_rejects_mutation(
    tmp_path, empty_directory,
):
    from capability_pipeline.regrade import create_plan_bundle, validate_plan_bundle

    source = Path(__file__).resolve().parents[1] / "data/c17-repeated-evaluation-003"
    if not source.is_dir():
        pytest.skip("frozen c17 fixture unavailable")
    if empty_directory:
        copied_source = tmp_path / "evaluation"
        shutil.copytree(source, copied_source)
        source = copied_source
        (source / "bundle/controls/initial-empty").mkdir(parents=True)
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["directories"] = sorted(
            path.relative_to(source).as_posix()
            for path in source.rglob("*") if path.is_dir()
        )
        manifest_path.write_text(json.dumps(manifest))
    taskcompendium = tmp_path / "taskcompendium"
    taskcompendium.mkdir()
    (taskcompendium / "pyproject.toml").write_text("fixed")
    (taskcompendium / "uv.lock").write_text("fixed")
    planned = create_plan_bundle(source, taskcompendium, tmp_path / "planned")
    archive = tmp_path / "regrade.tar.gz"
    transport = tmp_path / "regrade.transport.json"
    restore_bundle.build_regrade_transport(
        tmp_path / "planned", archive, transport, plan_sha256=planned["plan_sha256"]
    )
    restored = tmp_path / "restored"
    restore_bundle.extract_regrade_transport(
        archive,
        transport,
        restored,
        expected_manifest_sha256=planned["manifest_sha256"],
        expected_plan_sha256=planned["plan_sha256"],
    )
    assert (
        validate_plan_bundle(restored, taskcompendium, planned["plan_sha256"])[0][
            "cell_count"
        ]
        == 70
    )
    if empty_directory:
        assert (restored / "input/bundle/controls/initial-empty").is_dir()
        (restored / "input/unexpected-empty").mkdir()
        with pytest.raises(ValueError, match="directory set differs"):
            restore_bundle.regrade_members(restored)
        (restored / "input/unexpected-empty").rmdir()
    (restored / "input/bundle/controls.json").write_text("{}")
    with pytest.raises(ValueError, match="member hash mismatch"):
        restore_bundle.regrade_members(restored)
