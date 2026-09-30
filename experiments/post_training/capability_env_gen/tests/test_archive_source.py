import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "scripts/archive_source.py"
SPEC = importlib.util.spec_from_file_location("archive_source", MODULE)
archive_source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive_source)


def test_snapshot_binds_checklists_audits_extension_and_actual_launch_inputs(tmp_path):
    root = tmp_path / "root"
    files = {
        "capability_pipeline/synthesis.py": "controller",
        "vendor/task_spec/composite_extension.lock.json": "extension lock",
        "vendor/task_spec/patches/composite_required_extension.patch": "patch",
        "docs/build_acceptance_001.md": "shared checklist",
        "docs/build_acceptance/proposal.md": "exact checklist",
        "docs/audits/c05_c30_construction_001.md": "measured audit",
        "docs/partial_controls.md": "partial controls",
        "uv.lock": "dependencies",
        "new_catalog.json": "new catalog fixture",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    (root / "capability_pipeline/._synthesis.py").write_text("metadata")
    source = tmp_path / "source"
    source.mkdir()
    (source / "accepted.json").write_text("[]")
    (source / "manifest.json").write_text('{"lineage":"fixture"}')
    (source / "daytona-health.json").write_text('{"state":"passed"}')
    (source / "models.template.yml").write_text("secret fixture must not be copied")
    pilot = tmp_path / "alternate-pilot.json"
    pilot.write_text('{"sample":"alternate"}')
    payload, manifest, encoded = archive_source.build_archive(
        root, source=source, pilot=pilot
    )
    assert set(files) <= manifest["files"].keys()
    assert (
        manifest["files"]["inputs/source/accepted.json"]
        == hashlib.sha256(b"[]").hexdigest()
    )
    assert "inputs/source/manifest.json" in manifest["files"]
    assert "inputs/source/daytona-health.json" in manifest["files"]
    assert (
        manifest["files"]["inputs/pilot.json"]
        == hashlib.sha256(pilot.read_bytes()).hexdigest()
    )
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        assert set(archive.getnames()) == set(manifest["files"]) | {"manifest.json"}
        assert all(not item.pax_headers for item in archive.getmembers())
        assert not any(
            "._" in name or "models.template" in name for name in archive.getnames()
        )
        for name, expected in manifest["files"].items():
            assert (
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
            )
        assert json.loads(archive.extractfile("manifest.json").read()) == json.loads(
            encoded
        )


def test_snapshot_rejects_external_source_symlink(tmp_path):
    root = tmp_path / "root"
    (root / "capability_pipeline").mkdir(parents=True)
    outside = tmp_path / "private"
    outside.write_text("must not archive")
    (root / "capability_pipeline/unrelated.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink or external"):
        archive_source.build_archive(root)


def test_snapshot_rejects_stale_launch_contract_before_upload(tmp_path):
    root, source = tmp_path / "root", tmp_path / "source"
    (root / "docs").mkdir(parents=True)
    source.mkdir()
    contract = root / "docs/task_contract.md"
    contract.write_text("frozen contract")
    (source / "accepted.json").write_text("[]")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "accepted_sha256": hashlib.sha256(b"[]").hexdigest(),
                "required_contract_inputs": {
                    "docs/task_contract.md": hashlib.sha256(
                        contract.read_bytes()
                    ).hexdigest()
                },
            }
        )
    )
    archive_source.build_archive(root, source=source)
    contract.write_text("changed contract")
    with pytest.raises(ValueError, match="launch contract input differs"):
        archive_source.build_archive(root, source=source)
    contract.write_text("frozen contract")
    (source / "accepted.json").write_text("[{}]")
    with pytest.raises(ValueError, match="accepted-input digest differs"):
        archive_source.build_archive(root, source=source)


@pytest.mark.parametrize(
    "schema_version",
    [
        "capability-portable-runtime-bundle-v1",
        "capability-portable-runtime-bundle-v2",
    ],
)
def test_snapshot_complete_input_requires_and_archives_exact_bundle(
    tmp_path, schema_version
):
    root, source = tmp_path / "root", tmp_path / "source"
    (root / "capability_pipeline").mkdir(parents=True)
    (root / "capability_pipeline/controller.py").write_text("controller")
    (source / "restore-seed/items/x").mkdir(parents=True)
    (source / "accepted.json").write_text("[]")
    (source / "restore-seed/items/x/status.json").write_text("{}")
    nested = source / "restore-seed/items/x/harbor/manifest.json"
    nested.parent.mkdir(parents=True)
    nested.write_text('{"step_names":["one"]}')
    declared = {
        path.relative_to(source).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "files": declared,
            }
        )
    )

    payload, manifest, _ = archive_source.build_archive(
        root, source=source, complete_input=True
    )

    expected = {f"inputs/source/{name}" for name in declared} | {
        "inputs/source/manifest.json"
    }
    assert expected <= set(manifest["files"])
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        assert expected <= set(archive.getnames())

    (source / "restore-seed/items/x/status.json").write_text("changed")
    with pytest.raises(ValueError, match="member hash mismatch"):
        archive_source.build_archive(root, source=source, complete_input=True)


def test_snapshot_complete_evaluation_bundle_never_requires_accepted_json(tmp_path):
    root, source = tmp_path / "root", tmp_path / "evaluation"
    (root / "capability_pipeline").mkdir(parents=True)
    (root / "capability_pipeline/evaluation.py").write_text("controller")
    (source / "package").mkdir(parents=True)
    (source / "bundle").mkdir()
    (source / "package/manifest.json").write_text("{}")
    (source / "bundle/specification.json").write_text("{}")
    (source / "bundle/controls.json").write_text("{}")
    (source / "plan.json").write_text(
        json.dumps({"inputs": {"package": {"path": "package"}}})
    )
    files = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "capability-runtime-evaluation-bundle-v1",
                "files": files,
                "plan_sha256": hashlib.sha256((source / "plan.json").read_bytes()).hexdigest(),
            }
        )
    )
    _, manifest, _ = archive_source.build_archive(root, source=source, complete_input=True)
    assert "inputs/source/plan.json" in manifest["files"]
    assert "inputs/source/accepted.json" not in manifest["files"]
