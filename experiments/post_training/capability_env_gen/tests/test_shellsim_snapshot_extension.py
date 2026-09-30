"""Static and metadata-only tests for the private ShellSim snapshot overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from capability_pipeline import shellsim_snapshot_extension as extension

ROOT = Path(__file__).parents[1]


def test_snapshot_overlay_lock_binds_patch_and_every_changed_bridge_file():
    lock_path = ROOT / extension.LOCK_PATH
    patch_path = ROOT / extension.PATCH_PATH
    lock = json.loads(lock_path.read_text())

    assert lock["schema_version"] == "taskcompendium-shellsim-vfs-snapshot-overlay-v1"
    assert lock["extension_id"] == extension.EXTENSION_ID
    assert lock["base_revision"] == extension.BASE_REVISION
    assert lock["patch_sha256"] == hashlib.sha256(patch_path.read_bytes()).hexdigest()
    assert set(lock["files"]) == {
        extension.MAIN.as_posix(),
        extension.CARGO_TOML.as_posix(),
        extension.CARGO_LOCK.as_posix(),
        extension.README.as_posix(),
    }
    for hashes in lock["files"].values():
        assert set(hashes) == {"base_sha256", "patched_sha256"}
        assert all(len(value) == 64 for value in hashes.values())
        assert "PENDING" not in hashes.values()


def test_snapshot_patch_uses_raw_vfs_nodes_and_documents_complete_bounds():
    patch = (ROOT / extension.PATCH_PATH).read_text()

    snapshot_hunk = patch.split("Request::Snapshot", 1)[1].split("Request::Exec", 1)[0]
    assert "sim.vfs.walk(&root)" in snapshot_hunk
    assert "sim.vfs.raw_get(&absolute)" in snapshot_hunk
    assert "NodeKind::File(bytes)" in snapshot_hunk
    assert "sha256_hex(bytes)" in snapshot_hunk
    assert "read_limited" not in snapshot_hunk
    assert "snapshot entry limit exceeded" in snapshot_hunk
    assert "snapshot file-byte limit exceeded" in snapshot_hunk
    assert "snapshot response limit exceeded" in snapshot_hunk
    assert "NodeKind::Symlink(target)" in snapshot_hunk
    assert 'target: Some(target.clone())' in snapshot_hunk
    assert 'mode: node.mode' in snapshot_hunk
    assert "snapshot_sha256" in snapshot_hunk


def test_extension_record_exposes_only_pinned_provenance():
    record = extension.extension_record()
    assert record["id"] == extension.EXTENSION_ID
    assert record["base_revision"] == extension.BASE_REVISION
    assert record["snapshot_schema"] == "taskcompendium-shellsim-vfs-snapshot-v1"
    assert record["patch_sha256"] == extension.sha256(ROOT / extension.PATCH_PATH)
    assert set(record["bridge_files"]) == {
        extension.MAIN.as_posix(),
        extension.CARGO_TOML.as_posix(),
        extension.CARGO_LOCK.as_posix(),
        extension.README.as_posix(),
    }


def test_validate_base_rejects_any_unpinned_bridge_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    expected = {}
    for relative in (extension.MAIN, extension.CARGO_TOML, extension.CARGO_LOCK, extension.README):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix())
        expected[relative.as_posix()] = {
            "base_sha256": extension.sha256(path), "patched_sha256": "f" * 64,
        }
    monkeypatch.setattr(extension, "_load_lock", lambda: {"files": expected})

    assert extension.validate_base(source) == {"files": expected}
    (source / extension.MAIN).write_text("tampered")
    with pytest.raises(ValueError, match="base differs"):
        extension.validate_base(source)
