"""Reproducible private overlay for the bounded ShellSim VFS snapshot bridge."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

EXTENSION_ID = "taskcompendium-shellsim-vfs-snapshot-v1"
BASE_REVISION = "dc6b501c8604bcd2e3c20c1e9947679845fdfef8"
BRIDGE_ROOT = Path("shellsim-bridge")
MAIN = BRIDGE_ROOT / "src/main.rs"
CARGO_TOML = BRIDGE_ROOT / "Cargo.toml"
CARGO_LOCK = BRIDGE_ROOT / "Cargo.lock"
README = BRIDGE_ROOT / "README.md"
LOCK_PATH = Path("vendor/task_spec/shellsim_snapshot_extension.lock.json")
PATCH_PATH = Path("vendor/task_spec/patches/shellsim_vfs_snapshot_extension.patch")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lock() -> dict:
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / LOCK_PATH).read_text())
    if not isinstance(value, dict) or value.get("extension_id") != EXTENSION_ID:
        raise ValueError("ShellSim snapshot extension lock is invalid")
    if value.get("base_revision") != BASE_REVISION:
        raise ValueError("ShellSim snapshot extension revision differs")
    if value.get("patch") != PATCH_PATH.as_posix():
        raise ValueError("ShellSim snapshot extension patch path differs")
    if value.get("patch_sha256") != sha256(root / PATCH_PATH):
        raise ValueError("ShellSim snapshot extension patch hash differs")
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != {
        MAIN.as_posix(), CARGO_TOML.as_posix(), CARGO_LOCK.as_posix(), README.as_posix()
    }:
        raise ValueError("ShellSim snapshot extension lock files differ")
    return value


def validate_base(source: Path) -> dict:
    """Validate the unmodified TaskCompendium bridge against the extension base."""
    lock = _load_lock()
    for relative, hashes in lock["files"].items():
        path = source / relative
        if not isinstance(hashes, dict) or set(hashes) != {"base_sha256", "patched_sha256"}:
            raise ValueError(f"invalid snapshot extension hashes: {relative}")
        if not path.is_file() or sha256(path) != hashes["base_sha256"]:
            raise ValueError(f"ShellSim snapshot extension base differs: {relative}")
    return lock


def apply_overlay(source: Path, destination: Path) -> Path:
    """Copy an exact base and apply the one pinned patch with no in-place mutation."""
    source, destination = Path(source), Path(destination)
    lock = validate_base(source)
    if destination.exists():
        raise FileExistsError(f"ShellSim snapshot overlay destination exists: {destination}")
    shutil.copytree(source, destination, symlinks=True)
    patch = Path(__file__).resolve().parents[1] / PATCH_PATH
    import subprocess

    completed = subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
        cwd=destination,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        shutil.rmtree(destination, ignore_errors=True)
        raise ValueError(f"ShellSim snapshot patch failed: {completed.stderr.strip()}")
    try:
        for relative, hashes in lock["files"].items():
            if sha256(destination / relative) != hashes["patched_sha256"]:
                raise ValueError(f"ShellSim snapshot overlay hash differs: {relative}")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def extension_record() -> dict:
    """Return the immutable provenance record consumers bind into reset evidence."""
    lock = _load_lock()
    return {
        "id": EXTENSION_ID,
        "base_revision": BASE_REVISION,
        "patch_sha256": lock["patch_sha256"],
        "bridge_files": {
            relative: hashes["patched_sha256"] for relative, hashes in lock["files"].items()
        },
        "snapshot_schema": "taskcompendium-shellsim-vfs-snapshot-v1",
    }
