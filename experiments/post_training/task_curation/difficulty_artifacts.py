# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore relocated grader artifacts without changing frozen TaskSpec paths."""

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.runtime.local import ENVIRONMENT_MANIFEST, FILES_ARCHIVE, LOCK_FILE, load_manifest

REGIONAL_ENVIRONMENT_ROOT = "s3://marin-us-east-02a/marin/images"
RELOCATED_LOCK = "env-278169f50750da36/2026.10.08/requirements.lock"


def regional_lock_url(path: str) -> str:
    """Resolve the recorded relocation of the original QA grader environment."""
    if path.startswith("/") and path.endswith("/" + RELOCATED_LOCK):
        return str(StoragePath(REGIONAL_ENVIRONMENT_ROOT) / RELOCATED_LOCK)
    return path


def restore_original_lock(path: str) -> bool:
    """Restore verified lock siblings at the frozen path; never rewrite the task."""
    regional = regional_lock_url(path)
    if regional == path or not StoragePath(regional).exists():
        return False
    source = StoragePath(regional).parent
    manifest_bytes = (source / ENVIRONMENT_MANIFEST).read_bytes()
    manifest = load_manifest(manifest_bytes.decode())
    files = {LOCK_FILE: StoragePath(regional).read_bytes(), ENVIRONMENT_MANIFEST: manifest_bytes}
    if manifest.files:
        archive = (source / FILES_ARCHIVE).read_bytes()
        if hashlib.sha256(archive).hexdigest() != manifest.files_sha256:
            raise ValueError(f"Regional grader archive differs from its manifest: {source}")
        files[FILES_ARCHIVE] = archive
    destination = Path(path).parent
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        target = destination / name
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError(f"Frozen grader artifact differs from its regional copy: {target}")
            continue
        with NamedTemporaryFile(dir=destination, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        if target.read_bytes() != payload:
            raise ValueError(f"Restored grader artifact changed on readback: {target}")
    return True
