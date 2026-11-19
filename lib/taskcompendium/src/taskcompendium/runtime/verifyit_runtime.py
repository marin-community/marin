# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage pinned framework source separately from the task's declared environment."""

import hashlib
import json
from pathlib import Path

from taskcompendium.runtime.local import RUNTIME_PACKAGES, context_files
from taskcompendium.runtime.staging import StagedFile

RUNTIME_ROOT = "/tests/.taskcompendium-runtime"
ENTRY_PATH = f"{RUNTIME_ROOT}/entry.py"


def verifyit_runtime_files() -> tuple[list[StagedFile], str]:
    """Return the producer's exact Verifyit files and their manifest digest."""
    files = []
    checksums = {}
    for package in RUNTIME_PACKAGES:
        for metadata in context_files(package):
            name = f"{package.name}/{metadata.path}"
            data = (package / metadata.path).read_bytes()
            if hashlib.sha256(data).hexdigest() != metadata.sha256:
                raise ValueError(f"Verifyit runtime changed while staging: {name}")
            files.append(StagedFile(f"{RUNTIME_ROOT}/{name}", data, int(metadata.mode, 8) & 0o777))
            checksums[name] = metadata.sha256
    entry = Path(__file__).with_name("verifyit_entry.py").read_bytes()
    files.append(StagedFile(ENTRY_PATH, entry))
    checksums["entry.py"] = hashlib.sha256(entry).hexdigest()
    manifest = json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode()
    files.append(StagedFile(f"{RUNTIME_ROOT}/manifest.json", manifest))
    return files, hashlib.sha256(manifest).hexdigest()
