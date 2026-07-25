# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install a verified CUDA PJRT plugin into an NGC task overlay."""

import hashlib
import os
from pathlib import Path

import fsspec

COPY_CHUNK_BYTES = 8 << 20


def copy_artifact(source: str, destination: Path, expected_sha256: str) -> str:
    """Copy an artifact through fsspec and atomically install it after hash verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    digest = hashlib.sha256()
    with fsspec.open(source, "rb") as source_handle, temporary.open("wb") as destination_handle:
        while chunk := source_handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
            destination_handle.write(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        temporary.unlink()
        raise ValueError(f"artifact SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    os.replace(temporary, destination)
    return actual_sha256
