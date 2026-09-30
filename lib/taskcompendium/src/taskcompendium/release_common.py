# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared identifiers and file digests for local public release builders."""

import hashlib
from pathlib import Path

REPO_ID = "open-athena/taskcompendium-alpha-1"


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a local release input or output file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
