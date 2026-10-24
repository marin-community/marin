# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared route used to bind numerical reports to a serving process."""

import hashlib
from pathlib import Path

import levanter

RUN_PROVENANCE_PATH = "/levanter267/run"


def inference_source_digest():
    """Hash all Levanter Python sources, including the attention/MoE dependencies."""
    root = Path(levanter.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
