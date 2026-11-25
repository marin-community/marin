# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a pinned Hugging Face dataset file and verify its exact bytes."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class PinnedHfFile:
    """One published file at an immutable dataset revision, with its expected bytes."""

    repo: str
    revision: str
    filename: str
    sha256: str
    byte_size: int

    @property
    def uri(self) -> str:
        return f"hf://datasets/{self.repo}@{self.revision}/{self.filename}"


def read_pinned_hf_file(reference: PinnedHfFile) -> bytes:
    """Download the pinned revision through the cached Hugging Face login and verify it."""
    local = Path(
        hf_hub_download(
            repo_id=reference.repo, repo_type="dataset", revision=reference.revision, filename=reference.filename
        )
    )
    raw = local.read_bytes()
    if len(raw) != reference.byte_size or hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError(f"Pinned file differs from its reference: {reference.uri}")
    return raw
