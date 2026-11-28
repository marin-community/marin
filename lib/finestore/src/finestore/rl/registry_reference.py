# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read pinned registry documents from their committed, sha-verified references."""

import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field


class RegistryReference(BaseModel):
    """A pinned document: one private registry revision, path and byte hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def read_registry_bytes(reference: RegistryReference) -> bytes:
    """Download the exact registry revision and verify its bytes."""
    downloaded = hf_hub_download(
        repo_id=reference.repo,
        repo_type="dataset",
        revision=reference.revision,
        filename=reference.path,
    )
    data = Path(downloaded).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError(f"Registry bytes changed: {reference.repo}@{reference.revision}:{reference.path}")
    return data


def read_registry_document(reference: RegistryReference) -> dict[str, Any]:
    """Download the exact registry revision and verify its bytes before decoding."""
    document = json.loads(read_registry_bytes(reference))
    if not isinstance(document, dict):
        raise ValueError(f"Registry document must contain an object: {reference.path}")
    return document


def read_registry_json(path: Path) -> dict[str, Any]:
    """Load a committed reference file and fetch the pinned document it names."""
    reference = RegistryReference.model_validate_json(path.read_bytes())
    return read_registry_document(reference)
