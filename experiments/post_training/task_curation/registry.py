# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read immutable Atlas registry documents from their pinned private archive."""

import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field


class RegistryReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def read_registry_json(path: Path) -> dict[str, Any]:
    """Download an exact registry revision and verify its bytes before decoding."""
    reference = RegistryReference.model_validate_json(path.read_bytes())
    downloaded = hf_hub_download(
        repo_id=reference.repo,
        repo_type="dataset",
        revision=reference.revision,
        filename=reference.path,
    )
    data = Path(downloaded).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError(f"Registry bytes changed: {reference.repo}@{reference.revision}:{reference.path}")
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ValueError(f"Registry document must contain an object: {reference.path}")
    return document
