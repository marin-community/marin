# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read SHA-pinned files from private Hugging Face datasets."""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field


class DataReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DataReferenceIndex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    references: list[DataReference]


def read_reference_index(path: Path) -> DataReferenceIndex:
    return DataReferenceIndex.model_validate_json(path.read_bytes())


def read_referenced_document(index_path: Path, path: str) -> bytes:
    """Download one referenced document by its index path, verifying its bytes."""
    index = read_reference_index(index_path)
    reference = next((item for item in index.references if item.path == path), None)
    if reference is None:
        raise ValueError(f"Reference index does not list {path}: {index_path}")
    downloaded = hf_hub_download(
        repo_id=reference.repo,
        repo_type="dataset",
        revision=reference.revision,
        filename=reference.path,
    )
    data = Path(downloaded).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError(f"Referenced dataset bytes changed: {reference.repo}@{reference.revision}:{reference.path}")
    return data
