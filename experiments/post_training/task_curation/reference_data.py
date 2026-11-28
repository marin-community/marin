# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read reference indexes that pin private Hugging Face datasets instead of committed
data files.

A reference index pins one dataset revision and lists every file it serves with its
SHA-256. Readers download through the registry reference reader and verify bytes
before parsing. The write path that appends new documents lands with the receipt
archive flow that owns it.
"""

import json
from pathlib import Path
from typing import Any

from finestore.rl.registry_reference import RegistryReference, read_registry_bytes
from pydantic import BaseModel, ConfigDict, Field

DataReference = RegistryReference
"""One pinned document in a reference index; the shape is the registry reference."""


class DataReferenceIndex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    references: list[DataReference]


def read_reference_index(path: Path) -> DataReferenceIndex:
    return DataReferenceIndex.model_validate_json(path.read_bytes())


def read_referenced_documents(index_path: Path) -> dict[str, bytes]:
    """Download every referenced document at its pinned revision, verifying its bytes."""
    index = read_reference_index(index_path)
    documents: dict[str, bytes] = {}
    for reference in index.references:
        documents[reference.path] = read_registry_bytes(reference)
    return documents


def load_referenced_json(index_path: Path) -> dict[str, Any]:
    return {path: json.loads(data) for path, data in read_referenced_documents(index_path).items()}


def read_referenced_document(index_path: Path, path: str) -> bytes:
    """Download one referenced document by its index path, verifying its bytes."""
    index = read_reference_index(index_path)
    reference = next((item for item in index.references if item.path == path), None)
    if reference is None:
        raise ValueError(f"Reference index does not list {path}: {index_path}")
    return read_registry_bytes(reference)


def load_referenced_document(index_path: Path, path: str) -> Any:
    return json.loads(read_referenced_document(index_path, path))
