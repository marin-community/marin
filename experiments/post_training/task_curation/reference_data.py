# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read reference indexes that pin private Hugging Face datasets instead of committed
data files.

A reference index pins one dataset revision and lists every file it serves with its
SHA-256; the per-reference pins restate the index pin, and readers verify the two
agree before downloading.
"""

import json
from pathlib import Path
from typing import Any

from finestore.rl.registry_reference import RegistryReference, read_registry_bytes
from pydantic import BaseModel, ConfigDict, Field


class DataReferenceIndex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    references: list[RegistryReference]


def read_reference_index(path: Path) -> DataReferenceIndex:
    return DataReferenceIndex.model_validate_json(path.read_bytes())


def read_referenced_documents(index_path: Path) -> dict[str, bytes]:
    """Download every referenced document at its pinned revision, verifying its bytes."""
    index = read_reference_index(index_path)
    documents: dict[str, bytes] = {}
    for reference in index.references:
        if (reference.repo, reference.revision) != (index.repo, index.revision):
            raise ValueError(f"Reference pin differs from its index pin: {reference.path}")
        documents[reference.path] = read_registry_bytes(reference)
    return documents


def load_referenced_json(index_path: Path) -> dict[str, Any]:
    return {path: json.loads(data) for path, data in read_referenced_documents(index_path).items()}
