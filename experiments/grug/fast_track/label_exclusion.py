# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed label groups that must stay outside a classifier's evaluation pool."""

from typing import Literal

from marin.execution.fingerprint import register_fingerprint
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath


class LabelExclusion(BaseModel):
    """Label content identity and the union of train, development, and audit groups.

    A scorer without fitted labels must declare an empty group set explicitly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    label_revision: str = Field(min_length=1)
    duplicate_groups: frozenset[str]
    normalized_document_ids: frozenset[str] = frozenset()
    normalized_id_scheme: Literal["xxh3_128_utf8"] = "xxh3_128_utf8"

    @model_validator(mode="after")
    def validate_groups(self) -> "LabelExclusion":
        if any(
            len(group) != 64 or any(char not in "0123456789abcdef" for char in group) for group in self.duplicate_groups
        ):
            raise ValueError("label duplicate groups must be SHA-256 hashes of normalized UTF-8 text")
        if any(
            len(document_id) != 32 or any(char not in "0123456789abcdef" for char in document_id)
            for document_id in self.normalized_document_ids
        ):
            raise ValueError("excluded normalized document IDs must use the xxh3_128 UTF-8 content ID scheme")
        return self

    def identity(self) -> dict:
        """Return a stable identity that includes the complete group set."""
        return {
            "label_revision": self.label_revision,
            "duplicate_groups": sorted(self.duplicate_groups),
            "normalized_document_ids": sorted(self.normalized_document_ids),
            "normalized_id_scheme": self.normalized_id_scheme,
        }


register_fingerprint(LabelExclusion, LabelExclusion.identity)


def read_label_exclusion(path: str) -> LabelExclusion:
    """Read a local or object-storage exclusion manifest."""
    return LabelExclusion.model_validate_json(StoragePath(path).read_text())
