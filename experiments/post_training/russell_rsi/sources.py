# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical bounded source snapshots for corpus and task preparation."""

import hashlib
import json
import re
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, model_validator

MAX_SNAPSHOT_BYTES = 2_000_000
MAX_SNAPSHOT_FILES = 100


def compact_json_sha256(value: dict) -> str:
    """Hash sorted JSON with compact separators for loop and coding evidence identities."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or ".git" in path.parts or str(path) != value:
        raise ValueError(f"Unsafe snapshot path: {value!r}")
    return value


class SourceSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class SourceSnapshot(BaseModel):
    """Teacher-side source trees and license provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    repository: str
    parent_sha: str
    commit_sha: str
    parent_files: dict[str, str]
    reference_files: dict[str, str]
    license_paths: tuple[str, ...]
    split: SourceSplit

    @model_validator(mode="after")
    def validate_snapshot(self) -> "SourceSnapshot":
        for sha in (self.parent_sha, self.commit_sha):
            if not re.fullmatch(r"[0-9a-f]{40}", sha):
                raise ValueError("Snapshots require full commit SHAs")
        for files in (self.parent_files, self.reference_files):
            for path in files:
                source_path(path)
            if (
                len(files) > MAX_SNAPSHOT_FILES
                or sum(len(text.encode()) for text in files.values()) > MAX_SNAPSHOT_BYTES
            ):
                raise ValueError("Snapshot exceeds the source budget")
            if not self.license_paths or not all(path in files for path in self.license_paths):
                raise ValueError("Snapshots require license files")
        return self


def source_group_id(snapshot: SourceSnapshot) -> str:
    """Identify one source scope at a commit independently of its data split."""
    payload = snapshot.model_dump(mode="json", exclude={"split"})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{snapshot.commit_sha}-{digest[:16]}"
