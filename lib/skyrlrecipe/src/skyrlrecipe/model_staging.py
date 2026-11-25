# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model staging paths and Hub mirror records shared by SkyRL launch and comparison."""

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath

DEFAULT_TOKENIZER_REVISION = "main"


@dataclass(frozen=True)
class HuggingFaceModelCache:
    """CPU mirror bound to an immutable commit and its requested ref."""

    model_id: str
    revision: str
    uri: str
    identity: str
    resolved_revision: str


def model_cache_key(source: str, identity: str) -> str:
    """Hash a model source and its revision or artifact identity."""
    return hashlib.sha256(f"{source}@{identity}".encode()).hexdigest()


def model_metadata_relative_path(source: str, identity: str) -> PurePosixPath:
    """Return the metadata path relative to the worker's temporary directory."""
    return PurePosixPath("marinskyrl", "model_metadata", model_cache_key(source, identity))
