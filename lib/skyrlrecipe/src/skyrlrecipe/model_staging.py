# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache identities for native model and tokenizer staging."""

import hashlib
from pathlib import PurePosixPath


def immutable_model_cache_key(source: str, identity: str) -> str:
    """Return a stable cache key for one immutable model source."""
    return hashlib.sha256(f"{source}@{identity}".encode()).hexdigest()


def model_metadata_subpath(source: str, identity: str) -> PurePosixPath:
    """Return the staging path relative to the native task's temporary directory."""
    return PurePosixPath("marinskyrl", "model_metadata", immutable_model_cache_key(source, identity))


def tokenizer_metadata_subpath(source: str, revision: str | None) -> PurePosixPath:
    """Return the staging path for a separately declared tokenizer."""
    return model_metadata_subpath(source, revision or "main")
