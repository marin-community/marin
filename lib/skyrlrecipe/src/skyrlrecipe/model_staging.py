# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic paths for launch-time policy and tokenizer metadata."""

import hashlib
from pathlib import PurePosixPath


def immutable_model_cache_key(source: str, identity: str) -> str:
    """Return a stable cache key for one immutable model source."""
    return hashlib.sha256(f"{source}@{identity}".encode()).hexdigest()


def model_metadata_relative_path(source: str, identity: str) -> PurePosixPath:
    """Return the metadata path relative to the worker's temporary directory."""
    return PurePosixPath("marinskyrl", "model_metadata", immutable_model_cache_key(source, identity))
