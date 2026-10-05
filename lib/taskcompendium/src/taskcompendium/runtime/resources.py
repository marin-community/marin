# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inline resources used by the optional episode runtime."""

from taskcompendium.environment import EnvironmentFile


def inline_resource(path: str, data: bytes) -> EnvironmentFile:
    """Build a file from a source path relative to the filesystem root."""
    return EnvironmentFile(path="/" + path.lstrip("/"), content=data)
