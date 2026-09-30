# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming content hashes for local and object-store files."""

import hashlib

from fsspec import AbstractFileSystem

from rigging.filesystem.buckets import filesystem_for

HASH_CHUNK_BYTES = 8 * 1024 * 1024


def file_md5(url: str) -> bytes:
    """Calculate an MD5 digest by streaming the complete contents of ``url``."""
    filesystem, path = filesystem_for(url)
    return file_md5_for_path(filesystem, path)


def file_md5_for_path(filesystem: AbstractFileSystem, path: str) -> bytes:
    """Calculate an MD5 digest for a path on an already-routed filesystem."""
    md5 = hashlib.md5(usedforsecurity=False)
    with filesystem.open(path, "rb") as file:
        while chunk := file.read(HASH_CHUNK_BYTES):
            md5.update(chunk)
    return md5.digest()

