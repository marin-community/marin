# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming content hashes for local and object-store files."""

import base64
import hashlib
from dataclasses import dataclass
from enum import StrEnum

import google_crc32c
from fsspec import AbstractFileSystem

from rigging.filesystem.buckets import filesystem_for

HASH_CHUNK_BYTES = 8 * 1024 * 1024


class HashAlgorithm(StrEnum):
    MD5 = "md5"
    CRC32C = "crc32c"


@dataclass(frozen=True)
class FileHashes:
    md5: bytes | None
    crc32c: bytes | None


def file_hashes(url: str, algorithms: frozenset[HashAlgorithm]) -> FileHashes:
    """Calculate ``algorithms`` by streaming the complete contents of ``url``."""
    filesystem, path = filesystem_for(url)
    return file_hashes_for_path(filesystem, path, algorithms)


def file_hashes_for_path(
    filesystem: AbstractFileSystem,
    path: str,
    algorithms: frozenset[HashAlgorithm],
) -> FileHashes:
    """Calculate ``algorithms`` for a path on an already-routed filesystem."""
    md5 = hashlib.md5(usedforsecurity=False) if HashAlgorithm.MD5 in algorithms else None
    crc32c = google_crc32c.Checksum() if HashAlgorithm.CRC32C in algorithms else None
    with filesystem.open(path, "rb") as file:
        while chunk := file.read(HASH_CHUNK_BYTES):
            if md5 is not None:
                md5.update(chunk)
            if crc32c is not None:
                crc32c.update(chunk)
    return FileHashes(
        md5=md5.digest() if md5 is not None else None,
        crc32c=crc32c.digest() if crc32c is not None else None,
    )


def format_digest(digest: bytes | None, *, hexadecimal: bool) -> str:
    """Format a digest as lowercase hexadecimal or RFC 4648 base64."""
    if digest is None:
        return "-"
    if hexadecimal:
        return digest.hex()
    return base64.b64encode(digest).decode()
