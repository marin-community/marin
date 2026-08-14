# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming content hashes for local and object-store files."""

import base64
import hashlib
from dataclasses import dataclass

import google_crc32c

from rigging.filesystem.buckets import filesystem_for

HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class FileHashes:
    md5: bytes | None
    crc32c: bytes | None


def file_hashes(url: str, *, include_md5: bool, include_crc32c: bool) -> FileHashes:
    """Calculate selected hashes by streaming the complete contents of ``url``."""
    filesystem, path = filesystem_for(url)
    md5 = hashlib.md5(usedforsecurity=False) if include_md5 else None
    crc32c = google_crc32c.Checksum() if include_crc32c else None
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
