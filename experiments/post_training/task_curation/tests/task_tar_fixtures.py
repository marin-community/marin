# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic task-tar builders shared by the curation tests."""

import gzip
import io
import tarfile

DEFAULT_MODE = 0o644


def pack_tar(members: dict[str, bytes | tuple[bytes, int]], *, gz: bool = True) -> bytes:
    """Pack tar members in insertion order, gzipped with mtime=0 unless ``gz`` is false."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, member in members.items():
            content, mode = member if isinstance(member, tuple) else (member, DEFAULT_MODE)
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = mode
            archive.addfile(info, io.BytesIO(content))
    raw = buffer.getvalue()
    return gzip.compress(raw, mtime=0) if gz else raw
