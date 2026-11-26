# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded regular-file reads for verifier inputs and result artifacts."""

import errno
import os
import stat
from pathlib import Path

MAX_ARTIFACT_BYTES = 1_000_000


def read_regular_bytes(path: Path) -> bytes:
    """Read at most 1 MB without following the file or its immediate parent link."""
    try:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError("artifact parent is not a real directory") from error
        raise
    try:
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError("artifact file is a symbolic link") from error
            raise
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact is not a bounded regular file")
        payload = source.read(MAX_ARTIFACT_BYTES + 1)
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds the size limit")
    return payload


def read_text(path: Path, *, characters: int = -1, errors: str = "strict") -> str:
    """Read text under the caller's decoding policy, optionally retaining a character prefix."""
    with path.open(errors=errors) as source:
        return source.read(characters)
