# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic replacement of a file's whole content."""

import tempfile
from pathlib import Path


def write_atomic(path: Path, content: bytes) -> None:
    """Replace ``path`` with ``content`` through a temporary file in the same directory.

    A reader sees the old file or the new one, never a partial write. The temporary file is removed
    if the write fails.
    """
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete_on_close=False) as temp:
        temp.write(content)
        temp.flush()
        Path(temp.name).replace(path)
