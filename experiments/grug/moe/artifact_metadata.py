# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
