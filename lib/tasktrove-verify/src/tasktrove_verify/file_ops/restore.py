# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore task-supplied files before execution."""

import shutil
from collections.abc import Sequence
from pathlib import Path

from tasktrove_verify.grade import InvalidTask


def restore(entries: Sequence[str], tests_dir: Path, workspace: Path) -> None:
    """Copy each entry from ``tests_dir`` over the workspace, undoing agent edits to the tests."""
    for entry in entries:
        source = tests_dir / entry
        destination = workspace / entry
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            raise InvalidTask(f"restore entry {entry!r} is not in the tests directory")
