# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic TaskTrove archives for importer contract tests."""

import io
import tarfile

from taskcompendium.importers.tasktrove.convert import read_archive
from taskcompendium.importers.tasktrove.models import TaskArchive

FIXTURE_DATASET_URI = "https://example.invalid/tasktrove-fixture"
FIXTURE_REVISION = "synthetic-v1"


def exact_archive() -> TaskArchive:
    """Build a small exact-mode list task with no upstream content."""
    source = "laion__all-puzzles-v2"
    path = "synthetic-exact-task"
    return read_archive(
        _tar_gz(
            {
                "task.toml": (
                    f"""version = "1.0"

[metadata]
tasktrove_source = "{source}"
tasktrove_path = "{path}"
family = "math-answer"
converter = "all_puzzles"
mode = "exact"
tags = ["test", "puzzle", "ordered-list"]
"""
                ),
                "instruction.md": (
                    """# Sorting task

## Puzzle Type
A synthetic ordering task.

## Problem Statement
Sort `green`, `blue` into reverse alphabetical order and return the list.

## Task
Solve the task and write to `/app/answer.txt`.
"""
                ),
                "tests/verifier.toml": (
                    """mode = "exact"
expected = ["green", "blue"]
ignore_case = true
ignore_whitespace = true
ordered = true
output = "/app/answer.txt"
"""
                ),
            }
        ),
        source,
        path,
        FIXTURE_DATASET_URI,
        FIXTURE_REVISION,
    )


def exact_archive_bytes() -> bytes:
    """Build a tarball from the synthetic exact archive's member files."""
    return _tar_gz({name: content.decode() for name, content in exact_archive().files.items()})


def _tar_gz(files: dict[str, str]) -> bytes:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for name, content in files.items():
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return archive_bytes.getvalue()
