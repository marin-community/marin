# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate task output selections before capture and grader staging."""

from collections.abc import Mapping
from pathlib import PurePosixPath

from shellbox.machine import Machine
from shellbox.transfer import download_files

from taskcompendium.models import normalized_absolute_path, validate_output_paths

MAX_OUTPUT_FILES = 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def selected_output_files(paths: tuple[str, ...], files: Mapping[str, bytes]) -> dict[str, bytes]:
    """Select declared files and descendants, validating evidence before staging it."""
    validate_output_paths(paths)
    roots = tuple(PurePosixPath(path) for path in paths)
    selected = {}
    total = 0
    for path, data in files.items():
        candidate = PurePosixPath(path)
        if not any(candidate.is_relative_to(root) for root in roots):
            continue
        normalized_absolute_path(path)
        validate_output_paths((path,))
        selected[path] = data
        total += len(data)
        if len(selected) > MAX_OUTPUT_FILES or total > MAX_OUTPUT_BYTES:
            raise RuntimeError("Output capture exceeds its file or byte budget")
    return selected


async def captured_output_files(
    machine: Machine, paths: tuple[str, ...], *, timeout: float | None, limit_bytes: int
) -> dict[str, bytes]:
    """Capture declared output files and directories without private grading resources."""
    validate_output_paths(paths)
    files = await download_files(
        machine,
        paths,
        timeout=timeout,
        max_files=MAX_OUTPUT_FILES,
        max_bytes=MAX_OUTPUT_BYTES,
        max_file_bytes=limit_bytes,
    )
    selected = selected_output_files(paths, files)
    if selected != files:
        raise ValueError("Downloaded files fall outside the declared output paths")
    return selected
