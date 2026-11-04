# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content identity for evaluator files shipped in a Marin runtime bundle."""

import subprocess
from pathlib import Path

from iris.cluster.client.bundle import DEFAULT_EXCLUDE
from rigging.cache import combined_content_hash, file_content_hash


def source_content_hash(root: Path, directories: tuple[str, ...], files: tuple[str, ...]) -> str:
    """Hash declared shipped sources, including local edits and nonignored new files."""
    for directory in directories:
        source = root / directory
        if not source.is_dir():
            raise FileNotFoundError(f"Evaluation runtime source directory is unavailable: {source}")
    for filename in files:
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Evaluation runtime source file is unavailable: {root / filename}")

    if (root / ".git").exists():
        # Match Iris bundle membership: changed tracked files and nonignored new
        # files both ship. Gitignored scratch files do not identify the runtime.
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *directories, *files],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        candidates = {root / filename for filename in result.stdout.split("\0") if filename}
    else:
        # Extracted Iris bundles contain no Git metadata. Their shipped files
        # already reflect Git membership; inspect only the declared source roots.
        candidates = {root / filename for filename in files}
        candidates.update(candidate for directory in directories for candidate in (root / directory).rglob("*"))

    components = []
    for candidate in sorted(candidates):
        relative = candidate.relative_to(root)
        # Optional benchmark packages generate build metadata inside source roots.
        if DEFAULT_EXCLUDE.search(relative.as_posix()) or any(
            part in {"build", "dist"} or part.endswith((".egg-info", ".dist-info")) for part in relative.parts
        ):
            continue
        if candidate.is_file():
            components.append(f"{relative.as_posix()}:{file_content_hash(candidate)}")
    return combined_content_hash(components)
