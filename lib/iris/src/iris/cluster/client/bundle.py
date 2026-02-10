# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace bundle creation for job submission."""

import logging
import subprocess
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

EXCLUDE_EXTENSIONS = {".mov", ".pyc"}

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
    "venv",
}

EXCLUDE_SUBPATHS = {
    "docs/figures",
    "docs/images",
    "docs/reports",
    "docs/static",
    "tests/snapshot",
}


def _should_exclude(relative: Path) -> bool:
    """Check whether a relative path should be excluded from the bundle."""
    if relative.suffix in EXCLUDE_EXTENSIONS:
        return True
    # e.g. foo.egg-info
    if any(part.endswith(".egg-info") for part in relative.parts):
        return True
    if any(part in EXCLUDE_DIRS for part in relative.parts):
        return True
    rel_str = str(relative)
    return any(subpath in rel_str for subpath in EXCLUDE_SUBPATHS)


def _get_git_non_ignored_files(workspace: Path) -> set[Path] | None:
    """Get files that are not ignored by git.

    Returns None if git is not available or this isn't a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [Path(f) for f in result.stdout.splitlines() if f]
        files = [f for f in files if not _should_exclude(f)]
        return {workspace / f for f in files}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug("Git not available, using pattern-based exclusion: %s", e)
        return None


class BundleCreator:
    """Helper for creating workspace bundles.

    Bundles a user's workspace directory (containing pyproject.toml, uv.lock,
    and source code) into a zip file for job execution.

    The workspace must already have iris as a dependency in pyproject.toml.
    If uv.lock doesn't exist, it will be generated.
    """

    def __init__(self, workspace: Path):
        self._workspace = workspace

    def create_bundle(self) -> bytes:
        """Create a workspace bundle.

        Returns:
            Bundle as bytes (zip file contents)
        """
        git_files = _get_git_non_ignored_files(self._workspace)

        with tempfile.TemporaryDirectory(prefix="bundle_") as td:
            bundle_path = Path(td) / "bundle.zip"
            with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
                if git_files is not None:
                    for file in git_files:
                        if file.is_file():
                            zf.write(file, file.relative_to(self._workspace))
                else:
                    for file in self._workspace.rglob("*"):
                        rel = file.relative_to(self._workspace)
                        if file.is_file() and not _should_exclude(rel):
                            zf.write(file, rel)
            return bundle_path.read_bytes()
