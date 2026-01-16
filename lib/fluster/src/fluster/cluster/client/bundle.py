# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workspace bundle creation for job submission.

This module provides BundleCreator for packaging workspace directories
into zip files that can be sent to workers.
"""

import subprocess
import tempfile
import zipfile
from pathlib import Path


def _get_git_non_ignored_files(workspace: Path) -> set[Path] | None:
    """Get files that are not ignored by git.

    Uses git ls-files to get both tracked files and untracked files that
    would not be ignored by .gitignore rules.

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
        return {workspace / line for line in result.stdout.splitlines() if line}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class BundleCreator:
    """Helper for creating workspace bundles.

    Bundles a user's workspace directory (containing pyproject.toml, uv.lock,
    and source code) into a zip file for job execution.

    The workspace must already have fluster as a dependency in pyproject.toml.
    If uv.lock doesn't exist, it will be generated.
    """

    def __init__(self, workspace: Path):
        """Initialize bundle creator.

        Args:
            workspace: Path to workspace directory containing pyproject.toml
        """
        self._workspace = workspace

    def create_bundle(self) -> bytes:
        """Create a workspace bundle.

        Creates a zip file containing the workspace directory contents.
        Uses git to determine which files to include (respecting .gitignore),
        falling back to pattern-based exclusion if git is not available.

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
                        if file.is_file() and not self._should_exclude(file):
                            zf.write(file, file.relative_to(self._workspace))
            return bundle_path.read_bytes()

    def _should_exclude(self, path: Path) -> bool:
        """Check if a file should be excluded from the bundle."""
        exclude_patterns = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "*.pyc",
            "*.egg-info",
            ".venv",
            "venv",
            "node_modules",
        }
        parts = path.relative_to(self._workspace).parts
        for part in parts:
            for pattern in exclude_patterns:
                if pattern.startswith("*"):
                    if part.endswith(pattern[1:]):
                        return True
                elif part == pattern:
                    return True
        return False
