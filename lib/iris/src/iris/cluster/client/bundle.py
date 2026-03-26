# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace bundle creation for job submission."""

import atexit
import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum bundle size in bytes (25 MB)
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024

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

# Glob patterns for generated files that are gitignored but required at runtime.
# These are produced by build hooks (e.g. hatch_build.py protobuf generation)
# and must be included in task bundles so that `uv sync` inside containers can
# skip regeneration.
GENERATED_ARTIFACT_GLOBS = [
    "src/iris/rpc/*_pb2.py",
    "src/iris/rpc/*_pb2.pyi",
    "src/iris/rpc/*_connect.py",
    "lib/iris/src/iris/rpc/*_pb2.py",
    "lib/iris/src/iris/rpc/*_pb2.pyi",
    "lib/iris/src/iris/rpc/*_connect.py",
]


def _should_exclude(
    relative: Path,
    extra_dirs: set[str] | None = None,
    extra_extensions: set[str] | None = None,
    extra_subpaths: set[str] | None = None,
) -> bool:
    """Check whether a relative path should be excluded from the bundle.

    Default EXCLUDE_DIRS (e.g. __pycache__, .git) are matched at any depth.
    Caller-supplied extra_dirs are matched only against the top-level directory
    component, mirroring Ray's excludes behavior (e.g. "tests/" excludes only
    the top-level tests directory, not lib/foo/tests/).
    """
    all_extensions = EXCLUDE_EXTENSIONS | (extra_extensions or set())
    all_subpaths = EXCLUDE_SUBPATHS | (extra_subpaths or set())

    if relative.suffix in all_extensions:
        return True
    # e.g. foo.egg-info
    if any(part.endswith(".egg-info") for part in relative.parts):
        return True
    # Default dirs excluded at any depth
    if any(part in EXCLUDE_DIRS for part in relative.parts):
        return True
    # Caller-supplied dirs excluded only at the top level
    if extra_dirs and relative.parts and relative.parts[0] in extra_dirs:
        return True
    rel_str = str(relative)
    return any(subpath in rel_str for subpath in all_subpaths)


def get_git_non_ignored_files(
    workspace: Path,
    *,
    exclude_dirs: set[str] | None = None,
    exclude_extensions: set[str] | None = None,
    exclude_subpaths: set[str] | None = None,
) -> set[Path] | None:
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
        files = [f for f in files if not _should_exclude(f, exclude_dirs, exclude_extensions, exclude_subpaths)]
        return {workspace / f for f in files}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug("Git not available, using pattern-based exclusion: %s", e)
        return None


def include_generated_build_artifacts(
    workspace: Path,
    files: set[Path],
    *,
    exclude_dirs: set[str] | None = None,
    exclude_extensions: set[str] | None = None,
    exclude_subpaths: set[str] | None = None,
) -> None:
    """Add generated build artifacts that exist on disk but are gitignored."""
    added = 0
    for pattern in GENERATED_ARTIFACT_GLOBS:
        for path in workspace.glob(pattern):
            if (
                path.is_file()
                and path not in files
                and not _should_exclude(path.relative_to(workspace), exclude_dirs, exclude_extensions, exclude_subpaths)
            ):
                files.add(path)
                added += 1
    if added:
        logger.debug("Included %d generated build artifact(s) in bundle", added)


def create_workspace_zip(
    workspace: str | Path,
    *,
    exclude_dirs: set[str] | None = None,
    exclude_extensions: set[str] | None = None,
    exclude_subpaths: set[str] | None = None,
    max_size_bytes: int | None = MAX_BUNDLE_SIZE_BYTES,
) -> str:
    """Create a zip of the workspace suitable for Ray's working_dir or Iris bundles.

    Uses git ls-files to determine which files to include (respecting .gitignore),
    then adds back generated protobuf artifacts that are gitignored but needed at
    runtime. When git is unavailable, falls back to pattern-based exclusion.

    Args:
        workspace: Root directory to bundle.
        exclude_dirs: Additional directory names to exclude (merged with defaults).
        exclude_extensions: Additional file extensions to exclude (merged with defaults).
        exclude_subpaths: Additional subpath strings to exclude (merged with defaults).
        max_size_bytes: Maximum allowed zip size. Pass None to disable the check.

    Returns:
        Path to the created zip file.  The file is cleaned up automatically
        at process exit via atexit.  Ray accepts zip file paths directly as
        working_dir, so no intermediate directory is needed.
    """
    workspace = Path(workspace)

    git_files = get_git_non_ignored_files(
        workspace,
        exclude_dirs=exclude_dirs,
        exclude_extensions=exclude_extensions,
        exclude_subpaths=exclude_subpaths,
    )
    if git_files is not None:
        include_generated_build_artifacts(
            workspace,
            git_files,
            exclude_dirs=exclude_dirs,
            exclude_extensions=exclude_extensions,
            exclude_subpaths=exclude_subpaths,
        )

    # Create a temp *file* (not directory) so Ray can use the path directly as
    # working_dir.  Register cleanup on process exit.
    fd, zip_path_str = tempfile.mkstemp(suffix=".zip", prefix="workspace_")
    os.close(fd)
    zip_path = Path(zip_path_str)
    atexit.register(lambda p=zip_path_str: os.unlink(p) if os.path.exists(p) else None)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if git_files is not None:
            for file in git_files:
                if file.is_file():
                    zf.write(file, file.relative_to(workspace))
        else:
            for file in workspace.rglob("*"):
                rel = file.relative_to(workspace)
                if file.is_file() and not _should_exclude(rel, exclude_dirs, exclude_extensions, exclude_subpaths):
                    zf.write(file, rel)

    if max_size_bytes is not None:
        zip_size = zip_path.stat().st_size
        if zip_size > max_size_bytes:
            zip_size_mb = zip_size / (1024 * 1024)
            max_size_mb = max_size_bytes / (1024 * 1024)
            raise ValueError(
                f"Bundle size {zip_size_mb:.1f}MB exceeds maximum {max_size_mb:.0f}MB. "
                "Consider excluding large files or using .gitignore."
            )

    return str(zip_path)


class BundleCreator:
    """Helper for creating workspace bundles for Iris job submission.

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

        Raises:
            ValueError: If bundle size exceeds MAX_BUNDLE_SIZE_BYTES
        """
        zip_path = create_workspace_zip(self._workspace, max_size_bytes=MAX_BUNDLE_SIZE_BYTES)
        return Path(zip_path).read_bytes()
