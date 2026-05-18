# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace bundle creation for job submission."""

import atexit
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum bundle size in bytes (25 MB)
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024

# Default exclude pattern applied to all bundles.  Matches against the
# *relative* path (forward-slash separated, no leading slash).
DEFAULT_EXCLUDE = re.compile(
    r"""
      \.mov$                        # video files
    | \.pyc$                        # bytecode
    | \.egg-info(/|$)               # egg metadata
    | (^|/)__pycache__(/|$)         # pycache at any depth
    | (^|/)\.git(/|$)               # .git at any depth
    | (^|/)\.mypy_cache(/|$)
    | (^|/)\.pytest_cache(/|$)
    | (^|/)\.ruff_cache(/|$)
    | (^|/)\.venv(/|$)
    | (^|/)node_modules(/|$)
    | (^|/)venv(/|$)
    | (^|/)docs/figures(/|$)
    | (^|/)docs/images(/|$)
    | (^|/)docs/reports(/|$)
    | (^|/)docs/static(/|$)
    | (^|/)tests/snapshots(/|$)
    """,
    re.VERBOSE,
)

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


def _should_exclude(relative: str, exclude: re.Pattern[str]) -> bool:
    return bool(exclude.search(relative))


def _merge_exclude(extra: re.Pattern[str] | None) -> re.Pattern[str]:
    if extra is None:
        return DEFAULT_EXCLUDE
    return re.compile(f"(?:{DEFAULT_EXCLUDE.pattern})|(?:{extra.pattern})", re.VERBOSE)


def collect_workspace_files(
    workspace: str | Path,
    *,
    exclude: re.Pattern[str] | None = None,
) -> list[Path]:
    """Collect the list of files to include in a workspace bundle.

    Uses git ls-files when available (respecting .gitignore), then adds back
    generated protobuf artifacts that are gitignored but needed at runtime.
    Falls back to pattern-based exclusion when git is unavailable.

    Args:
        workspace: Root directory to bundle.
        exclude: Extra regex to exclude (merged with DEFAULT_EXCLUDE).

    Returns:
        Sorted list of absolute paths.
    """
    workspace = Path(workspace)
    merged = _merge_exclude(exclude)

    git_files = _get_git_non_ignored_files(workspace, merged)
    if git_files is not None:
        _include_generated_build_artifacts(workspace, git_files, merged)
        return sorted(f for f in git_files if f.is_file())

    return sorted(
        f for f in workspace.rglob("*") if f.is_file() and not _should_exclude(str(f.relative_to(workspace)), merged)
    )


def create_workspace_zip(
    workspace: str | Path,
    *,
    exclude: re.Pattern[str] | None = None,
    max_size_bytes: int | None = MAX_BUNDLE_SIZE_BYTES,
) -> bytes:
    """Create a zip of the workspace and return the raw bytes.

    Suitable for Iris bundle uploads where the caller sends bytes directly.
    """
    workspace = Path(workspace)
    files = collect_workspace_files(workspace, exclude=exclude)

    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="workspace_")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in files:
                zf.write(file, file.relative_to(workspace))
        buf = Path(tmp_path).read_bytes()
    finally:
        os.unlink(tmp_path)

    if max_size_bytes is not None and len(buf) > max_size_bytes:
        size_mb = len(buf) / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise ValueError(
            f"Bundle size {size_mb:.1f}MB exceeds maximum {max_mb:.0f}MB. "
            "Consider excluding large files or using .gitignore."
        )

    return buf


def create_workspace_dir(
    workspace: str | Path,
    *,
    exclude: re.Pattern[str] | None = None,
) -> str:
    """Copy workspace files into a temporary directory for upload.

    The Iris client zips and uploads the directory. The temp directory is
    cleaned up at process exit via atexit.
    """
    workspace = Path(workspace)
    files = collect_workspace_files(workspace, exclude=exclude)

    tmp_dir = tempfile.mkdtemp(prefix="workspace_")
    atexit.register(lambda d=tmp_dir: shutil.rmtree(d, ignore_errors=True))

    for file in files:
        rel = file.relative_to(workspace)
        dest = Path(tmp_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)

    return tmp_dir


def _get_git_non_ignored_files(
    workspace: Path,
    exclude: re.Pattern[str],
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
        files = [f for f in result.stdout.splitlines() if f and not _should_exclude(f, exclude)]
        return {workspace / f for f in files}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug("Git not available, using pattern-based exclusion: %s", e)
        return None


def _include_generated_build_artifacts(
    workspace: Path,
    files: set[Path],
    exclude: re.Pattern[str],
) -> None:
    """Add generated build artifacts that exist on disk but are gitignored."""
    added = 0
    for pattern in GENERATED_ARTIFACT_GLOBS:
        for path in workspace.glob(pattern):
            if path.is_file() and path not in files and not _should_exclude(str(path.relative_to(workspace)), exclude):
                files.add(path)
                added += 1
    if added:
        logger.debug("Included %d generated build artifact(s) in bundle", added)


class BundleCreator:
    """Helper for creating workspace bundles for Iris job submission."""

    def __init__(self, workspace: Path):
        self._workspace = workspace

    def create_bundle(self) -> bytes:
        """Create a workspace bundle.

        Returns:
            Bundle as bytes (zip file contents)

        Raises:
            ValueError: If bundle size exceeds MAX_BUNDLE_SIZE_BYTES
        """
        return create_workspace_zip(self._workspace, max_size_bytes=MAX_BUNDLE_SIZE_BYTES)
