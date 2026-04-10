# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Per-rollout repo working tree management for SWE-ZERO.

A ``WorkTree`` is a local directory containing a checkout of a SWE-rebench V2
PR's repo at the ``base_commit``, with the ``test_patch`` applied so the new
test files are visible to the agent. The agent's bash subprocess runs against
this directory.

The worktree is materialized in this priority order:

1. **Pre-built GCS cache** (created by ``clone_repos.py`` Zephyr pipeline):
   ``gs://<bucket>/swe_zero/repo_cache/<repo>/<short_sha>.tar.gz``.
2. **Lazy GitHub clone** as a fallback when the cache is missing.

Either way the result is a unique per-rollout directory under ``cache_root``
that gets cleaned up when the rollout finishes. We never share a worktree
between concurrent rollouts (each gets its own checkout) so ``sed -i`` /
``git apply`` from one trace does not contaminate another.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import fsspec

from experiments.swe_zero.data_loader import PRRecord

logger = logging.getLogger(__name__)

DEFAULT_GCS_CACHE_ROOT = "gs://marin-us-central2/swe_zero/repo_cache"
DEFAULT_LOCAL_CACHE_ROOT = "/tmp/swe_zero_worktrees"


def _short_sha(sha: str) -> str:
    return sha[:12]


def _safe_repo_name(repo: str) -> str:
    """Filesystem-safe repo name (org__name)."""
    return repo.replace("/", "__")


def gcs_cache_key(repo: str, base_commit: str, gcs_root: str = DEFAULT_GCS_CACHE_ROOT) -> str:
    """Return the canonical GCS path for a (repo, commit) tarball."""
    return f"{gcs_root}/{_safe_repo_name(repo)}/{_short_sha(base_commit)}.tar.gz"


@dataclass
class WorkTree:
    """A materialized per-rollout repo checkout that supports cleanup."""

    path: Path
    repo: str
    base_commit: str
    instance_id: str
    _temp_dir: tempfile._TemporaryFileWrapper | None = field(default=None, repr=False)

    def cleanup(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> WorkTree:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


def _gcs_exists(path: str) -> bool:
    fs, _ = fsspec.core.url_to_fs(path)
    return fs.exists(path)


def _extract_tarball_to(tarball_local: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "xzf", str(tarball_local), "-C", str(target), "--strip-components=1"],
        check=True,
    )


def _fetch_from_gcs_cache(gcs_path: str, target: Path) -> bool:
    """Download a tarball from GCS and extract it. Returns False on miss."""
    if not _gcs_exists(gcs_path):
        return False
    logger.info("Fetching cached worktree from %s", gcs_path)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        local_tar = Path(tmp.name)
    try:
        with fsspec.open(gcs_path, "rb") as src, open(local_tar, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        _extract_tarball_to(local_tar, target)
    finally:
        local_tar.unlink(missing_ok=True)
    return True


def _clone_from_github(repo: str, base_commit: str, target: Path) -> None:
    """Lazy fallback when the GCS cache is missing.

    Uses a shallow ``git init`` + ``git fetch <sha>`` so we don't pull the full
    history. This requires the host to allow upload-pack of arbitrary commits,
    which GitHub does for unauthenticated public repos.
    """
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    logger.info("Lazy cloning %s @ %s into %s", repo, _short_sha(base_commit), target)
    subprocess.run(["git", "init", "--quiet"], cwd=target, check=True)
    subprocess.run(["git", "remote", "add", "origin", url], cwd=target, check=True)
    subprocess.run(
        ["git", "fetch", "--quiet", "--depth=1", "origin", base_commit],
        cwd=target,
        check=True,
    )
    subprocess.run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=target, check=True)


def _apply_test_patch(worktree: Path, test_patch: str) -> None:
    """Apply the test_patch so the new test files are visible to the agent."""
    if not test_patch or not test_patch.strip():
        return
    patch_file = worktree / ".swe_zero_test.patch"
    patch_file.write_text(test_patch)
    try:
        result = subprocess.run(
            ["git", "apply", "--allow-empty", "--whitespace=nowarn", str(patch_file)],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Fall back to plain ``patch`` for diffs that ``git apply`` rejects
            # (e.g. test_patch references files outside index).
            patch_proc = subprocess.run(
                ["patch", "-p1", "--forward", "-i", str(patch_file)],
                cwd=worktree,
                capture_output=True,
                text=True,
            )
            if patch_proc.returncode != 0:
                logger.warning(
                    "Failed to apply test_patch (git apply: %s, patch: %s)",
                    result.stderr.strip(),
                    patch_proc.stderr.strip(),
                )
    finally:
        patch_file.unlink(missing_ok=True)


def materialize_worktree(
    pr: PRRecord,
    *,
    local_cache_root: str | Path = DEFAULT_LOCAL_CACHE_ROOT,
    gcs_cache_root: str | None = DEFAULT_GCS_CACHE_ROOT,
) -> WorkTree:
    """Build a fresh worktree directory for ``pr`` and return a handle.

    Each call returns a unique directory; callers must call ``cleanup`` (or
    use the worktree as a context manager) when the rollout finishes.
    """
    local_root = Path(local_cache_root)
    local_root.mkdir(parents=True, exist_ok=True)

    workdir = local_root / f"{_safe_repo_name(pr.repo)}__{_short_sha(pr.base_commit)}__{uuid.uuid4().hex[:8]}"

    fetched_from_cache = False
    if gcs_cache_root:
        try:
            fetched_from_cache = _fetch_from_gcs_cache(
                gcs_cache_key(pr.repo, pr.base_commit, gcs_cache_root),
                workdir,
            )
        except Exception as e:
            logger.warning("GCS cache fetch failed for %s @ %s: %s", pr.repo, pr.base_commit, e)

    if not fetched_from_cache:
        _clone_from_github(pr.repo, pr.base_commit, workdir)

    _apply_test_patch(workdir, pr.test_patch)

    return WorkTree(
        path=workdir,
        repo=pr.repo,
        base_commit=pr.base_commit,
        instance_id=pr.instance_id,
    )
