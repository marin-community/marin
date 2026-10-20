"""Pinned Hugging Face files with optional shared object-store staging."""

from __future__ import annotations

import re
import shutil
from pathlib import PurePosixPath

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import validate_repo_id
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.distributed_lock import create_lock, lease_refresh
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration, ExponentialBackoff

_STAGING_TIMEOUT = Duration.from_minutes(15)


def fetch_hf(repo_id: str, filename: str, *, revision: str, cache_prefix: str | None = None) -> str:
    """Return a pinned dataset file as a local path or fsspec-readable URL.

    Without ``cache_prefix``, use Hugging Face's local cache. An explicit local
    or object-store prefix stages the file under ``repo_id/revision/filename``.
    Completed shared entries can be read without contacting Hugging Face.
    ``revision`` must be the full commit SHA; mutable branches are not accepted.
    """
    validate_repo_id(repo_id)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Hugging Face dataset revision must be a full lowercase commit SHA")
    relative = PurePosixPath(filename)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != filename or filename == ".":
        raise ValueError(f"Dataset filename must be a normalized relative path: {filename!r}")
    if cache_prefix is None:
        return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)

    destination_path = StoragePath(cache_prefix) / repo_id / revision / filename
    destination = str(destination_path)
    filesystem, path = url_to_fs(destination)
    complete = f"{path}.complete"
    if filesystem.exists(complete) and filesystem.exists(path):
        return destination

    destination_path.parent.mkdirs(exist_ok=True)
    # Lease paths must retain their scheme so rigging selects the native
    # conditional-write implementation for S3 and GCS.
    lease = create_lock(f"{destination}.lock")
    ExponentialBackoff().wait_until_or_raise(
        lease.try_acquire, _STAGING_TIMEOUT, f"Timed out staging Hugging Face dataset file {destination}"
    )
    try:
        with lease_refresh(lease):
            if filesystem.exists(complete) and filesystem.exists(path):
                return destination
            source = hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)
            with atomic_rename(path, filesystem) as temporary:
                with open(source, "rb") as reader, filesystem.open(temporary, "wb") as writer:
                    shutil.copyfileobj(reader, writer)
            # Object-store renames copy then delete. A separate completion
            # marker keeps a failed or interrupted publication out of readers.
            with atomic_rename(complete, filesystem) as temporary:
                with filesystem.open(temporary, "wb") as marker:
                    marker.write(b"")
            return destination
    finally:
        lease.release()
