"""Pinned Hugging Face and HTTP files with shared object-store staging."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import BinaryIO
from urllib.parse import urlsplit
from urllib.request import urlopen

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import validate_repo_id
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.distributed_lock import create_lock, lease_refresh
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration, ExponentialBackoff

_STAGING_TIMEOUT = Duration.from_minutes(15)
_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_URL_KEY_LENGTH = 16


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
        return _local_hf_file(repo_id, filename, revision=revision)

    destination_path = StoragePath(cache_prefix) / repo_id / revision / filename

    def write_file(writer: BinaryIO) -> None:
        source = _local_hf_file(repo_id, filename, revision=revision)
        with open(source, "rb") as reader:
            shutil.copyfileobj(reader, writer)

    return _stage_file(destination_path, write_file)


def fetch_url(url: str, *, version: str, cache_prefix: str) -> str:
    """Return an HTTP(S) file staged under an explicit local or object-store prefix.

    ``version`` identifies an immutable publisher release, commit, or date using
    letters, digits, dots, underscores, or hyphens. The URL and version identify
    the cache entry; completed entries require no origin request.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Dataset URL must use HTTP or HTTPS")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version) is None:
        raise ValueError("Dataset version must be a release, commit, or date safe for a path component")
    if not cache_prefix:
        raise ValueError("URL datasets require an explicit cache prefix")
    # This identifies the URL without putting query credentials in stored keys.
    url_key = hashlib.sha256(url.encode()).hexdigest()[:_URL_KEY_LENGTH]
    destination_path = StoragePath(cache_prefix) / "urls" / url_key / version / "content"

    def write_file(writer: BinaryIO) -> None:
        with urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as reader:
            content_length = reader.headers.get("Content-Length")
            downloaded = 0
            while chunk := reader.read(_DOWNLOAD_CHUNK_BYTES):
                writer.write(chunk)
                downloaded += len(chunk)
            if content_length is not None and downloaded != int(content_length):
                raise OSError(f"Incomplete HTTP download: expected {content_length} bytes, received {downloaded}")

    return _stage_file(destination_path, write_file)


def _local_hf_file(repo_id: str, filename: str, *, revision: str) -> str:
    cached = try_to_load_from_cache(repo_id, filename, repo_type="dataset", revision=revision)
    if isinstance(cached, str):
        return cached
    return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)


def _stage_file(destination_path: StoragePath, write_file: Callable[[BinaryIO], None]) -> str:
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
        lease.try_acquire, _STAGING_TIMEOUT, f"Timed out staging dataset file {destination}"
    )
    try:
        with lease_refresh(lease):
            if filesystem.exists(complete) and filesystem.exists(path):
                return destination
            with atomic_rename(path, filesystem) as temporary:
                with filesystem.open(temporary, "wb") as writer:
                    write_file(writer)
            # Object-store renames copy then delete. A separate completion
            # marker keeps a failed or interrupted publication out of readers.
            with atomic_rename(complete, filesystem) as temporary:
                with filesystem.open(temporary, "wb") as marker:
                    marker.write(b"")
            return destination
    finally:
        lease.release()
