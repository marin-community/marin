# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage immutable task files in a shared host cache."""

import fcntl
import hashlib
import os
import tempfile
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.environment import EnvironmentAsset

ASSET_CACHE = Path(tempfile.gettempdir()) / f"rolloutengine-assets-{os.getuid()}"
COPY_CHUNK_BYTES = 1024**2
MAX_CACHE_BYTES = 1024**3


def cached_asset(asset: EnvironmentAsset, cache: Path = ASSET_CACHE) -> Path:
    """Return a host path whose bytes match the declared asset identity."""
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = cache / asset.sha256
    with (cache / f"{asset.sha256}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if target.stat().st_size != asset.size_bytes or digest != asset.sha256:
                raise ValueError(f"Cached task asset differs from its identity: {asset.path}")
            return target
        with (cache / ".capacity.lock").open("a") as capacity_lock:
            fcntl.flock(capacity_lock, fcntl.LOCK_EX)
            used = sum(path.stat().st_size for path in cache.iterdir() if path.is_file())
            if used + asset.size_bytes > MAX_CACHE_BYTES:
                raise ValueError("Task asset cache exceeds its 1 GiB host budget")
            return _download_asset(asset, cache, target)


def _download_asset(asset: EnvironmentAsset, cache: Path, target: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="asset-", dir=cache) as directory:
        staging = Path(directory) / "content"
        digest = hashlib.sha256()
        written = 0
        with StoragePath(asset.uri).open("rb") as source, staging.open("wb") as destination:
            while chunk := source.read(min(COPY_CHUNK_BYTES, asset.size_bytes - written + 1)):
                written += len(chunk)
                if written > asset.size_bytes:
                    raise ValueError(f"Task asset exceeds its declared size: {asset.path}")
                digest.update(chunk)
                destination.write(chunk)
        if written != asset.size_bytes or digest.hexdigest() != asset.sha256:
            raise ValueError(f"Task asset differs from its declared identity: {asset.path}")
        staging.rename(target)
    return target
