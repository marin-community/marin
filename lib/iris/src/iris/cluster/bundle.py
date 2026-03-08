# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical bundle interfaces and implementations for controller/worker/runtime."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Protocol
from urllib.request import urlopen

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.marin_fs import url_to_fs

logger = logging.getLogger(__name__)

BUNDLE_ID_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_bundle_id(bundle_id: str) -> None:
    """Validate a bundle identifier."""
    if not BUNDLE_ID_RE.fullmatch(bundle_id):
        raise ConnectError(Code.INVALID_ARGUMENT, f"Invalid bundle_id: {bundle_id!r}")


def bundle_id_for_zip(blob: bytes) -> str:
    """Return canonical bundle id for zip bytes."""
    return hashlib.sha256(blob).hexdigest()


class BundleStore(Protocol):
    """Bundle storage contract shared across controller/worker/runtime."""

    def write_zip(self, blob: bytes) -> str: ...

    def get_zip(self, bundle_id: str) -> bytes: ...

    def get_bundle(self, bundle_id: str) -> Path: ...

    def prefetch_bundle(self, bundle_id: str) -> None: ...


class ControllerBundleStore:
    """Content-addressed bundle storage used by the controller."""

    def __init__(self, bundle_prefix: str):
        self._prefix = bundle_prefix.rstrip("/")
        self._lock = threading.Lock()

    @property
    def prefix(self) -> str:
        return self._prefix

    def bundle_uri_for_id(self, bundle_id: str) -> str:
        validate_bundle_id(bundle_id)
        return f"{self._prefix}/{bundle_id}.zip"

    def write_zip(self, blob: bytes) -> str:
        bundle_id = bundle_id_for_zip(blob)
        bundle_uri = self.bundle_uri_for_id(bundle_id)
        with self._lock:
            try:
                fs, path = url_to_fs(bundle_uri)
                if fs.exists(path):
                    return bundle_id
                fs.makedirs(str(Path(path).parent), exist_ok=True)
                with fs.open(path, "wb") as f:
                    f.write(blob)
            except Exception as e:
                raise ConnectError(Code.INTERNAL, f"Failed to store bundle {bundle_id}: {e}") from e
        return bundle_id

    def get_zip(self, bundle_id: str) -> bytes:
        validate_bundle_id(bundle_id)
        bundle_uri = self.bundle_uri_for_id(bundle_id)
        try:
            fs, path = url_to_fs(bundle_uri)
            if not fs.exists(path):
                raise ConnectError(Code.NOT_FOUND, f"Bundle not found: {bundle_id}")
            with fs.open(path, "rb") as f:
                return f.read()
        except ConnectError:
            raise
        except Exception as e:
            raise ConnectError(Code.INTERNAL, f"Failed to read bundle {bundle_id}: {e}") from e

    def get_bundle(self, bundle_id: str) -> Path:
        """Return local filesystem path to bundle zip when available."""
        bundle_uri = self.bundle_uri_for_id(bundle_id)
        fs, path = url_to_fs(bundle_uri)
        if fs.protocol not in {"file", ("file", "local"), ("local", "file"), "local"}:
            raise ConnectError(Code.UNIMPLEMENTED, "get_bundle is only supported for local bundle stores")
        return Path(path)

    def prefetch_bundle(self, bundle_id: str) -> None:
        # Controller store is canonical source; fetch validates existence.
        self.get_zip(bundle_id)


class LocalBundleStore:
    """Filesystem-backed worker bundle store with controller fetch on miss."""

    _EVICTION_GRACE_SECONDS = 300

    def __init__(
        self,
        cache_dir: Path,
        controller_address: str | None = None,
        max_bundles: int = 100,
    ):
        self._cache_dir = cache_dir
        self._controller_address = controller_address.rstrip("/") if controller_address else ""
        self._bundles_dir = cache_dir / "bundles"
        self._extracts_dir = cache_dir / "extracts"
        self._max_bundles = max_bundles
        self._extract_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

        self._bundles_dir.mkdir(parents=True, exist_ok=True)
        self._extracts_dir.mkdir(parents=True, exist_ok=True)

    def _zip_path(self, bundle_id: str) -> Path:
        return self._bundles_dir / f"{bundle_id}.zip"

    def _extract_path(self, bundle_id: str) -> Path:
        return self._extracts_dir / bundle_id

    def _download_url(self, bundle_id: str) -> str:
        if not self._controller_address:
            raise ConnectError(Code.FAILED_PRECONDITION, "controller address is required for bundle fetches")
        return f"{self._controller_address}/bundles/{bundle_id}.zip"

    def write_zip(self, blob: bytes) -> str:
        bundle_id = bundle_id_for_zip(blob)
        zip_path = self._zip_path(bundle_id)
        if not zip_path.exists():
            zip_path.write_bytes(blob)
        self._validate_hash(zip_path, bundle_id)
        return bundle_id

    def get_zip(self, bundle_id: str) -> bytes:
        validate_bundle_id(bundle_id)
        zip_path = self._zip_path(bundle_id)
        if not zip_path.exists():
            self._download(bundle_id, zip_path)
        self._validate_hash(zip_path, bundle_id)
        return zip_path.read_bytes()

    def prefetch_bundle(self, bundle_id: str) -> None:
        self.get_bundle(bundle_id)

    def get_bundle(self, bundle_id: str) -> Path:
        validate_bundle_id(bundle_id)
        extract_path = self._extract_path(bundle_id)
        zip_path = self._zip_path(bundle_id)

        with self._extract_locks[bundle_id]:
            if extract_path.exists():
                extract_path.touch()
                return extract_path

            if not zip_path.exists():
                self._download(bundle_id, zip_path)
            else:
                self._validate_hash(zip_path, bundle_id)

            self._extract(zip_path, extract_path)
            self._evict_old_bundles()
            return extract_path

    def _download(self, bundle_id: str, local_path: Path) -> None:
        url = self._download_url(bundle_id)
        try:
            with urlopen(url, timeout=120) as resp:
                data = resp.read()
        except Exception as e:
            raise ConnectError(Code.UNAVAILABLE, f"failed to fetch bundle {bundle_id} from controller: {e}") from e

        local_path.write_bytes(data)
        self._validate_hash(local_path, bundle_id)

    @staticmethod
    def _validate_hash(path: Path, expected_bundle_id: str) -> None:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_bundle_id:
            raise ValueError(f"Bundle hash mismatch: {actual} != {expected_bundle_id}")

    def _extract(self, zip_path: Path, extract_path: Path) -> None:
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            base = extract_path.resolve()
            for member in zf.namelist():
                member_path = (extract_path / member).resolve()
                if not member_path.is_relative_to(base):
                    raise ValueError(f"Zip slip detected: {member} attempts to write outside extract path")
            zf.extractall(extract_path)

    def _evict_old_bundles(self) -> None:
        extracts = list(self._extracts_dir.iterdir())
        if len(extracts) <= self._max_bundles:
            return

        extracts.sort(key=lambda p: p.stat().st_mtime)
        for path in extracts[: len(extracts) - self._max_bundles]:
            if time.time() - path.stat().st_mtime < self._EVICTION_GRACE_SECONDS:
                break
            bundle_id = path.name
            with self._extract_locks[bundle_id]:
                if not path.exists():
                    continue
                if time.time() - path.stat().st_mtime < self._EVICTION_GRACE_SECONDS:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                zip_path = self._zip_path(bundle_id)
                if zip_path.exists():
                    zip_path.unlink()


def stage_bundle_to_local(
    *,
    bundle_id: str,
    workdir: Path,
    workdir_files: dict[str, bytes],
    bundle_store: BundleStore,
) -> None:
    """Fetch bundle and materialize it plus workdir files in workdir."""
    bundle_path = bundle_store.get_bundle(bundle_id)
    if bundle_path.suffix == ".zip":
        extract_dir = workdir / ".iris_bundle_extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path, "r") as zf:
            base = extract_dir.resolve()
            for member in zf.namelist():
                member_path = (extract_dir / member).resolve()
                if not member_path.is_relative_to(base):
                    raise ValueError(f"Zip slip detected: {member} attempts to write outside extract path")
            zf.extractall(extract_dir)
        shutil.copytree(extract_dir, workdir, dirs_exist_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
    else:
        shutil.copytree(bundle_path, workdir, dirs_exist_ok=True)

    for name, data in workdir_files.items():
        path = workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
