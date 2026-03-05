# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import fsspec
import pytest

from iris.mirror_fs import (
    MIRROR_COPY_LIMIT_BYTES,
    MirrorCopyLimitExceeded,
    MirrorFileSystem,
)


class FakeGCS:
    """In-memory fake GCS filesystem for testing MirrorFileSystem.

    Stores files as {path: bytes} where path has no gs:// prefix
    (e.g. "marin-us-east5/models/checkpoint.bin").
    """

    protocol = ("gcs", "gs")

    def __init__(self):
        self._files: dict[str, bytes] = {}

    def add_file(self, path: str, data: bytes) -> None:
        self._files[path] = data

    def exists(self, path: str, **kwargs) -> bool:
        # Check exact match or if path is a "directory" (prefix of some file)
        if path in self._files:
            return True
        prefix = path.rstrip("/") + "/"
        return any(k.startswith(prefix) for k in self._files)

    def size(self, path: str) -> int | None:
        data = self._files.get(path)
        return len(data) if data is not None else None

    def info(self, path: str, **kwargs) -> dict:
        if path not in self._files:
            raise FileNotFoundError(path)
        return {"name": path, "size": len(self._files[path]), "type": "file"}

    def open(self, path: str, mode: str = "rb", **kwargs):
        import io

        if "r" in mode:
            data = self._files.get(path)
            if data is None:
                raise FileNotFoundError(path)
            return io.BytesIO(data)
        else:
            buf = io.BytesIO()
            # Attach a close hook to capture writes
            original_close = buf.close
            _self = self
            _path = path

            def close_and_save():
                _self._files[_path] = buf.getvalue()
                original_close()

            buf.close = close_and_save
            return buf

    def cat_file(self, path: str, start=None, end=None, **kwargs) -> bytes:
        data = self._files.get(path)
        if data is None:
            raise FileNotFoundError(path)
        return data[start:end]

    def copy(self, path1: str, path2: str, **kwargs) -> None:
        data = self._files.get(path1)
        if data is None:
            raise FileNotFoundError(path1)
        self._files[path2] = data

    def ls(self, path: str, detail: bool = True, **kwargs) -> list:
        prefix = path.rstrip("/") + "/"
        matches = [k for k in self._files if k.startswith(prefix) or k == path]
        if not matches:
            raise FileNotFoundError(path)
        if detail:
            return [{"name": m, "size": len(self._files[m]), "type": "file"} for m in matches]
        return matches

    def mkdirs(self, path: str, exist_ok: bool = False) -> None:
        pass

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        pass

    def rm(self, path: str, recursive: bool = False, **kwargs) -> None:
        if recursive:
            prefix = path.rstrip("/") + "/"
            to_remove = [k for k in self._files if k.startswith(prefix) or k == path]
            for k in to_remove:
                del self._files[k]
        else:
            self._files.pop(path, None)

    def rm_file(self, path: str) -> None:
        self._files.pop(path, None)

    def put_file(self, lpath: str, rpath: str, **kwargs) -> None:
        with open(lpath, "rb") as f:
            self._files[rpath] = f.read()


@pytest.fixture()
def fake_gcs():
    return FakeGCS()


@pytest.fixture()
def mirror_fs(fake_gcs, tmp_path):
    """Create a MirrorFileSystem with a fake GCS backend and local lock paths."""
    fs = MirrorFileSystem.__new__(MirrorFileSystem)
    fsspec.AbstractFileSystem.__init__(fs)
    fs._local_bucket = "marin-us-east5"
    fs._all_buckets = ["marin-us-east5", "marin-us-central2", "marin-eu-west4"]
    fs._copy_limit_bytes = MIRROR_COPY_LIMIT_BYTES
    fs._bytes_copied = 0
    fs._holder_id = "test-holder"
    fs._gcs = fake_gcs
    # Override lock path to use local tmp dir for testing
    fs._lock_path_for = lambda path: str(tmp_path / f".mirror_locks/{path}.lock")
    return fs


def test_read_from_local_bucket(mirror_fs, fake_gcs):
    """File in local bucket is served directly without scanning remote."""
    fake_gcs.add_file("marin-us-east5/models/ckpt.bin", b"local-data")

    data = mirror_fs.cat_file("models/ckpt.bin")
    assert data == b"local-data"


def test_read_copies_from_remote_bucket(mirror_fs, fake_gcs):
    """File not in local bucket is found in remote and copied."""
    fake_gcs.add_file("marin-us-central2/models/ckpt.bin", b"remote-data")

    data = mirror_fs.cat_file("models/ckpt.bin")
    assert data == b"remote-data"
    # Should now exist in local bucket too
    assert fake_gcs.exists("marin-us-east5/models/ckpt.bin")
    assert fake_gcs._files["marin-us-east5/models/ckpt.bin"] == b"remote-data"


def test_file_not_found_anywhere(mirror_fs):
    """FileNotFoundError when file doesn't exist in any bucket."""
    with pytest.raises(FileNotFoundError, match="not found in any marin bucket"):
        mirror_fs.cat_file("nonexistent/file.bin")


def test_exists_checks_local_first(mirror_fs, fake_gcs):
    fake_gcs.add_file("marin-us-east5/data/file.txt", b"hello")
    assert mirror_fs.exists("data/file.txt")


def test_exists_checks_remote(mirror_fs, fake_gcs):
    fake_gcs.add_file("marin-eu-west4/data/file.txt", b"hello")
    assert mirror_fs.exists("data/file.txt")


def test_exists_returns_false_when_absent(mirror_fs):
    assert not mirror_fs.exists("no/such/file.txt")


def test_write_goes_to_local_bucket(mirror_fs, fake_gcs):
    """Writes target the local bucket."""
    f = mirror_fs._open("output/result.bin", "wb")
    f.write(b"result-data")
    f.close()
    assert fake_gcs._files["marin-us-east5/output/result.bin"] == b"result-data"


def test_copy_budget_tracks_bytes(mirror_fs, fake_gcs):
    fake_gcs.add_file("marin-us-central2/data/big.bin", b"x" * 1000)
    mirror_fs.cat_file("data/big.bin")
    assert mirror_fs.bytes_copied == 1000


def test_copy_budget_raises_when_exceeded(mirror_fs, fake_gcs):
    mirror_fs._copy_limit_bytes = 500  # Low limit for testing
    fake_gcs.add_file("marin-us-central2/data/big.bin", b"x" * 1000)

    with pytest.raises(MirrorCopyLimitExceeded, match="mirror copy limit"):
        mirror_fs.cat_file("data/big.bin")


def test_copy_budget_cumulative(mirror_fs, fake_gcs):
    mirror_fs._copy_limit_bytes = 1500
    fake_gcs.add_file("marin-us-central2/data/a.bin", b"x" * 800)
    fake_gcs.add_file("marin-us-central2/data/b.bin", b"y" * 800)

    mirror_fs.cat_file("data/a.bin")
    assert mirror_fs.bytes_copied == 800

    with pytest.raises(MirrorCopyLimitExceeded):
        mirror_fs.cat_file("data/b.bin")


def test_open_read_resolves_and_reads(mirror_fs, fake_gcs):
    fake_gcs.add_file("marin-us-east5/data/local.txt", b"local-content")
    with mirror_fs._open("data/local.txt", "rb") as f:
        assert f.read() == b"local-content"


def test_open_read_from_remote(mirror_fs, fake_gcs):
    fake_gcs.add_file("marin-eu-west4/data/remote.txt", b"remote-content")
    with mirror_fs._open("data/remote.txt", "rb") as f:
        assert f.read() == b"remote-content"


def test_protocol_registered():
    """The mirror:// protocol is registered with fsspec."""
    import iris.mirror_fs  # noqa: F401

    assert "mirror" in fsspec.registry


def test_second_read_uses_local_cache(mirror_fs, fake_gcs):
    """After copying from remote, second read uses local bucket."""
    fake_gcs.add_file("marin-us-central2/data/file.bin", b"data")

    # First read: copies from remote
    mirror_fs.cat_file("data/file.bin")
    assert mirror_fs.bytes_copied == 4

    # Remove from remote to prove local is used
    del fake_gcs._files["marin-us-central2/data/file.bin"]

    # Second read: from local, no additional copy
    data = mirror_fs.cat_file("data/file.bin")
    assert data == b"data"
    assert mirror_fs.bytes_copied == 4  # unchanged
