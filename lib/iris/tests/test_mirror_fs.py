# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

import fsspec
import pytest

from iris.mirror_fs import (
    MIRROR_COPY_LIMIT_BYTES,
    MirrorCopyLimitExceeded,
    MirrorFileSystem,
)


@pytest.fixture()
def mirror_env(tmp_path):
    """Set up a local mirror FS environment with fake regional dirs.

    Creates directories mimicking marin regional buckets on local disk,
    and patches marin_prefix to point at the "local" region.
    """
    local_dir = tmp_path / "marin-local"
    local_dir.mkdir()
    remote1 = tmp_path / "marin-us-central2"
    remote1.mkdir()
    remote2 = tmp_path / "marin-eu-west4"
    remote2.mkdir()
    return {
        "local_dir": local_dir,
        "remote1": remote1,
        "remote2": remote2,
        "local_prefix": str(local_dir),
        "remote_prefixes": [str(remote1), str(remote2)],
    }


@pytest.fixture()
def mirror_fs(mirror_env, tmp_path):
    """Create a MirrorFileSystem backed by local directories."""
    fs = MirrorFileSystem.__new__(MirrorFileSystem)
    fsspec.AbstractFileSystem.__init__(fs)
    fs._local_prefix = mirror_env["local_prefix"]
    fs._remote_prefixes = mirror_env["remote_prefixes"]
    fs._copy_limit_bytes = MIRROR_COPY_LIMIT_BYTES
    fs._bytes_copied = 0
    fs._holder_id = "test-holder"
    # Override lock paths to use local tmp dir
    lock_dir = str(tmp_path / "locks")
    fs._lock_path_for = lambda path: os.path.join(lock_dir, f"{path.replace('/', '_')}.lock")
    return fs


def _write_file(base_dir, rel_path, data):
    """Write data to base_dir/rel_path, creating parents."""
    full = os.path.join(str(base_dir), rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)


def test_read_from_local(mirror_fs, mirror_env):
    """File in local prefix is served directly without scanning remote."""
    _write_file(mirror_env["local_dir"], "models/ckpt.bin", b"local-data")
    data = mirror_fs.cat_file("models/ckpt.bin")
    assert data == b"local-data"


def test_read_copies_from_remote(mirror_fs, mirror_env):
    """File not in local prefix is found in remote and copied to local."""
    _write_file(mirror_env["remote1"], "models/ckpt.bin", b"remote-data")

    data = mirror_fs.cat_file("models/ckpt.bin")
    assert data == b"remote-data"
    # Should now exist locally too
    local_path = os.path.join(str(mirror_env["local_dir"]), "models/ckpt.bin")
    with open(local_path, "rb") as f:
        assert f.read() == b"remote-data"


def test_file_not_found_raises(mirror_fs):
    """FileNotFoundError when file doesn't exist in any prefix."""
    with pytest.raises(FileNotFoundError, match="not found in any marin bucket"):
        mirror_fs.cat_file("nonexistent/file.bin")


def test_copy_budget_raises_when_exceeded(mirror_fs, mirror_env):
    mirror_fs._copy_limit_bytes = 500
    _write_file(mirror_env["remote1"], "data/big.bin", b"x" * 1000)

    with pytest.raises(MirrorCopyLimitExceeded, match="mirror copy limit"):
        mirror_fs.cat_file("data/big.bin")


def test_copy_budget_cumulative(mirror_fs, mirror_env):
    """Budget is enforced across multiple copies."""
    mirror_fs._copy_limit_bytes = 1500
    _write_file(mirror_env["remote1"], "data/a.bin", b"x" * 800)
    _write_file(mirror_env["remote1"], "data/b.bin", b"y" * 800)

    mirror_fs.cat_file("data/a.bin")
    assert mirror_fs.bytes_copied == 800

    with pytest.raises(MirrorCopyLimitExceeded):
        mirror_fs.cat_file("data/b.bin")


def test_second_read_uses_local_cache(mirror_fs, mirror_env):
    """After copying from remote, second read uses local cache without re-copying."""
    _write_file(mirror_env["remote1"], "data/file.bin", b"data")

    mirror_fs.cat_file("data/file.bin")
    assert mirror_fs.bytes_copied == 4

    # Remove from remote to prove local is used
    os.remove(os.path.join(str(mirror_env["remote1"]), "data/file.bin"))

    data = mirror_fs.cat_file("data/file.bin")
    assert data == b"data"
    assert mirror_fs.bytes_copied == 4  # unchanged


def test_read_finds_file_in_second_remote(mirror_fs, mirror_env):
    """File only in the second remote prefix is still found and copied."""
    _write_file(mirror_env["remote2"], "data/remote2.txt", b"from-remote2")

    data = mirror_fs.cat_file("data/remote2.txt")
    assert data == b"from-remote2"
