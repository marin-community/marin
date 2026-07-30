# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the guarded fsspec factory: the url_to_fs/open_url/filesystem entry
points, unique_temp_path, atomic_rename, and fetch_file_atomic, plus the S3
request bounds those entry points inject."""

import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import pytest
from aiobotocore.config import AioConfig
from botocore.awsrequest import AWSRequest
from botocore.exceptions import ReadTimeoutError
from rigging.filesystem import s3_compat
from rigging.filesystem.cross_region import CrossRegionGuardedFS
from rigging.filesystem.factory import (
    _with_s3_timeout_defaults,
    atomic_rename,
    fetch_file_atomic,
    filesystem,
    open_url,
    unique_temp_path,
    url_to_fs,
)
from rigging.filesystem.s3_compat import (
    TotalDeadlineAIOHTTPSession,
    fsspec_s3_conf,
    s3_request_bounds_config_kwargs,
)


def test_unique_temp_path_produces_distinct_paths():
    """Each call to unique_temp_path returns a different path."""
    paths = {unique_temp_path("/some/output.txt") for _ in range(10)}
    assert len(paths) == 10
    for p in paths:
        assert p.startswith("/some/output.txt.tmp.")


def test_atomic_rename_uses_unique_temp_paths(tmp_path):
    """Concurrent atomic_rename calls use distinct temp paths (UUID collision avoidance)."""
    output = str(tmp_path / "out.txt")
    observed_temps = []

    for _ in range(5):
        with atomic_rename(output) as temp_path:
            observed_temps.append(temp_path)
            Path(temp_path).write_text("data")

    assert len(set(observed_temps)) == 5, "Each call should produce a unique temp path"
    for tp in observed_temps:
        assert ".tmp." in tp


def test_atomic_rename_cleans_up_on_error(tmp_path):
    """Temp file is removed when the context raises an exception."""
    output = str(tmp_path / "out.txt")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_rename(output) as temp_path:
            Path(temp_path).write_text("bad")
            raise RuntimeError("boom")

    assert not Path(temp_path).exists()
    assert not Path(output).exists()


# ---------------------------------------------------------------------------
# fetch_file_atomic
# ---------------------------------------------------------------------------


def test_fetch_file_atomic_copies_source(tmp_path):
    src = tmp_path / "remote" / "tokenizer.json"
    src.parent.mkdir(parents=True)
    src.write_bytes(b'{"version": 1}')
    dest = tmp_path / "cache" / "tokenizer.json"
    dest.parent.mkdir(parents=True)

    assert fetch_file_atomic(str(src), str(dest)) is True
    assert dest.read_bytes() == b'{"version": 1}'


def test_fetch_file_atomic_missing_source_returns_false(tmp_path):
    dest = tmp_path / "cache" / "tokenizer.json"
    dest.parent.mkdir(parents=True)

    assert fetch_file_atomic(str(tmp_path / "remote" / "absent.json"), str(dest)) is False
    assert not dest.exists()


def test_fetch_file_atomic_failure_preserves_dest_and_cleans_temp(tmp_path, monkeypatch):
    # Regression for marin#7167: a fetch that dies mid-finalize must not leave a
    # partial file at dest (poisoning a shared cache) and must not orphan its temp.
    src = tmp_path / "remote" / "tokenizer.json"
    src.parent.mkdir(parents=True)
    src.write_bytes(b'{"version": 2}')
    dest = tmp_path / "cache" / "tokenizer.json"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'{"version": 1}')  # a previous complete file

    def boom(*args, **kwargs):
        raise OSError("simulated failure finalizing the fetch")

    monkeypatch.setattr("os.replace", boom)

    with pytest.raises(OSError, match="simulated failure"):
        fetch_file_atomic(str(src), str(dest))

    assert dest.read_bytes() == b'{"version": 1}'
    assert [p.name for p in dest.parent.iterdir()] == ["tokenizer.json"]


# ---------------------------------------------------------------------------
# Guarded entry point tests
# ---------------------------------------------------------------------------


def test_url_to_fs_does_not_wrap_local(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    fs, _path = url_to_fs(str(test_file))
    assert not isinstance(fs, CrossRegionGuardedFS)


def test_open_url_local_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    result = open_url(str(test_file), "r")
    with result as f:
        assert f.read() == "hello"


def test_filesystem_local():
    fs = filesystem("file")
    assert not isinstance(fs, CrossRegionGuardedFS)


def test_s3_filesystems_are_built_with_the_deadline():
    """Without this wiring the deadline exists but nothing uses it, and every
    rigging-built S3 filesystem keeps the #6719 hang."""
    config_kwargs = _with_s3_timeout_defaults({})["config_kwargs"]

    assert config_kwargs["http_session_cls"] is TotalDeadlineAIOHTTPSession
    # AioConfig is the call s3fs makes; it rejects the kwarg before aiobotocore 2.12.2.
    assert AioConfig(**config_kwargs).http_session_cls is TotalDeadlineAIOHTTPSession


def test_request_to_a_silent_peer_gives_up(monkeypatch):
    """The production failure, reproduced: a peer accepts the whole request body
    and then never answers. Under the sock_* bounds alone no timer is armed and
    the caller waits forever (#6719)."""
    monkeypatch.setattr(s3_compat, "_S3_TOTAL_TIMEOUT", 1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        swallowed = threading.Event()
        held: list[socket.socket] = []

        def swallow_request():
            conn, _ = listener.accept()
            conn.recv(65536)  # take the request, then answer nothing at all
            held.append(conn)
            swallowed.set()

        threading.Thread(target=swallow_request, daemon=True).start()

        # sock_read is deliberately long: it is the bound that does NOT cover
        # this wait, so only `total` can end the call.
        session = TotalDeadlineAIOHTTPSession(timeout=(5, 300))
        request = AWSRequest(method="PUT", url=f"http://127.0.0.1:{port}/part", data=b"x" * 1024).prepare()

        async def send_and_close():
            async with session:
                return await session.send(request)

        started = time.monotonic()
        with pytest.raises(ReadTimeoutError):
            asyncio.run(send_and_close())
        elapsed = time.monotonic() - started

        for conn in held:
            conn.close()

    assert swallowed.is_set(), "peer never received the body; the test did not exercise the deadline"
    # Bounded below too: an immediate setup or wiring failure would also raise.
    assert 0.5 < elapsed < 30, f"expected the 1s deadline to end the call, took {elapsed:.2f}s"


def test_env_config_block_stays_json_serializable():
    """The FSSPEC_S3 block is json.dumps'd into every Iris task's environment.
    Moving the session class into the shared bounds helper would break task
    startup fleet-wide, so guard the boundary."""
    json.dumps(s3_request_bounds_config_kwargs())
    json.dumps(fsspec_s3_conf("http://cwlota.com"))
