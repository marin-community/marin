# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for guarded fsspec entry points and their S3 request bounds."""

import asyncio
import json
import socket
import threading
import time

import fsspec
import pytest
import rigging.filesystem.s3_compat as s3_compat
from aiobotocore.config import AioConfig
from botocore.awsrequest import AWSRequest
from botocore.exceptions import ReadTimeoutError
from rigging.filesystem.cross_region import CrossRegionGuardedFS
from rigging.filesystem.factory import _with_s3_timeout_defaults, filesystem, open_url, url_to_fs
from rigging.filesystem.listing_cache import configure_listing_cache_defaults
from rigging.filesystem.s3_compat import (
    TotalDeadlineAIOHTTPSession,
    configure_fsspec_s3,
    fsspec_s3_conf,
    s3_request_bounds_config_kwargs,
)


class _ExternallyMutableFileSystem(fsspec.AbstractFileSystem):
    """Object-store fake whose listings can change outside this filesystem instance."""

    protocol = ("externallistings", "gs")
    cachable = False

    def __init__(self, files: set[str], **storage_options: object):
        super().__init__(**storage_options)
        self.files = files

    def ls(self, path: str, detail: bool = True, **_kwargs: object) -> list[dict[str, str | int]] | list[str]:
        path = self._strip_protocol(path)
        try:
            listing = self.dircache[path]
        except KeyError:
            listing: list[dict[str, str | int]] = [
                {"name": name, "size": 0, "type": "file"} for name in sorted(self.files)
            ]
            self.dircache[path] = listing
        return listing if detail else [str(entry["name"]) for entry in listing]


class _ExternallyMutableS3FileSystem(_ExternallyMutableFileSystem):
    protocol = ("externals3listings", "s3")


fsspec.register_implementation("externallistings", _ExternallyMutableFileSystem, clobber=True)
fsspec.register_implementation("externals3listings", _ExternallyMutableS3FileSystem, clobber=True)


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


@pytest.mark.parametrize("entrypoint", ["url_to_fs", "filesystem", "fsspec"])
def test_cloud_filesystem_detects_externally_added_files_by_default(entrypoint):
    files = {"bucket/first"}
    if entrypoint == "url_to_fs":
        fs, path = url_to_fs("externallistings://bucket", files=files)
    elif entrypoint == "filesystem":
        fs, path = filesystem("externallistings", files=files), "bucket"
    else:
        fs, path = fsspec.core.url_to_fs("externallistings://bucket", files=files)

    assert fs.ls(path, detail=False) == ["bucket/first"]

    files.add("bucket/second")

    assert fs.ls(path, detail=False) == ["bucket/first", "bucket/second"]


def test_cloud_filesystem_preserves_explicit_listing_cache_opt_in():
    files = {"bucket/first"}
    fs, path = url_to_fs("externallistings://bucket", files=files, listings_expiry_time=60)
    assert fs.ls(path, detail=False) == ["bucket/first"]

    files.add("bucket/second")

    assert fs.ls(path, detail=False) == ["bucket/first"]


def test_cloud_filesystem_preserves_process_listing_cache_config(monkeypatch):
    monkeypatch.setitem(fsspec.config.conf, "gs", {"listings_expiry_time": 60})
    configure_listing_cache_defaults()
    files = {"bucket/first"}
    fs, path = fsspec.core.url_to_fs("externallistings://bucket", files=files)
    assert fs.ls(path, detail=False) == ["bucket/first"]

    files.add("bucket/second")

    assert fs.ls(path, detail=False) == ["bucket/first"]


def test_configure_fsspec_s3_preserves_process_listing_cache_config(monkeypatch):
    for key in ("AWS_ENDPOINT_URL", "AWS_REGION", "AWS_DEFAULT_REGION", "FSSPEC_S3"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(fsspec.config.conf, "s3", {"listings_expiry_time": 60})
    configure_fsspec_s3("https://objects.example.com")
    files = {"bucket/first"}
    fs, path = fsspec.core.url_to_fs("externals3listings://bucket", files=files)
    assert fs.ls(path, detail=False) == ["bucket/first"]

    files.add("bucket/second")

    assert fs.ls(path, detail=False) == ["bucket/first"]


def test_s3_filesystems_are_built_with_the_deadline():
    """Without this wiring the deadline exists but nothing uses it, and every
    rigging-built S3 filesystem keeps the #6719 hang."""
    config_kwargs = _with_s3_timeout_defaults({})["config_kwargs"]

    assert config_kwargs["http_session_cls"] is TotalDeadlineAIOHTTPSession
    # AioConfig is the call s3fs makes; it rejects the kwarg before aiobotocore 2.12.2.
    assert AioConfig(**config_kwargs).http_session_cls is TotalDeadlineAIOHTTPSession


def test_upload_part_to_a_peer_that_withholds_100_continue_gives_up(monkeypatch):
    """The production failure, reproduced (#6719).

    botocore sends `Expect: 100-continue` on UploadPart, so aiohttp waits for the
    interim response before writing the body. That wait is covered by none of
    aiobotocore's scalar bounds -- sock_read cannot arm because no read is in
    flight -- so a peer that withholds `100 Continue` hangs the caller forever.
    `total` is the only bound that ends it.
    """
    monkeypatch.setattr(s3_compat, "_S3_TOTAL_TIMEOUT", 1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        headers_seen = threading.Event()
        body_bytes: list[int] = []
        held: list[socket.socket] = []

        def withhold_continue():
            conn, _ = listener.accept()
            held.append(conn)
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            body_bytes.append(len(buf.split(b"\r\n\r\n", 1)[1]))
            headers_seen.set()  # never send `100 Continue`, never respond, never close

        threading.Thread(target=withhold_continue, daemon=True).start()

        # sock_read is deliberately long: it is not the bound that ends this call.
        session = TotalDeadlineAIOHTTPSession(timeout=(5, 300))
        request = AWSRequest(
            method="PUT",
            url=f"http://127.0.0.1:{port}/part",
            headers={"Expect": "100-continue", "Content-Length": "1024"},
            data=b"x" * 1024,
        ).prepare()

        async def send_and_close():
            async with session:
                return await session.send(request)

        started = time.monotonic()
        with pytest.raises(ReadTimeoutError):
            asyncio.run(send_and_close())
        elapsed = time.monotonic() - started

        for conn in held:
            conn.close()

    assert headers_seen.is_set(), "peer never got the headers; the test did not exercise the deadline"
    assert body_bytes == [0], f"body must still be unsent while awaiting 100 Continue, got {body_bytes}"
    # Bounded below too: an immediate setup or wiring failure would also raise.
    assert 0.5 < elapsed < 30, f"expected the 1s deadline to end the call, took {elapsed:.2f}s"


def test_env_config_block_stays_json_serializable():
    """The FSSPEC_S3 block is json.dumps'd into every Iris task's environment.
    Moving the session class into the shared bounds helper would break task
    startup fleet-wide, so guard the boundary."""
    json.dumps(s3_request_bounds_config_kwargs())
    json.dumps(fsspec_s3_conf("http://cwlota.com"))
