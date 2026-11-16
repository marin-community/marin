import asyncio
import copy

import fsspec.config
import pytest
import s3fs
from botocore.exceptions import ReadTimeoutError
from rigging.filesystem import s3_compat
from upath import UPath

from harbor.utils import s3_request_timeout


@pytest.mark.asyncio
async def test_s3_write_times_out_when_server_never_responds(monkeypatch):
    request_received = asyncio.Event()
    release_server = asyncio.Event()

    async def stalled_upload(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            request_received.set()
            await release_server.wait()
        finally:
            writer.close()
            await writer.wait_closed()

    previous = copy.deepcopy(fsspec.config.conf.get("s3", {}))
    server = await asyncio.start_server(stalled_upload, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(s3_compat, "_S3_TOTAL_TIMEOUT", 0.2)
    monkeypatch.setitem(
        fsspec.config.conf,
        "s3",
        {
            "key": "test-key",
            "secret": "test-secret",
            "client_kwargs": {"endpoint_url": f"http://127.0.0.1:{port}"},
            "config_kwargs": {
                "connect_timeout": 30,
                "read_timeout": 120,
                "retries": {"max_attempts": 0},
                "s3": {"addressing_style": "path"},
            },
        },
    )
    s3_request_timeout.configure_s3_request_timeout()
    destination = UPath("s3://test-bucket/acp-launch.sh")
    write = asyncio.create_task(
        asyncio.to_thread(destination.write_text, "#!/bin/sh\n")
    )
    try:
        # Use the same plain UPath close/commit path as Harbor's artifact writes.
        done, _ = await asyncio.wait({write}, timeout=10)
        assert write in done, "The S3 client's deadline did not end the write"
        with pytest.raises(ReadTimeoutError):
            write.result()
        assert request_received.is_set()
    finally:
        release_server.set()
        await asyncio.gather(write, return_exceptions=True)
        server.close()
        await server.wait_closed()
        await destination.fs.s3._endpoint.http_session.close()
        s3fs.S3FileSystem.clear_instance_cache()
        fsspec.config.conf["s3"] = previous
