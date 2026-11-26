# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pool verification must not stall concurrent trials or repeat the host probe."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from shellbox.backends.local import disk


@pytest.mark.asyncio
async def test_pool_verification_keeps_loop_responsive_and_reuses_concurrent_result(tmp_path, monkeypatch):
    started = threading.Event()
    released = threading.Event()
    real_stat = disk.os.stat

    def host_stat(path, *args, **kwargs):
        if str(path) == "/proc/self/ns/mnt":
            return SimpleNamespace(st_ino=123)
        return real_stat(path, *args, **kwargs)

    def probe(root):
        started.set()
        if not released.wait(timeout=5):
            raise TimeoutError("Host probe blocked the event loop")
        return SimpleNamespace(root=root)

    # Substitute the privileged host filesystem/probe boundaries, keeping the
    # public asynchronous acquisition and concurrent cache behavior real.
    monkeypatch.setattr(disk.os, "stat", host_stat)
    monkeypatch.setattr(disk, "LocalDiskPool", probe)
    first = asyncio.create_task(disk.verified_local_disk_pool(tmp_path))
    second = asyncio.create_task(disk.verified_local_disk_pool(tmp_path))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        heartbeat = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(heartbeat.set_result, None)
        await heartbeat
        assert not first.done() and not second.done()
    finally:
        released.set()
    first_pool, second_pool = await asyncio.gather(first, second)
    assert first_pool is second_pool
    assert await disk.verified_local_disk_pool(tmp_path / ".") is first_pool
