# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Blocking controller RPCs must leave capacity for ordinary RPCs."""

import asyncio
import threading

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.rpc.async_adapter import AsyncServiceAdapter, BoundedThreadExecutor
from rigging.server_auth import VerifiedIdentity, get_verified_identity, identity_scope


class _BlockingService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.executed_timeouts: list[int] = []

    def exec_in_container(self, timeout_seconds: int) -> tuple[int, VerifiedIdentity | None]:
        self.executed_timeouts.append(timeout_seconds)
        identity = get_verified_identity()
        self.started.set()
        try:
            assert self.release.wait(timeout=5), "blocking fake was not released"
            return timeout_seconds, identity
        finally:
            self.finished.set()

    def list_peers(self) -> str:
        return "reachable"


def test_blocking_exec_keeps_ordinary_rpc_responsive_and_preserves_context():
    async def scenario() -> None:
        service = _BlockingService()
        executor = BoundedThreadExecutor(max_workers=1, max_pending=0, thread_name_prefix="test-exec")
        adapter = AsyncServiceAdapter(service, isolated_methods={"exec_in_container": executor})
        identity = VerifiedIdentity(user_id="test-user", role="admin")
        try:
            with identity_scope(identity):
                exec_call = asyncio.create_task(adapter.exec_in_container(-1))
                assert await asyncio.to_thread(service.started.wait, 2)
                assert not exec_call.done()
                assert await asyncio.wait_for(adapter.list_peers(), 2) == "reachable"
                service.release.set()
                assert await asyncio.wait_for(exec_call, 2) == (-1, identity)
        finally:
            service.release.set()
            executor.shutdown()

    asyncio.run(scenario())


def test_cancelled_exec_keeps_capacity_until_worker_finishes():
    async def scenario() -> None:
        service = _BlockingService()
        executor = BoundedThreadExecutor(max_workers=1, max_pending=0, thread_name_prefix="test-exec")
        adapter = AsyncServiceAdapter(service, isolated_methods={"exec_in_container": executor})
        try:
            exec_call = asyncio.create_task(adapter.exec_in_container(60))
            assert await asyncio.to_thread(service.started.wait, 2)
            exec_call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await exec_call
            with pytest.raises(ConnectError) as error:
                await adapter.exec_in_container(60)
            assert error.value.code == Code.RESOURCE_EXHAUSTED

            service.release.set()
            assert await asyncio.to_thread(service.finished.wait, 2)
        finally:
            service.release.set()
            executor.shutdown()

    asyncio.run(scenario())


def test_cancelled_queued_exec_frees_capacity_without_running():
    async def scenario() -> None:
        service = _BlockingService()
        executor = BoundedThreadExecutor(max_workers=1, max_pending=1, thread_name_prefix="test-exec")
        adapter = AsyncServiceAdapter(service, isolated_methods={"exec_in_container": executor})
        try:
            first = asyncio.create_task(adapter.exec_in_container(1))
            assert await asyncio.to_thread(service.started.wait, 2)
            queued = asyncio.create_task(adapter.exec_in_container(2))
            await asyncio.sleep(0)  # Let the queued call submit before checking saturation.
            with pytest.raises(ConnectError) as error:
                await adapter.exec_in_container(3)
            assert error.value.code == Code.RESOURCE_EXHAUSTED

            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            replacement = asyncio.create_task(adapter.exec_in_container(4))
            service.release.set()
            assert (await asyncio.wait_for(first, 2))[0] == 1
            assert (await asyncio.wait_for(replacement, 2))[0] == 4
            assert service.executed_timeouts == [1, 4]
        finally:
            service.release.set()
            executor.shutdown()

    asyncio.run(scenario())
