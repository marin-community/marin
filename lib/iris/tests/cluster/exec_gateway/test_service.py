# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket
import sys
from contextlib import asynccontextmanager

import pytest
import uvicorn
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.exec_gateway.auth import ExecPodTarget, exec_audience, mint_exec_grant
from iris.cluster.exec_gateway.main import gateway_application
from iris.cluster.exec_gateway.service import ExecGatewayService
from iris.rpc import exec_gateway_pb2 as rpc
from iris.rpc.exec_gateway_connect import ExecGatewayServiceClient
from iris.testing.exec_gateway import PausedCloseConnector, ProcessExecConnector
from rigging.token_authority import JwksVerifier, JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem


@pytest.fixture
def keypair():
    return generate_ed25519_keypair()


@pytest.fixture
def signer(keypair):
    return JwtSigner(signing_key_from_private_pem(keypair.private_pem), issuer="test-controller")


@pytest.fixture
def verifier(keypair):
    return JwksVerifier(
        issuers={"test-controller": [keypair.public_pem]},
        expected_audiences=frozenset({exec_audience("gateway")}),
        leeway_seconds=0,
    )


def grant(signer, command, *, grant_id="grant-one", container="task"):
    return mint_exec_grant(
        signer,
        gateway_id="gateway",
        grant_id=grant_id,
        subject="alice",
        target=ExecPodTarget("iris", "pod", "uid", container, "attempt"),
        command=[sys.executable, "-u", "-c", command],
    )


@asynccontextmanager
async def running_gateway(verifier, *, connector=None, **kwargs):
    connector = connector or ProcessExecConnector()
    service = ExecGatewayService(verifier, connector, **kwargs)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        application = gateway_application(service)
        async with application.router.lifespan_context(application):
            server = uvicorn.Server(uvicorn.Config(application, log_level="error", lifespan="off", log_config=None))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                async with ExecGatewayServiceClient(address=f"http://127.0.0.1:{listener.getsockname()[1]}") as client:
                    yield client, connector
            finally:
                server.should_exit = True
                await serving


async def read_to_exit(client, token, session_id):
    stdout, stderr = bytearray(), bytearray()
    async with asyncio.timeout(10):
        while True:
            result = await client.receive(rpc.ReceiveRequest(grant=token, session_id=session_id, wait_ms=1000))
            stdout.extend(result.stdout)
            stderr.extend(result.stderr)
            if result.exited:
                return bytes(stdout), bytes(stderr), result


@pytest.mark.asyncio
async def test_gateway_connect_session_streams_binary_stdio_and_exit_status(signer, verifier):
    token = grant(
        signer,
        "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data[::-1]); "
        "sys.stderr.buffer.write(b'problem'); sys.exit(7)",
    )
    async with running_gateway(verifier) as (client, connector):
        opened = await client.open(rpc.OpenRequest(grant=token))
        await client.send(rpc.SendRequest(grant=token, session_id=opened.session_id, stdin=b"a\x00\xffb"))
        await client.send(rpc.SendRequest(grant=token, session_id=opened.session_id, stdin=b"c", eof=True))
        stdout, stderr, result = await read_to_exit(client, token, opened.session_id)
        assert stdout == b"cb\xff\x00a"
        assert stderr == b"problem"
        assert result.exit_code == 7 and not result.error
        assert connector.targets[0].container_name == "task"
        await client.close_session(rpc.CloseRequest(grant=token, session_id=opened.session_id))
        with pytest.raises(ConnectError) as consumed:
            await client.open(rpc.OpenRequest(grant=token))
        assert consumed.value.code == Code.ALREADY_EXISTS


@pytest.mark.asyncio
async def test_gateway_rejects_foreign_plane_forgery_and_cross_container_session(signer, verifier):
    token = grant(signer, "import sys; sys.stdin.buffer.read()")
    other = grant(signer, "import sys; sys.stdin.buffer.read()", grant_id="other", container="service-api")
    foreign = signer.mint({"sub": "alice"}, audience="iris", ttl_seconds=60)
    async with running_gateway(verifier) as (client, connector):
        opened = await client.open(rpc.OpenRequest(grant=token))
        for invalid in [foreign, token[:-8] + "bad"]:
            with pytest.raises(ConnectError) as rejected:
                await client.open(rpc.OpenRequest(grant=invalid))
            assert rejected.value.code == Code.UNAUTHENTICATED
        with pytest.raises(ConnectError) as cross:
            await client.send(rpc.SendRequest(grant=other, session_id=opened.session_id, stdin=b"bad"))
        assert cross.value.code == Code.PERMISSION_DENIED
        assert len(connector.processes) == 1
        await client.close_session(rpc.CloseRequest(grant=token, session_id=opened.session_id))
        assert connector.processes[0].returncode is not None


@pytest.mark.asyncio
async def test_gateway_bounds_sessions_output_and_stdin_and_cleans_processes(signer, verifier):
    token = grant(signer, "import sys; sys.stdin.buffer.read(1); sys.stdout.buffer.write(b'x'*8192); sys.stdout.flush()")
    other = grant(signer, "pass", grant_id="another")
    async with running_gateway(verifier, max_sessions=1, max_buffer_bytes=4096) as (client, connector):
        opened = await client.open(rpc.OpenRequest(grant=token))
        with pytest.raises(ConnectError) as full:
            await client.open(rpc.OpenRequest(grant=other))
        assert full.value.code == Code.RESOURCE_EXHAUSTED
        with pytest.raises(ConnectError) as large:
            await client.send(rpc.SendRequest(grant=token, session_id=opened.session_id, stdin=b"x" * 65537))
        assert large.value.code == Code.RESOURCE_EXHAUSTED
        await client.send(rpc.SendRequest(grant=token, session_id=opened.session_id, stdin=b"x"))
        _, _, result = await read_to_exit(client, token, opened.session_id)
        assert result.error
        assert result.exit_code == 1
    assert all(process.returncode is not None for process in connector.processes)


@pytest.mark.asyncio
async def test_gateway_shutdown_kills_unfinished_process(signer, verifier):
    token = grant(signer, "import sys; sys.stdin.buffer.read()")
    async with running_gateway(verifier) as (client, connector):
        await client.open(rpc.OpenRequest(grant=token))
        assert connector.processes[0].returncode is None
    assert connector.processes[0].returncode is not None


@pytest.mark.asyncio
async def test_gateway_backpressures_large_output_and_reaps_abandoned_session(signer, verifier):
    token = grant(signer, "import sys; sys.stdout.buffer.write(b'x'*(3*1024*1024)); sys.stdout.flush()")
    async with running_gateway(verifier, max_buffer_bytes=65536) as (client, connector):
        opened = await client.open(rpc.OpenRequest(grant=token))
        stdout, stderr, result = await read_to_exit(client, token, opened.session_id)
        assert stdout == b"x" * (3 * 1024 * 1024)
        assert stderr == b"" and result.exit_code == 0 and not result.error
    abandoned = grant(signer, "import sys; sys.stdin.buffer.read()", grant_id="abandoned")
    async with running_gateway(verifier, idle_timeout_ms=1) as (client, connector):
        opened = await client.open(rpc.OpenRequest(grant=abandoned))
        async with asyncio.timeout(3):
            await connector.processes[0].wait()
        with pytest.raises(ConnectError) as expired:
            await client.receive(rpc.ReceiveRequest(grant=abandoned, session_id=opened.session_id))
        assert expired.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_gateway_close_wakes_pending_receive_without_long_poll_delay(signer, verifier):
    token = grant(signer, "import sys; sys.stdin.buffer.read()")
    service = ExecGatewayService(verifier, ProcessExecConnector())
    await service.start()
    try:
        opened = await service.open(rpc.OpenRequest(grant=token), None)
        receiving = asyncio.create_task(
            service.receive(
                rpc.ReceiveRequest(grant=token, session_id=opened.session_id, wait_ms=20_000),
                None,
            )
        )
        # Let the receive coroutine reach its event wait before closing.
        await asyncio.sleep(0)
        await service.close_session(rpc.CloseRequest(grant=token, session_id=opened.session_id), None)
        async with asyncio.timeout(1):
            result = await receiving
        assert result.exited and result.exit_code == 137
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_gateway_cancelled_close_remains_owned_through_shutdown(signer, verifier):
    connector = PausedCloseConnector()
    token = grant(signer, "import sys; sys.stdin.buffer.read()")
    async with running_gateway(verifier, connector=connector) as (client, _):
        opened = await client.open(rpc.OpenRequest(grant=token))
        closing = asyncio.create_task(client.close_session(rpc.CloseRequest(grant=token, session_id=opened.session_id)))
        await asyncio.wait_for(connector.close_started.wait(), timeout=2)
        closing.cancel()
        closing.cancel()
        await asyncio.gather(closing, return_exceptions=True)
        assert connector.processes[0].returncode is None
        connector.release_close.set()
    assert connector.processes[0].returncode is not None


@pytest.mark.asyncio
async def test_gateway_shutdown_waits_for_reaper_disposal_despite_cancellation(signer, verifier):
    connector = PausedCloseConnector()
    service = ExecGatewayService(verifier, connector, idle_timeout_ms=1)
    token = grant(signer, "import sys; sys.stdin.buffer.read()")
    await service.start()
    await service.open(rpc.OpenRequest(grant=token), None)
    await asyncio.wait_for(connector.close_started.wait(), timeout=2)
    shutdown = asyncio.create_task(service.shutdown())
    # Start shutdown, then cancel it while the reaper's external close is paused.
    await asyncio.sleep(0)
    shutdown.cancel()
    await asyncio.sleep(0)
    shutdown.cancel()
    assert connector.processes[0].returncode is None
    connector.release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(shutdown, timeout=2)
    assert connector.processes[0].returncode is not None
