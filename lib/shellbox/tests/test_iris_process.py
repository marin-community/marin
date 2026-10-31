# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct Iris process pipes over real loopback sockets and subprocesses."""

import asyncio
import fcntl
import json
import os
import socket
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("iris")

from iris.cluster.types import DriverConnectionSpec, JobName
from iris.rpc import controller_pb2
from shellbox.backends.iris import process_guest
from shellbox.backends.iris.machine import IrisMachine
from shellbox.backends.iris.process import IrisProcessSession
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, MachineTerminated


class ConnectionController:
    """Authorization returns an allocation identity, never carries process bytes."""

    def __init__(self, port):
        self.port = port
        self.opened = []
        self.admitted: threading.Event | None = None
        self.release: threading.Event | None = None

    def get_task_connection(self, request, *, timeout_ms):
        del timeout_ms
        self.opened.append(request)
        if self.admitted is not None:
            self.admitted.set()
        if self.release is not None:
            assert self.release.wait(10)
        return controller_pb2.Controller.GetTaskConnectionResponse(
            pod_ip="127.0.0.1", port=self.port, pod_uid="pod-uid", attempt_uid="attempt-one"
        )

    def exec_in_container(self, *args, **kwargs):
        raise AssertionError("Primary process data must not use controller exec")


class Endpoint:
    def close(self):
        pass


class Client:
    def shutdown(self):
        pass


class Job:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


@asynccontextmanager
async def direct_machine(tmp_path: Path):
    capability = os.urandom(32)
    connections = set()

    def connected(reader, writer):
        task = asyncio.create_task(process_guest.serve_process(reader, writer, capability))
        connections.add(task)
        task.add_done_callback(connections.discard)

    server = await process_guest.start_server(connected, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    controller = ConnectionController(port)
    status = SimpleNamespace(
        current_attempt_number=0, attempts=(SimpleNamespace(attempt_number=0, attempt_uid="attempt-one"),)
    )
    machine = IrisMachine(
        Endpoint(),
        Client(),
        controller,
        Job(),
        SimpleNamespace(task_id=JobName.from_wire("/test-user/shellbox/0"), status=lambda: status),
        MachineSpec(RegistryImage("fixture"), workdir=str(tmp_path)),
        capability=capability,
        driver=DriverConnectionSpec(JobName.from_wire("/test-user/driver/0"), "driver-attempt", port),
    )
    try:
        yield machine, controller, connections
    finally:
        await machine.close()
        server.close()
        await server.wait_closed()
        for task in tuple(connections):
            task.cancel()
        await asyncio.gather(*connections, return_exceptions=True)


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_iris_http_relay_streams_large_binary_frames_concurrently_and_reclaims_pending_work(tmp_path):
    async def check():
        entered, release, waiting, cancelled = (asyncio.Event() for _ in range(4))

        async def handler(request: HttpRequest) -> HttpResponse:
            if request.path == "/slow":
                entered.set()
                await release.wait()
                return HttpResponse(200, b"slow finished")
            if request.path == "/wait":
                waiting.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return HttpResponse(201, request.body[::-1], "application/octet-stream")

        async with direct_machine(tmp_path) as (machine, controller, _):
            async with httpx.AsyncClient(trust_env=False) as client:
                pending = []
                try:
                    async with serve_http(machine, handler, port=available_port()) as url:
                        slow = asyncio.create_task(client.post(url + "/slow", timeout=20))
                        pending.append(slow)
                        await asyncio.wait_for(entered.wait(), 5)
                        payload = bytes(range(256)) * 8192
                        result = await client.post(url + "/binary", content=payload, timeout=20)
                        assert result.status_code == 201 and result.content == payload[::-1]
                        release.set()
                        assert (await slow).content == b"slow finished"
                        pending.append(asyncio.create_task(client.post(url + "/wait", timeout=20)))
                        await asyncio.wait_for(waiting.wait(), 5)
                    assert cancelled.is_set()
                    assert len(controller.opened) == 1
                finally:
                    release.set()
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(check())


def test_iris_process_chunks_binary_stdin_and_stdout_and_drains_bounded_stderr(tmp_path):
    payload = bytes(range(256)) * 8192
    script = (
        "import os,sys\n"
        "assert os.getcwd() == os.environ['EXPECTED_CWD']\n"
        "sys.stderr.buffer.write(b'x' * 2097152); sys.stderr.buffer.flush()\n"
        "data = sys.stdin.buffer.read(2097152)\n"
        "sys.stdout.buffer.write(data[::-1]); sys.stdout.buffer.flush(); sys.exit(7)\n"
    )

    async def check():
        async with direct_machine(tmp_path) as (machine, controller, _):
            session = await machine.open_process(
                Command(("python3", "-u", "-c", script), env={"EXPECTED_CWD": str(tmp_path)}, output_limit_bytes=32)
            )
            received = bytearray()
            async with asyncio.timeout(20):
                await session.write(payload)
                while chunk := await session.read(16384):
                    received.extend(chunk)
                assert await session.wait() == 7
            assert received == payload[::-1]
            assert session.stderr == b"x" * 32
            await session.close()
            assert len(controller.opened) == 1

    asyncio.run(check())


def test_iris_process_cancelled_admission_waits_for_ownership_and_reclaims_process(tmp_path):
    async def check():
        async with direct_machine(tmp_path) as (machine, controller, _):
            controller.admitted, controller.release = threading.Event(), threading.Event()
            opening = asyncio.create_task(machine.open_process(Command(("sh", "-c", "read line"))))
            try:
                assert await asyncio.to_thread(controller.admitted.wait, 5)
                opening.cancel("original cancellation")
                # Let cancellation enter ownership recovery before interrupting it again.
                await asyncio.sleep(0)
                opening.cancel("second cancellation")
                await asyncio.sleep(0)
                assert not opening.done()
                controller.release.set()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await asyncio.wait_for(opening, 5)
                assert caught.value.args == ("original cancellation",)
                assert len(controller.opened) == 1
            finally:
                controller.release.set()
                await asyncio.gather(opening, return_exceptions=True)

    asyncio.run(check())


def test_iris_machine_close_reclaims_every_open_process_and_rejects_more_writes(tmp_path):
    async def check():
        async with direct_machine(tmp_path) as (machine, controller, _):
            first = await machine.open_process(Command(("cat",)))
            second = await machine.open_process(Command(("cat",)))
            await first.write(b"first process\n")
            await second.write(b"second process\n")
            assert await first.read() == b"first process\n"
            assert await second.read() == b"second process\n"
            await asyncio.gather(machine.close(), machine.close())
            assert len(controller.opened) == 1
            for session in (first, second):
                with pytest.raises(MachineTerminated):
                    await session.write(b"late")

    asyncio.run(check())


def test_direct_primary_commands_and_large_files_bypass_controller_exec(tmp_path):
    payload = bytes(range(256)) * 8192

    async def check():
        async with direct_machine(tmp_path) as (machine, controller, _):
            source, remote, target = (tmp_path / name for name in ("input", "remote", "output"))
            source.write_bytes(payload)
            await machine.upload(source, str(remote))
            result = await machine.run(Command(("sh", "-c", "cat; printf '\\000\\377' >&2"), stdin=b"input\x00\xff"))
            assert result.exit_code == 0 and result.stdout == b"input\x00\xff" and result.stderr == b"\x00\xff"
            await machine.download(str(remote), target)
            assert target.read_bytes() == payload
            assert len(controller.opened) == 1

    asyncio.run(check())


def test_lost_ready_after_open_never_replays_and_reclaims_the_process(tmp_path, monkeypatch):
    marker, lockfile = tmp_path / "executions", tmp_path / "running.lock"
    program = (
        "import fcntl,signal,sys; lock=open(sys.argv[2],'w'); fcntl.flock(lock,fcntl.LOCK_EX); "
        "open(sys.argv[1],'a').write('once\\n'); print('started',flush=True); signal.pause()"
    )

    async def check():
        connect = asyncio.open_connection
        proxies = set()
        async with direct_machine(tmp_path) as (machine, controller, connections):

            async def lose_ready(reader, writer):
                guest_reader, guest_writer = await connect("127.0.0.1", controller.port)
                try:
                    # Forward the real mutual handshake and Open. Drop the
                    # acknowledgment only after the real child proves it ran.
                    for source, target, count in (
                        (guest_reader, writer, len(process_guest.MAGIC) + 32),
                        (reader, guest_writer, 64),
                        (guest_reader, writer, 32),
                    ):
                        target.write(await source.readexactly(count))
                        await target.drain()
                    kind, data = await process_guest.read_frame(reader)
                    assert kind == process_guest.OPEN
                    await process_guest.write_frame(guest_writer, kind, data)
                    assert await process_guest.read_frame(guest_reader) == (process_guest.READY, b"")
                    output = bytearray()
                    while b"\n" not in output:
                        kind, data = await process_guest.read_frame(guest_reader)
                        assert kind == process_guest.STDOUT
                        output.extend(data)
                    assert output == b"started\n"
                finally:
                    writer.close()
                    guest_writer.close()
                    await asyncio.gather(writer.wait_closed(), guest_writer.wait_closed())

            def accepted(reader, writer):
                task = asyncio.create_task(lose_ready(reader, writer))
                proxies.add(task)

            proxy = await asyncio.start_server(accepted, "127.0.0.1", 0)

            async def redirected_tcp(*args, **kwargs):
                return await connect("127.0.0.1", proxy.sockets[0].getsockname()[1], **kwargs)

            monkeypatch.setattr(asyncio, "open_connection", redirected_tcp)
            async with proxy:
                with pytest.raises(asyncio.IncompleteReadError):
                    await machine.open_process(Command(("python3", "-u", "-c", program, str(marker), str(lockfile))))
                await asyncio.wait_for(asyncio.gather(*proxies, *connections), 5)
            assert machine.job.cancelled
            assert marker.read_text() == "once\n"
            with lockfile.open() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    asyncio.run(check())


def test_process_close_reaps_descendants_even_with_unread_output(tmp_path):
    lockfile = tmp_path / "descendant.lock"
    child_program = (
        "import fcntl,sys; lock=open(sys.argv[1],'w'); fcntl.flock(lock,fcntl.LOCK_EX); "
        "print('ready',flush=True); sys.stdin.read()"
    )
    script = (
        "import os,subprocess,sys\n"
        f"child=subprocess.Popen([sys.executable,'-u','-c',{child_program!r},sys.argv[1]],"
        "stdin=subprocess.PIPE,stdout=subprocess.PIPE)\n"
        "assert child.stdout.readline()==b'ready\\n'\n"
        "print('ready',flush=True)\n"
        "while True: os.write(1,b'x'*65536)\n"
    )

    async def check():
        async with direct_machine(tmp_path) as (machine, _, _):
            session = await machine.open_process(Command(("python3", "-u", "-c", script, str(lockfile))))
            readiness = bytearray()
            while b"\n" not in readiness:
                readiness.extend(await session.read())
            assert readiness.startswith(b"ready\n")
            await asyncio.wait_for(session.close(), 5)
            with lockfile.open() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(MachineTerminated):
                await session.write(b"late")

    asyncio.run(check())


def test_broker_rejects_another_machine_capability_without_running_its_command(tmp_path):
    marker = tmp_path / "unauthorized"

    async def check():
        capability = os.urandom(32)
        server = await process_guest.start_server(
            lambda reader, writer: process_guest.serve_process(reader, writer, capability), "127.0.0.1", 0
        )
        async with server:
            with pytest.raises((PermissionError, asyncio.IncompleteReadError)):
                await IrisProcessSession.connect(
                    "127.0.0.1",
                    server.sockets[0].getsockname()[1],
                    os.urandom(32),
                    Command(("touch", str(marker)), cwd=str(tmp_path)),
                    on_close=lambda _: None,
                )
            assert not marker.exists()

    asyncio.run(check())


@pytest.mark.parametrize("blocked_stdin", [False, True])
def test_broker_disconnect_reclaims_the_running_process_group(tmp_path, blocked_stdin):
    lockfile = tmp_path / "process.lock"
    program = (
        "import fcntl,sys; f=open(sys.argv[1],'w'); fcntl.flock(f,fcntl.LOCK_EX); "
        "print('ready',flush=True); " + ("__import__('signal').pause()" if blocked_stdin else "sys.stdin.read()")
    )

    async def check():
        capability = os.urandom(32)
        finished = asyncio.Event()

        async def connected(reader, writer):
            try:
                await process_guest.serve_process(reader, writer, capability)
            finally:
                finished.set()

        server = await process_guest.start_server(connected, "127.0.0.1", 0)
        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
            await process_guest.authenticate_client(reader, writer, capability)
            import_frame = json.dumps(
                {"argv": ["python3", "-u", "-c", program, str(lockfile)], "cwd": str(tmp_path), "env": {}}
            ).encode()
            await process_guest.write_frame(writer, process_guest.OPEN, import_frame)
            assert await process_guest.read_frame(reader) == (process_guest.READY, b"")
            readiness = bytearray()
            while b"\n" not in readiness:
                kind, data = await process_guest.read_frame(reader)
                assert kind == process_guest.STDOUT
                readiness.extend(data)
            assert readiness == b"ready\n"
            if blocked_stdin:
                # This valid sender has at most one unacknowledged chunk, so EOF
                # reaches the broker even when its child's pipe cannot drain.
                await process_guest.write_frame(writer, process_guest.STDIN, b"x" * process_guest.CHUNK_BYTES)
                assert await process_guest.read_frame(reader) == (process_guest.WRITTEN, b"")
                await process_guest.write_frame(writer, process_guest.STDIN, b"x" * process_guest.CHUNK_BYTES)
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(finished.wait(), 5)
            with lockfile.open() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    asyncio.run(check())
