# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local task peers, service readiness, and coordinated cleanup."""

import asyncio
import contextlib
import errno
import json
import os
import shutil
import socket
from pathlib import Path
from typing import cast

import pytest
from shellbox.backends.local.runtime import LocalMachine
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.machine import (
    Command,
    ExitReason,
    HostImage,
    MachineSpec,
    MachineTerminated,
    NetworkPolicy,
)
from shellbox.task import ContainerSpec, ReadinessProbe, ServiceEndpoint, TaskSpec, single_container_task

from lib.shellbox.tests.local_machine_support import SPEC, make_factory, on_machine, run_command

HTTP_CLIENT = """import sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    sys.stdout.buffer.write(response.read())
"""


SERVICE_SERVER = """import http.server, os, pathlib, sys
pathlib.Path('/app/member-secret').write_text(sys.argv[1])
pathlib.Path('/app/service-pid').write_text(str(os.getpid()))
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = pathlib.Path('/app/member-secret').read_bytes()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
http.server.ThreadingHTTPServer(('127.0.0.1', 8123), Handler).serve_forever()
"""


CONNECT_PROBE = """import socket, sys
try:
    socket.create_connection((sys.argv[2], int(sys.argv[1])), timeout=1).close()
except OSError:
    print('blocked')
else:
    print('connected')
"""


def private_task_spec(marker="service", readiness=None, peer_name="runtime"):
    return TaskSpec(
        containers=(
            ContainerSpec(
                name=peer_name,
                machine=MachineSpec(HostImage(), workdir="/app", network=NetworkPolicy.DENY),
                endpoints=(ServiceEndpoint("http", 8123),),
                entrypoint=("python3", "-c", SERVICE_SERVER),
                arguments=(marker,),
                readiness=readiness
                or ReadinessProbe(
                    command=Command(
                        ("python3", "-c", "import socket; socket.create_connection(('127.0.0.1', 8123)).close()")
                    ),
                    timeout=10,
                    interval=0.05,
                ),
            ),
            ContainerSpec("actor", MachineSpec(HostImage(), workdir="/app", network=NetworkPolicy.DENY)),
        ),
        primary="actor",
    )


def test_persistent_machine_hides_host_processes_and_drops_all_privilege(factory):
    script = """import json, os, pathlib
status = dict(line.split(':', 1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
print(json.dumps({key: status[key].strip() for key in ('NoNewPrivs', 'CapEff', 'CapPrm', 'CapInh', 'CapAmb', 'CapBnd')}))
print(pathlib.Path('/proc/' + os.environ['HOST_PROCESS']).exists())
links = []
for descriptor in pathlib.Path('/proc/self/fd').iterdir():
    try:
        links.append(os.readlink(descriptor))
    except FileNotFoundError:
        pass
print(json.dumps(links))
"""
    result = run_command(factory, Command(("python3", "-c", script), env={"HOST_PROCESS": str(os.getpid())}))
    assert result.exit_code == 0, result.stderr
    status, sees_host, descriptors = result.stdout.decode().splitlines()
    assert json.loads(status) == {
        "NoNewPrivs": "1",
        **{key: "0000000000000000" for key in ("CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd")},
    }
    assert sees_host == "False"
    assert not any(
        target.startswith(("user:[", "mnt:[", "pid:[", "net:[")) or "shellbox-config" in target
        for target in json.loads(descriptors)
    )


def test_real_local_http_relay_serves_concurrent_requests(factory):
    async def scenario(machine):
        blocked, release = asyncio.Event(), asyncio.Event()

        async def handler(request: HttpRequest):
            if request.path == "/slow":
                blocked.set()
                await release.wait()
            return HttpResponse(200, request.path.encode(), "text/plain")

        async with (
            asyncio.TaskGroup() as service_tasks,
            serve_http(machine, handler, task_group=service_tasks, port=8787) as endpoint,
        ):
            slow = asyncio.create_task(
                machine.run(Command(("python3", "-c", HTTP_CLIENT, endpoint + "/slow"), timeout=15))
            )
            try:
                await asyncio.wait_for(blocked.wait(), 10)
                fast = await machine.run(Command(("python3", "-c", HTTP_CLIENT, endpoint + "/fast"), timeout=10))
                assert (fast.exit_code, fast.stdout) == (0, b"/fast")
                release.set()
                result = await slow
                assert (result.exit_code, result.stdout) == (0, b"/slow")
            finally:
                release.set()
                if not slow.done():
                    slow.cancel()
                    await asyncio.gather(slow, return_exceptions=True)

    on_machine(factory, MachineSpec(HostImage(), workdir="/app", network=NetworkPolicy.DENY), scenario)


@pytest.mark.parametrize(
    "family,host,peer_name",
    [(socket.AF_INET, "127.0.0.1", "runtime_peer"), (socket.AF_INET6, "::1", "runtime")],
    ids=["ipv4-underscore-peer", "ipv6"],
)
def test_local_task_peers_have_private_filesystems_and_no_host_network(factory, family, host, peer_name):
    async def scenario(host_port):
        groups = await asyncio.gather(
            factory.create_task(private_task_spec("first", peer_name=peer_name)),
            factory.create_task(private_task_spec("second", peer_name=peer_name)),
        )
        roots = [cast(LocalMachine, machine).root for group in groups for machine in group.containers.values()]
        try:
            await asyncio.gather(*(group.wait_ready() for group in groups))
            for group, expected in zip(groups, (b"first", b"second"), strict=True):
                written = await group.primary.run(Command(("sh", "-c", "echo actor >/app/member-secret")))
                assert written.exit_code == 0
                own = await group.primary.run(Command(("cat", "/app/member-secret")))
                assert own.stdout == b"actor\n"
                answer = await group.primary.run(
                    Command(("python3", "-c", HTTP_CLIENT, "http://" + group.endpoint(peer_name, "http")))
                )
                assert (answer.exit_code, answer.stdout) == (0, expected)
                connection = await group.primary.run(Command(("python3", "-c", CONNECT_PROBE, str(host_port), host)))
                assert connection.stdout == b"blocked\n"
        finally:
            await asyncio.gather(*(group.close() for group in groups))
        assert not any(root.exists() for root in roots)

    with contextlib.ExitStack() as cleanup:
        try:
            listener = cleanup.enter_context(socket.socket(family))
            listener.bind((host, 0))
        except OSError as error:
            if family == socket.AF_INET6 and error.errno in (
                errno.EAFNOSUPPORT,
                errno.EPROTONOSUPPORT,
                errno.EADDRNOTAVAIL,
            ):
                pytest.skip("Host IPv6 loopback is unavailable")
            raise
        listener.listen()
        asyncio.run(scenario(listener.getsockname()[1]))


def test_local_task_retries_a_timed_readiness_probe_without_destroying_the_companion(factory):
    probe = """import pathlib, signal, socket
marker = pathlib.Path('/app/probe-attempted')
if not marker.exists():
    marker.touch()
    signal.pause()
socket.create_connection(('127.0.0.1', 8123), timeout=1).close()
"""

    async def scenario():
        spec = private_task_spec(
            readiness=ReadinessProbe(
                command=Command(("python3", "-c", probe), timeout=0.15),
                timeout=10,
                interval=0.05,
                attempts=5,
            )
        )
        group = await factory.create_task(spec)
        try:
            await group.wait_ready()
            result = await group.containers["runtime"].run(Command(("test", "-f", "/app/probe-attempted")))
            assert result.exit_code == 0
            response = await group.primary.run(Command(("python3", "-c", HTTP_CLIENT, "http://runtime:8123")))
            assert response.stdout == b"service"
        finally:
            await group.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
def test_local_task_command_deadline_closes_every_member(undelegated_factory, cancel):
    async def scenario():
        group = await undelegated_factory.create_task(private_task_spec())
        roots = [cast(LocalMachine, machine).root for machine in group.containers.values()]
        try:
            await group.wait_ready()
            command = Command(
                ("sh", "-c", "touch /app/command-started; exec sleep 3600"), timeout=None if cancel else 0.2
            )
            pending = asyncio.create_task(group.primary.run(command))
            if cancel:
                started = await group.primary.run(
                    Command(("sh", "-c", "until test -f /app/command-started; do sleep 0.01; done"), timeout=5)
                )
                assert started.exit_code == 0
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                result = await pending
                assert result.reason is ExitReason.TIMED_OUT
            for machine in group.containers.values():
                with pytest.raises(MachineTerminated):
                    await machine.run(Command(("true",)))
        finally:
            await group.close()
        assert not any(root.exists() for root in roots)

    asyncio.run(scenario())


def test_cancelled_command_start_closes_the_machine(undelegated_factory, monkeypatch):
    async def scenario():
        machine = await undelegated_factory.create(SPEC)
        connection_made = asyncio.subprocess.SubprocessStreamProtocol.connection_made

        def cancel_before_handle_returns(protocol, transport):
            connection_made(protocol, transport)
            monkeypatch.setattr(asyncio.subprocess.SubprocessStreamProtocol, "connection_made", connection_made)
            # asyncio calls this after spawning the process but before returning
            # its handle. Cancel without adding an artificial await to startup.
            pending.cancel()

        try:
            monkeypatch.setattr(
                asyncio.subprocess.SubprocessStreamProtocol, "connection_made", cancel_before_handle_returns
            )
            pending = asyncio.create_task(machine.run(Command(("sleep", "3600"))))
            with pytest.raises(asyncio.CancelledError):
                await pending
            with pytest.raises(MachineTerminated):
                await machine.run(Command(("true",)))
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "companions,service_exit",
    [(False, False), (True, False), (True, True)],
    ids=["singleton-ttl", "multi-container-ttl", "companion-exit"],
)
def test_local_task_failure_and_ttl_reclaim_all_members(companions, service_exit):
    factory = make_factory(ttl=None if service_exit else 3)

    async def scenario():
        spec = private_task_spec() if companions else single_container_task(SPEC)
        group = await factory.create_task(spec)
        roots = [cast(LocalMachine, machine).root for machine in group.containers.values()]
        try:
            await group.wait_ready()
            waiting = asyncio.create_task(group.primary.run(Command(("sleep", "3600"))))
            if service_exit:
                with contextlib.suppress(MachineTerminated):
                    await group.containers["runtime"].run(Command(("sh", "-c", "kill -TERM $(cat /app/service-pid)")))
            with pytest.raises(MachineTerminated):
                await asyncio.wait_for(waiting, 8)
            for machine in group.containers.values():
                with pytest.raises(MachineTerminated):
                    await machine.run(Command(("true",)))
        finally:
            await group.close()
        assert not any(root.exists() for root in roots)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "entrypoint,arguments,expected",
    [
        (None, None, b"from-image"),
        (
            (
                "/bin/sh",
                "-c",
                'test "$#" -eq 0 || exit 23; printf entrypoint >index.html; exec httpd -f -p 8123 -h .',
                "entrypoint",
            ),
            None,
            b"entrypoint",
        ),
        (None, ("printf arguments >index.html; exec httpd -f -p 8123 -h .",), b"arguments"),
        (
            ("/bin/sh", "-c", 'printf "%s" "$1" >index.html; exec httpd -f -p 8123 -h .', "entrypoint"),
            ("both",),
            b"both",
        ),
    ],
    ids=["image-defaults", "entrypoint-only", "arguments-only", "both-overridden"],
)
def test_prepared_local_task_applies_startup_overrides_with_private_writable_roots(
    executable_oci_image, tmp_path, entrypoint, arguments, expected
):
    umoci = shutil.which("umoci")
    if umoci is None:
        pytest.skip("OCI preparation requires umoci; the Shellbox CI lane installs it")
    factory = make_factory(umoci=Path(umoci))
    image = executable_oci_image
    original = {path.relative_to(image.layout): path.read_bytes() for path in image.layout.rglob("*") if path.is_file()}
    task = MachineSpec(image, workdir="", network=NetworkPolicy.DENY)
    service = ContainerSpec(
        "runtime",
        task,
        endpoints=(ServiceEndpoint("http", 8123),),
        readiness=ReadinessProbe(Command(("wget", "-q", "-O", "-", "http://runtime:8123")), timeout=10, interval=0.05),
        entrypoint=entrypoint,
        arguments=arguments,
    )

    async def scenario():
        group = await factory.create_task(TaskSpec((ContainerSpec("actor", task), service), primary="actor"))
        roots = [cast(LocalMachine, machine).root for machine in group.containers.values()]
        try:
            await group.wait_ready()
            identity = await group.primary.run(
                Command(("sh", "-c", 'pwd; printf "%s\\n" "$IMAGE_VALUE"; id -u; printf "%s" "$PATH"'))
            )
            assert identity.exit_code == 0, identity.stderr
            workdir, value, user, path = identity.stdout.decode().splitlines()
            assert (workdir, value, user) == ("/app", "from-image", "0")
            assert "/run/current-system" not in path
            written = await group.primary.run(Command(("sh", "-c", "printf actor >index.html")))
            assert written.exit_code == 0
            response = await group.primary.run(Command(("wget", "-q", "-O", "-", "http://runtime:8123")))
            assert (response.exit_code, response.stdout) == (0, expected)
            await group.primary.download("/app/index.html", tmp_path / "actor")
            await group.containers["runtime"].download("/app/index.html", tmp_path / "runtime")
            assert (tmp_path / "actor").read_bytes() == b"actor"
            assert (tmp_path / "runtime").read_bytes() == expected
        finally:
            await group.close()
        assert not any(root.exists() for root in roots)

    asyncio.run(scenario())
    assert {
        path.relative_to(image.layout): path.read_bytes() for path in image.layout.rglob("*") if path.is_file()
    } == original
