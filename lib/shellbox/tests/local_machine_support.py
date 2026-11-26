# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared guest programs and scenarios for local machine tests."""

import asyncio
import io
import os
import tarfile
from collections.abc import Awaitable, Callable

import pytest
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.backends.local.runtime import LocalMachine
from shellbox.backends.local.sandbox import SandboxUnavailable
from shellbox.machine import (
    Command,
    HostImage,
    MachineSpec,
    NetworkPolicy,
    Result,
)
from shellbox.task import ContainerSpec, ReadinessProbe, ServiceEndpoint, TaskSpec

# Prints "ok" or the errno name for each (operation, target) pair in argv, so one command reports several checks.
ACCESS_PROBE = """
import errno, os, socket, sys

def attempt(operation, target):
    if operation == "read":
        open(target).close()
    elif operation == "write":
        open(target, "w").close()
    elif operation == "signal":
        os.kill(int(target), 0)
    else:
        socket.create_connection(("127.0.0.1", int(target)), timeout=10).close()

for operation, target in zip(sys.argv[1::2], sys.argv[2::2]):
    try:
        attempt(operation, target)
        print("ok")
    except OSError as error:
        print(errno.errorcode[error.errno])
"""
# Starts up to argv[1] threads, which RLIMIT_NPROC counts as it counts processes, and prints how many started.
# The threads end with the command, so a test leaves no stray processes behind.
THREAD_BOMB = """
import sys, threading

threading.stack_size(1 << 16)
release = threading.Event()
started = 0
try:
    while started < int(sys.argv[1]):
        threading.Thread(target=release.wait, daemon=True).start()
        started += 1
except RuntimeError:
    pass
release.set()
print(started)
"""
# The shared native root-delivery grading quiesce (taskcompendium's harbor.root_delivery,
# whose grading collect runs it first, before restoring private verifier material): kill
# every process outside the exec's own ancestry, sparing PID 1 and zombies. The grader
# also spares the keeper sleeper by its exact argv; these copies omit that spare, so a
# machine survives them only because its init respawns the sleeper.
ROOT_DELIVERY_QUIESCE_PROGRAM = """\
import os, signal

mine = set()
pid = os.getpid()
while pid > 1:
    mine.add(pid)
    try:
        with open(f"/proc/{pid}/stat") as stream:
            fields = stream.read().rsplit(") ", 1)[1].split()
        pid = int(fields[1])
    except (OSError, IndexError, ValueError):
        break
killed = []
for entry in os.scandir("/proc"):
    if not entry.name.isdecimal():
        continue
    victim = int(entry.name)
    if victim in mine or victim == 1:
        continue
    try:
        with open(f"/proc/{victim}/stat") as stream:
            fields = stream.read().rsplit(") ", 1)[1].split()
        state = fields[0]
    except (OSError, IndexError, ValueError):
        continue
    if state.startswith("Z"):
        continue
    try:
        os.kill(victim, signal.SIGKILL)
        killed.append(victim)
    except ProcessLookupError:
        continue
print(len(killed))
"""

ROOT_DELIVERY_QUIESCE_SCRIPT = """\
set -eu
self=$$
ancestry=" $self "
pid=$self
while [ "$pid" -gt 1 ]; do
  line=$(cat "/proc/$pid/stat" 2>/dev/null) || break
  rest=${line##*) }
  state=${rest%% *}
  ppid=${rest#* }
  ppid=${ppid%% *}
  case $ppid in ''|*[!0-9]*) break;; esac
  ancestry="$ancestry $ppid "
  pid=$ppid
done
killed=0
for entry in /proc/[0-9]*; do
  victim=${entry#/proc/}
  case $ancestry in *" $victim "*) continue;; esac
  [ "$victim" -gt 1 ] || continue
  line=$(cat "/proc/$victim/stat" 2>/dev/null) || continue
  rest=${line##*) }
  case ${rest%% *} in Z*) continue;; esac
  if kill -9 "$victim" 2>/dev/null; then
    killed=$((killed + 1))
  elif [ ! -d "/proc/$victim" ]; then
    continue
  else
    echo "quiesce kill failed for $victim" >&2
    exit 1
  fi
done
echo "$killed"
"""
SPEC = MachineSpec(HostImage(), workdir="/app")
PYTHON_VERSION = "3.12.13"
# A hash lock as ``uv pip compile --generate-hashes`` writes it, for a small pure-Python package.
LOCK = """\
six==1.17.0 \\
    --hash=sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274 \\
    --hash=sha256:ff70335d468e7eb6ec65b95b99d3a2836546063f63acc5171de367e834932a81
"""


def make_factory(**options) -> LocalMachineFactory:
    try:
        return LocalMachineFactory(**options)
    except SandboxUnavailable as error:
        pytest.skip(str(error))


def on_machine[T](
    factory: LocalMachineFactory, spec: MachineSpec, scenario: Callable[[LocalMachine], Awaitable[T]]
) -> T:
    async def run() -> T:
        machine = await factory.create(spec)
        try:
            return await scenario(machine)
        finally:
            await machine.close()

    return asyncio.run(run())


def run_command(factory: LocalMachineFactory, command: Command, spec: MachineSpec = SPEC) -> Result:
    return on_machine(factory, spec, lambda machine: machine.run(command))


def access_outcomes(factory: LocalMachineFactory, *checks: tuple[str, object], spec: MachineSpec = SPEC) -> list[str]:
    argv = [str(part) for check in checks for part in check]
    return run_command(factory, Command(("python3", "-c", ACCESS_PROBE, *argv)), spec).stdout.decode().split()


def _stream(*members: tuple[tarfile.TarInfo, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for member, data in members:
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def _member(name: str, kind: bytes = tarfile.REGTYPE, linkname: str = "") -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type, member.linkname, member.mode = kind, linkname, 0o755 if kind == tarfile.DIRTYPE else 0o644
    return member


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


# Copy-on-write retains root-owned system file permissions.
root_commands = pytest.mark.skipif(os.geteuid() != 0, reason="Writing root-owned system directories needs root")


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
