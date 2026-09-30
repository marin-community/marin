# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures: an in-memory container runtime, and in-process host/broker servers.

``FakeRuntime`` runs real local processes for ``/bin/sh -c`` commands and
session commands, so the session machinery (host-side output files, exit codes
recorded by a watcher thread) is exercised for real. Guest paths map into a
per-container temp directory. File transfer recognises the exact busybox argv
the agent builds, which pins that argv as a side effect.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO

import pytest
import uvicorn
from silo.host.runtime import BUSYBOX, ContainerSpec, ExecOutcome, RuntimeError_
from silo_testing import wait_until


class FakeRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.images: set[str] = set()
        self.pulls: list[str] = []
        self.builds: list[tuple[str, str]] = []
        self.started: list[ContainerSpec] = []
        self.containers: dict[str, Path] = {}
        # Deletion blocks here until a test releases it, which is how the
        # "delete returns before the resource is gone" window is made observable.
        self.remove_gate = threading.Event()
        self.remove_gate.set()
        self.fail_start = False

    def _guest(self, name: str, path: str) -> Path:
        return self.containers[name] / path.lstrip("/")

    def ensure_image(self, ref: str) -> None:
        self.pulls.append(ref)
        self.images.add(ref)

    def build(self, tag: str, dockerfile: str) -> None:
        self.builds.append((tag, dockerfile))
        self.images.add(tag)

    def start(self, spec: ContainerSpec) -> None:
        if self.fail_start:
            raise RuntimeError_("injected start failure")
        if spec.image not in self.images:
            raise RuntimeError_(f"image {spec.image} not present")
        directory = self.root / spec.name
        directory.mkdir(parents=True)
        self.containers[spec.name] = directory
        self.started.append(spec)

    def exec(
        self,
        name: str,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        user: str | None = None,
        stdin: bytes | None = None,
        timeout: float | None = None,
        merge_stderr: bool = True,
    ) -> ExecOutcome:
        if name not in self.containers:
            raise RuntimeError_(f"no such container {name}")
        argv = list(argv)
        if argv[:3] == [BUSYBOX, "sh", "-c"] and argv[4] == "silo-upload":
            target = self._guest(name, argv[5])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stdin or b"")
            return ExecOutcome(0, b"")
        if argv[:3] == [BUSYBOX, "test", "-f"]:
            return ExecOutcome(0 if self._guest(name, argv[3]).is_file() else 1, b"")
        if argv[:2] == [BUSYBOX, "cat"]:
            path = self._guest(name, argv[2])
            if not path.is_file():
                return ExecOutcome(1, b"", stderr=b"cat: no such file")
            return ExecOutcome(0, path.read_bytes())
        if argv[:2] == ["/bin/sh", "-c"]:
            workdir = self.containers[name] if not cwd else self._guest(name, cwd)
            try:
                proc = subprocess.run(
                    argv,
                    cwd=workdir,
                    input=stdin,
                    env={**os.environ, **(env or {})},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                return ExecOutcome(137, error.stdout or b"", timed_out=True)
            return ExecOutcome(proc.returncode, proc.stdout)
        raise AssertionError(f"unexpected argv {argv}")

    def spawn(
        self,
        name: str,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None,
        cwd: str | None,
        user: str | None,
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(list(argv), cwd=self.containers[name], stdout=stdout, stderr=stderr)

    def remove(self, name: str) -> None:
        self.remove_gate.wait(timeout=30)
        self.containers.pop(name, None)

    def exists(self, name: str) -> bool:
        return name in self.containers


@pytest.fixture
def fake_runtime(tmp_path: Path) -> FakeRuntime:
    return FakeRuntime(tmp_path / "containers")


class ServerThread:
    """uvicorn on a kernel-assigned port in a background thread."""

    def __init__(self, app) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off", ws="none"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> ServerThread:
        self.thread.start()
        wait_until(lambda: self.server.started, timeout=10)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def serve() -> Iterator:
    servers: list[ServerThread] = []

    def start(app) -> ServerThread:
        server = ServerThread(app).__enter__()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.__exit__(None, None, None)
