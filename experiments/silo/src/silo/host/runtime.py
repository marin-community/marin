# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The container runtime a host agent drives, and its nerdctl implementation.

The agent talks to a small ``ContainerRuntime`` protocol rather than to nerdctl
directly, so the lifecycle logic (ids, sessions, async deletion, cpusets) is
testable against an in-memory fake without a container daemon.

Why nerdctl over the containerd gRPC API: nerdctl-full ships containerd, runc,
buildkit and the CNI plugins as one tarball, its CLI surface maps one-to-one onto
what the provider needs (run/exec/rm/pull/build), and the Phase 0 spike proved it
works inside a privileged Iris task on cw-us-east-02a. A subprocess per operation
costs milliseconds against sandbox lifetimes measured in minutes.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol

logger = logging.getLogger(__name__)

# Where the host's static busybox is bind-mounted inside every sandbox. It is the
# PID 1 keepalive and the plumbing for file transfer, so the provider works for
# images with no shell utilities at all. User commands never run through it: the
# pipeline wraps its own commands in the GUEST's /bin/sh, as it did on Daytona.
TOOLS_MOUNT = "/.silo"
BUSYBOX = f"{TOOLS_MOUNT}/busybox"

RUNTIME_RUNC = "runc"
RUNTIME_RUNSC = "runsc"
RUNTIMES = (RUNTIME_RUNC, RUNTIME_RUNSC)


@dataclass(frozen=True)
class ContainerSpec:
    """Everything needed to start one sandbox container."""

    name: str
    image: str
    runtime: str
    cpus: int
    cpuset: tuple[int, ...]
    memory_bytes: int
    network_none: bool = True
    user: str | None = None
    workdir: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.runtime not in RUNTIMES:
            raise ValueError(f"unknown runtime {self.runtime!r}")
        if not self.network_none:
            # There is no code path that starts a sandbox with a network. If a
            # caller ever asks for one, that is a policy decision to make
            # explicitly here, not a flag to thread through quietly.
            raise ValueError("sandboxes are always started with --network none")


@dataclass(frozen=True)
class ExecOutcome:
    exit_code: int
    output: bytes
    timed_out: bool = False
    # Populated only when the exec was run with merge_stderr=False.
    stderr: bytes = b""


class ContainerRuntime(Protocol):
    def ensure_image(self, ref: str) -> None: ...

    def build(self, tag: str, dockerfile: str) -> None: ...

    def start(self, spec: ContainerSpec) -> None: ...

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
    ) -> ExecOutcome: ...

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
    ) -> subprocess.Popen[bytes]: ...

    def remove(self, name: str) -> None: ...

    def exists(self, name: str) -> bool: ...


class RuntimeError_(Exception):
    """A runtime operation failed. Carries the command's own output."""


class NerdctlRuntime:
    """``ContainerRuntime`` over the nerdctl CLI against a private containerd."""

    def __init__(
        self,
        *,
        address: str,
        namespace: str = "silo",
        tools_dir: Path,
        runsc_binary: Path | None = None,
        docker_config_dir: Path | None = None,
        nerdctl: str = "nerdctl",
        ctr: str = "ctr",
    ) -> None:
        self._base = [nerdctl, "--address", address, "--namespace", namespace]
        # Commands run through containerd's own `ctr task exec`, not `nerdctl
        # exec`: nerdctl 2.1.2 exits 1 for ANY non-zero exit and appends
        # `level=fatal msg="exec failed with exit code N"` to the command's
        # stderr (measured, spike phase0c). The pipeline reads both -- 124 means
        # "timed out" to it -- so exec must report exactly what the command did.
        self._ctr = [ctr, "--address", address, "--namespace", namespace]
        # nerdctl --name -> the containerd container id ctr needs.
        self._ids: dict[str, str] = {}
        self._ids_lock = threading.Lock()
        self._tools_dir = tools_dir
        # gVisor is run through containerd's ordinary runc v2 shim with runsc as
        # the OCI binary (`--runtime /abs/path/runsc`), NOT through the dedicated
        # containerd-shim-runsc-v1. Measured in the Phase 0b spike on
        # cw-us-east-02a: the dedicated shim hangs when nested inside a
        # privileged pod (a bare `true` timed out at 150 s), while the runc-shim
        # route starts a sandbox in ~0.6 s.
        self._runsc_binary = runsc_binary
        self._env = dict(os.environ)
        if docker_config_dir is not None:
            # Registry credentials live only in this directory and only in the
            # environment of the nerdctl subprocesses. They are never passed to
            # a container: `nerdctl exec` forwards exactly the -e flags we give it.
            self._env["DOCKER_CONFIG"] = str(docker_config_dir)
        self._pull_locks: dict[str, threading.Lock] = {}
        self._pull_locks_guard = threading.Lock()

    # -- helpers -------------------------------------------------------------

    def _run(self, args: Sequence[str], *, timeout: float | None = None, stdin: bytes | None = None) -> bytes:
        cmd = [*self._base, *args]
        proc = subprocess.run(
            cmd,
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self._env,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError_(
                f"nerdctl {shlex.join(args[:2])} exited {proc.returncode}: "
                f"{proc.stdout.decode(errors='replace')[-800:]}"
            )
        return proc.stdout

    def _container_id(self, name: str) -> str:
        with self._ids_lock:
            cid = self._ids.get(name)
        if cid is None:
            cid = self._run(["inspect", "--format", "{{.ID}}", name], timeout=60).decode().strip()
            with self._ids_lock:
                self._ids[name] = cid
        return cid

    def _exec_command(
        self,
        name: str,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None,
        cwd: str | None,
        user: str | None,
    ) -> list[str]:
        command = [*self._ctr, "task", "exec", "--exec-id", f"silo-{uuid.uuid4().hex}"]
        if cwd:
            command += ["--cwd", cwd]
        if user:
            command += ["--user", user]
        command.append(self._container_id(name))
        if env:
            # ctr has no --env; the container's own environment stays as the base.
            command += [BUSYBOX, "env", *(f"{key}={value}" for key, value in env.items())]
        return [*command, *argv]

    # -- images --------------------------------------------------------------

    def _lock_for(self, ref: str) -> threading.Lock:
        with self._pull_locks_guard:
            return self._pull_locks.setdefault(ref, threading.Lock())

    def has_image(self, ref: str) -> bool:
        try:
            self._run(["image", "inspect", ref], timeout=60)
            return True
        except RuntimeError_:
            return False

    def ensure_image(self, ref: str) -> None:
        # One pull per ref at a time: fifty sandboxes from one snapshot should
        # cost one pull, not fifty concurrent ones racing the same blobs.
        with self._lock_for(ref):
            if self.has_image(ref):
                return
            self._run(["pull", "-q", ref], timeout=1800)

    def build(self, tag: str, dockerfile: str) -> None:
        with self._lock_for(tag):
            if self.has_image(tag):
                return
            # An empty context: the recipes we accept never COPY/ADD. A directory
            # we create ourselves, not /var/empty, which the Iris task image lacks
            # (acceptance silo3-01: "lstat /var/empty: no such file or directory").
            context = self._tools_dir.parent / "empty-build-context"
            context.mkdir(parents=True, exist_ok=True)
            self._run(["build", "-t", tag, "-f", "-", str(context)], timeout=3600, stdin=dockerfile.encode())

    # -- containers ----------------------------------------------------------

    def start(self, spec: ContainerSpec) -> None:
        args = [
            "run",
            "-d",
            "--name",
            spec.name,
            "--network",
            "none",
            # CFS quota: this is what the guest reads back as cpu.max.
            "--cpus",
            str(spec.cpus),
            # cpuset: without it nproc reports the whole node (192), and a
            # `make -j$(nproc)` forks 192 compilers into an 8 GB cgroup.
            "--cpuset-cpus",
            ",".join(str(c) for c in spec.cpuset),
            "--memory",
            str(spec.memory_bytes),
            # No swap beyond the memory limit, so memory.max means what it says.
            "--memory-swap",
            str(spec.memory_bytes),
            "--pids-limit",
            "32768",
            "-v",
            f"{self._tools_dir}:{TOOLS_MOUNT}:ro",
            "--entrypoint",
            BUSYBOX,
        ]
        if spec.runtime == RUNTIME_RUNSC:
            if self._runsc_binary is None:
                raise RuntimeError_("runsc requested but this host has no runsc binary configured")
            args += ["--runtime", str(self._runsc_binary)]
        if spec.user:
            args += ["-u", spec.user]
        if spec.workdir:
            args += ["-w", spec.workdir]
        for key, value in spec.env.items():
            args += ["-e", f"{key}={value}"]
        for key, value in spec.labels.items():
            args += ["--label", f"{key}={value}"]
        # PID 1 is a keepalive; all work arrives via exec.
        args += [spec.image, "sleep", "2147483647"]
        output = self._run(args, timeout=600).decode().strip().splitlines()
        cid = output[-1].strip() if output else ""
        if len(cid) == 64 and all(c in "0123456789abcdef" for c in cid):
            with self._ids_lock:
                self._ids[spec.name] = cid

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
        # The deadline is enforced INSIDE the sandbox. Killing the client on the
        # host does not kill the process it started in the container, so a
        # host-side timeout alone would leave the command running.
        if timeout is not None:
            argv = [BUSYBOX, "timeout", "-s", "KILL", str(max(1, int(timeout))), *argv]
        cmd = self._exec_command(name, argv, env=env, cwd=cwd, user=user)
        try:
            proc = subprocess.run(
                cmd,
                # No input means an empty stdin, never the agent's own.
                input=stdin if stdin is not None else b"",
                stdout=subprocess.PIPE,
                # Merged for command output (Daytona's `result` interleaves them);
                # separate for file downloads, where one stray stderr byte would
                # corrupt the payload.
                stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
                env=self._env,
                # Host-side backstop, well past the in-guest one.
                timeout=None if timeout is None else timeout + 60,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return ExecOutcome(exit_code=124, output=error.stdout or b"", timed_out=True)
        # busybox timeout -s KILL reports the killed child as 137.
        timed_out = timeout is not None and proc.returncode == 137
        return ExecOutcome(
            exit_code=proc.returncode,
            output=proc.stdout,
            timed_out=timed_out,
            stderr=b"" if merge_stderr else (proc.stderr or b""),
        )

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
        return subprocess.Popen(
            self._exec_command(name, argv, env=env, cwd=cwd, user=user),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=self._env,
        )

    def remove(self, name: str) -> None:
        self._run(["rm", "-f", name], timeout=300)
        with self._ids_lock:
            self._ids.pop(name, None)

    def exists(self, name: str) -> bool:
        try:
            self._run(["container", "inspect", name], timeout=60)
            return True
        except RuntimeError_:
            return False
