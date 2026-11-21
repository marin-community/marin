# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An Iris machine whose controller runs each exec as a local subprocess, with scripted faults."""

import shutil
import subprocess
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import httpx
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from shellbox.backends.iris.machine import EXEC_ADMISSION_REFUSED, EXEC_SETUP_FAILURE, IrisMachine
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec

# Guest images' /bin/sh is a POSIX shell such as dash; macOS's sh is bash, which tolerates more.
GUEST_SH = shutil.which("dash") or "sh"


# The Iris Kubernetes backend serializes a lost exec websocket into the reply's error field.
STREAM_LOST = "Connection to remote host was lost."


class Fault(StrEnum):
    """How one RPC fails; each injection also says whether the controller ran the exec first."""

    CONNECT_REFUSED = "connect-refused"
    """The request never opens a connection."""
    ADMISSION_REFUSED = "admission-refused"
    """The controller's exec pool is full and refuses the request before its handler runs."""
    SETUP_FAILED = "setup-failed"
    """The reply carries the Kubernetes client's WebSocket setup-failure text."""
    SETUP_FAILED_LATE = "setup-failed-late"
    """As ``SETUP_FAILED``, but the admitted exec is still running: it finishes after the controller's next exec."""
    REPLY_TOO_LARGE = "reply-too-large"
    """The generated client rejects a reply above its configured read limit as RESOURCE_EXHAUSTED."""
    LOSE_REPLY = "lose-reply"
    """The reply is lost as UNAVAILABLE with no recognized cause."""
    POST_DROPPED = "post-dropped"
    """The controller consumed the request, then dropped the connection: UNAVAILABLE from a write error."""
    STREAM_LOST = "stream-lost"
    """The exec stream drops; the controller serializes the loss into its reply."""
    STREAM_LOST_RAISED = "stream-lost-raised"
    """The exec stream drops and the loss arrives as a ConnectRPC error."""
    OOM_KILLED = "oom-killed"
    """The sandbox task is OOM-killed: its exec stream drops and the status lookup reports OOMKilled."""


NEVER_AFTER_RUN = {
    Fault.CONNECT_REFUSED: "nothing reaches the controller, so no exec can have run",
    Fault.ADMISSION_REFUSED: "the pool refuses before the exec handler runs",
}
"""Faults that cannot follow an executed exec, and why."""

REPLIES = {
    Fault.SETUP_FAILED: EXEC_SETUP_FAILURE,
    Fault.SETUP_FAILED_LATE: EXEC_SETUP_FAILURE,
    Fault.STREAM_LOST: STREAM_LOST,
    Fault.OOM_KILLED: STREAM_LOST,
}
"""Faults the controller reports in its reply's error field rather than as an RPC error."""

Injection = tuple[str, Fault, bool]
"""An RPC site, its fault, and whether the controller runs the exec before the fault."""


# Every RPC an ``IrisMachine.run`` without stdin can make. Status is the sandbox task lookup; the rest are exec scripts.
RPC_SITES: dict[str, Callable[[str], bool]] = {
    "command": lambda script: "2> /tmp/.shellbox-" in script,
    "status": lambda script: script == "status",
    "stdout-size": lambda script: script.startswith("wc -c") and script.endswith("-out"),
    "stdout-chunk": lambda script: script.startswith("tail -c") and "-out |" in script,
    "stderr-size": lambda script: script.startswith("wc -c") and script.endswith("-err"),
    "stderr-chunk": lambda script: script.startswith("tail -c") and "-err |" in script,
    "cleanup": lambda script: script.startswith("rm -f /tmp/.shellbox-"),
}
# The helpers of ``IrisMachine.upload`` and ``download_file``: create or truncate the target, append a chunk, read one.
TRANSFER_SITES: dict[str, Callable[[str], bool]] = {
    "truncate": lambda script: 'true > "$p"' in script or (script.startswith("mkdir -p") and " && : > " in script),
    "append": lambda script: "| base64 -d >>" in script,
    "read-chunk": lambda script: script.startswith("tail -c") and "/tmp/.shellbox-" not in script,
}
SITES = RPC_SITES | TRANSFER_SITES


def injections(site: str, fault: Fault) -> tuple[Injection, ...]:
    """Each physically possible way ``fault`` can hit ``site``: before the exec ran and, where it can, after."""
    if fault is Fault.SETUP_FAILED_LATE:
        return () if site == "status" else ((site, fault, True),)
    unsent = (site, fault, False)
    return (unsent,) if fault in NEVER_AFTER_RUN or site == "status" else (unsent, (site, fault, True))


def _raise(fault: Fault) -> NoReturn:
    if fault is Fault.CONNECT_REFUSED:
        raise ConnectError(Code.UNAVAILABLE, "connection refused") from httpx.ConnectError("refused")
    if fault is Fault.ADMISSION_REFUSED:
        raise ConnectError(Code.RESOURCE_EXHAUSTED, EXEC_ADMISSION_REFUSED)
    if fault is Fault.REPLY_TOO_LARGE:
        raise ConnectError(Code.RESOURCE_EXHAUSTED, "message is larger than configured max 24")
    if fault is Fault.LOSE_REPLY:
        raise ConnectError(Code.UNAVAILABLE, "reply lost")
    if fault is Fault.POST_DROPPED:
        raise ConnectError(Code.UNAVAILABLE, "server disconnected") from httpx.WriteError("connection reset")
    if fault is Fault.STREAM_LOST_RAISED:
        raise ConnectError(Code.INTERNAL, STREAM_LOST)
    # A status lookup has no reply error field; these faults reach it as the transport's raw error text.
    raise RuntimeError(REPLIES[fault])


class LocalController:
    """Run each exec script locally after ``prelude``; each fault fires once, on the first RPC it selects."""

    def __init__(self, faults: Sequence[Injection] = (), prelude: str = "") -> None:
        self.faults, self.prelude = list(faults), prelude
        self.scripts: list[str] = []
        self.oom_killed = False
        self.late: list[tuple[list[str], str]] = []

    def _fault(self, script: str) -> Injection | None:
        entry = next((entry for entry in self.faults if SITES[entry[0]](script)), None)
        if entry is not None:
            self.faults.remove(entry)
        return entry

    def sent(self, site: str) -> int:
        return sum(SITES[site](script) for script in self.scripts)

    def status(self) -> SimpleNamespace:
        self.scripts.append("status")
        if (entry := self._fault("status")) is not None:
            if entry[1] is not Fault.OOM_KILLED:
                _raise(entry[1])
            self.oom_killed = True
        if self.oom_killed:
            return SimpleNamespace(state=TaskState.FAILED, error_message="OOMKilled")
        return SimpleNamespace(state=TaskState.RUNNING, error_message="")

    def _execute(self, shell: list[str], script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([GUEST_SH, *shell[1:], self.prelude + script], capture_output=True, text=True, timeout=30)

    def exec_in_container(self, request: object, timeout_ms: int) -> SimpleNamespace:
        *shell, script = request.command  # type: ignore[attr-defined]
        self.scripts.append(script)
        entry = self._fault(script)
        if entry is None:
            result = self._execute(shell, script)
            for late in self.late:
                self._execute(*late)
            self.late.clear()
            return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error="")
        _, fault, ran = entry
        if ran and fault is Fault.SETUP_FAILED_LATE:
            self.late.append((shell, script))
        elif ran:
            self._execute(shell, script)
        assert not (ran and fault in NEVER_AFTER_RUN), NEVER_AFTER_RUN.get(fault)
        self.oom_killed = self.oom_killed or fault is Fault.OOM_KILLED
        if fault in REPLIES:
            return SimpleNamespace(exit_code=0, stdout="", stderr="", error=REPLIES[fault])
        _raise(fault)


def local_iris_machine(workdir: Path, controller: LocalController) -> IrisMachine:
    task = SimpleNamespace(task_id=JobName.from_wire("/fixture/sandbox/0"), status=controller.status)
    return IrisMachine(
        SimpleNamespace(close=lambda: None),  # type: ignore[arg-type]
        SimpleNamespace(shutdown=lambda: None),  # type: ignore[arg-type]
        controller,  # type: ignore[arg-type]
        SimpleNamespace(cancel=lambda: None),  # type: ignore[arg-type]
        task,  # type: ignore[arg-type]
        MachineSpec(source=RegistryImage("fixture"), workdir=str(workdir)),
    )
