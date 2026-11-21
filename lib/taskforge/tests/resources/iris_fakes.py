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
from shellbox.backends.iris.machine import EXEC_SETUP_FAILURE, IrisMachine
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec

# Guest images' /bin/sh is a POSIX shell such as dash; macOS's sh is bash, which tolerates more.
GUEST_SH = shutil.which("dash") or "sh"


# The Iris Kubernetes backend serializes a lost exec websocket into the reply's error field.
STREAM_LOST = "Connection to remote host was lost."


class Fault(StrEnum):
    LOSE_REPLY = "lose-reply"
    """The controller runs the exec, then its reply is lost as UNAVAILABLE."""
    STREAM_LOST = "stream-lost"
    """The command runs, then its exec stream drops; the controller serializes the loss into its reply."""
    STREAM_LOST_RAISED = "stream-lost-raised"
    """The command runs, then its exec stream drops and the loss arrives as a ConnectRPC error."""
    SETUP_FAILED = "setup-failed"
    """The exec websocket fails before the command is sent."""
    CONNECT_REFUSED = "connect-refused"
    """The request never opens a connection."""


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


def _raise(fault: Fault) -> NoReturn:
    if fault is Fault.CONNECT_REFUSED:
        raise ConnectError(Code.UNAVAILABLE, "connection refused") from httpx.ConnectError("refused")
    if fault is Fault.LOSE_REPLY:
        raise ConnectError(Code.UNAVAILABLE, "reply lost after the controller ran the exec")
    if fault is Fault.STREAM_LOST_RAISED:
        raise ConnectError(Code.INTERNAL, STREAM_LOST)
    # A status lookup has no reply error field; these faults reach it as the transport's raw error text.
    raise RuntimeError(EXEC_SETUP_FAILURE if fault is Fault.SETUP_FAILED else STREAM_LOST)


class LocalController:
    """Run each exec script locally after ``prelude``; each fault fires once, on the first RPC it selects."""

    def __init__(self, faults: Sequence[tuple[str, Fault]] = (), prelude: str = "") -> None:
        self.faults, self.prelude = list(faults), prelude
        self.scripts: list[str] = []

    def _fault(self, script: str) -> Fault | None:
        entry = next((entry for entry in self.faults if RPC_SITES[entry[0]](script)), None)
        if entry is None:
            return None
        self.faults.remove(entry)
        return entry[1]

    def status(self) -> SimpleNamespace:
        self.scripts.append("status")
        if (fault := self._fault("status")) is not None:
            _raise(fault)
        return SimpleNamespace(state=TaskState.RUNNING, error_message="")

    def exec_in_container(self, request: object, timeout_ms: int) -> SimpleNamespace:
        *shell, script = request.command  # type: ignore[attr-defined]
        self.scripts.append(script)
        fault = self._fault(script)
        if fault is Fault.SETUP_FAILED:
            return SimpleNamespace(exit_code=0, stdout="", stderr="", error=EXEC_SETUP_FAILURE)
        if fault is Fault.CONNECT_REFUSED:
            _raise(fault)
        result = subprocess.run(
            [GUEST_SH, *shell[1:], self.prelude + script], capture_output=True, text=True, timeout=30
        )
        if fault is Fault.STREAM_LOST:
            return SimpleNamespace(exit_code=0, stdout="", stderr="", error=STREAM_LOST)
        if fault is not None:
            _raise(fault)
        return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error="")


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
