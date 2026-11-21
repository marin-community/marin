# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An Iris machine whose controller runs each exec as a local subprocess, with scripted faults."""

import shutil
import subprocess
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

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


class Fault(StrEnum):
    LOSE_REPLY = "lose-reply"
    """The controller runs the exec, then its reply is lost."""
    SETUP_FAILED = "setup-failed"
    """The exec websocket fails before the command is sent."""
    CONNECT_REFUSED = "connect-refused"
    """The request never opens a connection."""


class LocalController:
    """Run each exec script locally after ``prelude``; each fault fires once, on the first script it selects."""

    def __init__(self, faults: Sequence[tuple[Callable[[str], bool], Fault]] = (), prelude: str = "") -> None:
        self.faults, self.prelude = list(faults), prelude
        self.scripts: list[str] = []

    def exec_in_container(self, request: object, timeout_ms: int) -> SimpleNamespace:
        *shell, script = request.command  # type: ignore[attr-defined]
        self.scripts.append(script)
        fault = next((entry for entry in self.faults if entry[0](script)), None)
        if fault is not None:
            self.faults.remove(fault)
        if fault is not None and fault[1] is Fault.SETUP_FAILED:
            return SimpleNamespace(exit_code=0, stdout="", stderr="", error=EXEC_SETUP_FAILURE)
        if fault is not None and fault[1] is Fault.CONNECT_REFUSED:
            raise ConnectError(Code.UNAVAILABLE, "connection refused") from httpx.ConnectError("refused")
        result = subprocess.run(
            [GUEST_SH, *shell[1:], self.prelude + script], capture_output=True, text=True, timeout=30
        )
        if fault is not None:
            raise ConnectError(Code.UNAVAILABLE, "reply lost after the controller ran the exec")
        return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error="")


def local_iris_machine(workdir: Path, controller: LocalController) -> IrisMachine:
    task = SimpleNamespace(
        task_id=JobName.from_wire("/fixture/sandbox/0"), status=lambda: SimpleNamespace(state=TaskState.RUNNING)
    )
    return IrisMachine(
        SimpleNamespace(close=lambda: None),  # type: ignore[arg-type]
        SimpleNamespace(shutdown=lambda: None),  # type: ignore[arg-type]
        controller,  # type: ignore[arg-type]
        SimpleNamespace(cancel=lambda: None),  # type: ignore[arg-type]
        task,  # type: ignore[arg-type]
        MachineSpec(source=RegistryImage("fixture"), workdir=str(workdir)),
    )
