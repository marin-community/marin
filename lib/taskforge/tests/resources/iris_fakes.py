# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An Iris machine whose controller runs each exec as a local subprocess, with scripted faults."""

import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from shellbox.backends.iris.machine import IrisMachine
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec


class LocalController:
    """Run each exec script locally after ``prelude``; the first script ``lose_reply`` selects runs, then fails."""

    def __init__(self, lose_reply: Callable[[str], bool] = lambda script: False, prelude: str = "") -> None:
        self.lose_reply, self.prelude = lose_reply, prelude
        self.lost = False

    def exec_in_container(self, request: object, timeout_ms: int) -> SimpleNamespace:
        *shell, script = request.command  # type: ignore[attr-defined]
        result = subprocess.run([*shell, self.prelude + script], capture_output=True, text=True, timeout=30)
        if not self.lost and self.lose_reply(script):
            self.lost = True
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
