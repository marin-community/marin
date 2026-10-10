# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind shell tasks to Shellbox machines without exposing private resources."""

import asyncio
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from shellbox.machine import (
    Command,
    Machine,
    MachineFactory,
    MachineSpec,
    QemuBundle,
)

from taskcompendium.models import (
    DEFAULT_WORKSPACE,
    FunctionCall,
    FunctionDefinition,
    TaskResource,
    TaskSpec,
    validate_output_paths,
)
from taskcompendium.runtime.environment import prepare_machine_spec, validate_machine_spec
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.output_capture import captured_output_files
from taskcompendium.runtime.resources import resource_bytes

OUTPUT_PATH = "/output/command_capture.txt"
CONTROL_PATH = "/controls/reference.sh"

SHELL = FunctionDefinition(
    name="shell",
    description="Run a shell command in the task workspace. Files persist between commands.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
)


def machine_spec_identity(machine_spec: MachineSpec) -> dict[str, Any]:
    """Serialize machine parameters with a JSON-compatible local bundle path."""
    identity = asdict(machine_spec)
    if isinstance(machine_spec.source, QemuBundle):
        identity["source"] = {"path": str(machine_spec.source.path)}
    return identity


async def upload_resources(machine: Machine, resources: Sequence[TaskResource], timeout: float) -> None:
    """Write task resources at their absolute paths, applying declared file modes."""
    with TemporaryDirectory() as directory:
        for index, resource in enumerate(resources):
            if resource.mtime_ns is not None:
                raise ValueError("Shell factory cannot mount resource timestamps")
            local = Path(directory) / str(index)
            local.write_bytes(resource_bytes(resource))
            target = f"/{resource.path}"
            await machine.upload(local, target)
            if resource.mode is not None:
                permissions = await machine.run(Command(("chmod", resource.mode, target), timeout=timeout))
                if permissions.exit_code != 0:
                    raise RuntimeError(f"Could not set resource permissions: {target}")


@dataclass
class ShellEnvironment:
    machine: Machine
    output_paths: tuple[str, ...]
    command_timeout: float
    output_limit_bytes: int
    workdir: str = DEFAULT_WORKSPACE

    async def step(self, call: FunctionCall) -> str:
        command = call.arguments.get("command")
        if call.name != SHELL.name or set(call.arguments) != {"command"} or not isinstance(command, str):
            return json.dumps({"error": "shell requires one string command"})
        result = await self.machine.run(
            Command(
                ("bash", "-c", command),
                cwd=self.workdir,
                timeout=self.command_timeout,
                output_limit_bytes=self.output_limit_bytes,
            )
        )
        return json.dumps(
            {
                "exit_code": result.exit_code,
                "reason": result.reason.value,
                "stdout": result.stdout.decode(errors="replace"),
                "stderr": result.stderr.decode(errors="replace"),
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            }
        )

    async def evidence(self) -> RuntimeEvidence:
        files = await captured_output_files(
            self.machine, self.output_paths, timeout=self.command_timeout, limit_bytes=self.output_limit_bytes
        )
        return RuntimeEvidence(files, "{}")

    async def close(self) -> None:
        await self.machine.close()


@dataclass(frozen=True)
class ShellFactory:
    machine_factory: MachineFactory
    machine_spec: MachineSpec
    backend_identity: dict
    command_timeout: float
    output_limit_bytes: int
    mounted_roles: tuple[Literal["worker", "oracle"], ...] = ("worker",)

    @property
    def identity(self) -> dict:
        return {
            **self.backend_identity,
            "backend": self.machine_factory.backend.value,
            "workdir": self.machine_spec.workdir,
            "network": self.machine_spec.network.value,
            "memory_mb": self.machine_spec.memory_mb,
            "env_sha256": hashlib.sha256(json.dumps(self.machine_spec.env, sort_keys=True).encode()).hexdigest(),
            "command_timeout": self.command_timeout,
            "output_limit_bytes": self.output_limit_bytes,
            "mounted_roles": self.mounted_roles,
        }

    async def create(self, task: TaskSpec) -> ShellEnvironment:
        validate_machine_spec(task.environment_requirements, self.machine_factory, self.machine_spec)
        validate_output_paths(task.output_paths)
        requirements = task.environment_requirements
        if (
            set(requirements.capabilities) - {"shell", "filesystem"}
            or requirements.tool_providers
            or task.interaction_tools
        ):
            raise ValueError("Shell factory cannot satisfy these environment requirements")
        async with asyncio.timeout(self.machine_spec.startup_timeout):
            prepared = await asyncio.to_thread(
                prepare_machine_spec,
                requirements,
                self.machine_factory,
                self.machine_spec,
                dict(os.environ),
            )
            machine = await self.machine_factory.create(prepared)
        try:
            initialized = await machine.run(
                Command(("mkdir", "-p", prepared.workdir, "/output"), timeout=self.command_timeout)
            )
            if initialized.exit_code != 0:
                raise RuntimeError("Could not initialize shell workspace")
            roles = {
                "worker": task.resources.worker,
                "oracle": task.resources.oracle,
            }
            resources = list(task.resources.all)
            for role in self.mounted_roles:
                resources.extend(roles[role])
            await upload_resources(machine, resources, self.command_timeout)
            for setup in requirements.setup_commands:
                result = await machine.run(Command(("sh", "-c", setup), timeout=self.command_timeout, user="0"))
                if result.exit_code != 0:
                    raise RuntimeError("Shell environment setup failed")
        except BaseException:
            await machine.close()
            raise
        return ShellEnvironment(
            machine,
            task.output_paths,
            self.command_timeout,
            self.output_limit_bytes,
            prepared.workdir,
        )
