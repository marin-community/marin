# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind shell tasks to Shellbox machines without exposing private resources."""

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from shellbox.image import RegistryImage
from shellbox.machine import (
    Backend,
    Command,
    DockerImage,
    HostImage,
    Machine,
    MachineFactory,
    MachineSpec,
    QemuBundle,
    UnsupportedMachineSpec,
)

from taskcompendium.models import (
    EnvironmentRequirements,
    FunctionCall,
    FunctionDefinition,
    OutputDirectory,
    TaskResource,
    TaskSpec,
    grader_workspace,
    require_compatible_backend,
)
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.output_capture import (
    CAPTURE_METADATA_BYTES,
    DIRECTORY_CAPTURE_PROBE,
    DIRECTORY_CAPTURE_SCRIPT,
    selected_directory_files,
    validate_output_directories,
)
from taskcompendium.runtime.resources import resource_bytes

INTERFACE = "shell:v1"
OUTPUT_PATH = "/output/command_capture.txt"
CONTROL_PATH = "/controls/reference.sh"
MISSING_CAPTURE_EXIT_CODE = 44

BASH = FunctionDefinition(
    name="Bash",
    description="Run a shell command in /workspace",
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


def require_environment_source(machine_spec: MachineSpec, environment: EnvironmentRequirements) -> None:
    """Check the machine against the environment: the host for a local environment, otherwise its pinned image."""
    if isinstance(machine_spec.source, HostImage) and Backend.LOCAL in environment.compatible_backends:
        return
    if environment.docker_image is None:
        raise ValueError("The environment names no image for the machine to use")
    require_image(machine_spec, environment.docker_image)


def require_image(machine_spec: MachineSpec, image: str) -> None:
    """Check the task image against a direct reference or staged guest metadata."""
    source = machine_spec.source
    if source in (DockerImage(image), RegistryImage(image)):
        return
    if isinstance(source, QemuBundle):
        metadata = json.loads((source.path / "image.json").read_text())
        if metadata.get("image_reference") == image:
            return
    raise ValueError("Machine must use the task's pinned image")


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


async def captured_output_files(
    machine: Machine, paths: tuple[str, ...], *, timeout: float | None, limit_bytes: int, user: str | None = None
) -> dict[str, bytes]:
    """The regular files at ``paths`` on ``machine``, omitting missing ones.

    Reading through the machine keeps capture sizes bounded, including symlinks. A file that is
    unreadable or larger than ``limit_bytes`` raises ``RuntimeError``.
    """
    files = {}
    for path in paths:
        result = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    f'if [ -f "$1" ]; then head -c "$2" -- "$1"; else exit {MISSING_CAPTURE_EXIT_CODE}; fi',
                    "capture-output",
                    path,
                    str(limit_bytes + 1),
                ),
                timeout=timeout,
                user=user,
                output_limit_bytes=limit_bytes + 1,
            )
        )
        if result.exit_code == MISSING_CAPTURE_EXIT_CODE:
            continue
        if result.exit_code != 0 or result.stdout_truncated or len(result.stdout) > limit_bytes:
            raise RuntimeError(f"Capture unavailable or exceeds budget: {path}")
        files[path] = result.stdout
    return files


@dataclass
class ShellEnvironment:
    machine: Machine
    output_paths: tuple[str, ...]
    command_timeout: float
    output_limit_bytes: int
    output_directories: tuple[OutputDirectory, ...] = ()

    async def step(self, call: FunctionCall) -> str:
        command = call.arguments.get("command")
        if call.name != "Bash" or set(call.arguments) != {"command"} or not isinstance(command, str):
            return json.dumps({"error": "Bash requires one string command"})
        result = await self.machine.run(
            Command(
                ("/bin/bash", "-lc", command),
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
        for selection in self.output_directories:
            result = await self.machine.run(
                Command(
                    (
                        "python3",
                        "-c",
                        DIRECTORY_CAPTURE_SCRIPT,
                        selection.model_dump_json(),
                        str(self.output_limit_bytes),
                        str(CAPTURE_METADATA_BYTES),
                    ),
                    timeout=self.command_timeout,
                    output_limit_bytes=selection.max_bytes * 2 + CAPTURE_METADATA_BYTES,
                )
            )
            if result.exit_code != 0 or result.stdout_truncated:
                raise RuntimeError(
                    f"Directory capture unavailable: {selection.root}; {result.stderr.decode(errors='replace')[-2000:]}"
                )
            captured = {
                path: base64.b64decode(encoded, validate=True) for path, encoded in json.loads(result.stdout).items()
            }
            selected = selected_directory_files(selection, captured)
            if selected != captured:
                raise RuntimeError(f"Directory capture returned files outside selection: {selection.root}")
            # Exact paths retain their existing values and precedence. Unsorted
            # directory insertion order is retained for source discovery.
            for path, data in selected.items():
                files.setdefault(path, data)
            selected_directory_files(selection, files)
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
        require_compatible_backend(task.environment_requirements, self.machine_factory.backend)
        validate_output_directories(task.output_directories, grader_workspace(task.grader))
        if task.output_directories and (
            "python3" not in task.environment_requirements.capabilities
            or self.machine_factory.backend == Backend.SHELLSIM
        ):
            raise UnsupportedMachineSpec("Directory capture requires a real POSIX Python 3 runtime")
        provider = task.environment_requirements.tool_providers.get("shell")
        if provider is None or provider.action_interface != INTERFACE or provider.initial_state != {}:
            raise ValueError("Unsupported shell fixture")
        requirements = task.environment_requirements
        if requirements.docker_image is not None:
            require_image(self.machine_spec, requirements.docker_image)
        if (
            set(requirements.capabilities) - {"shell", "filesystem", "python3"}
            or requirements.working_directory is not None
            or requirements.setup_commands
            or requirements.environment_variables
            or set(requirements.tool_providers) != {"shell"}
        ):
            raise ValueError("Shell factory cannot satisfy these environment requirements")
        machine = await self.machine_factory.create(self.machine_spec)
        try:
            if task.output_directories:
                probe = await machine.run(
                    Command(("python3", "-c", DIRECTORY_CAPTURE_PROBE), timeout=self.command_timeout)
                )
                if probe.exit_code != 0:
                    raise UnsupportedMachineSpec("Directory capture requires a real POSIX Python 3 runtime")
            initialized = await machine.run(
                Command(("mkdir", "-p", self.machine_spec.workdir, "/output"), timeout=self.command_timeout)
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
        except BaseException:
            await machine.close()
            raise
        return ShellEnvironment(
            machine, task.output_paths, self.command_timeout, self.output_limit_bytes, task.output_directories
        )
