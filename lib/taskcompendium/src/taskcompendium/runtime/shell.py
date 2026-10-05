# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind shell tasks to Shellbox machines without exposing private resources."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from shellbox.machine import Command, Machine, MachineFactory, MachineSpec

from taskcompendium.models import FunctionCall, TaskSpec
from taskcompendium.runtime.models import RuntimeEvidence

INTERFACE = "shell:v1"
OUTPUT_PATH = "/output/command_capture.txt"
CONTROL_PATH = "/controls/reference.sh"
MISSING_CAPTURE_EXIT_CODE = 44


@dataclass
class ShellEnvironment:
    machine: Machine
    output_paths: tuple[str, ...]
    command_timeout: float
    output_limit_bytes: int

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
        files = {}
        for path in self.output_paths:
            # Reading through the machine keeps capture sizes bounded, including symlinks.
            result = await self.machine.run(
                Command(
                    (
                        "/bin/bash",
                        "-c",
                        f'if test -f "$1"; then head -c "$2" -- "$1"; else exit {MISSING_CAPTURE_EXIT_CODE}; fi',
                        "capture",
                        path,
                        str(self.output_limit_bytes + 1),
                    ),
                    timeout=self.command_timeout,
                    output_limit_bytes=self.output_limit_bytes + 1,
                )
            )
            if result.exit_code == MISSING_CAPTURE_EXIT_CODE:
                continue
            if result.exit_code != 0 or result.stdout_truncated or len(result.stdout) > self.output_limit_bytes:
                raise RuntimeError(f"Capture unavailable or exceeds budget: {path}")
            files[path] = result.stdout
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
            "workdir": self.machine_spec.workdir,
            "network": self.machine_spec.network.value,
            "memory_mb": self.machine_spec.memory_mb,
            "env_sha256": hashlib.sha256(json.dumps(self.machine_spec.env, sort_keys=True).encode()).hexdigest(),
            "command_timeout": self.command_timeout,
            "output_limit_bytes": self.output_limit_bytes,
            "mounted_roles": self.mounted_roles,
        }

    async def create(self, task: TaskSpec) -> ShellEnvironment:
        environment = task.environment
        provider = environment.tool_providers.get("shell")
        if provider is None or provider.action_interface != INTERFACE or provider.initial_state != {}:
            raise ValueError("Unsupported shell fixture")
        requirements = task.environment_requirements
        if (
            set(requirements.capabilities) - {"shell", "filesystem"}
            or environment.image is not None
            or environment.setup
            or environment.env
            or set(environment.tool_providers) != {"shell"}
        ):
            raise ValueError("Shell factory cannot satisfy these environment requirements")
        machine = await self.machine_factory.create(self.machine_spec)
        try:
            initialized = await machine.run(
                Command(("mkdir", "-p", self.machine_spec.workdir, "/output"), timeout=self.command_timeout)
            )
            if initialized.exit_code != 0:
                raise RuntimeError("Could not initialize shell workspace")
            with TemporaryDirectory() as directory:
                roles = {
                    "worker": environment.files,
                    "oracle": task.oracle_files,
                }
                resources = []
                for role in self.mounted_roles:
                    resources.extend(roles[role])
                for index, resource in enumerate(resources):
                    if resource.mode != 0o644 or resource.mtime_ns is not None:
                        raise ValueError("Shell factory cannot mount resource metadata")
                    local = Path(directory) / str(index)
                    local.write_bytes(resource.content)
                    await machine.upload(local, resource.path)
        except BaseException:
            await machine.close()
            raise
        return ShellEnvironment(machine, task.output_paths, self.command_timeout, self.output_limit_bytes)
