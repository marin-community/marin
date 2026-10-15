# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind shell tasks to Shellbox machines without exposing private resources."""

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from shellbox.machine import (
    DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    Command,
    ExitReason,
    Machine,
    MachineFactory,
    MachineSpec,
    QemuBundle,
    Result,
)

from taskcompendium.models import (
    ANSWER_CALL_NAME,
    DEFAULT_WORKSPACE,
    AnswerCall,
    FunctionCall,
    FunctionDefinition,
    TaskResource,
    TaskSpec,
    validate_output_paths,
)
from taskcompendium.runtime.environment import prepare_machine_spec, setup_user, validate_machine_spec
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.output_capture import captured_output_files
from taskcompendium.runtime.staging import resource_files, write_archive

OUTPUT_PATH = "/output/command_capture.txt"
CONTROL_PATH = "/controls/reference.sh"
SPILL_DIRECTORY = "/tmp/tool-output"
SPILL_FILE_BYTES = 64 * 1024 * 1024
"""Bytes each spilled stream file keeps; output past it is read and discarded, and the command keeps running."""
# Arguments: $0 is the spill path prefix, $1 the command, $2 the preview read size, $3 the spill threshold,
# and $4 the per-file cap. Each stream passes through a pipeline that saves its first $4 bytes and drains the
# rest, so the files stay bounded and the shell waits for both before reading previews. The host reads only
# each file's head, and files within the threshold are removed.
SPILL_SCRIPT = (
    'mkdir -p "${0%/*}" || exit; set -o pipefail; '
    '{ { bash -c "$1" 2>&1 1>&3 3>&- | { head -c "$4" >"$0.stderr"; cat >/dev/null; }; } 3>&1 1>&2 '
    '| { head -c "$4" >"$0.stdout"; cat >/dev/null; }; }; status=$?; '
    'head -c "$2" "$0.stdout"; head -c "$2" "$0.stderr" >&2; '
    'for stream in stdout stderr; do [ "$(wc -c <"$0.$stream")" -gt "$3" ] || rm -f "$0.$stream"; done; '
    'exit "$status"'
)


class ShellToolConfig(BaseModel):
    """Harness presentation of command execution, independent of task semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="shell", min_length=1)
    command_parameter: str = Field(default="command", min_length=1)


@dataclass(frozen=True)
class BoundShellTool:
    """The advertised definition and the argument the Bash executor consumes."""

    definition: FunctionDefinition
    command_parameter: str


def resolve_shell_tool(task: TaskSpec, config: ShellToolConfig) -> BoundShellTool | None:
    """Preserve an explicit task interface, or bind the harness shell capability."""
    names = {tool.name for tool in task.final_tools}
    if isinstance(task.answer_format, AnswerCall):
        names.add(ANSWER_CALL_NAME)
    if task.tool_bindings:
        if len(task.tool_bindings) != 1:
            raise ValueError("The shell session requires exactly one shell binding")
        name, binding = next(iter(task.tool_bindings.items()))
        definitions = [tool for tool in task.interaction_tools if tool.name == name]
        if len(definitions) != 1 or name in names:
            raise ValueError(f"Shell binding must name one unambiguous interaction tool: {name}")
        definition = definitions[0]
        parameters = definition.parameters
        properties = parameters.get("properties")
        parameter = binding.command_parameter
        allowed_annotations = {"title", "description"}
        if (
            parameters.get("type") != "object"
            or parameters.get("required") != [parameter]
            or parameters.get("additionalProperties") is not False
            or set(parameters) - {"type", "properties", "required", "additionalProperties"} - allowed_annotations
            or not isinstance(properties, dict)
            or set(properties) != {parameter}
        ):
            raise ValueError("Shell tools require exactly one required string argument and no additional properties")
        argument = properties[parameter]
        if (
            not isinstance(argument, dict)
            or argument.get("type") != "string"
            or set(argument) - {"type"} - allowed_annotations
        ):
            raise ValueError("Shell command arguments must be unconstrained strings")
        return BoundShellTool(definition, parameter)
    if "shell" not in task.environment_requirements.capabilities:
        return None
    names.update(tool.name for tool in task.interaction_tools)
    if config.name in names:
        raise ValueError(f"Shell tool collides with a task-owned tool: {config.name}")
    return BoundShellTool(
        FunctionDefinition(
            name=config.name,
            description="Run a Bash command in the task workspace. Files persist between commands.",
            parameters={
                "type": "object",
                "properties": {config.command_parameter: {"type": "string"}},
                "required": [config.command_parameter],
                "additionalProperties": False,
            },
        ),
        config.command_parameter,
    )


async def run_shell_call(
    machine: Machine,
    call: FunctionCall,
    tool: BoundShellTool,
    *,
    timeout: float | None,
    output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    cwd: str | None = None,
    spill_bytes: int | None = None,
) -> str:
    """Decode a harness call and return the Bash command's observation.

    With ``spill_bytes``, each stream is written to a file under ``SPILL_DIRECTORY`` that keeps at most
    ``SPILL_FILE_BYTES``. A stream longer than ``spill_bytes`` keeps its file, and the observation holds
    its first ``spill_bytes`` bytes and its path in ``stdout_path`` or ``stderr_path``. A timed-out
    command reports both paths, since its files can hold output the machine did not return.
    """
    command = call.arguments.get(tool.command_parameter)
    if (
        call.name != tool.definition.name
        or set(call.arguments) != {tool.command_parameter}
        or not isinstance(command, str)
    ):
        return json.dumps({"error": f"{tool.definition.name} requires one string {tool.command_parameter}"})
    if spill_bytes is None:
        result = await machine.run(
            Command(("bash", "-c", command), cwd=cwd, timeout=timeout, output_limit_bytes=output_limit_bytes)
        )
        return json.dumps(
            _observation(result, result.stdout, result.stderr, result.stdout_truncated, result.stderr_truncated)
        )
    prefix = f"{SPILL_DIRECTORY}/{uuid.uuid4().hex}"
    # One byte past the preview tells the host that the stream exceeded the threshold.
    read_bytes = spill_bytes + 1
    result = await machine.run(
        Command(
            ("bash", "-c", SPILL_SCRIPT, prefix, command, str(read_bytes), str(spill_bytes), str(SPILL_FILE_BYTES)),
            cwd=cwd,
            timeout=timeout,
            output_limit_bytes=min(output_limit_bytes, read_bytes),
        )
    )
    timed_out = result.reason == ExitReason.TIMED_OUT
    stdout_spilled = result.stdout_truncated or len(result.stdout) > spill_bytes
    stderr_spilled = result.stderr_truncated or len(result.stderr) > spill_bytes
    return json.dumps(
        {
            **_observation(
                result, result.stdout[:spill_bytes], result.stderr[:spill_bytes], stdout_spilled, stderr_spilled
            ),
            "stdout_path": f"{prefix}.stdout" if stdout_spilled or timed_out else None,
            "stderr_path": f"{prefix}.stderr" if stderr_spilled or timed_out else None,
        }
    )


def _observation(
    result: Result, stdout: bytes, stderr: bytes, stdout_truncated: bool, stderr_truncated: bool
) -> dict[str, object]:
    return {
        "exit_code": result.exit_code,
        "reason": result.reason.value,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


def machine_spec_identity(machine_spec: MachineSpec) -> dict[str, Any]:
    """Serialize machine parameters with a JSON-compatible local bundle path."""
    identity = asdict(machine_spec)
    if isinstance(machine_spec.source, QemuBundle):
        identity["source"] = {"path": str(machine_spec.source.path)}
    return identity


async def upload_resources(
    machine: Machine, resources: Sequence[TaskResource], timeout: float, *, user: str | None
) -> None:
    """Write task resources at their absolute paths, applying declared file modes."""
    if not resources:
        return
    with TemporaryDirectory() as directory:
        for resource in resources:
            if resource.mtime_ns is not None:
                raise ValueError("Shell factory cannot mount resource timestamps")
        archive = Path(directory) / "resources.tar"
        target = f"/tmp/taskcompendium-resources-{uuid.uuid4().hex}.tar"
        # Batch resources into one upload to avoid a machine RPC for every file.
        # File-only entries preserve existing guest directory ownership and modes.
        write_archive(archive, resource_files(resources, ""), [])
        await machine.upload(archive, target)
        for arguments in (("tar", "-xf", target, "-C", "/"), ("rm", "-f", target)):
            result = await machine.run(Command(arguments, cwd="/", timeout=timeout, user=user))
            if result.exit_code != 0:
                raise RuntimeError(f"Could not stage task resources: {result.stderr.decode(errors='replace')}")


@dataclass
class ShellEnvironment:
    machine: Machine
    output_paths: tuple[str, ...]
    command_timeout: float
    output_limit_bytes: int
    workdir: str = DEFAULT_WORKSPACE
    shell_tool: BoundShellTool | None = None

    @property
    def tools(self) -> tuple[FunctionDefinition, ...]:
        return () if self.shell_tool is None else (self.shell_tool.definition,)

    async def step(self, call: FunctionCall) -> str:
        if self.shell_tool is None:
            return json.dumps({"error": "No shell tool is available"})
        return await run_shell_call(
            self.machine,
            call,
            self.shell_tool,
            timeout=self.command_timeout,
            output_limit_bytes=self.output_limit_bytes,
            cwd=self.workdir,
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
    shell_tool: ShellToolConfig = field(default_factory=ShellToolConfig)

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
            "shell_tool": self.shell_tool.model_dump(),
        }

    async def create(self, task: TaskSpec) -> ShellEnvironment:
        tool = resolve_shell_tool(task, self.shell_tool)
        validate_machine_spec(task.environment_requirements, self.machine_factory, self.machine_spec)
        validate_output_paths(task.output_paths)
        requirements = task.environment_requirements
        if requirements.tool_providers or any(
            tool is None or definition.name != tool.definition.name for definition in task.interaction_tools
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
            await upload_resources(
                machine, resources, self.command_timeout, user=setup_user(self.machine_factory.backend)
            )
            for setup in requirements.setup_commands:
                result = await machine.run(
                    Command(
                        ("sh", "-c", setup), timeout=self.command_timeout, user=setup_user(self.machine_factory.backend)
                    )
                )
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
            tool,
        )
