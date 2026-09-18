# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct and execute a bounded ShellSim agent workspace."""

import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

import shellsim

MAX_WORKSPACE_FILES = 500
MAX_WORKSPACE_FILE_BYTES = 256 * 1024
MAX_WORKSPACE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_WORKSPACE_COMMANDS = 32
MAX_WORKSPACE_COMMAND_BYTES = 16 * 1024
MAX_WORKSPACE_HISTORY_BYTES = 128 * 1024
_SHELLSIM_CPU_LIMIT = 50_000_000
_SHELLSIM_MEMORY_LIMIT = 64 * 1024 * 1024
_SHELLSIM_DISK_LIMIT = 16 * 1024 * 1024
_SHELLSIM_OUTPUT_LIMIT = 2 * 1024 * 1024
_WORKSPACE_ROOT = "/work"


@dataclass(frozen=True)
class ShellWorkspaceRequest:
    """One command plus the state required to reconstruct its workspace."""

    files: dict[str, str]
    history: tuple[str, ...]
    command: str

    def to_json_bytes(self) -> bytes:
        return json.dumps(dataclasses.asdict(self)).encode()

    @classmethod
    def from_payload(cls, payload: object) -> "ShellWorkspaceRequest":
        if not isinstance(payload, dict):
            raise ValueError("Shell workspace request must be an object")
        files = _workspace_files(payload.get("files"))
        history = _command_history(payload.get("history"))
        command = payload.get("command")
        if not isinstance(command, str):
            raise ValueError("Shell workspace command must be a string")
        _validate_command(command)
        return cls(files=files, history=history, command=command)


@dataclass(frozen=True)
class ShellCommandResult:
    exit_code: int
    stdout: str
    stderr: str
    stop_reason: str | None
    unsupported: tuple[str, ...]
    partial_commands: tuple[str, ...]

    def to_json_bytes(self) -> bytes:
        return json.dumps(dataclasses.asdict(self)).encode()


def _workspace_files(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Shell workspace files must be an object mapping paths to text")
    if len(value) > MAX_WORKSPACE_FILES:
        raise ValueError(f"Shell workspace may contain at most {MAX_WORKSPACE_FILES} initial files")

    files: dict[str, str] = {}
    total_bytes = 0
    for name, content in value.items():
        if not isinstance(name, str) or not isinstance(content, str):
            raise ValueError("Shell workspace file paths and contents must be strings")
        path = PurePosixPath(name)
        if (
            not name
            or "\x00" in name
            or path.is_absolute()
            or path.as_posix() != name
            or path == PurePosixPath(".")
            or ".." in path.parts
            or ".git" in path.parts
        ):
            raise ValueError(f"Invalid shell workspace path: {name!r}")
        content_bytes = len(content.encode())
        if content_bytes > MAX_WORKSPACE_FILE_BYTES:
            raise ValueError(f"Shell workspace file {name!r} exceeds {MAX_WORKSPACE_FILE_BYTES} bytes")
        total_bytes += len(name.encode()) + content_bytes
        files[name] = content
    if total_bytes > MAX_WORKSPACE_TOTAL_BYTES:
        raise ValueError(f"Shell workspace initial files exceed {MAX_WORKSPACE_TOTAL_BYTES} bytes")
    return files


def _command_history(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(command, str) for command in value):
        raise ValueError("Shell workspace history must be an array of command strings")
    if len(value) > MAX_WORKSPACE_COMMANDS:
        raise ValueError(f"Shell workspace history may contain at most {MAX_WORKSPACE_COMMANDS} commands")
    for command in value:
        _validate_command(command)
    if sum(len(command.encode()) for command in value) > MAX_WORKSPACE_HISTORY_BYTES:
        raise ValueError(f"Shell workspace history exceeds {MAX_WORKSPACE_HISTORY_BYTES} bytes")
    return tuple(value)


def _validate_command(command: str) -> None:
    if not command.strip():
        raise ValueError("Shell workspace command may not be empty")
    if len(command.encode()) > MAX_WORKSPACE_COMMAND_BYTES:
        raise ValueError(f"Shell workspace command exceeds {MAX_WORKSPACE_COMMAND_BYTES} bytes")


def _seed_workspace(environment: shellsim.Environment, files: dict[str, str]) -> None:
    for name, content in sorted(files.items()):
        parent = PurePosixPath(name).parent
        if parent != PurePosixPath("."):
            environment.mkdir(f"{_WORKSPACE_ROOT}/{parent}", parents=True)
        environment.write_file(f"{_WORKSPACE_ROOT}/{name}", content)

    initialized = environment.run(f"cd {_WORKSPACE_ROOT}; git init")
    if initialized.returncode != 0:
        raise ValueError(f"Could not initialize simulated Git workspace: {initialized.stderr_text.strip()}")
    if files:
        committed = environment.run("git add .; git commit -m baseline")
        if committed.returncode != 0:
            raise ValueError(f"Could not create simulated Git baseline: {committed.stderr_text.strip()}")


def shell_command_result(request: ShellWorkspaceRequest) -> ShellCommandResult:
    environment = shellsim.Environment(
        cpu=_SHELLSIM_CPU_LIMIT,
        memory=_SHELLSIM_MEMORY_LIMIT,
        disk=_SHELLSIM_DISK_LIMIT,
        output=_SHELLSIM_OUTPUT_LIMIT,
    )
    _seed_workspace(environment, request.files)
    for index, command in enumerate(request.history, start=1):
        replayed = environment.run(command)
        if environment.terminated:
            reason = replayed.stop_reason or "resource limit"
            raise ValueError(f"Shell workspace history command {index} terminated the simulation: {reason}")

    result = environment.run(request.command)
    return ShellCommandResult(
        exit_code=result.returncode,
        stdout=result.stdout_text,
        stderr=result.stderr_text,
        stop_reason=result.stop_reason,
        unsupported=tuple(result.unsupported),
        partial_commands=tuple(result.partial_commands),
    )


def _main() -> None:
    if len(sys.argv) != 1:
        raise ValueError(f"Expected no shell workspace arguments, got {sys.argv[1:]!r}")
    request = ShellWorkspaceRequest.from_payload(json.load(sys.stdin))
    sys.stdout.buffer.write(shell_command_result(request).to_json_bytes())


if __name__ == "__main__":
    _main()
