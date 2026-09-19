# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct and execute a bounded ShellSim agent workspace."""

import dataclasses
import json
import shlex
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

import shellsim

MAX_WORKSPACE_FILES = 500
MAX_WORKSPACE_FILE_BYTES = 256 * 1024
MAX_WORKSPACE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_WORKSPACE_COMMANDS = 32
MAX_WORKSPACE_COMMAND_BYTES = 16 * 1024
MAX_WORKSPACE_HISTORY_BYTES = 128 * 1024
MAX_IMPORTED_COMMITS = 32
MAX_IMPORTED_HISTORY_BYTES = MAX_WORKSPACE_TOTAL_BYTES + 64 * 1024
MAX_COMMIT_METADATA_BYTES = 4 * 1024
_SHELLSIM_CPU_LIMIT = 50_000_000
_SHELLSIM_MEMORY_LIMIT = 64 * 1024 * 1024
_SHELLSIM_DISK_LIMIT = 16 * 1024 * 1024
_SHELLSIM_OUTPUT_LIMIT = 2 * 1024 * 1024
_WORKSPACE_ROOT = "/work"


@dataclass(frozen=True)
class ImportedGitCommit:
    """One filtered source commit to recreate in ShellSim."""

    message: str
    author_name: str
    author_email: str
    changes: dict[str, str | None]


def imported_git_history_bytes(commits: Iterable[ImportedGitCommit]) -> int:
    """Count UTF-8 bytes charged against the imported-history request limit."""
    return sum(
        len(commit.message.encode())
        + len(commit.author_name.encode())
        + len(commit.author_email.encode())
        + sum(
            len(path.encode()) + (len(content.encode()) if content is not None else 0)
            for path, content in commit.changes.items()
        )
        for commit in commits
    )


@dataclass(frozen=True)
class ShellWorkspaceRequest:
    """One command plus the state required to reconstruct its workspace."""

    files: dict[str, str]
    commits: tuple[ImportedGitCommit, ...]
    history: tuple[str, ...]
    command: str

    def to_json_bytes(self) -> bytes:
        return json.dumps(dataclasses.asdict(self)).encode()

    @classmethod
    def from_payload(cls, payload: object) -> "ShellWorkspaceRequest":
        if not isinstance(payload, dict):
            raise ValueError("Shell workspace request must be an object")
        files = _workspace_files(payload.get("files"))
        commits = _imported_commits(payload.get("commits"), files)
        history = _command_history(payload.get("history"))
        command = payload.get("command")
        if not isinstance(command, str):
            raise ValueError("Shell workspace command must be a string")
        _validate_command(command)
        return cls(files=files, commits=commits, history=history, command=command)


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
        if not is_workspace_path(name):
            raise ValueError(f"Invalid shell workspace path: {name!r}")
        content_bytes = len(content.encode())
        if content_bytes > MAX_WORKSPACE_FILE_BYTES:
            raise ValueError(f"Shell workspace file {name!r} exceeds {MAX_WORKSPACE_FILE_BYTES} bytes")
        total_bytes += len(name.encode()) + content_bytes
        files[name] = content
    if total_bytes > MAX_WORKSPACE_TOTAL_BYTES:
        raise ValueError(f"Shell workspace initial files exceed {MAX_WORKSPACE_TOTAL_BYTES} bytes")
    return files


def is_workspace_path(value: str) -> bool:
    """Return whether a relative path is safe for the simulated workspace."""
    path = PurePosixPath(value)
    return (
        bool(value)
        and "\x00" not in value
        and not path.is_absolute()
        and path.as_posix() == value
        and path != PurePosixPath(".")
        and ".." not in path.parts
        and ".git" not in path.parts
    )


def _imported_commits(value: object, files: dict[str, str]) -> tuple[ImportedGitCommit, ...]:
    if not isinstance(value, list):
        raise ValueError("Imported Git commits must be an array")
    if len(value) > MAX_IMPORTED_COMMITS:
        raise ValueError(f"A shell workspace may import at most {MAX_IMPORTED_COMMITS} Git commits")

    commits: list[ImportedGitCommit] = []
    reconstructed_files: dict[str, str] = {}
    for raw_commit in value:
        if not isinstance(raw_commit, dict):
            raise ValueError("Each imported Git commit must be an object")
        message = _commit_metadata(raw_commit.get("message"), "message")
        author_name = _commit_metadata(raw_commit.get("author_name"), "author name")
        author_email = _commit_metadata(raw_commit.get("author_email"), "author email")
        raw_changes = raw_commit.get("changes")
        if not isinstance(raw_changes, dict) or not raw_changes:
            raise ValueError("Each imported Git commit must contain file changes")
        changes: dict[str, str | None] = {}
        for name, content in raw_changes.items():
            if content is not None and not isinstance(content, str):
                raise ValueError("Imported Git file contents must be strings or null")
            validated = _workspace_files({name: content}) if content is not None else _workspace_files({name: ""})
            path = next(iter(validated))
            changes[path] = content
            if content is None:
                reconstructed_files.pop(path, None)
            else:
                reconstructed_files[path] = content
        _workspace_files(reconstructed_files)
        commits.append(
            ImportedGitCommit(
                message=message,
                author_name=author_name,
                author_email=author_email,
                changes=changes,
            )
        )
    if imported_git_history_bytes(commits) > MAX_IMPORTED_HISTORY_BYTES:
        raise ValueError(f"Imported Git history exceeds {MAX_IMPORTED_HISTORY_BYTES} bytes")
    if commits and reconstructed_files != files:
        raise ValueError("Imported Git commits do not reconstruct the shell workspace files")
    return tuple(commits)


def _commit_metadata(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Imported Git {label} must be a non-empty string")
    if len(value.encode()) > MAX_COMMIT_METADATA_BYTES:
        raise ValueError(f"Imported Git {label} exceeds {MAX_COMMIT_METADATA_BYTES} bytes")
    return value


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


def _seed_workspace(
    environment: shellsim.Environment,
    files: dict[str, str],
    commits: tuple[ImportedGitCommit, ...],
) -> None:
    initialized = environment.run(f"cd {_WORKSPACE_ROOT}; git init")
    if initialized.returncode != 0:
        raise ValueError(f"Could not initialize simulated Git workspace: {initialized.stderr_text.strip()}")
    if commits:
        for commit in commits:
            _apply_imported_commit(environment, commit)
        return
    for name, content in sorted(files.items()):
        _write_workspace_file(environment, name, content)
    if not files:
        return
    committed = environment.run(f"cd {_WORKSPACE_ROOT}; git add .; git commit -m baseline")
    if committed.returncode != 0:
        raise ValueError(f"Could not create simulated Git baseline: {committed.stderr_text.strip()}")


def _apply_imported_commit(environment: shellsim.Environment, commit: ImportedGitCommit) -> None:
    deleted_paths = [path for path, content in commit.changes.items() if content is None]
    written_paths = [path for path, content in commit.changes.items() if content is not None]
    replaced_directories = [
        written_path
        for written_path in written_paths
        if any(deleted_path.startswith(f"{written_path}/") for deleted_path in deleted_paths)
    ]
    removed_paths = set(deleted_paths + replaced_directories)
    if removed_paths:
        quoted_paths = " ".join(shlex.quote(path) for path in sorted(removed_paths, reverse=True))
        removed = environment.run(f"cd {_WORKSPACE_ROOT}; rm -rf -- {quoted_paths}")
        if removed.returncode != 0:
            raise ValueError(f"Could not apply imported Git deletions: {removed.stderr_text.strip()}")
    for name, content in sorted(commit.changes.items()):
        if content is not None:
            _write_workspace_file(environment, name, content)

    author = shlex.quote(f"{commit.author_name} <{commit.author_email}>")
    message = shlex.quote(commit.message)
    committed = environment.run(f"cd {_WORKSPACE_ROOT}; git add -A; git commit --author={author} -m {message}")
    if committed.returncode != 0:
        raise ValueError(f"Could not recreate imported Git commit: {committed.stderr_text.strip()}")


def _write_workspace_file(environment: shellsim.Environment, name: str, content: str) -> None:
    parent = PurePosixPath(name).parent
    if parent != PurePosixPath("."):
        environment.mkdir(f"{_WORKSPACE_ROOT}/{parent}", parents=True)
    environment.write_file(f"{_WORKSPACE_ROOT}/{name}", content)


def shell_command_result(request: ShellWorkspaceRequest) -> ShellCommandResult:
    environment = shellsim.Environment(
        cpu=_SHELLSIM_CPU_LIMIT,
        memory=_SHELLSIM_MEMORY_LIMIT,
        disk=_SHELLSIM_DISK_LIMIT,
        output=_SHELLSIM_OUTPUT_LIMIT,
    )
    _seed_workspace(environment, request.files, request.commits)
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
