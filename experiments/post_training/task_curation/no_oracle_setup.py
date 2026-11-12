# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a rebuilt setup script in Docker or the lowered runtime and compare its writes.

Preservation proves the staged bytes travel; this proves the setup script lands
them. Each execution runs the declared setup commands in the task's own base
image, then scans the workspace the native recipe's COPY destinations define.
"""

import base64
import shlex
import subprocess
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.calibforge_envelope import LoweredSpec
from experiments.post_training.task_curation.calibforge_fixtures import SETUP_ROOT
from experiments.post_training.task_curation.calibforge_rebuild import setup_script_name
from experiments.post_training.task_curation.environment_fidelity import DEFAULT_WORKDIR
from experiments.post_training.task_curation.no_oracle_liveness import DETAIL_LIMIT
from experiments.post_training.task_curation.no_oracle_preservation import (
    FileExpectation,
    _compare_workspace,
    _native_dockerfile,
    copied_destinations,
    native_workdir,
    workspace_expectations,
)
from experiments.post_training.task_curation.task_tars import open_task_tar


def _docker(*arguments: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, **kwargs)


class SetupOutcome(StrEnum):
    """How one rebuilt setup execution ended."""

    EXECUTED = "executed"
    NO_SETUP_COMMANDS = "no_setup_commands"
    MISSING_SETUP_SCRIPT = "missing_setup_script"
    DOCKER_ERROR = "docker_error"
    TIMEOUT = "timeout"
    SCRIPT_ERROR = "script_error"
    INVALID_TASK = "invalid_task"


@dataclass(frozen=True)
class SetupExecution:
    """One rebuilt setup script executed in its base image."""

    task_id: str
    outcome: SetupOutcome
    written: int = 0
    missing: tuple[str, ...] = ()
    """Expected paths the workspace scan found nowhere."""
    altered: tuple[str, ...] = ()
    """Expected paths present with different bytes."""
    extra: tuple[str, ...] = ()
    remoded: tuple[str, ...] = ()
    workspace_stdout: str | None = None
    """The raw workspace scan output, when the script ran to completion."""
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Whether every expected file is present with the native bytes.

        A mode difference is reported in ``remoded`` and does not fail the check,
        because the native recipe's own ``chmod`` steps are replayed verbatim by the
        rebuild, so a faithful rebuild legitimately differs from the source byte mode.
        """
        return self.outcome == SetupOutcome.EXECUTED and not (self.missing or self.altered)


DOCKER_SETUP_TIMEOUT = 300
"""Seconds one rebuilt setup script may run in its base image."""

APT_SHIM = """#!/bin/sh
# Skip package installation during payload-only diagnostics.
exit 0
"""


def _execute_container_setup(
    task_id: str,
    image: str,
    command: str,
    copies: tuple[tuple[Path, str], ...],
    dns: tuple[str, ...],
    timeout: int,
) -> SetupExecution:
    resolvers = tuple(part for server in dns for part in ("--dns", server))
    created = _docker("create", *resolvers, image, "sh", "-c", command)
    if created.returncode:
        return SetupExecution(task_id=task_id, outcome=SetupOutcome.DOCKER_ERROR, detail=created.stderr[-DETAIL_LIMIT:])
    container = created.stdout.strip()
    try:
        for source, destination in copies:
            copied = _docker("cp", str(source), f"{container}:{destination}")
            if copied.returncode:
                return SetupExecution(
                    task_id=task_id, outcome=SetupOutcome.DOCKER_ERROR, detail=copied.stderr[-DETAIL_LIMIT:]
                )
        started = _docker("start", "-a", container, timeout=timeout)
    except subprocess.TimeoutExpired:
        return SetupExecution(task_id=task_id, outcome=SetupOutcome.TIMEOUT, detail=f"{timeout}s")
    finally:
        removed = _docker("rm", "-f", container)
        if removed.returncode:
            raise RuntimeError(f"cannot remove Docker setup container {container}: {removed.stderr[-DETAIL_LIMIT:]}")
    if started.returncode:
        return SetupExecution(task_id=task_id, outcome=SetupOutcome.SCRIPT_ERROR, detail=started.stderr[-DETAIL_LIMIT:])
    return SetupExecution(task_id=task_id, outcome=SetupOutcome.EXECUTED, workspace_stdout=started.stdout)


def _workspace_scan(roots: tuple[str, ...]) -> str:
    paths = " ".join(shlex.quote(path) for path in dict.fromkeys(roots))
    return (
        f'for root in {paths}; do if [ -d "$root" ]; then '
        '(find "$root" -type f -printf \'%m %p\\n\' && find "$root" -type f -exec sha256sum {} +); '
        'elif [ -e "$root" ]; then (stat -c \'%a %n\' "$root" && sha256sum "$root"); fi; done'
    )


def _setup_comparison(task_id: str, expectations: tuple[FileExpectation, ...], stdout: str) -> SetupExecution:
    comparison = _compare_workspace(expectations, stdout)
    return SetupExecution(
        task_id=task_id,
        outcome=SetupOutcome.EXECUTED,
        written=comparison.written,
        missing=comparison.missing,
        altered=comparison.altered,
        extra=comparison.extra,
        remoded=comparison.remoded,
        workspace_stdout=stdout,
    )


def execute_setup_in_docker(
    task_id: str,
    native_payload: bytes,
    rebuilt_payload: bytes,
    *,
    image: str,
    workdir: str,
    setup_script: str,
    destinations: tuple[str, ...] = (),
    stub_apt: bool = False,
    dns: tuple[str, ...] = (),
    timeout: int = DOCKER_SETUP_TIMEOUT,
) -> SetupExecution:
    """Run one rebuilt setup script in its base image and compare the written files."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with open_task_tar(rebuilt_payload) as archive:
            archive.extractall(root, filter="data")
        setup = root / SETUP_ROOT.lstrip("/")
        script = setup / setup_script
        if not script.exists():
            return SetupExecution(task_id=task_id, outcome=SetupOutcome.MISSING_SETUP_SCRIPT)
        shim = root / "shim"
        shim.mkdir()
        for tool in ("apt-get", "apt"):
            (shim / tool).write_text(APT_SHIM)
            (shim / tool).chmod(0o755)
        command = (
            ("PATH=/shim:$PATH; " if stub_apt else "")
            + f"sh {SETUP_ROOT}/{setup_script} && "
            + _workspace_scan((workdir, *destinations))
        )
        result = _execute_container_setup(task_id, image, command, ((setup, SETUP_ROOT), (shim, "/shim")), dns, timeout)
    if result.workspace_stdout is None:
        return result
    return _setup_comparison(task_id, workspace_expectations(native_payload), result.workspace_stdout)


def execute_lowered_setup(
    task: TaskSpec,
    expectations: tuple[FileExpectation, ...],
    *,
    destinations: tuple[str, ...] = (),
    timeout: int = DOCKER_SETUP_TIMEOUT,
    dns: tuple[str, ...] = (),
) -> SetupExecution:
    """Run a lowered task's setup commands in its image and compare the written files.

    Resources carry the fixtures and ``environment_requirements.setup_commands``
    stage them. This executes exactly those commands, so a task that declares a
    hook nothing runs cannot pass.
    """
    requirements = task.environment_requirements
    commands = tuple(requirements.setup_commands or ())
    if not commands:
        return SetupExecution(task_id=task.id, outcome=SetupOutcome.NO_SETUP_COMMANDS)
    workdir = requirements.working_directory or DEFAULT_WORKDIR
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for resource in task.resources.all or ():
            target = root / "setup_files" / Path(resource.path).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(resource.source.content_base64, validate=True))
            if resource.mode is not None:
                target.chmod(int(resource.mode, 8))
        command = " && ".join(commands) + " && " + _workspace_scan((workdir, *destinations))
        result = _execute_container_setup(
            task.id, requirements.docker_image, command, ((root / "setup_files", SETUP_ROOT),), dns, timeout
        )
    if result.workspace_stdout is None:
        return result
    return _setup_comparison(task.id, expectations, result.workspace_stdout)


@dataclass(frozen=True)
class RebuiltMetadata:
    image: str
    workdir: str
    setup_script: str


def _rebuilt_metadata(rebuilt_payload: bytes) -> RebuiltMetadata:
    """The image, workdir and setup script a rebuilt task declares."""
    with open_task_tar(rebuilt_payload) as archive:
        member = next((item for item in archive.getmembers() if item.name == "task.toml"), None)
        if member is None:
            raise ValueError("rebuilt task has no task.toml")
        document = tomllib.loads(archive.extractfile(member).read().decode(errors="replace"))
    environment = document.get("environment") or {}
    setup = environment.get("setup") or {}
    command = str(setup.get("command") or "")
    script = setup_script_name(command)
    workdir = environment.get("workdir") or DEFAULT_WORKDIR
    return RebuiltMetadata(environment["docker_image"], workdir, script)


def _native_destinations(native_payload: bytes, workdir: str) -> tuple[str, ...]:
    """Absolute destinations the native recipe writes, resolved against its own workdir."""
    dockerfile = _native_dockerfile(native_payload)
    if dockerfile is None:
        return (workdir,)
    return copied_destinations(dockerfile, native_workdir(dockerfile))


def _execute_setups(
    native: dict[str, bytes],
    rebuilt: dict[str, bytes],
    workers: int,
    *,
    stub_apt: bool = False,
    dns: tuple[str, ...] = (),
) -> list[SetupExecution]:
    def run(task_id: str) -> SetupExecution:
        try:
            metadata = _rebuilt_metadata(rebuilt[task_id])
        except ValueError as error:
            # A missing rebuilt[task_id] is a cohort-join bug and propagates;
            # only an unusable rebuilt tar is a task-level refusal.
            return SetupExecution(task_id=task_id, outcome=SetupOutcome.INVALID_TASK, detail=str(error))
        return execute_setup_in_docker(
            task_id,
            native[task_id],
            rebuilt[task_id],
            image=metadata.image,
            workdir=metadata.workdir,
            setup_script=metadata.setup_script,
            destinations=_native_destinations(native[task_id], metadata.workdir),
            stub_apt=stub_apt,
            dns=dns,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, sorted(native)))


def _execute_lowered(
    lowered: dict[str, LoweredSpec], native: dict[str, bytes], workers: int, *, dns: tuple[str, ...] = ()
) -> list[SetupExecution]:
    def run(task_id: str) -> SetupExecution:
        expectations = workspace_expectations(native[task_id])
        requirements = lowered[task_id].task.environment_requirements
        workdir = requirements.working_directory or DEFAULT_WORKDIR
        return execute_lowered_setup(
            lowered[task_id].task,
            expectations,
            destinations=_native_destinations(native[task_id], workdir),
            dns=dns,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, sorted(lowered)))
