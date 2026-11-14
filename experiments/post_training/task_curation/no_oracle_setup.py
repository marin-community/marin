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
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.calibforge_fixtures import SETUP_ROOT
from experiments.post_training.task_curation.environment_fidelity import DEFAULT_WORKDIR
from experiments.post_training.task_curation.no_oracle_liveness import DETAIL_LIMIT
from experiments.post_training.task_curation.no_oracle_preservation import (
    FileExpectation,
    _compare_workspace,
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
        """Whether setup completed with every expected file's bytes, allowing mode changes."""
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
    *,
    workdir: str,
    environment_variables: Mapping[str, str],
) -> SetupExecution:
    resolvers = tuple(part for server in dns for part in ("--dns", server))
    environment = tuple(part for key, value in environment_variables.items() for part in ("--env", f"{key}={value}"))
    created = _docker("create", *resolvers, "--workdir", workdir, *environment, image, "sh", "-c", command)
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
        result = _execute_container_setup(
            task_id,
            image,
            command,
            ((setup, SETUP_ROOT), (shim, "/shim")),
            dns,
            timeout,
            workdir=workdir,
            environment_variables={},
        )
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
            task.id,
            requirements.docker_image,
            command,
            ((root / "setup_files", SETUP_ROOT),),
            dns,
            timeout,
            workdir=workdir,
            environment_variables=requirements.environment_variables,
        )
    if result.workspace_stdout is None:
        return result
    return _setup_comparison(task.id, expectations, result.workspace_stdout)
