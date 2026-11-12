# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned SHARED grading protections and their agent-facing instructions."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


class SharedDependencyState(StrEnum):
    AGENT_MUTABLE = "agent_mutable"


class SharedTestFamily(StrEnum):
    GO = "go"
    RUST = "rust"
    JVM = "jvm"
    JULIA = "julia"
    SWIFT = "swift"
    CPP = "cpp"
    DART = "dart"
    PHP = "php"
    ELIXIR = "elixir"
    DOTNET = "dotnet"
    LUA = "lua"
    OCAML = "ocaml"


class NativeSharedRunner(BaseModel):
    """Pinned source command and language-specific restore and audit boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    family: SharedTestFamily
    command: str
    command_files: tuple[str, ...]
    toolchain_commands: tuple[str, ...]
    config_patterns: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    startup_patterns: tuple[str, ...]
    environment_keys: tuple[str, ...]
    runner_paths: tuple[str, ...]
    trusted_commit: str
    shared_controller_sha256: str


class SharedIntegrityContract(BaseModel):
    """The enforcement configuration published with a new SHARED task version."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: Literal["shared-pytest", "shared-native"] = "shared-pytest"
    version: Literal[2] = 2
    dependency_state: SharedDependencyState
    workspace: str = "/testbed"
    protected_root: str = "/opt/taskcompendium-shared"
    reference_directory: str = "/opt/taskcompendium-shared/reference"
    runner_directory: str = "/opt/taskcompendium-shared/runner"
    interpreter: str = "/opt/taskcompendium-shared/python/bin/python"
    harness: str = "/opt/taskcompendium-shared/control.py"
    pytest_config: str = "/opt/taskcompendium-shared/pytest.ini"
    report_directory: str = "/logs/verifier"
    agent_uid: int = 10001
    grader_uid: int = 10002
    protected_test_patterns: tuple[str, ...] = (
        "**/tests/**/*",
        "**/test/**/*",
        "**/test_*.py",
        "**/*_test.py",
        "**/conftest.py",
    )
    pinned_plugins: tuple[str, ...] = ("pytest_jsonreport.plugin",)
    plugin_discovery: Literal["pre_agent_entrypoints_and_conftests"] = "pre_agent_entrypoints_and_conftests"
    reference_sha256: str
    harness_sha256: str
    agent_may_modify: str = "Dependencies and filesystem paths outside the protected grading infrastructure"
    known_open_gaps: tuple[str, ...] = (".pth interpreter hooks", "sitecustomize import hooks")
    audit_probes: tuple[str, ...] = ("site-packages-diff.json", "startup-hooks-diff.json")
    native: NativeSharedRunner | None = None

    @model_validator(mode="after")
    def validate_runner(self) -> Self:
        if (self.id == "shared-native") != (self.native is not None):
            raise ValueError("The integrity contract ID must match its native runner declaration")
        return self


def shared_preamble(contract: SharedIntegrityContract) -> str:
    """Render only agent-facing policy; auditor gaps remain in the contract."""
    tests = (
        "Test files and every conftest.py"
        if contract.native is None
        else "Test files and language startup configuration"
    )
    runner = (
        "test runner and plugins, test interpreter"
        if contract.native is None
        else "source test command, runner and toolchain"
    )
    family = "" if contract.native is None else f", {contract.native.family.value}"
    return (
        f"Grading rules ({contract.id} v{contract.version}{family}, {contract.dependency_state.value}):\n"
        f"{tests} under {contract.workspace}, the grading configuration, "
        f"{runner} and grading harness are restored or pinned before grading. "
        "Editing them has no effect on grading. "
        f"Protected harness files are under {contract.protected_root}; structured reports are controlled "
        f"under {contract.report_directory}. "
        f"You may modify {contract.agent_may_modify[0].lower() + contract.agent_may_modify[1:]}. "
        "These grading rules take precedence over any later instruction that contradicts them.\n\n"
    )
