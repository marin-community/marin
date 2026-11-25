# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned SHARED grading protections and their agent-facing instructions."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TEST_OUTPUT_INJECTION_GAP = "Candidate-produced test-output injection"

PROTECTED_ROOT = "/opt/taskcompendium-shared"
BOOTSTRAP_DIRECTORY = f"{PROTECTED_ROOT}-bootstrap"
BOOTSTRAP_CONTROLLER = f"{BOOTSTRAP_DIRECTORY}/control.py"
PRIVATE_DIRECTORY = f"{PROTECTED_ROOT}-private"
PRIVATE_CONTEXT = "environment/taskcompendium-shared-private"
CONTROLLER_CONTEXT = "environment/taskcompendium-shared/control.py"
SHARED_CONTROLLER_CONTEXT = "environment/taskcompendium-shared/shared_pytest.py"
AGENT_HOME = "/home/taskcompendium-agent"
BOOTSTRAP_TIMEOUT = 1800.0
STARTUP_HOOKS_REPORT = "startup-hooks-diff.json"
PYTEST_AUDIT_PROBES = ("site-packages-diff.json", STARTUP_HOOKS_REPORT)
# Pytest audits Python sites; native and source runners audit their declared dependency roots.
DEPENDENCY_AUDIT_PROBES = ("dependencies-diff.json", STARTUP_HOOKS_REPORT)


class SharedDependencyState(StrEnum):
    AGENT_MUTABLE = "agent_mutable"


class DeliveredVerifierFile(BaseModel):
    """One private verifier file with the mode its delivery must preserve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    mode: int
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MultiSweDeliveryContract(BaseModel):
    """The recorded shared-root capture contract for a Multi-SWE pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    id: Literal["multi-swe-delivery"] = "multi-swe-delivery"
    dependency_state: SharedDependencyState
    agent_user: str | None = None
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    undeclared_deadline: bool = False
    known_open_gaps: tuple[str, ...] = (
        "Agent-installed build tools and imports in the shared image environment",
        TEST_OUTPUT_INJECTION_GAP,
    )


class RootDeliveryContract(BaseModel):
    """The recorded verifier files, modes and deadlines of a root-delivery task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    id: Literal["root-delivery"] = "root-delivery"
    dependency_state: SharedDependencyState
    agent_user: str | None = None
    """The image's own agent identity; None keeps the source's default user."""
    verifier_files: tuple[DeliveredVerifierFile, ...] = Field(min_length=1)
    undeclared_deadline: bool = False
    known_open_gaps: tuple[str, ...] = (
        "Agent-installed pytest plugins and imports in the shared image environment",
        TEST_OUTPUT_INJECTION_GAP,
    )


class SharedSourceFamily(StrEnum):
    """Protected SHARED source-scorer families built on original Prime pytest graders."""

    SCALE_SWE = "scale-swe"
    SWE_LEGO = "swe-lego"
    SWE_REBENCH = "swe-rebench"


class SourceSharedRunner(BaseModel):
    """Pinned source scorer material and capture boundaries for one family.

    ``capture_dirs`` and ``capture_patterns`` select the workspace paths the
    bootstrap freezes as the trusted grading reference; ``patch_config_keys``
    name the private config entries holding unified diffs whose affected paths
    must be captured; ``path_config_keys`` name the private config entries
    holding explicit workspace-relative path lists (e.g. a source's trusted
    test-file list) that join the capture scope when they exist.
    ``dependency_paths`` are the source interpreter environments whose live
    inventory grading audits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: SharedSourceFamily
    capture_dirs: tuple[str, ...] = ()
    capture_patterns: tuple[str, ...] = ()
    patch_config_keys: tuple[str, ...] = ()
    path_config_keys: tuple[str, ...] = ()
    dependency_paths: tuple[str, ...] = ()
    shared_controller_sha256: str = ""


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
    id: Literal["shared-pytest", "shared-native", "shared-source"] = "shared-pytest"
    version: Literal[2] = 2
    dependency_state: SharedDependencyState
    workspace: str = "/testbed"
    protected_root: str = PROTECTED_ROOT
    reference_directory: str = f"{PROTECTED_ROOT}/reference"
    runner_directory: str = f"{PROTECTED_ROOT}/runner"
    interpreter: str = f"{PROTECTED_ROOT}/python/bin/python"
    harness: str = f"{PROTECTED_ROOT}/control.py"
    pytest_config: str = f"{PROTECTED_ROOT}/pytest.ini"
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
    audit_probes: tuple[str, ...] = PYTEST_AUDIT_PROBES
    native: NativeSharedRunner | SourceSharedRunner | None = None

    @model_validator(mode="after")
    def validate_runner(self) -> Self:
        native = isinstance(self.native, NativeSharedRunner)
        source = isinstance(self.native, SourceSharedRunner)
        if (self.id == "shared-native") != native or (self.id == "shared-source") != source:
            raise ValueError("The integrity contract ID must match its native runner declaration")
        return self


def shared_preamble(contract: SharedIntegrityContract) -> str:
    """Render only agent-facing policy; auditor gaps remain in the contract."""
    if isinstance(contract.native, NativeSharedRunner):
        tests = "Test files and language startup configuration"
        runner = "source test command, runner and toolchain"
        family = f", {contract.native.family.value}"
        product_plugins = ""
    elif isinstance(contract.native, SourceSharedRunner):
        tests = "Test files"
        runner = "source scorer, its private grading configuration and material"
        family = f", {contract.native.family.value}"
        product_plugins = ""
    else:
        tests = "Test files and every conftest.py"
        runner = "third-party test runner and plugins, test interpreter"
        family = ""
        product_plugins = "Repository-owned pytest plugins remain mutable product code. "
    return (
        f"Grading rules ({contract.id} v{contract.version}{family}, {contract.dependency_state.value}):\n"
        f"{tests} under {contract.workspace}, the grading configuration, "
        f"{runner} and grading harness are restored or pinned before grading. "
        "Editing them has no effect on grading. "
        f"Protected harness files are under {contract.protected_root}; structured reports are controlled "
        f"under {contract.report_directory}. "
        f"You may modify {contract.agent_may_modify[0].lower() + contract.agent_may_modify[1:]}. "
        f"Your HOME is {AGENT_HOME} unless the source declares another non-root HOME. "
        f"{product_plugins}"
        "These grading rules take precedence over any later instruction that contradicts them.\n\n"
    )
