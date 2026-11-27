# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned SHARED grading protections declared by producer TaskSpecs."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SharedDependencyState(StrEnum):
    AGENT_MUTABLE = "agent_mutable"


class DeliveredVerifierFile(BaseModel):
    """One private verifier file with the mode its delivery must preserve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    mode: int
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MultiSweDeliveryContract(BaseModel):
    """The typed SHARED integrity contract for Multi-SWE delivered-pipeline sources.

    Same original-root delivery semantics as the root-delivery contract, with
    the captured ``/home`` artifacts restored verbatim (the archive itself
    carries modes and mtimes) and the reward parsed host-side by the pinned
    upstream parser from the copied artifacts. ``archive_sha256`` pins the
    delivered capture archive bytes; ``undeclared_deadline`` records sources
    with no verifier deadline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    id: Literal["multi-swe-delivery"] = "multi-swe-delivery"
    dependency_state: SharedDependencyState
    agent_user: str | None = None
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    undeclared_deadline: bool = False
    known_open_gaps: tuple[str, ...] = (
        "Agent-installed build tools and imports in the shared image environment",
        "Candidate-produced test-output injection",
    )


class RootDeliveryContract(BaseModel):
    """The typed SHARED integrity contract for original-root delivery sources.

    Sources graded by delivery (RST family) keep the original root actor: no UID
    demotion and no in-container protected root. The agent phase is untouched;
    at grading time, trusted root collect commands quiesce the container, deliver
    the private verifier files (never present during the actor phase), and run
    the source's original verifier script verbatim, which writes its own reward.
    Copied artifacts carry the reward to the grading machine; the container is
    never trusted to score beyond the source's own script.

    ``verifier_files`` records the delivered per-package verifier material with
    its original modes; ``undeclared_deadline`` records that the source declares
    no verifier deadline, so the grading command waits unbounded under the same
    outer enforcement the source runs under.
    """

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
        "Candidate-produced test-output injection",
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
    native: NativeSharedRunner | SourceSharedRunner | None = None

    @model_validator(mode="after")
    def validate_runner(self) -> Self:
        native = isinstance(self.native, NativeSharedRunner)
        source = isinstance(self.native, SourceSharedRunner)
        if (self.id == "shared-native") != native or (self.id == "shared-source") != source:
            raise ValueError("The integrity contract ID must match its native runner declaration")
        return self
