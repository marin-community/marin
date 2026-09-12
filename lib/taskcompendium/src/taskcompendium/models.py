# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned semantic contracts, independent of Harbor's on-disk layout."""

import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

import msgspec
from tasktrove_verify.spec import Mode

SCHEMA_VERSION = "0.1"
VERIFIER_REVISION = "b2b68d8b0a770cdc0ab3903780172c4b3eea81b1"
HARBOR_REVISION = "9551f376157d90104011107dcbf9ca3621228126"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def relative_path(path: str) -> None:
    """Reject paths that can escape a resource's declared root."""
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"Expected a normalized relative resource path: {path!r}")


def image_digest(image: str) -> None:
    digest = image.removeprefix("sha256:") if image.startswith("sha256:") else image.rsplit("@sha256:", 1)[-1]
    if not _SHA256.fullmatch(digest) or (not image.startswith("sha256:") and "@sha256:" not in image):
        raise ValueError("Container images require an immutable @sha256 digest")


def validate_workdir(path: str) -> None:
    """Keep candidate files in a normalized directory outside verifier/system roots."""
    parsed = PurePosixPath(path)
    protected = (
        "/tests",
        "/input",
        "/result",
        "/snapshot",
        "/opt/runtime",
        "/proc",
        "/sys",
        "/dev",
        "/tmp",
        "/etc",
        "/usr",
        "/bin",
        "/lib",
    )
    if (
        not parsed.is_absolute()
        or path == "/"
        or str(parsed) != path
        or ".." in parsed.parts
        or any(path == root or path.startswith(root + "/") for root in protected)
    ):
        raise ValueError(f"Unsafe candidate workdir: {path!r}")


def validate_directories(workdir: str, directories: tuple[str, ...]) -> None:
    roots = (workdir, *directories)
    for index, root in enumerate(roots):
        validate_workdir(root)
        for other in roots[:index]:
            if root == other or root.startswith(other + "/") or other.startswith(root + "/"):
                raise ValueError("Candidate filesystem roots must not overlap")


class ResourceRole(StrEnum):
    AGENT = "agent"
    VERIFIER = "verifier"
    ORACLE = "oracle"


class Embedded(msgspec.Struct, frozen=True, tag_field="kind", tag="embedded", forbid_unknown_fields=True):
    data: bytes


class ResourceRef(msgspec.Struct, frozen=True, tag_field="kind", tag="reference", forbid_unknown_fields=True):
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri or not _SHA256.fullmatch(self.sha256):
            raise ValueError("A resource reference requires a URI and SHA256 digest")


class Resource(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    path: str
    roles: tuple[ResourceRole, ...]
    content: Embedded | ResourceRef
    executable: bool = False

    def __post_init__(self) -> None:
        relative_path(self.path)
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("Resource roles must be explicit and unique")
        if ResourceRole.ORACLE in self.roles and ResourceRole.AGENT in self.roles:
            raise ValueError("Oracle material cannot be agent-visible")


class NoEnvironment(msgspec.Struct, frozen=True, tag_field="kind", tag="none", forbid_unknown_fields=True):
    pass


class ShellSimEnvironment(msgspec.Struct, frozen=True, tag_field="kind", tag="shellsim", forbid_unknown_fields=True):
    workdir: str = "/app"
    max_steps: int = 100_000
    max_output_bytes: int = 1_048_576
    setup_commands: tuple[str, ...] = ()
    additional_directories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_directories(self.workdir, self.additional_directories)
        if self.max_steps <= 0 or self.max_output_bytes <= 0:
            raise ValueError("ShellSim budgets must be positive")


class DockerEnvironment(msgspec.Struct, frozen=True, tag_field="kind", tag="docker", forbid_unknown_fields=True):
    image: str
    workdir: str = "/app"
    setup_commands: tuple[str, ...] = ()
    additional_directories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        image_digest(self.image)
        validate_directories(self.workdir, self.additional_directories)


EnvironmentRequirement = NoEnvironment | ShellSimEnvironment | DockerEnvironment


class PythonRuntime(msgspec.Struct, frozen=True, tag_field="kind", tag="python", forbid_unknown_fields=True):
    """Trusted answer graders only; never executes source programs."""

    revision: str = VERIFIER_REVISION

    def __post_init__(self) -> None:
        if self.revision != VERIFIER_REVISION:
            raise ValueError("Unsupported verifier implementation revision")


class EmptyWorkspace(msgspec.Struct, frozen=True, tag_field="kind", tag="empty", forbid_unknown_fields=True):
    pass


class ImageOverlay(msgspec.Struct, frozen=True, tag_field="kind", tag="image_overlay", forbid_unknown_fields=True):
    """Retain image dependencies while replacing all submitted workspace content."""

    preserved_directories: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.preserved_directories or len(set(self.preserved_directories)) != len(self.preserved_directories):
            raise ValueError("Image overlay requires unique preserved dependency directories")
        for path in self.preserved_directories:
            relative_path(path)
            if "/" in path or path == "__external__":
                raise ValueError("Preserved dependency directories must be top-level workspace directories")


class ContainerRuntime(msgspec.Struct, frozen=True, tag_field="kind", tag="container", forbid_unknown_fields=True):
    image: str
    timeout: float = 120.0
    revision: str = VERIFIER_REVISION
    workspace: EmptyWorkspace | ImageOverlay = EmptyWorkspace()
    supervisor_python: str = "python3"

    def __post_init__(self) -> None:
        image_digest(self.image)
        if self.revision != VERIFIER_REVISION:
            raise ValueError("Unsupported verifier implementation revision")
        if not self.supervisor_python or any(c.isspace() for c in self.supervisor_python):
            raise ValueError("Supervisor Python must be a single executable path")
        if self.timeout <= 0:
            raise ValueError("Verifier timeout must be positive")


class JudgeModelPolicy(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    model: str
    size_class: Literal["small", "medium", "large"]
    provider: str
    base_url: str
    samples: int = 1
    aggregation: Literal["mean"] = "mean"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model or not self.provider or not self.base_url or self.samples < 1:
            raise ValueError("Judge provider, endpoint, model and positive sample count are required")


class JudgeView(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    transcript: bool = False
    files: tuple[str, ...] = ()
    reference_context: tuple[str, ...] = ()


class JudgeConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    policy: JudgeModelPolicy
    view: JudgeView


class VerifierSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Parameters use the pinned tasktrove-verify ontology, without submission paths."""

    mode: Mode
    parameters: dict[str, Any]
    judge: JudgeConfig | None = None

    def __post_init__(self) -> None:
        if {"mode", "output", "workspace"} & self.parameters.keys():
            raise ValueError("Mode, submission location and execution workspace are separate contracts")
        if (self.mode == Mode.JUDGE) != (self.judge is not None):
            raise ValueError("Judge mode requires explicit judge configuration")


class AnswerRequirements(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    kind: Literal["value", "literal", "json", "xml", "csv", "final_state"] = "value"


class Source(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    dataset: str
    revision: str
    row: str
    importer_revision: str

    def __post_init__(self) -> None:
        if not all((self.dataset, self.revision, self.row, self.importer_revision)):
            raise ValueError("Source provenance must be complete")


class TaskMetadata(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    source: Source
    competencies: tuple[str, ...] = ()
    task_shape: str = "answer"


class TaskSpecification(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    id: str
    instructions: str  # Agent-facing task request; no evaluation machinery.
    environment: EnvironmentRequirement
    resources: tuple[Resource, ...]
    verifier: VerifierSpec
    verifier_runtime: PythonRuntime | ContainerRuntime
    metadata: TaskMetadata
    answer_requirements: AnswerRequirements = AnswerRequirements()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported task schema: {self.schema_version}")
        if not self.id or not self.instructions.strip():
            raise ValueError("Task id and instructions are required")
        occupied: set[tuple[ResourceRole, str]] = set()
        for resource in self.resources:
            for role in resource.roles:
                key = (role, resource.path)
                if key in occupied:
                    raise ValueError(f"Resource placement is ambiguous: {key}")
                occupied.add(key)


class Chat(msgspec.Struct, frozen=True, tag_field="kind", tag="chat", forbid_unknown_fields=True):
    pass


class ChatWithTools(msgspec.Struct, frozen=True, tag_field="kind", tag="chat_with_tools", forbid_unknown_fields=True):
    tools: tuple[Literal["terminal"], ...] = ("terminal",)


class PlainText(msgspec.Struct, frozen=True, tag_field="kind", tag="plain", forbid_unknown_fields=True):
    pass


class BoxedLatex(msgspec.Struct, frozen=True, tag_field="kind", tag="boxed_latex", forbid_unknown_fields=True):
    pass


class JsonPath(msgspec.Struct, frozen=True, tag_field="kind", tag="json_path", forbid_unknown_fields=True):
    path: str = "$.answer"


class XmlPath(msgspec.Struct, frozen=True, tag_field="kind", tag="xml_path", forbid_unknown_fields=True):
    path: str = "/answer"


Extractor = PlainText | BoxedLatex | JsonPath | XmlPath


class AssistantFinal(msgspec.Struct, frozen=True, tag_field="kind", tag="assistant_final", forbid_unknown_fields=True):
    extractor: Extractor = PlainText()


class FileSubmission(msgspec.Struct, frozen=True, tag_field="kind", tag="file", forbid_unknown_fields=True):
    path: str
    extractor: Extractor = PlainText()


class FinalState(msgspec.Struct, frozen=True, tag_field="kind", tag="final_state", forbid_unknown_fields=True):
    paths: tuple[str, ...]
    excluded_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for path in self.excluded_paths:
            relative_path(path)


class Protocol(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    id: str
    interaction: Chat | ChatWithTools
    submission: AssistantFinal | FileSubmission | FinalState
    version: str = "0.1"


class ExecutionConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    agent: Literal["replay", "chat", "tool_chat", "terminus-2", "mini-swe-agent"]
    environment: EnvironmentRequirement
    timeout: float = 120.0


class Outcome(StrEnum):
    GRADED = "graded"
    EXTRACTION_ERROR = "extraction_error"
    INVALID_TASK = "invalid_task"
    INFRA_ERROR = "infra_error"


class GradingResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    status: Outcome
    reward: float | None
    detail: dict[str, Any] = msgspec.field(default_factory=dict)


class RejectionReason(StrEnum):
    BROKEN_GRADER = "broken_grader"
    GOLD_LEAKAGE = "gold_leakage"
    UNDERSPECIFIED = "underspecified"
    NULL_ANSWER_PASSES = "null_answer_passes"
    UNRECOVERABLE_SOURCE = "unrecoverable_source"
    UNSUPPORTED_ENVIRONMENT = "unsupported_environment"
    UNSUPPORTED_VERIFIER = "unsupported_verifier"
    DUPLICATE = "duplicate"


class Rejected(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    source: Source
    reason: RejectionReason
    detail: str
