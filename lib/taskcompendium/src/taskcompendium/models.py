# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned semantic contracts, independent of Harbor's on-disk layout."""

import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

import msgspec
from tasktrove_verify.spec import Mode

SCHEMA_VERSION = "0.6"
VERIFIER_REVISION = "b76d03131cd88bd9fc711dba206659027edba3a8"
LEGACY_VERIFIER_REVISION = "b2b68d8b0a770cdc0ab3903780172c4b3eea81b1"
SUPPORTED_VERIFIER_REVISIONS = frozenset({LEGACY_VERIFIER_REVISION, VERIFIER_REVISION})
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


class Capability(StrEnum):
    FILESYSTEM = "filesystem"
    SHELL = "shell"
    PROCESS = "process"


class WorkspaceState(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Required initial state, independent of the service that materializes it."""

    image: str | None = None
    workdir: str = "/app"
    setup_commands: tuple[str, ...] = ()
    additional_directories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.image is not None:
            image_digest(self.image)
        validate_directories(self.workdir, self.additional_directories)


class ActionInterface(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A semantic, provider-independent stateful action surface and pinned seed digest."""

    name: str
    version: str
    seed_sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.version or not _SHA256.fullmatch(self.seed_sha256):
            raise ValueError("Action interfaces require a name, version, and immutable seed digest")


class TaskRequirements(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    capabilities: tuple[Capability, ...] = ()
    state: WorkspaceState = WorkspaceState()
    action_interfaces: tuple[ActionInterface, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("Required capabilities must be unique")
        if self.state != WorkspaceState() and Capability.FILESYSTEM not in self.capabilities:
            raise ValueError("Workspace state requires filesystem capability")
        names = tuple(interface.name for interface in self.action_interfaces)
        if len(set(names)) != len(names):
            raise ValueError("Action interface names must be unique")


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
    revision: str = LEGACY_VERIFIER_REVISION
    workspace: EmptyWorkspace | ImageOverlay = EmptyWorkspace()
    supervisor_python: str = "python3"

    def __post_init__(self) -> None:
        image_digest(self.image)
        if self.revision not in SUPPORTED_VERIFIER_REVISIONS:
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


EXECUTABLE_MODES = frozenset({Mode.STDIO, Mode.PYTEST, Mode.JUNIT, Mode.GOTEST, Mode.SCRIPT})


class TaskTroveVerifier(
    msgspec.Struct, frozen=True, tag_field="kind", tag="tasktrove", forbid_unknown_fields=True, omit_defaults=True
):
    """Parameters use the pinned tasktrove-verify ontology, without submission paths."""

    mode: Mode
    parameters: dict[str, Any]
    judge: JudgeConfig | None = None
    runtime: ContainerRuntime | None = None
    implementation_revision: str = LEGACY_VERIFIER_REVISION

    def __post_init__(self) -> None:
        if self.implementation_revision not in SUPPORTED_VERIFIER_REVISIONS:
            raise ValueError("Unsupported verifier implementation revision")
        if (self.mode in EXECUTABLE_MODES) != (self.runtime is not None):
            raise ValueError("Only executable verifiers require an isolated container runtime")
        if self.runtime is not None and self.runtime.revision != self.implementation_revision:
            raise ValueError("Verifier runtime must match the verifier implementation revision")
        if {"mode", "output", "workspace"} & self.parameters.keys():
            raise ValueError("Mode, submission location and execution workspace are separate contracts")
        if (self.mode == Mode.JUDGE) != (self.judge is not None):
            raise ValueError("Judge mode requires explicit judge configuration")


class FunctionCall(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A source-native function call, retained without dispatching it."""

    name: str
    arguments: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Function calls require a name")


class ExpectedMessage(msgspec.Struct, frozen=True, tag_field="type", tag="message", forbid_unknown_fields=True):
    content: str


class ExpectedFunctionCall(
    msgspec.Struct, frozen=True, tag_field="type", tag="function_call", forbid_unknown_fields=True
):
    name: str
    arguments: str


class ExpectedFunctionCallBatch(
    msgspec.Struct, frozen=True, tag_field="type", tag="function_call_batch", forbid_unknown_fields=True
):
    calls: tuple[FunctionCall, ...]

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError("A function-call batch requires at least one call")


ExpectedAction = ExpectedMessage | ExpectedFunctionCall | ExpectedFunctionCallBatch


class ParallelToolCallRewardMode(StrEnum):
    BINARY_STRICT = "binary_strict"
    FRACTIONAL = "fractional"
    F1 = "f1"


class ToolCallComparatorConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Pinned source comparator settings; defaults preserve existing NeMo rows."""

    word_count_similarity_threshold: float
    floating_point_comparison_threshold: float = 1e-6
    parallel_tool_call_rewarding: bool = False
    allow_subset: bool = False
    allow_superset: bool = False
    parallel_tool_call_reward_mode: ParallelToolCallRewardMode = ParallelToolCallRewardMode.BINARY_STRICT

    def __post_init__(self) -> None:
        if self.word_count_similarity_threshold < 0 or self.floating_point_comparison_threshold < 0:
            raise ValueError("Comparator thresholds must be nonnegative")


class PredictedActionVerifier(
    msgspec.Struct, frozen=True, tag_field="kind", tag="predicted_action", forbid_unknown_fields=True
):
    """Private NeMo next-action target and source comparison behavior."""

    expected_action: ExpectedAction
    comparator: ToolCallComparatorConfig
    source_revision: str

    def __post_init__(self) -> None:
        if not self.source_revision:
            raise ValueError("Predicted-action verifier requires a pinned source revision")


class InstructionConstraint(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Instruction constraints require a source name")


class ConstraintVerifier(
    msgspec.Struct, frozen=True, tag_field="kind", tag="instruction_constraints", forbid_unknown_fields=True
):
    """Private IFEval checks evaluated through the pinned source registry."""

    constraints: tuple[InstructionConstraint, ...]
    aggregation: Literal["binary", "fraction"]

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("Constraint verifier requires at least one source constraint")


class ProviderStateVerifier(
    msgspec.Struct, frozen=True, tag_field="kind", tag="provider_state", forbid_unknown_fields=True
):
    """Private source-adapter verifier over authoritative provider state, not agent claims."""

    interface: ActionInterface
    adapter: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.adapter:
            raise ValueError("Provider-state verification requires a source adapter")


class CodeAnswerVerifier(msgspec.Struct, frozen=True, tag_field="kind", tag="code_answer", forbid_unknown_fields=True):
    """Materialize an answer-only code payload before an isolated source verifier runs."""

    verifier: TaskTroveVerifier
    output_path: str

    def __post_init__(self) -> None:
        relative_path(self.output_path)
        if self.verifier.mode not in EXECUTABLE_MODES:
            raise ValueError("Code-answer verifier requires an executable source verifier")


Verifier = TaskTroveVerifier | PredictedActionVerifier | ConstraintVerifier | ProviderStateVerifier | CodeAnswerVerifier


def verifier_runtime(verifier: Verifier) -> ContainerRuntime | None:
    """Return an isolated runtime only for verifier contracts that declare one."""
    if isinstance(verifier, TaskTroveVerifier):
        return verifier.runtime
    if isinstance(verifier, CodeAnswerVerifier):
        return verifier.verifier.runtime
    return None


def tasktrove_verifier(verifier: Verifier) -> TaskTroveVerifier | None:
    """Unwrap a source verifier only where its pinned ontology remains applicable."""
    if isinstance(verifier, TaskTroveVerifier):
        return verifier
    if isinstance(verifier, CodeAnswerVerifier):
        return verifier.verifier
    return None


class AnswerRequirements(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The semantic form of a task's submitted answer.

    ``text`` accepts an unconstrained assistant response. Renderings can add a
    transport wrapper, but they do not change this intrinsic answer form.
    """

    kind: Literal["text", "literal", "json", "xml", "csv", "final_state"] = "text"


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


class ContextRequirement(StrEnum):
    INSTRUCTION_AND_WORKSPACE = "instruction_and_workspace"
    PRIOR_CONVERSATION = "prior_conversation"


class TaskSuccessPolicy(StrEnum):
    ALL_REQUIRED_STEPS = "all_required_steps"
    FINAL = "final"
    MEAN = "mean"


class StepSpecification(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    instructions: str
    verifier: Verifier
    answer_requirements: AnswerRequirements = AnswerRequirements()
    resources: tuple[Resource, ...] = ()
    context_requirement: ContextRequirement = ContextRequirement.INSTRUCTION_AND_WORKSPACE

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("Step instructions are required")


class TaskSpecification(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    """One fixed semantic instance, including private correctness criteria."""

    id: str
    steps: tuple[StepSpecification, ...]
    requirements: TaskRequirements
    resources: tuple[Resource, ...]
    metadata: TaskMetadata
    success_policy: TaskSuccessPolicy = TaskSuccessPolicy.ALL_REQUIRED_STEPS
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported task schema: {self.schema_version}")
        if not self.id or not self.steps:
            raise ValueError("Task id and at least one step are required")
        for step in self.steps:
            occupied: set[tuple[ResourceRole, str]] = set()
            for resource in (*self.resources, *step.resources):
                for role in resource.roles:
                    key = (role, resource.path)
                    if key in occupied:
                        raise ValueError(f"Resource placement is ambiguous: {key}")
                    occupied.add(key)


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


class NativeFunction(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A provider-native advertised function definition, without an execution binding."""

    name: str
    parameters: dict[str, Any]
    description: str | None = None
    strict: bool | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.parameters, dict):
            raise ValueError("Native functions require a name and parameter schema")


class FinalActionSubmission(
    msgspec.Struct, frozen=True, tag_field="kind", tag="final_action", forbid_unknown_fields=True
):
    """A final model action represented natively, with no domain-action dispatch."""

    functions: tuple[NativeFunction, ...]
    allow_message: bool = True

    def __post_init__(self) -> None:
        if not self.functions:
            raise ValueError("Final-action submissions require advertised functions")
        names = tuple(function.name for function in self.functions)
        if len(set(names)) != len(names):
            raise ValueError("Final-action function names must be unique")


class Rendering(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    id: str
    submission: AssistantFinal | FileSubmission | FinalState | FinalActionSubmission
    version: str = "0.2"
    instruction_surface: Literal["original"] = "original"


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


class PublicResource(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    path: str
    content: Embedded | ResourceRef
    executable: bool = False

    def __post_init__(self) -> None:
        relative_path(self.path)


class TaskStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    instructions: str
    resources: tuple[PublicResource, ...]
    submission: AssistantFinal | FileSubmission | FinalState | FinalActionSubmission
    context_requirement: ContextRequirement


class Task(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Public rendered instance; private evaluation stays with its specification."""

    id: str
    specification_sha256: str
    steps: tuple[TaskStep, ...]
    requirements: TaskRequirements
    resources: tuple[PublicResource, ...]
    metadata: TaskMetadata
