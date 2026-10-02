# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export direct-chat or bounded Docker-file tasks as Harbor packages."""

import base64
import hashlib
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator
from rigging.filesystem.path_validation import validate_relative_file_path
from tasktrove_verify.spec import Mode, StructuredExactSpec

from taskcompendium.direct_chat import unsupported_direct_chat_features
from taskcompendium.grading import resolve_verifier, supports_verifier, validate_verifier
from taskcompendium.models import (
    SCHEMA_VERSION,
    DatasetPath,
    EnvironmentRequirements,
    TaskResource,
    TaskSpec,
    VerifierSpec,
)
from taskcompendium.submission import (
    JsonFile,
    SubmissionConvention,
    TextFile,
    render_instruction,
    submission_compatibility,
)

DIRECT_CHAT_ENVIRONMENT = "direct_chat"
DOCKER_ENVIRONMENT = "docker"
MAX_RESOURCE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 64 * 1024 * 1024
SPECIFICATION_FILE = "specification.json"
SUBMISSION_CONVENTION_FILE = "submission_convention.json"
ENVIRONMENT_CONFIG_FILE = "environment_config.json"
DEFAULT_INLINE_FILE_MODE = "0644"


class HarborEnvironmentConfig(BaseModel):
    """The environment and tools this Harbor lowering exposes to the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str = DIRECT_CHAT_ENVIRONMENT
    tools: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_environment(self) -> "HarborEnvironmentConfig":
        if self.environment not in (DIRECT_CHAT_ENVIRONMENT, DOCKER_ENVIRONMENT) or self.tools:
            raise ValueError("This lowering supports direct chat or Harbor Docker without task-owned tools")
        return self


@dataclass(frozen=True)
class LoweringCandidate:
    """A compatible submission convention and Harbor environment configuration."""

    convention: SubmissionConvention
    environment_config: HarborEnvironmentConfig


class SelectionPolicy(StrEnum):
    """How a caller chooses from compatible lowerings."""

    ALL = "all"
    FIRST = "first"
    SAMPLE = "sample"


def direct_chat_verifier_supported(specification: VerifierSpec) -> bool:
    return supports_verifier(specification) and not isinstance(resolve_verifier(specification), StructuredExactSpec)


def compatible_lowerings(
    specification: TaskSpec,
    convention_library: Sequence[SubmissionConvention],
    environment_configs: Sequence[HarborEnvironmentConfig],
) -> tuple[LoweringCandidate, ...]:
    """Enumerate conventions and environments that preserve this task's contract."""
    candidates = []
    for convention in convention_library:
        if not submission_compatibility(specification, convention).compatible:
            continue
        for environment_config in environment_configs:
            try:
                validate_environment_config(specification, environment_config, convention)
            except (NotImplementedError, ValueError):
                continue
            candidates.append(LoweringCandidate(convention, environment_config))
    return tuple(candidates)


def select_lowerings(
    candidates: Sequence[LoweringCandidate],
    policy: SelectionPolicy,
    *,
    required_environment: str | None = None,
    rng_key: int | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Select compatible candidates, honoring an explicit environment request."""
    if required_environment is not None:
        candidates = tuple(
            candidate for candidate in candidates if candidate.environment_config.environment == required_environment
        )
    if not candidates:
        if required_environment is not None:
            raise ValueError(f"No compatible lowerings for environment {required_environment!r}")
        raise ValueError("No compatible lowerings")
    if policy == SelectionPolicy.SAMPLE:
        if rng_key is None:
            raise ValueError("Sample selection requires an RNG key")
        digest = hashlib.sha256(str(rng_key).encode()).digest()
        return (candidates[int.from_bytes(digest, "big") % len(candidates)],)
    if rng_key is not None:
        raise ValueError("An RNG key is only used by sample selection")
    if policy == SelectionPolicy.ALL:
        return tuple(candidates)
    if policy == SelectionPolicy.FIRST:
        return (candidates[0],)
    raise ValueError(f"Unknown selection policy: {policy}")


def _resource_content(resource: TaskResource) -> bytes:
    if isinstance(resource.source, DatasetPath):
        raise NotImplementedError("Harbor Docker supports inline file resources only")
    content = base64.b64decode(resource.source.content_base64, validate=True)
    if len(content) > MAX_RESOURCE_BYTES:
        raise ValueError("Resource exceeds the inline byte limit")
    return content


def validate_environment_config(
    specification: TaskSpec,
    environment_config: HarborEnvironmentConfig,
    convention: SubmissionConvention | None = None,
) -> None:
    """Reject semantic features the selected Harbor runtime cannot preserve."""
    if environment_config.environment == DIRECT_CHAT_ENVIRONMENT:
        unsupported = unsupported_direct_chat_features(specification)
        if unsupported:
            raise NotImplementedError(f"Direct chat cannot satisfy requirements: {', '.join(unsupported)}")
        if not direct_chat_verifier_supported(specification.verifier):
            raise NotImplementedError(f"Direct chat cannot submit to verifier: {specification.verifier.kind!r}")
        if isinstance(convention, (TextFile, JsonFile)):
            raise NotImplementedError("Direct chat cannot submit a workspace file")
        return
    requirements = specification.environment_requirements
    if requirements.docker_image is None or requirements.working_directory is None:
        raise ValueError("Harbor Docker requires a pinned image and declared working_directory")
    validate_relative_file_path(requirements.working_directory[1:])
    if (
        requirements.working_directory == "/"
        or requirements.working_directory == "/logs"
        or requirements.working_directory.startswith("/logs/")
    ):
        raise ValueError("Docker workdir must be separate from Harbor log mounts")
    if len(requirements.working_directory.strip("/").split("/")) != 1:
        raise NotImplementedError("Harbor Docker supports one root-child working_directory only")
    if (
        set(requirements.capabilities) - {"filesystem", "shell"}
        or requirements.setup_commands
        or requirements.environment_variables
        or requirements.tool_providers
        or specification.final_tools
    ):
        raise NotImplementedError("Harbor Docker cannot preserve these capabilities, setup, environment, or tools")
    if specification.verifier.environment_requirements != EnvironmentRequirements():
        raise NotImplementedError("File grading supports pure host-side verifiers only")
    if not isinstance(convention, (TextFile, JsonFile)):
        raise NotImplementedError("Harbor Docker requires one TextFile or JsonFile submission")
    if "/" in convention.path:
        raise NotImplementedError("Harbor Docker supports one flat final filename only")
    supported = {Mode.EXACT, Mode.NUMERIC} if isinstance(convention, TextFile) else {Mode.STRUCTURED_EXACT}
    if specification.verifier.kind not in supported:
        raise NotImplementedError("File convention does not support this private verifier")
    resources = specification.resources
    total = sum(
        len(_resource_content(resource))
        for group in (resources.all, resources.worker, resources.oracle, resources.verifier)
        for resource in group
    )
    if total > MAX_TOTAL_RESOURCE_BYTES:
        raise ValueError("Resources exceed the total inline byte limit")
    for resource in resources.all + resources.worker:
        if resource.mtime_ns is not None:
            raise NotImplementedError("Harbor Docker cannot preserve declared public-file nanosecond timestamps")
        if not int(resource.mode or DEFAULT_INLINE_FILE_MODE, 8) & 0o400:
            raise NotImplementedError("Harbor Docker public inline files require owner-read permission for upload")
        if resource.path.split("/")[0] in {"Dockerfile", "docker-compose.yaml"}:
            raise ValueError("Worker resources cannot override the Docker definition")


def _task_config(specification: TaskSpec, environment_config: HarborEnvironmentConfig) -> dict:
    environment: dict[str, str | bool | None] = {"allow_internet": False}
    if environment_config.environment == DOCKER_ENVIRONMENT:
        requirements = specification.environment_requirements
        environment.update(docker_image=requirements.docker_image, workdir=requirements.working_directory)
    return {"version": "1.0", "environment": environment, "verifier": {"environment_mode": "shared"}}


def validate_exported_task(specification: TaskSpec, environment_config: HarborEnvironmentConfig, task_dir: Path) -> None:
    """Recheck Docker inputs and exclude private data from worker uploads."""
    if environment_config.environment != DOCKER_ENVIRONMENT:
        return
    expected_root = {
        SPECIFICATION_FILE,
        SUBMISSION_CONVENTION_FILE,
        ENVIRONMENT_CONFIG_FILE,
        "environment",
        "instruction.md",
        "task.toml",
    }
    if {path.name for path in task_dir.iterdir()} != expected_root or any(
        path.is_symlink() for path in task_dir.iterdir()
    ):
        raise ValueError("Exported Docker task contains undeclared files or symlinks")
    if tomllib.loads((task_dir / "task.toml").read_text()) != _task_config(specification, environment_config):
        raise ValueError("Exported Docker configuration differs from the task requirements")
    environment = task_dir / "environment"
    expected = {resource.path: resource for resource in specification.resources.all + specification.resources.worker}
    actual = set()
    for path in environment.rglob("*"):
        if path.is_symlink():
            raise ValueError("Exported worker inputs contain a symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(environment).as_posix()
        actual.add(relative)
        if relative not in expected or not path.is_file() or path.read_bytes() != _resource_content(expected[relative]):
            raise ValueError("Exported worker inputs differ from declared resources")
        mode = expected[relative].mode
        if path.stat().st_mode & 0o7777 != int(mode or DEFAULT_INLINE_FILE_MODE, 8):
            raise ValueError("Exported worker input mode differs from declared resource")
    if actual != set(expected):
        raise ValueError("Exported worker inputs differ from declared resources")


def read_specification(path: Path) -> TaskSpec:
    data = json.loads(path.read_text())
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported TaskSpec schema: {data['schema_version']}")
    return TaskSpec.model_validate(data)


def read_environment_config(path: Path) -> HarborEnvironmentConfig:
    return HarborEnvironmentConfig.model_validate_json(path.read_text())


def read_submission_convention(path: Path) -> SubmissionConvention:
    return TypeAdapter(SubmissionConvention).validate_json(path.read_text())


def lower_to_harbor(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    destination: Path,
) -> Path:
    """Write one custom-verifier task; launch agent selection remains separate."""
    validate_environment_config(specification, environment_config, convention)
    validate_verifier(specification.verifier)
    instruction = render_instruction(specification, convention)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "environment").mkdir()
    (destination / "instruction.md").write_text(instruction)
    config = _task_config(specification, environment_config)
    task_toml = 'version = "1.0"\n\n[environment]\n'
    for key, value in config["environment"].items():
        task_toml += f"{key} = {json.dumps(value)}\n"
    task_toml += '\n[verifier]\nenvironment_mode = "shared"\n'
    (destination / "task.toml").write_text(task_toml)
    if environment_config.environment == DOCKER_ENVIRONMENT:
        for resource in specification.resources.all + specification.resources.worker:
            path = destination / "environment" / resource.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_resource_content(resource))
            path.chmod(int(resource.mode or DEFAULT_INLINE_FILE_MODE, 8))
    (destination / SPECIFICATION_FILE).write_text(specification.model_dump_json(indent=2) + "\n")
    (destination / ENVIRONMENT_CONFIG_FILE).write_text(environment_config.model_dump_json(indent=2) + "\n")
    (destination / SUBMISSION_CONVENTION_FILE).write_text(convention.model_dump_json(indent=2) + "\n")
    return destination
