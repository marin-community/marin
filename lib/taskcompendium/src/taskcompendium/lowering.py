# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a TaskSpec submission as a Harbor task package."""

import hashlib
import json
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from taskcompendium.models import SCHEMA_VERSION, AnswerType, TaskSpec
from taskcompendium.submission import WORKSPACE_ROOT, SubmissionConvention, render_instruction, submission_compatibility
from taskcompendium.verifier_registry import resolve_verifier
from taskcompendium.verifiers.script import (
    MAX_RESOURCE_BYTES,
    PINNED_IMAGE,
    ResourceResolver,
    ScriptVerifier,
    materialize_private_resources,
)

DIRECT_CHAT_ENVIRONMENT = "direct_chat"
WORKSPACE_DOCKER_ENVIRONMENT = "workspace_docker"
SPECIFICATION_FILE = "specification.json"
SUBMISSION_CONVENTION_FILE = "submission_convention.json"
ENVIRONMENT_CONFIG_FILE = "environment_config.json"
PRIVATE_RESOURCES_DIR = "private_resources"


class HarborEnvironmentConfig(BaseModel):
    """The environment and tools this Harbor lowering exposes to the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str = DIRECT_CHAT_ENVIRONMENT
    tools: tuple[str, ...] = ()
    docker_image: str | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "HarborEnvironmentConfig":
        if self.environment == DIRECT_CHAT_ENVIRONMENT:
            if self.tools or self.docker_image is not None:
                raise ValueError("Direct chat cannot bind tools or a Docker image")
        elif self.environment == WORKSPACE_DOCKER_ENVIRONMENT:
            if self.docker_image is None or not PINNED_IMAGE.fullmatch(self.docker_image):
                raise ValueError("Workspace Docker environment requires a pinned image")
        else:
            raise ValueError(f"Unknown Harbor environment: {self.environment}")
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


def compatible_lowerings(
    specification: TaskSpec,
    convention_library: Sequence[SubmissionConvention],
    environment_configs: Sequence[HarborEnvironmentConfig],
) -> tuple[LoweringCandidate, ...]:
    """Enumerate conventions and environments that preserve this task's contract."""
    return tuple(
        LoweringCandidate(convention, environment_config)
        for convention in convention_library
        if submission_compatibility(specification, convention).compatible
        for environment_config in environment_configs
        if _is_compatible(specification, environment_config)
    )


def _is_compatible(specification: TaskSpec, environment_config: HarborEnvironmentConfig) -> bool:
    try:
        validate_environment_config(specification, environment_config)
    except ValueError:
        return False
    return True


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


def validate_environment_config(specification: TaskSpec, environment_config: HarborEnvironmentConfig) -> None:
    """Require the selected environment to satisfy the task's operations."""
    if environment_config.environment == DIRECT_CHAT_ENVIRONMENT:
        if (
            specification.environment_requirements.capabilities
            or specification.environment_requirements.action_interfaces
        ):
            raise ValueError("Direct chat cannot satisfy capability or action-interface requirements")
        if specification.answer_type in (AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE):
            raise ValueError("Direct chat cannot capture a final workspace")
        return
    if specification.environment_requirements.action_interfaces:
        raise ValueError("A provider-backed task needs an explicit verifier-side snapshot or bridge")
    if specification.answer_type == AnswerType.NATIVE_ACTION:
        raise ValueError("Native-action submissions need a provider-side snapshot or bridge")
    if not set(specification.environment_requirements.capabilities).issubset(set(environment_config.tools)):
        raise ValueError("Workspace Docker environment does not satisfy capability requirements")
    if specification.answer_type in (AnswerType.FILE, AnswerType.WORKSPACE_STATE) and not isinstance(
        resolve_verifier(specification.verifier), ScriptVerifier
    ):
        raise ValueError("File and workspace-state submissions require a script verifier")


def read_specification(path: Path) -> TaskSpec:
    data = json.loads(path.read_text())
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported TaskSpec schema: {data['schema_version']}")
    specification = TaskSpec.model_validate(data)
    resolve_verifier(specification.verifier)
    return specification


def read_environment_config(path: Path) -> HarborEnvironmentConfig:
    return HarborEnvironmentConfig.model_validate_json(path.read_text())


def read_submission_convention(path: Path) -> SubmissionConvention:
    return TypeAdapter(SubmissionConvention).validate_json(path.read_text())


def validate_exported_private_resources(specification: TaskSpec, task_dir: Path) -> None:
    """Check staged private bytes and paths before launching a trial."""
    verifier = resolve_verifier(specification.verifier)
    if not isinstance(verifier, ScriptVerifier):
        return
    root = task_dir / PRIVATE_RESOURCES_DIR
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing private resource directory")
    for resource in verifier.resources:
        path = root
        for part in resource.path.split("/"):
            path = path / part
            if path.is_symlink():
                raise ValueError(f"Private resource symlink: {resource.path}")
        if (
            not path.is_file()
            or path.stat().st_size > MAX_RESOURCE_BYTES
            or hashlib.sha256(path.read_bytes()).hexdigest() != resource.sha256
        ):
            raise ValueError(f"Private resource digest mismatch: {resource.path}")
        if bool(path.stat().st_mode & stat.S_IXUSR) != resource.executable:
            raise ValueError(f"Private resource executable bit differs: {resource.path}")


def lower_to_harbor(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    destination: Path,
    *,
    resource_resolver: ResourceResolver | None = None,
) -> Path:
    """Write one custom-verifier task; launch agent selection remains separate."""
    instruction = render_instruction(specification, convention)
    validate_environment_config(specification, environment_config)
    verifier = resolve_verifier(specification.verifier)
    with tempfile.TemporaryDirectory(prefix="taskcompendium-private-") as temporary:
        private_source = Path(temporary) / PRIVATE_RESOURCES_DIR
        if isinstance(verifier, ScriptVerifier):
            materialize_private_resources(verifier.resources, private_source, resource_resolver)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "environment").mkdir()
        if environment_config.environment == WORKSPACE_DOCKER_ENVIRONMENT:
            (destination / "environment" / "docker-compose.yaml").write_text(
                f"services:\n  main:\n    working_dir: {WORKSPACE_ROOT}\n"
            )
        (destination / "instruction.md").write_text(instruction)
        environment_lines = ['version = "1.0"', "", "[environment]", "allow_internet = false"]
        if environment_config.environment == WORKSPACE_DOCKER_ENVIRONMENT:
            environment_lines.extend(
                [
                    f"docker_image = {json.dumps(environment_config.docker_image)}",
                    f"workdir = {json.dumps(WORKSPACE_ROOT)}",
                ]
            )
        (destination / "task.toml").write_text("\n".join(environment_lines) + "\n")
        (destination / SPECIFICATION_FILE).write_text(specification.model_dump_json(indent=2) + "\n")
        (destination / ENVIRONMENT_CONFIG_FILE).write_text(environment_config.model_dump_json(indent=2) + "\n")
        (destination / SUBMISSION_CONVENTION_FILE).write_text(convention.model_dump_json(indent=2) + "\n")
        if isinstance(verifier, ScriptVerifier):
            shutil.copytree(private_source, destination / PRIVATE_RESOURCES_DIR)
    return destination
