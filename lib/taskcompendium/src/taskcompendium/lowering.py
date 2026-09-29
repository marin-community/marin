# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a direct-chat TaskSpec submission as a Harbor task package."""

import hashlib
import importlib
import json
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.models import SCHEMA_VERSION, AnswerType, TaskSpec
from taskcompendium.resources import (
    MAX_RESOURCE_BYTES,
    MAX_TOTAL_RESOURCE_BYTES,
    ResourceResolver,
    ResourceVisibility,
    materialize_resources,
    validate_resource_path,
    validate_resources,
)
from taskcompendium.submission import SubmissionConvention, render_instruction
from taskcompendium.verifier_registry import validate_verifier

DIRECT_CHAT_ENVIRONMENT = "direct_chat"
STATEFUL_ENVIRONMENT = "stateful"
SPECIFICATION_FILE = "specification.json"
SUBMISSION_CONVENTION_FILE = "submission_convention.json"
ENVIRONMENT_CONFIG_FILE = "environment_config.json"
REGISTERED_PROVIDERS = {
    "nemo_workplace:v1": "taskcompendium.providers.nemo_workplace.provider:NemoWorkplaceEnvironment",
}


class HarborEnvironmentConfig(BaseModel):
    """The environment and tools this Harbor lowering exposes to the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str = DIRECT_CHAT_ENVIRONMENT
    tools: tuple[str, ...] = ()
    action_interface: str | None = None
    seed_sha256: str | None = None
    provider: str | None = None
    provider_revision: str | None = None
    tools_sha256: str | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "HarborEnvironmentConfig":
        if self.environment == DIRECT_CHAT_ENVIRONMENT:
            if self.tools or any(
                value is not None
                for value in (
                    self.action_interface,
                    self.seed_sha256,
                    self.provider,
                    self.provider_revision,
                    self.tools_sha256,
                )
            ):
                raise ValueError("Direct chat cannot bind tools or a provider")
        elif self.environment == STATEFUL_ENVIRONMENT:
            if self.provider not in REGISTERED_PROVIDERS:
                raise ValueError("Unknown stateful provider")
            if not all((self.action_interface, self.seed_sha256, self.provider_revision, self.tools_sha256)):
                raise ValueError("Stateful binding requires interface, seed, provider revision, and tools digest")
            for digest in (self.seed_sha256, self.tools_sha256):
                if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                    raise ValueError("Stateful binding requires lowercase SHA256 digests")
        else:
            raise ValueError(f"Unknown Harbor environment: {self.environment}")
        return self


def provider_class(config: HarborEnvironmentConfig) -> type:
    """Resolve the one registered implementation named by a stateful binding."""
    if config.provider is None:
        raise ValueError("Stateful provider is required")
    module_name, class_name = REGISTERED_PROVIDERS[config.provider].split(":")
    return getattr(importlib.import_module(module_name), class_name)


def validate_provider_surface(config: HarborEnvironmentConfig) -> None:
    """Check provider identity and action schemas before an export or launch."""
    provider = provider_class(config)
    for field, expected in (
        ("ACTION_INTERFACE", config.action_interface),
        ("SEED_SHA256", config.seed_sha256),
        ("PROVIDER_REVISION", config.provider_revision),
    ):
        if getattr(provider, field) != expected:
            raise ValueError(f"Provider {field} differs from Harbor binding")
    definitions = provider.TOOL_DEFINITIONS
    digest = hashlib.sha256(
        json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if digest != config.tools_sha256:
        raise ValueError("Provider tool schemas differ from Harbor binding")
    names = tuple(definition["function"]["name"] for definition in definitions)
    if names != config.tools or len(set(names)) != len(names):
        raise ValueError("Provider tool names differ from Harbor binding")


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
        if convention.supports(specification.answer_type)
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
    """Require the selected binding to satisfy the task's semantic requirements."""
    if environment_config.environment == DIRECT_CHAT_ENVIRONMENT:
        if specification.requirements.capabilities or specification.requirements.action_interfaces:
            raise ValueError("Direct chat cannot satisfy capability or action-interface requirements")
        if specification.answer_type == AnswerType.STATE:
            raise ValueError("Direct chat cannot grade environment state")
        return
    if specification.answer_type != AnswerType.STATE:
        raise ValueError("Stateful binding requires a state task")
    requirements = specification.requirements
    if requirements.capabilities or requirements.action_interfaces != (environment_config.action_interface,):
        raise ValueError("Stateful binding does not satisfy action-interface requirements")
    if requirements.seed_sha256 != environment_config.seed_sha256:
        raise ValueError("Stateful binding seed differs from task seed")
    validate_provider_surface(environment_config)


def read_specification(path: Path) -> TaskSpec:
    data = json.loads(path.read_text())
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported TaskSpec schema: {data['schema_version']}")
    specification = TaskSpec.model_validate(data)
    validate_verifier(specification.verifier)
    return specification


def read_environment_config(path: Path) -> HarborEnvironmentConfig:
    return HarborEnvironmentConfig.model_validate_json(path.read_text())


def read_submission_convention(path: Path) -> SubmissionConvention:
    return SubmissionConvention.model_validate_json(path.read_text())


def validate_exported_resources(specification: TaskSpec, task_dir: Path) -> None:
    """Recheck pinned payloads and path safety in the exported task before launch."""
    total_bytes = 0
    for resource in specification.resources:
        root = (
            task_dir / "environment" / "inputs"
            if resource.visibility == ResourceVisibility.AGENT
            else task_dir / "private_resources"
        )
        relative = validate_resource_path(resource.path)
        if root.is_symlink() or root.parent.is_symlink():
            raise ValueError(f"Exported resource root is a symlink: {root}")
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Exported resource symlink: {current}")
        if not current.is_file():
            raise ValueError(f"Missing exported resource: {resource.path}")
        if current.stat().st_size > MAX_RESOURCE_BYTES:
            raise ValueError("Exported resources exceed size limits")
        payload = current.read_bytes()
        total_bytes += len(payload)
        if len(payload) > MAX_RESOURCE_BYTES or total_bytes > MAX_TOTAL_RESOURCE_BYTES:
            raise ValueError("Exported resources exceed size limits")
        if resource.reference is not None:
            digest = resource.reference.sha256
        else:
            assert resource.content is not None
            digest = hashlib.sha256(resource.content.encode()).hexdigest()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Exported resource digest mismatch: {resource.path}")
        if bool(current.stat().st_mode & stat.S_IXUSR) != resource.executable:
            raise ValueError(f"Exported resource executable bit differs: {resource.path}")


def lower_to_harbor(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    destination: Path,
    *,
    trusted_resolver: ResourceResolver | None = None,
) -> Path:
    """Write one custom-verifier task; launch agent selection remains separate."""
    validate_environment_config(specification, environment_config)
    validate_verifier(specification.verifier)
    validate_resources(specification.resources, trusted_resolver=trusted_resolver)
    if environment_config.environment == DIRECT_CHAT_ENVIRONMENT and any(
        resource.visibility == ResourceVisibility.AGENT for resource in specification.resources
    ):
        raise ValueError("Direct chat cannot expose agent-visible files")
    instruction = render_instruction(specification, convention)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "environment").mkdir()
    (destination / "instruction.md").write_text(instruction)
    (destination / "task.toml").write_text(
        'version = "1.0"\n\n[environment]\nallow_internet = false\n\n[verifier]\nenvironment_mode = "shared"\n'
    )
    (destination / SPECIFICATION_FILE).write_text(specification.model_dump_json(indent=2) + "\n")
    (destination / ENVIRONMENT_CONFIG_FILE).write_text(environment_config.model_dump_json(indent=2) + "\n")
    (destination / SUBMISSION_CONVENTION_FILE).write_text(convention.model_dump_json(indent=2) + "\n")
    if any(resource.visibility == ResourceVisibility.AGENT for resource in specification.resources):
        materialize_resources(
            specification.resources,
            destination / "environment" / "inputs",
            visibility=ResourceVisibility.AGENT,
            trusted_resolver=trusted_resolver,
        )
    if any(resource.visibility != ResourceVisibility.AGENT for resource in specification.resources):
        materialize_resources(
            specification.resources,
            destination / "private_resources",
            visibility=frozenset({ResourceVisibility.VERIFIER, ResourceVisibility.ORACLE}),
            trusted_resolver=trusted_resolver,
        )
    return destination
