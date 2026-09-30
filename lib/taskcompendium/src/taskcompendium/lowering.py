# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a TaskSpec and selected chat binding as a Harbor task package."""

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from taskcompendium.container_service import ContainerService, validate_container_image
from taskcompendium.models import SCHEMA_VERSION, AnswerType, TaskSpec
from taskcompendium.path_validation import validate_relative_file_paths
from taskcompendium.submission import (
    ANSWER_CALL_NAME,
    AnswerFormat,
    ProviderState,
    SubmissionConvention,
    render_instruction,
    submission_compatible,
)
from taskcompendium.verifier_registry import validate_verifier

ENVIRONMENT_DIR = "environment"
SPECIFICATION_FILE = "specification.json"
SUBMISSION_CONVENTION_FILE = "submission_convention.json"
ENVIRONMENT_CONFIG_FILE = "environment_config.json"


class ToolBinding(BaseModel):
    """A declared tool surface served by a pinned, isolated container."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_interface: str = Field(min_length=1)
    seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_revision: str = Field(min_length=1)
    runtime: ContainerService
    tools: tuple[str, ...] = Field(min_length=1)
    tools_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_definitions: tuple[dict[str, Any], ...]
    state_available: bool

    @model_validator(mode="after")
    def validate_binding(self) -> "ToolBinding":
        if len(set(self.tools)) != len(self.tools) or any(not name for name in self.tools):
            raise ValueError("Tool binding requires unique nonempty tool names")
        validate_provider_surface(self)
        return self


class HarborEnvironmentConfig(BaseModel):
    """A chat environment with zero or more pinned tool surfaces."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_providers: dict[str, ToolBinding] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bindings(self) -> "HarborEnvironmentConfig":
        validate_relative_file_paths(self.tool_providers)
        if any("/" in name for name in self.tool_providers):
            raise ValueError("Tool provider names must be single path components")
        tool_names = [name for binding in self.tool_providers.values() for name in binding.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("Tool names must be unique across providers")
        return self


def selected_tool_definitions(definitions: Sequence[dict[str, Any]], tool_names: Sequence[str]) -> list[dict[str, Any]]:
    """Select the pinned public surface from a provider that may expose more tools."""
    available: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        name = definition["function"]["name"]
        if not isinstance(name, str) or name in available:
            raise ValueError("Provider tool definitions require unique names")
        available[name] = definition
    if any(name not in available for name in tool_names):
        raise ValueError("Provider is missing a bound tool")
    return [available[name] for name in tool_names]


def validate_provider_surface(binding: ToolBinding) -> None:
    """Validate a caller's declared schemas without importing provider code."""
    definitions = selected_tool_definitions(binding.tool_definitions, binding.tools)
    digest = hashlib.sha256(
        json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if digest != binding.tools_sha256:
        raise ValueError("Provider tool schemas differ from Harbor binding")


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
        if submission_compatible(specification, convention).compatible
        for environment_config in environment_configs
        if is_compatible_lowering(specification, convention, environment_config)
    )


def is_compatible_lowering(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
) -> bool:
    """Whether the convention and selected providers can run this task together."""
    try:
        validate_environment_config(specification, convention, environment_config)
        validate_submission_tools(specification, convention, environment_config)
    except ValueError:
        return False
    return True


def select_lowerings(
    candidates: Sequence[LoweringCandidate],
    policy: SelectionPolicy,
    *,
    rng_key: int | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Select from the caller's compatible lowering candidates."""
    if not candidates:
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


def _validate_provider_requirements(
    specification: TaskSpec, convention: SubmissionConvention, environment_config: HarborEnvironmentConfig
) -> str | None:
    """Validate host-chat requirements and locate a provider-state submission."""
    bindings = environment_config.tool_providers
    requirements = specification.environment_requirements
    if requirements.capabilities:
        raise ValueError("Host chat cannot satisfy workspace capability requirements")
    if not bindings:
        if specification.tool_providers:
            raise ValueError("Chat without tools cannot satisfy provider requirements")
        if specification.answer_type == AnswerType.STATE:
            raise ValueError("Host chat without a provider cannot expose state")
        return None
    required = specification.tool_providers
    selected = bindings
    if required.keys() != selected.keys():
        raise ValueError("Tool bindings do not satisfy provider requirements")
    for name, binding in selected.items():
        requirement = required[name]
        if binding.action_interface != requirement.action_interface:
            raise ValueError(f"Tool binding interface differs from task requirement: {name}")
        if binding.seed_sha256 != requirement.seed_sha256:
            raise ValueError(f"Tool binding seed differs from task requirement: {name}")
    if isinstance(convention, ProviderState):
        if convention.provider not in selected:
            raise ValueError("State convention names a missing provider")
        return convention.provider
    return None


def validate_environment_config(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
) -> None:
    """Check selected container manifests against the task's semantic requirements."""
    state_provider = _validate_provider_requirements(specification, convention, environment_config)
    for name, binding in environment_config.tool_providers.items():
        validate_provider_surface(binding)
        validate_container_image(
            binding.runtime,
            {
                "action_interface": binding.action_interface,
                "seed_sha256": binding.seed_sha256,
                "provider_revision": binding.provider_revision,
            },
            list(binding.tool_definitions),
        )
        if name == state_provider and not binding.state_available:
            raise ValueError("State task requires canonical provider state")


def validate_submission_tools(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
) -> None:
    """Keep terminal submission functions distinct from executable provider tools."""
    terminal_names = {function.name for function in specification.final_tools}
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        terminal_names.add(ANSWER_CALL_NAME)
    provider_names = {name for binding in environment_config.tool_providers.values() for name in binding.tools}
    if provider_names & terminal_names:
        raise ValueError("Submission function names collide with provider tools")


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
    return TypeAdapter(SubmissionConvention).validate_json(path.read_text())


def lower_to_harbor(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    destination: Path,
) -> Path:
    """Write one custom-verifier task; launch agent selection remains separate."""
    validate_environment_config(specification, convention, environment_config)
    compatibility = submission_compatible(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention is incompatible: {'; '.join(compatibility.reasons)}")
    validate_submission_tools(specification, convention, environment_config)
    validate_verifier(specification.verifier)
    instruction = render_instruction(specification, convention)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / ENVIRONMENT_DIR).mkdir()
        _write_harbor_task(specification, convention, environment_config, destination, instruction)
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def _write_harbor_task(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    destination: Path,
    instruction: str,
) -> None:
    """Write files after all selected provider sources have been checked."""
    (destination / "instruction.md").write_text(instruction)
    (destination / "task.toml").write_text(
        'version = "1.0"\n\n[environment]\nallow_internet = false\n\n[verifier]\nenvironment_mode = "shared"\n'
    )
    # Harbor's pinned revision requires a test script even when a custom verifier runs.
    tests_dir = destination / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (destination / SPECIFICATION_FILE).write_text(specification.model_dump_json(indent=2) + "\n")
    (destination / ENVIRONMENT_CONFIG_FILE).write_text(environment_config.model_dump_json(indent=2) + "\n")
    (destination / SUBMISSION_CONVENTION_FILE).write_text(convention.model_dump_json(indent=2) + "\n")
