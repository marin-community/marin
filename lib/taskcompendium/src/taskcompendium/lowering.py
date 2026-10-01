# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a TaskSpec and selected chat binding as a Harbor task package."""

import hashlib
import inspect
import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from taskcompendium.models import SCHEMA_VERSION, SHA256_PATTERN, AnswerType, TaskSpec
from taskcompendium.path_validation import validate_relative_file_paths
from taskcompendium.provider_sources import (
    PROVIDER_SOURCES_DIR,
    ToolProviderCache,
    parse_git_provider,
    stage_git_provider,
)
from taskcompendium.submission import (
    ANSWER_CALL_NAME,
    AnswerFormat,
    ProviderState,
    SubmissionConvention,
    render_instruction,
    submission_compatibility,
)
from taskcompendium.tool_provider import ToolProviderFactory, tool_schema_sha256
from taskcompendium.verifier_registry import validate_verifier

SPECIFICATION_FILE = "specification.json"
SUBMISSION_CONVENTION_FILE = "submission_convention.json"
ENVIRONMENT_CONFIG_FILE = "environment_config.json"
ENVIRONMENT_DIR = "environment"


class ToolBinding(BaseModel):
    """Bind a task's named tool service to an implementation and selected schemas.

    A tool provider advertises functions and executes their calls against its
    trial-local state. It is composed into the Harbor environment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Versioned action contract, such as "nemo_workplace:v1".
    action_interface: str = Field(min_length=1)
    seed_sha256: str = Field(pattern=SHA256_PATTERN)
    # Python class locator selected by the code that prepares and launches tasks.
    provider: str
    # Immutable implementation revision expected at export and launch.
    provider_revision: str = Field(min_length=1)
    # Ordered names of functions selected from the provider for this task.
    tools: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)
    tools_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_binding(self) -> "ToolBinding":
        if parse_git_provider(self.provider) is None:
            scheme, separator, import_path = self.provider.partition(":")
            if scheme != "python" or not separator:
                raise ValueError("Tool provider must be a Python import or pinned HTTPS Git source")
            module_name, separator, class_name = import_path.partition(":")
            if (
                not separator
                or not class_name.isidentifier()
                or not all(part.isidentifier() for part in module_name.split("."))
            ):
                raise ValueError("Tool provider must be a python:module:Class import path")
        if len(set(self.tools)) != len(self.tools):
            raise ValueError("Tool binding requires unique tool names")
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


def provider_class(
    binding: ToolBinding, provider_source: Path | None = None, *, cache: ToolProviderCache
) -> ToolProviderFactory:
    """Resolve an implementation for the lifetime of the caller's cache."""
    return cache.stage(binding.provider, provider_source).factory


def selected_tool_definitions(definitions: Sequence[dict[str, Any]], tool_names: Sequence[str]) -> list[dict[str, Any]]:
    """Advertise only the bound functions; other provider functions stay hidden."""
    available: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        name = definition["function"]["name"]
        if not isinstance(name, str) or name in available:
            raise ValueError("Provider tool definitions require unique names")
        available[name] = definition
    if any(name not in available for name in tool_names):
        raise ValueError("Provider is missing a bound tool")
    return [available[name] for name in tool_names]


def validate_provider_surface(
    binding: ToolBinding, provider_source: Path | None = None, *, cache: ToolProviderCache
) -> None:
    """Check provider identity and action schemas before an export or launch."""
    provider = provider_class(binding, provider_source, cache=cache)
    for method in ("native_tool_definitions", "dispatch_action"):
        if not inspect.iscoroutinefunction(getattr(provider, method, None)):
            raise ValueError(f"Tool provider requires async {method}")
    for field, expected in (
        ("ACTION_INTERFACE", binding.action_interface),
        ("SEED_SHA256", binding.seed_sha256),
        ("PROVIDER_REVISION", binding.provider_revision),
    ):
        if getattr(provider, field) != expected:
            raise ValueError(f"Provider {field} differs from Harbor binding")
    definitions = selected_tool_definitions(provider.TOOL_DEFINITIONS, binding.tools)
    digest = tool_schema_sha256(definitions)
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
    *,
    trusted_provider_sources: dict[str, Path] | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Enumerate conventions and environments that preserve this task's contract."""
    return tuple(
        LoweringCandidate(convention, environment_config)
        for convention in convention_library
        if submission_compatibility(specification, convention).compatible
        for environment_config in environment_configs
        if is_compatible_lowering(
            specification, convention, environment_config, trusted_provider_sources=trusted_provider_sources
        )
    )


def is_compatible_lowering(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    *,
    trusted_provider_sources: dict[str, Path] | None = None,
) -> bool:
    """Whether the convention and selected providers can run this task together."""
    try:
        validate_environment_candidate(specification, convention, environment_config, trusted_provider_sources)
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
    if specification.resources:
        raise ValueError("Host chat cannot expose task resources")
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


def validate_environment_candidate(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    trusted_provider_sources: dict[str, Path] | None,
) -> None:
    """Check requirements and pinned provider implementations before selection."""
    _validate_provider_requirements(specification, convention, environment_config)
    with tempfile.TemporaryDirectory(prefix="taskcompendium-candidate-") as temporary, ToolProviderCache() as cache:
        sources: dict[str, Path] = {}
        for name, binding in environment_config.tool_providers.items():
            if parse_git_provider(binding.provider) is None:
                continue
            if trusted_provider_sources is None or name not in trusted_provider_sources:
                raise ValueError(f"Git provider requires a trusted source checkout: {name}")
            source = Path(temporary) / name
            stage_git_provider(binding.provider, trusted_provider_sources[name], source)
            sources[name] = source
        validate_environment_config(specification, convention, environment_config, provider_sources=sources, cache=cache)


def validate_environment_config(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    *,
    provider_sources: dict[str, Path] | None = None,
    cache: ToolProviderCache,
) -> None:
    """Check all selected provider implementations against their pinned surfaces."""
    state_provider = _validate_provider_requirements(specification, convention, environment_config)
    for name, binding in environment_config.tool_providers.items():
        source = provider_sources[name] if provider_sources is not None and name in provider_sources else None
        validate_provider_surface(binding, source, cache=cache)
        if name == state_provider and not callable(
            getattr(provider_class(binding, source, cache=cache), "canonical_state", None)
        ):
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
    *,
    trusted_provider_sources: dict[str, Path] | None = None,
) -> Path:
    """Write one custom-verifier task; launch agent selection remains separate."""
    git_bindings = {
        name: binding
        for name, binding in environment_config.tool_providers.items()
        if parse_git_provider(binding.provider) is not None
    }
    compatibility = submission_compatibility(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention is incompatible: {'; '.join(compatibility.reasons)}")
    validate_submission_tools(specification, convention, environment_config)
    validate_verifier(specification.verifier)
    instruction = render_instruction(specification, convention)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with ToolProviderCache() as cache:
            (destination / ENVIRONMENT_DIR).mkdir()
            staged_sources: dict[str, Path] = {}
            for name, binding in git_bindings.items():
                if trusted_provider_sources is None or name not in trusted_provider_sources:
                    raise ValueError(f"Git provider requires a trusted source checkout: {name}")
                source = destination / ENVIRONMENT_DIR / PROVIDER_SOURCES_DIR / name
                source.parent.mkdir(parents=True, exist_ok=True)
                stage_git_provider(binding.provider, trusted_provider_sources[name], source)
                staged_sources[name] = source
            validate_environment_config(
                specification, convention, environment_config, provider_sources=staged_sources, cache=cache
            )
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
