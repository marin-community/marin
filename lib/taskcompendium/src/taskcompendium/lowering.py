# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a direct-chat TaskSpec submission as a Harbor task package."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.grading import validate_verifier
from taskcompendium.models import SCHEMA_VERSION, TaskSpec
from taskcompendium.submission import SubmissionConvention, render_instruction

DIRECT_CHAT_ENVIRONMENT = "direct_chat"
SPECIFICATION_FILE = "specification.json"
SUBMISSION_CONVENTION_FILE = "submission_convention.json"
ENVIRONMENT_CONFIG_FILE = "environment_config.json"


class HarborEnvironmentConfig(BaseModel):
    """The environment and tools this Harbor lowering exposes to the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str = DIRECT_CHAT_ENVIRONMENT
    tools: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_direct_chat(self) -> "HarborEnvironmentConfig":
        if self.environment != DIRECT_CHAT_ENVIRONMENT or self.tools:
            raise ValueError("This lowering supports direct chat without tools")
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
    if specification.requirements.capabilities or specification.requirements.action_interfaces:
        return ()
    return tuple(
        LoweringCandidate(convention, environment_config)
        for convention in convention_library
        if convention.supports(specification.answer_type)
        for environment_config in environment_configs
    )


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
    """Require direct chat to satisfy every declared semantic operation."""
    if environment_config != HarborEnvironmentConfig():
        raise ValueError("Only direct-chat environment configuration is supported")
    if specification.requirements.capabilities or specification.requirements.action_interfaces:
        raise ValueError("Direct chat cannot satisfy capability or action-interface requirements")


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


def lower_to_harbor(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    destination: Path,
) -> Path:
    """Write one custom-verifier task; launch agent selection remains separate."""
    validate_environment_config(specification, environment_config)
    validate_verifier(specification.verifier)
    instruction = render_instruction(specification, convention)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "environment").mkdir()
    (destination / "instruction.md").write_text(instruction)
    (destination / "task.toml").write_text('version = "1.0"\n\n[environment]\nallow_internet = false\n')
    (destination / SPECIFICATION_FILE).write_text(specification.model_dump_json(indent=2) + "\n")
    (destination / ENVIRONMENT_CONFIG_FILE).write_text(environment_config.model_dump_json(indent=2) + "\n")
    (destination / SUBMISSION_CONVENTION_FILE).write_text(convention.model_dump_json(indent=2) + "\n")
    return destination
