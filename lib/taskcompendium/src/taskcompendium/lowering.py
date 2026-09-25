# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a direct-chat TaskSpec rendering as a Harbor task package."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.models import TaskSpec
from taskcompendium.rendering import Rendering, render_instruction

DIRECT_CHAT_ENVIRONMENT = "direct_chat"
SPECIFICATION_FILE = "specification.json"
RENDERING_FILE = "rendering.json"
BINDING_FILE = "binding.json"


class HarborTaskBinding(BaseModel):
    """The environment and tools this Harbor lowering exposes to the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str = DIRECT_CHAT_ENVIRONMENT
    tools: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_direct_chat(self) -> "HarborTaskBinding":
        if self.environment != DIRECT_CHAT_ENVIRONMENT or self.tools:
            raise ValueError("This lowering supports direct chat without tools")
        return self


@dataclass(frozen=True)
class LoweringCandidate:
    """A compatible rendering and Harbor environment binding."""

    rendering: Rendering
    binding: HarborTaskBinding


class SelectionPolicy(StrEnum):
    """How a caller chooses from compatible lowerings."""

    ALL = "all"
    FIRST = "first"
    SAMPLE = "sample"


def compatible_lowerings(
    specification: TaskSpec,
    rendering_library: Sequence[Rendering],
    bindings: Sequence[HarborTaskBinding],
) -> tuple[LoweringCandidate, ...]:
    """Enumerate renderings and bindings that preserve this task's contract."""
    if specification.requirements.capabilities or specification.requirements.action_interfaces:
        return ()
    return tuple(
        LoweringCandidate(rendering, binding)
        for rendering in rendering_library
        if rendering.answer_format in specification.permitted_answer_formats
        for binding in bindings
    )


def select_lowerings(
    candidates: Sequence[LoweringCandidate],
    policy: SelectionPolicy,
    *,
    rng_key: int | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Select all, the first, or one keyed sample without global RNG state."""
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


def validate_binding(specification: TaskSpec, binding: HarborTaskBinding) -> None:
    """Require direct chat to satisfy every declared semantic operation."""
    if binding != HarborTaskBinding():
        raise ValueError("Only direct-chat binding is supported")
    if specification.requirements.capabilities or specification.requirements.action_interfaces:
        raise ValueError("Direct chat cannot satisfy capability or action-interface requirements")


def read_specification(path: Path) -> TaskSpec:
    return TaskSpec.model_validate_json(path.read_text())


def read_binding(path: Path) -> HarborTaskBinding:
    return HarborTaskBinding.model_validate_json(path.read_text())


def read_rendering(path: Path) -> Rendering:
    return Rendering.model_validate_json(path.read_text())


def lower_to_harbor(
    specification: TaskSpec,
    rendering: Rendering,
    binding: HarborTaskBinding,
    destination: Path,
) -> Path:
    """Write one custom-verifier task; launch agent selection remains separate."""
    validate_binding(specification, binding)
    instruction = render_instruction(specification, rendering)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "environment").mkdir()
    (destination / "instruction.md").write_text(instruction)
    (destination / "task.toml").write_text('version = "1.0"\n\n[environment]\nallow_internet = false\n')
    (destination / SPECIFICATION_FILE).write_text(specification.model_dump_json(indent=2) + "\n")
    (destination / BINDING_FILE).write_text(binding.model_dump_json(indent=2) + "\n")
    (destination / RENDERING_FILE).write_text(rendering.model_dump_json(indent=2) + "\n")
    return destination
