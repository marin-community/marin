# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a direct-chat TaskSpec rendering as a Harbor task package."""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from taskcompendium.models import AnswerFormat, ExactAnswer, Source, TaskRequirements, TaskSpec
from taskcompendium.rendering import Rendering, render_instruction

DIRECT_CHAT_ENVIRONMENT = "direct_chat"


@dataclass(frozen=True)
class HarborTaskBinding:
    """The environment and tools this Harbor lowering exposes to the agent."""

    environment: str = DIRECT_CHAT_ENVIRONMENT
    tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.environment != DIRECT_CHAT_ENVIRONMENT or self.tools:
            raise ValueError("This lowering supports direct chat without tools")


def validate_binding(specification: TaskSpec, binding: HarborTaskBinding) -> None:
    """Require direct chat to satisfy every declared semantic operation."""
    if binding != HarborTaskBinding():
        raise ValueError("Only direct-chat binding is supported")
    if specification.requirements.capabilities or specification.requirements.action_interfaces:
        raise ValueError("Direct chat cannot satisfy capability or action-interface requirements")


def read_specification(path: Path) -> TaskSpec:
    """Read the private semantic record from an exported Harbor task."""
    data = json.loads(path.read_text())
    return TaskSpec(
        id=data["id"],
        instructions=data["instructions"],
        verifier=ExactAnswer(**data["verifier"]),
        source=Source(**data["source"]),
        requirements=TaskRequirements(
            capabilities=tuple(data["requirements"]["capabilities"]),
            action_interfaces=tuple(data["requirements"]["action_interfaces"]),
        ),
        permitted_answer_formats=tuple(AnswerFormat(value) for value in data["permitted_answer_formats"]),
        schema_version=data["schema_version"],
    )


def read_binding(path: Path) -> HarborTaskBinding:
    """Read the target binding stored in an exported Harbor task."""
    data = json.loads(path.read_text())
    return HarborTaskBinding(environment=data["environment"], tools=tuple(data["tools"]))


def read_rendering(path: Path) -> Rendering:
    """Read the selected output convention from an exported Harbor task."""
    data = json.loads(path.read_text())
    return Rendering(id=data["id"], answer_format=AnswerFormat(data["answer_format"]))


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
    (destination / "specification.json").write_text(json.dumps(dataclasses.asdict(specification), indent=2) + "\n")
    (destination / "binding.json").write_text(json.dumps(dataclasses.asdict(binding), indent=2) + "\n")
    (destination / "rendering.json").write_text(json.dumps(dataclasses.asdict(rendering), indent=2) + "\n")
    return destination
