# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original source identities and public preparation for physical nop controls."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import ExecutionRequirements, NoGrader, TaskSpec


class NegativeControlSetupKind(StrEnum):
    NATIVE_SCRIPT = "native_script"
    TASK_SPEC_PREPARATION = "task_spec_preparation"


class NegativeControlSource(BaseModel):
    """The original source and exact final package/budgets a control must consume."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    source_task_sha256: str
    source_task_checksum: str
    task_sha256: str
    runtime_sha256: str | None
    task_checksum: str
    execution_requirements: ExecutionRequirements
    verifier_execution_requirements: ExecutionRequirements | None


class NativeInstructionSetup(BaseModel):
    """Select one verbatim public bash block from the original instruction."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    block_index: int = Field(ge=0)


def public_task_preparation(task: TaskSpec) -> dict:
    """Describe the environment and files installed before a TaskSpec actor starts."""
    return {
        "environment_requirements": task.environment_requirements.model_dump(mode="json"),
        "resources": {
            "all": [resource.model_dump(mode="json") for resource in task.resources.all],
            "worker": [resource.model_dump(mode="json") for resource in task.resources.worker],
        },
    }


def negative_control_setup_kind(task: TaskSpec) -> NegativeControlSetupKind:
    """Distinguish native Harbor payloads from executable converted TaskSpecs."""
    if isinstance(task.grader, NoGrader) and "harbor" in task.tags:
        return NegativeControlSetupKind.NATIVE_SCRIPT
    return NegativeControlSetupKind.TASK_SPEC_PREPARATION
