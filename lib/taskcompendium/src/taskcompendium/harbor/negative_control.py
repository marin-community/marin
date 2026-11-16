# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original source identities and public preparation for physical nop controls."""

import hashlib
import tomllib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.harbor.execution import native_execution_limits
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, NoGrader, TaskSpec
from taskcompendium.runtime.resources import resource_bytes


class NegativeControlSetupKind(StrEnum):
    NATIVE_SCRIPT = "native_script"
    DECLARED_SETUP = "declared_setup"
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


class DeclaredSourceSetup(BaseModel):
    """Bind native Trial preparation to the unchanged original source TaskSpec."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    task_spec: TaskSpec


def declared_setup_preparation(task: TaskSpec, source: NegativeControlSource, config_bytes: bytes) -> dict:
    """Verify the consumed native setup against its original source declaration."""
    config = tomllib.loads(config_bytes.decode())
    setup = config["environment"]["setup"]
    original = next(resource for resource in task.resources.verifier if resource.path == "task.toml")
    original_setup = tomllib.loads(resource_bytes(original).decode())["environment"]["setup"]
    if (
        negative_control_setup_kind(task) is not NegativeControlSetupKind.DECLARED_SETUP
        or task.id != source.task_id
        or canonical_sha256(task.model_dump(mode="json")) != source.source_task_sha256
        or task.environment_requirements.setup_commands != (setup["command"],)
        or setup != original_setup
        or native_execution_limits(config).requirements != source.execution_requirements
        or not setup["command"].strip()
        or setup["timeout_sec"] <= 0
    ):
        raise ValueError("Declared negative-control setup differs from the original source")
    return {
        "version": 1,
        "setup_kind": NegativeControlSetupKind.DECLARED_SETUP.value,
        "task_spec_file": "source-task-spec.json",
        "task_spec_sha256": source.source_task_sha256,
        "source_task_checksum": source.source_task_checksum,
        "source_native_checksum": source.task_checksum,
        "native_config_file": "consumed-task.toml",
        "native_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "declared_setup": setup,
        "declared_setup_executed_once_by_trial": True,
    }


def require_nop_trial_config(config: dict) -> None:
    """Reject actor, environment, grading and timeout overrides in a physical control."""
    agent = config.get("agent") or {}
    environment = config.get("environment") or {}
    verifier = config.get("verifier") or {}
    if agent.get("name") != "nop" or any(
        agent.get(name) for name in ("model_name", "model_alias", "kwargs", "env", "skills", "mcp_servers")
    ):
        raise ValueError("A physical negative control requires the unmodified nop agent")
    if (
        config.get("timeout_multiplier", 1) != 1
        or config.get("agent_timeout_multiplier") not in (None, 1)
        or config.get("verifier_timeout_multiplier") not in (None, 1)
        or any(
            agent.get(name) is not None
            for name in ("override_timeout_sec", "override_setup_timeout_sec", "max_timeout_sec")
        )
        or any(verifier.get(name) is not None for name in ("override_timeout_sec", "max_timeout_sec", "import_path"))
        or verifier.get("disable")
        or verifier.get("kwargs")
        or verifier.get("env")
        or config.get("artifacts")
        or config.get("extra_instruction_paths")
        or environment.get("force_build")
        or environment.get("mounts")
        or environment.get("env")
        or any(
            environment.get(name) is not None
            for name in ("override_cpus", "override_memory_mb", "override_storage_mb", "override_gpus", "override_tpu")
        )
    ):
        raise ValueError("A physical negative control cannot override its source grading, environment or budgets")


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
        if task.environment_requirements.setup_commands:
            return NegativeControlSetupKind.DECLARED_SETUP
        return NegativeControlSetupKind.NATIVE_SCRIPT
    return NegativeControlSetupKind.TASK_SPEC_PREPARATION
