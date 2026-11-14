# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package unchanged TaskSpecs for Harbor's TaskSpec trial workload."""

import json
import re
import tomllib
from pathlib import Path

import tomlkit

from taskcompendium.grading_environment import GradingEnvironmentMode, assess_grading_environment
from taskcompendium.harbor.execution import native_execution_limits
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    ExecutionRequirements,
    NoGrader,
    TaskResource,
    TaskSpec,
)
from taskcompendium.repository_shell import SHELL_ACTION_INTERFACE, SHELL_TOOL, repository_shell_environment
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import conversation_messages

TASK_SPEC_FILE = "task-spec.json"
ARCHIVED_SOURCE_CONFIG = "source_archive/task.toml"
ARCHIVED_SOURCE_PROVENANCE = "taskcompendium/crosscode-remediation.json"


def source_task_configurations(task: TaskSpec) -> tuple[TaskResource, ...]:
    """Return source task configurations, requiring provenance for repaired oracle archives."""
    configurations = [
        resource
        for resources in (task.resources.all, task.resources.worker, task.resources.verifier, task.resources.oracle)
        for resource in resources
        if resource.path == "task.toml"
    ]
    oracle = {resource.path: resource for resource in task.resources.oracle}
    if ARCHIVED_SOURCE_CONFIG in oracle:
        if ARCHIVED_SOURCE_PROVENANCE not in oracle:
            raise ValueError(f"{task.id}: archived source configuration requires repair provenance")
        provenance = json.loads(resource_bytes(oracle[ARCHIVED_SOURCE_PROVENANCE]))
        if (
            not provenance["version"]
            or provenance["version"] != task.labels.get("remediation_version")
            or re.fullmatch(r"[0-9a-f]{64}", provenance["original_task_sha256"]) is None
        ):
            raise ValueError(f"{task.id}: archived source configuration has conflicting repair provenance")
        configurations.append(oracle[ARCHIVED_SOURCE_CONFIG])
    return tuple(configurations)


def declared_execution_limits(task: TaskSpec) -> dict[str, int | float]:
    """Read source limits, including provenance-bound oracle archives, rejecting disagreements."""
    declared = task.execution_requirements.model_dump() if task.execution_requirements is not None else {}
    for resource in source_task_configurations(task):
        limits = native_execution_limits(tomllib.loads(resource_bytes(resource).decode()))
        for name, value in limits.declared.items():
            if name in declared and declared[name] != value:
                raise ValueError(f"{task.id}: conflicting original {name} declarations")
            declared[name] = value
    return declared


def unsupported_harbor_contract(task: TaskSpec) -> str | None:
    """Describe contracts that require a different executor, without rewriting them."""
    if isinstance(task.grader, NoGrader):
        return "no_executable_task_spec_grader"
    if task.answer_type == AnswerType.STATE:
        return "task_spec_state_capture_required"
    for requirement in task.environment_requirements.tool_providers.values():
        if requirement.action_interface != SHELL_ACTION_INTERFACE:
            return "task_spec_tool_provider_required"
    try:
        state = repository_shell_environment(task.environment_requirements)
    except ValueError:
        return "task_spec_provider_contract_invalid"
    if state is not None:
        if (
            task.tool_bindings
            or any(tool != SHELL_TOOL for tool in task.interaction_tools)
            or any(tool.name == SHELL_TOOL.name for tool in task.final_tools)
        ):
            return "task_spec_provider_contract_invalid"
    if task.integrity_contract is not None or assess_grading_environment(task).required == GradingEnvironmentMode.SHARED:
        return "task_spec_shared_executor_required"
    environments = [task.environment_requirements]
    if task.grader.environment is not None:
        environments.append(task.grader.environment)
    if any(environment.docker_build is not None for environment in environments):
        return "task_spec_unresolved_build"
    return None


def actor_machine_required(task: TaskSpec) -> bool:
    return task.environment_requirements != EnvironmentRequirements() or bool(task.resources.worker)


def require_trial_budget(task: TaskSpec, requirements: ExecutionRequirements) -> None:
    """Require wrapper budgets to preserve every declared TaskSpec limit."""
    if task.execution_requirements is not None and task.execution_requirements != requirements:
        raise ValueError(f"{task.id}: trial budgets conflict with execution_requirements")
    for name, value in declared_execution_limits(task).items():
        if getattr(requirements, name) != value:
            raise ValueError(f"{task.id}: trial budget conflicts with original {name}")


def write_harbor_task(task: TaskSpec, directory: Path, requirements: ExecutionRequirements) -> None:
    """Write a private TaskSpec payload and its trial budgets, preserving the task itself.

    ``requirements`` supplies a calibration shape when the original lacks one. Explicit
    requirements must match exactly. No environment, file delivery or grader is lowered.
    """
    unsupported = unsupported_harbor_contract(task)
    if unsupported is not None:
        raise ValueError(f"{task.id}: {unsupported}")
    require_trial_budget(task, requirements)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "environment").mkdir()
    (directory / TASK_SPEC_FILE).write_text(task.model_dump_json() + "\n")
    # This is the human-readable task view; the TaskSpec agent consumes the complete conversation.
    (directory / "instruction.md").write_text(json.dumps(conversation_messages(task.context.events), ensure_ascii=False))
    (directory / "task.toml").write_text(
        tomlkit.dumps(
            {
                "environment": {
                    "cpus": requirements.cpus,
                    "memory_mb": requirements.memory_mb,
                    "storage_mb": requirements.storage_mb,
                },
                "agent": {"timeout_sec": requirements.agent_timeout_sec},
                "verifier": {"timeout_sec": requirements.verifier_timeout_sec, "environment_mode": "separate"},
            }
        )
    )
