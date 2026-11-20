# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark environment identities for canonical tasks."""

from taskcompendium.convert.nemotron_ultra import blend_component
from taskcompendium.grader import grader_config
from taskcompendium.models import TaskSpec

NEMOTRON_ULTRA_DATASET = "nvidia/Nemotron-RL-Ultra-Training-Blends"


def benchmark_env_id(task: TaskSpec, *, source_env_id: str | None) -> str:
    """Identify the task's benchmark environment from its labels and source."""
    explicit_id = task.labels.get("env_id") or source_env_id
    if explicit_id:
        return explicit_id

    # The blend shares one source dataset across several grading environments.
    if task.source.dataset == NEMOTRON_ULTRA_DATASET and any(
        resource.path == "config.json" for resource in task.resources.verifier
    ):
        contract = grader_config(task).get("contract")
        if isinstance(contract, dict):
            component = contract.get("dataset")
            if isinstance(component, str) and component:
                return component
            agent = contract.get("agent_ref")
            if isinstance(agent, dict) and isinstance(agent.get("name"), str) and agent["name"]:
                return blend_component({"agent_ref": agent})

    return task.source.dataset
