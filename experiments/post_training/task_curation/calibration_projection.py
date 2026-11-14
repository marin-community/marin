# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish resolved allocations without changing the source execution contract."""

from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.runtime.resources import inline_resource


def calibrated_task_spec(
    task: TaskSpec,
    actor: ExecutionRequirements,
    verifier: ExecutionRequirements | None,
    native_configuration: bytes,
) -> TaskSpec:
    """Apply measured allocations and retain the exact final native configuration."""
    configuration = inline_resource("task.toml", native_configuration)
    return task.model_copy(
        update={
            "execution_requirements": actor,
            **({"verifier_execution_requirements": verifier} if verifier is not None else {}),
            "resources": task.resources.model_copy(
                update={
                    "verifier": tuple(
                        (
                            resource.model_copy(update={"source": configuration.source})
                            if resource.path == "task.toml"
                            else resource
                        )
                        for resource in task.resources.verifier
                    )
                }
            ),
        }
    )
