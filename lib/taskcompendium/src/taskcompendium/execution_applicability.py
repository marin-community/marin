# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decide which task contracts require an agent harness and its execution gates."""

from taskcompendium.grading_environment import GradingEnvironmentMode, assess_grading_environment
from taskcompendium.harbor.adapter import actor_machine_required
from taskcompendium.models import AnswerType, TaskSpec


def is_nonagentic_task(task: TaskSpec) -> bool:
    """Identify answer-only tasks with no actor, interaction harness or workspace."""
    return not (
        actor_machine_required(task)
        or task.interaction_tools
        or task.tool_bindings
        or task.output_paths
        or task.answer_type in {AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE}
        or assess_grading_environment(task).required is GradingEnvironmentMode.SHARED
    )
