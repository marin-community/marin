"""Apply SkyRL-owned graders to completed rollout groups."""

import asyncio
from collections.abc import Callable, Mapping, Sequence

from rolloutengine.contracts import RolloutData
from taskcompendium.models import TaskSpec

from skyrl_train.rollouts.genrm_grading import grade_genrm_rollouts
from skyrl_train.rollouts.group_grader import GENRM_GROUP_GRADER, GroupGraderSpec
from skyrl_train.rollouts.task_projections import rollout_loss_eligible
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig

GroupGrader = Callable[[TaskSpec, GroupGraderSpec, Sequence[RolloutData], Sequence[bool], str], Sequence[RolloutData]]
GROUP_GRADERS: Mapping[str, GroupGrader] = {GENRM_GROUP_GRADER: grade_genrm_rollouts}


async def grade_groups(
    tasks: Sequence[TaskSpec],
    specifications: Sequence[GroupGraderSpec | None],
    rollouts: Sequence[RolloutData],
    group_ids: Sequence[str],
    graders: Mapping[str, GroupGrader],
    phase: str,
    error_handling: Sequence[ErrorHandlingConfig],
    *,
    logprobs_required: bool,
) -> list[RolloutData]:
    """Grade each task group in request order before training conversion.

    Graders receive native records, execution eligibility, and the training phase.
    They return one record per input, in the same order. Private task data stays
    outside model requests.
    """
    groups: dict[str, list[int]] = {}
    for index, (task, rollout, group_id) in enumerate(zip(tasks, rollouts, group_ids, strict=True)):
        if rollout.task_id != task.id:
            raise ValueError("Rollout task ID does not match its request")
        groups.setdefault(group_id, []).append(index)
    result = list(rollouts)
    for indices in groups.values():
        task = tasks[indices[0]]
        if all(specifications[index] is None for index in indices):
            continue
        if any(tasks[index] != task for index in indices):
            raise ValueError(f"Task group {task.id!r} contains different task definitions")
        specification = specifications[indices[0]]
        if specification is None or any(specifications[index] != specification for index in indices):
            raise ValueError(f"Task group {task.id!r} contains different group graders")
        grader = graders[specification.name]
        records = [rollouts[index] for index in indices]
        eligible = [
            rollout_loss_eligible(rollouts[index], error_handling[index], logprobs_required=logprobs_required)
            for index in indices
        ]
        graded = await asyncio.to_thread(grader, task, specification, records, eligible, phase)
        for index, record in zip(indices, graded, strict=True):
            if record.task_id != task.id:
                raise ValueError("Group grader changed a rollout task ID")
            result[index] = record
    return result
