"""Task and preference inputs shared by dataset loading and rollout requests."""

from dataclasses import dataclass
from typing import Any

from rolloutengine.harness.tools import task_prompt
from rolloutengine.harness.prefill import prefill_prompt
from rolloutengine.models import TaskRolloutSpec
from skyrl_gym.rollout_specs import PREFILL_HARNESS, is_shell_harness, shell_harness
from skyrl_gym.task_records import episode_prompt
from taskcompendium.runtime.shell import resolve_shell_tool

from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.rollouts.group_grader import GroupGraderSpec


@dataclass(frozen=True)
class TrajectoryMetadata:
    uid: str
    metric_class: str
    data_source: str | None = None
    teacher_route: str | None = None
    metric_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskInput:
    spec: TaskRolloutSpec
    metadata: TrajectoryMetadata
    group_grader: GroupGraderSpec | None = None


@dataclass(frozen=True)
class PreferenceInput:
    prompt: ConversationType | list[int]
    preference: dict[str, Any]
    metadata: TrajectoryMetadata


def task_messages(spec: TaskRolloutSpec) -> ConversationType:
    if spec.harness == PREFILL_HARNESS:
        return list(prefill_prompt(spec.task).messages)
    if not is_shell_harness(spec.harness):
        return list(episode_prompt(spec.task).messages)
    tool = resolve_shell_tool(spec.task, shell_harness(spec.harness).shell_tool)
    return list(task_prompt(spec.task, tool, ()).messages)
