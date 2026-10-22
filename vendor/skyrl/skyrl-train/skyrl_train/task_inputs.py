"""Task and preference inputs shared by dataset loading and rollout requests."""

from dataclasses import dataclass
from typing import Any

from rolloutengine.harness.shell import task_prompt
from rolloutengine.models import TaskRolloutSpec
from skyrl_gym.rollout_specs import is_shell_harness, shell_harness
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


@dataclass(frozen=True)
class FixedCompletionInput:
    """One pre-generated training row: a prompt plus its fixed completion.

    ``prompt`` is a message list (text rows) or prompt token IDs (tokenized rows).
    ``completion`` is completion text or, for tokenized rows, response token IDs that
    begin at the first supervised token. ``completion_loss_mask`` is None for text
    rows (every completion token is supervised) and aligns one-to-one with tokenized
    completion IDs.
    """

    prompt: ConversationType | list[int]
    completion: str | list[int]
    metadata: TrajectoryMetadata
    completion_loss_mask: tuple[int, ...] | None = None


def task_messages(spec: TaskRolloutSpec) -> ConversationType:
    if not is_shell_harness(spec.harness):
        return list(episode_prompt(spec.task).messages)
    tool = resolve_shell_tool(spec.task, shell_harness(spec.harness).shell_tool)
    return list(task_prompt(spec.task, tool, ()).messages)
