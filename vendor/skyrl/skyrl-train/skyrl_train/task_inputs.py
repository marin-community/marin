"""Task and preference inputs shared by dataset loading and rollout requests."""

from dataclasses import dataclass
from typing import Any

from rolloutengine.lowering import SHELLBOX_SESSION
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.task_session import session_start
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
    lowered: LoweredTaskSpec
    metadata: TrajectoryMetadata
    group_grader: GroupGraderSpec | None = None


@dataclass(frozen=True)
class PreferenceInput:
    prompt: ConversationType | list[int]
    preference: dict[str, Any]
    metadata: TrajectoryMetadata


def task_messages(lowered: LoweredTaskSpec) -> ConversationType:
    tool = (
        resolve_shell_tool(lowered.task, lowered.session.shell_tool)
        if lowered.session.task_session == SHELLBOX_SESSION
        else None
    )
    return list(session_start(lowered.task, tool, ()).messages)
