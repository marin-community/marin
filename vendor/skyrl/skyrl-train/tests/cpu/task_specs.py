"""Explicit runtime settings for task execution tests."""

from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from skyrl_gym.rollout_specs import is_shell_harness, rollout_spec
from shellbox.machine import NetworkPolicy
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from skyrl_train.task_inputs import TaskInput, TrajectoryMetadata

SHELL_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.SHELL_SIMULATOR, capabilities=("shell", "filesystem")
)
NATIVE_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.LINUX_PROCESS,
    docker_image="fixture@sha256:" + "0" * 64,
)


def text_task(task_id: str = "task", *, prompt="Question", source="fixture") -> TaskSpec:
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="Projection fixture"),
        source=Source(dataset=source, revision="1", row="0", importer_revision="1"),
    )


def task_input(
    lowered: TaskRolloutSpec,
    *,
    metric_class=None,
    teacher_route=None,
    data_source=None,
    group_grader=None,
    metric_tags=(),
) -> TaskInput:
    return TaskInput(
        spec=lowered,
        metadata=TrajectoryMetadata(
            uid=lowered.task.id,
            metric_class=metric_class or ("taskcompendium" if is_shell_harness(lowered.harness) else lowered.harness),
            data_source=lowered.task.source.dataset if data_source is None else data_source,
            teacher_route=teacher_route,
            metric_tags=metric_tags,
        ),
        group_grader=group_grader,
    )


def machine_runtime(backend: str) -> MachineRuntimeSpec:
    return MachineRuntimeSpec(
        backend=backend,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=5,
        cleanup_timeout=5,
    )


def session_spec(
    session_name: str = "shellbox",
    *,
    max_turns: int = 2,
    **limits,
) -> dict:
    settings = dict(
        task_session=session_name,
        max_turns=max_turns,
        model_turn_timeout=5,
        command_timeout=1,
        tool_turn_timeout=5,
        total_turn_timeout=None,
        attempt_timeout=None,
        verifier_timeout=5,
        cleanup_timeout=5,
        max_total_tokens=None,
    )
    settings.update(limits)
    return settings


def session_limits(*, max_turns: int = 2, **limits) -> dict:
    return {key: value for key, value in session_spec(max_turns=max_turns, **limits).items() if key != "task_session"}


def lowered_task(
    task: TaskSpec,
    session_name: str = "shellbox",
    *,
    max_turns: int = 2,
    backend: str | None = None,
    verifier_backend: str | None = None,
    **limits,
) -> TaskRolloutSpec:
    return rollout_spec(
        task,
        session_spec(session_name, max_turns=max_turns, **limits),
        task_machine=None if backend is None else machine_runtime(backend),
        verifier_machine=None if verifier_backend is None else machine_runtime(verifier_backend),
    )
