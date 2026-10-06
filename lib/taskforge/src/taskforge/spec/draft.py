# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a TaskCompendium ``TaskSpec`` from builder outputs.

These helpers own the serialized details a builder should not repeat: verifier
``parameters_json``, stage contexts, and the capability requirements of
executable environments. ``assemble`` adds the checks TaskSpec itself does not
make and returns a spec that survives a JSON round trip.

A task's execution settings are not part of the ``TaskSpec`` (TaskCompendium
0.24): deadlines, the agent user and each stage's working files, setup and
healthcheck live in a ``TaskExecution`` that travels beside the task.
``assemble`` checks the ``TaskExecution`` against the task; the builder returns
both (``BuildOutput.execution``), and validation passes both to RolloutEngine.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue
from taskcompendium.environment import (
    DockerBuild,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    ExitCodeReward,
    FileReward,
    HealthcheckSpec,
    RegistryImage,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    StdoutReward,
    VerifierArtifact,
)
from taskcompendium.execution import TaskExecution
from taskcompendium.grading import validate_verifier, verifier_descriptor
from taskcompendium.models import (
    FILESYSTEM_CAPABILITY,
    SHELL_CAPABILITY,
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    Source,
    StageRewardStrategy,
    StageVerifierSpec,
    TaskSpec,
    TaskStage,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from verifyit.candidate import CandidateSpec

type Reward = StdoutReward | ExitCodeReward | FileReward

TEXT_ANSWER_KINDS = frozenset({VerifierKind.EXACT_ANSWER, VerifierKind.NUMERIC_ANSWER, VerifierKind.MCQ_ANSWER})
TEXT_ANSWER_TYPES = frozenset({AnswerType.TEXT, AnswerType.NUMBER})
MACHINE_ANSWER_TYPES = frozenset({AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE})
DEFAULT_REWARD = "reward"
"""The one reward component every graded result reports."""


@dataclass(frozen=True)
class Resources:
    """Machine limits. ``None`` leaves the limit to the machine factory."""

    memory_mb: int | None = None
    cpus: int | None = None
    storage_mb: int | None = None
    gpus: int = 0


def file(path: str, content: str, mode: int = 0o644) -> EnvironmentFile:
    """A UTF-8 text file at an absolute machine path."""
    return EnvironmentFile(path=path, content=content.encode(), mode=mode)


def shell_command(script: str, timeout: float, user: str | None = None, cwd: str | None = None) -> EnvironmentCommand:
    """Run ``script`` with ``sh -c``."""
    return EnvironmentCommand(argv=("sh", "-c", script), timeout=timeout, user=user, cwd=cwd)


def environment(
    kind: EnvironmentKind,
    image: RegistryImage | DockerBuild | None = None,
    files: Sequence[EnvironmentFile] = (),
    setup: Sequence[EnvironmentCommand] = (),
    healthcheck: HealthcheckSpec | None = None,
    workdir: str = "/workspace",
    network: bool = False,
    resources: Resources | None = None,
    env: Mapping[str, str] | None = None,
    startup_timeout: float | None = None,
) -> EnvironmentSpec:
    """The machine the agent works in; everything here is visible to the agent."""
    resources = resources or Resources()
    return EnvironmentSpec(
        kind=kind,
        image=image,
        workdir=workdir,
        files=tuple(files),
        env=dict(env or {}),
        setup=tuple(setup),
        healthcheck=healthcheck,
        startup_timeout=startup_timeout,
        network=network,
        memory_mb=resources.memory_mb,
        cpus=resources.cpus,
        storage_mb=resources.storage_mb,
        gpus=resources.gpus,
    )


def reward_file(
    path: str, file_format: RewardFileFormat, key: str = "reward", pass_above: float | None = None
) -> FileReward:
    """Read the score from one reward file the grader writes."""
    return FileReward(files=(RewardFile(path=path, format=file_format, key=key),), pass_above=pass_above)


def shell_verifier(
    argv: Sequence[str],
    reward: Reward,
    timeout: float,
    files: Sequence[EnvironmentFile] = (),
    user: str | None = None,
    grading_environment: EnvironmentSpec | None = None,
    collect: Sequence[EnvironmentCommand] = (),
    artifacts: Sequence[VerifierArtifact] = (),
    env: Mapping[str, str] | None = None,
) -> VerifierSpec:
    """A task-specific grader script.

    ``files`` are private: the engine installs them only after the final model
    response. With ``grading_environment`` the script runs in a fresh machine
    that receives ``artifacts`` copied from the agent's machine.
    """
    parameters = ShellVerifierSpec(
        argv=tuple(argv),
        timeout=timeout,
        env=dict(env or {}),
        user=user,
        reward=reward,
        collect=tuple(collect),
        artifacts=tuple(artifacts),
    )
    return VerifierSpec(
        kind=VerifierKind.SHELL,
        parameters_json=parameters.model_dump_json(),
        files=tuple(files),
        environment=grading_environment,
    )


def answer_verifier(spec: CandidateSpec) -> VerifierSpec:
    """A generic verifyit answer grader; the build SDK's name for ``verifier_descriptor``.

    Only verifyit's candidate modes (exact, numeric, mcq, predicted_action)
    grade a final answer without a private runtime; RolloutEngine refuses the
    other modes on a task without an application-supplied session.
    """
    return verifier_descriptor(spec)


def emits_reward_components(verifier: VerifierSpec) -> bool:
    """Whether RolloutEngine can report named reward components for ``verifier``.

    Only a shell grader whose reward files are all JSON reports one component
    per numeric key; every other grader reports just ``reward``.
    """
    if verifier.kind != VerifierKind.SHELL:
        return False
    reward = ShellVerifierSpec.model_validate_json(verifier.parameters_json).reward
    return isinstance(reward, FileReward) and all(item.format == RewardFileFormat.JSON for item in reward.files)


def staged(strategy: StageRewardStrategy) -> VerifierSpec:
    """The task-level verifier of a staged task: reduce stage grades with ``strategy``."""
    return VerifierSpec(kind=VerifierKind.STAGED, parameters_json=StageVerifierSpec(strategy=strategy).model_dump_json())


def stage(
    name: str,
    verifier: VerifierSpec,
    instruction: str | None = None,
    minimum_rewards: Mapping[str, float] | None = None,
) -> TaskStage:
    """One stage on the shared task machine.

    The first stage uses the task instruction and takes no ``instruction``; every
    later stage needs one. The task stops after a stage whose rewards fall below
    any of its ``minimum_rewards``. The stage's working files, setup, healthcheck,
    deadline and user are a ``StageExecution`` in ``TaskExecution.stages[name]``.
    """
    return TaskStage(
        name=name,
        context=None if instruction is None else _user_context(instruction),
        verifier=verifier,
        minimum_rewards=dict(minimum_rewards or {}),
    )


def assemble(
    task_id: str,
    instruction: str,
    answer_type: AnswerType,
    environment: EnvironmentSpec,
    verifier: VerifierSpec,
    source: Source,
    *,
    execution: TaskExecution,
    system: str | None = None,
    stages: Sequence[TaskStage] = (),
    final_tools: Sequence[FunctionDefinition] = (),
    metadata: Mapping[str, JsonValue] | None = None,
    tags: Sequence[str] = (),
) -> TaskSpec:
    """Build a TaskSpec and reject one that RolloutEngine could not grade as intended.

    ``execution`` holds the deadlines, agent user and stage preparation the task
    runs with. It is checked here but not stored in the TaskSpec: keep it beside
    the task and pass both to RolloutEngine.

    Raises:
        ValueError: TaskSpec validation failed, ``execution`` does not name
            exactly the task's stages, a grader does not fit the answer type or
            environment, a private grader file is agent-visible, or a stage gates
            on a reward component its grader never reports.
    """
    executable = environment.kind != EnvironmentKind.NULL
    events = (
        *(() if system is None else (TextMessage(role="system", content=system),)),
        TextMessage(role="user", content=instruction),
    )
    spec = TaskSpec(
        id=task_id,
        context=ConversationInput(events=events),
        environment_requirements=EnvironmentRequirements(
            capabilities=(SHELL_CAPABILITY, FILESYSTEM_CAPABILITY) if executable else ()
        ),
        final_tools=tuple(final_tools),
        answer_type=answer_type,
        verifier=verifier,
        environment=environment,
        stages=tuple(stages),
        source=source,
        metadata=dict(metadata or {}),
        tags=tuple(tags),
    )
    check_execution(spec, execution)
    stage_graders = tuple(item.verifier for item in spec.stages)
    for grader in (spec.verifier, *stage_graders):
        validate_verifier(grader)
    for grader in stage_graders or (spec.verifier,):
        _check_grader(spec, execution, grader)
    for item in spec.stages:
        named = sorted(set(item.minimum_rewards) - {DEFAULT_REWARD})
        if named and not emits_reward_components(item.verifier):
            raise ValueError(f"Stage {item.name!r} gates on {named}, but its grader reports only {DEFAULT_REWARD!r}")
    restored = TaskSpec.model_validate_json(spec.model_dump_json())
    if restored != spec:
        raise ValueError(f"Task {task_id!r} does not survive a JSON round trip")
    return restored


def check_execution(task: TaskSpec, execution: TaskExecution) -> None:
    """Raise ``ValueError`` unless ``execution`` prepares exactly the stages ``task`` has."""
    names = [item.name for item in task.stages]
    if sorted(execution.stages) != sorted(names):
        raise ValueError(f"Execution stages {sorted(execution.stages)} do not match the task's stages {sorted(names)}")


def _user_context(instruction: str) -> ConversationInput:
    return ConversationInput(events=(TextMessage(role="user", content=instruction),))


def _agent_visible_files(spec: TaskSpec, execution: TaskExecution) -> tuple[EnvironmentFile, ...]:
    environment = spec.environment
    build = environment.image.files if isinstance(environment.image, DockerBuild) else ()
    return (*environment.files, *build, *(item for stage in execution.stages.values() for item in stage.workdir_files))


def _check_grader(spec: TaskSpec, execution: TaskExecution, grader: VerifierSpec) -> None:
    if grader.kind == VerifierKind.SKIPPED:
        return
    if grader.kind == VerifierKind.SHELL:
        _check_shell_grader(spec, execution, grader)
        return
    if spec.answer_type in MACHINE_ANSWER_TYPES:
        raise ValueError(f"A {spec.answer_type.value!r} answer requires a shell verifier, not {grader.kind!r}")
    if grader.kind == VerifierKind.PREDICTED_ACTION and spec.answer_type != AnswerType.NATIVE_ACTION:
        raise ValueError("A predicted_action verifier requires a native_action answer")
    if grader.kind in TEXT_ANSWER_KINDS and spec.answer_type not in TEXT_ANSWER_TYPES:
        raise ValueError(f"A {grader.kind!r} verifier requires a text or number answer")


def _check_shell_grader(spec: TaskSpec, execution: TaskExecution, grader: VerifierSpec) -> None:
    environment = spec.environment
    if environment.kind == EnvironmentKind.NULL:
        raise ValueError("A shell verifier requires an executable task environment")
    visible = {item.content for item in _agent_visible_files(spec, execution)}
    leaked = sorted(item.path for item in grader.files if item.content in visible)
    if leaked:
        raise ValueError(f"Private verifier file content is also agent-visible: {leaked}")
    if grader.environment is None:
        shadowed = sorted({item.path for item in grader.files} & {item.path for item in environment.files})
        if shadowed:
            raise ValueError(f"Private verifier files overwrite agent environment files: {shadowed}")
