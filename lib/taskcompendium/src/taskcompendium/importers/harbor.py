# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Harbor task directories into portable tasks with private shell grading."""

from pathlib import Path

from harbor_config.models.task.config import (
    ArtifactConfig,
    EnvironmentConfig,
    HealthcheckConfig,
    StepConfig,
    TaskConfig,
    TaskOS,
    VerifierEnvironmentMode,
)

from taskcompendium.environment import (
    ArtifactKind,
    DockerBuild,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    FileReward,
    HealthcheckSpec,
    MissingArtifactPolicy,
    RegistryImage,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    Source,
    StageRewardStrategy,
    StageVerifierSpec,
    TaskSpec,
    TaskStage,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)

ARTIFACTS_PATH = "/logs/artifacts"
REWARD_PATH = "/logs/verifier"
AGENT_LOG_PATH = "/logs/agent"
GRADER_DIRECTORY = "/tests"
GRADER_PATH = f"{GRADER_DIRECTORY}/test.sh"


def _directory_files(directory: Path, target: str) -> tuple[EnvironmentFile, ...]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Task files cannot contain symlinks: {path}")
        if path.is_file():
            files.append(
                EnvironmentFile(
                    path=f"{target.rstrip('/')}/{path.relative_to(directory).as_posix()}",
                    content=path.read_bytes(),
                    mode=path.stat().st_mode & 0o777,
                )
            )
    return tuple(files)


def _environment(config: EnvironmentConfig, context: Path, files: tuple[EnvironmentFile, ...]) -> EnvironmentSpec:
    unsupported = {
        key: value
        for key, value in config.model_dump().items()
        if key in {"gpu_types", "tpu", "mcp_servers", "skills_dir"} and value
    }
    if config.os != TaskOS.LINUX or unsupported:
        raise ValueError(f"Unsupported Harbor machine configuration: os={config.os}, fields={sorted(unsupported)}")
    image = (
        RegistryImage(reference=config.docker_image)
        if config.docker_image
        else DockerBuild(files=_directory_files(context, "/"))
    )
    return EnvironmentSpec(
        kind=EnvironmentKind.DOCKER,
        image=image,
        workdir=config.workdir or "",
        env=config.env,
        memory_mb=config.memory_mb,
        cpus=config.cpus,
        storage_mb=config.storage_mb,
        gpus=config.gpus or 0,
        network=config.allow_internet,
        startup_timeout=config.build_timeout_sec,
        files=files,
        healthcheck=None if config.healthcheck is None else _healthcheck(config.healthcheck, None),
        setup=(
            EnvironmentCommand(
                argv=("mkdir", "-p", ARTIFACTS_PATH, REWARD_PATH, AGENT_LOG_PATH),
                timeout=config.build_timeout_sec,
                user="0",
            ),
            EnvironmentCommand(
                argv=("chmod", "777", ARTIFACTS_PATH, REWARD_PATH, AGENT_LOG_PATH),
                timeout=config.build_timeout_sec,
                user="0",
            ),
        ),
    )


def _healthcheck(config: HealthcheckConfig, user: str | None) -> HealthcheckSpec:
    return HealthcheckSpec(
        command=EnvironmentCommand(argv=("sh", "-c", config.command), timeout=config.timeout_sec, user=user),
        interval=config.interval_sec,
        start_period=config.start_period_sec,
        start_interval=config.start_interval_sec,
        retries=config.retries,
    )


def _shell_verifier(config: TaskConfig, directory: Path, tests: Path) -> VerifierSpec:
    setup_files = _directory_files(directory / "setup_files", "/setup_files")
    verifier_environment = None
    private_by_path = {file.path: file for file in _directory_files(directory / "tests", GRADER_DIRECTORY)}
    private_by_path.update({file.path: file for file in _directory_files(tests, GRADER_DIRECTORY)})
    private_files = tuple(private_by_path.values())
    separate = (
        config.verifier.environment_mode == VerifierEnvironmentMode.SEPARATE or config.verifier.environment is not None
    )
    if separate:
        verifier_environment = _environment(
            config.verifier.environment or config.environment,
            tests,
            setup_files,
        )
        # A separate Harbor verifier image owns /tests, including its entrypoint.
        private_files = ()
    elif GRADER_PATH not in private_by_path:
        raise ValueError("A shared Harbor verifier requires tests/test.sh")
    artifact_configs = [ArtifactConfig(source=item) if isinstance(item, str) else item for item in config.artifacts]
    if not any(artifact.source.rstrip("/") == ARTIFACTS_PATH for artifact in artifact_configs):
        artifact_configs.append(ArtifactConfig(source=ARTIFACTS_PATH))
    verifier = ShellVerifierSpec(
        argv=("bash", GRADER_PATH),
        files=private_files,
        timeout=config.verifier.timeout_sec,
        env=config.verifier.env,
        user=None if config.verifier.user is None else str(config.verifier.user),
        environment=verifier_environment,
        collect=tuple(
            EnvironmentCommand(
                argv=("sh", "-c", hook.command),
                timeout=hook.timeout_sec,
                user=None if hook.user is None else str(hook.user),
            )
            for hook in config.verifier.collect
        ),
        artifacts=(
            tuple(
                VerifierArtifact(
                    source=artifact.source,
                    target=artifact.source,
                    kind=ArtifactKind.AUTO,
                    exclude=tuple(artifact.exclude),
                    missing=MissingArtifactPolicy.SKIP,
                )
                for artifact in artifact_configs
            )
            if separate
            else ()
        ),
        reward=FileReward(
            pass_above=0,
            files=(
                RewardFile(path=f"{REWARD_PATH}/reward.json", format=RewardFileFormat.JSON),
                RewardFile(path=f"{REWARD_PATH}/reward.txt", format=RewardFileFormat.NUMBER),
            ),
        ),
    )
    return VerifierSpec(kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json())


def _stage_config(config: TaskConfig, step: StepConfig) -> TaskConfig:
    mode = (
        step.verifier.environment_mode
        or (VerifierEnvironmentMode.SEPARATE if step.verifier.environment is not None else None)
        or config.verifier.environment_mode
        or (
            VerifierEnvironmentMode.SEPARATE
            if config.verifier.environment is not None
            else VerifierEnvironmentMode.SHARED
        )
    )
    environment = (
        step.verifier.environment or config.verifier.environment or config.environment
        if mode == VerifierEnvironmentMode.SEPARATE
        else None
    )
    verifier = step.verifier.model_copy(
        update={
            "environment_mode": mode,
            "environment": environment,
            "env": {**config.verifier.env, **step.verifier.env},
            "user": config.verifier.user if step.verifier.user is None else step.verifier.user,
            "collect": [*config.verifier.collect, *step.verifier.collect],
        }
    )
    agent = step.agent.model_copy(
        update={
            "timeout_sec": config.agent.timeout_sec if step.agent.timeout_sec is None else step.agent.timeout_sec,
            "user": config.agent.user if step.agent.user is None else step.agent.user,
        }
    )
    return config.model_copy(
        update={"agent": agent, "verifier": verifier, "artifacts": [*config.artifacts, *step.artifacts]}
    )


def _conversation(path: Path) -> ConversationInput:
    return ConversationInput(events=(TextMessage(role="user", content=path.read_text()),))


def harbor_task(directory: Path, *, source: Source, verifier_override: VerifierSpec | None = None) -> TaskSpec:
    """Package Harbor tasks without exposing their test files to the model.

    Environment variables retain their templates until execution. The converter
    rejects task features that the execution schema cannot yet represent.
    A verifier override replaces private grading for every stage and permits
    task packages without shell grader files.
    """
    config = TaskConfig.model_validate_toml((directory / "task.toml").read_text())
    environment = _environment(
        config.environment, directory / "environment", _directory_files(directory / "setup_files", "/setup_files")
    )
    stages = []
    for index, step in enumerate(config.steps or []):
        if Path(step.name).name != step.name or step.name in {".", ".."}:
            raise ValueError("Harbor step names must be path components")
        effective = _stage_config(config, step)
        step_dir = directory / "steps" / step.name
        user = None if effective.agent.user is None else str(effective.agent.user)
        tests = step_dir / "tests" if (step_dir / "tests").is_dir() else directory / "tests"
        stages.append(
            TaskStage(
                name=step.name,
                context=None if index == 0 else _conversation(step_dir / "instruction.md"),
                verifier=verifier_override or _shell_verifier(effective, directory, tests),
                workdir_files=_directory_files(step_dir / "workdir", "/"),
                setup=(
                    (
                        EnvironmentCommand(
                            argv=("bash", "setup.sh"), timeout=config.environment.build_timeout_sec, user=user
                        ),
                    )
                    if (step_dir / "workdir/setup.sh").is_file()
                    else ()
                ),
                healthcheck=None if step.healthcheck is None else _healthcheck(step.healthcheck, user),
                agent_timeout=effective.agent.timeout_sec,
                agent_user=user,
                minimum_rewards=(
                    step.min_reward
                    if isinstance(step.min_reward, dict)
                    else {} if step.min_reward is None else {"reward": step.min_reward}
                ),
            )
        )
    context_path = directory / "steps" / stages[0].name / "instruction.md" if stages else directory / "instruction.md"
    verifier = (
        VerifierSpec(
            kind=VerifierKind.STAGED,
            parameters_json=StageVerifierSpec(
                strategy=StageRewardStrategy(config.multi_step_reward_strategy or StageRewardStrategy.MEAN)
            ).model_dump_json(),
        )
        if stages
        else verifier_override or _shell_verifier(config, directory, directory / "tests")
    )
    return TaskSpec(
        id=f"{source.dataset}:{source.row}",
        context=_conversation(context_path),
        environment_requirements=EnvironmentRequirements(capabilities=("shell", "filesystem")),
        answer_type=AnswerType.STATE,
        environment=environment,
        agent_timeout=config.agent.timeout_sec,
        agent_user=None if config.agent.user is None else str(config.agent.user),
        verifier=verifier,
        stages=tuple(stages),
        source=source,
        metadata={"harbor": config.metadata},
    )
