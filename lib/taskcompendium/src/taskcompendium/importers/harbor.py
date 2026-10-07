# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert single-stage Harbor packages with prebuilt images to TaskSpec."""

from pathlib import Path

from harbor_config.models.task.config import (
    ArtifactConfig,
    EnvironmentConfig,
    TaskConfig,
    TaskOS,
    VerifierEnvironmentMode,
)

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.shell_verifier import (
    ArtifactKind,
    FileReward,
    MissingArtifactPolicy,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)

ARTIFACTS_PATH = "/logs/artifacts"
REWARD_PATH = "/logs/verifier"
AGENT_LOG_PATH = "/logs/agent"
GRADER_DIRECTORY = "/tests"
GRADER_PATH = f"{GRADER_DIRECTORY}/test.sh"


def _directory_resources(directory: Path, prefix: str = "") -> tuple[TaskResource, ...]:
    resources = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Task files cannot contain symlinks: {path}")
        if path.is_file():
            resources.append(
                inline_resource(f"{prefix}{path.relative_to(directory).as_posix()}", path.read_bytes()).model_copy(
                    update={"mode": f"{path.stat().st_mode & 0o777:03o}"}
                )
            )
    return tuple(resources)


def _environment(config: EnvironmentConfig, *, capabilities: tuple[str, ...] = ()) -> EnvironmentRequirements:
    unsupported = {
        key: value
        for key, value in config.model_dump().items()
        if key in {"gpu_types", "tpu", "mcp_servers", "skills_dir", "healthcheck"} and value
    }
    if config.os != TaskOS.LINUX or unsupported:
        raise NotImplementedError(f"Unsupported Harbor environment: os={config.os}, fields={sorted(unsupported)}")
    if not config.docker_image:
        raise NotImplementedError(
            "Harbor tasks require a prebuilt, digest-pinned image; Dockerfile tasks are unsupported"
        )
    return EnvironmentRequirements(
        capabilities=capabilities,
        docker_image=config.docker_image,
        working_directory=config.workdir,
        environment_variables=config.env,
    )


def harbor_task(directory: Path, *, source: Source) -> TaskSpec:
    """Preserve task semantics and keep verifier files outside the agent workspace.

    Shared verifier mode, multi-stage packages, image builds, and healthchecks are unsupported.
    Machine limits, users, and deadlines belong to runtime lowering.
    """
    config = TaskConfig.model_validate_toml((directory / "task.toml").read_text())
    if config.steps or config.multi_step_reward_strategy:
        raise NotImplementedError("Multi-stage Harbor tasks are unsupported")
    if config.verifier.environment_mode != VerifierEnvironmentMode.SEPARATE and config.verifier.environment is None:
        raise NotImplementedError("Harbor shell grading requires a separate verifier environment")
    requirements = _environment(config.environment, capabilities=("shell", "filesystem"))
    requirements = requirements.model_copy(
        update={
            "setup_commands": (
                f"mkdir -p {ARTIFACTS_PATH} {REWARD_PATH} {AGENT_LOG_PATH}",
                f"chmod 777 {ARTIFACTS_PATH} {REWARD_PATH} {AGENT_LOG_PATH}",
            )
        }
    )
    worker = _directory_resources(directory / "setup_files", "setup_files/")
    private = _directory_resources(directory / "tests")
    verifier_requirements = _environment(config.verifier.environment or config.environment).model_copy(
        update={
            "environment_variables": {
                **(config.verifier.environment or config.environment).env,
                **config.verifier.env,
            },
        }
    )
    if config.verifier.collect:
        raise NotImplementedError("Harbor collect hooks with per-hook users and deadlines are unsupported")
    artifacts = [ArtifactConfig(source=item) if isinstance(item, str) else item for item in config.artifacts]
    if not any(artifact.source.rstrip("/") == ARTIFACTS_PATH for artifact in artifacts):
        artifacts.append(ArtifactConfig(source=ARTIFACTS_PATH))
    verifier = ShellVerifierSpec(
        argv=("bash", GRADER_PATH),
        artifacts=tuple(
            VerifierArtifact(
                source=artifact.source,
                target=artifact.source,
                kind=ArtifactKind.AUTO,
                exclude=tuple(artifact.exclude),
                missing=MissingArtifactPolicy.SKIP,
            )
            for artifact in artifacts
        ),
        reward=FileReward(
            pass_above=0,
            files=(
                RewardFile(path=f"{REWARD_PATH}/reward.json", format=RewardFileFormat.JSON),
                RewardFile(path=f"{REWARD_PATH}/reward.txt", format=RewardFileFormat.NUMBER),
            ),
        ),
    )
    specification = VerifierSpec(
        kind="shell",
        parameters_json=verifier.model_dump_json(),
        environment_requirements=verifier_requirements,
    )
    return TaskSpec(
        id=f"{source.dataset}:{source.row}",
        context=ConversationInput(
            events=(TextMessage(role="user", content=(directory / "instruction.md").read_text()),)
        ),
        environment_requirements=requirements,
        answer_type=AnswerType.WORKSPACE_STATE,
        verifier=specification,
        resources=ResourceGroups(worker=worker, verifier=private),
        source=source,
        tags=("harbor",),
    )
