# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower one Harbor task package to a TaskSpec and its per-task episode inputs.

A package directory holds ``instruction.md``, ``task.toml``, ``environment/``, ``tests/``, and
optionally ``solution/`` and ``setup_files/``. The instruction becomes the user message;
``environment/`` becomes a digest-pinned image, a build context with pinned bases, or a primary
environment with Compose sidecars. ``tests/`` are verifier resources at ``/tests``,
``solution/`` oracle resources at ``/solution``, and ``setup_files/`` worker resources at
``/setup_files``. A dataset's grading rule decides how attempts are graded.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
from shellbox.machine import NetworkPolicy

from taskcompendium.evaluation.spec import EpisodeInputs, MachineResources
from taskcompendium.importers.harbor.compose import (
    ComposeGroup,
    lower_sidecars,
    mcp_providers,
    read_compose,
    sidecar_source,
)
from taskcompendium.importers.harbor.config import (
    ArtifactConfig,
    EnvironmentConfig,
    HarborTaskConfig,
    Healthcheck,
    UnsupportedHarborFeature,
    VerifierMode,
    read_task_config,
)
from taskcompendium.importers.harbor.sources import (
    DOCKERFILE,
    BuildSource,
    ContainerSource,
    ImageSource,
    LoweredSource,
    Pin,
    directory_resources,
    extend_source,
    lower_source,
    pinner,
    source_references,
)
from taskcompendium.models import (
    AgentMachine,
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    Grader,
    GraderMachine,
    GraderPlacement,
    MissingArtifactPolicy,
    PlainText,
    ReadinessRequirement,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
    VerifierCommand,
    VerifyitGrader,
    validate_output_paths,
)

INSTRUCTION = "instruction.md"
ENVIRONMENT_DIR = "environment"
TESTS_DIR = "tests"
SOLUTION_DIR = "solution"
SETUP_FILES_DIR = "setup_files"
TEST_SCRIPT = "/tests/test.sh"
ARTIFACTS_DIR = "/logs/artifacts"
AGENT_LOGS_DIR = "/logs/agent"
AGENT_CAPABILITIES = ("shell", "filesystem", "process")
LOG_DIRECTORIES = f"mkdir -p {AGENT_LOGS_DIR} {ARTIFACTS_DIR} && chmod 777 {AGENT_LOGS_DIR} {ARTIFACTS_DIR}"
PACKAGE_REWARD = FileReward(
    files=(
        RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON_METRICS),
        RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),
    )
)
"""Harbor's reward files: ``reward.json`` first, with its other numeric keys as detail, then ``reward.txt``."""
METADATA_LABELS = ("category", "difficulty")
# Minimal task images may provide only sh. The script is passed once, as "$1", so that a long script does not
# appear twice in the remote argv.
HARBOR_SHELL_PREFIX = (
    "/bin/sh",
    "-c",
    'if [ -x /bin/bash ]; then exec /bin/bash -c "$1"; else exec /bin/sh -c "$1"; fi',
    "harbor-command",
)
"""The argv before a command that Harbor runs through Bash when the image has it, else ``sh``."""


def harbor_shell(command: str) -> tuple[str, ...]:
    return (*HARBOR_SHELL_PREFIX, command)


@dataclass(frozen=True)
class PackageTests:
    """Grade with the package's ``tests/test.sh`` and the reward file it writes, as Harbor does."""

    kind: Literal["package_tests"] = "package_tests"


@dataclass(frozen=True)
class AnswerFile:
    """Capture the file at ``path`` and grade its text on the host with ``grader``.

    Use it where ``test.sh`` only compares a string or calls a fixed judge.
    """

    path: str
    grader: VerifyitGrader
    kind: Literal["answer_file"] = "answer_file"

    def __post_init__(self) -> None:
        validate_output_paths((self.path,))
        if self.grader.placement is not None:
            raise ValueError("An answer file is graded on the host, so its grader declares no placement")


GradingRule = Annotated[PackageTests | AnswerFile, Field(discriminator="kind")]


@dataclass(frozen=True)
class JobSettings:
    """What a Harbor job sets for every package it runs.

    ``agent_timeout``, ``verifier_timeout`` and ``storage_mb`` replace the package's own values, as the
    job's ``override_timeout_sec`` and ``override_storage_mb`` do; ``None`` keeps them. ``agent_install``
    holds Dockerfile instructions, built after the package's own, that install an agent the package's
    image lacks, which Harbor would install at setup.

    A deployment that supplies ``agent_user`` or ``setup_timeout`` must match the corresponding
    package declarations. The runtime enforces the user and bounds startup, including setup.
    """

    agent_timeout: float | None = None
    verifier_timeout: float | None = None
    storage_mb: int | None = None
    agent_install: str | None = None
    agent_user: str | None = None
    setup_timeout: float | None = None


def _job_config(config: HarborTaskConfig, job: JobSettings) -> HarborTaskConfig:
    """The package's configuration with the job's replacements."""
    agent, verifier, environment = config.agent, config.verifier, config.environment
    if job.agent_timeout is not None:
        agent = agent.model_copy(update={"timeout_sec": job.agent_timeout})
    if job.verifier_timeout is not None:
        verifier = verifier.model_copy(update={"timeout_sec": job.verifier_timeout})
    if job.storage_mb is not None:
        environment = environment.model_copy(update={"storage_mb": job.storage_mb})
    return config.model_copy(update={"agent": agent, "verifier": verifier, "environment": environment})


@dataclass(frozen=True)
class HarborTask:
    task: TaskSpec
    episode: EpisodeInputs


@dataclass(frozen=True)
class _PrimarySources:
    main: ContainerSource
    compose: ComposeGroup | None


def _primary_sources(config: HarborTaskConfig, directory: Path) -> _PrimarySources:
    environment_dir = directory / ENVIRONMENT_DIR
    return _PrimarySources(
        _container_source(config.environment, environment_dir, ENVIRONMENT_DIR), read_compose(environment_dir)
    )


def _container_source(environment: EnvironmentConfig, build_context: Path, section: str) -> ContainerSource:
    """Harbor's precedence: ``docker_image`` wins over the build context's Dockerfile."""
    if environment.docker_image:
        return ImageSource(environment.docker_image)
    if (build_context / DOCKERFILE).is_file():
        return BuildSource(build_context)
    raise UnsupportedHarborFeature(f"neither [{section}].docker_image nor {build_context.name}/{DOCKERFILE} exists")


def _verifier_source(config: HarborTaskConfig, directory: Path) -> ContainerSource | None:
    """The separate verifier's container; Harbor builds it from ``tests/``, never ``environment/``."""
    if config.verifier.mode != VerifierMode.SEPARATE:
        return None
    return _container_source(config.verifier_environment, directory / TESTS_DIR, "verifier.environment")


def image_references(directory: Path) -> tuple[str, ...]:
    """Every external image the package's machines need, so a recipe can resolve digests once."""
    config = read_task_config(directory)
    primary = _primary_sources(config, directory)
    sources = [primary.main]
    if primary.compose is not None:
        sources.extend(sidecar_source(primary.compose, name) for name in primary.compose.sidecars)
    verifier = _verifier_source(config, directory)
    if verifier is not None:
        sources.append(verifier)
    return tuple(dict.fromkeys(reference for source in sources for reference in source_references(source)))


def _readiness(healthcheck: Healthcheck | None) -> ReadinessRequirement | None:
    """Harbor polls during the start period; the requirement waits it out, then allows ``retries`` failures."""
    if healthcheck is None:
        return None
    return ReadinessRequirement(
        argv=harbor_shell(healthcheck.command),
        command_timeout=healthcheck.timeout_sec,
        timeout=healthcheck.start_period_sec
        + healthcheck.retries * (healthcheck.timeout_sec + healthcheck.interval_sec),
        interval=healthcheck.interval_sec,
        start_period=healthcheck.start_period_sec,
        attempts=healthcheck.retries,
    )


def _network_capabilities(allow_internet: bool) -> tuple[str, ...]:
    return ("network",) if allow_internet else ()


def _agent_internet(config: HarborTaskConfig, sources: _PrimarySources) -> bool:
    """``allow_internet``, unless the Compose file takes ``main`` off the network."""
    return config.environment.allow_internet and (sources.compose is None or not sources.compose.main_network_disabled)


def _environment_files(config: HarborTaskConfig, directory: Path, sources: _PrimarySources) -> tuple[TaskResource, ...]:
    """Harbor copies ``environment/`` into the working directory of a prebuilt image without build inputs."""
    environment_dir = directory / ENVIRONMENT_DIR
    if not isinstance(sources.main, ImageSource) or sources.compose is not None:
        return ()
    if (environment_dir / DOCKERFILE).exists():
        return ()
    files = directory_resources(environment_dir)
    if not files:
        return ()
    workdir = config.environment.workdir
    if workdir is None:
        raise UnsupportedHarborFeature("environment/ files for a prebuilt image require [environment].workdir")
    prefix = workdir.strip("/")
    return tuple(file.model_copy(update={"path": f"{prefix}/{file.path}" if prefix else file.path}) for file in files)


def _agent_environment(
    config: HarborTaskConfig, main: LoweredSource, sources: _PrimarySources, pin: Pin
) -> EnvironmentRequirements:
    environment = config.environment
    services = () if sources.compose is None else lower_sidecars(sources.compose, pin)
    if environment.mcp_servers and sources.compose is None:
        raise UnsupportedHarborFeature("MCP servers outside a Compose service group")
    compose_environment = {} if sources.compose is None else sources.compose.main_environment
    compose_workdir = None if sources.compose is None else sources.compose.main_working_directory
    return EnvironmentRequirements(
        capabilities=AGENT_CAPABILITIES + _network_capabilities(_agent_internet(config, sources)),
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=main.docker_image,
        docker_build=main.docker_build,
        working_directory=environment.workdir or compose_workdir or main.workdir,
        setup_commands=(LOG_DIRECTORIES,) + ((environment.setup.command,) if environment.setup else ()),
        environment_variables={**environment.env, **compose_environment, **config.agent.env},
        tool_providers=mcp_providers(environment.mcp_servers, services),
        services=services,
        readiness=_readiness(environment.healthcheck),
    )


def _verifier_environment(config: HarborTaskConfig, directory: Path, pin: Pin) -> EnvironmentRequirements | None:
    source = _verifier_source(config, directory)
    if source is None:
        return None
    environment = config.verifier_environment
    lowered = lower_source(source, pin)
    return EnvironmentRequirements(
        capabilities=_network_capabilities(environment.allow_internet),
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=lowered.docker_image,
        docker_build=lowered.docker_build,
        working_directory=environment.workdir or lowered.workdir,
        # The grading machine is the verifier's own, so its variables carry the verifier's env.
        environment_variables={**environment.env, **config.verifier.env},
    )


def _artifacts(config: HarborTaskConfig) -> tuple[VerifierArtifact, ...]:
    """Harbor's artifacts, always including ``/logs/artifacts``, copied to the same paths for the verifier."""
    artifacts = list(config.artifacts)
    if not any(artifact.source.rstrip("/") == ARTIFACTS_DIR for artifact in artifacts):
        artifacts.insert(0, ArtifactConfig(source=ARTIFACTS_DIR))
    return tuple(
        VerifierArtifact(
            source=artifact.source,
            # Harbor's destination names a host collection path, not a verifier path.
            target=artifact.source,
            kind=ArtifactKind.AUTO,
            exclude=artifact.exclude,
            missing=MissingArtifactPolicy.SKIP,
        )
        for artifact in artifacts
    )


def _shared_verifier_env(config: HarborTaskConfig) -> dict[str, str]:
    """``verifier.env`` for a test run in the agent's container, which offers no collect step and no host values."""
    if config.verifier.collect:
        raise UnsupportedHarborFeature("collect hooks for a verifier in the agent's container")
    host_values = sorted(name for name, value in config.verifier.env.items() if "${" in value)
    if host_values:
        raise UnsupportedHarborFeature(
            f"verifier variables {', '.join(host_values)} that read host values in the agent's container"
        )
    return dict(config.verifier.env)


def package_grader(
    config: HarborTaskConfig,
    rule: GradingRule,
    agent: EnvironmentRequirements,
    verifier: EnvironmentRequirements | None,
) -> Grader:
    """The task's grader, placed where Harbor would run its verifier.

    ``verifier`` is the separate verifier's machine, or ``None`` when Harbor grades in the
    agent's container after the agent stops (``shared``, Harbor's default). Harbor runs the
    tests in the container's working directory; when the package names none, that is the
    image's ``WORKDIR``, so the grader runs in the machine's working directory.
    """
    if isinstance(rule, AnswerFile):
        return rule.grader
    if verifier is None:
        placement: GraderPlacement = AgentMachine()
        cwd, env = agent.working_directory, _shared_verifier_env(config)
    else:
        placement = GraderMachine(
            environment=verifier,
            collect=tuple(
                VerifierCommand(argv=harbor_shell(hook.command), timeout=hook.timeout_sec)
                for hook in config.verifier.collect
            ),
            artifacts=_artifacts(config),
        )
        cwd, env = verifier.working_directory, {}
    return ScriptGrader(
        argv=("bash", TEST_SCRIPT),
        cwd=cwd,
        env=env,
        placement=placement,
        answer_path=None,
        reward=PACKAGE_REWARD,
        timeout=config.verifier.timeout_sec,
    )


def _labels(config: HarborTaskConfig, directory: Path) -> dict[str, str]:
    labels = {name: str(config.metadata[name]) for name in METADATA_LABELS if name in config.metadata}
    tags = config.metadata.get("tags")
    if tags:
        labels["tags"] = tags if isinstance(tags, str) else ",".join(str(tag) for tag in tags)
    return {**labels, "harbor_path": directory.name}


def lower_package(
    directory: Path,
    *,
    task_id: str,
    source: Source,
    rule: GradingRule,
    image_digests: Mapping[str, str],
    job: JobSettings,
) -> HarborTask:
    """Lower a package as ``job`` runs it; ``image_digests`` maps each :func:`image_references` entry to a digest.

    Raises ``UnsupportedHarborFeature`` for packages the TaskSpec cannot represent faithfully, and
    ``ValueError`` when an image reference has no resolved digest.
    """
    config = _job_config(read_task_config(directory, agent_user=job.agent_user, setup_timeout=job.setup_timeout), job)
    pin = pinner(image_digests)
    sources = _primary_sources(config, directory)
    main = lower_source(sources.main, pin)
    if job.agent_install is not None:
        main = extend_source(main, job.agent_install)
    agent = _agent_environment(config, main, sources, pin)
    verifier = _verifier_environment(config, directory, pin)
    grader = package_grader(config, rule, agent, verifier)
    task = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=(directory / INSTRUCTION).read_text()),)),
        environment_requirements=agent,
        output_paths=(rule.path,) if isinstance(rule, AnswerFile) else (),
        answer_type=AnswerType.FILE if isinstance(rule, AnswerFile) else AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=grader,
        source=source,
        resources=ResourceGroups(
            worker=(
                *directory_resources(directory / SETUP_FILES_DIR, f"{SETUP_FILES_DIR}/"),
                *_environment_files(config, directory, sources),
            ),
            oracle=directory_resources(directory / SOLUTION_DIR, f"{SOLUTION_DIR}/"),
            verifier=directory_resources(directory / TESTS_DIR),
        ),
        labels=_labels(config, directory),
    )
    environment = config.environment
    episode = EpisodeInputs(
        agent_timeout=config.agent.timeout_sec,
        verifier_timeout=config.verifier.timeout_sec,
        cpus=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        network=NetworkPolicy.ALLOW if _agent_internet(config, sources) else NetworkPolicy.DENY,
        verifier_resources=(
            None
            if verifier is None
            else MachineResources(
                cpus=config.verifier_environment.cpus,
                memory_mb=config.verifier_environment.memory_mb,
                storage_mb=config.verifier_environment.storage_mb,
            )
        ),
    )
    return HarborTask(task, episode)
