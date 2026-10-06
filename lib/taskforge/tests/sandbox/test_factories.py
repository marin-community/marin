# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.image import DockerfileSource
from shellbox.image import RegistryImage as ShellboxRegistryImage
from shellbox.machine import MachineSpec, NetworkPolicy, ShellSimBuiltins, UnsupportedMachineSpec
from taskcompendium.environment import (
    ArtifactKind,
    DockerBuild,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    RegistryImage,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.execution import StageExecution, TaskExecution
from taskcompendium.grading import numeric_answer
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
from taskcompendium.submission import PlainText

from taskforge.sandbox import factories
from taskforge.sandbox.factories import (
    IRIS_DOCKER,
    LOCAL_DOCKER,
    SHELLSIM,
    FactoryCapabilities,
    MachineHost,
    RefusalReason,
    factory_capabilities,
    machine_factories,
    task_refusals,
)
from taskforge.spec.draft import task_execution

IMAGE = RegistryImage(reference="registry.example/task@sha256:" + "0" * 64)
BUILD = DockerBuild(files=(EnvironmentFile(path="/Dockerfile", content=b"FROM busybox\n"),))


def task(environment: EnvironmentSpec, **update) -> TaskSpec:
    return TaskSpec(
        id="t",
        context=ConversationInput(events=(TextMessage(role="user", content="hi"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer("1", tolerance_abs=0, tolerance_rel=0),
        environment=environment,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    ).model_copy(update=update)


def reasons(spec: TaskSpec, capabilities, execution: TaskExecution = task_execution()) -> set[tuple[RefusalReason, str]]:
    return {(refusal.reason, refusal.where) for refusal in task_refusals(spec, execution, capabilities)}


IRIS = {EnvironmentKind.SHELLSIM: SHELLSIM, EnvironmentKind.DOCKER: IRIS_DOCKER}
# What the Iris backend accepts once its create works.
IRIS_WORKING = {EnvironmentKind.SHELLSIM: SHELLSIM, EnvironmentKind.DOCKER: replace(IRIS_DOCKER, unavailable=None)}
LAPTOP = {EnvironmentKind.SHELLSIM: SHELLSIM, EnvironmentKind.DOCKER: LOCAL_DOCKER}


def test_docker_tasks_are_refused_on_iris_today():
    spec = task(EnvironmentSpec(kind=EnvironmentKind.DOCKER, image=IMAGE, network=True))
    assert reasons(spec, IRIS) == {(RefusalReason.NO_FACTORY, "task")}
    assert reasons(task(EnvironmentSpec(kind=EnvironmentKind.SHELLSIM)), IRIS) == set()


def test_docker_build_task_is_refused_on_iris_and_accepted_on_local_docker():
    spec = task(EnvironmentSpec(kind=EnvironmentKind.DOCKER, image=BUILD))
    agent = task_execution(agent_user="agent")
    assert reasons(spec, IRIS_WORKING, agent) == {
        (RefusalReason.IMAGE_SOURCE, "task"),
        (RefusalReason.EXECUTION_USER, "task"),
    }
    assert reasons(spec, LAPTOP, agent) == set()
    published = spec.model_copy(update={"environment": EnvironmentSpec(kind=EnvironmentKind.DOCKER, image=IMAGE)})
    assert reasons(published, IRIS_WORKING) == set()
    networked = published.model_copy(
        update={"environment": EnvironmentSpec(kind=EnvironmentKind.DOCKER, image=IMAGE, network=True)}
    )
    assert reasons(networked, IRIS_WORKING) == {(RefusalReason.NETWORK, "task")}
    assert reasons(networked, LAPTOP) == set()


def test_laptop_without_docker_names_the_missing_factory():
    missing = FactoryCapabilities(
        frozenset(), frozenset(), False, False, False, False, unavailable="docker CLI not found on PATH"
    )
    spec = task(EnvironmentSpec(kind=EnvironmentKind.DOCKER, image=IMAGE))
    [refusal] = task_refusals(
        spec, task_execution(), {EnvironmentKind.SHELLSIM: SHELLSIM, EnvironmentKind.DOCKER: missing}
    )
    assert refusal.reason is RefusalReason.NO_FACTORY
    assert "docker CLI not found" in refusal.detail


def test_shell_verifier_grading_environment_and_collect_users_are_checked():
    verifier = ShellVerifierSpec(
        argv=("sh", "/grade.sh"),
        timeout=5,
        collect=(EnvironmentCommand(argv=("true",), timeout=5, user="grader"),),
    )
    spec = task(
        EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
        answer_type=AnswerType.FILE,
        verifier=shell_verifier(verifier, EnvironmentSpec(kind=EnvironmentKind.SHELLSIM, network=True)),
    )
    assert reasons(spec, LAPTOP) == {(RefusalReason.EXECUTION_USER, "task"), (RefusalReason.NETWORK, "verifier")}


def shell_verifier(
    verifier: ShellVerifierSpec, environment: EnvironmentSpec | None = None, files: tuple[EnvironmentFile, ...] = ()
) -> VerifierSpec:
    return VerifierSpec(
        kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json(), environment=environment, files=files
    )


DOCKER_TASK = EnvironmentSpec(kind=EnvironmentKind.DOCKER, image=IMAGE)


@pytest.mark.parametrize(
    "artifact",
    [
        VerifierArtifact(source="/work", target="/work", kind=ArtifactKind.AUTO),
        VerifierArtifact(source="/work/out", target="/out", kind=ArtifactKind.FILE, missing="skip"),
        VerifierArtifact(source="/work", target="/work", kind=ArtifactKind.DIRECTORY, exclude=(".git",)),
    ],
)
def test_grading_artifacts_the_engine_fetches_as_root_need_execution_users(artifact):
    verifier = ShellVerifierSpec(argv=("sh", "/grade.sh"), timeout=5, artifacts=(artifact,))
    spec = task(DOCKER_TASK, answer_type=AnswerType.FILE, verifier=shell_verifier(verifier, DOCKER_TASK))
    assert reasons(spec, IRIS_WORKING) == {(RefusalReason.EXECUTION_USER, "task")}
    assert reasons(spec, LAPTOP) == set()

    plain = VerifierArtifact(source="/work/out", target="/out", kind=ArtifactKind.FILE)
    plain_spec = spec.model_copy(
        update={"verifier": shell_verifier(verifier.model_copy(update={"artifacts": (plain,)}), DOCKER_TASK)}
    )
    assert reasons(plain_spec, IRIS_WORKING) == set()


def test_stage_grader_files_removed_as_root_need_execution_users():
    grader = shell_verifier(
        ShellVerifierSpec(argv=("sh", "/grade.sh"), timeout=5),
        files=(EnvironmentFile(path="/grade.sh", content=b"echo 1\n"),),
    )
    spec = task(
        DOCKER_TASK,
        answer_type=AnswerType.FILE,
        verifier=VerifierSpec(
            kind=VerifierKind.STAGED,
            parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.FINAL).model_dump_json(),
        ),
        stages=(TaskStage(name="only", verifier=grader),),
    )
    execution = task_execution(stages={"only": StageExecution()})
    assert reasons(spec, IRIS_WORKING, execution) == {(RefusalReason.EXECUTION_USER, "task")}
    assert reasons(spec, LAPTOP, execution) == set()


def test_explicit_file_timestamps_are_refused_where_they_are_not_kept():
    stamped = EnvironmentFile(path="/work/data.txt", content=b"1\n", mtime_ns=1_700_000_000_000_000_000)
    shellsim = task(EnvironmentSpec(kind=EnvironmentKind.SHELLSIM, files=(stamped,)))
    assert reasons(shellsim, LAPTOP) == {(RefusalReason.FILE_TIMESTAMPS, "task")}

    grader = shell_verifier(ShellVerifierSpec(argv=("sh", "/grade.sh"), timeout=5), files=(stamped,))
    docker = task(DOCKER_TASK, answer_type=AnswerType.FILE, verifier=grader)
    assert reasons(docker, LAPTOP) == set()
    assert reasons(docker, IRIS_WORKING) == {(RefusalReason.FILE_TIMESTAMPS, "task")}


def test_laptop_without_docker_reports_and_omits_the_docker_factory(monkeypatch):
    monkeypatch.setattr(factories.shutil, "which", lambda name: None)
    factories.local_docker.cache_clear()
    try:
        docker = factory_capabilities(MachineHost.LAPTOP)[EnvironmentKind.DOCKER]
        assert docker.unavailable == "docker CLI not found on PATH"
        assert EnvironmentKind.DOCKER not in machine_factories(MachineHost.LAPTOP, controller_url=None)
    finally:
        factories.local_docker.cache_clear()


def test_iris_docker_factory_submits_to_the_given_controller():
    docker = machine_factories(MachineHost.IRIS, controller_url="http://controller.example:10000")[
        EnvironmentKind.DOCKER
    ]
    assert isinstance(docker, IrisMachineFactory)
    assert docker.controller_url == "http://controller.example:10000"


SOURCES = {"registry": ShellboxRegistryImage("busybox:1"), "build": DockerfileSource(Path("/c"), Path("/c/Dockerfile"))}


async def assert_refused(factory, spec: MachineSpec) -> None:
    with pytest.raises(UnsupportedMachineSpec):
        await factory.create(spec)


async def test_capability_table_matches_what_the_shellbox_factories_refuse():
    """Each unsupported combination the table declares is refused by shellbox before any I/O.

    When an upstream backend gains a capability (for example NetworkPolicy.DENY on Iris), this
    test fails until the table is updated.
    """
    iris = IrisMachineFactory(controller_url="http://127.0.0.1:9")
    for kind in {"registry", "build"} - IRIS_DOCKER.image_sources:
        await assert_refused(iris, MachineSpec(source=SOURCES[kind], network=NetworkPolicy.ALLOW))
    await assert_refused(iris, MachineSpec(source=SOURCES["registry"], network=NetworkPolicy.ALLOW, gpus=1))
    # IRIS_DOCKER.network describes the patched backend. The shipped one still refuses DENY; when the
    # patch lands this fails, and IRIS_DOCKER.unavailable and this check go.
    assert IRIS_DOCKER.unavailable is not None
    await assert_refused(iris, MachineSpec(source=SOURCES["registry"], network=NetworkPolicy.DENY))

    shellsim = ShellSimMachineFactory()
    for policy in set(NetworkPolicy) - SHELLSIM.network:
        await assert_refused(shellsim, MachineSpec(source=ShellSimBuiltins(), network=policy))
    await assert_refused(shellsim, MachineSpec(source=ShellSimBuiltins(), cpus=1))
    await assert_refused(shellsim, MachineSpec(source=SOURCES["registry"]))
    assert await shellsim.create(MachineSpec(source=ShellSimBuiltins()))

    # Local Docker refuses registry images and Dockerfiles only when it has no Skopeo image cache.
    await assert_refused(DockerMachineFactory(), MachineSpec(source=SOURCES["build"]))


async def test_rollout_engine_refuses_the_file_timestamps_shellsim_lacks():
    """``SHELLSIM.file_timestamps`` mirrors RolloutEngine's own check, made before any machine starts."""

    async def unused_model(request: ModelRequest) -> ModelTurn:
        raise AssertionError("the engine refuses the task before any model call")

    stamped = EnvironmentFile(path="/work/data.txt", content=b"1\n", mtime_ns=1_700_000_000_000_000_000)
    engine = ShellboxRolloutEngine(
        unused_model,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        max_turns=1,
        command_timeout=1,
        cleanup_timeout=1,
        convention=PlainText(id="plain"),
    )
    assert not SHELLSIM.file_timestamps
    with pytest.raises(ValueError, match="timestamps"):
        await engine.run(
            task(EnvironmentSpec(kind=EnvironmentKind.SHELLSIM, files=(stamped,))), execution=task_execution()
        )
