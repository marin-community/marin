# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

import pytest
from rolloutengine.lowering import SHELLBOX_SESSION, validate_lowered_task
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import Backend, Command, MachineSpec, NetworkPolicy, ShellSimBuiltins, UnsupportedMachineSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    Grader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import NumericSpec

from taskforge.sandbox import factories
from taskforge.sandbox.factories import (
    IRIS_GVISOR,
    LOCAL_DOCKER,
    LOCAL_SANDBOX,
    SHELLSIM,
    LocalDocker,
    LocalSandbox,
    MachineHost,
    MachineRole,
    RefusalReason,
    container_backend,
    factory_capabilities,
    machine_factories,
    task_refusals,
)

IMAGE = "registry.example/task@sha256:" + "0" * 64
SHELL = ("shell", "filesystem")
IRIS = {Backend.SHELLSIM.value: SHELLSIM, Backend.GVISOR.value: IRIS_GVISOR, Backend.LOCAL.value: LOCAL_SANDBOX}
LAPTOP = {Backend.SHELLSIM.value: SHELLSIM, Backend.DOCKER.value: LOCAL_DOCKER}
SESSION = TaskSessionSpec(
    task_session=SHELLBOX_SESSION,
    max_turns=1,
    model_turn_timeout=None,
    command_timeout=1,
    tool_turn_timeout=2,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=5,
    cleanup_timeout=1,
)
STAMPED = inline_resource("work/data.txt", b"1\n").model_copy(update={"mtime_ns": 1_700_000_000_000_000_000})


def selection(backend: Backend, **update) -> MachineRuntimeSpec:
    fields = dict(
        backend=backend.value,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=None,
        cleanup_timeout=None,
    )
    return MachineRuntimeSpec(**(fields | update))


LOCKED = EnvironmentRequirements(compatible_backends=(Backend.LOCAL,), packages_lock="gs://bucket/env/uv.lock")


@dataclass(frozen=True)
class FakeFactory:
    backend: Backend

    async def create(self, spec: MachineSpec):
        raise AssertionError(f"Unexpected machine for {spec}")


def script_grader(environment: EnvironmentRequirements = EnvironmentRequirements(docker_image=IMAGE)) -> ScriptGrader:
    return ScriptGrader(argv=("sh", "/tests/grade.sh"), environment=environment)


def lowered(
    task_machine: MachineRuntimeSpec | None,
    *,
    image: str | None = None,
    grader: Grader | None = None,
    verifier_machine: MachineRuntimeSpec | None = None,
    worker: tuple[TaskResource, ...] = (),
    verifier_files: tuple[TaskResource, ...] = (),
) -> LoweredTaskSpec:
    environment = (
        EnvironmentRequirements()
        if task_machine is None
        else EnvironmentRequirements(capabilities=SHELL, docker_image=image)
    )
    task = TaskSpec(
        id="t",
        context=ConversationInput(events=(TextMessage(role="user", content="hi"),)),
        environment_requirements=environment,
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=grader or verifyit_package(NumericSpec("1", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        resources=ResourceGroups(worker=worker, verifier=verifier_files),
    )
    return LoweredTaskSpec(
        task=task, runtime=TaskRuntimeSpec(task_machine=task_machine, verifier_machine=verifier_machine), session=SESSION
    )


def reasons(spec: LoweredTaskSpec, capabilities) -> set[tuple[RefusalReason, MachineRole]]:
    return {(refusal.reason, refusal.where) for refusal in task_refusals(spec, capabilities)}


def test_shellsim_and_machineless_tasks_run_everywhere():
    for capabilities in (IRIS, LAPTOP):
        assert reasons(lowered(None), capabilities) == set()
        assert reasons(lowered(selection(Backend.SHELLSIM)), capabilities) == set()


def test_image_tasks_run_on_each_hosts_container_backend():
    for host, capabilities in ((MachineHost.IRIS, IRIS), (MachineHost.LAPTOP, LAPTOP)):
        spec = lowered(selection(container_backend(host), network=NetworkPolicy.ALLOW), image=IMAGE)
        assert reasons(spec, capabilities) == set()
    # A draft lowered on the laptop names a backend Iris has no factory for.
    laptop_draft = lowered(selection(Backend.DOCKER), image=IMAGE)
    assert reasons(laptop_draft, IRIS) == {(RefusalReason.NO_FACTORY, MachineRole.TASK)}


def test_image_must_match_whether_the_backend_takes_one():
    assert reasons(lowered(selection(Backend.SHELLSIM), image=IMAGE), IRIS) == {(RefusalReason.IMAGE, MachineRole.TASK)}
    assert reasons(lowered(selection(Backend.GVISOR)), IRIS) == {(RefusalReason.IMAGE, MachineRole.TASK)}


def test_shellsim_refuses_network_limits_gpus_and_non_root_users():
    spec = lowered(
        selection(Backend.SHELLSIM, network=NetworkPolicy.ALLOW, cpus=2, gpus=1, user="agent"),
    )
    assert reasons(spec, LAPTOP) == {
        (RefusalReason.NETWORK, MachineRole.TASK),
        (RefusalReason.RESOURCE_LIMITS, MachineRole.TASK),
        (RefusalReason.GPUS, MachineRole.TASK),
        (RefusalReason.EXECUTION_USER, MachineRole.TASK),
    }
    assert reasons(lowered(selection(Backend.SHELLSIM, user="root")), LAPTOP) == set()


def test_gvisor_accepts_only_root_while_local_docker_accepts_any_user():
    assert reasons(lowered(selection(Backend.GVISOR, user="agent"), image=IMAGE), IRIS) == {
        (RefusalReason.EXECUTION_USER, MachineRole.TASK)
    }
    assert reasons(lowered(selection(Backend.GVISOR, user="0"), image=IMAGE), IRIS) == set()
    assert reasons(lowered(selection(Backend.DOCKER, user="agent"), image=IMAGE), LAPTOP) == set()


def test_verifier_machine_is_checked_against_its_own_backend():
    spec = lowered(
        selection(Backend.GVISOR),
        image=IMAGE,
        grader=script_grader(),
        verifier_machine=selection(Backend.GVISOR, user="grader", gpus=1),
    )
    assert reasons(spec, IRIS) == {
        (RefusalReason.EXECUTION_USER, MachineRole.VERIFIER),
        (RefusalReason.GPUS, MachineRole.VERIFIER),
    }


def test_explicit_file_timestamps_are_refused_where_they_are_not_kept():
    assert reasons(lowered(selection(Backend.SHELLSIM), worker=(STAMPED,)), LAPTOP) == {
        (RefusalReason.FILE_TIMESTAMPS, MachineRole.TASK)
    }
    # Verifier files go only to the verifier machine.
    graded = lowered(
        selection(Backend.SHELLSIM),
        grader=script_grader(),
        verifier_machine=selection(container_backend(MachineHost.IRIS)),
        verifier_files=(STAMPED,),
    )
    assert reasons(graded, IRIS) == {(RefusalReason.FILE_TIMESTAMPS, MachineRole.VERIFIER)}
    on_laptop = graded.model_copy(
        update={
            "runtime": TaskRuntimeSpec(
                task_machine=selection(Backend.SHELLSIM), verifier_machine=selection(Backend.DOCKER)
            )
        }
    )
    assert reasons(on_laptop, LAPTOP) == set()


def test_laptop_without_docker_names_the_missing_factory_and_omits_it(monkeypatch, tmp_path):
    monkeypatch.setattr(factories.shutil, "which", lambda name: None)
    factories.local_docker.cache_clear()
    try:
        capabilities = factory_capabilities(MachineHost.LAPTOP)
        built = machine_factories(MachineHost.LAPTOP, controller_url=None, image_cache=tmp_path)
    finally:
        factories.local_docker.cache_clear()
    [refusal] = task_refusals(lowered(selection(Backend.DOCKER), image=IMAGE), capabilities)
    assert (refusal.reason, refusal.where) == (RefusalReason.NO_FACTORY, MachineRole.TASK)
    assert "docker CLI not found" in refusal.detail
    assert set(built) == {Backend.SHELLSIM.value}


def test_a_lock_only_grader_needs_the_local_factory_of_an_iris_task(monkeypatch):
    graded = lowered(None, grader=script_grader(LOCKED), verifier_machine=selection(Backend.LOCAL))

    assert reasons(graded, IRIS) == set()
    assert reasons(graded, LAPTOP) == {(RefusalReason.NO_FACTORY, MachineRole.VERIFIER)}
    monkeypatch.setattr(factories, "local_sandbox", lambda: LocalSandbox(factory=None, unavailable="no bwrap here"))
    [refusal] = task_refusals(graded, factory_capabilities(MachineHost.IRIS))
    assert (refusal.reason, refusal.where, refusal.detail) == (
        RefusalReason.NO_FACTORY,
        MachineRole.VERIFIER,
        "local: no bwrap here",
    )


@pytest.mark.parametrize("host", list(MachineHost))
def test_factories_and_capabilities_cover_the_hosts_backends(monkeypatch, tmp_path, host):
    skopeo = Path("/opt/bin/skopeo")
    local = FakeFactory(Backend.LOCAL)
    monkeypatch.setattr(factories, "local_docker", lambda: LocalDocker(skopeo=skopeo, unavailable=None))
    monkeypatch.setattr(factories, "local_sandbox", lambda: LocalSandbox(factory=local, unavailable=None))
    is_iris = host is MachineHost.IRIS
    built = machine_factories(
        host,
        controller_url="http://controller.example:10000" if is_iris else None,
        image_cache=None if is_iris else tmp_path / "images",
    )
    expected = {Backend.SHELLSIM.value, container_backend(host).value} | ({Backend.LOCAL.value} if is_iris else set())
    assert set(built) == set(factory_capabilities(host)) == expected
    assert all(factory.backend.value == key for key, factory in built.items())
    container = built[container_backend(host).value]
    if is_iris:
        assert isinstance(container, IrisMachineFactory)
        assert container.controller_url == "http://controller.example:10000"
    else:
        assert isinstance(container, DockerMachineFactory)
        assert (container.image_cache, container.skopeo) == (tmp_path / "images", skopeo)


def test_factory_arguments_must_match_the_host(tmp_path):
    with pytest.raises(ValueError, match="controller URL"):
        machine_factories(MachineHost.LAPTOP, controller_url="http://c:1", image_cache=tmp_path)
    with pytest.raises(ValueError, match="image cache"):
        machine_factories(MachineHost.IRIS, controller_url="http://c:1", image_cache=tmp_path)


async def assert_refused(factory, spec: MachineSpec) -> None:
    with pytest.raises(UnsupportedMachineSpec):
        await factory.create(spec)


async def test_capability_table_matches_what_the_shellbox_factories_refuse():
    """Each unsupported combination the table declares is refused by shellbox before any I/O.

    When an upstream backend gains a capability (for example GPUs on Iris), this test fails until the
    table is updated.
    """
    registry = RegistryImage(IMAGE)
    iris = IrisMachineFactory(controller_url="http://127.0.0.1:9")
    assert IRIS_GVISOR.image and not IRIS_GVISOR.gpus
    await assert_refused(iris, MachineSpec(source=ShellSimBuiltins()))
    await assert_refused(iris, MachineSpec(source=DockerfileSource(Path("/c"), Path("/c/Dockerfile"))))
    await assert_refused(iris, MachineSpec(source=registry, gpus=1))

    shellsim = ShellSimMachineFactory()
    for policy in set(NetworkPolicy) - SHELLSIM.network:
        await assert_refused(shellsim, MachineSpec(source=ShellSimBuiltins(), network=policy))
    await assert_refused(shellsim, MachineSpec(source=ShellSimBuiltins(), cpus=1))
    await assert_refused(shellsim, MachineSpec(source=ShellSimBuiltins(), gpus=1))
    await assert_refused(shellsim, MachineSpec(source=registry))
    machine = await shellsim.create(MachineSpec(source=ShellSimBuiltins()))
    assert SHELLSIM.execution_users is not None
    for user in SHELLSIM.execution_users:
        assert (await machine.run(Command(("true",), user=user))).exit_code == 0
    with pytest.raises(UnsupportedMachineSpec):
        await machine.run(Command(("true",), user="agent"))
    await machine.close()


def test_lowering_refuses_the_file_timestamps_shellsim_lacks():
    """``SHELLSIM.file_timestamps`` mirrors RolloutEngine's own check, made before any machine starts."""
    assert not SHELLSIM.file_timestamps
    with pytest.raises(NotImplementedError, match="timestamps"):
        validate_lowered_task(
            lowered(selection(Backend.SHELLSIM), worker=(STAMPED,)),
            factories={Backend.SHELLSIM.value: ShellSimMachineFactory()},
            sessions={},
        )
