# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _machine_spec, _prepare_machine
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    MachineSpec,
    NetworkPolicy,
    UnsupportedMachineSpec,
)
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    Grader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import JudgeSpec, NumericSpec

from taskforge.sandbox import factories
from taskforge.sandbox.factories import (
    LocalSandbox,
    MachineHost,
    MachineRole,
    RefusalReason,
    container_backend,
    grading_environment,
    machine_factories,
    task_refusals,
    warm_local_python,
)
from taskforge.spec import draft

IMAGE = "registry.example/task@sha256:" + "0" * 64
SHELL = ("shell", "filesystem")
LIMITS = draft.rollout_limits(
    max_turns=1,
    model_turn_timeout=None,
    tool_turn_timeout=2,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=5,
    cleanup_timeout=1,
)


def selection(backend: Backend, **update) -> MachineRuntimeSpec:
    fields: dict[str, Any] = dict(
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


LOCKED = EnvironmentRequirements(
    command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="gs://bucket/env/uv.lock"
)


@dataclass(frozen=True)
class FakeFactory:
    """A factory for a backend whose shellbox module installs only on Linux; it accepts every machine."""

    backend: Backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        pass

    async def create(self, spec: MachineSpec):
        raise AssertionError(f"Unexpected machine for {spec}")


GVISOR = IrisMachineFactory(controller_url="http://127.0.0.1:9")
SHELLSIM = ShellSimMachineFactory()
IRIS = {Backend.SHELLSIM.value: SHELLSIM, Backend.GVISOR.value: GVISOR, Backend.LOCAL.value: FakeFactory(Backend.LOCAL)}
LAPTOP = {Backend.SHELLSIM.value: SHELLSIM, Backend.DOCKER.value: DockerMachineFactory()}


def script_grader(
    environment: EnvironmentRequirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE
    )
) -> ScriptGrader:
    return ScriptGrader(argv=("sh", "/tests/grade.sh"), environment=environment)


def lowered(
    task_machine: MachineRuntimeSpec | None,
    *,
    image: str | None = None,
    capabilities: tuple[str, ...] = SHELL,
    grader: Grader | None = None,
    verifier_machine: MachineRuntimeSpec | None = None,
) -> TaskRolloutSpec:
    environment = (
        EnvironmentRequirements()
        if task_machine is None
        else EnvironmentRequirements(
            capabilities=capabilities,
            command_semantics=CommandSemantics.SHELL_SIMULATOR if image is None else CommandSemantics.LINUX_PROCESS,
            docker_image=image,
        )
    )
    task = TaskSpec(
        id="t",
        context=ConversationInput(events=(TextMessage(role="user", content="hi"),)),
        environment_requirements=environment,
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=grader or verifyit_package(NumericSpec("1", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        resources=ResourceGroups(),
    )
    return TaskRolloutSpec(
        task=task,
        harness=draft.SHELL_HARNESS,
        task_machine=task_machine,
        verifier_machine=verifier_machine,
        limits=LIMITS,
    )


def reasons(spec: TaskRolloutSpec, machines) -> set[tuple[RefusalReason, MachineRole]]:
    return {(refusal.reason, refusal.where) for refusal in task_refusals(spec, machines)}


def test_shellsim_and_machineless_tasks_run_everywhere():
    for machines in (IRIS, LAPTOP):
        assert reasons(lowered(None), machines) == set()
        assert reasons(lowered(selection(Backend.SHELLSIM)), machines) == set()


def test_shellsim_accepts_a_storage_limit_and_refuses_a_cpu_limit():
    assert reasons(lowered(selection(Backend.SHELLSIM, storage_mb=64)), LAPTOP) == set()
    [refusal] = task_refusals(lowered(selection(Backend.SHELLSIM, cpus=2)), LAPTOP)
    assert (refusal.reason, refusal.where) == (RefusalReason.UNSUPPORTED, MachineRole.TASK)
    assert "CPU" in refusal.detail


def test_image_tasks_run_on_each_hosts_container_backend():
    for host, machines in ((MachineHost.IRIS, IRIS), (MachineHost.LAPTOP, LAPTOP)):
        spec = lowered(selection(container_backend(host), network=NetworkPolicy.ALLOW), image=IMAGE)
        assert reasons(spec, machines) == set()
    # A draft lowered on the laptop names a backend Iris has no factory for.
    [refusal] = task_refusals(lowered(selection(Backend.DOCKER), image=IMAGE), IRIS)
    assert (refusal.reason, refusal.where) == (RefusalReason.NO_FACTORY, MachineRole.TASK)
    assert refusal.detail.startswith("docker:")


def test_a_network_task_on_a_deny_policy_is_refused_with_the_factorys_message():
    spec = lowered(selection(Backend.GVISOR), image=IMAGE, capabilities=(*SHELL, "network"))
    with pytest.raises(UnsupportedMachineSpec) as rejected:
        GVISOR.check_environment(spec.task.environment_requirements, MachineSpec(source=RegistryImage(IMAGE)))

    assert [(r.reason, r.where, r.detail) for r in task_refusals(spec, IRIS)] == [
        (RefusalReason.UNSUPPORTED, MachineRole.TASK, f"gvisor: {rejected.value}")
    ]
    allowed = lowered(
        selection(Backend.GVISOR, network=NetworkPolicy.ALLOW),
        image=IMAGE,
        capabilities=spec.task.environment_requirements.capabilities,
    )
    assert reasons(allowed, IRIS) == set()


def test_image_must_match_whether_the_backend_takes_one():
    assert reasons(lowered(selection(Backend.SHELLSIM), image=IMAGE), IRIS) == {
        (RefusalReason.UNSUPPORTED, MachineRole.TASK)
    }
    assert reasons(lowered(selection(Backend.GVISOR)), IRIS) == {(RefusalReason.UNSUPPORTED, MachineRole.TASK)}


def test_verifier_machine_is_checked_against_its_own_backend():
    spec = lowered(
        selection(Backend.SHELLSIM),
        grader=script_grader(),
        verifier_machine=selection(Backend.GVISOR, gpus=1),
    )
    [refusal] = task_refusals(spec, IRIS)
    assert (refusal.reason, refusal.where) == (RefusalReason.UNSUPPORTED, MachineRole.VERIFIER)
    assert "GPU" in refusal.detail


def test_a_judge_takes_no_verifier_machine_whatever_environment_its_grader_names():
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE)
    judge = verifyit_package(JudgeSpec(references=("r",), question="q", model="m"), environment=environment).grader
    numeric = verifyit_package(NumericSpec("1", tolerance_abs=0, tolerance_rel=0), environment=environment).grader

    assert grading_environment(lowered(None, grader=judge).task) is None
    assert grading_environment(lowered(None, grader=numeric).task) == environment


def test_laptop_without_docker_omits_its_factory_and_logs_why(monkeypatch, caplog):
    monkeypatch.setattr(factories.shutil, "which", lambda name: None)
    factories.docker_unavailable.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger=factories.__name__):
            built = machine_factories(MachineHost.LAPTOP, controller_url=None)
    finally:
        factories.docker_unavailable.cache_clear()
    assert set(built) == {Backend.SHELLSIM.value}
    assert "docker CLI not found" in caplog.text
    assert reasons(lowered(selection(Backend.DOCKER), image=IMAGE), built) == {
        (RefusalReason.NO_FACTORY, MachineRole.TASK)
    }


def test_a_lock_only_grader_needs_the_local_factory_of_an_iris_task(monkeypatch, caplog):
    graded = lowered(None, grader=script_grader(LOCKED), verifier_machine=selection(Backend.LOCAL))

    assert reasons(graded, IRIS) == set()
    assert reasons(graded, LAPTOP) == {(RefusalReason.NO_FACTORY, MachineRole.VERIFIER)}
    monkeypatch.setattr(factories, "local_sandbox", lambda: LocalSandbox(factory=None, unavailable="no bwrap here"))
    with caplog.at_level(logging.WARNING, logger=factories.__name__):
        built = machine_factories(MachineHost.IRIS, controller_url="http://controller.example:10000")
    assert "no bwrap here" in caplog.text
    assert reasons(graded, built) == {(RefusalReason.NO_FACTORY, MachineRole.VERIFIER)}


@pytest.mark.parametrize("host", list(MachineHost))
def test_factories_cover_the_hosts_backends(monkeypatch, host):
    local = FakeFactory(Backend.LOCAL)
    monkeypatch.setattr(factories, "docker_unavailable", lambda: None)
    monkeypatch.setattr(factories, "local_sandbox", lambda: LocalSandbox(factory=local, unavailable=None))
    is_iris = host is MachineHost.IRIS
    built = machine_factories(host, controller_url="http://controller.example:10000" if is_iris else None)
    expected = {Backend.SHELLSIM.value, container_backend(host).value} | ({Backend.LOCAL.value} if is_iris else set())
    assert set(built) == expected
    assert all(factory.backend.value == key for key, factory in built.items())
    container = built[container_backend(host).value]
    if is_iris:
        assert isinstance(container, IrisMachineFactory)
        assert container.controller_url == "http://controller.example:10000"
    else:
        assert isinstance(container, DockerMachineFactory)


@dataclass(frozen=True)
class FakeLocalFactory(FakeFactory):
    """The local factory's shape: a package cache and a Python version the warm-up resolves."""

    package_cache: Path = Path("/cache/python")
    python_version: str = "3.12.12"


def test_warming_resolves_the_bare_cpython_environment_once_off_the_caller_thread(monkeypatch, caplog):
    calls: list[tuple[Path, tuple[str, ...], str, str]] = []
    monkeypatch.setattr(
        factories,
        "resolve_python_environment",
        lambda cache, requirements, version: calls.append(
            (cache, requirements, version, threading.current_thread().name)
        ),
    )
    built = {**IRIS, Backend.LOCAL.value: FakeLocalFactory(Backend.LOCAL)}

    with caplog.at_level(logging.INFO, logger=factories.__name__):
        thread = warm_local_python(built)
        assert thread is not None
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert calls == [(Path("/cache/python"), (), "3.12.12", factories.WARM_THREAD)]
    assert "Warmed the local Python 3.12.12 environment" in caplog.text


def test_a_failed_warm_up_is_logged_and_leaves_the_first_machine_to_resolve(monkeypatch, caplog):
    def refuse(cache, requirements, version):
        raise RuntimeError("uv is not installed")

    monkeypatch.setattr(factories, "resolve_python_environment", refuse)
    built = {**IRIS, Backend.LOCAL.value: FakeLocalFactory(Backend.LOCAL)}

    with caplog.at_level(logging.WARNING, logger=factories.__name__):
        thread = warm_local_python(built)
        assert thread is not None
        thread.join(timeout=10)

    assert "Warming the local Python 3.12.12 environment in /cache/python failed" in caplog.text
    assert "uv is not installed" in caplog.text


def test_without_a_local_factory_there_is_nothing_to_warm(monkeypatch):
    monkeypatch.setattr(factories, "resolve_python_environment", lambda *args: pytest.fail("resolved without a factory"))

    assert warm_local_python({Backend.SHELLSIM.value: SHELLSIM}) is None
    assert warm_local_python(IRIS) is None


def test_factory_arguments_must_match_the_host():
    with pytest.raises(ValueError, match="controller URL"):
        machine_factories(MachineHost.LAPTOP, controller_url="http://c:1")
    with pytest.raises(ValueError, match="controller URL"):
        machine_factories(MachineHost.IRIS, controller_url=None)


@pytest.mark.manual
async def test_a_taskforge_shellsim_machine_runs_the_catalogs_cpython():
    """Fetches ``SHELLSIM_CATALOG``'s CPython, which the unit suite leaves to ShellSim's own tests."""
    factory = machine_factories(MachineHost.LAPTOP, controller_url=None)[Backend.SHELLSIM.value]
    spec = _machine_spec(draft.requirements(image=None), selection(Backend.SHELLSIM))
    machine = await factory.create(spec)
    try:
        result = await machine.run(
            Command(argv=("python3", "-c", "import sys; print(6 * 7, sys.implementation.name)"), timeout=120)
        )
    finally:
        await machine.close()

    assert (result.exit_code, result.stdout) == (0, b"42 cpython\n")


@pytest.mark.manual
async def test_a_grader_base_machine_runs_cpython_312_on_the_local_backend():
    """Needs bubblewrap (Linux, as root or with user namespaces) and uv; installs CPython into the package cache."""
    sandbox = factories.local_sandbox()
    assert sandbox.factory is not None, sandbox.unavailable
    built = {Backend.LOCAL.value: sandbox.factory}
    requirements = draft.host_requirements(draft.grader_environment(None), built)
    runtime = draft.machine_runtime(draft.machine(startup_timeout=600), requirements, MachineHost.IRIS)
    async with AsyncExitStack() as owned:
        machine = await _prepare_machine(requirements, runtime, (), built, _Cleanup(60), owned)
        assert machine is not None
        result = await machine.run(
            Command(argv=("python3", "-c", "import sys; print(*sys.version_info[:2])"), cwd="/app", timeout=60)
        )

    assert (runtime.backend, result.exit_code, result.stdout) == (Backend.LOCAL.value, 0, b"3 12\n")
