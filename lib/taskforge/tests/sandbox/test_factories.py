# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invariants: the task runs on the machine it declares (refusals name the machine a host cannot provide;
``machine_factories`` raises rather than shrink the selection); the solver machine has no network (a network task on a
``DENY`` machine is refused)."""

import subprocess
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

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
    unset = dict.fromkeys(("cpus", "memory_mb", "storage_mb", "user", "startup_timeout", "cleanup_timeout"))
    return MachineRuntimeSpec(**(unset | dict(backend=backend.value, network=NetworkPolicy.DENY, gpus=0) | update))


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
    ),
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
        harness=draft.TOOL_HARNESS,
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


def no_docker_cli(monkeypatch):
    monkeypatch.setattr(factories.shutil, "which", lambda name: None)


def hung_docker_daemon(monkeypatch):
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], factories.DOCKER_PROBE_TIMEOUT)

    monkeypatch.setattr(factories.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(factories.subprocess, "run", hang)


def no_bwrap(monkeypatch):
    def unavailable():
        raise RuntimeError("no bwrap here")

    monkeypatch.setattr(factories, "local_factory", unavailable)


@pytest.mark.parametrize(
    "host,requested,message,other",
    [
        (no_docker_cli, Backend.DOCKER, "Requested docker backend unavailable: docker CLI not found", Backend.SHELLSIM),
        (hung_docker_daemon, Backend.DOCKER, "docker daemon did not answer", Backend.SHELLSIM),
        (no_bwrap, Backend.LOCAL, "no bwrap here", Backend.GVISOR),
    ],
)
def test_a_requested_backend_the_host_cannot_start_fails_without_shrinking_the_selection(
    monkeypatch, host, requested, message, other
):
    host(monkeypatch)
    controller = "http://controller.example:10000"

    with pytest.raises(RuntimeError, match=message):
        machine_factories((other, requested), controller_url=controller)
    # The same host can explicitly request the other backend alone.
    assert set(machine_factories((other,), controller_url=controller)) == {other.value}


@pytest.mark.parametrize("backends", [(Backend.SHELLSIM,), (Backend.LOCAL, Backend.GVISOR), (Backend.DOCKER,)])
def test_factories_construct_exactly_the_selected_backends(monkeypatch, backends):
    monkeypatch.setattr(factories, "docker_unavailable_reason", lambda: None)
    monkeypatch.setattr(factories, "local_factory", lambda: FakeFactory(Backend.LOCAL))
    controller = "http://controller.example:10000"
    built = machine_factories(backends, controller_url=controller)

    assert set(built) == {backend.value for backend in backends}
    assert all(factory.backend.value == key for key, factory in built.items())
    if Backend.GVISOR in backends:
        iris = built[Backend.GVISOR.value]
        assert isinstance(iris, IrisMachineFactory)
        assert iris.controller_url == controller


@dataclass(frozen=True)
class FakeLocalFactory(FakeFactory):
    """The local factory's shape: a package cache and a Python version the warm-up resolves."""

    package_cache: Path = Path("/cache/python")
    python_version: str = "3.12.12"


def test_warming_resolves_the_bare_cpython_environment_once_off_the_caller_thread_and_only_with_a_local_factory(
    monkeypatch,
):
    calls: list[tuple[Path, tuple[str, ...], str, str]] = []
    monkeypatch.setattr(
        factories,
        "resolve_python_environment",
        lambda cache, requirements, version: calls.append(
            (cache, requirements, version, threading.current_thread().name)
        ),
    )
    assert warm_local_python({Backend.SHELLSIM.value: SHELLSIM}) is None
    assert warm_local_python(IRIS) is None

    thread = warm_local_python({**IRIS, Backend.LOCAL.value: FakeLocalFactory(Backend.LOCAL)})
    assert thread is not None
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert calls == [(Path("/cache/python"), (), "3.12.12", factories.WARM_THREAD)]


@pytest.mark.manual
async def test_a_taskforge_shellsim_machine_runs_the_catalogs_cpython():
    """Fetches ``SHELLSIM_CATALOG``'s CPython, which the unit suite leaves to ShellSim's own tests."""
    factory = machine_factories((Backend.SHELLSIM,), controller_url=None)[Backend.SHELLSIM.value]
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
    built = machine_factories((Backend.LOCAL,), controller_url=None)
    requirements = draft.host_requirements(draft.grader_environment(None), built)
    runtime = draft.machine_runtime(draft.machine(startup_timeout=600), requirements, MachineHost.IRIS)
    async with AsyncExitStack() as owned:
        machine = await _prepare_machine(requirements, runtime, (), built, _Cleanup(60), owned)
        assert machine is not None
        result = await machine.run(
            Command(argv=("python3", "-c", "import sys; print(*sys.version_info[:2])"), cwd="/app", timeout=60)
        )

    assert (runtime.backend, result.exit_code, result.stdout) == (Backend.LOCAL.value, 0, b"3 12\n")
