# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

import pytest
from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _machine_spec, _prepare_machine
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    MachineSpec,
    NetworkPolicy,
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
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import JudgeSpec, NumericSpec

from taskforge.sandbox import factories
from taskforge.sandbox.factories import (
    grading_environment,
    machine_factories,
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


@dataclass(frozen=True)
class FakeFactory:
    """A factory for a backend whose shellbox module installs only on Linux; it accepts every machine."""

    backend: Backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        pass

    async def create(self, spec: MachineSpec):
        raise AssertionError(f"Unexpected machine for {spec}")


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


def test_a_judge_takes_no_verifier_machine_whatever_environment_its_grader_names():
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE)
    judge = verifyit_package(JudgeSpec(references=("r",), question="q", model="m"), environment=environment).grader
    numeric = verifyit_package(NumericSpec("1", tolerance_abs=0, tolerance_rel=0), environment=environment).grader

    assert grading_environment(lowered(None, grader=judge).task) is None
    assert grading_environment(lowered(None, grader=numeric).task) == environment


def test_requested_docker_fails_if_the_host_has_no_docker(monkeypatch):
    monkeypatch.setattr(factories.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="Requested docker backend unavailable: docker CLI not found"):
        machine_factories((Backend.SHELLSIM, Backend.DOCKER), controller_url=None)
    # The same host can explicitly request ShellSim without silently changing its selection.
    built = machine_factories((Backend.SHELLSIM,), controller_url=None)
    assert set(built) == {Backend.SHELLSIM.value}


def test_requested_docker_fails_if_its_daemon_hangs(monkeypatch):
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], factories.DOCKER_PROBE_TIMEOUT)

    monkeypatch.setattr(factories.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(factories.subprocess, "run", hang)
    with pytest.raises(RuntimeError, match="docker daemon did not answer"):
        machine_factories((Backend.DOCKER,), controller_url=None)


def test_requested_local_backend_propagates_its_availability_failure(monkeypatch):
    def unavailable():
        raise RuntimeError("no bwrap here")

    monkeypatch.setattr(factories, "local_factory", unavailable)
    with pytest.raises(RuntimeError, match="no bwrap here"):
        machine_factories((Backend.LOCAL,), controller_url=None)
    built = machine_factories((Backend.GVISOR,), controller_url="http://controller.example:10000")
    assert set(built) == {Backend.GVISOR.value}


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
    runtime = draft.machine_runtime(draft.machine(startup_timeout=600), requirements, built)
    async with AsyncExitStack() as owned:
        machine = await _prepare_machine(requirements, runtime, (), built, _Cleanup(60), owned)
        assert machine is not None
        result = await machine.run(
            Command(argv=("python3", "-c", "import sys; print(*sys.version_info[:2])"), cwd="/app", timeout=60)
        )

    assert (runtime.backend, result.exit_code, result.stdout) == (Backend.LOCAL.value, 0, b"3 12\n")
