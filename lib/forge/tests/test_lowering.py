# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest
from rolloutengine.machines import _machine_spec
from rolloutengine.models import RolloutLimits, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task, machine_runtime, rollout_limits
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineFactory, MachineSpec
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    ScriptGrader,
    StdoutReward,
)

from forge.lowering import (
    GRADING_MACHINE,
    LoweringSettings,
    host_requirements,
    lower,
    lower_with_settings,
    lowering_settings,
    machine_backend,
    requirements,
)
from forge.policy import load_policy
from forge.resources.sandbox.factories import MachineHost
from forge.resources.sandbox.images import GRADER_BASE_IMAGE


@dataclass(frozen=True, slots=True)
class AvailableFactory(MachineFactory):
    backend: Backend

    @override
    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        """The Iris container backend supports this fixture; lowering starts no machine."""

    @override
    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError("Lowering must not create a machine")


@pytest.mark.parametrize("host,backend", [(MachineHost.IRIS, Backend.GVISOR)])
def test_lowering_binds_an_image_task_to_its_host_and_survives_engine_wire_format(
    host: MachineHost, backend: Backend
) -> None:
    image = "docker.io/example/task@sha256:" + "0" * 64
    task = arithmetic_task(environment_requirements=requirements(image=image))
    lowered = lower(
        task,
        host=host,
        task_machine=GRADING_MACHINE,
        verifier_machine=None,
        limits=RolloutLimits(
            max_turns=2,
            max_total_tokens=None,
            model_turn_timeout=None,
            tool_turn_timeout=None,
            total_turn_timeout=None,
            attempt_timeout=None,
            verifier_timeout=None,
            cleanup_timeout=10,
        ),
        factories={backend.value: AvailableFactory(backend)},
    )
    restored = TaskRolloutSpec.model_validate_json(lowered.model_dump_json())
    assert restored.task_machine is not None
    assert restored.task_machine.backend == backend.value
    assert restored.verifier_machine is None
    assert restored.task == task


@pytest.mark.parametrize(
    "image",
    [
        None,
        GRADER_BASE_IMAGE,
        "registry.example.com/python@sha256:" + "a" * 64,
    ],
)
def test_grader_image_precheck_respects_available_host_factories(image: str | None) -> None:
    task = arithmetic_task(
        grader=ScriptGrader(
            argv=("sh", "-c", "echo 1"),
            reward=StdoutReward(),
            environment=requirements(image=image),
            answer_path=None,
            timeout=1,
        )
    )
    settings = LoweringSettings(MachineHost.IRIS, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    factories = {backend.value: AvailableFactory(backend) for backend in (Backend.GVISOR, Backend.SHELLSIM)}
    if image not in (None, GRADER_BASE_IMAGE):
        with pytest.raises(ValueError, match="grader image must be") as raised:
            lower_with_settings(task, settings, factories)
        assert image in str(raised.value) and GRADER_BASE_IMAGE in str(raised.value)
    else:
        lowered = lower_with_settings(task, settings, factories)
        assert lowered.verifier_machine is not None
        assert lowered.verifier_machine.backend == (Backend.SHELLSIM if image is None else Backend.GVISOR)


@pytest.mark.parametrize("directory", [None, "/custom"])
def test_missing_workdir_resolves_before_machine_creation(directory: str | None) -> None:
    declared = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=GRADER_BASE_IMAGE, working_directory=directory
    )
    resolved = host_requirements(declared)
    assert resolved.working_directory == (directory or "/workspace")
    assert _machine_spec(resolved, machine_runtime(backend="gvisor")).workdir == (directory or "/workspace")
    assert _machine_spec(declared, machine_runtime(backend="gvisor")).workdir == (directory or "/workspace")


def test_image_free_linux_process_is_rejected_before_starting_a_machine() -> None:
    with pytest.raises(ValueError, match="need a container image on Iris"):
        machine_backend(requirements(image=None, packages=("numpy",)), MachineHost.IRIS)


@pytest.mark.parametrize("memory_mb", [8192, 16384])
def test_policy_memory_and_actual_grader_timeout_reach_both_machine_specs(memory_mb: int) -> None:
    policy = load_policy(Path(__file__).parents[1] / "src/forge/policy/defaults")
    assert policy.setting("sandbox.memory_mb", int) == 8192
    policy = policy.overlaid({"sandbox.memory_mb": memory_mb})
    settings = lowering_settings(policy)
    package = ScriptGrader(
        argv=("true",),
        reward=StdoutReward(),
        environment=requirements(image=GRADER_BASE_IMAGE),
        answer_path=None,
        timeout=900,
    )
    task = arithmetic_task(grader=package, environment_requirements=requirements(image=GRADER_BASE_IMAGE))
    spec = lower_with_settings(task, settings, {"gvisor": AvailableFactory(Backend.GVISOR)})
    assert spec.task_machine is not None and spec.verifier_machine is not None
    assert spec.task_machine.memory_mb == spec.verifier_machine.memory_mb == memory_mb
    assert spec.task_machine.cpus == spec.verifier_machine.cpus == 1
    assert spec.limits.verifier_timeout == 1500
