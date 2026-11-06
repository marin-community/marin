# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest
from rolloutengine.machines import _machine_spec
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
    lower_with_settings,
    lowering_settings,
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


def test_policy_memory_and_actual_grader_timeout_reach_both_machine_specs() -> None:
    memory_mb = 16384
    policy = load_policy(Path(__file__).parents[1] / "src/forge/policy/defaults")
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
