# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

import pytest
from rolloutengine.models import RolloutLimits, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task, rollout_limits
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineFactory, MachineSpec
from taskcompendium.models import AnswerType, CommandSemantics, EnvironmentRequirements, PlainText, Source, StdoutReward
from verifyit.spec import ExactSpec

from forge.lowering import (
    GRADING_MACHINE,
    LoweringSettings,
    answer_grader,
    assemble,
    lower,
    lower_with_settings,
    machine,
    requirements,
    script_grader,
)
from forge.resources.sandbox.factories import MachineHost
from forge.resources.sandbox.images import GRADER_BASE_IMAGE


@dataclass(frozen=True, slots=True)
class AvailableFactory(MachineFactory):
    backend: Backend

    @override
    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        """Both container hosts support this fixture; no machine is started by lowering."""

    @override
    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError("Lowering must not create a machine")


@pytest.mark.parametrize("host,backend", [(MachineHost.LAPTOP, Backend.DOCKER), (MachineHost.IRIS, Backend.GVISOR)])
def test_lowering_binds_an_image_task_to_its_host_and_survives_engine_wire_format(
    host: MachineHost, backend: Backend
) -> None:
    image = "docker.io/example/task@sha256:" + "0" * 64
    task = assemble(
        "sum",
        "Return 1 + 1",
        AnswerType.TEXT,
        PlainText(),
        answer_grader(ExactSpec(expected=("2",))),
        Source(dataset="forge", revision="v1", row="1", importer_revision="v1"),
        environment=requirements(image=image),
    )
    lowered = lower(
        task,
        host=host,
        task_machine=machine(startup_timeout=30),
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
    "image,local",
    [
        (None, False),
        (GRADER_BASE_IMAGE, False),
        ("registry.example.com/python@sha256:" + "a" * 64, False),
        ("registry.example.com/python@sha256:" + "a" * 64, True),
    ],
)
def test_grader_image_precheck_respects_available_host_factories(image: str | None, local: bool) -> None:
    task = arithmetic_task(
        grader=script_grader(
            ("sh", "-c", "echo 1"), StdoutReward(), environment=requirements(image=image), answer_path=None, timeout=1
        ).grader
    )
    settings = LoweringSettings(MachineHost.IRIS, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    factories = {backend.value: AvailableFactory(backend) for backend in (Backend.GVISOR, Backend.SHELLSIM)}
    if local:
        factories[Backend.LOCAL.value] = AvailableFactory(Backend.LOCAL)
    if image not in (None, GRADER_BASE_IMAGE) and not local:
        with pytest.raises(ValueError, match="grader image must be") as raised:
            lower_with_settings(task, settings, factories)
        assert image in str(raised.value) and GRADER_BASE_IMAGE in str(raised.value)
    else:
        lowered = lower_with_settings(task, settings, factories)
        assert lowered.verifier_machine is not None
        assert lowered.verifier_machine.backend == (Backend.SHELLSIM if image is None else Backend.GVISOR)


def test_missing_machine_error_names_the_author_environment_fix() -> None:
    task = arithmetic_task(
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="lockfile-placeholder"
        )
    )
    settings = LoweringSettings(MachineHost.IRIS, GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    with pytest.raises(ValueError) as raised:
        lower_with_settings(task, settings, {})
    message = str(raised.value)
    assert 'set capabilities to ["shell"]' in message
    assert GRADER_BASE_IMAGE in message and "command_semantics shell_simulator" in message
    assert "candidate files require it" in message
    repaired = task.model_copy(update={"environment_requirements": requirements(image=None)})
    spec = lower_with_settings(repaired, settings, {"shellsim": AvailableFactory(Backend.SHELLSIM)})
    assert spec.task_machine is not None and spec.task_machine.backend == Backend.SHELLSIM
