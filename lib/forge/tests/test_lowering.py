# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

import pytest
from rolloutengine.models import RolloutLimits, TaskRolloutSpec
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineFactory, MachineSpec
from taskcompendium.models import AnswerType, PlainText, Source
from verifyit.spec import ExactSpec

from forge.lowering import answer_grader, assemble, lower, machine, requirements
from forge.resources.sandbox.factories import MachineHost


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
