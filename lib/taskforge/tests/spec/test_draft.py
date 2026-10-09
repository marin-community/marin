# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from shellbox.machine import Backend
from taskcompendium.models import AnswerType, PlainText, Source
from verifyit.spec import NumericSpec

from taskforge.sandbox.factories import MachineHost, machine_factories
from taskforge.spec.draft import answer_grader, assemble, file, lower, machine, requirements, session

SESSION = session(
    max_turns=4,
    model_turn_timeout=None,
    command_timeout=None,
    tool_turn_timeout=None,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=60.0,
    cleanup_timeout=30.0,
)


def task(image: str | None):
    return assemble(
        "multiply",
        "Multiply the two numbers in /workspace/numbers.txt.",
        AnswerType.TEXT,
        PlainText(),
        answer_grader(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0)),
        Source(dataset="test", revision="r1", row="0", importer_revision="test"),
        environment=requirements(image=image),
        files=(file("workspace/numbers.txt", "6 7\n"),),
    )


def test_an_image_less_task_lowers_onto_shellsim_and_an_image_task_has_no_factory_here(tmp_path):
    factories = machine_factories(MachineHost.LAPTOP, None, tmp_path)
    settings = machine(startup_timeout=60.0)

    lowered = lower(
        task(None),
        host=MachineHost.LAPTOP,
        task_machine=settings,
        verifier_machine=None,
        session=SESSION,
        factories=factories,
    )

    assert lowered.runtime.task_machine is not None
    assert lowered.runtime.task_machine.backend == Backend.SHELLSIM.value
    with pytest.raises(ValueError):
        lower(
            task("registry.example/task@sha256:" + "0" * 64),
            host=MachineHost.LAPTOP,
            task_machine=settings,
            verifier_machine=None,
            session=SESSION,
            factories=factories,
        )
