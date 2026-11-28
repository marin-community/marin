# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade complete rollouts in real bubblewrap sandboxes built from package locks."""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable
from shellbox.machine import Backend, Machine, MachineFactory, MachineSpec, TaskRequirements
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.evalchemy.code import convert_python
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    GraderMachine,
    ResourceGroups,
    ScriptGrader,
    Source,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource

from rolloutengine.test_support import TWELVE, ReplayModel, arithmetic_task, engine, machine_runtime, rollout_spec

SCRIPT = b"""
import os
from pathlib import Path
import six
from verifyit.numeric import numeric_literal

assert isinstance(Path('/app/ready').read_text(), six.string_types)
assert os.environ['GRADER_SETTING'] == 'configured'
assert os.environ['PYTHONHASHSEED'] == '17'
print(float(numeric_literal(Path('/app/answer.txt').read_text()) == 12))
"""


@dataclass(frozen=True)
class DelegatingLocalFactory:
    base: MachineFactory

    @property
    def backend(self) -> Backend:
        return Backend.LOCAL

    def check_environment(self, requirements: TaskRequirements, spec: MachineSpec) -> None:
        self.base.check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        return await self.base.create(replace(spec, env={**spec.env, "WRAPPER_MARKER": "wrapped"}))


@pytest.fixture
def local_grading_factory():
    if os.geteuid() != 0:
        pytest.skip("Local grader staging requires a root host process")
    try:
        return LocalMachineFactory(hash_seed="17")
    except SandboxUnavailable as error:
        pytest.skip(str(error))


@pytest.fixture
def lock_environment(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_bytes((Path(__file__).parent / "fixtures" / "local-grader.lock").read_bytes())
    return EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        packages_lock=str(lock),
        setup_commands=("printf ready > /app/ready",),
        environment_variables={"GRADER_SETTING": "configured"},
    )


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "grader_kind,answer,reward",
    [
        ("script", "12", 1.0),
        ("script", "13", 0.0),
        ("verifyit", "12", 1.0),
        ("verifyit", "13", 0.0),
        ("humanevalplus", "def identity(value):\n    return value", 1.0),
        ("humanevalplus", "def identity(value):\n    return 0", 0.0),
        ("mbppplus", "def identity(value):\n    return value", 1.0),
        ("mbppplus", "def identity(value):\n    return 0", 0.0),
    ],
)
async def test_rollout_grades_answer_in_lock_environment(
    local_grading_factory, lock_environment, grader_kind, answer, reward
):
    if grader_kind == "script":
        grader = ScriptGrader(placement=GraderMachine(environment=lock_environment), argv=("python3", "/tests/grade.py"))
        resources = ResourceGroups(verifier=(inline_resource("grade.py", SCRIPT),))
        task = arithmetic_task(grader=grader, resources=resources)
    elif grader_kind == "verifyit":
        package = verifyit_package(TWELVE)
        grader = package.grader.model_copy(update={"placement": GraderMachine(environment=lock_environment)})
        resources = ResourceGroups(verifier=package.resources)
        task = arithmetic_task(grader=grader, resources=resources)
    else:
        problem = {
            "prompt": "def identity(value):\n    ",
            "evaluation_prompt": "Return the input unchanged in a function named identity.",
            "test": (
                "def check(candidate):\n    assert candidate(7) == 7\n    assert candidate(-3) == -3"
                if grader_kind == "humanevalplus"
                else "assert identity(7) == 7\nassert identity(-3) == -3"
            ),
            "entry_point": "identity",
        }
        row = RawRow(
            "identity", Source(dataset=grader_kind, revision="fixture", row="0", importer_revision="1"), problem
        )
        task = convert_python(row, ConversionContext(inputs={}, grader_environment=lock_environment), suite=grader_kind)
    model = ReplayModel([{"role": "assistant", "content": answer}])
    result = await engine(model, {"local": local_grading_factory}).run_task(
        rollout_spec(task, verifier_machine=machine_runtime(startup_timeout=120), verifier_timeout=150)
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward), result.grade
    assert result.rollout.conversations[-1].response_token_ids == (20,)
    assert "grade.py" not in json.dumps(model.requests[0].messages)


@pytest.mark.timeout(180)
async def test_wrapped_local_factory_grades_lock_environment(local_grading_factory, lock_environment):
    grader = ScriptGrader(placement=GraderMachine(environment=lock_environment), argv=("python3", "/tests/grade.py"))
    script = b"import os\nassert os.environ['WRAPPER_MARKER'] == 'wrapped'\n" + SCRIPT
    task = arithmetic_task(grader=grader, resources=ResourceGroups(verifier=(inline_resource("grade.py", script),)))
    model = ReplayModel([{"role": "assistant", "content": "12"}])
    factory = DelegatingLocalFactory(local_grading_factory)

    result = await engine(model, {"local": factory}).run_task(
        rollout_spec(task, verifier_machine=machine_runtime(startup_timeout=120), verifier_timeout=150)
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0), result.grade.error
