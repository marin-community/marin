# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade complete rollouts in real bubblewrap sandboxes built from package locks."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable
from shellbox.machine import Backend
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import EnvironmentRequirements, ResourceGroups, ScriptGrader
from taskcompendium.runtime.resources import inline_resource

from .test_rollout import TWELVE, ReplayModel, arithmetic_task, engine, lowered, machine_runtime

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
    (tmp_path / ".artifact.json").write_text(
        json.dumps({"result": {"lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "data": []}})
    )
    return EnvironmentRequirements(
        compatible_backends=(Backend.LOCAL,),
        packages_lock=str(lock),
        setup_commands=("printf ready > /app/ready",),
        environment_variables={"GRADER_SETTING": "configured"},
    )


@pytest.mark.timeout(180)
@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
@pytest.mark.parametrize("grader_kind", ["script", "verifyit"])
async def test_rollout_grades_answer_in_lock_environment(
    local_grading_factory, lock_environment, grader_kind, answer, reward
):
    if grader_kind == "script":
        grader = ScriptGrader(environment=lock_environment, argv=("python3", "/tests/grade.py"))
        resources = ResourceGroups(verifier=(inline_resource("grade.py", SCRIPT),))
    else:
        package = verifyit_package(TWELVE)
        grader = package.grader.model_copy(update={"environment": lock_environment})
        resources = ResourceGroups(verifier=package.resources)
    task = arithmetic_task(grader=grader, resources=resources)
    model = ReplayModel([{"role": "assistant", "content": answer}])
    result = await engine(model, {"local": local_grading_factory}).run(
        lowered(task, verifier_machine=machine_runtime(startup_timeout=120), verifier_timeout=150)
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward), result.grade.error
    assert result.response_token_ids == (20,)
    assert "grade.py" not in json.dumps(model.requests[0].messages)
