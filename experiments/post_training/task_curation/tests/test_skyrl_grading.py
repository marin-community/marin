# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL grade scripts score replies with the vendored scorers in the grader image.

Each test converts a fixture row and grades it the way a campaign does, in a fresh container of the
locally built grader image, which stands in for the task's pinned grader image. Build it from the
repository root with::

    docker build --platform linux/amd64 --build-context verifyit=lib/verifyit/src/verifyit \\
        -t local/task-curation-grader:test experiments/post_training/task_curation/images/grader
"""

import asyncio
import json
import shutil
import subprocess
from dataclasses import replace
from typing import Any

import pytest
from shellbox.backends.docker.machine import DockerMachine, DockerMachineFactory
from shellbox.machine import Backend, DockerImage, MachineSpec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.tests.conversion import converted_task
from experiments.post_training.task_curation.tests.test_skyrl import PIPELINES, ROWS, SUM_SOLUTION

pytestmark = pytest.mark.docker

GRADER_IMAGE = "local/task-curation-grader:test"
GRADING_MEMORY_MB = 5120
NOISY_ADD = '```python\ndef add(a, b):\n    print("x" * 20000)\n    return a + b\n```'
"""A correct function that prints more than the runtime keeps of a grader's stdout."""


class LocalGraderImage:
    """Starts the locally built grader image whatever grader image the task pins."""

    backend = Backend.DOCKER

    async def create(self, spec: MachineSpec) -> DockerMachine:
        return await DockerMachineFactory().create(replace(spec, source=DockerImage(GRADER_IMAGE)))


class LocalGradingMachines:
    """Grading machines from the local Docker daemon, requested the way a campaign's controls request them."""

    def identity(self) -> dict[str, Any]:
        return {"backend": Backend.DOCKER}

    def machine(self, image: str, memory_mb: int) -> tuple[LocalGraderImage, MachineSpec]:
        return LocalGraderImage(), MachineSpec(DockerImage(image), memory_mb=memory_mb)


@pytest.fixture(scope="module")
def machines() -> LocalGradingMachines:
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")
    inspected = subprocess.run(["docker", "image", "inspect", GRADER_IMAGE], capture_output=True, check=False)
    if inspected.returncode != 0:
        pytest.skip(f"{GRADER_IMAGE} is not built; see this module's docstring")
    return LocalGradingMachines()


def grade(task: TaskSpec, reply: str, machines: LocalGradingMachines) -> GradeResult:
    assert task.grader.environment is not None and task.grader.environment.docker_image is not None
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=reply)))
    )
    machine = machines.machine(task.grader.environment.docker_image, GRADING_MEMORY_MB)
    return asyncio.run(grade_in_sandbox(task, attempt, *machine))


@pytest.mark.parametrize(
    ("name", "golden"),
    [
        ("apps", CheckStatus.PASS),
        ("eurus2_code", CheckStatus.SKIPPED),
        ("verifiable_code", CheckStatus.PASS),
        ("gretel_text_to_sql", CheckStatus.PASS),
        ("nemotron_if", CheckStatus.SKIPPED),
        ("rlvr_ifeval", CheckStatus.SKIPPED),
    ],
)
def test_declared_controls_pass_in_the_grader_image(name, golden, machines):
    """An empty and a wrong reply score 0, and the source's known solution, where it has one, scores 1."""
    pipeline = PIPELINES[name]
    assert pipeline.controls is not None
    report = run_controls(converted_task(pipeline, ROWS[name]), controls=pipeline.controls, machines=machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"empty": CheckStatus.PASS, "golden": golden, "negative": CheckStatus.PASS}, report.checks


@pytest.mark.parametrize(
    ("name", "scorer"),
    [
        ("apps", "apps_testing_util.py"),
        ("verifiable_code", "livecodebench.py"),
        ("gretel_text_to_sql", "text_to_sql_scoring.py"),
        ("rlvr_ifeval", "ifeval_utils.py"),
    ],
)
def test_grade_script_fails_rather_than_scoring_when_its_scorer_cannot_import(name, scorer, machines):
    task = converted_task(PIPELINES[name], ROWS[name])
    broken = inline_resource(scorer, b"import package_missing_from_the_grader_image\n")
    verifier = tuple(broken if resource.path == scorer else resource for resource in task.resources.verifier)
    task = task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": verifier})})
    result = grade(task, f"```python\n{SUM_SOLUTION}\n```", machines)
    assert result.status == Outcome.INFRA_ERROR
    assert result.diagnostics is not None and result.diagnostics["exit_code"] != 0


@pytest.mark.parametrize(
    ("reply", "reward"),
    [("a calm haiku about rain.", 1.0), ("a calm haiku, about rain.", 0.5), ("A calm haiku, about rain.", 0.0)],
)
def test_ifeval_rewards_the_fraction_of_constraints_a_reply_meets(reply, reward, machines):
    row = {
        **ROWS["nemotron_if"],
        "args": {
            "instruction_id_list": ["change_case:english_lowercase", "punctuation:no_comma"],
            "instruction_kwargs": [{}, {}],
        },
    }
    result = grade(converted_task(PIPELINES["nemotron_if"], row), reply, machines)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize(
    ("reply", "reward"),
    [
        (NOISY_ADD, 1.0),
        ("```python\ndef add(a, b):\n    raise SystemExit(0)\n```", 0.0),
    ],
)
def test_apps_reward_survives_a_program_that_floods_stdout_or_exits(reply, reward, machines):
    row = {**ROWS["apps"], "input_output": json.dumps({"inputs": [[1, 2]], "outputs": [3], "fn_name": "add"})}
    result = grade(converted_task(PIPELINES["apps"], row), reply, machines)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


def test_lcb_reward_survives_a_program_that_floods_stdout(machines):
    cases = [{"type": "functional", "fn_name": "add", "input": [1, 2], "output": 3}]
    row = {**ROWS["verifiable_code"], "verification_info": {"language": "python", "test_cases": cases}}
    result = grade(converted_task(PIPELINES["verifiable_code"], row), NOISY_ADD, machines)
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
