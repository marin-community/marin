# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fragment adapter preserves native grading and exports a private file grader."""

import sys
import tomllib
from pathlib import Path

import pytest
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.grade import Status
from verifyit.modes.grade_script import grade
from verifyit.spec import ScriptSpec

from experiments.post_training.task_curation.crosscode_python import (
    GRADER_PATH,
    SOURCE,
    VERDICT_FILE,
    repair_task,
    structured_grader,
)
from experiments.post_training.task_curation.tasktrove.harbor_export import harbor_payload

NATIVE_SCRIPT = Path(__file__).parent / "fixtures/crosscode-native-test.sh"
ACTOR_IMAGE = "registry.example/crosscode-actor@sha256:" + "a" * 64
GRADER_IMAGE = "registry.example/crosscode-grader@sha256:" + "b" * 64


def original_task() -> TaskSpec:
    return TaskSpec(
        id="crosscode-fragment",
        context=ConversationInput(
            events=(TextMessage(role="user", content="Complete target(). Write /app/solution.txt."),)
        ),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM python:3.10-slim\n"),)),
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="Native Harbor verifier"),
        source=Source(
            dataset="open-thoughts/TaskTrove",
            revision="original-pin",
            row=SOURCE + "/tasks.parquet:0",
            importer_revision="original-converter",
        ),
        resources=ResourceGroups(
            oracle=(inline_resource("solution/solution.py", b"target()"),),
            verifier=(
                inline_resource("tests/test.sh", NATIVE_SCRIPT.read_bytes()),
                inline_resource("tests/solution.txt", b"target()\n"),
                inline_resource("task.toml", b"[agent]\ntimeout_sec=900.0\n[verifier]\ntimeout_sec=720.0\n"),
            ),
        ),
    )


def test_repaired_fragment_exports_the_exact_oracle_and_a_separate_private_grader():
    original = original_task()
    repaired = repair_task(original, actor_image=ACTOR_IMAGE, grader_image=GRADER_IMAGE).task
    payload = harbor_payload(
        {"task_json": repaired.model_dump_json(), "source_row": SOURCE + "/0", "original_path": "crosscode-fragment"},
        grader_image=GRADER_IMAGE,
        family="crosscode",
    )
    assert payload.solution["solution/solution.py"] == b"target()"
    assert payload.files["tests/solution.txt"] == b"target()\n"
    config = tomllib.loads(payload.files["task.toml"].decode())
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["artifacts"] == [{"source": "/app/solution.txt", "destination": "app/solution.txt"}]
    assert payload.files["environment/Dockerfile"] == f"FROM {ACTOR_IMAGE}\n".encode()
    assert all("tests/" not in name for name in payload.files if name.startswith("environment/"))
    archived = {r.path: resource_bytes(r) for r in repaired.resources.oracle}
    assert archived["source_archive/tests/test.sh"] == NATIVE_SCRIPT.read_bytes()
    assert archived["source_archive/tests/solution.txt"] == b"target()\n"


@pytest.mark.skipif(sys.platform != "linux", reason="The archived comparator requires GNU sed and tac")
@pytest.mark.parametrize(
    "candidate,reference,reward,tier",
    [
        (b"target()", b"target()\n", 1.0, "1/exact"),
        (b"\n \t\ntarget()  \r\n\t\n", b"target()", 1.0, "1/exact"),
        (b"target()\nextra\nmore\nlast\n", b"target()", 1.0, "2/startswith"),
        (b"target anything entirely wrong", b"target()", 1.0, "3/leading-identifier"),
        (b"y" * 399, b"target()", 0.25, "4/non-empty-partial"),
        (b"y" * 400, b"target()", 0.0, ""),
        (b"123 + ---", b"target()", 0.0, ""),
        (b"", b"target()", 0.0, ""),
        (b"", b"", 1.0, "1/exact"),
    ],
)
def test_structured_producer_preserves_the_four_tier_rewards(tmp_path, candidate, reference, reward, tier):
    tests = tmp_path / "tests"
    workspace = tmp_path / "app"
    tests.mkdir()
    workspace.mkdir()
    (tests / GRADER_PATH).write_bytes(structured_grader(NATIVE_SCRIPT.read_bytes()))
    (tests / "solution.txt").write_bytes(reference)
    (workspace / "solution.txt").write_bytes(candidate)
    result = grade(ScriptSpec(path=GRADER_PATH, workspace=str(workspace), verdict_file=VERDICT_FILE), tests, workspace)
    assert result.status is Status.SCORED
    assert result.reward == reward
    assert result.detail["tier"] == tier
    assert result.detail["total"] == 1


def test_missing_reference_is_unscored_but_a_missing_submission_is_a_completed_negative(tmp_path):
    tests = tmp_path / "tests"
    workspace = tmp_path / "app"
    tests.mkdir()
    workspace.mkdir()
    (tests / GRADER_PATH).write_bytes(structured_grader(NATIVE_SCRIPT.read_bytes()))
    spec = ScriptSpec(path=GRADER_PATH, workspace=str(workspace), verdict_file=VERDICT_FILE)
    invalid = grade(spec, tests, workspace)
    assert invalid.status is Status.INVALID_TASK
    (tests / "solution.txt").write_bytes(b"target()")
    negative = grade(spec, tests, workspace)
    assert negative.status is Status.SCORED
    assert negative.reward == 0
    assert negative.detail["reason"] == "missing_submission"
    assert negative.detail["total"] == 1
