# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fragment adapter preserves native grading and exports a private file grader."""

import sys
import tomllib
from pathlib import Path

import pytest
from taskcompendium.harbor.adapter import declared_execution_limits, write_harbor_task
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
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS
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
from experiments.post_training.task_curation.daytona_census import measure_envelopes
from experiments.post_training.task_curation.daytona_identity import native_task_toml
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


def test_repaired_fragment_exports_the_exact_oracle_and_a_separate_private_grader(tmp_path):
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
    assert declared_execution_limits(repaired) == {"agent_timeout_sec": 900.0, "verifier_timeout_sec": 720.0}
    with pytest.raises(ValueError, match="original agent_timeout_sec"):
        write_harbor_task(repaired, tmp_path / "shortened", INITIAL_REQUIREMENTS)
    requirements = INITIAL_REQUIREMENTS.model_copy(update={"agent_timeout_sec": 900.0, "verifier_timeout_sec": 720.0})
    directory = tmp_path / "preserved"
    write_harbor_task(repaired, directory, requirements)
    wrapper = tomllib.loads((directory / "task.toml").read_text())
    assert (wrapper["agent"]["timeout_sec"], wrapper["verifier"]["timeout_sec"]) == (900.0, 720.0)
    assert TaskSpec.model_validate_json((directory / "task-spec.json").read_text()) == repaired


def test_repaired_fragment_census_uses_declared_images_and_preserved_source_configuration():
    pytest.importorskip("harbor_config")
    original = original_task().model_copy(update={"tags": ("harbor",)})
    repaired = [
        repair_task(original, actor_image=image, grader_image=GRADER_IMAGE).task
        for image in (ACTOR_IMAGE, "registry.example/mirrored-actor@sha256:" + "a" * 64)
    ]
    documents = [(task.model_dump(mode="json"), {}) for task in repaired]

    measured = measure_envelopes(documents)

    assert measured.converted == measured.machine_slots == 2
    assert len(measured.snapshot_sizes) == 2
    assert measured.execution_failure_tasks == 0
    assert measured.actor_sources == {"docker_image": 2}
    assert measured.roles["grader"]["daytona"]["machine_slots"] == 2
    assert not measured.roles["grader"]["daytona"]["failures"]
    assert native_task_toml(documents[0][0]) == native_task_toml(original.model_dump(mode="json"))


@pytest.mark.parametrize("conflict", ["current_config", "explicit_limits"])
def test_repaired_archived_deadlines_reject_conflicting_limits(tmp_path, conflict):
    repaired = repair_task(original_task(), actor_image=ACTOR_IMAGE, grader_image=GRADER_IMAGE).task
    if conflict == "current_config":
        repaired = repaired.model_copy(
            update={
                "resources": repaired.resources.model_copy(
                    update={"all": (inline_resource("task.toml", b"[agent]\ntimeout_sec=300.0\n"),)}
                )
            }
        )
    else:
        repaired = repaired.model_copy(update={"execution_requirements": INITIAL_REQUIREMENTS})
    with pytest.raises(ValueError, match="conflicting original agent_timeout_sec"):
        write_harbor_task(repaired, tmp_path / "conflicting", INITIAL_REQUIREMENTS)


@pytest.mark.parametrize("provenance", ["missing", "different_version"])
def test_archived_limits_require_the_repair_provenance(tmp_path, provenance):
    repaired = repair_task(original_task(), actor_image=ACTOR_IMAGE, grader_image=GRADER_IMAGE).task
    if provenance == "missing":
        oracle = tuple(r for r in repaired.resources.oracle if r.path != "taskcompendium/crosscode-remediation.json")
        repaired = repaired.model_copy(update={"resources": repaired.resources.model_copy(update={"oracle": oracle})})
    else:
        repaired = repaired.model_copy(update={"labels": {**repaired.labels, "remediation_version": "other-repair"}})
    requirements = INITIAL_REQUIREMENTS.model_copy(update={"agent_timeout_sec": 900.0, "verifier_timeout_sec": 720.0})
    with pytest.raises(ValueError, match="repair provenance"):
        write_harbor_task(repaired, tmp_path / "unbound", requirements)


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
