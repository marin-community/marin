# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import DockerImage, MachineSpec
from verifyit.modes.grade_judge import JudgeConnection

from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.scienceagentbench import (
    KnowledgeCondition,
    VerifiedAnnotation,
    actor_context,
    read_verified_annotations,
    scienceagentbench_task,
)
from taskcompendium.importers.scienceagentbench_assets.release import copy_member
from taskcompendium.importers.scienceagentbench_grading import deterministic_grader, visual_grader
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    GradingAttempt,
    PrivateVerifierResource,
    ResourceGroups,
    ScienceAgentBenchVerifierFile,
    TaskSpec,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import grade_task

from .grading_machines import GRADER_IMAGE, REPLAY_IMAGE, ReplayMachines, completed_trace


def test_release_member_rejects_changed_publisher_data_with_the_same_length():
    member = "benchmark/datasets/example/example.csv"
    expected = b"direction,score\nleft,10\n"
    modified = expected.replace(b"10", b"11")
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(member, modified)
    with zipfile.ZipFile(archive_bytes) as archive, pytest.raises(ValueError, match="Unexpected contents"):
        copy_member(archive, member, hashlib.sha256(expected).hexdigest(), io.BytesIO(), len(expected))


def test_verifier_publisher_member_cannot_be_deserialized_as_a_worker_input():
    private = PrivateVerifierResource(
        path="gold.csv",
        source=ScienceAgentBenchVerifierFile(
            member="benchmark/eval_programs/gold_results/gold.csv", sha256="a" * 64, size=42
        ),
    )
    serialized = private.model_dump(mode="json")
    assert ResourceGroups.model_validate({"verifier": [serialized]}).verifier == (private,)
    for role in ("worker", "all", "oracle"):
        with pytest.raises(ValueError):
            ResourceGroups.model_validate({role: [serialized]})


@pytest.fixture
def annotation():
    return VerifiedAnnotation(
        instance_id=1,
        domain="synthetic-test-domain",
        subtask_categories="regression",
        github_name="private-source-repository",
        task_inst="Fit the observations and save predictions in pred_results/output.csv.",
        domain_knowledge="EXPERT_ONLY: the response is quadratic.",
        dataset_folder_tree="|-- observations/\n|---- train.csv\n|---- test.csv",
        dataset_preview="column_a,column_b\n" + "0,1\n" * 1000 + "LAST_PREVIEW_ROW",
        src_file_or_path="private-source-program.py",
        gold_program_name="private-reference.py",
        output_fname="pred_results/output.csv",
        eval_script_name="private-evaluator.py",
    )


def test_paired_inputs_change_only_expert_knowledge_and_preserve_full_preview(annotation):
    off = actor_context(annotation, KnowledgeCondition.OFF).model_dump_json()
    on = actor_context(annotation, KnowledgeCondition.ON).model_dump_json()
    assert annotation.domain_knowledge not in off
    assert annotation.domain_knowledge in on
    for private in (
        annotation.github_name,
        annotation.src_file_or_path,
        annotation.gold_program_name,
        annotation.eval_script_name,
    ):
        assert private not in off
        assert private not in on
    off_text = actor_context(annotation, KnowledgeCondition.OFF).events[0].content
    on_text = actor_context(annotation, KnowledgeCondition.ON).events[0].content
    assert on_text.replace(f"## Expert knowledge\n\n{annotation.domain_knowledge}\n\n", "") == off_text
    assert annotation.task_inst in off_text
    assert annotation.dataset_preview in off_text
    assert "benchmark/datasets/observations" in off_text


def test_null_optional_inputs_do_not_render_as_literal_none(annotation):
    without_optional = annotation.model_copy(update={"domain_knowledge": None, "dataset_preview": None})
    on = actor_context(without_optional, KnowledgeCondition.ON)
    off = actor_context(without_optional, KnowledgeCondition.OFF)
    assert on == off
    assert "None" not in on.events[0].content
    assert annotation.task_inst in on.events[0].content


def test_panel_reader_preserves_source_order_and_nulls(tmp_path, annotation):
    rows = [annotation.model_copy(update={"instance_id": i}).model_dump() for i in range(102, 0, -1)]
    rows[0]["domain_knowledge"] = None
    path = tmp_path / "verified.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    loaded = list(read_verified_annotations(StoragePath(str(path)), ConversionContext({}, None)))
    assert loaded == rows


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unexpected"])
def test_incomplete_panel_yields_nothing(tmp_path, annotation, corruption):
    rows = [annotation.model_copy(update={"instance_id": i}).model_dump() for i in range(1, 103)]
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["instance_id"] = 103
    path = tmp_path / "verified.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    reader = read_verified_annotations(StoragePath(str(path)), ConversionContext({}, None))
    with pytest.raises(ValueError):
        next(reader)


@pytest.mark.parametrize("condition", list(KnowledgeCondition))
def test_converted_task_grades_replayed_program_and_keeps_reference_private(tmp_path, annotation, condition):
    environment = EnvironmentRequirements(
        docker_image=REPLAY_IMAGE,
        command_semantics=CommandSemantics.LINUX_PROCESS,
        working_directory="/testbed",
        capabilities=("shell", "filesystem"),
    )
    script = b"""
from pathlib import Path
def eval():
    assert not Path('../controls/reference.py').exists()
    assert not Path('solution.py').exists()
    return int(Path('pred_results/output.csv').read_text() == Path('../tests/gold.csv').read_text()), 'match'
"""
    package = deterministic_grader(
        annotation,
        environment=EnvironmentRequirements(
            docker_image=GRADER_IMAGE,
            command_semantics=CommandSemantics.LINUX_PROCESS,
            working_directory="/testbed",
        ),
        replay_environment=environment,
        evaluator_program=script,
        replay_timeout=10,
        verifier_timeout=30,
    )
    reference = b"PRIVATE_REFERENCE_PROGRAM"
    task = scienceagentbench_task(
        annotation,
        condition,
        environment=environment,
        package=GraderPackage(package.grader, (*package.resources, inline_resource("gold.csv", b"42"))),
        reference_program=reference,
        importer_revision="test-converter",
    )
    task = TaskSpec.model_validate_json(task.model_dump_json())
    assert reference.decode() not in task.context.model_dump_json()
    machines = ReplayMachines(tmp_path / "machines")
    for prediction, expected in (("42", 1.0), ("wrong", 0.0)):
        program = f"""
from pathlib import Path
assert not Path('../controls/reference.py').exists()
assert not Path('../tests/gold.csv').exists()
Path('pred_results').mkdir()
Path('pred_results/output.csv').write_text({prediction!r})
""".encode()
        result = grade_task(
            task,
            GradingAttempt(completed_trace(task), {"/testbed/solution.py": program}),
            machine_factory=machines,
            machine_spec=MachineSpec(DockerImage(GRADER_IMAGE)),
        )
        assert (result.status, result.reward) == (Outcome.GRADED, expected), result
    assert all(machine.closed for machine in machines.machines)


@pytest.mark.parametrize(
    "reply,status,expected",
    [
        ("[FINAL SCORE]: 90", 200, 1.0),
        ("no marker", 200, 0.0),
        ("[FINAL SCORE]: 90", 503, None),
    ],
)
def test_visual_package_replays_then_grades_both_figures(tmp_path, annotation, judge_server, reply, status, expected):
    judge_server.reply = reply
    judge_server.status_code = status
    replay = EnvironmentRequirements(
        docker_image=REPLAY_IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS, working_directory="/testbed"
    )
    evaluator = b"""
from gpt4_visual_judge import encode_image, score_figure

def eval():
    results = []
    responses = []
    for figure in ('one', 'two'):
        texts, score = score_figure(
            encode_image(f'pred_results/{figure}.png'), encode_image(f'../tests/gold/{figure}.png')
        )
        results.append(score >= 60)
        responses.extend(texts)
    return int(all(results)), responses
"""
    package = visual_grader(
        annotation.model_copy(update={"output_fname": "pred_results/one.png"}),
        environment=replay.model_copy(update={"docker_image": GRADER_IMAGE}),
        replay_environment=replay,
        evaluator_program=evaluator,
        judge_prompt="pinned prompt fixture",
        replay_timeout=10,
        verifier_timeout=30,
    )
    package = GraderPackage(
        package.grader,
        (*package.resources, inline_resource("gold/one.png", b"gold one"), inline_resource("gold/two.png", b"gold two")),
    )
    task = scienceagentbench_task(
        annotation.model_copy(update={"output_fname": "pred_results/one.png"}),
        condition=KnowledgeCondition.OFF,
        environment=replay,
        package=package,
        reference_program=b"reference fixture",
        importer_revision="test-visual",
    )
    program = b"""
import os
from pathlib import Path
assert 'VERIFYIT_JUDGE_API_KEY' not in os.environ
assert not Path('../tests/gold/one.png').exists()
Path('pred_results').mkdir()
Path('pred_results/one.png').write_bytes(b'candidate one')
Path('pred_results/two.png').write_bytes(b'candidate two')
"""
    machines = ReplayMachines(tmp_path / "machines")
    result = grade_task(
        TaskSpec.model_validate_json(task.model_dump_json()),
        GradingAttempt(completed_trace(task), {"/testbed/solution.py": program}),
        machine_factory=machines,
        machine_spec=MachineSpec(DockerImage(GRADER_IMAGE)),
        judge_connection=JudgeConnection(f"http://127.0.0.1:{judge_server.server_port}/v1", "judge-fixture-key"),
    )
    assert (result.status, result.reward) == (
        Outcome.INFRA_ERROR if expected is None else Outcome.GRADED,
        expected,
    ), result
    if expected is not None:
        assert result.detail is not None
        assert len(result.detail["judge_receipts"]) == 6
    assert len(judge_server.requests) == (1 if expected is None else 6)
    for request in judge_server.requests:
        assert request["n"] == 1
        assert request["model"] == "MiniMaxAI/MiniMax-M3-MXFP8"
        assert "provider" not in request
    assert len(machines.machines) == 2 and all(machine.closed for machine in machines.machines)
    assert "VERIFYIT_JUDGE_API_KEY" not in machines.machines[0].env
    assert machines.machines[1].env["VERIFYIT_JUDGE_API_KEY"] == "judge-fixture-key"
