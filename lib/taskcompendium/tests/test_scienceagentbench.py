# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import DockerImage, MachineSpec

from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.scienceagentbench import (
    KnowledgeCondition,
    VerifiedAnnotation,
    actor_context,
    read_verified_annotations,
    scienceagentbench_task,
)
from taskcompendium.importers.scienceagentbench_grading import deterministic_grader
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    GradingAttempt,
    TaskSpec,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import grade_task

from .grading_machines import GRADER_IMAGE, REPLAY_IMAGE, ReplayMachines, finished


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
            GradingAttempt(finished(task), {"/testbed/solution.py": program}),
            machine_factory=machines,
            machine_spec=MachineSpec(DockerImage(GRADER_IMAGE)),
        )
        assert (result.status, result.reward) == (Outcome.GRADED, expected), result
    assert all(machine.closed for machine in machines.machines)
