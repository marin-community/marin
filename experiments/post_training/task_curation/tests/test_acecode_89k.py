# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ScriptGrader, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import acecode_89k

NATIVE = {
    "id": "acecode-0001",
    "source": "bigcode_python_fns",
    "question": "Implement a function that simulates a simple staking mechanism for token deposits.",
    "test_cases": [
        "assert staking_simulation([(1500, 1)], [1500], 1) == [1500]",
        "assert staking_simulation([(1000, 2)], [1000], 1) == [0]",
    ],
    "inferences": [{"completion": "def staking_simulation(deposits, withdraws, time_passed):\n    return [1500]"}],
    "context_messages": [{"role": "user", "content": "task"}],
}


def converted(native):
    row = RawRow(
        id="original-acecode-row",
        source=Source(dataset=acecode_89k.REPO, revision=acecode_89k.REVISION, row="fixture", importer_revision="test"),
        data=native,
    )
    result = acecode_89k.convert_problem(
        row,
        ConversionContext(
            {},
            EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=acecode_89k.GRADER_IMAGE
            ),
        ),
    )
    assert isinstance(result, NormalizedTask)
    return result


def rejected(native):
    row = RawRow(
        id="original-acecode-row",
        source=Source(dataset=acecode_89k.REPO, revision=acecode_89k.REVISION, row="fixture", importer_revision="test"),
        data=native,
    )
    result = acecode_89k.convert_problem(
        row,
        ConversionContext(
            {},
            EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=acecode_89k.GRADER_IMAGE
            ),
        ),
    )
    assert isinstance(result, ImportRejection)
    return result


def test_context_is_the_question_with_the_completion_instruction_recorded():
    result = converted(NATIVE)
    task = result.task
    (user,) = task.context.events
    assert user.role == "user" and user.content == NATIVE["question"] + acecode_89k.COMPLETION_INSTRUCTION
    assert len(result.changes) == 1 and "completion format instruction" in result.changes[0].reason
    assert task.labels["native_source"] == "bigcode_python_fns"
    assert not task.resources.worker and not task.resources.oracle


def test_the_assert_cases_are_the_grader_and_the_inferences_stay_out():
    task = converted(NATIVE).task
    assert isinstance(task.grader, ScriptGrader) and task.grader.judge is None
    config = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    payload = json.loads(resource_bytes(config))
    assert payload["test_cases"] == NATIVE["test_cases"]
    assert "inferences" not in json.dumps(payload)
    grade = next(resource for resource in task.resources.verifier if resource.path == "grade.py")
    script = resource_bytes(grade).decode()
    assert "exec(case" in script and "print(1)" in script


def test_contract_defects_fail_closed():
    assert rejected({**NATIVE, "question": ""}).reason.startswith("incomplete_acecode_contract")
    assert rejected({**NATIVE, "test_cases": []}).reason.startswith("invalid_test_cases")
    assert rejected({**NATIVE, "test_cases": ["x = 1"]}).reason.startswith("non_assert_test_case")
    assert rejected({**NATIVE, "id": ""}).reason.startswith("invalid_acecode_metadata")
    assert rejected({"question": NATIVE["question"]}).reason.startswith("incomplete_acecode_contract")


def test_source_module_binds_the_sealed_population():
    source = acecode_89k.source(tuple(f"data/train-{i:05d}-of-00011.parquet" for i in range(11)))
    assert source.info.id == acecode_89k.SOURCE_ID
    assert source.info.count == 87149
    assert source.info.origin.value == "Tiger-lab"
    assert source.config.source.revision == acecode_89k.REVISION
    assert "assert cases" in source.info.notes and "private provenance" in source.info.notes
