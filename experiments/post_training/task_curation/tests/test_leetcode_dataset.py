# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ScriptGrader, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import leetcode_dataset

NATIVE = json.loads((Path(__file__).parent / "fixtures/leetcode_problem.json").read_text())


def converted(native):
    row = RawRow(
        id="original-leetcode-row",
        source=Source(
            dataset=leetcode_dataset.REPO, revision=leetcode_dataset.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = leetcode_dataset.convert_problem(
        row,
        ConversionContext(
            {},
            EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=leetcode_dataset.GRADER_IMAGE
            ),
        ),
    )
    assert isinstance(result, NormalizedTask)
    return result


def rejected(native):
    row = RawRow(
        id="original-leetcode-row",
        source=Source(
            dataset=leetcode_dataset.REPO, revision=leetcode_dataset.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = leetcode_dataset.convert_problem(
        row,
        ConversionContext(
            {},
            EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=leetcode_dataset.GRADER_IMAGE
            ),
        ),
    )
    assert isinstance(result, ImportRejection)
    return result


def grader_config(task):
    config = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    return json.loads(resource_bytes(config))


def test_context_is_the_native_query_framing_with_the_completion_instruction_recorded():
    result = converted(NATIVE)
    task = result.task
    (user,) = task.context.events
    assert user.role == "user" and user.content == NATIVE["query"] + leetcode_dataset.COMPLETION_INSTRUCTION
    assert NATIVE["problem_description"] in user.content  # the framing carries the actual problem
    assert len(result.changes) == 1 and "completion format instruction" in result.changes[0].reason
    assert task.labels["native_task_id"] == "two-sum" and task.labels["difficulty"] == "Easy"
    assert not task.resources.worker and not task.resources.oracle


def test_the_source_harness_is_the_grader_and_the_reference_stays_out():
    task = converted(NATIVE).task
    assert isinstance(task.grader, ScriptGrader) and task.grader.judge is None
    config = grader_config(task)
    assert config["preamble"] == NATIVE["prompt"]
    assert config["starter_code"] == NATIVE["starter_code"]  # provenance; the runner never execs it
    assert config["entry_point"] == "Solution().twoSum"
    assert config["test_source"] == NATIVE["test"]
    assert "reference solution" not in json.dumps(config)
    grade = next(resource for resource in task.resources.verifier if resource.path == "grade.py")
    script = resource_bytes(grade).decode()
    assert "harness_reward" in script and "/tests/config.json" in script


def test_context_carries_the_problem_statement():
    """Coordinator-required regression: the context must carry the actual problem statement.

    The v1 converter bound the boilerplate prompt field, which contains the problem in 0/2,869
    native rows; this test fails if the context ever loses the problem statement again.
    """
    (user,) = converted(NATIVE).task.context.events
    assert NATIVE["problem_description"] in user.content
    assert "Given an array of integers" in user.content


def test_the_reference_qualification_rejects_a_broken_harness():
    broken = {**NATIVE, "test": "def check(candidate):\n    this is not python ==="}
    result = rejected(broken)
    assert result.reason.startswith("harness_defect")
    assert "two-sum" in leetcode_dataset.REFERENCE_QUALIFICATION_EVIDENCE


def test_the_reference_qualification_rejects_when_the_reference_fails():
    failing = {**NATIVE, "test": NATIVE["test"] + "\n    assert candidate([1, 2], 3) == [99]\n"}
    result = rejected(failing)
    assert result.reason.startswith("harness_rejects_reference_solution")
    assert "two-sum" in leetcode_dataset.REFERENCE_QUALIFICATION_EVIDENCE


def test_contract_defects_fail_closed():
    assert rejected({**NATIVE, "query": ""}).reason.startswith("incomplete_leetcode_contract")
    assert rejected({**NATIVE, "test": ""}).reason.startswith("incomplete_leetcode_contract")
    assert rejected(
        {
            **NATIVE,
            "entry_point": "Solution().threeSum",
            "starter_code": "class Solution:\n    def twoSum(self, nums):\n        ",
        }
    ).reason.startswith("entry_point_not_in_starter")
    assert rejected({**NATIVE, "test": "def verify(x):\n    pass"}).reason.startswith("missing_check_harness")
    assert rejected({"prompt": NATIVE["prompt"]}).reason.startswith("incomplete_leetcode_contract")  # no query


def test_source_module_binds_the_train_split():
    source = leetcode_dataset.source(("LeetCodeDataset-train.jsonl",))
    assert source.info.id == leetcode_dataset.SOURCE_ID
    assert source.info.count == 2641
    assert source.info.origin.value == "Newfacade"
    assert source.config.source.revision == leetcode_dataset.REVISION
    assert "harness" in source.info.notes and "private provenance" in source.info.notes
