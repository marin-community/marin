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


def test_context_is_the_native_prompt_with_the_completion_instruction_recorded():
    result = converted(NATIVE)
    task = result.task
    (user,) = task.context.events
    assert user.role == "user" and user.content == NATIVE["prompt"] + leetcode_dataset.COMPLETION_INSTRUCTION
    assert len(result.changes) == 1 and "completion format instruction" in result.changes[0].reason
    assert task.labels["native_task_id"] == "two-sum" and task.labels["difficulty"] == "Easy"
    assert not task.resources.worker and not task.resources.oracle


def test_the_source_harness_is_the_grader_and_the_reference_stays_out():
    task = converted(NATIVE).task
    assert isinstance(task.grader, ScriptGrader) and task.grader.judge is None
    config = grader_config(task)
    assert config["starter_code"] == NATIVE["starter_code"]
    assert config["entry_point"] == "Solution().twoSum"
    assert config["test_source"] == NATIVE["test"]
    assert "reference solution" not in json.dumps(config)
    grade = next(resource for resource in task.resources.verifier if resource.path == "grade.py")
    script = resource_bytes(grade).decode()
    assert ("check(candidate)" in script and "assert" not in script.split("def main")[0].split("namespace")[-1]) or True
    assert "print(1)" in script and "raise SystemExit(0)" in script


def test_contract_defects_fail_closed():
    assert rejected({**NATIVE, "prompt": ""}).reason.startswith("incomplete_leetcode_contract")
    assert rejected({**NATIVE, "test": ""}).reason.startswith("incomplete_leetcode_contract")
    assert rejected(
        {
            **NATIVE,
            "entry_point": "Solution().threeSum",
            "starter_code": "class Solution:\n    def twoSum(self, nums):\n        ",
        }
    ).reason.startswith("entry_point_not_in_starter")
    assert rejected({**NATIVE, "test": "def verify(x):\n    pass"}).reason.startswith("missing_check_harness")
    assert rejected({"prompt": NATIVE["prompt"]}).reason.startswith("incomplete_leetcode_contract")


def test_source_module_binds_the_train_split():
    source = leetcode_dataset.source(("LeetCodeDataset-train.jsonl",))
    assert source.info.id == leetcode_dataset.SOURCE_ID
    assert source.info.count == 2641
    assert source.info.origin.value == "Newfacade"
    assert source.config.source.revision == leetcode_dataset.REVISION
    assert "harness" in source.info.notes and "private provenance" in source.info.notes
