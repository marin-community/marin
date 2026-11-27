# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ScriptGrader, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import deepcoder_preview

PRIME = {
    "problem": "Given a list of URLs, count the ones belonging to the abacaba domain.",
    "tests": json.dumps(
        [
            {"type": "stdin_stdout", "input": "2\nhttp://abacaba.ru/a\nhttp://other.com/\n", "output": "1\n"},
        ]
    ),
    "solutions": ["```python\nprint(0)\n```"],
}
TACO = {
    "problem": "Sort the sequence with the given operations.",
    "tests": json.dumps({"inputs": ["3 1\n1 2 3\n", "1 1\n5\n"], "outputs": ["1 2 3\n", "5\n"]}),
}


def converted(native):
    row = RawRow(
        id="original-deepcoder-row",
        source=Source(
            dataset=deepcoder_preview.REPO, revision=deepcoder_preview.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = deepcoder_preview.convert_problem(
        row,
        # The census compiles the recipe's pinned requirements to a hash lock and supplies its URL.
        ConversionContext(
            {},
            EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                packages_lock="s3://test/deepcoder-requirements.lock",
            ),
        ),
    )
    assert isinstance(result, NormalizedTask)
    return result


def rejected(native):
    row = RawRow(
        id="original-deepcoder-row",
        source=Source(
            dataset=deepcoder_preview.REPO, revision=deepcoder_preview.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = deepcoder_preview.convert_problem(row, _context())
    assert isinstance(result, ImportRejection)
    return result


def _context():
    return ConversionContext(
        {},
        EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="s3://test/deepcoder-requirements.lock"
        ),
    )


def test_the_array_tests_normalize_into_the_apps_contract():
    task = converted(PRIME).task
    assert isinstance(task.grader, ScriptGrader) and task.grader.judge is None
    config = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    payload = json.loads(resource_bytes(config))
    assert payload["input_output"] == {"inputs": ["2\nhttp://abacaba.ru/a\nhttp://other.com/\n"], "outputs": ["1\n"]}
    assert "solutions" not in json.dumps(payload)  # the reference solutions stay private
    grade = next(resource for resource in task.resources.verifier if resource.path == "grade.py")
    assert b"apps_testing_util" in resource_bytes(grade)


def test_taco_paired_tests_pass_through():
    task = converted(TACO).task
    config = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    payload = json.loads(resource_bytes(config))
    assert payload["input_output"]["inputs"] == ["3 1\n1 2 3\n", "1 1\n5\n"]


def test_the_context_is_the_verbatim_problem_with_the_completion_instruction():
    result = converted(PRIME)
    (user,) = result.task.context.events
    assert user.content == PRIME["problem"] + deepcoder_preview.CODE_INSTRUCTION
    assert len(result.changes) == 1 and "completion format instruction" in result.changes[0].reason


def test_contract_defects_fail_closed():
    assert rejected({**PRIME, "problem": ""}).reason.startswith("incomplete_deepcoder_contract")
    assert rejected({**PRIME, "tests": "not json"}).reason.startswith("invalid_test_contract")
    assert rejected({**PRIME, "tests": "[]"}).reason.startswith("invalid_test_contract")
    assert rejected({**PRIME, "tests": json.dumps([{"input": "1\n"}])}).reason.startswith("invalid_test_contract")
    assert rejected({key: value for key, value in PRIME.items() if key != "tests"}).reason.startswith(
        "incomplete_deepcoder_contract"
    )


def test_the_source_module_binds_the_sealed_population():
    source = deepcoder_preview.source(deepcoder_preview.FILES)
    assert source.info.id == deepcoder_preview.SOURCE_ID
    assert source.info.count == 23688
    assert source.info.origin.value == "agentica-org"
    assert len(source.config.source.files) == 9
    assert "private provenance" in source.info.notes and "codeforces" in source.info.notes
