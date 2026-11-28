# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest
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


@pytest.fixture
def staged_grader(tmp_path: Path) -> Path:
    task = converted(NATIVE).task
    for resource in task.resources.verifier:
        (tmp_path / resource.path).write_bytes(resource_bytes(resource))
    script = tmp_path / "grade.py"
    # Map the sandbox's absolute input paths into the test directory.
    script.write_text(
        script.read_text()
        .replace('"/tests/config.json"', repr(str(tmp_path / "config.json")))
        .replace('"/app/answer.txt"', repr(str(tmp_path / "answer.txt")))
    )
    return script


@pytest.mark.parametrize(
    ("answer", "reward"),
    [
        ("```python\ndef staking_simulation(xs: List[int], ys, t):\n    return [0]\n```", 0),
        ("def staking_simulation(:", 0),
        ("import unavailable_acecode_dependency", 0),
        ("This is a prose reply.", 0),
        ("print(1)\nraise SystemExit(0)", 0),
        ("def staking_simulation(xs, ys, t):\n    return [1500]", 0),
        (
            "```python\nraise RuntimeError\n```\n"
            "```python\nprint('debug output', end='')\n"
            "def staking_simulation(xs, ys, t):\n    return [1500 if t == 1 else 0]\n```",
            1,
        ),
    ],
    ids=("missing_annotation_import", "syntax_error", "missing_package", "prose", "early_exit", "failed_case", "pass"),
)
def test_shipped_grader_emits_only_its_reward(staged_grader: Path, answer: str, reward: int):
    (staged_grader.parent / "answer.txt").write_text(answer)
    result = subprocess.run([sys.executable, str(staged_grader)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{reward}\n"


def test_shipped_grader_scores_a_timed_out_case_zero(staged_grader: Path):
    (staged_grader.parent / "answer.txt").write_text(
        "def staking_simulation(xs, ys, t):\n    while True:\n        pass\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['main'](timeout=0.2)",
            str(staged_grader),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0\n"


@pytest.mark.parametrize("cases", [[], ["assert ("]], ids=("empty", "invalid_python"))
def test_shipped_grader_exits_nonzero_for_invalid_cases(staged_grader: Path, cases: list[str]):
    (staged_grader.parent / "config.json").write_text(json.dumps({"test_cases": cases}))
    (staged_grader.parent / "answer.txt").write_text("def staking_simulation(xs, ys, t):\n    return [0]\n")
    result = subprocess.run([sys.executable, str(staged_grader)], capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert result.stdout == ""


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
