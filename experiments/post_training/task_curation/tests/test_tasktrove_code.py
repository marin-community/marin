# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove executable, structured-output and repository sources turn archives into graded tasks."""

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from taskcompendium.convert.executable import SOLUTION_PATHS, broken_submission, solve_script
from taskcompendium.convert.tasktrove import SOLVE_SH, TEST_SH, archive_files, unpack_task_binary
from taskcompendium.convert.tasktrove_nl2bash import OUTPUT_PATH
from taskcompendium.models import TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus, ImportFailureKind, ImportRejection, NormalizedTask
from taskcompendium.runtime.resources import resource_bytes
from verifyit.grade import grade
from verifyit.spec import ScriptSpec, StdioSpec

from experiments.post_training.task_curation.datasets.tasktrove import (
    code,
    nl2bash,
    python_tests,
    repositories,
    structured_outputs,
)
from experiments.post_training.task_curation.images import (
    TASKTROVE_EXECUTABLE_IMAGE,
    TASKTROVE_NL2BASH_IMAGE,
    TASKTROVE_PYTHON_TESTS_IMAGE,
    TASKTROVE_STACK_PYTEST_IMAGE,
)
from experiments.post_training.task_curation.tests.conversion import convert_row, converted_task, tasktrove_row

PIPELINES = {
    pipeline.name: pipeline
    for module in (code, python_tests, nl2bash, structured_outputs, repositories)
    for pipeline in module.pipelines()
}

CODENET = (Path(__file__).parent / "fixtures/codenet.tar.gz").read_bytes()
"""A pinned codeforces-layout archive: three cases, a ``solve.sh`` oracle and the source's runner."""

DOCKERFILE = b"FROM python:3.12-slim\nWORKDIR /app\n"
SUM_PROMPT = "Read two integers from stdin and print their sum. Write your program to `/app/solution.py`."
SUM_SOLUTION = b"a, b = map(int, input().split())\nprint(a + b)\n"
SUM_CASES = {"inputs": ["3 4\n", "-5 3\n"], "outputs": ["7\n", "-2\n"]}
NAME_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
SEED_SCRIPT = b"mkdir -p /workspace && printf 'x' > /workspace/notes.txt\n"
STRUCTURED_PROMPT = (
    "Extract the person's name from the note below as JSON.\n\nNote: Ada wrote this.\n\n"
    "Write your final JSON to `/app/answer.txt`."
)


def archive(instruction: str, files: dict[str, bytes]) -> dict:
    """A source-shaped TaskTrove row: instruction, Dockerfile, the source's runner and ``files``."""
    return tasktrove_row(
        {
            "instruction.md": instruction.encode(),
            "environment/Dockerfile": DOCKERFILE,
            TEST_SH: b"#!/bin/bash\nexit 1\n",
            **files,
        }
    )


def python_row(test_file: str = "tests/test_solution.py") -> dict:
    return archive(
        "Implement `add(a, b)`, returning the sum of two integers, in `/app/solution.py`.",
        {
            test_file: b"from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            "solution/solution.py": b"def add(a, b):\n    return a + b\n",
        },
    )


def stdio_dirs(inputs: list[str], outputs: list[str]) -> dict[str, bytes]:
    files = {}
    for index, (stdin, stdout) in enumerate(zip(inputs, outputs, strict=True)):
        files[f"tests/inputs/input_{index}.txt"] = stdin.encode()
        files[f"tests/outputs/output_{index}.txt"] = stdout.encode()
    return files


def nl2bash_row(verifier_data: dict | None) -> dict:
    files = {
        "setup_files/setup_seeds.sh": SEED_SCRIPT,
        "tests/setup_files/setup_seeds.sh": SEED_SCRIPT,
        SOLVE_SH: f"#!/bin/bash\nbash /tests/setup_seeds.sh\ncd /workspace && ls > {OUTPUT_PATH} 2>&1\n".encode(),
    }
    if verifier_data is not None:
        files["tests/verifier_data.json"] = json.dumps(verifier_data).encode()
    return archive(
        f"Run `bash /setup_files/setup_seeds.sh`, then list /workspace and save the output to `{OUTPUT_PATH}`.",
        files,
    )


def structured_row(schema: dict, schema_type: str, instruction: str = STRUCTURED_PROMPT) -> dict:
    return archive(
        instruction, {"tests/verifier_data.json": json.dumps({"schema": schema, "schema_type": schema_type}).encode()}
    )


def repository_row() -> dict:
    config = {"repo": "octo/widgets", "FAIL_TO_PASS": ["tests/test_spin.py::test_spin"], "PASS_TO_PASS": []}
    return archive(
        "Fix the spin bug in octo/widgets.\n\n```\n"
        "git clone https://github.com/octo/widgets.git . && git checkout 1a2b3c4\n```\n",
        {"tests/config.json": json.dumps(config).encode(), SOLVE_SH: b"#!/bin/bash\ngit apply /solution/fix.patch\n"},
    )


ROWS: dict[str, dict] = {
    "tasktrove-code_contests": archive(SUM_PROMPT, {"tests/test_data.json": json.dumps(SUM_CASES).encode()}),
    "tasktrove-codeforces": {"path": "codenet", "task_binary": CODENET},
    "tasktrove-competitive_coding": archive(SUM_PROMPT, {"tests/verifier_data.json": json.dumps(SUM_CASES).encode()}),
    "tasktrove-taco": archive(
        SUM_PROMPT, {**stdio_dirs(SUM_CASES["inputs"], SUM_CASES["outputs"]), "solution/solution.py": SUM_SOLUTION}
    ),
    "tasktrove-nl2bash": nl2bash_row({"expected_output": "notes.txt\n"}),
    "tasktrove-curriculum_easy": python_row("tests/test_curriculum.py"),
    "tasktrove-curriculum_medium": python_row("tests/test_curriculum.py"),
    "tasktrove-e2egit": python_row(),
    "tasktrove-e2egit_large": python_row(),
    "tasktrove-multifile": python_row("tests/test_multifile.py"),
    "tasktrove-pymethods": python_row(),
    "tasktrove-pymethods_large": python_row(),
    "tasktrove-unitsyn": python_row(),
    "tasktrove-unitsyn_large": python_row(),
    "tasktrove-stack_pytest": python_row(),
    "tasktrove-structured_outputs": structured_row(NAME_SCHEMA, "json"),
    "tasktrove-swe_rebench": repository_row(),
    "tasktrove-swesmith": repository_row(),
}

PYTHON_FILE = ("/app/solution.py",)
EXPECTED = {
    "tasktrove-code_contests": ("stdio", SOLUTION_PATHS, TASKTROVE_EXECUTABLE_IMAGE),
    "tasktrove-codeforces": ("stdio", SOLUTION_PATHS, TASKTROVE_EXECUTABLE_IMAGE),
    "tasktrove-competitive_coding": ("stdio", PYTHON_FILE, TASKTROVE_EXECUTABLE_IMAGE),
    "tasktrove-taco": ("stdio", SOLUTION_PATHS, TASKTROVE_EXECUTABLE_IMAGE),
    "tasktrove-nl2bash": ("script", (OUTPUT_PATH,), TASKTROVE_NL2BASH_IMAGE),
    "tasktrove-curriculum_easy": ("pytest", PYTHON_FILE, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-curriculum_medium": ("pytest", PYTHON_FILE, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-e2egit": ("pytest", PYTHON_FILE, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-e2egit_large": ("pytest", PYTHON_FILE, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-multifile": ("pytest", PYTHON_FILE, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-pymethods": ("pytest", PYTHON_FILE, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-pymethods_large": ("pytest", PYTHON_FILE, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-unitsyn": ("pytest", SOLUTION_PATHS, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-unitsyn_large": ("pytest", PYTHON_FILE, TASKTROVE_PYTHON_TESTS_IMAGE),
    "tasktrove-stack_pytest": ("pytest", PYTHON_FILE, TASKTROVE_STACK_PYTEST_IMAGE),
    "tasktrove-structured_outputs": ("json-schema", (), None),
    "tasktrove-swe_rebench": ("none", (), None),
    "tasktrove-swesmith": ("none", (), None),
}
"""Each declaration's grader mode (``none`` for an ungraded task), captured files and grader image."""


def grader_name(task: TaskSpec) -> str:
    return task.grader.mode if isinstance(task.grader, VerifyitGrader) else task.grader.kind


def write_verifier(task: TaskSpec, tests: Path) -> None:
    for resource in task.resources.verifier:
        target = tests / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))


def resource_map(resources) -> dict[str, bytes]:
    return {resource.path: resource_bytes(resource) for resource in resources}


def test_rows_cover_every_declaration():
    assert set(ROWS) == set(PIPELINES) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_row_converts_to_declared_grader(name):
    task = converted_task(PIPELINES[name], ROWS[name])
    mode, output_paths, image = EXPECTED[name]
    assert grader_name(task) == mode
    assert task.output_paths == output_paths
    grader_environment = task.grader.environment if isinstance(task.grader, VerifyitGrader) else None
    expected_image = image.requirements().docker_image if image is not None else None
    assert (grader_environment.docker_image if grader_environment is not None else None) == expected_image
    # Hidden tests and oracle files never reach the agent's machine.
    worker = {resource.path for resource in task.resources.worker}
    assert not any(path.startswith(("tests/", "solution/", "cases/")) for path in worker)


@pytest.mark.parametrize(
    "name, row, kind, reason",
    [
        ("tasktrove-code_contests", archive(SUM_PROMPT, {}), ImportFailureKind.SOURCE_DEFECT, "null_grader"),
        (
            "tasktrove-code_contests",
            archive(
                f"{SUM_PROMPT}\n\nExamples: `3 4` and `-5 3`.", {"tests/test_data.json": json.dumps(SUM_CASES).encode()}
            ),
            ImportFailureKind.SOURCE_DEFECT,
            "gold_in_instruction",
        ),
        (
            "tasktrove-taco",
            archive(
                SUM_PROMPT,
                {
                    **stdio_dirs(SUM_CASES["inputs"], SUM_CASES["outputs"]),
                    "solution/solution.py": b"class Solution:\n    def add(self, a, b):\n        return a + b\n",
                },
            ),
            ImportFailureKind.UNSUPPORTED,
            "unsupported_variant",
        ),
        (
            "tasktrove-competitive_coding",
            archive(SUM_PROMPT, {"tests/verifier_data.json": json.dumps({"inputs": ["1\n"], "outputs": []}).encode()}),
            ImportFailureKind.SOURCE_DEFECT,
            "null_grader",
        ),
        ("tasktrove-e2egit", archive(SUM_PROMPT, {}), ImportFailureKind.UNSUPPORTED, "unsupported_variant"),
        (
            "tasktrove-pymethods",
            archive(
                "Implement the requested helper.",
                {"tests/test_solution.py": b"from helpers import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"},
            ),
            ImportFailureKind.UNSUPPORTED,
            "unsupported_public_output_contract",
        ),
        (
            "tasktrove-swesmith",
            archive("Fix the spin bug.", {"tests/config.json": b'{"repo": "octo/widgets"}'}),
            ImportFailureKind.UNSUPPORTED,
            "missing_repository_ref",
        ),
    ],
)
def test_ungradable_archives_are_rejected_with_typed_cause(name, row, kind, reason):
    rejection = convert_row(PIPELINES[name], row)
    assert isinstance(rejection, ImportRejection)
    assert (rejection.kind, rejection.reason) == (kind, reason)


def test_python_tests_infer_module_file_from_hidden_test_imports():
    row = archive(
        "Implement `add(a, b)` returning the sum of two integers.",
        {"tests/test_solution.py": b"from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"},
    )
    result = convert_row(PIPELINES["tasktrove-e2egit"], row)
    assert isinstance(result, NormalizedTask)
    assert result.task.output_paths == ("/app/calculator.py",)
    prompt = result.task.context.events[0].content
    assert prompt.endswith("Delivery: write the requested implementation to `/app/calculator.py`.\n")


def local_stdio(task: TaskSpec, workspace: Path) -> StdioSpec:
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, StdioSpec)
    return replace(spec, workspace=str(workspace))


def test_codeforces_oracle_passes_and_negative_fails_on_hidden_cases(tmp_path):
    task = converted_task(PIPELINES["tasktrove-codeforces"], ROWS["tasktrove-codeforces"])
    write_verifier(task, tmp_path / "tests")
    workspace = tmp_path / "app"
    spec = local_stdio(task, workspace)
    assert solve_script(task) is not None
    oracle = tmp_path / "solve.sh"
    oracle.write_bytes(resource_map(task.resources.oracle)[SOLVE_SH])
    subprocess.run(["bash", str(oracle)], check=True, env={**os.environ, "APP_DIR": str(workspace)})
    assert grade(spec, tmp_path / "tests", workspace).reward == 1.0
    for path, data in broken_submission(task).files.items():
        (workspace / Path(path).name).write_bytes(data)
    assert grade(spec, tmp_path / "tests", workspace).reward == 0.0


@pytest.mark.parametrize("exit_code, expected_reward", [(0, 1.0), (7, 0.0)])
def test_codeforces_source_runner_and_converted_grader_agree_on_exit_status(tmp_path, exit_code, expected_reward):
    # The submission prints every expected output, so only its exit status decides the reward.
    source = archive_files(unpack_task_binary(ROWS["tasktrove-codeforces"], {}))
    task = converted_task(PIPELINES["tasktrove-codeforces"], ROWS["tasktrove-codeforces"])
    tests = tmp_path / "tests"
    write_verifier(task, tests)
    for path, data in source.under("tests/").items():
        (tests / path.removeprefix("tests/")).parent.mkdir(parents=True, exist_ok=True)
        (tests / path.removeprefix("tests/")).write_bytes(data)
    answers = {
        data.decode(): source.text(path.replace("/inputs/input_", "/outputs/output_"))
        for path, data in source.under("tests/inputs/").items()
    }
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "solution.py").write_text(
        f"import sys\nanswers={answers!r}\nsys.stdout.write(answers[sys.stdin.read()])\nsys.exit({exit_code})\n"
    )
    runner = source.text(TEST_SH)
    for old, new in (
        ("/logs/verifier", str(tmp_path / "logs")),
        ("/tmp/codenet-actual.txt", str(tmp_path / "actual.txt")),
        ("/tests", str(tests)),
        ("/app", str(workspace)),
    ):
        runner = runner.replace(old, new)
    subprocess.run(["bash", "-c", runner], check=False, capture_output=True)
    assert float((tmp_path / "logs/reward.txt").read_text()) == expected_reward
    assert grade(local_stdio(task, workspace), tests, workspace).reward == expected_reward


@pytest.mark.parametrize(
    "capture, reward",
    [("notes.txt\n", 1.0), ("./notes.txt\nextra.log\n", 1.0), ("unexpected error: missing input\n", 0.0)],
)
def test_nl2bash_checker_grades_the_captured_output(tmp_path, capture, reward):
    task = converted_task(PIPELINES["tasktrove-nl2bash"], ROWS["tasktrove-nl2bash"])
    write_verifier(task, tmp_path / "tests")
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, ScriptSpec)
    output = tmp_path / "capture.txt"
    output.write_text(capture)
    local = replace(spec, args=(str(output),), workspace=str(tmp_path))
    assert grade(local, tmp_path / "tests", tmp_path).reward == reward


def test_nl2bash_oracle_seeds_from_the_mounted_oracle_script():
    task = converted_task(PIPELINES["tasktrove-nl2bash"], ROWS["tasktrove-nl2bash"])
    oracle = resource_map(task.resources.oracle)
    assert "bash /tests/setup_files/setup_seeds.sh" in oracle[SOLVE_SH].decode()
    assert oracle["tests/setup_files/setup_seeds.sh"] == SEED_SCRIPT
    assert resource_map(task.resources.worker) == {"setup_files/setup_seeds.sh": SEED_SCRIPT}


@pytest.mark.parametrize(
    "verifier_data, kind",
    [({"expected_output": None}, ImportFailureKind.SOURCE_DEFECT), (None, ImportFailureKind.CONVERTER_ERROR)],
)
def test_nl2bash_without_expected_output_is_rejected(verifier_data, kind):
    rejection = convert_row(PIPELINES["tasktrove-nl2bash"], nl2bash_row(verifier_data))
    assert isinstance(rejection, ImportRejection)
    assert rejection.kind is kind and rejection.detail


def test_structured_outputs_ask_for_the_answer_in_the_reply():
    footer = f"{structured_outputs.SUBMISSION_FOOTER}Your chat reply is NOT graded; only `/app/answer.txt` is read.\n"
    result = convert_row(
        PIPELINES["tasktrove-structured_outputs"], structured_row(NAME_SCHEMA, "json", STRUCTURED_PROMPT + footer)
    )
    assert isinstance(result, NormalizedTask)
    prompt = result.task.context.events[-1].content
    assert prompt.endswith("Return your final JSON in the assistant response.")
    assert "/app/answer.txt" not in prompt
    assert [change.field for change in result.changes] == ["instruction"]


def test_structured_outputs_reject_a_required_field_the_schema_forbids():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    rejection = convert_row(PIPELINES["tasktrove-structured_outputs"], structured_row(schema, "json"))
    assert isinstance(rejection, ImportRejection)
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.SOURCE_DEFECT, "unsatisfiable_schema")


@pytest.mark.parametrize(
    "schema_type, golden", [("json", CheckStatus.SKIPPED), ("xml", CheckStatus.PASS), ("csv", CheckStatus.PASS)]
)
def test_structured_outputs_controls_grade_as_expected(schema_type, golden):
    pipeline = PIPELINES["tasktrove-structured_outputs"]
    task = converted_task(pipeline, structured_row(NAME_SCHEMA, schema_type))
    assert pipeline.controls is not None
    report = run_controls(task, controls=pipeline.controls, machines=None)
    assert {check.check: check.status for check in report.checks} == {
        "empty": CheckStatus.PASS,
        "golden": golden,
        "negative": CheckStatus.PASS,
    }


def test_repository_tasks_keep_source_grading_terms():
    task = converted_task(PIPELINES["tasktrove-swe_rebench"], ROWS["tasktrove-swe_rebench"])
    contract = task.grader.model_dump()["contract"]["contract"]
    assert (contract["repository"], contract["source_ref"], contract["workspace"]) == (
        "octo/widgets",
        "1a2b3c4",
        "/testbed",
    )
    assert set(resource_map(task.resources.verifier)) == {
        "taskcompendium/archive-provenance.json",
        "config.json",
        "test.sh",
    }
    assert set(resource_map(task.resources.oracle)) == {"instruction.md", "environment/Dockerfile", SOLVE_SH}
