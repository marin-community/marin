# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove executable, structured-output and repository sources turn archives into graded tasks."""

import json
import os
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from taskcompendium.convert.executable import solve_script
from taskcompendium.convert.tasktrove import SOLVE_SH, TEST_SH, archive_files, unpack_task_binary
from taskcompendium.convert.tasktrove_nl2bash import OUTPUT_PATH
from taskcompendium.harbor.export import harbor_record
from taskcompendium.models import TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, NormalizedTask
from taskcompendium.runtime.resources import resource_bytes
from verifyit.grade import Status, grade
from verifyit.spec import ScriptSpec, StdioSpec

from experiments.post_training.task_curation.datasets.tasktrove import (
    code,
    nl2bash,
    python_tests,
    structured_outputs,
)
from experiments.post_training.task_curation.tests.conversion import (
    BASE_IMAGE,
    convert_row,
    converted_task,
    fixture_context,
    tasktrove_row,
)

pytest_plugins = ("lib.verifyit.tests.test_judge",)

PIPELINES = {
    source.name: source.pipeline
    for module in (code, python_tests, nl2bash, structured_outputs)
    for source in module.sources()
    if source.pipeline is not None
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
}


def write_verifier(task: TaskSpec, tests: Path) -> None:
    for resource in task.resources.verifier:
        target = tests / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))


def resource_map(resources) -> dict[str, bytes]:
    return {resource.path: resource_bytes(resource) for resource in resources}


@pytest.mark.parametrize("name", sorted(ROWS))
def test_conversion_keeps_hidden_tests_and_oracles_private(name):
    task = converted_task(PIPELINES[name], ROWS[name])
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
        (
            "tasktrove-competitive_coding",
            archive(
                f"{SUM_PROMPT}\n\nExamples: `3 4` and `-5 3`.",
                {"tests/verifier_data.json": json.dumps(SUM_CASES).encode()},
            ),
            ImportFailureKind.SOURCE_DEFECT,
            "gold_in_instruction",
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


def test_codeforces_oracle_passes_on_hidden_cases(tmp_path):
    task = converted_task(PIPELINES["tasktrove-codeforces"], ROWS["tasktrove-codeforces"])
    write_verifier(task, tmp_path / "tests")
    workspace = tmp_path / "app"
    spec = local_stdio(task, workspace)
    assert solve_script(task) is not None
    oracle = tmp_path / "solve.sh"
    oracle.write_bytes(resource_map(task.resources.oracle)[SOLVE_SH])
    subprocess.run(["bash", str(oracle)], check=True, env={**os.environ, "APP_DIR": str(workspace)})
    assert grade(spec, tmp_path / "tests", workspace).reward == 1.0


@pytest.mark.parametrize("exit_code, expected_reward", [(0, 1.0), (7, 0.0)])
def test_codeforces_source_runner_and_converted_grader_agree_on_exit_status(tmp_path, exit_code, expected_reward):
    # The submission prints every expected output, so only its exit status decides the reward.
    pipeline = PIPELINES["tasktrove-codeforces"]
    source = archive_files(unpack_task_binary(ROWS["tasktrove-codeforces"], fixture_context(pipeline)))
    task = converted_task(pipeline, ROWS["tasktrove-codeforces"])
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
    prompt = cast(str, result.task.context.events[-1].content)
    assert "Return your final JSON in the assistant response." in prompt
    assert "Missing data convention" in prompt
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
    "schema_type,candidate",
    [
        ("json", '{"name":"Ada"}'),
        ("yaml", "name: Ada"),
        ("toml", 'name="Ada"'),
        ("xml", "<person><name>Ada</name></person>"),
        ("csv", "name\nAda\n"),
    ],
)
@pytest.mark.parametrize("label,reward", [("PASS", 1.0), ("FAIL", 0.0)])
def test_structured_outputs_require_grounding_after_valid_format(
    tmp_path, fake_judge, monkeypatch, schema_type, candidate, label, reward
):
    pipeline = PIPELINES["tasktrove-structured_outputs"]
    task = converted_task(pipeline, structured_row(NAME_SCHEMA, schema_type))
    tests = tmp_path / "tests"
    write_verifier(task, tests)
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "answer.txt").write_text(candidate)
    for name in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    fake_judge.replies = [label]
    verdict = grade(verifyit_spec(task.grader), tests, workspace)
    assert (verdict.status, verdict.reward) == (Status.SCORED, reward), verdict.detail
    assert "Ada wrote this." in fake_judge.prompts[0]
    assert candidate in fake_judge.prompts[0]


def test_structured_outputs_invalid_format_scores_zero_before_grounding(tmp_path, fake_judge):
    task = converted_task(PIPELINES["tasktrove-structured_outputs"], structured_row(NAME_SCHEMA, "json"))
    tests = tmp_path / "tests"
    write_verifier(task, tests)
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("not JSON")
    verdict = grade(verifyit_spec(task.grader), tests, workspace)
    assert (verdict.status, verdict.reward) == (Status.SCORED, 0.0)
    assert fake_judge.requests == []


@pytest.mark.parametrize("name", ["code_contests", "taco"])
def test_source_stdio_keeps_shared_recipe_and_grades_single_hidden_case(name, tmp_path):
    cases = {"inputs": ["3 4\n"], "outputs": ["7\n"]}
    data = (
        {"tests/test_data.json": json.dumps(cases).encode()}
        if name == "code_contests"
        else {
            **stdio_dirs(cases["inputs"], cases["outputs"]),
            "solution/solution.py": SUM_SOLUTION,
        }
    )
    recipe = b"FROM python:3.12-slim\nWORKDIR /app\nRUN mkdir /source-dependency\n"
    data["environment/Dockerfile"] = recipe
    task = converted_task(PIPELINES[f"tasktrove-{name}"], archive(SUM_PROMPT, data))
    record = harbor_record(
        {"task_json": task.model_dump_json(), "original_path": "fixture", "source_row": f"{name}/tasks.parquet:0"},
        grader_image=None,
        fallback_actor_image=BASE_IMAGE,
        family="competitive-programming",
    )
    files = archive_files(
        unpack_task_binary(
            {"path": "fixture", "task_binary": record.task_binary}, fixture_context(PIPELINES[f"tasktrove-{name}"])
        )
    ).files
    assert files["environment/Dockerfile"].decode().split("# --- verifyit ---")[0].strip() == recipe.decode().strip()
    config = tomllib.loads(files["task.toml"].decode())
    assert config["verifier"]["environment_mode"] == "shared"
    assert not config["artifacts"] and "tests/Dockerfile" not in files
    assert not any(path.startswith(("environment/tests/", "environment/solution/", "solution/")) for path in files)
    tests = tmp_path / "tests"
    write_verifier(task, tests)
    workspace = tmp_path / "app"
    workspace.mkdir()
    spec = local_stdio(task, workspace)
    spec = replace(spec, command=spec.command.replace("/app/", str(workspace) + "/"))
    (workspace / "solution.py").write_bytes(SUM_SOLUTION)
    assert grade(spec, tests, workspace).reward == 1.0
    (workspace / "solution.py").write_text("print('wrong')\n")
    assert grade(spec, tests, workspace).reward == 0.0
