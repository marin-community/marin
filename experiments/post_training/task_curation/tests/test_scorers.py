# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVARC execution and APPS grading against independent expected results."""

import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

DATASETS = Path(__file__).resolve().parents[1] / "datasets"
ULTRA = DATASETS / "nemotron_ultra" / "scorers"
SKYRL = DATASETS / "skyrl" / "scorers"
ARC = DATASETS / "arc" / "scorers"
ULTRA_ENVS = "skyrl_gym/envs/nemotron_ultra"
APPS_EVALUATOR = SKYRL / "apps_testing_util.py"


def shipped(root: Path, *paths: str) -> dict[str, Path]:
    """Map each path under ``/tests`` to the vendored file at the same path under ``root``."""
    return {path: root / path for path in paths}


@dataclass(frozen=True)
class ScorerClosure:
    files: dict[str, Path]
    """Path under ``/tests`` -> vendored file."""
    requires: tuple[str, ...] = ()
    """Third-party modules the grader image provides."""

    def stage(self, root: Path) -> Path:
        tests = root / "tests"
        for path, source in self.files.items():
            (tests / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, tests / path)
        return tests

    def skip_if_unavailable(self) -> None:
        missing = [module for module in self.requires if importlib.util.find_spec(module) is None]
        if missing:
            pytest.skip(f"grader-image modules absent locally: {', '.join(missing)}")


ULTRA_BASE = shipped(
    ULTRA,
    "skyrl_gym/__init__.py",
    "skyrl_gym/envs/__init__.py",
    "skyrl_gym/envs/aime/utils.py",
    f"{ULTRA_ENVS}/__init__.py",
    f"{ULTRA_ENVS}/answer_extraction.py",
)
"""Every Ultra and NVARC task ships the final-answer extractor and the package around it."""


NVARC = ScorerClosure(
    {**ULTRA_BASE, **shipped(ARC, f"{ULTRA_ENVS}/nvarc.py", f"{ULTRA_ENVS}/sandbox.py", "local_sandbox.py")},
    ("requests", "numpy"),
)

STRUCTURED_OUTPUT = ScorerClosure(
    {**ULTRA_BASE, **shipped(ULTRA, f"{ULTRA_ENVS}/structured_outputs.py")},
    ("openapi_schema_validator", "xmltodict", "yaml"),
)


NVARC_RECORD = {"test_input": [[1, 2], [3, 4]], "expected_output": [[2, 3], [4, 5]]}
NVARC_CHECK = """
import json, sys
sys.path.insert(0, sys.argv[1])
from local_sandbox import LocalSandbox
from skyrl_gym.envs.nemotron_ultra import nvarc
reward, detail = nvarc.grade_inductive_arc(
    sys.stdin.read(), json.loads(sys.argv[2]), sandbox=LocalSandbox(user=None), python_timeout_seconds=1
)
print(json.dumps([reward, detail["execution_output"]["process_status"]]))
"""

NVARC_TRANSDUCTIVE_CHECK = """
import json, sys
sys.path.insert(0, sys.argv[1])
from skyrl_gym.envs.nemotron_ultra.nvarc import parse_grid
print(json.dumps(parse_grid(sys.stdin.read())))
"""

STRUCTURED_OUTPUT_CHECK = """
import json, sys
sys.path.insert(0, sys.argv[1])
from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output
record = {
    "schema_str": json.dumps({"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}),
    "schema_type": "json",
}
reward, detail = grade_structured_output(sys.stdin.read(), record, {})
print(json.dumps([reward, detail["error_type"]]))
"""

ANSWER_EXTRACTION_CHECK = """
import json, sys
sys.path.insert(0, sys.argv[1])
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
print(json.dumps(final_answer_text(sys.stdin.read())))
"""


def test_ultra_final_answer_excludes_hidden_reasoning(tmp_path: Path):
    tests = ScorerClosure(ULTRA_BASE).stage(tmp_path)
    result = subprocess.run(
        (sys.executable, "-I", "-c", ANSWER_EXTRACTION_CHECK, str(tests)),
        input="<|start_think|>Tea, maybe coffee.<|end_think|>Tea is brewed from leaves.",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == "Tea is brewed from leaves."


def test_structured_output_accepts_a_markdown_fenced_document(tmp_path: Path):
    STRUCTURED_OUTPUT.skip_if_unavailable()
    tests = STRUCTURED_OUTPUT.stage(tmp_path)
    result = subprocess.run(
        (sys.executable, "-I", "-c", STRUCTURED_OUTPUT_CHECK, str(tests)),
        input='The requested document follows.\n```json\n{"count": 2}\n```',
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [1.0, None]


@pytest.mark.parametrize(
    "answer",
    [
        "```text\n1 2\n3 4\n```",
        "```json\n[[1, 2], [3, 4]]\n```",
        "[[1, 2], [3, 4]]",
        "\\boxed{12\n34}",
        "<|start_think|>try a transform<|end_think|>\n```\n12\n34\n```",
    ],
    ids=["fenced_rows", "fenced_json", "json", "boxed_compact", "reasoning_then_fenced_compact"],
)
def test_nvarc_transductive_accepts_common_grid_formats(answer: str, tmp_path: Path):
    NVARC.skip_if_unavailable()
    tests = NVARC.stage(tmp_path)
    result = subprocess.run(
        (sys.executable, "-I", "-c", NVARC_TRANSDUCTIVE_CHECK, str(tests)),
        input=answer,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    ("transform", "reward", "status"),
    [
        ("import numpy as np\ndef transform(grid):\n    return np.array(grid) + 1", 1.0, "completed"),
        ("def transform(grid):\n    return grid", 0.0, "completed"),
        ("def transform(grid):\n    while True:\n        pass", 0.0, "timeout"),
    ],
    ids=["solves", "wrong_grid", "never_returns"],
)
def test_nvarc_runs_transforms_in_local_sandbox(transform: str, reward: float, status: str, tmp_path: Path):
    NVARC.skip_if_unavailable()
    tests = NVARC.stage(tmp_path)
    result = subprocess.run(
        (sys.executable, "-I", "-c", NVARC_CHECK, str(tests), json.dumps(NVARC_RECORD)),
        input=f"```python\n{transform}\n```",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [reward, status]


# APPS-format problems and the per-case results the pinned evaluator returns for them under Python 3.10
# with pyext: True/False per case, -1 for a runtime error or a timeout (4 s), -2 when the program does
# not compile. ``__name__`` is never "__main__" inside the evaluator, so a guarded main() never runs.
APPS_PROBLEMS = {
    "call_based": {
        "input_output": {"fn_name": "add", "inputs": [[1, 2], [3, 4], [5, 6]], "outputs": [[3], [7], [12]]},
        "solution": "def add(a, b):\n    return a + b\n",
        "grade": [True, True, False],
    },
    "solution_class": {
        "input_output": {
            "fn_name": "twoSum",
            "inputs": [[[2, 7, 11, 15], 9], [[3, 2, 4], 6], [[1, 5], 6]],
            "outputs": [[[0, 1]], [[1, 2]], [[1, 0]]],
        },
        "solution": (
            "class Solution:\n"
            "    def twoSum(self, nums, target):\n"
            "        seen = {}\n"
            "        for index, value in enumerate(nums):\n"
            "            if target - value in seen:\n"
            "                return [seen[target - value], index]\n"
            "            seen[value] = index\n"
        ),
        "grade": [True, True, False],
    },
    "standard_input": {
        "input_output": {"inputs": ["2\n1 2 3\n", "3\n4 5\n"], "outputs": ["12\n", "28\n"]},
        "solution": "n = int(input())\nvalues = list(map(int, input().split()))\nprint(sum(values) * n)\n",
        "grade": [True, False],
    },
    "standard_input_main_guard": {
        "input_output": {"inputs": ["1 2\n"], "outputs": ["3\n"]},
        "solution": (
            "import sys\n\n"
            "def main():\n"
            "    a, b = map(int, sys.stdin.read().split())\n"
            "    print(a + b)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        "grade": [False],
    },
    "compile_error": {
        "input_output": {"inputs": ["1\n"], "outputs": ["1\n"]},
        "solution": "print(int(input()) +\n",
        "grade": [-2],
    },
    "runtime_error": {
        "input_output": {"fn_name": "divide", "inputs": [[4, 2], [1, 0]], "outputs": [[2], [0]]},
        "solution": "def divide(a, b):\n    return a // b\n",
        "grade": [True, -1],
    },
    "timeout": {
        "input_output": {"inputs": ["1\n"], "outputs": ["1\n"]},
        "solution": "while True:\n    pass\n",
        "grade": [-1],
    },
}

APPS_RUN = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("testing_util", sys.argv[1])
testing_util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(testing_util)
problem = json.loads(sys.stdin.read())
results = testing_util.run_test(problem={"input_output": problem["input_output"]}, test=problem["solution"])
print(json.dumps(results, default=lambda value: value.item()))
"""


def apps_grade(python: Sequence[str], evaluator: Path, problem: dict) -> list:
    # run_test disables os, shutil and subprocess functions in its process, so each problem gets its own.
    result = subprocess.run(
        (*python, "-c", APPS_RUN, str(evaluator)),
        input=json.dumps(problem),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize("name", sorted(APPS_PROBLEMS))
def test_patched_apps_evaluator_grades_fixture(name: str):
    problem = APPS_PROBLEMS[name]
    assert apps_grade((sys.executable,), APPS_EVALUATOR, problem) == problem["grade"]
