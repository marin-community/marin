# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendored scorers import from the files a task ships, and the patched APPS evaluator grades like the original.

``CLOSURES`` names, per scorer, the files a task copies into ``/tests`` and the third-party modules the
grader image provides. Every check runs in a fresh interpreter: the APPS evaluator installs a SIGALRM
handler when imported, and the closures stage different ``skyrl_gym`` packages.
"""

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
APPS_PATCH = SKYRL / "apps_testing_util_py312.patch"
ORIGINAL_APPS_PYTHON = (
    *("uv", "run", "--no-project", "--python", "3.10"),
    *("--with", "pyext==0.7", "--with", "numpy==1.23.5", "python"),
)
"""The interpreter and packages of the APPS image the patch replaces."""


def shipped(root: Path, *paths: str) -> dict[str, Path]:
    """Map each path under ``/tests`` to the vendored file at the same path under ``root``."""
    return {path: root / path for path in paths}


@dataclass(frozen=True)
class ScorerClosure:
    modules: tuple[str, ...]
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


def ultra_scorer(name: str, *requires: str) -> ScorerClosure:
    return ScorerClosure(
        (f"skyrl_gym.envs.nemotron_ultra.{name}",), {**ULTRA_BASE, **shipped(ULTRA, f"{ULTRA_ENVS}/{name}.py")}, requires
    )


def flat_scorer(name: str, *requires: str) -> ScorerClosure:
    return ScorerClosure((name.removesuffix(".py"),), shipped(SKYRL, name), requires)


NVARC = ScorerClosure(
    ("skyrl_gym.envs.nemotron_ultra.nvarc", "local_sandbox"),
    {**ULTRA_BASE, **shipped(ARC, f"{ULTRA_ENVS}/nvarc.py", f"{ULTRA_ENVS}/sandbox.py", "local_sandbox.py")},
    ("requests", "numpy"),
)

CLOSURES = {
    "ultra_calendar": ultra_scorer("calendar"),
    "ultra_code_gen": ScorerClosure(
        ("skyrl_gym.envs.nemotron_ultra.code_gen",),
        {
            **ULTRA_BASE,
            **shipped(ULTRA, f"{ULTRA_ENVS}/code_gen.py"),
            "skyrl_gym/envs/lcb/livecodebench.py": SKYRL / "livecodebench.py",
        },
        ("numpy", "pandas"),
    ),
    "ultra_format_verification": ultra_scorer("format_verification"),
    "ultra_instruction_following": ultra_scorer("instruction_following", "verifiable_instructions"),
    "ultra_mcqa": ultra_scorer("mcqa"),
    "ultra_rdkit_chemistry": ultra_scorer("rdkit_chemistry"),
    "ultra_structured_outputs": ultra_scorer("structured_outputs", "openapi_schema_validator", "xmltodict", "yaml"),
    "ultra_tool_call": ultra_scorer("tool_call"),
    "arc_nvarc": NVARC,
    "skyrl_livecodebench": flat_scorer("livecodebench.py", "numpy", "pandas"),
    "skyrl_text_to_sql": flat_scorer("text_to_sql_scoring.py"),
    "skyrl_ifeval": flat_scorer("ifeval_utils.py"),
    "skyrl_apps": flat_scorer("apps_testing_util.py", "numpy"),
}

IMPORT_CHECK = """
import importlib, pathlib, sys
tests = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(tests))
for name in sys.argv[2:]:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).is_relative_to(tests), f"{name} imported from {module.__file__}"
"""


@pytest.mark.parametrize("name", sorted(CLOSURES))
def test_scorer_imports_from_shipped_files(name: str, tmp_path: Path):
    closure = CLOSURES[name]
    closure.skip_if_unavailable()
    tests = closure.stage(tmp_path)
    result = subprocess.run(
        (sys.executable, "-I", "-c", IMPORT_CHECK, str(tests), *closure.modules), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


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


@pytest.fixture(scope="module")
def original_apps_evaluator(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned upstream evaluator, recovered by reversing the committed patch."""
    if shutil.which("uv") is None:
        pytest.skip("uv is unavailable to run the original APPS evaluator")
    directory = tmp_path_factory.mktemp("apps")
    if subprocess.run(("uv", "python", "find", "--no-project", "3.10"), cwd=directory, capture_output=True).returncode:
        pytest.skip("Python 3.10 is unavailable to run the original APPS evaluator")
    shutil.copyfile(APPS_EVALUATOR, directory / APPS_EVALUATOR.name)
    subprocess.run(("git", "apply", "--reverse", str(APPS_PATCH)), cwd=directory, check=True)
    return directory / APPS_EVALUATOR.name


@pytest.mark.parametrize("name", sorted(APPS_PROBLEMS))
def test_patched_apps_evaluator_grades_fixture(name: str):
    problem = APPS_PROBLEMS[name]
    assert apps_grade((sys.executable,), APPS_EVALUATOR, problem) == problem["grade"]


@pytest.mark.parametrize("name", sorted(APPS_PROBLEMS))
def test_original_apps_evaluator_grades_fixture(name: str, original_apps_evaluator: Path):
    problem = APPS_PROBLEMS[name]
    assert apps_grade(ORIGINAL_APPS_PYTHON, original_apps_evaluator, problem) == problem["grade"]
