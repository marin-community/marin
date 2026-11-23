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
    ("answer", "expected"),
    [
        ("```text\n1 2\n3 4\n```", [[1, 2], [3, 4]]),
        ("```json\n[[1, 2], [3, 4]]\n```", [[1, 2], [3, 4]]),
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
        ("\\boxed{12\n34}", [[1, 2], [3, 4]]),
        ("<|start_think|>try a transform<|end_think|>\n```\n12\n34\n```", [[1, 2], [3, 4]]),
        ("12", [[1, 2]]),
    ],
    ids=["fenced_rows", "fenced_json", "json", "boxed_compact", "reasoning_then_fenced_compact", "single_row"],
)
def test_nvarc_transductive_accepts_common_grid_formats(answer: str, expected: list[list[int]], tmp_path: Path):
    NVARC.skip_if_unavailable()
    tests = NVARC.stage(tmp_path)
    result = subprocess.run(
        (sys.executable, "-I", "-c", NVARC_TRANSDUCTIVE_CHECK, str(tests)),
        input=answer,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


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


LIVECODEBENCH = ScorerClosure(shipped(SKYRL, "livecodebench.py"), ("numpy", "pandas"))

LIVECODEBENCH_CHECK = """
import json, sys
sys.path.insert(0, sys.argv[1])
import livecodebench
tests = json.loads(livecodebench.normalize_lcb_ground_truth(json.loads(sys.argv[2])))
bound = json.loads(sys.argv[3])
limits = None if bound is None else livecodebench.VerifierLimits(total_timeout_seconds=bound)
print(json.dumps(livecodebench.compute_score(sys.stdin.read(), tests, limits=limits)[1]))
"""

STDIN_ADD = [
    {"input": "1 2\n", "output": "3\n", "testtype": "stdin"},
    {"input": "40 2\n", "output": "42\n", "testtype": "stdin"},
]
FUNCTIONAL_ADD = [
    {"input": [1, 2], "output": 3, "testtype": "functional", "fn_name": "add"},
    {"input": [40, 2], "output": 42, "testtype": "functional", "fn_name": "add"},
]
FUNCTIONAL_RANGE = [
    {"input": [3], "output": [0, 1, 2], "testtype": "functional", "fn_name": "first"},
    {"input": [1], "output": [0], "testtype": "functional", "fn_name": "first"},
]

READS_EXPECTED_OUTPUT = """import sys

frame = sys._getframe()
while frame is not None and "gt_out" not in frame.f_locals:
    frame = frame.f_back
print(frame.f_locals["gt_out"] if frame is not None else "", end="")"""
"""Finds the evaluator's local holding the expected output of the running test and prints it."""

READS_TEST_CASES = """import sys

data = sys.stdin.read()
frame = sys._getframe()
while frame is not None:
    for value in frame.f_locals.values():
        if isinstance(value, dict) and "outputs" in value and data in value.get("inputs", ()):
            print(value["outputs"][value["inputs"].index(data)], end="")
            sys.exit()
    frame = frame.f_back"""
"""Finds any local dict of test cases on the stack and prints the output stored for its own input."""

REPORTS_ITS_VERDICT = """import gc, os
from multiprocessing.connection import Connection

for item in gc.get_objects():
    if isinstance(item, Connection) and item.writable:
        item.send(([True], {}))
        os._exit(0)"""
"""Sends a passing verdict over the verifier child's pipe and exits before any comparison."""

STALLS_MID_MESSAGE = """import gc, os, signal, time
from multiprocessing.connection import Connection

signal.signal(signal.SIGALRM, signal.SIG_IGN)
for item in gc.get_objects():
    if isinstance(item, Connection) and item.writable:
        os.write(item.fileno(), bytes(1))
        time.sleep(600)"""
"""Writes the first byte of a message to the verifier child's pipe, then stalls with the per-test alarm ignored."""

EQUAL_TO_EVERYTHING = """class _Any:
    def __eq__(self, other):
        return True

class Solution:
    def add(self, a, b):
        return _Any()"""

INT_EQUAL_TO_EVERYTHING = """class _Int(int):
    def __eq__(self, other):
        return True

class Solution:
    def add(self, a, b):
        return _Int(0)"""


def livecodebench_reward(
    tmp_path: Path, cases: list[dict], program: str, total_timeout_seconds: float | None = None
) -> float:
    LIVECODEBENCH.skip_if_unavailable()
    tests = LIVECODEBENCH.stage(tmp_path)
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            LIVECODEBENCH_CHECK,
            str(tests),
            json.dumps(cases),
            json.dumps(total_timeout_seconds),
        ),
        input=f"```python\n{program}\n```",
        capture_output=True,
        text=True,
        timeout=None if total_timeout_seconds is None else total_timeout_seconds + 60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize(
    ("cases", "program", "reward"),
    [
        (STDIN_ADD, "a, b = map(int, input().split())\nprint(a + b)", 1.0),
        (STDIN_ADD, READS_EXPECTED_OUTPUT, 0.0),
        (STDIN_ADD, READS_TEST_CASES, 0.0),
        (STDIN_ADD, REPORTS_ITS_VERDICT, 0.0),
        (FUNCTIONAL_ADD, "class Solution:\n    def add(self, a, b):\n        return a + b", 1.0),
        (FUNCTIONAL_ADD, EQUAL_TO_EVERYTHING, 0.0),
        (FUNCTIONAL_ADD, INT_EQUAL_TO_EVERYTHING, 0.0),
    ],
    ids=[
        "stdin_solves",
        "stdin_reads_expected_output",
        "stdin_reads_test_cases",
        "stdin_reports_its_verdict",
        "call_solves",
        "call_returns_object_equal_to_everything",
        "call_returns_int_equal_to_everything",
    ],
)
def test_livecodebench_program_cannot_read_expected_outputs_or_decide_comparisons(
    cases: list[dict], program: str, reward: float, tmp_path: Path
):
    assert livecodebench_reward(tmp_path, cases, program) == reward


def test_livecodebench_program_that_stalls_mid_message_fails_at_the_deadline(tmp_path: Path):
    assert livecodebench_reward(tmp_path, STDIN_ADD, STALLS_MID_MESSAGE, total_timeout_seconds=10.0) == 0.0


@pytest.mark.parametrize(
    "program",
    [
        "def first(n):\n    return tuple(range(n))",
        "import numpy as np\n\ndef first(n):\n    return list(np.arange(n))",
    ],
    ids=["tuple_for_list", "list_of_numpy_integers"],
)
def test_livecodebench_return_value_compares_as_the_value_the_program_returned(program: str, tmp_path: Path):
    assert livecodebench_reward(tmp_path, FUNCTIONAL_RANGE, program) == 1.0


ULTRA_CODE = ScorerClosure(
    {
        **ULTRA_BASE,
        **shipped(ULTRA, f"{ULTRA_ENVS}/code_gen.py"),
        "skyrl_gym/envs/lcb/livecodebench.py": SKYRL / "livecodebench.py",
    },
    LIVECODEBENCH.requires,
)
"""What the Ultra code grade script ships: ``code_gen`` imports the one vendored LiveCodeBench evaluator."""

ULTRA_CODE_CHECK = """
import json, sys, time, tracemalloc
sys.path.insert(0, sys.argv[1])
from skyrl_gym.envs.nemotron_ultra.code_gen import grade_code
tracemalloc.start()
cpu_start = time.process_time()
reward, detail = grade_code(sys.stdin.read(), {"verifier_metadata": {"unit_tests": json.loads(sys.argv[2])}})
cpu_seconds = time.process_time() - cpu_start
peak_bytes = tracemalloc.get_traced_memory()[1]
print(json.dumps([reward, detail["result"], detail["test_results"], peak_bytes, cpu_seconds], default=str))
"""
"""Grades the reply on stdin with Ultra's ``grade_code`` and prints the result with the peak memory and CPU time it
used in the grading process; values JSON cannot hold print as text, as in ``code_grade.py``."""

PRINTS_64_MIB = 'print("x" * (64 << 20))'
"""A stdin program whose output is 64 MiB, for tests that expect a few bytes."""

WRITES_COLLIDING_DICT = """import gc, json, os, struct
from multiprocessing.connection import Connection

P = (1 << 61) - 1
# The fractions (1 + 2jP) / 2 all have one hash, and comparing two of them runs Python code.
pairs = [[["fraction", 1 + 2 * j * P, 2], 0] for j in range(24000)]
frame = json.dumps(["value", ["dict", pairs], 0], separators=(",", ":")).encode()

class Solution:
    def add(self, a, b):
        for item in gc.get_objects():
            if isinstance(item, Connection) and item.writable:
                data = struct.pack("!Q", len(frame)) + frame
                while data:
                    data = data[os.write(item.fileno(), data):]
        return a + b"""
"""Writes its own result to the verifier child's pipe: a dict whose 24,000 keys share one hash, in just under 1 MiB."""

NUMPY_FRAME_ITEMS = 8000
FUNCTIONAL_ZEROS = [
    {"input": [NUMPY_FRAME_ITEMS], "output": [0] * NUMPY_FRAME_ITEMS, "testtype": "functional", "fn_name": "zeros"}
]

WRITES_NUMPY_ARRAY = f"""import gc, json, os, struct
from multiprocessing.connection import Connection

frame = json.dumps(["value", ["numpy", "float64", [[0]] * {NUMPY_FRAME_ITEMS}], 0], separators=(",", ":")).encode()

def zeros(n):
    for item in gc.get_objects():
        if isinstance(item, Connection) and item.writable:
            data = struct.pack("!Q", len(frame)) + frame
            while data:
                data = data[os.write(item.fileno(), data):]
    return [0] * n"""
"""Writes its own result to the verifier child's pipe: a NumPy float64 form holding a list of one-item lists, which
the NumPy constructor would build into an (n, 1) array, in 32 KB."""


def ultra_code_grade(tmp_path: Path, cases: list[dict], program: str) -> tuple[float, str, list, int, float]:
    ULTRA_CODE.skip_if_unavailable()
    tests = ULTRA_CODE.stage(tmp_path)
    result = subprocess.run(
        (sys.executable, "-I", "-c", ULTRA_CODE_CHECK, str(tests), json.dumps(cases)),
        input=f"```python\n{program}\n```",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    reward, outcome, test_results, peak_bytes, cpu_seconds = json.loads(result.stdout.splitlines()[-1])
    return reward, outcome, test_results, peak_bytes, cpu_seconds


def test_ultra_code_grade_scores_a_program_without_the_tested_function_zero(tmp_path: Path):
    program = "class Solution:\n    def solve(self, a, b):\n        return a + b"
    assert ultra_code_grade(tmp_path, FUNCTIONAL_ADD, program)[:3] == (0.0, "failed_tests", [-4])


def test_ultra_code_grade_fails_an_oversized_output_without_reading_it(tmp_path: Path):
    reward, outcome, test_results, peak_bytes, _ = ultra_code_grade(tmp_path, STDIN_ADD, PRINTS_64_MIB)
    assert (reward, outcome, test_results) == (0.0, "failed_tests", [-2])
    # The grading process reads the 64 MiB result in 1 MiB pieces and drops it.
    assert peak_bytes < 16 << 20


def test_ultra_code_grade_fails_a_dict_with_colliding_keys_without_building_it(tmp_path: Path):
    reward, outcome, test_results, _, cpu_seconds = ultra_code_grade(tmp_path, FUNCTIONAL_ADD, WRITES_COLLIDING_DICT)
    assert (reward, outcome, test_results) == (0.0, "failed_tests", [False])
    # Building the dict would take about 24,000 ** 2 / 2 key comparisons in the grading process.
    assert cpu_seconds < 10


def test_ultra_code_grade_fails_a_numpy_frame_holding_a_list_without_broadcasting(tmp_path: Path):
    reward, outcome, test_results, peak_bytes, _ = ultra_code_grade(tmp_path, FUNCTIONAL_ZEROS, WRITES_NUMPY_ARRAY)
    assert (reward, outcome, test_results) == (0.0, "failed_tests", [False])
    # An (n, 1) array compared with the n expected items would allocate n * n = 64 MB in the grading process.
    assert peak_bytes < 16 << 20
