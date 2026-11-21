# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge a LiveCodeBench Pro reply as the official local judge does: 1 when the program passes every test.

The program is the last ```cpp block containing ``#include``, else the ``#include`` lines and ``main``
around the last ``int main(``, as ``extract_longest_cpp_code`` in the official harness finds it. It is
compiled with ``g++ -O2 -pipe -std=gnu++17`` and run on each test in order with the problem's CPU time
limit, twice that in wall time, and its memory limit as address space and stack. The problem's testlib
checker, compiled against ``/srv/testlib``, accepts an output by exiting 0; the first failure ends judging.
Judging past ``deadline`` seconds fails, so a slow program scores 0 before the run's grading timeout.
A checker that fails to compile is an infrastructure error and raises.
"""

import json
import re
import resource
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ANSWER_PATH = Path("/app/answer.txt")
CONFIG_PATH = Path("/tests/config.json")
TESTS_PATH = Path("/tests/tests.zip")
TESTLIB_DIR = "/srv/testlib"
COMPILE = ("g++", "-O2", "-pipe", "-std=gnu++17")
COMPILE_CPU = 10
CHECKER_CPU = 10
OUTPUT_BYTES = 128 << 20

FENCED_CPP = re.compile(r"^```cpp\s*\n(.*?)\n```", re.MULTILINE | re.DOTALL)
MAIN = re.compile(r"int\s+main\s*\(")
INCLUDE_LINE = re.compile(r"^\s*#include")


def main_function_code(text: str, match: re.Match) -> str:
    """The ``#include`` lines directly above ``main`` through ``main``'s closing brace."""
    brace = text.find("{", match.end())
    if brace == -1:
        return ""
    depth, end = 0, brace
    while end < len(text):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        end += 1
        if depth == 0:
            break
    lines = text.splitlines()
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line) + 1)
    main_line = next((i for i in range(len(lines)) if starts[i] <= match.start() < starts[i + 1]), 0)
    first = None
    for index in range(main_line, -1, -1):
        if INCLUDE_LINE.match(lines[index]):
            first = index
        elif first is not None:
            break
    return text[starts[main_line if first is None else first] : end].strip()


def extract_code(text: str) -> str | None:
    for block in reversed(FENCED_CPP.findall(text)):
        if "#include" in block:
            return block.strip()
    for match in reversed(list(MAIN.finditer(text))):
        candidate = main_function_code(text, match)
        if "#include" in candidate:
            return candidate
    return None


def limits(cpu: float, memory: int | None):
    """Set the child's CPU, address-space, stack and output-file limits before it runs."""

    def apply() -> None:
        seconds = int(-(-cpu // 1))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_BYTES, OUTPUT_BYTES))
        if memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_STACK, (memory, memory))

    return apply


def run(argv: list[str], cpu: float, wall: float, memory: int | None = None, **streams) -> bool:
    """Whether ``argv`` exits 0 within its limits."""
    try:
        completed = subprocess.run(argv, timeout=wall, preexec_fn=limits(cpu, memory), check=False, **streams)
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


def compile_program(source: Path, binary: Path, *include: str) -> bool:
    argv = [*COMPILE, *(f"-I{path}" for path in include), str(source), "-o", str(binary)]
    return run(argv, COMPILE_CPU, 2 * COMPILE_CPU, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def reward(answer: str, config: dict, tests: Path, workdir: Path) -> float:
    deadline = time.monotonic() + config["deadline"]
    with zipfile.ZipFile(tests) as archive:
        archive.extractall(workdir / "tests")
    checker = workdir / "checker"
    if not compile_program(workdir / "tests" / "checker.cpp", checker, TESTLIB_DIR):
        raise RuntimeError("The problem's checker does not compile")
    code = extract_code(answer)
    if code is None:
        return 0.0
    (workdir / "main.cpp").write_text(code)
    program = workdir / "program"
    if not compile_program(workdir / "main.cpp", program):
        return 0.0
    output = workdir / "output"
    time_limit, memory = config["time_limit"], config["memory_mb"] << 20
    for test_input, test_answer in config["cases"]:
        if time.monotonic() > deadline:
            return 0.0
        test_input, test_answer = workdir / "tests" / test_input, workdir / "tests" / test_answer
        with test_input.open("rb") as stdin, output.open("wb") as stdout:
            if not run(
                [str(program)], time_limit, 2 * time_limit, memory, stdin=stdin, stdout=stdout, stderr=subprocess.DEVNULL
            ):
                return 0.0
        checked = [str(checker), str(test_input), str(output), str(test_answer)]
        if not run(checked, CHECKER_CPU, 2 * CHECKER_CPU, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
            return 0.0
    return 1.0


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    with tempfile.TemporaryDirectory() as workdir:
        score = reward(ANSWER_PATH.read_text(), config, TESTS_PATH, Path(workdir))
    sys.stdout.write(f"{score}\n")


if __name__ == "__main__":
    main()
