# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import json
import shutil
import sys
import zipfile

import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.runtime.resources import resource_bytes

from experiments.evaluation.benchmarks.livecodebench_pro import HERE, PROMPT, TESTS_INPUT, source
from experiments.post_training.task_curation.tests.conversion import convert_row, fixture_context

PROBLEM = "1983A"
CONFIG = (
    "type: default\ntime_limit: 1s\nmemory_limit: 256m\nchecker: checker.cpp\ninput_prefix: ''\n"
    "output_prefix: ''\ninput_suffix: .in\noutput_suffix: .ans\nsubtasks:\n  - score: 100\n    n_cases: 2\n"
)
# Accepts an output whose single integer equals the answer's; no testlib needed.
CHECKER = (
    "#include <fstream>\n"
    "int main(int argc, char** argv) {\n"
    "  std::ifstream out(argv[2]), ans(argv[3]);\n"
    "  long long got, want;\n"
    "  return (out >> got) && (ans >> want) && got == want ? 0 : 1;\n"
    "}\n"
)
SUM = "#include <iostream>\nint main() { long long a, b; std::cin >> a >> b; std::cout << a + b << std::endl; }"
DIFFERENCE = "#include <iostream>\nint main() { long long a, b; std::cin >> a >> b; std::cout << a - b << std::endl; }"
SPIN = "#include <iostream>\nint main() { volatile unsigned long long i = 0; while (true) ++i; }"
CRASH = "#include <cstdlib>\nint main() { std::abort(); }"
SLOW_SUM = (
    "#include <ctime>\n#include <iostream>\n"
    "int main() { while (std::clock() < 3 * CLOCKS_PER_SEC / 4) {}\n"
    "long long a, b; std::cin >> a >> b; std::cout << a + b << std::endl; }"
)


def grader():
    spec = importlib.util.spec_from_file_location("livecodebench_pro_grade", HERE / "livecodebench_pro_grade.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TESTS = {"1.in": "1 2\n", "1.ans": "3\n", "2.in": "-5 5\n", "2.ans": "0\n"}


def archive_bytes(checker: str = CHECKER, tests: dict[str, str] = TESTS) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("config.yaml", CONFIG)
        archive.writestr("checker.cpp", checker)
        for name, text in tests.items():
            archive.writestr(f"testdata/{name}", text)
    return buffer.getvalue()


def source_row(statement: str = "Print a + b.") -> dict:
    return {
        "problem_id": PROBLEM,
        "problem_title": "Sum",
        "difficulty": "easy",
        "problem_statement": statement,
        "platform": "codeforces",
        "link": "https://codeforces.com/problemset/problem/1983/A",
        "time_limit": 1000,
        "memory_limit": 256,
    }


def recipe():
    dataset = source().config
    assert dataset is not None
    return dataset.recipe


def selected(tmp_path, row: dict, archive: bytes | None = None) -> bool:
    (tmp_path / f"{PROBLEM}.zip").write_bytes(archive if archive is not None else archive_bytes())
    context = fixture_context(recipe(), {TESTS_INPUT: StoragePath(str(tmp_path))})
    return recipe().source.select(row, context)


@pytest.mark.parametrize(
    ("reply", "program"),
    [
        (f"Plan.\n```cpp\n{DIFFERENCE}\n```\nFixed:\n```cpp\n{SUM}\n```", SUM),
        (f"```cpp\n{SUM}\n```\nUnused helper:\n```cpp\nint helper();\n```", SUM),
        (f"Final program:\n{SUM}\nDone.", SUM),
        ("```python\nprint(3)\n```", None),
    ],
)
def test_extraction_takes_the_last_cpp_block_with_an_include_else_the_code_around_main(reply, program):
    assert grader().extract_code(reply) == program


def test_conversion_prompts_with_the_official_prefix_and_ships_the_archive_and_its_cases(tmp_path):
    (tmp_path / f"{PROBLEM}.zip").write_bytes(archive_bytes())
    task = convert_row(recipe(), source_row(), inputs={TESTS_INPUT: StoragePath(str(tmp_path))})

    assert task.context.events[0].content == PROMPT + "Print a + b."
    files = {resource.path: resource for resource in task.resources.verifier}
    config = json.loads(resource_bytes(files["config.json"]))
    assert config["cases"] == [["testdata/1.in", "testdata/1.ans"], ["testdata/2.in", "testdata/2.ans"]]
    assert (config["time_limit"], config["memory_mb"]) == (1.0, 256)


def test_interactive_problems_are_not_selected(tmp_path):
    assert selected(tmp_path, source_row())
    assert not selected(tmp_path, source_row("This is an interactive problem.\n\nGuess the number."))


@pytest.mark.parametrize(("reads_answer", "judgeable"), [(True, False), (False, True)])
def test_tests_that_share_an_input_must_agree_when_the_checker_reads_the_answer(tmp_path, reads_answer, judgeable):
    checker = "int main() { long long x = ans.readLong(); }" if reads_answer else "int main() { ouf.readLong(); }"
    tests = {"1.in": "1 2\n", "1.ans": "3\n", "2.in": "1 2\n", "2.ans": "4\n"}

    assert selected(tmp_path, source_row(), archive_bytes(checker, tests)) == judgeable


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("g++") is None, reason="judging sets Linux rlimits and compiles with g++"
)
@pytest.mark.parametrize(
    ("reply", "reward"),
    [
        (f"```cpp\n{SUM}\n```", 1.0),
        (f"```cpp\n{DIFFERENCE}\n```", 0.0),
        (f"```cpp\n{SPIN}\n```", 0.0),
        (f"```cpp\n{CRASH}\n```", 0.0),
        ("```cpp\n#include <iostream>\nint main( {\n```", 0.0),
        ("No code.", 0.0),
    ],
)
def test_judge_accepts_only_a_program_that_passes_every_test(tmp_path, reply, reward):
    assert judge(tmp_path, reply, time_limit=1.0) == reward


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("g++") is None, reason="judging sets Linux rlimits and compiles with g++"
)
@pytest.mark.parametrize(("program", "reward"), [(SUM, 1.0), (SLOW_SUM, 0.0)])
def test_judge_enforces_a_fractional_time_limit(tmp_path, program, reward):
    assert judge(tmp_path, f"```cpp\n{program}\n```", time_limit=0.5) == reward


def judge(tmp_path, reply: str, time_limit: float) -> float:
    archive = tmp_path / "tests.zip"
    archive.write_bytes(archive_bytes())
    config = {
        "cases": [["testdata/1.in", "testdata/1.ans"], ["testdata/2.in", "testdata/2.ans"]],
        "time_limit": time_limit,
        "memory_mb": 256,
        "deadline": 60,
    }
    workdir = tmp_path / "work"
    workdir.mkdir()
    return grader().reward(reply, config, archive, workdir)
