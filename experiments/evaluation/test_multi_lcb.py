# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import sys
import zlib

import pytest
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from experiments.evaluation.benchmarks.multi_lcb import CPP, HERE, SCORERS, source
from experiments.post_training.task_curation.tests.conversion import convert_row

CORRECT = "#include <iostream>\nint main() { long a, b; std::cin >> a >> b; std::cout << a + b << std::endl; }"
PUBLIC_ONLY = "#include <iostream>\nint main() { std::cout << 3 << std::endl; }"
QUESTION = (
    "Return the sum of a and b.\n\nExample 1:\n\nInput: a = 1, b = 2\nOutput: 3\n\nConstraints:\n\n-10 <= a, b <= 10"
)


def source_row() -> dict:
    return {
        "question_title": "Add",
        "question_content": QUESTION,
        "platform": "leetcode",
        "question_id": "9001",
        "contest_id": "weekly-contest-1",
        "contest_date": "2025-01-04T00:00:00",
        "starter_code": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        ",
        "difficulty": "easy",
        "public_test_cases": json.dumps([{"input": "1\n2", "output": "3", "testtype": "functional"}]),
        "private_test_cases": json.dumps([{"input": "-5\n5", "output": "0", "testtype": "functional"}]),
        "metadata": json.dumps({"func_name": "add"}),
    }


def converted() -> TaskSpec:
    dataset = source(CPP).config
    assert dataset is not None
    task = convert_row(dataset.recipe, source_row())
    assert isinstance(task, TaskSpec)
    return task


def grade(monkeypatch, reply: str) -> bool:
    """Run the grade script's harness call on ``reply`` with the task's packaged tests and config."""
    monkeypatch.syspath_prepend(str(SCORERS))
    monkeypatch.setenv("PATH", os.environ["PATH"])
    files = {resource.path: resource_bytes(resource) for resource in converted().resources.verifier}
    spec = importlib.util.spec_from_file_location("multi_lcb_grade", HERE / "multi_lcb_grade.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_pkill()
    config = json.loads(files["config.json"])
    tests = json.loads(zlib.decompress(files["tests.json.zlib"]))
    return module.passes(module.extract_code(reply), tests, config["language"])


LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="the harness sets RLIMIT_AS, which macOS rejects")


@LINUX_ONLY
@pytest.mark.parametrize(("program", "passed"), [(CORRECT, True), (PUBLIC_ONLY, False)])
def test_grader_runs_the_program_on_converted_public_and_private_tests(monkeypatch, program, passed):
    assert grade(monkeypatch, f"Here it is.\n```cpp\n{program}\n```") is passed


def test_leetcode_prompt_shows_examples_as_stdin():
    user = converted().context.events[1].content
    assert "Sample Input 1:\n\n1\n2\n\nSample Output 1:\n\n3" in user
    assert "```cpp\n// YOUR CODE HERE\n```" in user
