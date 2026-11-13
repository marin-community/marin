# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LiveCodeBench rows convert to tasks whose packaged grader scores replies as the official harness does."""

import base64
import importlib.util
import json
import pickle
import zlib

import pytest
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ImportRejection
from taskcompendium.runtime.resources import resource_bytes

from experiments.evaluation.benchmarks.livecodebench import HERE, source
from experiments.post_training.task_curation.datasets.skyrl.code import SCORERS
from experiments.post_training.task_curation.tests.conversion import convert_row

CORRECT = "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b"
PUBLIC_ONLY = "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return 3"


def encoded_private_tests(payload: object) -> str:
    """Encode a payload the way LiveCodeBench stores ``private_test_cases``."""
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode()


def source_row(private_test_cases: str) -> dict:
    return {
        "question_title": "Add",
        "question_content": "Return the sum of a and b.",
        "platform": "leetcode",
        "question_id": "9001",
        "contest_id": "weekly-contest-1",
        "contest_date": "2025-01-04T00:00:00",
        "starter_code": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        ",
        "difficulty": "easy",
        "public_test_cases": json.dumps([{"input": "1\n2", "output": "3", "testtype": "functional"}]),
        "private_test_cases": private_test_cases,
        "metadata": json.dumps({"func_name": "add"}),
    }


def converted(row: dict) -> TaskSpec | ImportRejection:
    dataset = source().config
    assert dataset is not None
    return convert_row(dataset.recipe, row)


def grade(task: TaskSpec, reply: str) -> float:
    """Run the task's packaged grade script on ``reply`` with its packaged tests and config."""
    files = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    spec = importlib.util.spec_from_file_location("livecodebench_grade", HERE / "livecodebench_grade.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(files["config.json"])
    cases = json.loads(zlib.decompress(files["tests.json.zlib"]))
    return module.reward(reply, cases, config["per_test_timeout"], config["deadline"])


@pytest.mark.parametrize(
    ("reply", "reward"),
    [
        (f"```python\n{CORRECT}\n```", 1.0),
        (f"```python\n{PUBLIC_ONLY}\n```", 0.0),
        (f"Draft:\n```python\n{PUBLIC_ONLY}\n```\nFixed:\n```python\n{CORRECT}\n```", 1.0),
        (f"Draft:\n```python\n{CORRECT}\n```\nFinal:\n```python\n{PUBLIC_ONLY}\n```", 0.0),
    ],
)
def test_grader_scores_the_last_program_on_public_and_private_tests(monkeypatch, reply, reward):
    monkeypatch.syspath_prepend(str(SCORERS))
    private = json.dumps([{"input": "-5\n5", "output": "0", "testtype": "functional"}])
    task = converted(source_row(encoded_private_tests(private)))
    assert isinstance(task, TaskSpec)
    assert grade(task, reply) == reward


class _ReferencesGlobal:
    def __reduce__(self):
        return (print, ("payload executed",))


def test_private_tests_that_reference_a_global_are_rejected_without_running(capsys):
    result = converted(source_row(encoded_private_tests(_ReferencesGlobal())))
    assert isinstance(result, ImportRejection)
    assert "payload executed" not in capsys.readouterr().out
