# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Completion scorer packages execute from their shipped resources."""

import json
import os
import subprocess
import sys

import pytest

from taskcompendium.importers.evalchemy.fewshot_grader import source_package
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.runtime.resources import resource_bytes


@pytest.mark.parametrize(
    "scorer,config,correct,wrong",
    [
        ("gsm8k_cot", {"document": {"answer": "Two.\n#### 2"}}, "The answer is 2.", "The answer is 3."),
        (
            "minerva_math",
            {"document": {"solution": r"\boxed{2}"}},
            "Final Answer: The final answer is 2. I hope it is correct.",
            "Final Answer: The final answer is 3. I hope it is correct.",
        ),
        ("bbh_cot", {"document": {"target": "True"}}, "the answer is True.", "the answer is False."),
        (
            "harness_generation",
            {"profile": "gsm8k", "references": [["25"]], "question": "What percent?"},
            r"The answer is \boxed{\frac{1}{4}}",
            r"The answer is \boxed{3}",
        ),
    ],
)
def test_packaged_completion_scorer_grades_answers(tmp_path, scorer, config, correct, wrong):
    package = source_package(
        scorer,
        config,
        environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="fixture.lock"
        ),
    )
    for resource in package.resources:
        destination = tmp_path / resource.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(resource_bytes(resource))
    module = package.grader.argv[2]
    program = (
        "import importlib,json,sys; "
        "module=importlib.import_module(sys.argv[1]); "
        "print(module.score(sys.argv[2],json.loads(sys.argv[3]),sys.argv[4]).reward)"
    )
    for response, expected in ((correct, 1.0), (wrong, 0.0)):
        result = subprocess.run(
            [sys.executable, "-c", program, module, scorer, json.dumps(config), response],
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert float(result.stdout) == expected
