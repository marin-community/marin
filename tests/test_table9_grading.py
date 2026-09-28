# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pytest

pytest.importorskip("math_verify")

from experiments.domain_phase_mix import evaluate_table9_accuracy as inference
from experiments.domain_phase_mix.grade_table9_accuracy import grade_math, grade_task, graded_result, python_program


def test_math_equivalent_answer_and_wrong_answer():
    sample = {
        "metadata": {"solution_text": r"The answer is $\boxed{\frac{1}{2}}$."},
        "generation": r"Final Answer: The final answer is $0.5$. I hope it is correct.",
    }
    assert grade_math(sample)["math_verify"] == 1.0
    sample["generation"] = r"Final Answer: The final answer is $7$. I hope it is correct."
    assert grade_math(sample) == {"exact_match": 0.0, "math_verify": 0.0}


def test_code_completion_preserves_function_indentation_and_excludes_fence():
    sample = {"task": "codex_humaneval", "metadata": {"answer_prefix": "def f(x):\n", "test": "assert f(2) == 3"}}
    program, _ = python_program(sample, "    return x + 1\n```\nignore this")
    assert program == "def f(x):\n    return x + 1\n\nassert f(2) == 3\n"


MBPP_TEST = 'assert remove_dirty_chars("probasscurve", "pros") == "bacuve"\nassert remove_dirty_chars("exoticmiles", "toxic") == "emles"'


def mbpp_program(generation: str) -> str:
    program, _ = python_program({"task": "mbpp", "metadata": {"test": MBPP_TEST}}, generation + "```")
    return program


def test_mbpp_tests_call_the_completions_entry_point_under_its_own_name():
    # The helper is defined last, so only the entry-point rule binds the tested name to remove_chars.
    generation = (
        "def remove_chars(s, banned):\n    return ''.join(c for c in s if keep(c, banned))\n\n"
        "def keep(c, banned):\n    return c not in banned\n"
    )
    exec(mbpp_program(generation), {})


def test_mbpp_binding_does_not_pass_wrong_or_call_incompatible_completions():
    with pytest.raises(AssertionError):
        exec(mbpp_program("def remove_chars(s, banned):\n    return s\n"), {})
    with pytest.raises(NameError):
        exec(mbpp_program("def remove_chars(s):\n    return s\n"), {})


def test_generation_is_not_accuracy_until_separate_grading_completes(tmp_path):
    task = "minerva_math_algebra"
    row = {"name": "checkpoint"}
    plan = {"output_root": str(tmp_path), "rows": [row], "request_manifest": {"tasks": {task: {"count": 1}}}}
    root = inference.result_root(plan, row, task, 0)
    samples = [{"task": task, "doc_id": 0, "metadata": {"solution_text": r"$\boxed{2}$"}, "generation": r"$\boxed{2}$"}]
    artifact = inference.write_verified(root + "/samples.json.gz", gzip.compress(json.dumps(samples).encode()))
    marker = {
        "row": row,
        "protocol_sha256": inference.digest(inference.protocol(plan)),
        "limit": 0,
        "task": task,
        "count": 1,
        "stage": "generated_ungraded",
        "artifact": artifact,
    }
    inference.write_verified(root + "/SUCCESS.json", inference.existing.canonical_json(marker))
    assert graded_result(plan, row, task, 0) is None
    result = grade_task(plan, row, task, 0)
    assert result["metrics"]["math_verify"] == 1.0
    assert graded_result(plan, row, task, 0) == result
    assert grade_task(plan, row, task, 0) == result
