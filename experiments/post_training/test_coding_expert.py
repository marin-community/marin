# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.coding_expert import CodeProblem, normalized_prompt_fingerprint, partition_problems


def test_partition_keeps_related_problems_in_one_split() -> None:
    problems = [
        CodeProblem("a", "source:shared", normalized_prompt_fingerprint("first")),
        CodeProblem("b", "source:shared", normalized_prompt_fingerprint("second")),
        CodeProblem("c", "source:c", normalized_prompt_fingerprint("duplicate prompt")),
        CodeProblem("d", "source:d", normalized_prompt_fingerprint(" duplicate   prompt ")),
        CodeProblem("e", None, normalized_prompt_fingerprint("fifth")),
        CodeProblem("f", None, normalized_prompt_fingerprint("sixth")),
        CodeProblem("g", None, normalized_prompt_fingerprint("seventh")),
        CodeProblem("h", None, normalized_prompt_fingerprint("eighth")),
    ]

    splits = partition_problems(problems, {"final": 2, "development": 2, "train": 4}, seed=9528)

    assert {name: len(selected) for name, selected in splits.items()} == {
        "final": 2,
        "development": 2,
        "train": 4,
    }
    split_by_id = {
        problem.problem_id: split for split, selected in splits.items() for problem in selected
    }
    assert split_by_id["a"] == split_by_id["b"]
    assert split_by_id["c"] == split_by_id["d"]


def test_partition_is_stable_under_input_order() -> None:
    problems = [
        CodeProblem(str(index), None, normalized_prompt_fingerprint(f"problem {index}")) for index in range(12)
    ]

    forward = partition_problems(problems, {"final": 2, "development": 2, "train": 4}, seed=17)
    reverse = partition_problems(reversed(problems), {"final": 2, "development": 2, "train": 4}, seed=17)

    assert {
        split: {problem.problem_id for problem in selected} for split, selected in forward.items()
    } == {split: {problem.problem_id for problem in selected} for split, selected in reverse.items()}


def test_verifier_accepts_correct_code_and_bounds_nonterminating_code() -> None:
    verifier = pytest.importorskip("experiments.post_training.coding_expert_verifier")

    cases = (verifier.CodeCase(b"2 3\n", "5\n"),)

    assert verifier.verify_source("a, b = map(int, input().split()); print(a + b)", cases).reward == 1
    assert verifier.verify_source("print(0)", cases).reward == 0
    assert verifier.verify_source("while True: pass", cases).reward == 0


def test_verifier_uses_a_fresh_simulation_for_each_case() -> None:
    verifier = pytest.importorskip("experiments.post_training.coding_expert_verifier")

    source = """\
try:
    open("/tmp/marker").read()
except FileNotFoundError:
    print("fresh")
    open("/tmp/marker", "w").write("seen")
else:
    print("stale")
"""
    cases = (verifier.CodeCase(b"", "fresh\n"), verifier.CodeCase(b"", "fresh\n"))

    assert verifier.verify_source(source, cases).reward == 1
