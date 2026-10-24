# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.weight_merging.playground.evaluation import (
    numeric_answer,
    paired_summary,
    python_code,
    score,
    sign_test_pvalue,
    unclosed_thinking,
    validate_vocabulary,
)


def test_gsm_extracts_marked_final_answer_over_intermediate_numbers():
    assert numeric_answer("12 + 8 = 20, then double. #### 40") == numeric_answer("40")
    assert numeric_answer("Answer: \\boxed{1,234.5}") == numeric_answer("1234.5")
    assert numeric_answer("no numerical answer") is None


def test_code_extracts_fenced_python():
    assert python_code("Here is code:\n```python\ndef f(x):\n    return x + 1\n```") == "def f(x):\n    return x + 1\n"


def test_paired_summary_compares_matching_ids_and_seeds():
    rows = [
        {"candidate": "base", "dataset": "gsm8k", "mode": "sample", "seed": 17, "item_id": "a", "correct": False},
        {"candidate": "base", "dataset": "gsm8k", "mode": "sample", "seed": 29, "item_id": "a", "correct": True},
        {"candidate": "merge", "dataset": "gsm8k", "mode": "sample", "seed": 17, "item_id": "a", "correct": True},
        {"candidate": "merge", "dataset": "gsm8k", "mode": "sample", "seed": 29, "item_id": "a", "correct": False},
        {"candidate": "merge", "dataset": "gsm8k", "mode": "sample", "seed": 17, "item_id": "b", "correct": True},
    ]
    summary = next(
        r
        for r in paired_summary(rows, "base")
        if (r["candidate"], r["dataset"], r["mode"]) == ("merge", "gsm8k", "sample")
    )
    assert (summary["n"], summary["correct"], summary["paired_n"], summary["wins"], summary["losses"]) == (3, 2, 2, 1, 1)


@pytest.mark.docker
def test_mbpp_runs_tests_and_rejects_wrong_function_in_container():
    item = {"dataset": "mbpp", "setup": [], "tests": ["assert f(3) == 4"]}
    assert score(item, "def f(x):\n    return x + 1")["correct"]
    assert not score(item, "def f(x):\n    return x - 1")["correct"]


def test_vocabulary_allows_only_an_added_padding_row():
    original = {"a": 0, "eos": 1}
    validate_vocabulary(original, original, "eos")
    validate_vocabulary(original, {**original, "pad": 2}, "pad")
    for incompatible in ({"a": 1, "eos": 0}, {**original, "pad": 0}, {**original, "think": 2}):
        with pytest.raises(ValueError):
            validate_vocabulary(original, incompatible, "pad")


def test_exact_sign_test_uses_discordant_pairs_only():
    assert sign_test_pvalue(0, 0) == 1
    assert sign_test_pvalue(5, 5) == 1
    assert sign_test_pvalue(10, 0) == pytest.approx(2 / 1024)
    assert sign_test_pvalue(1, 9) == pytest.approx(22 / 1024)
    assert sign_test_pvalue(9, 1) == sign_test_pvalue(1, 9)


def test_unclosed_thinking_tracks_multiple_sections():
    assert not unclosed_thinking("plain answer")
    assert not unclosed_thinking("<think>reasoning</think>answer")
    assert unclosed_thinking("<think>unfinished")
    assert unclosed_thinking("<think>done</think><think>unfinished")
    assert unclosed_thinking("</think><think>unfinished")
