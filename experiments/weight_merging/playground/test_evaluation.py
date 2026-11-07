# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from types import SimpleNamespace

import pytest

from experiments.weight_merging.playground.evaluation import (
    grade_completion,
    item_bootstrap,
    numeric_answer,
    paired_summary,
    python_code,
    remove_timed_out_container,
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


def test_deferred_code_grading_does_not_execute_generated_code(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU generation must not execute code")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    assert grade_completion({"dataset": "mbpp"}, "print('generated')", defer_code=True) == {
        "correct": None,
        "pending_execution": True,
    }
    assert grade_completion({"dataset": "gsm8k", "answer": "42"}, "#### 42", defer_code=True)["correct"]


def test_bootstrap_resamples_whole_items_instead_of_individual_decodes():
    summary = item_bootstrap([[0, 0, 0], [1, 1, 1]], resamples=200)
    assert summary["paired_items"] == 2
    assert summary["mean_item_gain"] == 0.5
    assert summary["item_bootstrap_ci95"] == [0, 1]


@pytest.mark.parametrize("message", ["removal of container NAME is already in progress", "No such container: NAME"])
def test_timeout_auto_remove_race_remains_a_failed_grade_after_disappearance(monkeypatch, message):
    polls = iter([True, False])

    class Container:
        attempts = 0

        def communicate(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.TimeoutExpired("docker run", 10)

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: Container())
    monkeypatch.setattr("time.sleep", lambda _: None)

    def run(command, **kwargs):
        if command[1] == "rm":
            raise subprocess.CalledProcessError(1, command, stderr=message.replace("NAME", command[-1]))
        assert command[1] == "ps" and command[-1] == "{{.Names}}"
        name_filter = command[command.index("--filter") + 1]
        assert name_filter.startswith("name=merge-eval-")
        return SimpleNamespace(stdout=name_filter.removeprefix("name=") + "\n" if next(polls) else "")

    monkeypatch.setattr("subprocess.run", run)
    assert score({"dataset": "mbpp", "setup": [], "tests": []}, "while True: pass") == {
        "correct": False,
        "error": "timeout",
    }
    with pytest.raises(StopIteration):
        next(polls)


def test_cleanup_race_does_not_accept_a_remaining_container_or_other_daemon_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    def run(command, **kwargs):
        if command[1] == "rm":
            raise subprocess.CalledProcessError(1, command, stderr="No such container: merge-eval-test")
        return SimpleNamespace(stdout="merge-eval-test\n")

    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(RuntimeError, match="did not remove"):
        remove_timed_out_container("merge-eval-test")

    def daemon_error(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="permission denied")

    monkeypatch.setattr("subprocess.run", daemon_error)
    with pytest.raises(RuntimeError, match="permission denied"):
        remove_timed_out_container("merge-eval-test")
