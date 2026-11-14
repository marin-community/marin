# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source symbolic comparison remains bounded in grading workers."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from verifyit.grade import GradingInfraError

from taskcompendium.importers.evalchemy.scorers import minerva
from taskcompendium.importers.evalchemy.scorers.fewshot import score
from taskcompendium.importers.evalchemy.scorers.harness_generation import grade_harness_generation_candidate
from taskcompendium.importers.evalchemy.scorers.minerva import is_equiv


def test_harness_math_symbolic_fallback_grades_correctly_in_a_worker_thread():
    answer = "1.2345678901234567"
    with ThreadPoolExecutor(max_workers=1) as workers:
        result = workers.submit(grade_harness_generation_candidate, "gsm8k", ((answer,),), "#### " + answer, "").result(
            timeout=15
        )
    assert result.reward == 1
    assert result.detail["metrics"] == {
        "exact_match,strict-match": 1.0,
        "exact_match,flexible-extract": 1.0,
    }


@pytest.mark.parametrize(
    ("response", "reference", "question", "extracted", "expected"),
    [
        (r"Reasoning considered 25. Final answer: \boxed{\$600}", "25", "How many TVs?", r"\$600", 0),
        (r"The answer is \boxed{11:30}", "11", "What is the time?", "11:30", 0),
        (r"The answer is \boxed{2:00}", "2", "What is the time?", "2:00", 1),
        (r"The answer is \boxed{\frac{1}{4}}", "25", "What percent?", r"\frac{1}{4}", 1),
        (r"The answer is \boxed{3 + 1}", "3", "How many?", "3 + 1", 0),
        (r"The answer is \boxed{3 + 1}", "4", "How many?", "3 + 1", 1),
        (r"The answer is \boxed{448 mL}", "448", "What volume?", "448 mL", 1),
    ],
)
def test_gsm8k_flexible_grader_preserves_complete_final_box(response, reference, question, extracted, expected):
    verdict = score(
        "harness_generation",
        {"profile": "gsm8k", "references": ((reference,),), "question": question},
        response,
    )
    assert verdict.detail["extracted"]["flexible-extract"] == extracted
    assert verdict.detail["metrics"]["exact_match,flexible-extract"] == expected
    assert verdict.reward == expected


@pytest.mark.parametrize("threaded", [False, True])
def test_minerva_symbolic_timeout_is_a_nonmatch(threaded):
    if threaded:
        with ThreadPoolExecutor(max_workers=1) as workers:
            result = workers.submit(is_equiv, r"9^{9^{9}}", "2").result(timeout=45)
    else:
        result = is_equiv(r"9^{9^{9}}", "2")
    assert result is False


def test_minerva_unexpected_sympy_error_does_not_score_as_wrong(monkeypatch):
    def fail(_expression):
        raise RuntimeError("sympy unavailable")

    monkeypatch.setattr(minerva.sympy, "simplify", fail)
    with pytest.raises(RuntimeError, match="sympy unavailable"):
        is_equiv(r"\sqrt{4}", "2")


@pytest.mark.slow
@pytest.mark.timeout(90)
def test_minerva_concurrent_workers_complete_symbolic_matches():
    with ThreadPoolExecutor(max_workers=48) as workers:
        results = list(workers.map(is_equiv, [r"\sqrt{4}"] * 48, ["2"] * 48))
    assert results == [True] * 48


def test_minerva_worker_crash_remains_an_infrastructure_failure(tmp_path, monkeypatch):
    (tmp_path / "sitecustomize.py").write_text("import os; os._exit(23)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with ThreadPoolExecutor(max_workers=1) as workers:
        with pytest.raises(GradingInfraError):
            workers.submit(is_equiv, r"\sqrt{4}", "2").result(timeout=45)
