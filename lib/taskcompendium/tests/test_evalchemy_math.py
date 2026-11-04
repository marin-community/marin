# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source symbolic comparison remains bounded in grading workers."""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import pytest
from verifyit.grade import GradingInfraError

from taskcompendium.importers.evalchemy.grader import DEPENDENCIES, scorer_files
from taskcompendium.importers.evalchemy.scorers import minerva
from taskcompendium.importers.evalchemy.scorers.harness_generation import grade_harness_generation_candidate
from taskcompendium.importers.evalchemy.scorers.main import score
from taskcompendium.importers.evalchemy.scorers.math import grade_boxed_math
from taskcompendium.importers.evalchemy.scorers.minerva import is_equiv
from taskcompendium.importers.evalchemy.scorers.olympiad import grade_olympiad
from taskcompendium.runtime.resources import resource_bytes


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


# The reference of OlympiadBench id 1122: a relation without free symbols.
RELATION = r"I_{e}(0)=\gamma I_{i}(0)"
GRADERS = {
    "olympiad": lambda reference, response: grade_olympiad((reference,), response),
    "boxed_math": grade_boxed_math,
}


@pytest.mark.parametrize(
    ("reference", "candidate", "correct"),
    [
        # Equations that solve to no value, and other relations without free symbols, do not match.
        (RELATION, "x = x", False),
        (RELATION, "x = x + 1", False),
        (RELATION, r"I_{i}(0) = I_{e}(0)", False),
        ("3 = 4", "1 = 2", False),
        # The reference's own relation matches, either way round.
        (RELATION, r"I_{e}(0)=\gamma I_{i}(0)", True),
        (RELATION, r"\gamma I_{i}(0) = I_{e}(0)", True),
        # Other references still go through math-verify.
        ("x=1", "x = 1", True),
        ("42", "42", True),
        ("0.5", r"\frac{1}{2}", True),
    ],
)
@pytest.mark.parametrize("grade", GRADERS.values(), ids=GRADERS.keys())
def test_symbol_free_relation_reference_matches_only_its_own_relations(grade, reference, candidate, correct):
    assert grade(reference, rf"The answer is \boxed{{{candidate}}}.").reward == float(correct)


@pytest.mark.parametrize("scorer", sorted(DEPENDENCIES))
def test_scorer_package_ships_every_module_its_files_import(scorer):
    # main.py imports each scorer's module only when that scorer runs, so it is not checked here.
    shipped = {resource.path.removeprefix("source_grader/"): resource for resource in scorer_files(scorer)}
    for path, resource in shipped.items():
        if not path.endswith(".py") or path == "main.py":
            continue
        package = PurePosixPath(path).parent
        for node in ast.walk(ast.parse(resource_bytes(resource))):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                for name in [node.module] if node.module else [alias.name for alias in node.names]:
                    assert str(package / (name.replace(".", "/") + ".py")) in shipped, f"{path} imports {name}"
