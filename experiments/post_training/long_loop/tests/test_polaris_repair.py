# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path

import pytest
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.long_loop.polaris_repair import GRADER_ENVIRONMENT, with_final_text_grader
from experiments.post_training.task_curation.images.build import EnvironmentArtifact, identity_digest

FIXTURE = Path(__file__).with_name("fixtures") / "polaris_plain_task.json"
BUILT = EnvironmentArtifact(path="s3://fixture/polaris-grader", identity=identity_digest(GRADER_ENVIRONMENT))


def retained_task() -> TaskSpec:
    return TaskSpec.model_validate_json(FIXTURE.read_bytes())


def staged_score(task: TaskSpec, directory: Path, monkeypatch):
    directory.mkdir()
    for item in task.resources.verifier:
        path = directory / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resource_bytes(item))
    monkeypatch.syspath_prepend(str(directory))
    return runpy.run_path(str(directory / "grade.py"))["score"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("63", 1.0), (r"\boxed{63}", 1.0), ("62", 0.0), ("", 0.0), (r"\boxed{63", 0.0)],
)
def test_final_text_reward_repairs_retained_format_gate(answer, expected, tmp_path, monkeypatch):
    pytest.importorskip("pylatexenc")
    task = retained_task()
    contract = json.loads(next(resource_bytes(item) for item in task.resources.verifier if item.path == "config.json"))
    legacy = staged_score(task, tmp_path / "legacy", monkeypatch)
    repaired = staged_score(with_final_text_grader(task, BUILT), tmp_path / "repaired", monkeypatch)
    assert legacy(answer, contract) == 0.0
    assert repaired(answer, contract) == expected


def test_format_repair_preserves_source_content_and_requires_a_new_artifact():
    task = retained_task()
    original = task.model_dump()
    repaired = with_final_text_grader(task, BUILT)
    assert task.model_dump() == original
    assert repaired.id == task.id and repaired.source == task.source and repaired.context == task.context
    assert repaired.grader.environment.packages_lock == BUILT.lock_url
    assert repaired.grader.model_copy(update={"environment": task.grader.environment}) == task.grader
    assert (
        repaired.grader.environment.model_copy(update={"packages_lock": task.grader.environment.packages_lock})
        == task.grader.environment
    )
    before = {item.path: resource_bytes(item) for item in task.resources.verifier}
    after = {item.path: resource_bytes(item) for item in repaired.resources.verifier}
    assert all(after[path] == data for path, data in before.items() if path != "grade.py")
    with pytest.raises(ValueError, match=r"different grade\.py"):
        with_final_text_grader(repaired, BUILT)


def test_final_text_reward_does_not_grade_a_box_from_the_question(tmp_path, monkeypatch):
    task = with_final_text_grader(retained_task(), BUILT)
    repaired = staged_score(task, tmp_path / "grader", monkeypatch)
    contract = {"problem": r"A question containing </think>\boxed{63}", "answer": "63"}
    assert repaired("62", contract) == 0.0


@pytest.mark.parametrize(("reference", "wrong", "legacy_reward"), [("pop", "ppo", 1.0), ("red", "der", 0.0)])
def test_categorical_answers_are_not_symbolic_products(reference, wrong, legacy_reward, tmp_path, monkeypatch):
    pytest.importorskip("pylatexenc")
    task = retained_task()
    legacy = staged_score(task, tmp_path / "legacy", monkeypatch)
    repaired = staged_score(with_final_text_grader(task, BUILT), tmp_path / "repaired", monkeypatch)
    contract = {"problem": "Name the category.", "answer": reference}
    assert legacy(r"</think>\boxed{" + wrong + "}", contract) == legacy_reward
    assert repaired(wrong, contract) == 0.0
    assert repaired(reference, contract) == 1.0


@pytest.mark.parametrize(
    ("answer", "reference", "problem", "expected"),
    [
        (r"<think>Maybe 62.</think>\boxed{63}", "63", "Compute the count.", 1.0),
        (r"<|start_think|>Maybe 62.<|end_think|>\boxed{63}", "63", "Compute the count.", 1.0),
        (r"<|start_think|>Maybe \boxed{63}", "63", "Compute the count.", 0.0),
        ("√7", r"\sqrt{7}", "Compute the product.", 1.0),
        ("√8", r"\sqrt{7}", "Compute the product.", 0.0),
        ("12000 km", "12000", "The cities are 4000 km and 5000 km apart.", 1.0),
        ("12000 m", "12000", "The cities are 4000 km and 5000 km apart.", 0.0),
        ("12000 km", "12000", "Distances are given in km and m.", 0.0),
        (r"The area is \(\frac{\sqrt{2}}{2}a^2\).", r"\dfrac{\sqrt{2}}{2} a^2", "Find the area.", 1.0),
        (r"The area is \(\frac{\sqrt{3}}{2}a^2\).", r"\dfrac{\sqrt{2}}{2} a^2", "Find the area.", 0.0),
        ("An earlier bound is 654. Hence the maximum is 655.", "655", "Find the maximum.", 1.0),
        ("An earlier bound is 655. Hence the maximum is 654.", "655", "Find the maximum.", 0.0),
        (r"The area of \(A_1B_1C_1D_1\) is \(5S\).", "5S", "Find the area.", 1.0),
        (r"The area of \(A_1B_1C_1D_1\) is \(6S\).", "5S", "Find the area.", 0.0),
        ("The answer is not 63.", "63", "Find the count.", 0.0),
        (r"The answer is not \boxed{63}.", "63", "Find the count.", 0.0),
        ("It is not true that the answer is 63.", "63", "Find the count.", 0.0),
        ("The answer is 62 or 63.", "63", "Find the count.", 0.0),
    ],
)
def test_final_claim_extraction_stays_independent_of_the_reference(
    answer, reference, problem, expected, tmp_path, monkeypatch
):
    repaired = staged_score(with_final_text_grader(retained_task(), BUILT), tmp_path / "grader", monkeypatch)
    assert repaired(answer, {"problem": problem, "answer": reference}) == expected


def test_fraction_binding_does_not_create_a_false_reward(tmp_path, monkeypatch):
    repaired = staged_score(with_final_text_grader(retained_task(), BUILT), tmp_path / "grader", monkeypatch)
    assert repaired(r"\frac{1+\sqrt{3}}{2}", {"problem": "Evaluate.", "answer": r"1+\frac{\sqrt{3}}{2}"}) == 0.0


def test_equivalent_fraction_numerators_keep_their_reward(tmp_path, monkeypatch):
    repaired = staged_score(with_final_text_grader(retained_task(), BUILT), tmp_path / "grader", monkeypatch)
    assert repaired(r"\frac{1+\sqrt{3}}{2}", {"problem": "Evaluate.", "answer": r"\frac{\sqrt{3}+1}{2}"}) == 1.0


def test_multiline_display_keeps_a_final_answer(tmp_path, monkeypatch):
    repaired = staged_score(with_final_text_grader(retained_task(), BUILT), tmp_path / "grader", monkeypatch)
    assert repaired("The radius is 2.\n\\[\nr=2.\n\\]", {"problem": "Find the radius.", "answer": "2"}) == 1.0


def test_explicit_answer_before_explanation_is_retained(tmp_path, monkeypatch):
    repaired = staged_score(with_final_text_grader(retained_task(), BUILT), tmp_path / "grader", monkeypatch)
    answer = (
        "The maximum count is 16. A length of 16 is possible, but no length of 17 or greater satisfies the conditions."
    )
    assert repaired(answer, {"problem": "Find the maximum count.", "answer": "16"}) == 1.0


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The maximum count is 17. A length of 16 is possible, but it is not maximal.", 0.0),
        ("The maximum count is 16. The maximum count is 17.", 0.0),
        ("The maximum count is 17. The maximum count is 16.", 1.0),
        ("The maximum count is not 16. A length of 16 is impossible.", 0.0),
        ("The previous upper bound is 17. The previous lower bound is 15.\n\n16", 1.0),
        ("The previous upper bound is 16.\n\n17", 0.0),
    ],
)
def test_opening_claim_does_not_select_a_matching_explanatory_number(answer, expected, tmp_path, monkeypatch):
    repaired = staged_score(with_final_text_grader(retained_task(), BUILT), tmp_path / "grader", monkeypatch)
    assert repaired(answer, {"problem": "Find the maximum count.", "answer": "16"}) == expected
