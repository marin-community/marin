# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import reasoning_gym
from tasktrove_verify.grade import InvalidTask, Status, run
from tasktrove_verify.modes import grade_reasoning_gym
from tasktrove_verify.spec import ReasoningGymSpec

NEEDLE_ENTRY = {
    "question": "Who savors playing the accordion? Reply only with a name.",
    "answer": "Richard",
    "metadata": {"source_dataset": "needle_haystack", "source_index": 0, "num_statements": 45},
}


@pytest.fixture
def workspace(tmp_path):
    directory = tmp_path / "app"
    directory.mkdir()
    return directory


@pytest.fixture
def tests_dir(tmp_path):
    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "entry.json").write_text(json.dumps(NEEDLE_ENTRY))
    return directory


def answer(workspace, text):
    (workspace / "answer.txt").write_text(text)


def grade(tests_dir, workspace, dataset="needle_haystack"):
    return grade_reasoning_gym.grade(ReasoningGymSpec(dataset=dataset), tests_dir, workspace)


def test_correct_answer_scores_one(tests_dir, workspace):
    answer(workspace, "Richard\n")
    reward = grade(tests_dir, workspace)
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)


def test_dataset_scorer_decides_equivalence_rather_than_string_equality(tests_dir, workspace):
    # needle_haystack lowercases before comparing, so a differently cased name is still correct.
    answer(workspace, "richard")
    assert grade(tests_dir, workspace).reward == 1.0


def test_wrong_answer_scores_zero(tests_dir, workspace):
    answer(workspace, "Oluwadamiloju")
    reward = grade(tests_dir, workspace)
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)


def test_prose_around_the_answer_is_not_accepted_by_this_dataset(tests_dir, workspace):
    answer(workspace, "The person who savors playing the accordion is Richard.")
    assert grade(tests_dir, workspace).reward == 0.0


def test_partial_credit_from_the_scorer_is_passed_through(tmp_path, workspace):
    # simple_equations keeps reasoning-gym's default scorer, which awards a fraction when the gold
    # answer is buried in a longer response.
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    entry = {"answer": "42", "metadata": {"source_dataset": "simple_equations"}}
    (tests_dir / "entry.json").write_text(json.dumps(entry))
    answer(workspace, "x = 42")
    reward = grade(tests_dir, workspace, dataset="simple_equations")
    assert 0.0 < reward.reward < 1.0
    answer(workspace, "42")
    assert grade(tests_dir, workspace, dataset="simple_equations").reward == 1.0


@pytest.mark.parametrize(
    "candidate,expected",
    [("ALPHA BETA", 1.0), ("alpha wrong", 0.5), ("alpha beta extra", 1.0), ("wrong wrong", 0.0)],
)
def test_extracted_letter_jumble_preserves_word_partial_credit(candidate, expected):
    entry = {"answer": "alpha beta", "metadata": {"source_dataset": "letter_jumble"}}
    result = grade_reasoning_gym.grade_reasoning_gym_candidate(
        ReasoningGymSpec(dataset="letter_jumble"), entry, candidate
    )
    assert (result.reward, result.status) == (expected, Status.SCORED)


def test_scorer_failure_is_an_unscored_infrastructure_error(tests_dir, workspace):
    # A missing field required by the selected evaluator is a grader failure,
    # not evidence that the candidate is incorrect.
    (tests_dir / "entry.json").write_text(
        json.dumps({"answer": "alpha, beta", "metadata": {"source_dataset": "word_sorting"}})
    )
    spec_path = tests_dir / "verifier.toml"
    spec_path.write_text('mode = "reasoning-gym"\ndataset = "word_sorting"\n')
    answer(workspace, "alpha, beta")
    result = run(spec_path, workspace)
    assert result.status is Status.INFRA_ERROR
    assert "KeyError" in result.detail["error"]


@pytest.mark.parametrize("valid,expected", [(True, 1.0), (False, 0.01)])
def test_generated_graph_coloring_with_null_gold_grades_candidate_and_file(tests_dir, workspace, valid, expected):
    entry = reasoning_gym.create_dataset(
        "graph_color", seed=231, size=1, min_num_vertices=4, max_num_vertices=4, num_colors=4, edge_probability=0.8
    )[0]
    assert entry["answer"] is None
    coloring = (
        entry["metadata"]["possible_answer"] if valid else dict.fromkeys(entry["metadata"]["puzzle"]["vertices"], 1)
    )
    candidate = json.dumps(coloring)
    spec = ReasoningGymSpec(dataset="graph_color")
    result = grade_reasoning_gym.grade_reasoning_gym_candidate(spec, entry, candidate)
    assert (result.reward, result.status) == (expected, Status.SCORED)
    (tests_dir / "entry.json").write_text(json.dumps(entry))
    answer(workspace, candidate)
    result = grade_reasoning_gym.grade(spec, tests_dir, workspace)
    assert (result.reward, result.status) == (expected, Status.SCORED)
    assert reasoning_gym.get_score_answer_fn("graph_color")(candidate, entry) == expected


@pytest.mark.parametrize("answer_field", [{}, {"answer": [1, 2]}])
def test_null_gold_support_preserves_malformed_entry_errors(answer_field):
    entry = {**answer_field, "metadata": {"source_dataset": "graph_color"}}
    with pytest.raises(InvalidTask, match="string or null answer field"):
        grade_reasoning_gym.grade_reasoning_gym_candidate(ReasoningGymSpec(dataset="graph_color"), entry, "{}")


@pytest.mark.parametrize("text", [None, "", "   \n"])
def test_absent_or_blank_output_scores_zero_with_no_output(tests_dir, workspace, text):
    if text is not None:
        answer(workspace, text)
    reward = grade(tests_dir, workspace)
    assert reward.reward == 0.0
    assert reward.detail == {"reason": "no_output"}


def test_unknown_dataset_is_an_invalid_task(tests_dir, workspace):
    answer(workspace, "Richard")
    with pytest.raises(InvalidTask, match="unknown reasoning-gym dataset"):
        grade(tests_dir, workspace, dataset="not_a_reasoning_gym_dataset")


def test_missing_entry_file_is_an_invalid_task(tmp_path, workspace):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    answer(workspace, "Richard")
    with pytest.raises(InvalidTask, match="entry not found"):
        grade(tests_dir, workspace)


@pytest.mark.parametrize(
    "entry_text, message",
    [
        ("{not json", "is not JSON"),
        ('{"answer": "Richard"}', "metadata field"),
        ("[1, 2, 3]", "metadata field"),
    ],
)
def test_unusable_entry_file_is_an_invalid_task(tmp_path, workspace, entry_text, message):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "entry.json").write_text(entry_text)
    answer(workspace, "Richard")
    with pytest.raises(InvalidTask, match=message):
        grade(tests_dir, workspace)
