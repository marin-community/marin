# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from verifyit.grade import InvalidTask, Status, main
from verifyit.grade import grade as dispatch
from verifyit.modes import grade_mcq
from verifyit.spec import McqSpec


def _answer(workspace: Path, text: str) -> None:
    (workspace / "answer.txt").write_text(text)


@pytest.mark.parametrize(
    "candidate, expected_reward, detail",
    [
        (" c ", 1.0, {"extracted": "C", "expected": "C"}),
        ("B", 0.0, {"extracted": "B", "expected": "C"}),
        ("E", 0.0, {"reason": "out_of_range", "extracted": "E", "expected": "C"}),
        ("", 0.0, {"reason": "no_answer_line", "expected": "C"}),
    ],
)
def test_mcq_candidate_scores_extracted_option(candidate, expected_reward, detail):
    result = grade_mcq.grade_mcq_candidate(McqSpec(expected=" C ", options=4), candidate)
    assert (result.status, result.reward, result.detail) == (Status.SCORED, expected_reward, detail)


@pytest.mark.parametrize(
    "text, reward",
    [
        ("C", 1.0),
        (" \n c\t", 1.0),
        ("B", 0.0),
        ("E", 0.0),
        ("CC", 0.0),
        ("The third option fits.\nAnswer: C\n", 1.0),
        ("Answer: c", 1.0),
        ("Answer:C", 1.0),
        ("Answer : C", 1.0),
        ("Answer: B\n", 0.0),
        # Half of the Nemotron prompts ask for a boxed letter; models also decorate the line.
        ("Answer: \\boxed{C}\n", 1.0),
        ("**Answer:** C\n", 1.0),
        ("Answer: (C)\n", 1.0),
        ("Answer: `C`\n", 1.0),
        ("Answer: \\boxed{B}\n", 0.0),
        # The last stated answer wins: a model may revise itself.
        ("Answer: A\nOn reflection that is wrong.\nAnswer: C\n", 1.0),
        # Letters outside A..D cannot be the answer to a four-option question.
        ("Answer: E\n", 0.0),
        ("Answer: 3\n", 0.0),
        # Prose alone is not an answer.
        ("The answer is obviously C\n", 0.0),
        ("I worked through every option and settled on the third one.\nC\n", 0.0),
        ("", 0.0),
        ("   \n\n", 0.0),
    ],
)
def test_mcq_scores_a_standalone_letter_or_stated_answer(tmp_path, text, reward):
    _answer(tmp_path, text)
    assert grade_mcq.grade(McqSpec(expected="C"), tmp_path, tmp_path).reward == reward


def test_mcq_missing_answer_file_scores_zero(tmp_path):
    result = grade_mcq.grade(McqSpec(expected="C"), tmp_path, tmp_path)
    assert result.reward == 0.0
    assert result.detail["reason"] == "no_output"


def test_mcq_reward_detail_carries_the_extracted_letter(tmp_path):
    _answer(tmp_path, "Answer: b\n")
    assert grade_mcq.grade(McqSpec(expected="C"), tmp_path, tmp_path).detail["extracted"] == "B"


def test_mcq_letter_beyond_the_option_count_is_wrong_not_a_task_defect(tmp_path):
    _answer(tmp_path, "Answer: E\n")
    result = grade_mcq.grade(McqSpec(expected="C", options=4), tmp_path, tmp_path)
    assert (result.status, result.reward, result.detail["extracted"]) == (Status.SCORED, 0.0, "E")


@pytest.mark.parametrize("spec", [McqSpec(expected="E", options=4), McqSpec(expected="A", options=0)])
def test_mcq_expected_outside_the_options_is_an_invalid_task(tmp_path, spec):
    _answer(tmp_path, "Answer: A\n")
    assert dispatch(spec, tmp_path, tmp_path).status == Status.INVALID_TASK


def test_mcq_fifth_option_is_gradable_when_declared(tmp_path):
    _answer(tmp_path, "Answer: E\n")
    assert grade_mcq.grade(McqSpec(expected="E", options=5), tmp_path, tmp_path).reward == 1.0


def test_cli_grades_an_mcq_task_and_writes_the_verdict(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "verifier.toml").write_text('mode = "mcq"\nexpected = "C"\noptions = 4\n')
    _answer(tmp_path, "Working through the options...\nAnswer: C\n")
    logs = tmp_path / "logs"

    assert main([str(tests_dir / "verifier.toml"), "--logs-dir", str(logs), "--workspace", str(tmp_path)]) == 0
    assert json.loads((logs / "verdict.json").read_text()) == {
        "reward": 1.0,
        "status": "scored",
        "detail": {"extracted": "C", "expected": "C"},
    }
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert (logs / "reward.txt").read_text() == "1.0\n"


@pytest.mark.parametrize(
    ("response", "expected_reward"),
    [
        ("Answer: Banana", 0.0),
        ("Answer: BC", 0.0),
        ("Answer: B2", 0.0),
        ("Answer: B.", 1.0),
        ("Answer: (B)", 1.0),
        (r"Answer: \boxed{B}", 1.0),
        ("Answer: B because the evidence supports it", 1.0),
    ],
)
def test_mcq_option_must_be_a_complete_alphanumeric_token(tmp_path, response, expected_reward):
    _answer(tmp_path, response)
    verdict = dispatch(McqSpec(expected="B"), tmp_path, tmp_path)
    assert (verdict.reward, verdict.status) == (expected_reward, Status.SCORED)


def test_probability_mass_sums_correct_alternatives_without_duplicate_credit():
    scores = [math.log(value) for value in (0.2, 0.5, 0.3)]
    verdict = grade_mcq.grade_mcq_likelihoods(
        scores, [0, 2, 0], normalization_lengths=[1, 1, 1], policy=grade_mcq.LikelihoodScoring.PROBABILITY_MASS
    )
    assert verdict.reward == pytest.approx(0.5)
    assert verdict.detail["selected_index"] == 1
    assert verdict.detail["probabilities"] == pytest.approx([0.2, 0.5, 0.3])


def test_likelihood_choices_are_not_limited_to_alphabet_and_ties_choose_first():
    for policy, expected in [
        (grade_mcq.LikelihoodScoring.MOST_LIKELY, 1),
        (grade_mcq.LikelihoodScoring.PROBABILITY_MASS, 1 / 30),
    ]:
        result = grade_mcq.grade_mcq_likelihoods([-1] * 30, [0], normalization_lengths=[1] * 30, policy=policy)
        assert result.reward == pytest.approx(expected)
        assert result.detail["selected_index"] == 0


def test_all_correct_mass_is_exactly_one_despite_probability_rounding():
    likelihoods = [-4.6480028589515, -3.7412643461618327, -1.128800860050413, -3.7261393895864736]
    result = grade_mcq.grade_mcq_likelihoods(
        likelihoods, [0, 1, 2, 3], normalization_lengths=[1] * 4, policy=grade_mcq.LikelihoodScoring.PROBABILITY_MASS
    )
    assert result.reward == 1


def test_underflow_mass_matches_high_precision_reference():
    with localcontext() as context:
        context.prec = 80
        weights = [Decimal(-1000).exp(), Decimal(-1001).exp()]
        expected = float(weights[0] / sum(weights))
    result = grade_mcq.grade_mcq_likelihoods(
        [-1000, -1001], [0], normalization_lengths=[1, 1], policy=grade_mcq.LikelihoodScoring.PROBABILITY_MASS
    )
    assert result.reward == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    "likelihoods", [[0], None, [0, None], [0, True], [0, float("nan")], [0, float("-inf")], [0, 10**1000]]
)
def test_missing_or_malformed_likelihood_cannot_award_correct_first_choice(likelihoods):
    with pytest.raises(InvalidTask):
        grade_mcq.grade_mcq_likelihoods(
            likelihoods, [0], normalization_lengths=[1, 1], policy=grade_mcq.LikelihoodScoring.MOST_LIKELY
        )


@pytest.mark.parametrize("expected", ["AB", "", None, 7])
def test_malformed_reference_cannot_award_self_match(expected):
    with pytest.raises(InvalidTask):
        grade_mcq.grade_mcq_candidate(McqSpec(expected=expected, options=4), expected)


@pytest.mark.parametrize("expected", ["AB", ""])
def test_invalid_mcq_reference_removes_previous_cli_credit_before_reading_candidate(tmp_path, expected):
    spec = tmp_path / "verifier.toml"
    logs = tmp_path / "logs"
    spec.write_text('mode = "mcq"\nexpected = "A"\noptions = 4\n')
    _answer(tmp_path, "Answer: A")
    args = [str(spec), "--logs-dir", str(logs), "--workspace", str(tmp_path)]
    assert main(args) == 0
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    spec.write_text(f'mode = "mcq"\nexpected = "{expected}"\noptions = 4\n')
    (tmp_path / "answer.txt").unlink()
    assert main(args) == 0
    verdict = json.loads((logs / "verdict.json").read_text())
    assert (verdict["status"], verdict["reward"]) == ("invalid_task", 0)
    assert not (logs / "reward.json").exists()
    assert not (logs / "reward.txt").exists()


@pytest.mark.parametrize("options", [True, 4.0])
def test_noninteger_option_count_cannot_award_a_correct_answer(options):
    with pytest.raises(InvalidTask):
        grade_mcq.grade_mcq_candidate(McqSpec(expected="A", options=options), "A")


@pytest.mark.parametrize(
    "likelihoods,lengths,perplexity,bits",
    [
        ([-math.log(4), -math.log(8)], [1, 4], 2.0, 1.0),
        ([-2.0, -8.0], [2, 3], math.exp(2), 2 / math.log(2)),
        ([0.0], [3], 1.0, 0.0),
    ],
)
def test_corpus_diagnostics_weight_total_log_probability_by_total_units(likelihoods, lengths, perplexity, bits):
    result = grade_mcq.summarize_log_likelihoods(likelihoods, normalization_lengths=lengths)
    assert result.perplexity == pytest.approx(perplexity, rel=1e-15)
    assert result.bits_per_unit == pytest.approx(bits, rel=1e-15)
    assert result.mean_log_likelihood == pytest.approx(-math.log(perplexity), rel=1e-15)


@pytest.mark.parametrize(
    "likelihoods", [None, [None], [True], [float("nan")], [float("-inf")], [1.0], [-1000.0], [-1e308, -1e308]]
)
def test_invalid_or_overflowing_corpus_observations_cannot_publish_favorable_statistics(likelihoods):
    lengths = [1] * len(likelihoods) if isinstance(likelihoods, list) else [1]
    with pytest.raises(InvalidTask):
        grade_mcq.summarize_log_likelihoods(likelihoods, normalization_lengths=lengths)


def test_invalid_corpus_weights_are_validated_before_missing_observations():
    with pytest.raises(InvalidTask, match="normalization"):
        grade_mcq.summarize_log_likelihoods(None, normalization_lengths=[0])
