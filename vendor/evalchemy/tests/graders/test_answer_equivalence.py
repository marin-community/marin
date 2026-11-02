import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders import answer_equivalence
from eval.graders.judged import equivalence_judge, equivalence_label


def test_math_answers_equivalent_accepts_symbolically_equal_latex():
    assert answer_equivalence.math_answers_equivalent(r"\frac{1}{2}", ["0.5"])


def test_math_answers_equivalent_rejects_vacuous_symbol_free_equation_pairs():
    assert not answer_equivalence.math_answers_equivalent(r"$I_{i}(0) = I_{e}(0)$", [r"$I_{e}(0)=\gamma I_{i}(0)$"])
    assert not answer_equivalence.math_answers_equivalent("$1 = 2$", ["$3 = 4$"])


def test_math_answers_equivalent_keeps_nonvacuous_symbol_free_and_symbolic_matches():
    assert answer_equivalence.math_answers_equivalent("42", ["42"])
    assert answer_equivalence.math_answers_equivalent("x = 1", ["x=1"])


def test_judge_config_resolves_normalized_environment_without_exposing_key(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "secret-key")

    config = answer_equivalence.JudgeConfig.resolve(judge_model="auto")

    assert config.model == "judge-model"
    assert config.base_url == "https://judge.example/v1"
    assert config.api_key == "secret-key"
    assert "secret-key" not in repr(config)


def test_judge_config_does_not_fall_back_to_candidate_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "candidate-key")
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="JUDGE_API_KEY"):
        answer_equivalence.JudgeConfig.resolve()


@pytest.mark.parametrize(
    "base_url",
    [
        "judge.example/v1",
        "https://secret@judge.example/v1",
        "https://judge.example/v1?token=secret",
        "https://judge.example/v1#secret",
    ],
)
def test_judge_config_rejects_unsafe_endpoint_urls(base_url):
    with pytest.raises(ValueError, match="JUDGE_BASE_URL"):
        answer_equivalence.JudgeConfig("judge-model", base_url, "judge-key")


def _equivalence_spec():
    return EvalSpec(
        task="FinanceBench", source=SourceRef("fixture"), source_id="question-1", ordinal=0, requests=(),
        grader=GraderSpec("equivalence_judge", {"question": "How many centimeters are in one meter?", "answers": ["100 cm"], "answer": "100 cm"}),
    )


def test_equivalence_judge_builds_the_named_judge_plan():
    plan = equivalence_judge(_equivalence_spec(), ["1 m"])

    assert plan.grader == GraderSpec("equivalence_label", {"answer": "1 m"})
    assert plan.requests[0].generation == {"temperature": 0, "max_gen_toks": 2048}
    assert "How many centimeters are in one meter?" in plan.requests[0].prompt
    assert '["100 cm"]' in plan.requests[0].prompt
    assert "Candidate answer:\n1 m" in plan.requests[0].prompt


@pytest.mark.parametrize(
    ("response", "metrics"),
    [
        ("correct", {"accuracy": 1.0, "incorrect": 0.0, "not_attempted": 0.0, "judge_failed": 0.0}),
        ("incorrect", {"accuracy": 0.0, "incorrect": 1.0, "not_attempted": 0.0, "judge_failed": 0.0}),
    ],
)
def test_equivalence_label_maps_judge_responses_to_grades(response, metrics):
    grade = equivalence_label(_equivalence_spec(), [response])

    assert grade.metrics == metrics
    assert grade.audit["judge_label"] == response
