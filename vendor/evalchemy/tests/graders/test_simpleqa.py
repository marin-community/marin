import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.judged import simpleqa, simpleqa_label


def _simpleqa_spec():
    return EvalSpec(
        task="SimpleQA", source=SourceRef("fixture"), source_id="question-1", ordinal=0, requests=(),
        grader=GraderSpec("simpleqa"), doc={"question": "Which city?"}, target="Paris",
    )


@pytest.mark.parametrize(
    ("response", "metrics"),
    [
        ("A", {"accuracy": 1.0, "incorrect": 0.0, "not_attempted": 0.0, "judge_failed": 0.0}),
        ("b", {"accuracy": 0.0, "incorrect": 1.0, "not_attempted": 0.0, "judge_failed": 0.0}),
        (" C\n", {"accuracy": 0.0, "incorrect": 0.0, "not_attempted": 1.0, "judge_failed": 0.0}),
    ],
)
def test_simpleqa_named_graders_build_a_plan_and_parse_labels(response, metrics):
    plan = simpleqa(_simpleqa_spec(), ["The answer is Paris."])

    assert plan.grader == GraderSpec("simpleqa_label")
    assert plan.requests[0].generation == {"temperature": 0, "max_gen_toks": 2048}
    prompt = plan.requests[0].prompt
    assert "Question: Which city?" in prompt
    assert "Gold target: Paris" in prompt
    assert "Predicted answer: The answer is Paris." in prompt
    assert simpleqa_label(_simpleqa_spec(), [response]).metrics == metrics


@pytest.mark.parametrize("response", [
    "A because it is correct", "CORRECT", "", "D",
    "<think>reasoning</think>\nA", "I compared the answers. </think>\nA",
])
def test_simpleqa_label_rejects_noncanonical_responses(response):
    with pytest.raises(ValueError, match="unrecognized SimpleQA judgment"):
        simpleqa_label(_simpleqa_spec(), [response])
