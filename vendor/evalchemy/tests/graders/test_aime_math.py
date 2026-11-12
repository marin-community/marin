import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.math_specs import answer_number, boxed_math, jee
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed


@pytest.mark.parametrize("answer", ["25", r"\frac{50}{2}"])
def test_boxed_math_accepts_equivalent_final_answers(answer):
    spec = EvalSpec(task="AIME24", source=SourceRef("local"), source_id="1", ordinal=0, requests=(), grader=GraderSpec("boxed_math"), target="25")

    assert boxed_math(spec, [rf"Work. \boxed{{{answer}}}"]).metrics["accuracy"] == 1.0


def test_boxed_math_rejects_an_unboxed_answer():
    spec = EvalSpec(task="AIME24", source=SourceRef("local"), source_id="1", ordinal=0, requests=(), grader=GraderSpec("boxed_math"), target="25")

    assert boxed_math(spec, ["The answer is 25"]).metrics["accuracy"] == 0.0


@pytest.mark.parametrize("answer", ["25", r"\frac{50}{2}"])
def test_amc_boxed_math_uses_legacy_hendrycks_extraction_and_comparator(answer):
    response = rf"Work. \boxed{{{answer}}}"
    spec = EvalSpec(
        task="AMC23", source=SourceRef("local"), source_id="1", ordinal=0, requests=(),
        grader=GraderSpec("boxed_math", {"extractor": "hendrycks_math", "comparator": "hendrycks_math"}), target="25",
    )

    assert boxed_math(spec, [response]).metrics["accuracy"] == float(is_equiv("25", remove_boxed(last_boxed_only_string(response))))


@pytest.mark.parametrize("answer", ["25", "025"])
def test_aiw_number_grader_uses_legacy_hendrycks_comparator(answer):
    response = f"Answer: {answer}"
    spec = EvalSpec(
        task="AIW", source=SourceRef("local"), source_id="1", ordinal=0, requests=(),
        grader=GraderSpec("answer_number", {"comparator": "hendrycks_math"}), target="25",
    )

    assert answer_number(spec, [response]).metrics["accuracy"] == float(is_equiv("25", answer))


@pytest.mark.parametrize("response", [r"\boxed{1}<|im_end|>\boxed{2}", r"\boxed{2}</think>Done."])
def test_jee_retains_last_box_from_complete_response(response):
    spec = EvalSpec(
        task="JEEBench", source=SourceRef("fixture"), source_id="1", ordinal=0, requests=(),
        grader=GraderSpec("jee", {"type": "Integer"}), target="2",
    )
    assert jee(spec, [response]).metrics["accuracy"] == 1.0
