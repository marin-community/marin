import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.math_specs import boxed_math


@pytest.mark.parametrize(("candidate", "reference"), [("0.09", r"\frac{9}{100}"), ("58500", "58,500"), (r"5r^{5}", r"5r^5")])
def test_math_grader_preserves_symbolic_equivalence(candidate, reference):
    spec = EvalSpec(task="MATH500", source=SourceRef("local"), source_id="1", ordinal=0, requests=(), grader=GraderSpec("boxed_math"), target=reference)

    assert boxed_math(spec, [rf"\boxed{{{candidate}}}"]).metrics["accuracy"] == 1.0
