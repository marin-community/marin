from scripts.evaluation.evaluation_harness import EvaluationHarness
from scripts.evaluation.noop_evaluation_harness import NoopEvaluationHarness

# Supported evaluation harnesses
NAME_TO_EVALUATION_HARNESS = {
    "noop": NoopEvaluationHarness,
}


def get_evaluation_harness(evaluation_harness_name: str) -> EvaluationHarness:
    """
    Returns the evaluation harness for the given name.
    """
    assert (
        evaluation_harness_name in NAME_TO_EVALUATION_HARNESS
    ), f"Unknown evaluation harness: {evaluation_harness_name}"
    return NAME_TO_EVALUATION_HARNESS[evaluation_harness_name]()
