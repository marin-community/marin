from scripts.evaluation.evaluator import Evaluator
from scripts.evaluation.noop_evaluator import NoopEvaluator

# Supported evaluators
NAME_TO_EVALUATOR = {
    "noop": NoopEvaluator,
}


def get_evaluator(evaluator_name: str) -> Evaluator:
    """
    Returns the evaluator for the given name.
    """
    assert evaluator_name in NAME_TO_EVALUATOR, f"Unknown evaluation harness: {evaluator_name}"
    return NAME_TO_EVALUATOR[evaluator_name]()
