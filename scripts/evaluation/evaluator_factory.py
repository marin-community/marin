from scripts.evaluation.evaluator import Evaluator
from scripts.evaluation.simple_evaluator import SimpleEvaluator

# Supported evaluators
NAME_TO_EVALUATOR = {
    "simple": SimpleEvaluator,
}


def get_evaluator(evaluator_name: str) -> Evaluator:
    """
    Returns the evaluator for the given name.
    """
    if evaluator_name not in NAME_TO_EVALUATOR:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")

    return NAME_TO_EVALUATOR[evaluator_name]()
