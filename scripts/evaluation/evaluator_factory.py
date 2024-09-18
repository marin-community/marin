from scripts.evaluation.evaluator import Evaluator, EvaluatorConfig
from scripts.evaluation.helm_evaluator import HELMEvaluator
from scripts.evaluation.simple_evaluator import SimpleEvaluator
from scripts.evaluation.eleuther_evaluator import EleutherEvaluator

# Supported evaluators
NAME_TO_EVALUATOR = {
    "helm": HELMEvaluator,
    "debug": SimpleEvaluator,
    "eleuther": EleutherEvaluator,
}


def get_evaluator(config: EvaluatorConfig) -> Evaluator:
    """
    Returns the evaluator for the given name.
    """
    if config.name not in NAME_TO_EVALUATOR:
        raise ValueError(f"Unknown evaluator: {config.name}")

    return NAME_TO_EVALUATOR[config.name](config)
