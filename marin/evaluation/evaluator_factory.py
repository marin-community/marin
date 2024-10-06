from marin.evaluation.evaluator import Evaluator
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.alpaca_evaluator import AlpacaEvaluator
from marin.evaluation.helm_evaluator import HELMEvaluator
from marin.evaluation.simple_evaluator import SimpleEvaluator
from marin.evaluation.eleuther_evaluator import EleutherEvaluator

# Supported evaluators
NAME_TO_EVALUATOR = {
    "alpaca": AlpacaEvaluator,
    "helm": HELMEvaluator,
    "debug": SimpleEvaluator,
    "eleuther": EleutherEvaluator,
}


def get_evaluator(config: EvaluationConfig) -> Evaluator:
    """
    Returns the evaluator for the given name.
    """
    if config.evaluator not in NAME_TO_EVALUATOR:
        raise ValueError(f"Unknown evaluator: {config.evaluator}")

    return NAME_TO_EVALUATOR[config.evaluator]()
