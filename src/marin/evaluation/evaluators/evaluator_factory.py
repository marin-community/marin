# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.evaluators.alpaca_evaluator import AlpacaEvaluator
from marin.evaluation.evaluators.evaluator import Evaluator
from marin.evaluation.evaluators.helm_evaluator import HELMEvaluator
from marin.evaluation.evaluators.levanter_lm_eval_evaluator import LevanterLmEvalEvaluator
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LMEvaluationHarnessEvaluator
from marin.evaluation.evaluators.simple_evaluator import SimpleEvaluator

# Supported evaluators
NAME_TO_EVALUATOR = {
    "alpaca": AlpacaEvaluator,
    "helm": HELMEvaluator,
    "debug": SimpleEvaluator,
    "lm_evaluation_harness": LMEvaluationHarnessEvaluator,
    "levanter_lm_evaluation_harness": LevanterLmEvalEvaluator,
}


def get_evaluator(config: EvaluationConfig) -> Evaluator:
    """
    Returns the evaluator for the given name.
    """
    if config.evaluator not in NAME_TO_EVALUATOR:
        raise ValueError(f"Unknown evaluator: {config.evaluator}")

    return NAME_TO_EVALUATOR[config.evaluator]()
