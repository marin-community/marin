# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig


class LevanterTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with Levanter (primarily Lm Eval Harness) on TPUs."""

    @staticmethod
    def model_name_or_path(model: ModelConfig) -> str:
        """Return a reference Levanter can read without staging to local disk."""
        if model.path is None:
            return model.name
        return model.path
