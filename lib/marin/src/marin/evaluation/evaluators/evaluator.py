# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.inference.config import InferenceModelConfig


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model: InferenceModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """What to run to evaluate."""
        pass
