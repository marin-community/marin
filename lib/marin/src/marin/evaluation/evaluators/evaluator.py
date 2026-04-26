# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.inference.model_config import ModelConfig


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """What to run to evaluate."""
        pass
