# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from marin.evaluation.evaluation_config import EvalTaskConfig


@dataclass
class ModelConfig:
    """Model configuration for the legacy checkpoint-native evaluators."""

    name: str
    path: str | None
    engine_kwargs: dict[str, Any]
    generation_params: dict | None = None
    apply_chat_template: bool = False
    base_eval_run_name: str | None = None


class Evaluator(ABC):
    """Interface retained for checkpoint-native data-mixing evaluations."""

    @abstractmethod
    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """Evaluate a checkpoint and persist results."""
