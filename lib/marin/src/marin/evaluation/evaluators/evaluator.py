# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from marin.evaluation.evaluation_config import EvalTaskConfig


@dataclass
class ModelConfig:
    name: str
    """The name of the model e.g., allenai/olmo-7b"""

    path: str | None
    """
    The path to the model checkpoint. Can be a local path or a path on GCS.
    """

    engine_kwargs: dict[str, Any]
    """
    Additional keyword arguments to pass to the vLLM engine.
    """

    generation_params: dict | None = None
    """
    Additional keyword arguments passed to the SamplingParams for the vLLM engine
    """

    apply_chat_template: bool = False
    """
    Whether or not this model was trained with a Chat Template in the tokenizer
    """

    base_eval_run_name: str | None = None
    """Custom base name for wandb runs."""


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
