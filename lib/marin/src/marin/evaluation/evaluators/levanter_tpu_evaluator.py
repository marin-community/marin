# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from fray.v1.cluster import ResourceConfig

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig, launch_evaluate_with_ray


class LevanterTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with Levanter (primarily Lm Eval Harness) on TPUs."""

    @staticmethod
    def model_name_or_path(model: ModelConfig) -> str:
        """Return a reference Levanter can read without staging to local disk."""
        if model.path is None:
            return model.name
        return model.path

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        resource_config: ResourceConfig,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Launches the evaluation run with Fray.
        """
        launch_evaluate_with_ray(
            evaluator=self,
            job_name="levanter-tpu-eval",
            model=model,
            evals=evals,
            output_path=output_path,
            resource_config=resource_config,
            max_eval_instances=max_eval_instances,
            wandb_tags=wandb_tags,
            extras=("eval", "tpu"),
            configure_logging=False,
        )
