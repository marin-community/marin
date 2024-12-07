import json
import logging
import os
import shutil
from typing import ClassVar

import levanter.eval_harness as eval_harness

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.levanter_tpu_evaluator import LevanterTpuEvaluator
from marin.evaluation.utils import (
    is_remote_path,
    upload_to_gcs,
)
from marin.execution.executor import ExecutorStep

logger = logging.getLogger(__name__)


class LevanterLmEvalEvaluator(LevanterTpuEvaluator):
    """For `Evaluator`s that runs inference with Levanter's Lm Eval Harness on TPUs."""

    _pip_packages: ClassVar[list[Dependency]] = [
        *LevanterTpuEvaluator.DEFAULT_PIP_PACKAGES,
    ]

    RESULTS_PATH: str = os.path.join(LevanterTpuEvaluator.CACHE_PATH, "levanter_lm_eval_harness_results.json")

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        step: ExecutorStep | None = None,
    ) -> None:
        """
        Runs Levanter's lm-eval harness on the specified model and set of tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            step (ExecutorStep | None): The step to evaluate. Used to get the config for the model and the trainer.
        """
        # Eval Harness code: https://github.com/stanford-crfm/levanter/blob/main/src/levanter/eval_harness.py
        # Run the harness with the model and the specified evals

        from transformers import AutoTokenizer

        try:

            # Download the model from GCS or HuggingFace
            model_name_or_path: str = self.download_model(model)

            logger.info(f"Running eval harness on model: {model_name_or_path}")

            # convert to the config that Levanter's eval_harness expects
            if step and step.config:
                trainer_config = step.config.trainer
                model_config = step.config.model

            tasks = []
            for eval_task_config in evals:
                task = eval_harness.TaskConfig(
                    task=eval_task_config.name,
                    num_fewshot=eval_task_config.num_fewshot,
                    task_alias=eval_task_config.task_alias,
                )
                tasks.append(task)

            eval_config = eval_harness.EvalHarnessMainConfig(
                eval_harness=eval_harness.EvalHarnessConfig(
                    task_spec=tasks,
                    max_examples=max_eval_instances,
                    log_samples=True,
                ),
                tokenizer=AutoTokenizer.from_pretrained(model.name, trust_remote_code=True),
                checkpoint_path=model.path,
                trainer=trainer_config,
                model=model_config,
            )

            outputs = eval_harness.run_eval_harness_main(eval_config)

            with open(self.RESULTS_PATH, "w") as f:
                json.dump(outputs, f, indent=2)

        except Exception as e:

            logger.error(f"Error running eval harness: {e}")
            raise e

        finally:

            if is_remote_path(output_path):
                try:
                    logger.info("Uploading eval results to GCS...")
                    upload_to_gcs(self.RESULTS_PATH, output_path)
                    logger.info("Upload completed successfully.")
                except Exception as upload_error:
                    logger.info(f"Failed to upload results to GCS: {upload_error}")

            self.cleanup(model)

            if os.path.exists(self.RESULTS_PATH):
                shutil.rmtree(self.RESULTS_PATH)
