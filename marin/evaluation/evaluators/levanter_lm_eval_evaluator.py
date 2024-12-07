import json
import logging
from typing import ClassVar

import fsspec
import jmp
import levanter.eval_harness as eval_harness
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.levanter_tpu_evaluator import LevanterTpuEvaluator
from marin.evaluation.utils import (
    is_remote_path,
)

logger = logging.getLogger(__name__)


class LevanterLmEvalEvaluator(LevanterTpuEvaluator):
    """For `Evaluator`s that runs inference with Levanter's Lm Eval Harness on TPUs."""

    _pip_packages: ClassVar[list[Dependency]] = [
        *LevanterTpuEvaluator.DEFAULT_PIP_PACKAGES,
    ]

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
    ) -> None:
        """
        Runs Levanter's lm-eval harness on the specified model and set of tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
        """
        # Eval Harness code: https://github.com/stanford-crfm/levanter/blob/main/src/levanter/eval_harness.py
        # Run the harness with the model and the specified evals

        from transformers import AutoTokenizer

        try:

            # Download the model from GCS or HuggingFace
            model_name_or_path: str = self.download_model(model)

            logger.info(f"Running eval harness on model: {model_name_or_path}")

            trainer_config = TrainerConfig(
                mp=jmp.get_policy("fp32"),
                per_device_eval_batch_size=32,
                ray=RayConfig(auto_start_cluster=False)
            )

            model_config = LlamaConfig()
\
            # convert to the config that Levanter's eval_harness expects
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
                checkpoint_is_hf=True,
                trainer=trainer_config,
                model=model_config,
            )

            outputs = eval_harness.run_eval_harness_main(eval_config)

            if is_remote_path(output_path):
                try:
                    logger.info("Uploading eval results to GCS...")

                    # write output JSON directly to output_path on GCS
                    fs = fsspec.filesystem("gcs")
                    with fs.open(output_path, "w") as f:
                        json.dump(outputs, f, indent=2)

                    logger.info("Upload completed successfully.")

                except Exception as upload_error:
                    logger.info(f"Failed to upload results to GCS: {upload_error}")

        except Exception as e:

            logger.error(f"Error running eval harness: {e}")
            raise e

        finally:
            # Clean up resources
            self.cleanup(model)

