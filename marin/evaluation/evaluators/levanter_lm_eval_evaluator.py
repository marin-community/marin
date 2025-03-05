import json
import logging
import os
import shutil
from typing import ClassVar

import fsspec
import jmp
import levanter.eval_harness as eval_harness
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig

from experiments.evals.task_configs import convert_to_levanter_task_config
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.levanter_tpu_evaluator import LevanterTpuEvaluator

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

        try:

            # Download the model from GCS or HuggingFace
            model_name_or_path: str = self.download_model(model)

            logger.info(f"Running eval harness on model: {model_name_or_path}")

            # NOTE(chris): Before, the batch size was 16, but this is too large for the 8B model.
            # In the future, we should make this user-configurable.
            trainer_config = TrainerConfig(
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                per_device_eval_parallelism=8,
                ray=RayConfig(auto_start_cluster=False),
            )

            model_config = LlamaConfig()

            # convert to the config that Levanter's eval_harness expects
            tasks = convert_to_levanter_task_config(evals)

            model_path = os.path.join(LevanterTpuEvaluator.CACHE_PATH, model.path)

            logger.info(f"Model path: {model_path}")
            logger.info(f"Levanter Cache Path: {LevanterTpuEvaluator.CACHE_PATH}")
            logger.info(f"Model name: {model.name}")
            logger.info(f"model_name_or_path: {model_name_or_path}")

            eval_config = eval_harness.EvalHarnessMainConfig(
                eval_harness=eval_harness.LmEvalHarnessConfig(
                    task_spec=tasks,
                    max_examples=max_eval_instances,
                    log_samples=False,
                    max_eval_length=4096,
                ),
                tokenizer=model_path,  # levanter picks up the tokenizer from the model path
                checkpoint_path=model_path,
                checkpoint_is_hf=True,
                trainer=trainer_config,
                model=model_config,
            )

            results = eval_harness.run_eval_harness_main(eval_config)

            try:
                # add a results.json to output path
                output_path = os.path.join(output_path, "results.json")

                logger.info(f"Uploading results to GCS: {output_path}")

                # write output JSON directly to output_path on GCS
                fs = fsspec.filesystem("gcs")
                with fs.open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

                logger.info("Upload completed successfully.")

            except Exception as upload_error:
                logger.info(f"Failed to upload results to GCS: {upload_error}")

        except Exception as e:

            logger.error(f"Error running eval harness: {e}")
            raise e

        finally:
            # Clean up resources
            self.cleanup(model)

            if os.path.exists(LevanterTpuEvaluator.CACHE_PATH):
                shutil.rmtree(LevanterTpuEvaluator.CACHE_PATH)
