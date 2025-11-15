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

import logging
import os
import shutil
import traceback

from marin.evaluation.evaluation_config import EvalTaskConfig, ModelConfig
from marin.evaluation.evaluators.evaluator import Evaluator
from marin.evaluation.utils import is_remote_path, upload_to_gcs

logger = logging.getLogger(__name__)


class LMEvaluationHarnessEvaluator(Evaluator):
    """
    Evaluator that runs lm-eval (https://github.com/EleutherAI/lm-evaluation-harness)
    using the inference pool via OpenAI-compatible API.
    """

    CACHE_PATH: str = "/tmp/lm-eval"
    RESULTS_PATH: str = os.path.join(CACHE_PATH, "eleuther_results")

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        openai_base_url: str,
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Runs EleutherAI's lm-eval harness on the specified model and set of tasks.

        Uses the inference pool via the OpenAI-compatible API.

        Args:
            model: Model configuration
            evals: List of evaluations to run
            openai_base_url: Base URL for the OpenAI-compatible inference pool API
            output_path: Path to save evaluation results
            max_eval_instances: Maximum number of evaluation instances to run
            wandb_tags: Tags to add to wandb run
        """
        try:
            from lm_eval.evaluator import simple_evaluate
            from lm_eval.loggers import EvaluationTracker, WandbLogger

            # Configure lm-eval to use OpenAI-compatible API
            # See: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md#openai-completions
            model_args = {
                "base_url": openai_base_url,
                "model": model.name,
            }

            for eval_task in evals:
                result_filepath = os.path.join(self.RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")

                output_dir = os.path.dirname(result_filepath)
                os.makedirs(output_dir, exist_ok=True)

                evaluation_tracker = EvaluationTracker(output_path=result_filepath)

                wandb_args_dict = {
                    "project": "marin",
                    "job_type": "eval",
                    "name": model.name,
                    "tags": wandb_tags,
                }
                wandb_logger = WandbLogger(init_args=wandb_args_dict)

                results = simple_evaluate(
                    model="openai-chat-completions",
                    tasks=[eval_task.name],
                    num_fewshot=eval_task.num_fewshot,
                    model_args=model_args,
                    apply_chat_template=model.apply_chat_template,
                    batch_size="auto",
                    confirm_run_unsafe_code=True,
                    limit=max_eval_instances if max_eval_instances is not None else None,
                    evaluation_tracker=evaluation_tracker,
                    log_samples=True,
                )
                if results is not None:
                    samples = results.pop("samples")
                    evaluation_tracker.save_results_aggregated(results=results, samples=samples)

                    try:
                        wandb_logger.post_init(results)
                        wandb_logger.log_eval_result()
                        wandb_logger.log_eval_samples(samples)
                        wandb_logger.run.finish()
                    except Exception as e:
                        print(f"Logging to Weights and Biases failed due to {e}")

                    for task_name in results["configs"].keys():
                        evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

                assert os.path.exists(result_filepath), f"Results file {result_filepath} does not exist."

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("lm-eval failed. Please check the logs for more information.") from e

        finally:
            # Upload results to GCS if needed
            if is_remote_path(output_path):
                try:
                    logger.info("Uploading eval results to GCS...")
                    upload_to_gcs(self.RESULTS_PATH, output_path)
                    logger.info("Upload completed successfully.")
                except Exception as upload_error:
                    logger.info(f"Failed to upload results to GCS: {upload_error}")

            # Clean up local results
            if os.path.exists(self.RESULTS_PATH):
                shutil.rmtree(self.RESULTS_PATH)
