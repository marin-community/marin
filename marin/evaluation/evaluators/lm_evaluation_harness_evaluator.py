import logging
import os
import shutil
import traceback
from typing import ClassVar

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import is_remote_path, run_bash_command, set_cuda_visible_devices, upload_to_gcs

logger = logging.getLogger(__name__)


# TODO: this currently doesn't work on TPUs: https://github.com/vllm-project/vllm/issues/8499
class LMEvaluationHarnessEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """

    RESULTS_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "eleuther_results")

    _pip_packages: ClassVar[list[Dependency]] = [
        *VllmTpuEvaluator.DEFAULT_PIP_PACKAGES,
        Dependency(name="lm_eval"),
        Dependency(name="lm-eval[api]"),
    ]

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
    ) -> None:
        """
        Runs EleutherAI's lm-eval harness on the specified model and set of  tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
        """
        # From https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#model-apis-and-inference-servers
        # Run lm_eval with the model and the specified evals
        try:
            set_cuda_visible_devices()

            # Download the model from GCS or HuggingFace
            model_name_or_path: str = self.download_model(model)

            for eval_task in evals:

                result_filepath = os.path.join(self.RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")

                # Create the output directory
                output_dir = os.path.dirname(result_filepath)
                os.makedirs(output_dir, exist_ok=True)

                command = [
                    "lm_eval",
                    "--model",
                    "vllm",
                    "--tasks",
                    eval_task.name,
                    "--num_fewshot",
                    str(eval_task.num_fewshot),
                    "--model_args",
                    f"pretrained={model_name_or_path}",
                    "--batch_size",
                    "auto",
                    "--trust_remote_code",
                    "--output_path",
                    result_filepath,
                ]

                if max_eval_instances is not None:
                    # According lm-eval-harness, --limit should only be used for testing purposes
                    command.extend(["--limit", str(max_eval_instances)])

                # run the command and check if the results file exists
                run_bash_command(command, check=True)
                assert os.path.exists(result_filepath), f"Results file {result_filepath} does not exist."

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("lm-eval failed. Please check the logs for more information.") from e

        finally:

            # this is in the finally block so even in the case of exceptions we will
            # write what has been saved
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
