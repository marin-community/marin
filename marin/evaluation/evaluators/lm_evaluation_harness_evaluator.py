import logging
import os
import shutil
import traceback
from typing import ClassVar

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import is_remote_path, run_bash_command, upload_to_gcs

logger = logging.getLogger(__name__)


# TODO: Multiple choice tasks currently don't work on TPUs: https://github.com/vllm-project/vllm/issues/8499
class LMEvaluationHarnessEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """

    CACHE_PATH: str = "/tmp/lm-eval"
    RESULTS_PATH: str = os.path.join(CACHE_PATH, "eleuther_results")

    _pip_packages: ClassVar[list[Dependency]] = [
        *VllmTpuEvaluator.DEFAULT_PIP_PACKAGES,
        # NOTE(chris): We put lm-eval[ifeval] in the Dockerfile.vllm, so this dependency is not needed
    ]
    _env_vars: ClassVar[dict[str, str]] = {
        # Human eval tests code from the model which requires permission to run
        "HF_ALLOW_CODE_EVAL": "1",
    }

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
            # NOTE(chris): This is not supported on TPUs
            # set_cuda_visible_devices()
            # Download the model from GCS or HuggingFace
            model_name_or_path: str = self.download_model(model)

            pretrained_args: str = f"pretrained={model_name_or_path}"
            if model.engine_kwargs:
                for key, value in model.engine_kwargs.items():
                    pretrained_args += f",{key}={value}"

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
                    pretrained_args,
                    "--batch_size",
                    "auto",
                    "--trust_remote_code",
                    "--output_path",
                    result_filepath,
                ]

                # Human eval tests code from the model which requires permission to run
                if eval_task.name == "humaneval":
                    command.append("--confirm_run_unsafe_code")

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
