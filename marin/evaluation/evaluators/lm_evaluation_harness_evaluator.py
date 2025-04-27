import logging
import os
import shutil
import traceback
from typing import ClassVar

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import is_remote_path, upload_to_gcs

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

            from lm_eval.evaluator import simple_evaluate
            from lm_eval.loggers import EvaluationTracker, WandbLogger
            from lm_eval.utils import simple_parse_args_string

            for eval_task in evals:

                result_filepath = os.path.join(self.RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")

                # Create the output directory
                output_dir = os.path.dirname(result_filepath)
                os.makedirs(output_dir, exist_ok=True)

                evaluation_tracker_args = simple_parse_args_string(f",output_path={result_filepath}")
                evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

                wandb_args_dict = simple_parse_args_string(f"project=marin,job_type=eval,name={model.name}")
                wandb_config_args_dict = simple_parse_args_string("")
                wandb_logger = WandbLogger(**wandb_args_dict, **wandb_config_args_dict)

                results = simple_evaluate(
                    model="vllm",
                    tasks=[eval_task.name],
                    num_fewshot=eval_task.num_fewshot,
                    model_args=pretrained_args,
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
