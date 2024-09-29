from typing import List
import os
import traceback

from scripts.evaluation.evaluator import Dependency, ModelConfig
from scripts.evaluation.vllm_tpu_evaluator import VllmTpuEvaluator
from scripts.evaluation.utils import is_remote_path, upload_to_gcs, run_bash_command


# TODO: this currently doesn't work on TPUs: https://github.com/vllm-project/vllm/issues/8499
class EleutherEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """

    RESULTS_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "eleuther_results")
    DEFAULT_MAX_EVAL_INSTANCES: int = 1000

    _pip_packages: List[Dependency] = VllmTpuEvaluator.DEFAULT_PIP_PACKAGES + [
        Dependency(name="lm_eval"),
        Dependency(name="lm-eval[api]"),
    ]

    def run(self, model: ModelConfig, evals: List[str], output_path: str, max_eval_instances: int | None = None) -> None:
        """
        Runs EleutherAI's lm-eval harness on the specified model and set of  tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[str]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
        """
        # From https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#model-apis-and-inference-servers
        # Run lm_eval with the model and the specified evals
        try:
            # Download the model from GCS or HuggingFace
            model_name_or_path: str = self.download_model(model)

            max_eval_instances = max_eval_instances or self.DEFAULT_MAX_EVAL_INSTANCES
            run_bash_command(
                [
                    "lm_eval",
                    "--model",
                    "vllm",
                    "--tasks",
                    ",".join(evals),
                    "--limit",
                    str(max_eval_instances),
                    "--model_args",
                    f"pretrained={model_name_or_path}",
                    "--batch_size",
                    "auto",
                    "--output_path",
                    self.RESULTS_PATH,
                ]
            )

            # Upload the results to GCS
            if is_remote_path(output_path):
                upload_to_gcs(self.RESULTS_PATH, output_path)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("lm-eval failed. Please check the logs for more information.") from e
        finally:
            self.cleanup(model)
