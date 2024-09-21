from typing import List
import os

import ray

from scripts.evaluation.evaluator import Dependency, ModelConfig
from scripts.evaluation.vllm_tpu_evaluator import VllmTpuEvaluator
from scripts.evaluation.utils import is_remote_path, upload_to_gcs, run_bash_command


# TODO: this currently doesn't work on TPUs: https://github.com/vllm-project/vllm/issues/8499
class EleutherEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """

    RESULTS_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "eleuther_results")

    _pip_packages: List[Dependency] = VllmTpuEvaluator.DEFAULT_PIP_PACKAGES + [
        Dependency(name="lm_eval"),
        Dependency(name="lm-eval[api]"),
    ]

    @ray.remote(memory=64 * 1024 * 1024 * 1024, resources={"TPU": 4})  # 64 GB of memory, always request 4 TPUs
    def run(self, model: ModelConfig, evals: List[str], output_path: str) -> None:
        # Download the model from GCS or HuggingFace
        model.ensure_downloaded(local_path=os.path.join(VllmTpuEvaluator.CACHE_PATH, model.name))

        # From https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#model-apis-and-inference-servers
        # Run lm_eval with the model and the specified evals
        model_name_or_path: str = model.name if model.path is None else model.path

        try:
            run_bash_command(
                [
                    "lm_eval",
                    "--model",
                    "vllm",
                    "--tasks",
                    ",".join(evals),
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
            print(f"An error occurred: {e}")
        finally:
            self.cleanup(model)
