from typing import List
import os

import ray

from scripts.evaluation.evaluator import Dependency, ModelConfig
from scripts.evaluation.vllm_tpu_evaluator import VllmTpuEvaluator
from scripts.evaluation.utils import is_remote_path, upload_to_gcs, run_bash_command


class EleutherEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """
    RESULTS_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "eleuther_results")
    
    _pip_packages: List[Dependency] = VllmTpuEvaluator.DEFAULT_PIP_PACKAGES + [Dependency(name="lm_eval")]

    @ray.remote(memory=64 * 1024 * 1024 * 1024, resources={"TPU": 4})  # 64 GB of memory, always request 4 TPUs
    def run(self, model: ModelConfig, evals: List[str], output_path: str) -> None:
        # Installs and starts the vLLM server in the background
        super().run(model, evals, output_path)

        # Download the model from GCS or HuggingFace and serve it with vLLM
        server_url: str = self.start_vllm_server_in_background(model)

        # From https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#model-apis-and-inference-servers
        # Run lm_eval with the model and the specified evals
        model_name_or_path: str = model.name if model.path is None else model.path
        run_bash_command(
            f"lm_eval --model local-completions --tasks {evals} "
            f"--model_args model={model_name_or_path},base_url={server_url}/completions,"
            # Used the default values from the link above
            f"num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=16 "
            f"--output_path {self.RESULTS_PATH}"
        )

        # Upload the results to GCS
        if is_remote_path(output_path):
            upload_to_gcs(self.RESULTS_PATH, output_path)
