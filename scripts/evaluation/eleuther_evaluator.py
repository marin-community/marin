from typing import Dict, List
import os

import ray

from scripts.evaluation.evaluator import Dependency, ModelConfig
from scripts.evaluation.vllm_tpu_evaluator import VllmTpuEvaluator
from scripts.evaluation.utils import is_remote_path, upload_to_gcs, run_bash_command, write_yaml


class EleutherEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """

    # Following the defaults set in HELM: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/run.py
    PROD_ENV_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "prod_env")
    BENCHMARK_OUTPUT_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "benchmark_output")
    RESULTS_FOLDER: str = "results"
    RESULTS_PATH: str = os.path.join(BENCHMARK_OUTPUT_PATH, "runs", RESULTS_FOLDER)
    DEFAULT_MAX_EVAL_INSTANCES: int = 1000
    
    _pip_packages: List[Dependency] = VllmTpuEvaluator.DEFAULT_PIP_PACKAGES + [
        Dependency(name="lm_eval"),
    ]

    @ray.remote(memory=64 * 1024 * 1024 * 1024, resources={"TPU": 4})  # 64 GB of memory, always request 4 TPUs
    def run(self, model: ModelConfig, evals: List[str], output_path: str) -> None:
        super().run(model, evals, output_path)

        # Download the model from GCS or HuggingFace and serve it with vLLM
        self.start_vllm_server_in_background(model)

        # Run HELM with the model and the specified evals
        run_bash_command(
            f"lm_eval --model local-completions --tasks {evals} --model_args model={model},base_url=http://http://localhost:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=16 --output_path {self.BENCHMARK_OUTPUT_PATH}"
        )
        assert os.path.exists(self.RESULTS_PATH), f"Results not found at {self.RESULTS_PATH}. Did HELM run?"

        # Upload the results to GCS
        if is_remote_path(output_path):
            upload_to_gcs(self.RESULTS_PATH, output_path)
