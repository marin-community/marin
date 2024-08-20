from typing import List
import os

import ray

from scripts.evaluation.evaluator import Dependency
from scripts.evaluation.vllm_tpu_evaluator import VllmTpuEvaluator


class HELMEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs HELM: https://github.com/stanford-crfm/helm
    """

    # Following the defaults set in HELM: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/run.p
    CACHE_FOLDER: str = "prod_env"
    RESULTS_FOLDER: str = "results"
    RESULTS_PATH: str = os.path.join("benchmark_output", "runs", RESULTS_FOLDER)
    DEFAULT_MAX_EVAL_INSTANCES: int = 1000

    ALL_RUN_ENTRIES_URL: str = "https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation"
    RUN_ENTRIES_TEMPLATE: str = (
        "https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/presentation/{run_entries_file}"
    )

    _pip_packages: List[Dependency] = VllmTpuEvaluator.DEFAULT_PIP_PACKAGES + [
        Dependency(name="crfm-helm@git+https://github.com/stanford-crfm/helm.git@helm_on_tpu"),
    ]

    @ray.remote(memory=8 * 1024 * 1024 * 1024, resources={"TPU": 4})  # 8 GB of memory, always request 4 TPUs
    def run(self, model_name_or_path: str, evals: List[str]) -> None:
        super().run(model_name_or_path, evals)

        from helm.common.general import ensure_file_downloaded

        # Download the model from GCS or HuggingFace and serve it with vLLM
        self.start_vllm_server_in_background(model_name_or_path)

        # Download the run_entries file specified in `evals`
        for run_entries_file in evals:
            run_entries_url: str = self.RUN_ENTRIES_TEMPLATE.format(run_entries_file=run_entries_file)
            ensure_file_downloaded(source_url=run_entries_url, target_path=run_entries_file)
            assert (
                os.path.exists(run_entries_file),
                f"Failed to download. Does {run_entries_file} exist at {self.ALL_RUN_ENTRIES_URL}?",
            )

        # Run HELM with the model and the specified evals
        HELMEvaluator.run_bash_command(
            f"helm-run --conf-paths {' '.join(evals)} "
            f"--models-to-run {model_name_or_path} "
            f"--max-eval-instances {self.DEFAULT_MAX_EVAL_INSTANCES} "
            f"--suite {self.RESULTS_FOLDER} "
            f"--local-path {self.CACHE_FOLDER} "
            "--num-threads 1 "
            "--exit-on-error"
        )
        assert os.path.exists(self.RESULTS_PATH), f"Failed to generate results at {self.RESULTS_PATH}."

        # Upload the results to GCS
        if VllmTpuEvaluator.is_gcs_path(self._config.output_path):
            self.upload_to_gcs(self.RESULTS_PATH, self._config.output_path)
