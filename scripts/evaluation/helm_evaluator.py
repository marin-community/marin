from typing import Dict, List
import os

import ray

from scripts.evaluation.evaluator import Dependency, Model
from scripts.evaluation.vllm_tpu_evaluator import VllmTpuEvaluator
from scripts.evaluation.utils import is_gcs_path, upload_to_gcs, run_bash_command, write_yaml


class HELMEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs HELM: https://github.com/stanford-crfm/helm
    """

    # Following the defaults set in HELM: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/run.py
    PROD_ENV_FOLDER: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "prod_env")
    RESULTS_FOLDER: str = "results"
    RESULTS_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "benchmark_output", "runs", RESULTS_FOLDER)
    DEFAULT_MAX_EVAL_INSTANCES: int = 1000

    # Required files to run inference on a particular model in HELM. All of these files are in `PROD_ENV_FOLDER`.
    MODEL_DEPLOYMENTS_FILE_PATH: str = os.path.join(PROD_ENV_FOLDER, "model_deployments.yaml")
    MODEL_METADATA_FILE_PATH: str = os.path.join(PROD_ENV_FOLDER, "model_metadata.yaml")
    TOKENIZER_CONFIGS_FILE_PATH: str = os.path.join(PROD_ENV_FOLDER, "tokenizer_configs.yaml")

    ALL_RUN_ENTRIES_URL: str = "https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation"
    RUN_ENTRIES_TEMPLATE: str = (
        "https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/presentation/{run_entries_file}"
    )

    _pip_packages: List[Dependency] = VllmTpuEvaluator.DEFAULT_PIP_PACKAGES + [
        Dependency(name="crfm-helm@git+https://github.com/stanford-crfm/helm.git@helm_on_tpu"),
    ]

    @staticmethod
    def write_model_config_files(model: Model) -> None:
        """
        Write out the necessary model configuration files for HELM.
        """
        # TODO: Works for our olmo checkpoints, but make this configurable to support any model
        model_name: str = model.name
        tokenizer_name: str = "allenai/olmo-7b"
        content: Dict = {
            "model_deployments": [
                {
                    "name": model_name,
                    "model_name": model_name,
                    "tokenizer_name": tokenizer_name,
                    "max_sequence_length": 4096,
                    "client_spec": {
                        "class_name": "helm.clients.vllm_client.VLLMClient",
                        "args": {"base_url": "http://localhost:8000/v1"},
                    },
                }
            ]
        }
        write_yaml(content, HELMEvaluator.MODEL_DEPLOYMENTS_FILE_PATH)

        content = {
            "models": [
                {
                    "name": model_name,
                    "display_name": model_name,
                    "description": "OLMo is a series of Open Language Models trained on the Dolma dataset.",
                    "creator_organization_name": "Allen Institute for AI",
                    "access": "open",
                    "num_parameters": 7000000000,
                    "release_date": "2024-02-01",
                    "tags": ["TEXT_MODEL_TAG", "LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG"],
                }
            ]
        }
        write_yaml(content, HELMEvaluator.MODEL_METADATA_FILE_PATH)

        content = {
            "tokenizer_configs": [
                {
                    "name": tokenizer_name,
                    "tokenizer_spec": {
                        "class_name": "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
                        "args": {"trust_remote_code": True},
                    },
                    "end_of_text_token": "<|endoftext|>",
                    "prefix_token": "",
                }
            ]
        }
        write_yaml(content, HELMEvaluator.TOKENIZER_CONFIGS_FILE_PATH)

    @ray.remote(memory=8 * 1024 * 1024 * 1024, resources={"TPU": 4})  # 8 GB of memory, always request 4 TPUs
    def run(self, model: Model, evals: List[str], output_path: str) -> None:
        super().run(model, evals, output_path)

        from helm.common.general import ensure_file_downloaded

        # Download the model from GCS or HuggingFace and serve it with vLLM
        self.start_vllm_server_in_background(model)

        # Download the run_entries file specified in `evals`
        for run_entries_file in evals:
            run_entries_url: str = self.RUN_ENTRIES_TEMPLATE.format(run_entries_file=run_entries_file)
            ensure_file_downloaded(source_url=run_entries_url, target_path=run_entries_file)
            assert (
                os.path.exists(run_entries_file),
                f"Failed to download. Does {run_entries_file} exist at {self.ALL_RUN_ENTRIES_URL}?",
            )

        # Write the model configuration files necessary for HELM
        self.write_model_config_files(model)

        # Run HELM with the model and the specified evals
        run_bash_command(
            f"helm-run --conf-paths {' '.join(evals)} "
            f"--models-to-run {model.name} "
            f"--max-eval-instances {self.DEFAULT_MAX_EVAL_INSTANCES} "
            f"--suite {self.RESULTS_FOLDER} "
            f"--local-path {self.PROD_ENV_FOLDER} "
        )
        assert os.path.exists(self.RESULTS_PATH), f"Results not found at {self.RESULTS_PATH}. Did HELM run?"

        # Upload the results to GCS
        if is_gcs_path(output_path):
            upload_to_gcs(self.RESULTS_PATH, output_path)
