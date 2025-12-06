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
import tempfile
import traceback
from pathlib import Path

from fray.cluster.ray.deps import build_runtime_env_for_packages
from fray.isolated_env import TemporaryVenv

from marin.evaluation.evaluation_config import EvalTaskConfig, ModelConfig
from marin.evaluation.evaluators.evaluator import Evaluator
from marin.evaluation.utils import upload_to_gcs, write_yaml

# Following the defaults set in HELM: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/run.py
DEFAULT_MAX_EVAL_INSTANCES: int = 1000

MODEL_DEPLOYMENTS_FILE_PATH: str = "model_deployments.yaml"
MODEL_METADATA_FILE_PATH: str = "model_metadata.yaml"
TOKENIZER_CONFIGS_FILE_PATH: str = "tokenizer_configs.yaml"

ALL_RUN_ENTRIES_URL: str = "https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation"
RUN_ENTRIES_TEMPLATE: str = (
    "https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/presentation/{run_entries_file}"
)
ALL_SCHEMA_URL: str = "https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/static"
SCHEMA_TEMPLATE: str = (
    "https://raw.githubusercontent.com/stanford-crfm/helm/refs/heads/main/src/helm/benchmark/static/{schema_file}"
)

logger = logging.getLogger(__name__)


def ensure_file_downloaded(url: str, target_path: str) -> None:
    if os.path.exists(target_path):
        return

    import requests

    response = requests.get(url)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(response.content)


def write_model_config_files(model: ModelConfig, base_url: str, prod_env_path: Path) -> None:
    """
    Write out the necessary model configuration files for HELM.
    """
    from transformers import AutoTokenizer

    os.makedirs(prod_env_path, exist_ok=True)

    model_name: str = model.name
    # Use model.path for loading from HuggingFace, fallback to model.name if path is None
    model_path_or_name: str = model.path or model.name
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)

    content: dict = {
        "model_deployments": [
            {
                "name": model_name,
                "model_name": model_name,
                "tokenizer_name": model_name,
                "max_sequence_length": tokenizer.model_max_length,
                "client_spec": {
                    "class_name": "helm.clients.vllm_client.VLLMClient",
                    "args": {
                        "base_url": base_url,
                    },
                },
            }
        ]
    }
    deployments_path = prod_env_path / MODEL_DEPLOYMENTS_FILE_PATH
    write_yaml(content, deployments_path)

    content = {
        "models": [
            {
                "name": model_name,
                "display_name": model_name,
                "description": "",
                "creator_organization_name": "",
                "access": "open",
                "release_date": None,
                "tags": ["TEXT_MODEL_TAG", "LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG"],
            }
        ]
    }
    metadata_path = prod_env_path / MODEL_METADATA_FILE_PATH
    write_yaml(content, metadata_path)

    content = {
        "tokenizer_configs": [
            {
                "name": model_name,
                "tokenizer_spec": {
                    "class_name": "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
                    "args": {"pretrained_model_name_or_path": model_path_or_name, "trust_remote_code": True},
                },
                "prefix_token": tokenizer.bos_token,
                "end_of_text_token": tokenizer.eos_token,
            }
        ]
    }
    tokenizer_path = prod_env_path / TOKENIZER_CONFIGS_FILE_PATH
    write_yaml(content, tokenizer_path)


class HELMEvaluator(Evaluator):
    """
    Evaluator that runs HELM: https://github.com/stanford-crfm/helm
    """

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(extra=["eval", "tpu"])

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
        Runs HELM on the specified model and set of evaluations.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate

            evals (List[EvalTaskConfig]): The list of evaluations to run.
                As of now we don't use num_fewshot from the EvalTaskConfig.

            output_path (str): The path to save the evaluation results.

            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            wandb_tags (list[str] | None): Optional tags to add to the wandb run (unused).
        """
        results_tmp = tempfile.TemporaryDirectory()
        results_path = results_tmp.name
        prod_env_path = Path(results_path) / "prod_env"
        results_folder = Path(results_path) / "run" / "results"

        # Use isolated venv for HELM to avoid dependency conflicts
        with TemporaryVenv(
            pip_install_args=[
                "crfm-helm@git+https://github.com/stanford-crfm/helm.git",
                "openai",  # Required by HELM's OpenAI client
            ],
            prefix="helm_venv_",
        ) as venv:
            try:
                # Download the run_entries files and schema files for the specified evals
                assert len(evals) > 0, "Please specify at least one eval to run."
                run_entries_files: list[str] = []
                schema_files: list[str] = []
                for helm_eval in evals:
                    run_entries_file: str = f"run_entries_{helm_eval.name}.conf"
                    run_entries_url: str = RUN_ENTRIES_TEMPLATE.format(run_entries_file=run_entries_file)
                    ensure_file_downloaded(url=run_entries_url, target_path=run_entries_file)
                    assert os.path.exists(
                        run_entries_file
                    ), f"Failed to download. Does {run_entries_file} exist at {ALL_RUN_ENTRIES_URL}?"
                    run_entries_files.append(run_entries_file)

                    schema_file: str = f"schema_{helm_eval.name}.yaml"
                    schema_url: str = SCHEMA_TEMPLATE.format(schema_file=schema_file)
                    ensure_file_downloaded(url=schema_url, target_path=schema_file)
                    assert os.path.exists(
                        schema_file
                    ), f"Failed to download. Does {schema_file} exist at {ALL_SCHEMA_URL}?"
                    schema_files.append(schema_file)

                write_model_config_files(model, openai_base_url, prod_env_path)

                max_eval_instances = max_eval_instances or DEFAULT_MAX_EVAL_INSTANCES

                venv.run(
                    [
                        "helm-run",
                        "--conf-paths",
                        *run_entries_files,
                        "--models-to-run",
                        model.name,
                        "--max-eval-instances",
                        str(max_eval_instances),
                        "--output-path",
                        results_path,
                        "--suite",
                        str(results_folder),
                        "--local-path",
                        str(prod_env_path),
                        "--num-threads",
                        "1",
                        "--exit-on-error",
                    ],
                    check=True,
                )
                assert os.path.exists(results_folder), f"Results not found at {results_folder}. Did HELM run?"

                # Run helm-summarize, which aggregates all the results and generates tables for them.
                # See https://crfm-helm.readthedocs.io/en/latest/get_helm_rank for more information.
                venv.run(
                    [
                        "helm-summarize",
                        "--suite",
                        str(results_folder),
                        "--output-path",
                        results_path,
                        "--schema-path",
                        # helm-summarize only takes one schema file, so we just use the first one
                        schema_files[0],
                    ],
                    check=True,
                )

                upload_to_gcs(results_path, output_path)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError("HELM failed. Please check the logs for more information.") from e
