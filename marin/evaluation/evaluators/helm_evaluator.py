import os
import shutil
import traceback
from typing import ClassVar

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import is_remote_path, run_bash_command, upload_to_gcs, write_yaml


class HELMEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs HELM: https://github.com/stanford-crfm/helm
    """

    # Following the defaults set in HELM: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/run.py
    PROD_ENV_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "prod_env")
    BENCHMARK_OUTPUT_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "benchmark_output")
    RESULTS_FOLDER: str = "results"
    RESULTS_PATH: str = os.path.join(BENCHMARK_OUTPUT_PATH, "runs", RESULTS_FOLDER)
    DEFAULT_MAX_EVAL_INSTANCES: int = 1000

    # Required files to run inference on a particular model in HELM. All of these files are in `PROD_ENV_PATH`.
    MODEL_DEPLOYMENTS_FILE_PATH: str = os.path.join(PROD_ENV_PATH, "model_deployments.yaml")
    MODEL_METADATA_FILE_PATH: str = os.path.join(PROD_ENV_PATH, "model_metadata.yaml")
    TOKENIZER_CONFIGS_FILE_PATH: str = os.path.join(PROD_ENV_PATH, "tokenizer_configs.yaml")

    ALL_RUN_ENTRIES_URL: str = "https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation"
    RUN_ENTRIES_TEMPLATE: str = (
        "https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/presentation/{run_entries_file}"
    )
    ALL_SCHEMA_URL: str = "https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/static"
    SCHEMA_TEMPLATE: str = (
        "https://raw.githubusercontent.com/stanford-crfm/helm/refs/heads/main/src/helm/benchmark/static/{schema_file}"
    )

    _pip_packages: ClassVar[list[Dependency]] = [
        *VllmTpuEvaluator.DEFAULT_PIP_PACKAGES,
        Dependency(name="crfm-helm@git+https://github.com/stanford-crfm/helm.git@local_vllm"),
    ]

    @staticmethod
    def write_model_config_files(model: ModelConfig) -> None:
        """
        Write out the necessary model configuration files for HELM.
        """
        from transformers import AutoTokenizer

        os.makedirs(HELMEvaluator.PROD_ENV_PATH, exist_ok=True)

        model_name: str = model.name
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        content: dict = {
            "model_deployments": [
                {
                    "name": model_name,
                    "model_name": model_name,
                    "tokenizer_name": model_name,
                    "max_sequence_length": tokenizer.model_max_length,
                    "client_spec": {"class_name": "helm.clients.vllm_client.LocalVLLMClient"},
                }
            ]
        }
        write_yaml(content, HELMEvaluator.MODEL_DEPLOYMENTS_FILE_PATH)

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
        write_yaml(content, HELMEvaluator.MODEL_METADATA_FILE_PATH)

        content = {
            "tokenizer_configs": [
                {
                    "name": model_name,
                    "tokenizer_spec": {
                        "class_name": "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
                        "args": {"pretrained_model_name_or_path": model_name, "trust_remote_code": True},
                    },
                    "prefix_token": tokenizer.bos_token,
                    "end_of_text_token": tokenizer.eos_token,
                }
            ]
        }
        write_yaml(content, HELMEvaluator.TOKENIZER_CONFIGS_FILE_PATH)

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
    ) -> None:
        """
        Runs HELM on the specified model and set of evaluations.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate

            evals (List[EvalTaskConfig]): The list of evaluations to run.
                As of now we don't use num_fewshot from the EvalTaskConfig.

            output_path (str): The path to save the evaluation results.

            max_eval_instances (int | None): The maximum number of evaluation instances to run.
        """
        try:
            from helm.common.general import ensure_file_downloaded

            # Download the run_entries files and schema files for the specified evals
            assert len(evals) > 0, "Please specify at least one eval to run."
            run_entries_files: list[str] = []
            schema_files: list[str] = []
            for helm_eval in evals:
                run_entries_file: str = f"run_entries_{helm_eval.name}.conf"
                run_entries_url: str = self.RUN_ENTRIES_TEMPLATE.format(run_entries_file=run_entries_file)
                ensure_file_downloaded(source_url=run_entries_url, target_path=run_entries_file)
                assert os.path.exists(
                    run_entries_file
                ), f"Failed to download. Does {run_entries_file} exist at {self.ALL_RUN_ENTRIES_URL}?"
                run_entries_files.append(run_entries_file)

                schema_file: str = f"schema_{helm_eval.name}.yaml"
                schema_url: str = self.SCHEMA_TEMPLATE.format(schema_file=schema_file)
                ensure_file_downloaded(source_url=schema_url, target_path=schema_file)
                assert os.path.exists(
                    schema_file
                ), f"Failed to download. Does {schema_file} exist at {self.ALL_SCHEMA_URL}?"
                schema_files.append(schema_file)

            # Download the model checkpoint if necessary.
            self.download_model(model)
            # HELM requires the model name to match the local path
            if model.path is not None:
                model.name = model.path

            # Write the model configuration files necessary for HELM
            self.write_model_config_files(model)

            # Run HELM with the model and the specified evals
            # This commands evaluates the model on the specified evals and outputs the results to RESULTS_PATH
            max_eval_instances = max_eval_instances or self.DEFAULT_MAX_EVAL_INSTANCES
            run_bash_command(
                [
                    "helm-run",
                    "--conf-paths",
                    *run_entries_files,  # Use `*` to expand the list into separate arguments
                    "--models-to-run",
                    model.name,
                    "--max-eval-instances",
                    str(max_eval_instances),
                    "--output-path",
                    self.BENCHMARK_OUTPUT_PATH,
                    "--suite",
                    self.RESULTS_FOLDER,
                    "--local-path",
                    self.PROD_ENV_PATH,
                    "--num-threads",
                    "1",
                    "--exit-on-error",
                ]
            )
            assert os.path.exists(self.RESULTS_PATH), f"Results not found at {self.RESULTS_PATH}. Did HELM run?"

            # Run helm-summarize, which aggregates all the results and generates tables for them.
            # See https://crfm-helm.readthedocs.io/en/latest/get_helm_rank for more information.
            run_bash_command(
                [
                    "helm-summarize",
                    "--suite",
                    self.RESULTS_FOLDER,
                    "--output-path",
                    self.BENCHMARK_OUTPUT_PATH,
                    "--schema-path",
                    # helm-summarize only takes one schema file, so we just use the first one
                    schema_files[0],
                ]
            )

            # Upload the results to GCS
            if is_remote_path(output_path):
                upload_to_gcs(self.RESULTS_PATH, output_path)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("HELM failed. Please check the logs for more information.") from e
        finally:
            self.cleanup(model)
            shutil.rmtree(self.RESULTS_PATH, ignore_errors=True)
