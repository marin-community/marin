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
import shutil
import tempfile
import traceback

import fsspec
import requests

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from fray.cluster.ray.deps import build_runtime_env_for_packages

logger = logging.getLogger(__name__)


# TODO: Multiple choice tasks currently don't work on TPUs: https://github.com/vllm-project/vllm/issues/8499
class LMEvaluationHarnessEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """

    CACHE_PATH: str = "/tmp/lm-eval"
    RESULTS_PATH: str = os.path.join(CACHE_PATH, "eleuther_results")
    TOKENIZER_FILENAMES: tuple[str, ...] = (
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "config.json",
    )

    @classmethod
    def _stage_remote_tokenizer_dir(cls, remote_dir: str) -> str | None:
        local_dir = tempfile.mkdtemp(prefix="marin-tokenizer-")
        copied_any = False
        try:
            for filename in cls.TOKENIZER_FILENAMES:
                remote_path = f"{remote_dir.rstrip('/')}/{filename}"
                if not is_remote_path(remote_path):
                    continue
                fs, fs_path = fsspec.core.url_to_fs(remote_path)
                if not fs.exists(fs_path):
                    continue
                local_path = os.path.join(local_dir, filename)
                with fsspec.open(remote_path, "rb") as src:
                    data = src.read()
                with open(local_path, "wb") as dst:
                    dst.write(data)
                copied_any = True
        except Exception:
            shutil.rmtree(local_dir, ignore_errors=True)
            raise

        if not copied_any:
            shutil.rmtree(local_dir, ignore_errors=True)
            return None
        return local_dir

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(
            extra=["eval"],
            env_vars={
                "HF_ALLOW_CODE_EVAL": "1"
            },  # Human eval tests code from the model which requires permission to run
        )

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
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
        vllm_server = None
        staged_tokenizer_dir = None
        try:
            model_name_or_path, model = self.resolve_model_name_or_path(model)

            mode_str = os.environ.get("MARIN_VLLM_MODE", "docker").lower()
            if mode_str == "docker":
                vllm_server = self.start_vllm_server_in_background(
                    model=model,
                    mode="docker",
                )
                response = requests.get(f"{vllm_server.server_url}/models", timeout=30)
                response.raise_for_status()
                model_list = response.json().get("data", [])
                if not model_list:
                    raise RuntimeError(
                        f"No models returned from {vllm_server.server_url}/models: {response.text[:2000]}"
                    )
                model_id = model_list[0].get("id")
                if not model_id:
                    raise RuntimeError(
                        f"Missing model id in {vllm_server.server_url}/models response: {response.text[:2000]}"
                    )

                tokenizer = None
                if isinstance(model.engine_kwargs.get("tokenizer"), str):
                    tokenizer = model.engine_kwargs.get("tokenizer")
                elif is_remote_path(model_name_or_path):
                    staged_tokenizer_dir = self._stage_remote_tokenizer_dir(model_name_or_path)
                    if staged_tokenizer_dir is None:
                        raise ValueError(
                            "lm-eval's `local-completions` model requires a Hugging Face tokenizer name/path, but "
                            f"the served model id is a remote object-store URI: {model_id!r}, and no tokenizer files "
                            f"were found under {model_name_or_path!r}. "
                            "Set `engine_kwargs['tokenizer']` to an HF tokenizer id (e.g. "
                            "'meta-llama/Llama-3.1-8B-Instruct') or a local tokenizer path."
                        )
                    tokenizer = staged_tokenizer_dir

                # Use lm-eval's API model wrapper to talk to the sidecar's OpenAI-compatible endpoint.
                if model.apply_chat_template:
                    lm_eval_model = "local-chat-completions"
                    pretrained_args = (
                        f"model={model_id},"
                        f"base_url={vllm_server.server_url}/chat/completions,"
                        "tokenizer_backend=huggingface,"
                        "tokenized_requests=False"
                    )
                else:
                    lm_eval_model = "local-completions"
                    pretrained_args = (
                        f"model={model_id},"
                        f"base_url={vllm_server.server_url}/completions,"
                        "tokenizer_backend=huggingface,"
                        "tokenized_requests=False"
                    )
                if tokenizer is not None:
                    pretrained_args += f",tokenizer={tokenizer}"
                if model.engine_kwargs:
                    for key, value in model.engine_kwargs.items():
                        if key == "tokenizer":
                            continue
                        pretrained_args += f",{key}={value}"
            else:
                lm_eval_model = "vllm"
                pretrained_args = f"pretrained={model_name_or_path}"
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

                wandb_args_dict = {
                    "project": "marin",
                    "job_type": "eval",
                    "name": model.name,
                    "tags": wandb_tags,
                }
                # wandb_config_args_dict = simple_parse_args_string("")
                wandb_logger = WandbLogger(init_args=wandb_args_dict)

                results = simple_evaluate(
                    model=lm_eval_model,
                    tasks=[eval_task.name],
                    num_fewshot=eval_task.num_fewshot,
                    model_args=pretrained_args,
                    apply_chat_template=model.apply_chat_template,
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

            self.cleanup(model, vllm_server=vllm_server)
            if os.path.exists(self.RESULTS_PATH):
                shutil.rmtree(self.RESULTS_PATH)
            if staged_tokenizer_dir and os.path.exists(staged_tokenizer_dir):
                shutil.rmtree(staged_tokenizer_dir, ignore_errors=True)
