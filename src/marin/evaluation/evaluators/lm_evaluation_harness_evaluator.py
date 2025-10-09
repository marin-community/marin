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
import json
import os
import shutil
import traceback
import subprocess
import time

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from marin.run.ray_deps import build_runtime_env_for_packages

logger = logging.getLogger(__name__)


# TODO: Multiple choice tasks currently don't work on TPUs: https://github.com/vllm-project/vllm/issues/8499
class LMEvaluationHarnessEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """

    RESULTS_PATH: str = os.path.join(VllmTpuEvaluator.CACHE_PATH, "eleuther_results")

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(
            extra=["eval"],
            pip_packages=["pandas", "langdetect", "immutabledict"],
            env_vars={
                "HF_ALLOW_CODE_EVAL": "1"
            },  # Human eval tests code from the model which requires permission to run
        )

    def download_model(self, model: ModelConfig) -> str:
        """
        Download the model and ensure we get a local path for vLLM.
        """
        print(f"Downloading model: {model.name}, path: {model.path}")
        print(f"Cache path: {VllmTpuEvaluator.CACHE_PATH}")

        local_path = os.path.join(VllmTpuEvaluator.CACHE_PATH, model.name)
        print(f"Local download path: {local_path}")

        try:
            downloaded_path: str | None = model.ensure_downloaded(local_path=local_path)
            print(f"Download result: {downloaded_path}")
        except Exception as e:
            print(f"Download failed with exception: {e}")
            downloaded_path = None

        # Check if the local path exists even if ensure_downloaded returned None
        if downloaded_path is None and os.path.exists(local_path):
            print(f"Local path exists even though ensure_downloaded returned None: {local_path}")
            downloaded_path = local_path

        # For vLLM, we MUST have a local path, not a model name
        if downloaded_path is None:
            print(f"Final check - local_path exists: {os.path.exists(local_path)}")
            if os.path.exists(local_path):
                print(f"Using existing local path: {local_path}")
                return local_path
            else:
                raise ValueError(
                    f"Failed to download model {model.name} to local path {local_path}. \
                        vLLM requires a local filesystem path. Model path: {model.path}"
                )

        print(f"Final model path: {downloaded_path}")
        return downloaded_path

    def _cleanup_tpu_resources(self, wait_time: int = 2) -> None:
        """
        Clean up TPU resources (processes, lockfiles, cache).
        """
        try:
            # Kill any lingering vLLM processes
            subprocess.run(["pkill", "-f", "vllm"], check=False)
            subprocess.run(["pkill", "-f", "python.*tpu"], check=False)

            # Remove TPU lockfiles
            for i in range(8):
                try:
                    os.unlink(f"/tmp/libtpu_lockfile_{i}")
                except (FileNotFoundError, PermissionError):
                    pass

            # Wait for resources to be released
            if wait_time > 0:
                time.sleep(wait_time)

        except Exception as e:
            print(f"TPU resource cleanup failed: {e}")

    def cleanup(self, model: ModelConfig) -> None:
        """
        Clean up TPU resources and model checkpoint to prevent Ray runtime env cleanup issues.
        """
        print("Cleaning up TPU resources and model checkpoint...")

        try:
            # Clean up TPU resources
            self._cleanup_tpu_resources(wait_time=2)

            # Clean up model checkpoint
            model.destroy()

            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Cleanup failed: {e}")

    def _patch_rope_scaling_config_if_needed(self, model_local_dir: str) -> None:
        """
        Ensure rope_scaling in config.json is compatible with vLLM on TPU.

        - Adds missing rope_scaling["type"] as "su" when absent
        - Clamps rope_scaling["original_max_position_embeddings"] to be strictly
          less than max_position_embeddings when necessary
        """
        if not os.path.isdir(model_local_dir):
            return
        config_json_path = os.path.join(model_local_dir, "config.json")
        if not os.path.exists(config_json_path):
            return

        with open(config_json_path, "r") as f:
            cfg = json.load(f)

        rope_scaling = cfg.get("rope_scaling")
        max_pos = cfg.get("max_position_embeddings")
        updated = False
        removed = False

        if isinstance(rope_scaling, dict):
            # If type is missing or set to an unsupported type like 'su',
            # remove rope_scaling to avoid vLLM validation errors.
            rope_type = rope_scaling.get("type")
            if rope_type is None or rope_type == "su":
                cfg.pop("rope_scaling", None)
                updated = True
                removed = True
            else:
                if (
                    isinstance(max_pos, int)
                    and "original_max_position_embeddings" in rope_scaling
                    and isinstance(rope_scaling["original_max_position_embeddings"], int)
                    and rope_scaling["original_max_position_embeddings"] >= max_pos
                ):
                    rope_scaling["original_max_position_embeddings"] = max_pos - 1
                    updated = True

        if updated:
            if not removed:
                cfg["rope_scaling"] = rope_scaling
            with open(config_json_path, "w") as f:
                json.dump(cfg, f)

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        max_gen_toks: int | None = None,
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

            # Ensure model config is compatible with vLLM rope scaling on TPU
            self._patch_rope_scaling_config_if_needed(model_name_or_path)

            # Build model args string
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

                wandb_args_dict = {
                    "project": "marin",
                    "job_type": "eval",
                    "name": model.name,
                    "tags": wandb_tags,
                }

                wandb_logger = WandbLogger(**wandb_args_dict)

                # Use vLLM directly with TPU configuration that avoids deserialization issues
                # Set environment variables to help with TPU compilation
                # Clean up TPU resources before vLLM initialization
                print("Cleaning up TPU resources before vLLM initialization...")
                self._cleanup_tpu_resources(wait_time=5)
                print("TPU cleanup completed")

                os.environ["XLA_FLAGS"] = (
                    "--xla_gpu_enable_async_all_gather=false --xla_gpu_enable_async_all_reduce=false"
                )
                os.environ["XLA_USE_BF16"] = "1"

                # Note: vLLM expects 'pretrained' argument, not 'model'
                results = simple_evaluate(
                    model="vllm",
                    tasks=[eval_task.name],
                    num_fewshot=eval_task.num_fewshot,
                    model_args=f"pretrained={model_name_or_path},device=tpu,enforce_eager=True,dtype=bfloat16,tensor_parallel_size=8,pipeline_parallel_size=1,max_num_seqs=1",
                    apply_chat_template=model.apply_chat_template,
                    batch_size="auto",
                    confirm_run_unsafe_code=True,
                    limit=max_eval_instances if max_eval_instances is not None else None,
                    evaluation_tracker=evaluation_tracker,
                    log_samples=False,
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
            try:
                if is_remote_path(output_path):
                    try:
                        logger.info("Uploading eval results to GCS...")
                        upload_to_gcs(self.RESULTS_PATH, output_path)
                        logger.info("Upload completed successfully.")
                    except Exception as upload_error:
                        logger.info(f"Failed to upload results to GCS: {upload_error}")

                # Clean up TPU resources and model checkpoint
                self.cleanup(model)

                # Clean up results directory
                if os.path.exists(self.RESULTS_PATH):
                    shutil.rmtree(self.RESULTS_PATH)

                print("Final cleanup completed successfully")
            except Exception as cleanup_error:
                print(f"Final cleanup failed: {cleanup_error}")
                # Don't raise here to avoid masking the original error
