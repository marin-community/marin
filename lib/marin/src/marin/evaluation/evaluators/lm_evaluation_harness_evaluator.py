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
                "HF_ALLOW_CODE_EVAL": "1",
                # Reduce noisy service teardown behavior in short-lived Ray workers
                "WANDB_SERVICE_WAIT": "0",
                # Keep logs minimal from wandb in worker processes
                "WANDB_SILENT": "true",
                # Suppress noisy C++ / libtpu logs like build labels, warnings, etc.
                # TF_CPP_MIN_LOG_LEVEL: 0=all, 1=filter INFO, 2=filter INFO/WARNING, 3=filter INFO/WARNING/ERROR.
                # We use 3 here to hide INFO and WARNING; only FATAL errors will surface.
                "TF_CPP_MIN_LOG_LEVEL": "3",
                # Some libtpu builds also respect TPU_MIN_LOG_LEVEL for their own logging.
                # Conventionally: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL; we set 2 to hide INFO/WARNING.
                "TPU_MIN_LOG_LEVEL": "2",
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

    def cleanup(self, model: ModelConfig) -> None:
        """
        Clean up TPU resources and model checkpoint to prevent Ray runtime env cleanup issues.
        """
        print("Cleaning up TPU resources and model checkpoint...")

        try:

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

    def _patch_max_position_embeddings_if_needed(self, model_local_dir: str, max_model_len: int) -> None:
        """
        Update max_position_embeddings and related fields in config.json to match max_model_len if needed.
        This ensures vLLM uses the correct max length.
        vLLM reads max_model_len from config.json fields like max_position_embeddings, n_positions, or max_seq_len.
        """
        logger.info(f"Patching config.json in {model_local_dir} with max_model_len={max_model_len}")
        if not os.path.isdir(model_local_dir):
            logger.warning(f"Model directory does not exist: {model_local_dir}")
            return
        config_json_path = os.path.join(model_local_dir, "config.json")
        if not os.path.exists(config_json_path):
            logger.warning(f"config.json does not exist at: {config_json_path}")
            return
        logger.info(f"Found config.json at: {config_json_path}")

        with open(config_json_path, "r") as f:
            cfg = json.load(f)

        updated = False
        # Update all relevant fields that vLLM might read
        for key in ["max_position_embeddings", "n_positions", "max_seq_len"]:
            if key in cfg and cfg[key] != max_model_len:
                old_value = cfg[key]
                cfg[key] = max_model_len
                logger.info(f"Updating {key} from {old_value} to {max_model_len}")
                updated = True
        
        if updated:
            try:
                with open(config_json_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                logger.info(f"Updated config.json with max_model_len={max_model_len}")
                # Verify the patch was applied correctly
                with open(config_json_path, "r") as f:
                    verify_cfg = json.load(f)
                    for key in ["max_position_embeddings", "n_positions", "max_seq_len"]:
                        if key in verify_cfg:
                            logger.info(f"Verified {key} in config.json: {verify_cfg[key]}")
                            if verify_cfg[key] != max_model_len:
                                logger.warning(f"WARNING: {key} verification failed! Expected {max_model_len}, got {verify_cfg[key]}")
            except Exception as e:
                logger.error(f"Failed to write updated config.json: {e}")
                raise
        else:
            logger.info(f"No config updates needed - all values already match max_model_len={max_model_len}")

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        max_gen_toks: int | None = None,
        generation_params: dict | None = None,
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

            # Build model args string first to get engine_kwargs
            engine_kwargs = dict(model.engine_kwargs) if model.engine_kwargs else {}
            
            # Patch config.json BEFORE vLLM initializes to ensure it reads the correct values
            # Ensure model config is compatible with vLLM rope scaling on TPU
            self._patch_rope_scaling_config_if_needed(model_name_or_path)
            
            # Patch max_position_embeddings if max_model_len is specified and different
            # This must happen BEFORE vLLM initializes, as vLLM reads from config.json
            if "max_model_len" in engine_kwargs and engine_kwargs["max_model_len"] is not None:
                self._patch_max_position_embeddings_if_needed(model_name_or_path, engine_kwargs["max_model_len"])
            
            pretrained_args_parts = [
                f"pretrained={model_name_or_path}",
                "device=tpu",
                "enforce_eager=True",
                "dtype=bfloat16",
                "max_num_seqs=1",
                "pipeline_parallel_size=1",
            ]
            for key, value in engine_kwargs.items():
                # Skip None values to avoid passing "key=None" which might not be parsed correctly
                if value is not None:
                    pretrained_args_parts.append(f"{key}={value}")
            pretrained_args = ",".join(pretrained_args_parts)
            
            # Log the model args to debug max_model_len
            logger.info(f"Model args string: {pretrained_args}")
            if "max_model_len" in engine_kwargs:
                logger.info(f"max_model_len from engine_kwargs: {engine_kwargs['max_model_len']}")
            else:
                logger.warning("max_model_len not found in engine_kwargs! This may cause truncation issues.")

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

                # Remove XLA_FLAGS that were meant for GPU - not needed on TPU
                # os.environ["XLA_FLAGS"] = (
                #     "--xla_gpu_enable_async_all_gather=false --xla_gpu_enable_async_all_reduce=false"
                # )
                # os.environ["XLA_USE_BF16"] = "1"

                # === MONKEY PATCH ===
                # We have the issue that lm-eval's importlib.metadata.version("vllm") check fails.
                # Thus, we create minimal vllm package metadata to satisfy lm-eval's importlib.metadata.version("vllm") check
                import site
                site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
                if site_packages:
                    vllm_dist_info = os.path.join(site_packages, "vllm-0.11.1.dist-info")
                    if not os.path.exists(vllm_dist_info):
                        os.makedirs(vllm_dist_info, exist_ok=True)
                        metadata_content = "Metadata-Version: 2.1\nName: vllm\nVersion: 0.11.1\n"
                        with open(os.path.join(vllm_dist_info, "METADATA"), "w") as f:
                            f.write(metadata_content)
                        logger.info(f"Created vllm package metadata at {vllm_dist_info}")

                # Monkey-patch resolve_hf_chat_template if it's missing or has issues.
                # The lm-eval vLLM backend has changed this helper a few times, so we
                # wrap it defensively to handle different call signatures.
                try:
                    import lm_eval.models.vllm_causallms as vllm_module
                    if not hasattr(vllm_module, "resolve_hf_chat_template"):
                        def resolve_hf_chat_template(tokenizer, *args, **kwargs):
                            """Fallback implementation if the function is missing.

                            We accept flexible arguments to match whatever lm-eval expects:
                            - resolve_hf_chat_template(tokenizer, messages, **kwargs)
                            - resolve_hf_chat_template(tokenizer=..., messages=..., **kwargs)
                            - resolve_hf_chat_template(tokenizer=..., **kwargs)  # messages in kwargs
                            """
                            # Extract messages from positional args or kwargs
                            if args:
                                messages = args[0]
                            else:
                                messages = kwargs.get("messages")

                            if messages is None:
                                logger.warning("resolve_hf_chat_template called without messages; returning None")
                                return None

                            if hasattr(tokenizer, "apply_chat_template"):
                                # Drop unsupported kwargs like 'tools'
                                filtered_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
                                return tokenizer.apply_chat_template(messages, **filtered_kwargs)
                            return None
                        vllm_module.resolve_hf_chat_template = resolve_hf_chat_template
                        logger.info("Monkey-patched resolve_hf_chat_template (missing)")
                    else:
                        # Patch to handle unexpected kwargs like 'tools'
                        original_resolve = vllm_module.resolve_hf_chat_template
                        def patched_resolve_hf_chat_template(tokenizer, *args, **kwargs):
                            """Wrapper around the upstream helper that is tolerant to extra kwargs."""
                            # Pull out messages from args/kwargs in the same flexible way
                            messages = None
                            if args:
                                messages = args[0]
                                remaining_args = args[1:]
                            else:
                                remaining_args = ()
                                messages = kwargs.get("messages")

                            try:
                                if messages is not None:
                                    return original_resolve(tokenizer, messages, *remaining_args, **kwargs)
                                # Fall back to original signature if we couldn't infer messages cleanly
                                return original_resolve(tokenizer, *args, **kwargs)
                            except TypeError as e:
                                if "unexpected keyword argument" in str(e):
                                    # Filter out problematic kwargs
                                    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["tools"]}
                                    if messages is not None:
                                        return original_resolve(tokenizer, messages, *remaining_args, **filtered_kwargs)
                                    return original_resolve(tokenizer, *remaining_args, **filtered_kwargs)
                                raise
                        vllm_module.resolve_hf_chat_template = patched_resolve_hf_chat_template
                        logger.info("Monkey-patched resolve_hf_chat_template (to handle unexpected kwargs)")
                except Exception as e:
                    logger.warning(f"Failed to monkey-patch resolve_hf_chat_template: {e}")

                # Note: max_gen_toks is controlled by lm-eval's internal logic, not vLLM.
                # There is no way to set max_gen_toks on the vLLM model instance.

                results = simple_evaluate(
                    model="vllm",
                    tasks=[eval_task.name],
                    num_fewshot=eval_task.num_fewshot,
                    model_args=pretrained_args,
                    apply_chat_template=model.apply_chat_template,
                    batch_size="auto",
                    confirm_run_unsafe_code=True,
                    limit=max_eval_instances if max_eval_instances is not None else None,
                    evaluation_tracker=evaluation_tracker,
                    log_samples=True, # This controls whether to log samples to the results file
                )

                if results is not None:
                    # lm-eval sometimes omits "samples" (e.g., for some tasks or failure modes).
                    # Be defensive here instead of hard-crashing with KeyError.
                    samples = results.pop("samples", None)
                    if samples is None:
                        logger.warning(
                            "lm-eval results missing 'samples' key; available keys: %s",
                            list(results.keys()),
                        )
                        evaluation_tracker.save_results_aggregated(results=results, samples=None)
                    else:
                        evaluation_tracker.save_results_aggregated(results=results, samples=samples)

                        try:
                            wandb_logger.post_init(results)
                            wandb_logger.log_eval_result()
                            wandb_logger.log_eval_samples(samples)
                            wandb_logger.run.finish()
                        except Exception as e:
                            print(f"Logging to Weights and Biases failed due to {e}")

                        for task_name in results.get("configs", {}).keys():
                            evaluation_tracker.save_results_samples(
                                task_name=task_name,
                                samples=samples.get(task_name, []),
                            )

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
