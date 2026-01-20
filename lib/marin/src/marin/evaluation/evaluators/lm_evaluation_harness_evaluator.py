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

import hashlib
import inspect
import json
import logging
import os
import shutil
from contextlib import contextmanager

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
    CONFIG_CACHE_PATH: str = os.path.join(CACHE_PATH, "config_cache")

    # Config files needed for lm-eval (AutoConfig, tokenizer) but NOT model weights
    CONFIG_FILES: list[str] = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",  # For SentencePiece models
        "generation_config.json",
        "added_tokens.json",
        "chat_template.jinja",  # Chat template file (transformers picks this up automatically)
    ]

    def _download_config_files_from_gcs(self, gcs_path: str) -> str:
        """
        Download only config/tokenizer files from GCS to a local directory.
        This allows lm-eval to load config via AutoConfig.from_pretrained() while
        vLLM streams the actual model weights directly from GCS.

        Args:
            gcs_path: The GCS path to the model (e.g., gs://bucket/path/to/model)

        Returns:
            Local directory path containing the downloaded config files.
        """
        import fsspec

        # Create a unique local directory based on the GCS path
        path_hash = hashlib.md5(gcs_path.encode()).hexdigest()[:8]
        local_dir = os.path.join(self.CONFIG_CACHE_PATH, f"config_{path_hash}")
        os.makedirs(local_dir, exist_ok=True)

        fs = fsspec.filesystem("gcs")
        gcs_path_clean = gcs_path.rstrip("/")

        for filename in self.CONFIG_FILES:
            remote_file = f"{gcs_path_clean}/{filename}"
            local_file = os.path.join(local_dir, filename)
            try:
                if fs.exists(remote_file):
                    fs.get(remote_file, local_file)
                    logger.info(f"Downloaded {filename} from GCS to {local_file}")
            except Exception as e:
                # Not all files are required (e.g., vocab.json vs tokenizer.model)
                logger.debug(f"Could not download {filename}: {e}")

        return local_dir

    @contextmanager
    def _patch_autoconfig_for_gcs(self, gcs_path: str, local_config_dir: str):
        """
        Context manager that patches AutoConfig.from_pretrained and AutoTokenizer.from_pretrained
        to redirect GCS paths to a local config directory.

        This allows lm-eval to use AutoConfig/AutoTokenizer.from_pretrained() with GCS paths
        by transparently redirecting to the locally downloaded config files,
        while vLLM still uses the GCS path directly for streaming model weights.

        Args:
            gcs_path: The original GCS path that will be redirected
            local_config_dir: Local directory containing the downloaded config files
        """
        from transformers import AutoConfig, AutoTokenizer

        # Store the original methods (these are classmethods)
        original_config_from_pretrained = AutoConfig.from_pretrained.__func__
        original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained.__func__

        def patched_config_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            # If the path matches our GCS path, redirect to local config directory
            if pretrained_model_name_or_path == gcs_path:
                logger.info(f"Redirecting AutoConfig.from_pretrained from {gcs_path} to {local_config_dir}")
                return original_config_from_pretrained(cls, local_config_dir, *args, **kwargs)
            return original_config_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        def patched_tokenizer_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            # If the path matches our GCS path, redirect to local config directory
            if pretrained_model_name_or_path == gcs_path:
                logger.info(f"Redirecting AutoTokenizer.from_pretrained from {gcs_path} to {local_config_dir}")
                return original_tokenizer_from_pretrained(cls, local_config_dir, *args, **kwargs)
            return original_tokenizer_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Apply the patches (as classmethods)
        AutoConfig.from_pretrained = classmethod(patched_config_from_pretrained)
        AutoTokenizer.from_pretrained = classmethod(patched_tokenizer_from_pretrained)
        try:
            yield
        finally:
            # Restore the original methods
            AutoConfig.from_pretrained = classmethod(original_config_from_pretrained)
            AutoTokenizer.from_pretrained = classmethod(original_tokenizer_from_pretrained)

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        env_vars = {
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
            # Allow vLLM to use max_model_len larger than the model's native max_position_embeddings.
            # Required when the model config has a smaller max_position_embeddings but we want to use
            # a larger context window (e.g., for models with RoPE scaling).
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        }
        # Pass WANDB_API_KEY if set, so that wandb logging works on TPU nodes
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            env_vars["WANDB_API_KEY"] = wandb_api_key
        return build_runtime_env_for_packages(
            extra=["eval"],
            pip_packages=["pandas", "langdetect", "immutabledict"],
            env_vars=env_vars,
        )

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
        Update max_position_embeddings and related fields in config.json and tokenizer_config.json to match max_model_len if needed.
        This ensures vLLM and lm-eval use the correct max length.
        """
        if not os.path.isdir(model_local_dir):
            return
        
        # Patch model config.json
        config_json_path = os.path.join(model_local_dir, "config.json")
        if os.path.exists(config_json_path):
            self._patch_config_file(config_json_path, max_model_len)
        
        # Also patch tokenizer_config.json if it exists
        tokenizer_config_json_path = os.path.join(model_local_dir, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_json_path):
            self._patch_config_file(tokenizer_config_json_path, max_model_len)
    
    def _patch_config_file(self, config_json_path: str, max_model_len: int) -> None:
        """Helper method to patch a config.json or tokenizer_config.json file."""
        with open(config_json_path, "r") as f:
            cfg = json.load(f)

        updated = False
        for key in ["max_position_embeddings", "n_positions", "max_seq_len", "model_max_length"]:
            if key not in cfg:
                if key == "max_position_embeddings" or key == "model_max_length":
                    cfg[key] = max_model_len
                    updated = True
            elif cfg[key] != max_model_len:
                cfg[key] = max_model_len
                updated = True
        
        if updated:
            with open(config_json_path, "w") as f:
                json.dump(cfg, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

    def _get_valid_vllm_llm_parameters(self) -> set[str]:
        """
        Dynamically get the valid parameters for vLLM's LLM class.
        Returns a set of parameter names that are accepted by LLM.__init__.
        """
        # Always include max_model_len as it's a critical parameter
        critical_params = {"max_model_len"}
        try:
            from vllm import LLM
            sig = inspect.signature(LLM.__init__)
            # Get all parameter names, excluding 'self'
            valid_params = set(sig.parameters.keys()) - {"self"}
            # Always include critical params even if inspection fails to find them
            valid_params.update(critical_params)
            return valid_params
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not inspect vLLM LLM parameters: {e}. Using fallback set.")
            # Fallback: return a set of known valid parameters
            # This is a conservative set based on common vLLM parameters
            return {
                "model", "tensor_parallel_size", "pipeline_parallel_size", "max_model_len",
                "max_num_seqs", "max_num_batched_tokens", "dtype", "device", "enforce_eager",
                "trust_remote_code", "download_dir", "distributed_executor_backend",
            }

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
        local_config_dir = None
        try:
            # NOTE(chris): This is not supported on TPUs
            # set_cuda_visible_devices()
            # Download the model from GCS or HuggingFace
            model_name_or_path, model = self.resolve_model_name_or_path(model)

            # If model path is a GCS path, download config files locally for lm-eval
            # lm-eval's vllm wrapper uses AutoConfig.from_pretrained() which doesn't support GCS paths
            # We download only config/tokenizer files and patch AutoConfig to use them
            # vLLM still streams the actual model weights directly from GCS via runai_streamer
            if model_name_or_path.startswith("gs://"):
                logger.info(f"Model path is GCS, downloading config files for lm-eval: {model_name_or_path}")
                local_config_dir = self._download_config_files_from_gcs(model_name_or_path)
                logger.info(f"Config files downloaded to: {local_config_dir}")

            # Build model args string first to get engine_kwargs
            engine_kwargs = dict(model.engine_kwargs) if model.engine_kwargs else {}
            
            # Extract max_gen_toks from engine_kwargs if present, or use the function parameter
            # max_gen_toks should be passed to lm-eval via model_args, but NOT to vLLM's engine initialization
            max_gen_toks_value = engine_kwargs.pop("max_gen_toks", None) or max_gen_toks
            
            # Patch config.json BEFORE vLLM initializes to ensure it reads the correct values
            # Use local config directory if available (for GCS paths), otherwise use model path
            config_dir_for_patching = local_config_dir if local_config_dir else model_name_or_path

            # Ensure model config is compatible with vLLM rope scaling on TPU
            self._patch_rope_scaling_config_if_needed(config_dir_for_patching)

            # Patch max_position_embeddings if max_model_len is specified
            # This must happen BEFORE vLLM and lm-eval initialize
            max_model_len_value = None
            if "max_model_len" in engine_kwargs and engine_kwargs["max_model_len"] is not None:
                max_model_len_value = engine_kwargs["max_model_len"]
                self._patch_max_position_embeddings_if_needed(config_dir_for_patching, max_model_len_value)
            
            # Get configurable values from engine_kwargs with defaults
            max_num_seqs = engine_kwargs.pop("max_num_seqs", 1)
            enforce_eager = engine_kwargs.pop("enforce_eager", False)

            # Extract engine-level seed from generation_params for vLLM initialization
            # This is different from per-request seed (which TPU/JAX doesn't support)
            # The engine seed initializes vLLM's global random state
            engine_seed = 0  # vLLM default
            if generation_params is not None and "seed" in generation_params:
                engine_seed = generation_params["seed"]
                logger.info(f"Using engine seed: {engine_seed}")

            pretrained_args_parts = [
                f"pretrained={model_name_or_path}",
                "device=tpu",
                f"enforce_eager={enforce_eager}",
                "dtype=bfloat16",
                f"max_num_seqs={max_num_seqs}",
                "pipeline_parallel_size=1",
                f"seed={engine_seed}",
            ]
            # Add distributed_executor_backend=ray for TPU tensor parallelism
            # This is required when using tensor_parallel_size > 1 on TPU
            if engine_kwargs.get("tensor_parallel_size", 1) > 1:
                pretrained_args_parts.append("distributed_executor_backend=ray")
            # Get valid vLLM LLM parameters to filter engine_kwargs
            valid_vllm_params = self._get_valid_vllm_llm_parameters()
            # Add vLLM engine arguments, only including keys that are valid vLLM parameters
            for key, value in engine_kwargs.items():
                # Skip None values to avoid passing "key=None" which might not be parsed correctly
                # Only include keys that are valid vLLM LLM parameters
                if value is not None and key in valid_vllm_params:
                    pretrained_args_parts.append(f"{key}={value}")
                elif value is not None and key not in valid_vllm_params:
                    logger.warning(f"Skipping invalid vLLM engine argument: {key}={value}")
            # Add max_gen_toks to model_args for lm-eval (the vLLM wrapper should extract it before initializing vLLM)
            if max_gen_toks_value is not None:
                pretrained_args_parts.append(f"max_gen_toks={max_gen_toks_value}")
                logger.info(f"Setting max_gen_toks={max_gen_toks_value} in model_args for lm-eval")
            pretrained_args = ",".join(pretrained_args_parts)
            logger.info(f"Final model_args string: {pretrained_args}")

            from lm_eval.evaluator import simple_evaluate
            from lm_eval.loggers import EvaluationTracker, WandbLogger
            from lm_eval.utils import simple_parse_args_string

            # Patch the version function in lm_eval.models.vllm_causallms to return a fake version for vllm.
            # The tpu-inference vllm doesn't have proper package metadata, which causes
            # lm-eval's version check to fail. We return a version that's high enough to pass basic checks
            # but not so high that it enables features requiring newer lm-eval functions.
            # Note: We must patch the function in the lm_eval module since it already imported it.
            import lm_eval.models.vllm_causallms as vllm_module
            original_version = vllm_module.version
            def patched_version(package_name):
                if package_name == "vllm":
                    return "0.8.2"  # Return a version just below 0.8.3 to avoid enabling problematic features
                return original_version(package_name)
            vllm_module.version = patched_version

            # Define the evaluation function that will run inside the context manager
            def run_evaluations():
                for eval_task in evals:
                    self._run_single_evaluation(
                        eval_task=eval_task,
                        model=model,
                        pretrained_args=pretrained_args,
                        max_eval_instances=max_eval_instances,
                        wandb_tags=wandb_tags,
                        generation_params=generation_params,
                    )

            try:
                # If we have a local config directory (GCS path), use the AutoConfig patch
                if local_config_dir:
                    with self._patch_autoconfig_for_gcs(model_name_or_path, local_config_dir):
                        run_evaluations()
                else:
                    run_evaluations()
            finally:
                # Restore the original version function
                vllm_module.version = original_version

        except Exception as e:
            logger.exception("lm-eval failed")
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

                # Clean up config cache directory
                if local_config_dir and os.path.exists(local_config_dir):
                    shutil.rmtree(local_config_dir, ignore_errors=True)

                print("Final cleanup completed successfully")
            except Exception as cleanup_error:
                print(f"Final cleanup failed: {cleanup_error}")
                # Don't raise here to avoid masking the original error

    def _run_single_evaluation(
        self,
        eval_task: EvalTaskConfig,
        model: ModelConfig,
        pretrained_args: str,
        max_eval_instances: int | None,
        wandb_tags: list[str] | None,
        generation_params: dict | None,
    ) -> None:
        """Run a single evaluation task."""
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.loggers import EvaluationTracker, WandbLogger
        from lm_eval.utils import simple_parse_args_string

        result_filepath = os.path.join(self.RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")

        # Create the output directory
        output_dir = os.path.dirname(result_filepath)
        os.makedirs(output_dir, exist_ok=True)

        evaluation_tracker_args = simple_parse_args_string(f",output_path={result_filepath}")
        evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

        # Build wandb run name: model_name-task-seedN
        wandb_run_name = model.name
        if eval_task and eval_task.name:
            wandb_run_name = f"{wandb_run_name}-{eval_task.name}"
        if generation_params and "seed" in generation_params:
            wandb_run_name = f"{wandb_run_name}-seed{generation_params['seed']}"

        wandb_args_dict = {
            "project": "marin",
            "job_type": "eval",
            "name": wandb_run_name,
            "tags": wandb_tags,
        }
        wandb_logger = WandbLogger(init_args=wandb_args_dict)

        # Process generation_params: pass as gen_kwargs to override task-level generation settings
        gen_kwargs = None
        random_seed = 0  # lm-eval default
        if generation_params is not None:
            gen_kwargs = dict(generation_params)  # Make a copy
            # Extract seed for lm-eval's random seed params
            random_seed = gen_kwargs.pop("seed", 0)
            # Remove 'n' as it's not a valid lm-eval gen_kwarg
            gen_kwargs.pop("n", None)
            logger.info(f"Using gen_kwargs: {gen_kwargs}, random_seed: {random_seed}")

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
            log_samples=True,
            gen_kwargs=gen_kwargs,
            random_seed=random_seed,
            numpy_random_seed=random_seed,
            torch_random_seed=random_seed,
        )

        if results is not None:
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
