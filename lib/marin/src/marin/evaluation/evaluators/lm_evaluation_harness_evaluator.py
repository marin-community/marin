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

import inspect
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
from fray.cluster.ray.deps import build_runtime_env_for_packages
from marin.evaluation.evaluators.debug_logging import (
    log_tokenizer_details,
    log_vllm_initialization,
    log_sample_generation,
)

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

    def _patch_tokenizer_loading(self, max_model_len: int) -> None:
        """
        Monkey patch tokenizer loading to override model_max_length before lm-eval's vLLM wrapper uses it.
        This intercepts AutoTokenizer.from_pretrained to set model_max_length after loading.
        Also patches PreTrainedTokenizerBase.__getattribute__ to intercept model_max_length access.
        """
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerBase
            
            # Patch __getattribute__ to intercept model_max_length access
            original_getattribute = PreTrainedTokenizerBase.__getattribute__
            max_model_len_to_use = max_model_len
            
            def patched_getattribute(self, name):
                if name == 'model_max_length':
                    # Always return our max_model_len value
                    return max_model_len_to_use
                return original_getattribute(self, name)
            
            PreTrainedTokenizerBase.__getattribute__ = patched_getattribute
            
            # Also patch from_pretrained to set the value directly
            original_from_pretrained = AutoTokenizer.from_pretrained.__func__ if hasattr(AutoTokenizer.from_pretrained, '__func__') else AutoTokenizer.from_pretrained
            
            def patched_from_pretrained(cls, *args, **kwargs):
                tokenizer = original_from_pretrained(cls, *args, **kwargs)
                # Set model_max_length directly
                object.__setattr__(tokenizer, 'model_max_length', max_model_len_to_use)
                return tokenizer
            
            AutoTokenizer.from_pretrained = classmethod(patched_from_pretrained)
        except Exception as e:
            logger.warning(f"Failed to patch tokenizer loading: {e}")

    def _patch_lm_eval_vllm_wrapper(self, max_model_len: int) -> None:
        """
        Monkey patch lm-eval's vLLM wrapper class to override how it determines max_length.
        This patches the wrapper's __init__ to ensure tokenizer and any max_length attributes use our value.
        """
        try:
            # Import lm-eval's vLLM wrapper module before it's used
            import lm_eval.models.vllm_causallms as vllm_module
            
            # Find the vLLM wrapper class
            vllm_wrapper_class = None
            for attr_name in dir(vllm_module):
                if not attr_name.startswith('_'):
                    attr = getattr(vllm_module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, '__init__'):
                        if (hasattr(attr, 'generate_until') or 
                            hasattr(attr, 'tok_encode') or
                            'VLLM' in attr_name):
                            vllm_wrapper_class = attr
                            break
            
            if vllm_wrapper_class is not None:
                original_init = vllm_wrapper_class.__init__
                max_model_len_to_use = max_model_len
                
                def patched_init(self, *args, **kwargs):
                    original_init(self, *args, **kwargs)
                    # Override tokenizer's model_max_length
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        if hasattr(self.tokenizer, 'model_max_length'):
                            self.tokenizer.model_max_length = max_model_len_to_use
                    # Override vLLM model's max_model_len if it exists
                    if hasattr(self, 'model') and self.model is not None:
                        try:
                            if hasattr(self.model, 'llm_engine') and hasattr(self.model.llm_engine, 'model_config'):
                                self.model.llm_engine.model_config.max_model_len = max_model_len_to_use
                        except Exception:
                            pass
                    # Store max_model_len as an instance attribute for property patching
                    self._marin_max_model_len = max_model_len_to_use
                
                vllm_wrapper_class.__init__ = patched_init
                
                # Patch properties/methods that might return max_length or EvalPos.size
                # Patch EvalPos property if it exists
                if hasattr(vllm_wrapper_class, 'EvalPos'):
                    original_eval_pos = vllm_wrapper_class.EvalPos
                    if isinstance(original_eval_pos, property):
                        def patched_eval_pos_getter(self):
                            original = original_eval_pos.fget(self) if original_eval_pos.fget else None
                            if hasattr(self, '_marin_max_model_len'):
                                # Try to create a replacement with the correct size
                                try:
                                    if hasattr(original, 'size'):
                                        class EvalPosReplacement:
                                            def __init__(self, original_obj, new_size):
                                                self._original = original_obj
                                                self.size = new_size
                                                # Copy other attributes if needed
                                                for attr in ['name']:
                                                    if hasattr(original_obj, attr):
                                                        setattr(self, attr, getattr(original_obj, attr))
                                        return EvalPosReplacement(original, self._marin_max_model_len)
                                except Exception:
                                    pass
                            return original
                        vllm_wrapper_class.EvalPos = property(patched_eval_pos_getter)
                
                # Patch max_length property if it exists
                if hasattr(vllm_wrapper_class, 'max_length'):
                    original_max_length = vllm_wrapper_class.max_length
                    if isinstance(original_max_length, property):
                        def patched_max_length_getter(self):
                            if hasattr(self, '_marin_max_model_len'):
                                return self._marin_max_model_len
                            if original_max_length.fget:
                                return original_max_length.fget(self)
                            return getattr(self, '_max_length', None)
                        vllm_wrapper_class.max_length = property(patched_max_length_getter)
                    else:
                        # If it's not a property, try to override it as an attribute
                        # We'll set it in __init__ which we already patched
                        pass
                
                # Patch generate_until method to use our max_length
                if hasattr(vllm_wrapper_class, 'generate_until'):
                    original_generate_until = vllm_wrapper_class.generate_until
                    max_model_len_to_use_for_gen = max_model_len
                    
                    def patched_generate_until(self, requests):
                        # Force override max_length/EvalPos before calling original
                        if hasattr(self, '_marin_max_model_len'):
                            max_len = self._marin_max_model_len
                        else:
                            max_len = max_model_len_to_use_for_gen
                            self._marin_max_model_len = max_len
                        
                        # Override tokenizer's model_max_length
                        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                            if hasattr(self.tokenizer, 'model_max_length'):
                                object.__setattr__(self.tokenizer, 'model_max_length', max_len)
                        
                        # Override EvalPos if it exists - this is what lm-eval uses for max_length
                        if hasattr(self, 'EvalPos'):
                            try:
                                # Create a replacement that returns our max_length
                                class EvalPosWrapper:
                                    def __init__(self, original, max_size):
                                        self._original = original
                                        self.size = max_size
                                        # Copy other attributes
                                        for attr in ['name']:
                                            if hasattr(original, attr):
                                                setattr(self, attr, getattr(original, attr))
                                    
                                    def __getattr__(self, name):
                                        return getattr(self._original, name)
                                
                                if hasattr(self.EvalPos, 'size') and self.EvalPos.size != max_len:
                                    self.EvalPos = EvalPosWrapper(self.EvalPos, max_len)
                            except Exception:
                                pass
                        
                        # Override max_length attribute
                        if hasattr(self, 'max_length'):
                            object.__setattr__(self, 'max_length', max_len)
                        
                        return original_generate_until(self, requests)
                    
                    vllm_wrapper_class.generate_until = patched_generate_until
        except Exception as e:
            logger.warning(f"Failed to patch lm-eval vLLM wrapper: {e}")

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
        try:
            # NOTE(chris): This is not supported on TPUs
            # set_cuda_visible_devices()
            # Download the model from GCS or HuggingFace
            model_name_or_path: str = self.download_model(model)

            # === DEBUG LOGGING: Check tokenizer ===
            try:
                from transformers import AutoTokenizer
                logger.info("Loading tokenizer for inspection...")
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                log_tokenizer_details(tokenizer, model.name)
            except Exception as e:
                logger.error(f"Failed to inspect tokenizer: {e}")

            # Build model args string first to get engine_kwargs
            engine_kwargs = dict(model.engine_kwargs) if model.engine_kwargs else {}
            
            # Extract max_gen_toks from engine_kwargs if present, or use the function parameter
            # max_gen_toks should be passed to lm-eval via model_args, but NOT to vLLM's engine initialization
            max_gen_toks_value = engine_kwargs.pop("max_gen_toks", None) or max_gen_toks
            
            # Patch config.json BEFORE vLLM initializes to ensure it reads the correct values
            # Ensure model config is compatible with vLLM rope scaling on TPU
            self._patch_rope_scaling_config_if_needed(model_name_or_path)
            
            # Patch max_position_embeddings if max_model_len is specified
            # This must happen BEFORE vLLM and lm-eval initialize
            max_model_len_value = None
            if "max_model_len" in engine_kwargs and engine_kwargs["max_model_len"] is not None:
                max_model_len_value = engine_kwargs["max_model_len"]
                self._patch_max_position_embeddings_if_needed(model_name_or_path, max_model_len_value)
            
            # Monkey patch tokenizer loading and lm-eval's vLLM wrapper BEFORE any lm_eval imports
            # This must happen before any lm_eval imports
            if max_model_len_value is not None:
                self._patch_tokenizer_loading(max_model_len_value)
                self._patch_lm_eval_vllm_wrapper(max_model_len_value)
            
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
            # === DEBUG LOGGING: Log vLLM initialization ===
            log_vllm_initialization(model_name_or_path, pretrained_args, engine_kwargs)

            # === MONKEY PATCH resolve_hf_chat_template ===
            # This MUST happen before any lm_eval imports that might load vllm_causallms.
            # We patch the module early to ensure resolve_hf_chat_template is available
            # when the vllm_causallms module code executes.
            try:
                # Try to import from vllm if available
                try:
                    from vllm.entrypoints.chat_utils import resolve_hf_chat_template as vllm_resolve_hf_chat_template
                except ImportError:
                    vllm_resolve_hf_chat_template = None
                
                # Define fallback implementation
                def fallback_resolve_hf_chat_template(tokenizer, *args, **kwargs):
                    """Fallback implementation if the function is missing."""
                    if args:
                        messages = args[0]
                    else:
                        messages = kwargs.get("messages")
                    if messages is None:
                        logger.warning("resolve_hf_chat_template called without messages; returning None")
                        return None
                    if hasattr(tokenizer, "apply_chat_template"):
                        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
                        return tokenizer.apply_chat_template(messages, **filtered_kwargs)
                    return None
                
                # Import and patch the module NOW, before simple_evaluate imports it
                import lm_eval.models.vllm_causallms as vllm_module
                
                # Determine which function to use
                if vllm_resolve_hf_chat_template is not None:
                    def wrapped_vllm_resolve(tokenizer, *args, **kwargs):
                        try:
                            if args:
                                return vllm_resolve_hf_chat_template(tokenizer, *args, **kwargs)
                            return vllm_resolve_hf_chat_template(tokenizer, **kwargs)
                        except TypeError as e:
                            if "unexpected keyword argument" in str(e):
                                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["tools"]}
                                if args:
                                    return vllm_resolve_hf_chat_template(tokenizer, *args, **filtered_kwargs)
                                return vllm_resolve_hf_chat_template(tokenizer, **filtered_kwargs)
                            raise
                    wrapped_vllm_resolve._marin_patched = True
                    resolve_fn = wrapped_vllm_resolve
                else:
                    resolve_fn = fallback_resolve_hf_chat_template
                    resolve_fn._marin_patched = True
                
                # Inject into module namespace - both as attribute and in __dict__
                vllm_module.resolve_hf_chat_template = resolve_fn
                vllm_module.__dict__["resolve_hf_chat_template"] = resolve_fn
                # Also set it in globals() if the module has that
                if hasattr(vllm_module, "__globals__"):
                    vllm_module.__globals__["resolve_hf_chat_template"] = resolve_fn
            except Exception as e:
                logger.warning(f"Failed to pre-patch resolve_hf_chat_template: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")

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

                # === MONKEY PATCH: Log sample prompts and responses ===
                try:
                    import lm_eval.models.vllm_causallms as vllm_module

                    # Find the vLLM wrapper class
                    vllm_wrapper_class = None
                    for attr_name in dir(vllm_module):
                        attr = getattr(vllm_module, attr_name)
                        if isinstance(attr, type) and hasattr(attr, 'generate_until'):
                            vllm_wrapper_class = attr
                            break

                    if vllm_wrapper_class:
                        original_generate_until = vllm_wrapper_class.generate_until
                        logged_samples = {"count": 0}  # Track how many we've logged

                        def patched_generate_until_with_logging(self, requests):
                            # Log first 3 samples
                            if logged_samples["count"] < 3:
                                for i, request in enumerate(requests[:3 - logged_samples["count"]]):
                                    try:
                                        # Extract context (the prompt)
                                        if hasattr(request, 'args') and len(request.args) >= 2:
                                            context = request.args[0]
                                            logger.info(f"\n{'='*80}")
                                            logger.info(f"SAMPLE REQUEST {logged_samples['count'] + 1}")
                                            logger.info(f"{'='*80}")
                                            logger.info(f"Context type: {type(context)}")
                                            logger.info(f"Context: {context[:1000]}")  # First 1000 chars
                                            if len(str(context)) > 1000:
                                                logger.info(f"... (truncated, full length: {len(str(context))})")
                                            logger.info(f"{'='*80}\n")
                                    except Exception as e:
                                        logger.debug(f"Could not log request: {e}")

                            # Call original method
                            results = original_generate_until(self, requests)

                            # Log first 3 responses
                            if logged_samples["count"] < 3 and results:
                                for i, result in enumerate(results[:3 - logged_samples["count"]]):
                                    try:
                                        logger.info(f"\n{'='*80}")
                                        logger.info(f"SAMPLE RESPONSE {logged_samples['count'] + 1}")
                                        logger.info(f"{'='*80}")
                                        logger.info(f"Result type: {type(result)}")
                                        logger.info(f"Result: {str(result)[:2000]}")  # First 2000 chars
                                        if len(str(result)) > 2000:
                                            logger.info(f"... (truncated, full length: {len(str(result))})")
                                        logger.info(f"{'='*80}\n")
                                        logged_samples["count"] += 1
                                    except Exception as e:
                                        logger.debug(f"Could not log result: {e}")

                            return results

                        vllm_wrapper_class.generate_until = patched_generate_until_with_logging
                        logger.info("Successfully patched generate_until for logging")
                except Exception as e:
                    logger.warning(f"Failed to patch generate_until for logging: {e}")

                # Note: max_gen_toks is passed to lm-eval via model_args.
                # The vLLM wrapper in lm-eval should extract it from model_args before initializing vLLM.
                # If max_gen_toks is passed to vLLM's engine initialization, it will cause an error.

                # Process generation_params: pass as gen_kwargs to override task-level generation settings
                gen_kwargs = None
                random_seed = 0  # lm-eval default
                if generation_params is not None:
                    gen_kwargs = dict(generation_params)  # Make a copy
                    # Extract seed for lm-eval's random seed params
                    random_seed = gen_kwargs.pop("seed", 0)
                    # Remove 'n' as it's not a valid lm-eval gen_kwarg
                    gen_kwargs.pop("n", None)
                    # Note: We cannot pass 'seed' to vLLM on TPU/JAX - it raises
                    # "ValueError: JAX does not support per-request seed."
                    # The random_seed is still used for lm-eval's random state.
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
                    log_samples=True,  # This controls whether to log samples to the results file
                    gen_kwargs=gen_kwargs,  # Override task-level generation settings
                    random_seed=random_seed,
                    numpy_random_seed=random_seed,
                    torch_random_seed=random_seed,
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

                print("Final cleanup completed successfully")
            except Exception as cleanup_error:
                print(f"Final cleanup failed: {cleanup_error}")
                # Don't raise here to avoid masking the original error
