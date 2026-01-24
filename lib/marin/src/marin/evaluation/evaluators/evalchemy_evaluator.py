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

"""
Evalchemy evaluator for reasoning benchmarks.

Evalchemy (https://github.com/mlfoundations/evalchemy) builds on top of
lm-evaluation-harness and adds specialized reasoning tasks including:
- Math: AIME24, AIME25, AMC23, MATH500
- Code: HumanEval+, MBPP+, LiveCodeBench, BigCodeBench
- Science & Reasoning: GPQADiamond, Alice in Wonderland

This evaluator handles several compatibility issues:
1. lm-eval version differences (eval_logger location)
2. vllm-tpu lacking package metadata
3. TPU/JAX not supporting per-request seeds
4. GCS model paths not supported by transformers AutoConfig
"""

import glob
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from typing import Sequence

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator, remove_tpu_lockfile_on_exit
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from fray.cluster import ResourceConfig, current_cluster
from fray.cluster.base import JobRequest, Entrypoint, EnvironmentConfig
from fray.cluster.ray.deps import build_runtime_env_for_packages

logger = logging.getLogger(__name__)

# Evalchemy git repo and commit to use
EVALCHEMY_REPO = "https://github.com/mlfoundations/evalchemy.git"
EVALCHEMY_COMMIT = "6ed674159b37f740f2353a86f596f49f6ac13c19"  # 2025-01-08

# Wandb project name for evalchemy evaluations
# Note: Also defined in experiments/evals/evals.py. Kept separate to avoid
# library code (lib/marin) depending on experiments code (experiments/evals).
WANDB_PROJECT = "marin"


class EvalchemyEvaluator(VllmTpuEvaluator):
    """
    Evaluator that runs Evalchemy reasoning benchmarks on TPU via vLLM.

    Generation parameters can be passed via model.generation_params:
    - temperature: Sampling temperature (default 0)
    - max_gen_toks: Maximum generation tokens (e.g., 32768)
    - seed: Engine-level seed for vLLM (enables reproducible sampling with temp > 0)

    Note: TPU/JAX doesn't support per-request seeds in vLLM's sampling. To enable
    non-zero temperature with reproducibility, we use engine-level seed (passed to
    vLLM at initialization via --model_args seed=N) rather than per-request seeds.

    Note: Evalchemy is cloned at runtime because it has local file dependencies
    that prevent it from being installed via pip from git.
    """

    CACHE_PATH: str = "/tmp/evalchemy"
    EVALCHEMY_PATH: str = os.path.join(CACHE_PATH, "evalchemy_repo")
    RESULTS_PATH: str = os.path.join(CACHE_PATH, "evalchemy_results")
    CONFIG_CACHE_PATH: str = os.path.join(CACHE_PATH, "config_cache")

    # Config files needed for lm-eval (AutoConfig, tokenizer) but NOT model weights
    CONFIG_FILES: list[str] = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "generation_config.json",
        "added_tokens.json",
        "chat_template.jinja",
    ]

    def get_runtime_env(self) -> dict:
        """Returns the runtime environment for the Ray cluster."""
        env_vars = {"HF_ALLOW_CODE_EVAL": "1"}
        # Pass WANDB_API_KEY if set, so that wandb logging works on TPU nodes
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            env_vars["WANDB_API_KEY"] = wandb_api_key
        return build_runtime_env_for_packages(
            extra=["evalchemy"],
            env_vars=env_vars,
        )

    def _get_subprocess_env(self) -> dict:
        """Build environment for evalchemy subprocess.

        Includes vLLM env vars and propagates WANDB_API_KEY for post-evaluation
        wandb logging (done in _log_results_to_wandb after subprocess completes).
        """
        env = self._vllm_env()
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key and "WANDB_API_KEY" not in env:
            env["WANDB_API_KEY"] = wandb_api_key
        # Ensure Python output is unbuffered for real-time progress visibility
        env["PYTHONUNBUFFERED"] = "1"
        # Allow extending max_model_len beyond model's max_position_embeddings
        # This is needed when max_gen_toks + context_buffer > max_position_embeddings
        # Safe for RoPE models (like Qwen) which use relative position encoding
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        return env

    def _log_results_to_wandb(
        self,
        result_dir: str,
        run_name: str,
        model_name: str,
        task_name: str,
        engine_seed: int = 0,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """Log evaluation results to wandb after evalchemy completes.

        Reads results from evalchemy output files and logs metrics to wandb.
        This is done manually because the WandbLogger in lm-eval requires
        init_args/config_args format which is incompatible with CLI parsing.
        """
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")
            return

        if not os.environ.get("WANDB_API_KEY"):
            logger.info("WANDB_API_KEY not set, skipping wandb logging")
            return

        wandb_initialized = False
        try:
            # Find results.json file (evalchemy saves results as results.json)
            results_pattern = os.path.join(result_dir, "*", "results.json")
            results_files = glob.glob(results_pattern)

            if not results_files:
                # Try directly in result_dir
                results_pattern = os.path.join(result_dir, "results.json")
                results_files = glob.glob(results_pattern)

            if not results_files:
                logger.warning(f"No results.json found in {result_dir}, skipping wandb logging")
                return

            # Read the first results file
            results_file = results_files[0]
            logger.info(f"Reading results from {results_file}")

            with open(results_file, "r") as f:
                results = json.load(f)

            # Build wandb tags
            tags = ["evalchemy", task_name.lower()[:64], model_name.lower()[:64]]
            if engine_seed != 0:
                tags.append(f"seed{engine_seed}")
            if wandb_tags:
                tags.extend([tag[:64] for tag in wandb_tags])

            # Initialize wandb run
            wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                job_type="eval",
                tags=tags,
                config={
                    "model_name": model_name,
                    "task_name": task_name,
                    "engine_seed": engine_seed,
                },
                reinit=True,
            )
            wandb_initialized = True

            # Log metrics from results
            # Evalchemy results structure: {"results": {task_name: {metric: value}}}
            if "results" in results:
                for task, metrics in results["results"].items():
                    if isinstance(metrics, dict):
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                # Log with task prefix for clarity
                                wandb.log({f"{task}/{metric_name}": metric_value})

            # Also log raw results summary
            if "results" in results:
                wandb.log({"results_summary": results["results"]})

            wandb.finish()
            logger.info(f"Logged results to wandb run: {run_name}")

        except Exception as e:
            logger.warning(f"Failed to log results to wandb: {e}")
            # Only try to finish wandb if it was successfully initialized
            if wandb_initialized:
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    pass

    def _setup_evalchemy(self) -> str:
        """Clone evalchemy and apply necessary patches. Returns path to repo."""
        self._log_lmeval_version()
        self._clone_evalchemy()
        self._apply_patches()
        return self.EVALCHEMY_PATH

    def _log_lmeval_version(self) -> None:
        """Log lm-eval version for debugging."""
        try:
            import lm_eval
            logger.info(f"lm-eval version: {getattr(lm_eval, '__version__', 'unknown')}")
            logger.info(f"lm-eval location: {lm_eval.__file__}")
        except ImportError:
            logger.warning("lm-eval not found")

    def _clone_evalchemy(self) -> None:
        """Clone fresh copy of evalchemy repo."""
        if os.path.exists(self.EVALCHEMY_PATH):
            logger.info(f"Removing existing evalchemy repo at {self.EVALCHEMY_PATH}")
            shutil.rmtree(self.EVALCHEMY_PATH)

        os.makedirs(self.CACHE_PATH, exist_ok=True)

        logger.info(f"Cloning evalchemy from {EVALCHEMY_REPO} at commit {EVALCHEMY_COMMIT}")
        subprocess.run(
            ["git", "clone", EVALCHEMY_REPO, self.EVALCHEMY_PATH],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "checkout", EVALCHEMY_COMMIT],
            cwd=self.EVALCHEMY_PATH,
            check=True, capture_output=True, text=True,
        )
        logger.info(f"Evalchemy cloned successfully to {self.EVALCHEMY_PATH}")

    def _apply_patches(self) -> None:
        """Apply all necessary patches to evalchemy for TPU compatibility."""
        self._patch_eval_tracker_imports()
        self._patch_eval_py_imports()
        self._patch_vllm_version()
        self._patch_vllm_seed_for_tpu()

    def _patch_eval_tracker_imports(self) -> None:
        """Patch eval_tracker.py to handle different lm-eval versions."""
        path = os.path.join(self.EVALCHEMY_PATH, "eval", "eval_tracker.py")
        if not os.path.exists(path):
            return

        content = self._read_file(path)
        if "lm_eval.logging_utils" in content:
            return  # Already patched

        old = "from lm_eval.utils import eval_logger, handle_non_serializable, hash_string, simple_parse_args_string"
        new = """try:
    from lm_eval.utils import eval_logger, handle_non_serializable, hash_string, simple_parse_args_string
except ImportError:
    try:
        from lm_eval.logging_utils import eval_logger
        from lm_eval.utils import handle_non_serializable, hash_string, simple_parse_args_string
    except ImportError:
        import logging
        eval_logger = logging.getLogger("lm-eval")
        from lm_eval.utils import handle_non_serializable, hash_string, simple_parse_args_string"""

        if old in content:
            self._write_file(path, content.replace(old, new))
            logger.info("Patched eval_tracker.py to handle different lm-eval versions")
        else:
            logger.warning("Could not find expected import in eval_tracker.py - evalchemy may have changed")

    def _patch_eval_py_imports(self) -> None:
        """Patch eval.py to add eval_logger to utils module if missing."""
        path = os.path.join(self.EVALCHEMY_PATH, "eval", "eval.py")
        if not os.path.exists(path):
            return

        content = self._read_file(path)
        if "# Patch utils.eval_logger" in content:
            return  # Already patched

        patch = '''
# Patch utils.eval_logger for lm-eval compatibility
if not hasattr(utils, 'eval_logger'):
    try:
        from lm_eval.logging_utils import eval_logger as _eval_logger
        utils.eval_logger = _eval_logger
    except ImportError:
        import logging as _logging
        utils.eval_logger = _logging.getLogger("lm-eval")
'''
        marker = "from lm_eval.tasks import TaskManager as PretrainTaskManager"
        if marker in content:
            self._write_file(path, content.replace(marker, marker + patch))
            logger.info("Patched eval.py to handle different lm-eval versions")
        else:
            logger.warning("Could not find TaskManager import in eval.py - evalchemy may have changed")

    def _patch_vllm_version(self) -> None:
        """Patch eval.py to handle vllm-tpu lacking package metadata."""
        path = os.path.join(self.EVALCHEMY_PATH, "eval", "eval.py")
        if not os.path.exists(path):
            return

        content = self._read_file(path)
        if "# Patch vllm version" in content:
            return  # Already patched

        # Version 0.8.2 is returned because lm-eval checks vllm version for feature compatibility.
        # vllm-tpu doesn't have package metadata, so we return a version that lm-eval accepts.
        patch = '''
# Patch vllm version for vllm-tpu (lacks package metadata)
def _patch_vllm_version():
    try:
        import lm_eval.models.vllm_causallms as vllm_module
        if hasattr(vllm_module, 'version'):
            _original = vllm_module.version
            def _patched(pkg):
                # Return 0.8.2 for vllm - a version lm-eval accepts
                return "0.8.2" if pkg == "vllm" else _original(pkg)
            vllm_module.version = _patched
    except Exception:
        pass
_patch_vllm_version()
'''
        marker = "from lm_eval.utils import sanitize_model_name, simple_parse_args_string"
        if marker in content:
            self._write_file(path, content.replace(marker, marker + "\n" + patch))
            logger.info("Patched eval.py to handle vllm-tpu version")
        else:
            logger.warning("Could not find sanitize_model_name import in eval.py - evalchemy may have changed")

    def _patch_vllm_seed_for_tpu(self) -> None:
        """
        Patch lm-eval's vLLM wrapper to disable per-request seeds for TPU.

        TPU/JAX doesn't support per-request seeds, causing errors. This patch
        intercepts SamplingParams and sets seed=None.
        """
        path = os.path.join(self.EVALCHEMY_PATH, "eval", "eval.py")
        if not os.path.exists(path):
            return

        content = self._read_file(path)
        if "# Patch lm-eval vLLM seed" in content:
            return  # Already patched

        # This patch modifies lm-eval's VLLM._model_generate to strip seeds from SamplingParams
        patch = '''
# Patch lm-eval vLLM seed handling for TPU (JAX doesn't support per-request seeds)
def _patch_lmeval_vllm_seed():
    try:
        import lm_eval.models.vllm_causallms as vllm_module
        from vllm import SamplingParams

        if not hasattr(vllm_module, 'VLLM'):
            return

        VLLM = vllm_module.VLLM
        if hasattr(VLLM, '_tpu_seed_patched'):
            return

        _original = VLLM._model_generate

        def _strip_seed(sp):
            """Create new SamplingParams with seed=None, copying only supported attrs."""
            if sp.seed is None:
                return sp
            # Use only the core parameters that vllm-tpu supports.
            # Defaults match vLLM's SamplingParams defaults:
            # - n=1: single sample per request
            # - temperature=1.0: standard sampling temperature
            # - top_p=1.0: no nucleus sampling restriction
            # - top_k=-1: disabled (vLLM convention)
            # - max_tokens=16: vLLM default, typically overridden by caller
            kwargs = {
                'n': getattr(sp, 'n', 1),
                'temperature': getattr(sp, 'temperature', 1.0),
                'top_p': getattr(sp, 'top_p', 1.0),
                'top_k': getattr(sp, 'top_k', -1),
                'max_tokens': getattr(sp, 'max_tokens', 16),
                'seed': None,
            }
            # Add optional params if they exist on this vLLM version
            for attr in ['stop', 'stop_token_ids', 'ignore_eos', 'logprobs', 'skip_special_tokens']:
                if hasattr(sp, attr):
                    kwargs[attr] = getattr(sp, attr)
            return SamplingParams(**kwargs)

        def _patched(self, requests=None, generate=False, **kwargs):
            # Handle sampling_params kwarg
            sp = kwargs.pop('sampling_params', None)
            if sp is not None:
                if isinstance(sp, list):
                    kwargs['sampling_params'] = [_strip_seed(s) for s in sp]
                else:
                    kwargs['sampling_params'] = _strip_seed(sp)
                return _original(self, requests, generate, **kwargs)

            # Handle old-style requests list
            if requests is not None:
                patched = []
                for req in requests:
                    ctx, sp, *rest = req
                    patched.append((ctx, _strip_seed(sp), *rest))
                return _original(self, patched, generate, **kwargs)

            return _original(self, requests, generate, **kwargs)

        VLLM._model_generate = _patched
        VLLM._tpu_seed_patched = True
        print("Patched lm-eval VLLM to disable per-request seeds for TPU")
    except Exception as e:
        print(f"Warning: Could not patch lm-eval vllm seed: {e}")

_patch_lmeval_vllm_seed()
'''
        marker = "_patch_vllm_version()\n"
        if marker in content:
            self._write_file(path, content.replace(marker, marker + patch, 1))
            logger.info("Patched eval.py to disable per-request seeds for TPU")
        else:
            logger.warning("Could not find _patch_vllm_version() call in eval.py - evalchemy may have changed")

    def _download_config_files_from_gcs(self, gcs_path: str) -> str:
        """
        Download config/tokenizer files from GCS for lm-eval's AutoConfig.

        vLLM streams model weights directly from GCS, but transformers AutoConfig
        doesn't support GCS paths. We download only the config files locally.
        """
        try:
            import fsspec
        except ImportError as e:
            raise ImportError(
                "fsspec is required for GCS model paths. "
                "Install with: pip install fsspec gcsfs"
            ) from e

        path_hash = hashlib.md5(gcs_path.encode()).hexdigest()[:8]
        local_dir = os.path.join(self.CONFIG_CACHE_PATH, f"config_{path_hash}")
        os.makedirs(local_dir, exist_ok=True)

        fs = fsspec.filesystem("gcs")
        gcs_path_clean = gcs_path.rstrip("/")

        for filename in self.CONFIG_FILES:
            remote = f"{gcs_path_clean}/{filename}"
            local = os.path.join(local_dir, filename)
            try:
                if fs.exists(remote):
                    fs.get(remote, local)
                    logger.info(f"Downloaded {filename} from GCS to {local}")
            except Exception as e:
                logger.debug(f"Could not download {filename}: {e}")

        return local_dir

    def _patch_eval_py_for_gcs(self, gcs_path: str, local_config_dir: str) -> None:
        """Patch eval.py to redirect AutoConfig/AutoTokenizer from GCS to local."""
        path = os.path.join(self.EVALCHEMY_PATH, "eval", "eval.py")
        if not os.path.exists(path):
            return

        content = self._read_file(path)
        if "# Patch AutoConfig for GCS" in content:
            return  # Already patched

        patch = f'''
# Patch AutoConfig for GCS paths
def _patch_autoconfig_for_gcs():
    from transformers import AutoConfig, AutoTokenizer
    _gcs = "{gcs_path}".rstrip("/")
    _local = "{local_config_dir}"

    _orig_config = AutoConfig.from_pretrained.__func__
    _orig_tokenizer = AutoTokenizer.from_pretrained.__func__

    def _normalize(path):
        """Normalize path for comparison (handle trailing slashes)."""
        if isinstance(path, str):
            return path.rstrip("/")
        return path

    def _config(cls, path, *a, **kw):
        if _normalize(path) == _gcs:
            print(f"Redirecting AutoConfig from {{path}} to {{_local}}")
            return _orig_config(cls, _local, *a, **kw)
        return _orig_config(cls, path, *a, **kw)

    def _tokenizer(cls, path, *a, **kw):
        if _normalize(path) == _gcs:
            print(f"Redirecting AutoTokenizer from {{path}} to {{_local}}")
            return _orig_tokenizer(cls, _local, *a, **kw)
        return _orig_tokenizer(cls, path, *a, **kw)

    AutoConfig.from_pretrained = classmethod(_config)
    AutoTokenizer.from_pretrained = classmethod(_tokenizer)
    print(f"GCS patch installed: will redirect {{_gcs}} to {{_local}}")

_patch_autoconfig_for_gcs()
'''
        marker = 'utils.eval_logger = _logging.getLogger("lm-eval")'
        if marker in content:
            self._write_file(path, content.replace(marker, marker + "\n" + patch))
            logger.info("Patched eval.py to handle GCS model paths")
        else:
            logger.warning("Could not find eval_logger marker in eval.py for GCS patch - evalchemy may have changed")

    def _run_with_streaming(
        self,
        cmd: list[str],
        env: dict,
        cwd: str,
        log_file: str,
    ) -> int:
        """Run a subprocess with real-time output streaming.

        Streams stdout/stderr to the console in real-time for progress visibility,
        while also saving all output to a log file for error reporting.

        Note: We read stdout line-by-line in a loop, which prevents buffer deadlocks
        as long as the subprocess produces line-based output (which lm-eval does).

        Args:
            cmd: Command to run as list of strings
            env: Environment variables dict
            cwd: Working directory
            log_file: Path to save combined output

        Returns:
            Process return code
        """
        with open(log_file, "w") as lf:
            # Use Popen to get real-time output
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
            )

            # Stream output line by line
            try:
                for line in process.stdout:
                    # Write to log file
                    lf.write(line)
                    lf.flush()
                    # Print to console (will appear in Ray logs)
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except Exception as e:
                logger.warning(f"Error streaming output: {e}")

            # Wait for process to complete
            process.wait()
            return process.returncode

    def _get_max_model_len_from_config(self, config_dir: str) -> int | None:
        """Read max_position_embeddings from config.json for vLLM max_model_len."""
        config_path = os.path.join(config_dir, "config.json")
        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Try common config keys for max context length
            for key in ["max_position_embeddings", "n_positions", "max_seq_len", "seq_length"]:
                if key in config:
                    return config[key]
            return None
        except Exception as e:
            logger.warning(f"Could not read max_model_len from config: {e}")
            return None

    def _read_file(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def _write_file(self, path: str, content: str) -> None:
        with open(path, "w") as f:
            f.write(content)

    def evaluate(
        self,
        model: ModelConfig,
        evals: Sequence[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Run Evalchemy evaluations on the specified model and tasks.

        Args:
            model: Model configuration (generation_params can include temperature, max_gen_toks, seed)
            evals: List of evaluation tasks to run
            output_path: Path to save results (local or GCS)
            max_eval_instances: Maximum instances per task (for debugging)
            wandb_tags: Tags to add to wandb runs
        """
        local_config_dir = None
        try:
            evalchemy_path = self._setup_evalchemy()
            model_name_or_path, model = self.resolve_model_name_or_path(model)

            # Handle GCS model paths - download config files for lm-eval
            if model_name_or_path.startswith("gs://"):
                logger.info(f"Downloading config files for GCS model: {model_name_or_path}")
                local_config_dir = self._download_config_files_from_gcs(model_name_or_path)
                self._patch_eval_py_for_gcs(model_name_or_path, local_config_dir)

            os.makedirs(self.RESULTS_PATH, exist_ok=True)

            # Extract generation parameters
            gen_params = model.generation_params or {}
            temperature = gen_params.get("temperature", 0)  # Default to greedy
            max_gen_toks = gen_params.get("max_gen_toks")
            top_p = gen_params.get("top_p")
            engine_seed = gen_params.get("seed", 0)  # Engine-level seed for vLLM

            if engine_seed != 0:
                logger.info(f"Using engine seed: {engine_seed}")

            # Get max_model_len from config (needed for GCS paths where vLLM can't read config)
            # Then extend it if needed to accommodate max_gen_toks + context
            max_model_len = None
            if local_config_dir:
                max_model_len = self._get_max_model_len_from_config(local_config_dir)
                if max_model_len:
                    logger.info(f"Model's native max_position_embeddings: {max_model_len}")

            # Extend max_model_len if needed to accommodate generation + context
            # lm-eval computes: max_ctx_len = max_model_len - max_gen_toks
            # If max_model_len == max_gen_toks, then max_ctx_len = 0 and all context is truncated!
            # Solution: extend max_model_len to leave room for context
            #
            # Context buffer sizing:
            # - Math tasks (AIME, MATH500): ~200-500 tokens - 2048 is plenty
            # - Code tasks (HumanEval+, MBPP+): ~300-800 tokens - 2048 is fine
            # - Competitive programming (LiveCodeBench, CodeForces): ~500-3000 tokens - need 4096
            # - SWEbench: ~2000-8000+ tokens - may need even more
            # Default to 4096 to cover most cases safely
            if max_gen_toks:
                context_buffer = gen_params.get("context_buffer", 4096)
                required_max_model_len = max_gen_toks + context_buffer
                if max_model_len is None or required_max_model_len > max_model_len:
                    logger.info(
                        f"Extending max_model_len to {required_max_model_len} "
                        f"(max_gen_toks={max_gen_toks} + context_buffer={context_buffer})"
                    )
                    max_model_len = required_max_model_len

            for eval_task in evals:
                result_dir = os.path.join(
                    self.RESULTS_PATH,
                    f"{eval_task.name}_{eval_task.num_fewshot}shot"
                )
                os.makedirs(result_dir, exist_ok=True)

                # Build model_args for vLLM initialization
                # - batch_size=auto: Enable continuous batching for parallel inference
                # - max_model_len: Sets both vLLM's max model length AND lm-eval's _max_length
                #   (lm-eval line 161: self._max_length = max_model_len if max_model_len is not None else max_length)
                # - max_gen_toks: Maximum generation tokens (lm-eval default is only 256!)
                # - seed: Engine-level seed for reproducible sampling with temperature > 0
                model_args_parts = [
                    f"pretrained={model_name_or_path}",
                    "batch_size=auto",  # Enable batched inference (default is 1!)
                ]
                if engine_seed != 0:
                    model_args_parts.append(f"seed={engine_seed}")
                if max_model_len:
                    model_args_parts.append(f"max_model_len={max_model_len}")
                if max_gen_toks:
                    # Set at model level so lm-eval's default (256) is overridden
                    model_args_parts.append(f"max_gen_toks={max_gen_toks}")

                # Add engine_kwargs (e.g., tensor_parallel_size=4 for multi-chip TPU)
                if model.engine_kwargs:
                    for key, value in model.engine_kwargs.items():
                        model_args_parts.append(f"{key}={value}")

                model_args = ",".join(model_args_parts)
                logger.info(f"model_args: {model_args}")

                # Build wandb run name: evalchemy-{model_name}-{task_name}-seed{N} (lowercase)
                wandb_run_name = f"evalchemy-{model.name.lower()}"
                if eval_task.name:
                    wandb_run_name = f"{wandb_run_name}-{eval_task.name.lower()}"
                if engine_seed != 0:
                    wandb_run_name = f"{wandb_run_name}-seed{engine_seed}"

                # Build evalchemy CLI command
                cmd = [
                    "python", "-m", "eval.eval",
                    "--model", "vllm",
                    "--tasks", eval_task.name,
                    "--model_args", model_args,
                    "--output_path", result_dir,
                    "--verbosity", "INFO",
                ]

                if eval_task.num_fewshot > 0:
                    cmd.extend(["--num_fewshot", str(eval_task.num_fewshot)])

                if max_eval_instances is not None:
                    cmd.extend(["--limit", str(max_eval_instances)])

                if model.apply_chat_template:
                    cmd.append("--apply_chat_template")

                # Add generation kwargs (temperature, max_gen_toks, top_p)
                gen_kwargs = []
                if temperature is not None:
                    gen_kwargs.append(f"temperature={temperature}")
                if max_gen_toks is not None:
                    gen_kwargs.append(f"max_gen_toks={max_gen_toks}")
                if top_p is not None:
                    gen_kwargs.append(f"top_p={top_p}")
                if gen_kwargs:
                    cmd.extend(["--gen_kwargs", ",".join(gen_kwargs)])

                logger.info(f"Running: {' '.join(cmd)}")

                # Stream output in real-time while also capturing for error reporting
                log_file = os.path.join(result_dir, "evalchemy_output.log")
                returncode = self._run_with_streaming(
                    cmd=cmd,
                    env=self._get_subprocess_env(),
                    cwd=evalchemy_path,
                    log_file=log_file,
                )

                if returncode != 0:
                    error_file = os.path.join(result_dir, "evalchemy_error.txt")
                    with open(error_file, "w") as f:
                        f.write(f"Command: {' '.join(cmd)}\n")
                        f.write(f"Return code: {returncode}\n")
                        f.write(f"\n=== OUTPUT LOG ===\n")
                        if os.path.exists(log_file):
                            with open(log_file, "r") as lf:
                                f.write(lf.read())
                    raise RuntimeError(f"Evalchemy failed for {eval_task.name}, see {error_file}")

                logger.info(f"Completed {eval_task.name}")

                # Log results to wandb
                self._log_results_to_wandb(
                    result_dir=result_dir,
                    run_name=wandb_run_name,
                    model_name=model.name,
                    task_name=eval_task.name,
                    engine_seed=engine_seed,
                    wandb_tags=wandb_tags,
                )

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Evalchemy evaluation failed: {e}") from e

        finally:
            if is_remote_path(output_path):
                try:
                    logger.info("Uploading results to GCS...")
                    upload_to_gcs(self.RESULTS_PATH, output_path)
                except Exception as e:
                    logger.error(f"Failed to upload to GCS: {e}")

            self.cleanup(model)
            if os.path.exists(self.RESULTS_PATH):
                shutil.rmtree(self.RESULTS_PATH)
            if local_config_dir and os.path.exists(local_config_dir):
                shutil.rmtree(local_config_dir, ignore_errors=True)

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: Sequence[EvalTaskConfig],
        output_path: str,
        resource_config: ResourceConfig,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """Launch evaluation on Ray cluster with TPU resources."""

        def _run():
            with remove_tpu_lockfile_on_exit():
                import logging
                logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
                self.evaluate(model, evals, output_path, max_eval_instances, wandb_tags)

        job_request = JobRequest(
            name="evalchemy-tpu-evaluation",
            entrypoint=Entrypoint.from_callable(_run),
            resources=resource_config or ResourceConfig(),
            environment=EnvironmentConfig.create(extras=["evalchemy", "tpu", "vllm"]),
        )

        cluster = current_cluster()
        job_id = cluster.launch(job_request)
        cluster.wait(job_id, raise_on_failure=True)
