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

import gc
import glob
import hashlib
import io
import json
import logging
import os
import re
import runpy
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
# Reads from WANDB_PROJECT env var, defaults to "marin"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "marin")

# Evalchemy benchmarks that have hardcoded n_repeat values and their paths.
# These benchmarks run multiple repetitions with different seeds to compute
# averaged accuracy, but this significantly increases evaluation time. 
# For example, AIME25 defaults to n_repeat=10 
# (https://github.com/mlfoundations/evalchemy/blob/main/eval/chat_benchmarks/AIME25/eval_instruct.py)
N_REPEAT_BENCHMARK_PATHS = {
    "AIME25": "eval/chat_benchmarks/AIME25/eval_instruct.py",
    "AIME24": "eval/chat_benchmarks/AIME24/eval_instruct.py",
    "AMC23": "eval/chat_benchmarks/AMC23/eval_instruct.py",
    "HMMT": "eval/chat_benchmarks/HMMT/eval_instruct.py",
}


def _extract_step_suffix(path: str) -> str:
    """Extract step number from a model path and return as a suffix string.

    E.g., "gs://bucket/checkpoints/run/hf/step-7022/" -> "-step7022"
    E.g., "Qwen/Qwen2.5-7B-Instruct" -> ""
    """
    match = re.search(r'step-(\d+)', path)
    return f"-step{match.group(1)}" if match else ""


class _TeeWriter(io.TextIOBase):
    """Writes to both a log file and original stdout simultaneously."""

    def __init__(self, log_file, original_stdout):
        self._log_file = log_file
        self._original_stdout = original_stdout

    def write(self, s):
        if s:
            self._log_file.write(s)
            self._log_file.flush()
            self._original_stdout.write(s)
            self._original_stdout.flush()
        return len(s) if s else 0

    def flush(self):
        self._log_file.flush()
        self._original_stdout.flush()

    def fileno(self):
        return self._original_stdout.fileno()

    def isatty(self):
        return False


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
        # Pass wandb env vars so that logging works on TPU nodes
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            env_vars["WANDB_API_KEY"] = wandb_api_key
        wandb_entity = os.environ.get("WANDB_ENTITY")
        if wandb_entity:
            env_vars["WANDB_ENTITY"] = wandb_entity
        return build_runtime_env_for_packages(
            extra=["evalchemy"],
            env_vars=env_vars,
        )

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
            # Find results file. Evalchemy saves results as
            # <output_path>/<model_name_sanitized>/results_<ISO_TIMESTAMP>.json
            # e.g., result_dir/vllm/results_2024-01-22T18-04-30.123456.json
            results_pattern = os.path.join(result_dir, "*", "results_*.json")
            results_files = sorted(glob.glob(results_pattern))

            if not results_files:
                # Try directly in result_dir
                results_pattern = os.path.join(result_dir, "results_*.json")
                results_files = sorted(glob.glob(results_pattern))

            if not results_files:
                # Also try the exact name results.json as fallback
                results_pattern = os.path.join(result_dir, "*", "results.json")
                results_files = glob.glob(results_pattern)

            if not results_files:
                logger.warning(f"No results file found in {result_dir}, skipping wandb logging")
                return

            # Read the latest results file (sorted alphabetically, timestamp in name)
            results_file = results_files[-1]
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
            wandb_entity = os.environ.get("WANDB_ENTITY", "marin-community")
            wandb.init(
                project=WANDB_PROJECT,
                entity=wandb_entity,
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
        self._patch_vllm_stat_logging()

    def _patch_benchmark_n_repeat(self, task_name: str, n_repeat: int) -> None:
        """
        Override by patching an evalchemy benchmark's n_repeat value.
        Note: not all tasks support this.

        Args:
            task_name: Name of the task (e.g., "AIME25", "AIME24", "AMC23")
            n_repeat: Number of repetitions to use
        """
        if task_name not in N_REPEAT_BENCHMARK_PATHS:
            logger.warning(f"n_repeat patching not supported for task: {task_name}")
            return

        path = os.path.join(self.EVALCHEMY_PATH, N_REPEAT_BENCHMARK_PATHS[task_name])
        if not os.path.exists(path):
            logger.warning(f"Benchmark file not found: {path}")
            return

        content = self._read_file(path)

        # Replace n_repeat = N with the configured value
        new_content = re.sub(
            r'self\.n_repeat\s*=\s*\d+',
            f'self.n_repeat = {n_repeat}',
            content
        )

        if new_content != content:
            self._write_file(path, new_content)
            logger.info(f"Patched {task_name} n_repeat to {n_repeat}")

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

    def _patch_vllm_stat_logging(self) -> None:
        """
        Patch vLLM's LLM class to enable stat logging (tokens/sec throughput).

        vLLM's offline LLM class sets disable_log_stats=True by default, which
        suppresses periodic throughput logging. This patch overrides that default
        so that "Avg generation throughput: X tokens/s" lines are printed during
        generation. These log lines stream properly through the subprocess
        (unlike tqdm progress bars which use \\r and don't flush line-by-line).
        """
        path = os.path.join(self.EVALCHEMY_PATH, "eval", "eval.py")
        if not os.path.exists(path):
            return

        content = self._read_file(path)
        if "# Patch vLLM stat logging" in content:
            return  # Already patched

        patch = '''
# Patch vLLM stat logging to show tokens/sec throughput during generation
def _enable_vllm_stat_logging():
    try:
        from vllm import LLM
        _original_init = LLM.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs["disable_log_stats"] = False
            return _original_init(self, *args, **kwargs)

        LLM.__init__ = _patched_init
        print("Enabled vLLM throughput logging (disable_log_stats=False)")
    except Exception as e:
        print(f"Warning: Could not enable vLLM stat logging: {e}")

_enable_vllm_stat_logging()
'''
        marker = "_patch_lmeval_vllm_seed()\n"
        if marker in content:
            self._write_file(path, content.replace(marker, marker + patch, 1))
            logger.info("Patched eval.py to enable vLLM throughput logging")
        else:
            logger.warning("Could not find _patch_lmeval_vllm_seed() call in eval.py for stat logging patch")

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

    def _run_evalchemy_in_process(
        self,
        cmd: list[str],
        cwd: str,
        log_file: str,
    ) -> int:
        """Run evalchemy in-process using runpy instead of a subprocess.

        Executes the evalchemy CLI entrypoint (eval.eval) directly in the current
        process. This ensures that when the Ray worker dies (due to error or preemption),
        all TPU handles die with it — no orphaned subprocesses.

        Args:
            cmd: Original command list (e.g., [sys.executable, "-m", "eval.eval", ...])
            cwd: Working directory (evalchemy repo path)
            log_file: Path to save combined output

        Returns:
            0 on success, non-zero on failure
        """
        # Save state to restore after execution
        saved_argv = sys.argv[:]
        saved_path = sys.path[:]
        saved_cwd = os.getcwd()
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        saved_env = {}
        env_vars_to_set = {
            # Tell vLLM-TPU to use the vLLM model implementation
            "MODEL_IMPL_TYPE": "vllm",
            # Allow max_model_len > max_position_embeddings (safe for RoPE models)
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            # Disable Python output buffering for real-time log streaming
            "PYTHONUNBUFFERED": "1",
        }
        for key in env_vars_to_set:
            saved_env[key] = os.environ.get(key)

        try:
            # Set environment variables (replaces _get_subprocess_env)
            for key, value in env_vars_to_set.items():
                os.environ[key] = value

            # Insert evalchemy repo at front of sys.path
            if cwd not in sys.path:
                sys.path.insert(0, cwd)

            # Change to evalchemy directory (it uses relative paths)
            os.chdir(cwd)

            # Set sys.argv from cmd, stripping "python -m" prefix
            # cmd = [sys.executable, "-m", "eval.eval", "--model", "vllm", ...]
            # sys.argv should be ["eval/eval.py", "--model", "vllm", ...]
            # But runpy handles the module name, so we just need the args after the module
            argv_start = 0
            for i, arg in enumerate(cmd):
                if arg == "-m":
                    argv_start = i + 2  # Skip "-m" and "eval.eval"
                    break
            sys.argv = [os.path.join(cwd, "eval", "eval.py")] + cmd[argv_start:]

            # Tee output to both log file and console
            with open(log_file, "w") as lf:
                tee = _TeeWriter(lf, saved_stdout)
                sys.stdout = tee
                sys.stderr = tee

                runpy.run_module("eval.eval", run_name="__main__", alter_sys=True)

            return 0

        except SystemExit as e:
            # CLI tools use sys.exit(); extract the return code
            code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
            if code != 0:
                logger.error(f"evalchemy exited with code {code}")
            return code

        except Exception as e:
            logger.error(f"evalchemy failed in-process: {e}")
            traceback.print_exc(file=saved_stderr)
            return 1

        finally:
            # Restore all saved state
            sys.argv = saved_argv
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            os.chdir(saved_cwd)
            sys.path = saved_path

            # Restore environment variables
            for key, orig_value in saved_env.items():
                if orig_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = orig_value

            # Remove eval.* modules to prevent stale state between tasks
            modules_to_remove = [m for m in sys.modules if m == "eval" or m.startswith("eval.")]
            for m in modules_to_remove:
                del sys.modules[m]

    def _cleanup_vllm_between_tasks(self) -> None:
        """Clean up vLLM state between tasks to release TPU devices."""
        gc.collect()
        try:
            import vllm.distributed
            if hasattr(vllm.distributed, "destroy_model_parallel"):
                vllm.distributed.destroy_model_parallel()
                logger.info("Destroyed vLLM model parallel state")
        except (ImportError, Exception) as e:
            logger.debug(f"vLLM cleanup skipped: {e}")

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
                # Apply task-specific patches (e.g., n_repeat for AIME benchmarks)
                if eval_task.task_kwargs:
                    if "n_repeat" in eval_task.task_kwargs:
                        self._patch_benchmark_n_repeat(
                            eval_task.name,
                            eval_task.task_kwargs["n_repeat"]
                        )

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
                # Determine batch_size: use engine_kwargs override if provided, else default to "auto"
                # Note: batch_size is a separate lm-eval CLI arg, NOT a vLLM model arg
                batch_size = "auto"
                engine_kwargs = dict(model.engine_kwargs) if model.engine_kwargs else {}
                if "batch_size" in engine_kwargs:
                    batch_size = engine_kwargs.pop("batch_size")

                model_args_parts = [
                    f"pretrained={model_name_or_path}",
                ]
                if engine_seed != 0:
                    model_args_parts.append(f"seed={engine_seed}")
                if max_model_len:
                    model_args_parts.append(f"max_model_len={max_model_len}")
                if max_gen_toks:
                    # Set at model level so lm-eval's default (256) is overridden
                    model_args_parts.append(f"max_gen_toks={max_gen_toks}")

                # Add remaining engine_kwargs (e.g., tensor_parallel_size=4 for multi-chip TPU)
                for key, value in engine_kwargs.items():
                    model_args_parts.append(f"{key}={value}")

                model_args = ",".join(model_args_parts)
                logger.info(f"model_args: {model_args}")

                # Build wandb run name
                if model.base_eval_run_name:
                    step_suffix = _extract_step_suffix(model_name_or_path)
                    wandb_run_name = f"evalchemy-{model.base_eval_run_name}{step_suffix}"
                else:
                    wandb_run_name = f"evalchemy-{model.name}"
                if eval_task.name:
                    wandb_run_name = f"{wandb_run_name}-{eval_task.name}"
                if "seed" in gen_params:
                    wandb_run_name = f"{wandb_run_name}-seed{engine_seed}"

                # Build evalchemy CLI command
                cmd = [
                    sys.executable, "-m", "eval.eval",
                    "--model", "vllm",
                    "--tasks", eval_task.name,
                    "--model_args", model_args,
                    "--batch_size", str(batch_size),
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

                # Run evalchemy in-process (no subprocess, TPU handles die with worker)
                log_file = os.path.join(result_dir, "evalchemy_output.log")
                returncode = self._run_evalchemy_in_process(
                    cmd=cmd,
                    cwd=evalchemy_path,
                    log_file=log_file,
                )

                if returncode != 0:
                    # Read log file contents to include in the error message
                    log_contents = ""
                    if os.path.exists(log_file):
                        with open(log_file, "r") as lf:
                            log_contents = lf.read()

                    # Also write error file for reference
                    error_file = os.path.join(result_dir, "evalchemy_error.txt")
                    with open(error_file, "w") as f:
                        f.write(f"Command: {' '.join(cmd)}\n")
                        f.write(f"Return code: {returncode}\n")
                        f.write(f"\n=== OUTPUT LOG ===\n")
                        f.write(log_contents)

                    # Surface the last portion of the log directly in the error
                    # so it appears in the main logs without needing to find temp files
                    log_tail = log_contents[-5000:] if len(log_contents) > 5000 else log_contents
                    if len(log_contents) > 5000:
                        log_tail = "... [truncated] ...\n" + log_tail

                    error_msg = (
                        f"Evalchemy failed for {eval_task.name} (return code {returncode}).\n"
                        f"=== Command ===\n{' '.join(cmd)}\n"
                        f"=== Output (last 5000 chars) ===\n{log_tail}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

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

                # Clean up vLLM state between tasks to release TPU devices
                self._cleanup_vllm_between_tasks()

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
