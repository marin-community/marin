# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy evaluator driven via OpenAI-compatible HTTP.

Evalchemy (https://github.com/mlfoundations/evalchemy) extends lm-evaluation-harness
with reasoning benchmarks (AIME, MATH500, HumanEval+, MBPP+, LiveCodeBench,
GPQADiamond, ...). It is run via its `eval.eval` CLI entrypoint.

This evaluator:
- Clones evalchemy at a pinned commit.
- Patches a couple of lm-eval-version-compat imports inside evalchemy.
- Runs evalchemy with `--model {local-completions|local-chat-completions}` and
  `--model_args base_url=...`, i.e. as an HTTP client to an already-running
  OpenAI-compatible server. The caller stands the server up via a `ModelLauncher`.
- Manually logs a short results summary to wandb (evalchemy doesn't use lm-eval's
  WandbLogger because its config format is incompatible with CLI parsing).
"""

from __future__ import annotations

import glob
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

from marin.evaluation.api import LmEvalRun
from marin.evaluation.evaluation_config import WANDB_PROJECT
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import (
    _build_lm_eval_model_args,
    _lm_eval_kind,
)
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from marin.inference.model_launcher import RunningModel

logger = logging.getLogger(__name__)

# Evalchemy git repo and commit to use
EVALCHEMY_REPO = "https://github.com/teetone/evalchemy.git"
EVALCHEMY_COMMIT = "010412c"  # 2026-03-14 commit

# Evalchemy benchmarks that have hardcoded n_repeat values and their paths.
# These benchmarks run multiple repetitions with different seeds to compute
# averaged accuracy, but this significantly increases evaluation time.
# For example, AIME25 defaults to n_repeat=10.
_N_REPEAT_BENCHMARK_PATHS = {
    "AIME25": "eval/chat_benchmarks/AIME25/eval_instruct.py",
    "AIME24": "eval/chat_benchmarks/AIME24/eval_instruct.py",
    "AMC23": "eval/chat_benchmarks/AMC23/eval_instruct.py",
    "HMMT": "eval/chat_benchmarks/HMMT/eval_instruct.py",
    "LiveCodeBench": "eval/chat_benchmarks/LiveCodeBench/eval_instruct.py",
    "LiveCodeBenchv5_official": "eval/chat_benchmarks/LiveCodeBenchv5_official/eval_instruct.py",
    "LiveCodeBenchv6_official": "eval/chat_benchmarks/LiveCodeBenchv6_official/eval_instruct.py",
    "CodeForces": "eval/chat_benchmarks/CodeForces/eval_instruct.py",
    "CodeElo": "eval/chat_benchmarks/CodeElo/eval_instruct.py",
    "GPQADiamond": "eval/chat_benchmarks/GPQADiamond/eval_instruct.py",
    "JEEBench": "eval/chat_benchmarks/JEEBench/eval_instruct.py",
    "HLE": "eval/chat_benchmarks/HLE/eval_instruct.py",
    "AIME26": "eval/chat_benchmarks/AIME26/eval_instruct.py",
    "OlympiadBench": "eval/chat_benchmarks/OlympiadBench/eval_instruct.py",
}


class _TeeWriter(io.TextIOBase):
    """Writes to both a log file and original stdout simultaneously."""

    def __init__(self, log_file, original_stdout):
        self._log_file = log_file
        self._original_stdout = original_stdout

    def write(self, s):
        if not s:
            return 0
        self._log_file.write(s)
        self._log_file.flush()
        self._original_stdout.write(s)
        self._original_stdout.flush()
        return len(s)

    def flush(self):
        self._log_file.flush()
        self._original_stdout.flush()

    def fileno(self):
        return self._original_stdout.fileno()

    def isatty(self):
        return False


class EvalchemyEvaluator:
    """Runs Evalchemy reasoning benchmarks via its CLI against an OpenAI HTTP endpoint."""

    CACHE_PATH: str = "/tmp/evalchemy"
    EVALCHEMY_PATH: str = os.path.join(CACHE_PATH, "evalchemy_repo")
    RESULTS_PATH: str = os.path.join(CACHE_PATH, "evalchemy_results")

    def __init__(self, run: LmEvalRun) -> None:
        self.run_config = run

    def run(self, model: RunningModel) -> None:
        run = self.run_config
        try:
            self._setup_evalchemy()
            os.makedirs(self.RESULTS_PATH, exist_ok=True)

            lm_eval_kind = _lm_eval_kind(run)
            model_args = _build_lm_eval_model_args(model, run)
            logger.info(f"model_args: {model_args}")

            for eval_task in run.evals:
                if eval_task.task_kwargs and "n_repeat" in eval_task.task_kwargs:
                    self._patch_benchmark_n_repeat(eval_task.name, eval_task.task_kwargs["n_repeat"])

                result_dir = os.path.join(self.RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")
                os.makedirs(result_dir, exist_ok=True)

                wandb_run_name = self._build_wandb_run_name(
                    model=model, task_name=eval_task.name, base_eval_run_name=run.base_eval_run_name
                )

                cmd = self._build_cmd(
                    lm_eval_kind=lm_eval_kind,
                    model_args=model_args,
                    eval_task=eval_task,
                    result_dir=result_dir,
                    max_eval_instances=run.max_eval_instances,
                    apply_chat_template=run.apply_chat_template,
                    generation_params=run.generation_params,
                    batch_size=run.batch_size,
                )
                logger.info(f"Running: {' '.join(cmd)}")

                log_file = os.path.join(result_dir, "evalchemy_output.log")
                returncode = self._run_evalchemy_in_process(cmd=cmd, cwd=self.EVALCHEMY_PATH, log_file=log_file)
                if returncode != 0:
                    self._raise_with_log_tail(eval_task.name, cmd, log_file, returncode)

                logger.info(f"Completed {eval_task.name}")
                self._log_results_to_wandb(
                    result_dir=result_dir,
                    run_name=wandb_run_name,
                    model_name=model.endpoint.model,
                    task_name=eval_task.name,
                    engine_seed=int(run.generation_params.get("seed", 0)),
                    wandb_tags=run.wandb_tags,
                )
        finally:
            # Upload failures in the finally block are logged (not re-raised) so
            # they don't mask a primary eval exception; rmtree always runs.
            try:
                if is_remote_path(run.output_path):
                    logger.info("Uploading results to GCS...")
                    upload_to_gcs(self.RESULTS_PATH, run.output_path)
            except Exception:
                logger.exception("Failed to upload results to GCS")
            if os.path.exists(self.RESULTS_PATH):
                shutil.rmtree(self.RESULTS_PATH)

    @staticmethod
    def _build_wandb_run_name(*, model: RunningModel, task_name: str, base_eval_run_name: str | None) -> str:
        """Build the wandb run name for an evalchemy task.

        Callers that want a `-step{N}` suffix compute it themselves and pass it in
        via `base_eval_run_name` — the evaluator no longer sees the model path, so
        it cannot derive the step. Falls back to the endpoint model id.
        """
        prefix = f"evalchemy-{base_eval_run_name}" if base_eval_run_name else f"evalchemy-{model.endpoint.model}"
        return f"{prefix}-{task_name}" if task_name else prefix

    @staticmethod
    def _build_cmd(
        *,
        lm_eval_kind: str,
        model_args: str,
        eval_task,
        result_dir: str,
        max_eval_instances: int | None,
        apply_chat_template: bool,
        generation_params: dict,
        batch_size: str = "auto",
    ) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "eval.eval",
            "--model",
            lm_eval_kind,
            "--tasks",
            eval_task.name,
            "--model_args",
            model_args,
            "--batch_size",
            str(batch_size),
            "--output_path",
            result_dir,
            "--verbosity",
            "INFO",
        ]
        if eval_task.num_fewshot > 0:
            cmd.extend(["--num_fewshot", str(eval_task.num_fewshot)])
        if max_eval_instances is not None:
            cmd.extend(["--limit", str(max_eval_instances)])
        if apply_chat_template:
            cmd.append("--apply_chat_template")

        gen_kwargs = [
            f"{key}={generation_params[key]}"
            for key in ("temperature", "max_gen_toks", "top_p")
            if generation_params.get(key) is not None
        ]
        if gen_kwargs:
            cmd.extend(["--gen_kwargs", ",".join(gen_kwargs)])
        return cmd

    @staticmethod
    def _raise_with_log_tail(task_name: str, cmd: list[str], log_file: str, returncode: int) -> None:
        log_contents = ""
        if os.path.exists(log_file):
            with open(log_file) as lf:
                log_contents = lf.read()
        error_file = os.path.join(os.path.dirname(log_file), "evalchemy_error.txt")
        with open(error_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {returncode}\n\n=== OUTPUT LOG ===\n")
            f.write(log_contents)
        log_tail = log_contents[-5000:] if len(log_contents) > 5000 else log_contents
        if len(log_contents) > 5000:
            log_tail = "... [truncated] ...\n" + log_tail
        error_msg = (
            f"Evalchemy failed for {task_name} (return code {returncode}).\n"
            f"=== Command ===\n{' '.join(cmd)}\n"
            f"=== Output (last 5000 chars) ===\n{log_tail}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _setup_evalchemy(self) -> None:
        self._log_lmeval_version()
        self._clone_evalchemy()
        self._apply_patches()

    @staticmethod
    def _log_lmeval_version() -> None:
        import lm_eval

        logger.info("lm-eval version: %s (%s)", lm_eval.__version__, lm_eval.__file__)

    def _clone_evalchemy(self) -> None:
        if os.path.exists(self.EVALCHEMY_PATH):
            logger.info(f"Removing existing evalchemy repo at {self.EVALCHEMY_PATH}")
            shutil.rmtree(self.EVALCHEMY_PATH)
        os.makedirs(self.CACHE_PATH, exist_ok=True)
        logger.info(f"Cloning evalchemy from {EVALCHEMY_REPO} at commit {EVALCHEMY_COMMIT}")
        subprocess.run(
            ["git", "clone", EVALCHEMY_REPO, self.EVALCHEMY_PATH],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", EVALCHEMY_COMMIT],
            cwd=self.EVALCHEMY_PATH,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Evalchemy cloned successfully to {self.EVALCHEMY_PATH}")

    def _apply_patches(self) -> None:
        """Apply lm-eval-version-compat patches to evalchemy's eval_tracker.py and eval.py.

        The vLLM-specific patches (version faking, per-request seed stripping,
        stat-logging) are gone — evalchemy no longer loads `lm_eval.models.vllm_causallms`.
        """
        self._patch_eval_tracker_imports()
        self._patch_eval_py_imports()

    def _patch_benchmark_n_repeat(self, task_name: str, n_repeat: int) -> None:
        """Override n_repeat for benchmarks that hardcode it (e.g. AIME25 defaults to 10).

        Upstream evalchemy has no officially-supported override for this; regex
        rewrite of `self.n_repeat = N` is the only option. Raises if the pattern
        is absent so a future evalchemy version bump can't silently no-op.
        """
        if task_name not in _N_REPEAT_BENCHMARK_PATHS:
            raise ValueError(
                f"n_repeat patching not supported for task: {task_name}. "
                f"Known tasks: {sorted(_N_REPEAT_BENCHMARK_PATHS)}"
            )
        path = os.path.join(self.EVALCHEMY_PATH, _N_REPEAT_BENCHMARK_PATHS[task_name])
        with open(path) as f:
            content = f.read()
        new_content = re.sub(r"self\.n_repeat\s*=\s*\d+", f"self.n_repeat = {n_repeat}", content)
        if new_content == content:
            raise RuntimeError(
                f"Could not find `self.n_repeat = <int>` in {path}. "
                f"Evalchemy commit {EVALCHEMY_COMMIT} may have changed - update the patch marker."
            )
        with open(path, "w") as f:
            f.write(new_content)
        logger.info(f"Patched {task_name} n_repeat to {n_repeat}")

    def _patch_eval_tracker_imports(self) -> None:
        path = os.path.join(self.EVALCHEMY_PATH, "eval", "eval_tracker.py")
        with open(path) as f:
            content = f.read()
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
        if old not in content:
            raise RuntimeError(
                "Could not find expected import in eval_tracker.py. "
                f"Evalchemy commit {EVALCHEMY_COMMIT} may have changed - update the patch marker."
            )
        with open(path, "w") as f:
            f.write(content.replace(old, new))
        logger.info("Patched eval_tracker.py to handle different lm-eval versions")

    def _patch_eval_py_imports(self) -> None:
        path = os.path.join(self.EVALCHEMY_PATH, "eval", "eval.py")
        with open(path) as f:
            content = f.read()
        if "# Patch utils.eval_logger" in content:
            return  # Already patched
        patch = """
# Patch utils.eval_logger for lm-eval compatibility
if not hasattr(utils, 'eval_logger'):
    try:
        from lm_eval.logging_utils import eval_logger as _eval_logger
        utils.eval_logger = _eval_logger
    except ImportError:
        import logging as _logging
        utils.eval_logger = _logging.getLogger("lm-eval")
"""
        marker = "from lm_eval.tasks import TaskManager as PretrainTaskManager"
        if marker not in content:
            raise RuntimeError(
                "Could not find TaskManager import in eval.py. "
                f"Evalchemy commit {EVALCHEMY_COMMIT} may have changed - update the patch marker."
            )
        with open(path, "w") as f:
            f.write(content.replace(marker, marker + patch))
        logger.info("Patched eval.py to handle different lm-eval versions")

    def _run_evalchemy_in_process(self, *, cmd: list[str], cwd: str, log_file: str) -> int:
        """Run evalchemy's `eval.eval` entrypoint in-process via runpy.

        In-process matters on TPU: when the Ray worker dies, all open handles die
        with it, so no orphaned subprocesses linger.
        """
        saved_argv = sys.argv[:]
        saved_path = sys.path[:]
        saved_cwd = os.getcwd()
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        env_vars_to_set = {"PYTHONUNBUFFERED": "1"}
        saved_env = {k: os.environ.get(k) for k in env_vars_to_set}

        try:
            for key, value in env_vars_to_set.items():
                os.environ[key] = value
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
            os.chdir(cwd)

            # cmd = [sys.executable, "-m", "eval.eval", "--model", "...", ...]
            # Strip "python -m eval.eval" to form argv for runpy.
            argv_start = 0
            for i, arg in enumerate(cmd):
                if arg == "-m":
                    argv_start = i + 2
                    break
            sys.argv = [os.path.join(cwd, "eval", "eval.py"), *cmd[argv_start:]]

            with open(log_file, "w") as lf:
                tee = _TeeWriter(lf, saved_stdout)
                sys.stdout = tee
                sys.stderr = tee
                runpy.run_module("eval.eval", run_name="__main__", alter_sys=True)
            return 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
            if code != 0:
                logger.error(f"evalchemy exited with code {code}")
            return code
        except Exception as exc:
            logger.error(f"evalchemy failed in-process: {exc}")
            traceback.print_exc(file=saved_stderr)
            return 1
        finally:
            sys.argv = saved_argv
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            os.chdir(saved_cwd)
            sys.path = saved_path
            for key, orig_value in saved_env.items():
                if orig_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = orig_value
            modules_to_remove = [m for m in sys.modules if m == "eval" or m.startswith("eval.")]
            for m in modules_to_remove:
                del sys.modules[m]

    @staticmethod
    def _log_results_to_wandb(
        *,
        result_dir: str,
        run_name: str,
        model_name: str,
        task_name: str,
        engine_seed: int,
        wandb_tags: list[str] | None,
    ) -> None:
        """Log evaluation results to wandb after evalchemy completes.

        Done manually because lm-eval's WandbLogger expects init_args/config_args
        format that's incompatible with evalchemy's CLI parsing.
        """
        import wandb

        if not os.environ.get("WANDB_API_KEY"):
            logger.info("WANDB_API_KEY not set, skipping wandb logging")
            return

        # Evalchemy saves results as <output_path>/<model_name_sanitized>/results_<ISO>.json
        results_file = _pick_latest_results_file(result_dir)
        if results_file is None:
            logger.warning(f"No results file found in {result_dir}, skipping wandb logging")
            return
        with open(results_file) as f:
            results = json.load(f)

        tags = ["evalchemy", task_name.lower()[:64], model_name.lower()[:64]]
        if engine_seed != 0:
            tags.append(f"seed{engine_seed}")
        if wandb_tags:
            tags.extend(t[:64] for t in wandb_tags)

        wandb.init(
            project=WANDB_PROJECT,
            entity=os.environ.get("WANDB_ENTITY", "marin-community"),
            name=run_name,
            job_type="eval",
            tags=tags,
            config={"model_name": model_name, "task_name": task_name, "engine_seed": engine_seed},
            reinit=True,
        )

        for task, metrics in results.get("results", {}).items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, int | float):
                    wandb.log({f"{task}/{metric_name}": metric_value})
        if "results" in results:
            wandb.log({"results_summary": results["results"]})

        _finish_wandb_quietly()
        logger.info(f"Logged results to wandb run: {run_name}")


def _pick_latest_results_file(result_dir: str) -> str | None:
    """Evalchemy saves results as `<result_dir>/<model_name>/results_<ts>.json`."""
    for pattern in (
        os.path.join(result_dir, "*", "results_*.json"),
        os.path.join(result_dir, "results_*.json"),
        os.path.join(result_dir, "*", "results.json"),
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _finish_wandb_quietly() -> None:
    """`wandb.finish()` but filter its BrokenPipeError traceback from stderr.

    wandb's async finisher dumps a traceback on broken pipes during Ray worker
    shutdown. The pipe is already gone — it's noise, not signal.
    """
    import wandb

    saved_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        wandb.finish()
    finally:
        captured = sys.stderr.getvalue()
        sys.stderr = saved_stderr
        if captured and "BrokenPipeError" not in captured:
            logger.warning(f"wandb.finish() stderr: {captured.strip()}")
