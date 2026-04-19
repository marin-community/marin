# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor framework evaluator, driven via OpenAI-compatible HTTP.

Runs any Harbor dataset (https://harborframework.com/registry): AIME (60 math
problems), Terminal-Bench (89 tasks), SWE-bench Verified (500 tasks), etc.

Two operation modes, both through the same entry:

- **Local vLLM mode**: caller stands up a vLLM server via `VllmLauncher` and
  passes the `RunningModel`. Harbor talks to it via `agent_kwargs["api_base"]`
  and the `hosted_vllm/<name>` provider.
- **External-API mode** (Claude, GPT, ...): caller passes a `RunningModel` with
  `endpoint.url == LITELLM_PROVIDER_URL`. Harbor drops `api_base` and lets
  LiteLLM route by the model name directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from rigging.filesystem import open_url

from marin.evaluation.api import HarborRun
from marin.evaluation.utils import download_from_gcs, is_remote_path, upload_to_gcs
from marin.inference.model_launcher import LITELLM_PROVIDER_URL, RunningModel
from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger(__name__)


_HOSTED_VLLM_CANONICAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_DEFAULT_HOSTED_VLLM_MODEL_INFO: dict[str, Any] = {
    "max_input_tokens": 32768,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}


def sanitize_hosted_vllm_canonical_name(name: str) -> str:
    """Return a Harbor-safe canonical name for `hosted_vllm/<canonical>`.

    Harbor requires canonical names to match `[A-Za-z0-9._-]{1,64}`. Exposed so
    callers can sanitize before invoking `VllmLauncher(extra_args=["--served-model-name", ...])`.
    """
    candidate = re.sub(r"[^A-Za-z0-9._-]", "_", name.strip()).strip("_")
    if not candidate:
        candidate = "model"
    if len(candidate) > 64:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate[:55]}_{digest}"
    if not _HOSTED_VLLM_CANONICAL_NAME_PATTERN.fullmatch(candidate):
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
        candidate = f"model_{digest}"
    return candidate


def _is_external_api_model(model: RunningModel) -> bool:
    return model.endpoint.url == LITELLM_PROVIDER_URL


def _build_model_name_and_agent_kwargs(model: RunningModel, *, agent_kwargs: dict) -> tuple[str, dict]:
    """Decide Harbor's `model_name` (LiteLLM provider string) and seed agent_kwargs.

    External API: pass the model name through verbatim (e.g. "claude-opus-4"),
    let LiteLLM pick the provider.

    Local vLLM: route via `hosted_vllm/<canonical>` and point at our server's
    base URL.
    """
    resolved = dict(agent_kwargs)
    if _is_external_api_model(model):
        resolved.pop("api_base", None)  # don't mislead LiteLLM's routing
        return model.endpoint.model, resolved
    resolved.setdefault("api_base", model.endpoint.url)
    resolved.setdefault("model_info", _DEFAULT_HOSTED_VLLM_MODEL_INFO)
    return f"hosted_vllm/{model.endpoint.model}", resolved


def _generate_stable_job_name(dataset: str, version: str, model_name: str, agent: str, task_limit: int | None) -> str:
    """Generate a deterministic job name for resume capability."""
    key = f"{dataset}|{version}|{model_name}|{agent}|{task_limit}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    safe_dataset = re.sub(r"[^A-Za-z0-9_-]", "_", dataset)[:32]
    return f"harbor_{safe_dataset}_{digest}"


def _get_stable_local_workdir(job_name: str) -> Path:
    """Stable local working directory for Harbor jobs (survives preemptions)."""
    workdir = Path("/tmp/harbor_workdir") / job_name
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _restore_trials_from_gcs(gcs_output_path: str, local_job_dir: Path) -> int:
    """Restore completed trials from GCS to enable Harbor resume."""
    trials_gcs_path = os.path.join(gcs_output_path, "harbor_trials")
    if not fsspec_exists(trials_gcs_path):
        return 0
    result_files = fsspec_glob(os.path.join(trials_gcs_path, "*/result.json"))
    restored = 0
    for result_file in result_files:
        trial_gcs_dir = os.path.dirname(result_file)
        trial_name = os.path.basename(trial_gcs_dir)
        local_trial_dir = local_job_dir / trial_name
        if (local_trial_dir / "result.json").exists():
            continue
        # Per-trial restore is best-effort: missing a trial means we redo it.
        try:
            local_trial_dir.mkdir(parents=True, exist_ok=True)
            download_from_gcs(trial_gcs_dir, str(local_trial_dir))
            restored += 1
        except Exception:
            logger.exception(f"Failed to restore trial {trial_name} from GCS")
    return restored


def _create_trial_upload_hook(gcs_output_path: str, local_job_dir: Path):
    """Create an async hook that uploads each completed trial's dir to GCS."""

    async def on_trial_ended(event) -> None:
        if event.result is None:
            return
        trial_name = event.result.trial_name
        local_trial_dir = local_job_dir / trial_name
        if not local_trial_dir.exists():
            logger.warning(f"Trial directory not found for upload: {local_trial_dir}")
            return
        # Harbor writes trial dirs as root from Docker; sudo-relax perms so we
        # can read them back out. `check=False` — if sudo is missing, the
        # subsequent upload will fail with a real permission error.
        subprocess.run(["sudo", "chmod", "-R", "755", str(local_trial_dir)], check=False, capture_output=True)
        trial_gcs_path = os.path.join(gcs_output_path, "harbor_trials", trial_name)
        # Swallow per-trial upload failures: Harbor halts the whole job if a
        # hook raises, and one flaky upload out of hundreds shouldn't kill it.
        try:
            upload_to_gcs(str(local_trial_dir), trial_gcs_path)
        except Exception:
            logger.exception(f"Failed to upload trial {trial_name} to GCS")
            return
        logger.info(f"Uploaded trial {trial_name} to GCS")

    return on_trial_ended


class HarborEvaluator:
    """Runs any Harbor dataset against an OpenAI-compatible endpoint."""

    def __init__(self, run: HarborRun) -> None:
        self.run_config = run

    def run(self, model: RunningModel) -> None:
        run = self.run_config
        agent_kwargs = dict(run.agent_kwargs)
        if run.model_info is not None:
            agent_kwargs.setdefault("model_info", run.model_info)

        harbor_model_name, agent_kwargs = _build_model_name_and_agent_kwargs(model, agent_kwargs=agent_kwargs)

        logger.info("Running Harbor evaluation: %s@%s", run.dataset, run.version)
        logger.info(
            "Agent=%s Model=%s Concurrent=%s Env=%s",
            run.agent,
            harbor_model_name,
            run.n_concurrent,
            run.env,
        )
        if run.max_eval_instances is not None:
            logger.info("Limiting to first %s task(s)", run.max_eval_instances)

        results = self._run_harbor_trials(model_name=harbor_model_name, agent_kwargs=agent_kwargs)
        parsed_results = self._parse_results(results)
        parsed_results["meta"] = {
            "dataset": run.dataset,
            "version": run.version,
            "agent": run.agent,
            "n_concurrent": run.n_concurrent,
            "env": run.env,
            "max_eval_instances": run.max_eval_instances,
            "model": harbor_model_name,
        }
        self._save_results(parsed_results, harbor_model_name)
        logger.info("Harbor evaluation completed successfully")

    def _run_harbor_trials(self, *, model_name: str, agent_kwargs: dict) -> dict:
        """Run Harbor trials using its programmatic API.

        Robust to preemption: stable workdir, incremental GCS uploads on each
        completed trial, restore-from-GCS on resume, leverages Harbor's native
        resume capability.
        """
        from harbor.job import Job
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.job.config import JobConfig, OrchestratorConfig
        from harbor.models.orchestrator_type import OrchestratorType
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig

        run = self.run_config
        job_name = _generate_stable_job_name(run.dataset, run.version, model_name, run.agent, run.max_eval_instances)
        workdir = _get_stable_local_workdir(job_name)
        output_dir = workdir / "harbor_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        job_dir = output_dir / job_name
        if run.output_path and is_remote_path(run.output_path):
            restored = _restore_trials_from_gcs(run.output_path, job_dir)
            if restored > 0:
                logger.info(f"Restored {restored} completed trial(s) from GCS")

        # "local" in Marin means Docker in Harbor; otherwise pass through to
        # Harbor's enum (raises ValueError if unsupported).
        harbor_env_type = EnvironmentType.DOCKER if run.env == "local" else EnvironmentType(run.env)

        dataset_config = _build_dataset_config(
            dataset=run.dataset, version=run.version, workdir=workdir, task_limit=run.max_eval_instances
        )

        config = JobConfig(
            job_name=job_name,
            jobs_dir=output_dir,
            datasets=[dataset_config],
            agents=[AgentConfig(name=run.agent, model_name=model_name, kwargs=agent_kwargs)],
            orchestrator=OrchestratorConfig(
                type=OrchestratorType.LOCAL,
                n_concurrent_trials=run.n_concurrent,
            ),
            environment=EnvironmentConfig(type=harbor_env_type),
        )

        job = Job(config)
        if run.output_path and is_remote_path(run.output_path):
            upload_hook = _create_trial_upload_hook(run.output_path, job.job_dir)
            job.on_trial_ended(upload_hook)

        logger.info(
            f"Starting Harbor job {job.config.job_name} via programmatic API "
            f"(resuming={job.is_resuming}, total_trials={len(job)})"
        )
        asyncio.run(job.run())
        logger.info("Harbor execution completed")

        # See _create_trial_upload_hook for why we chmod: Docker wrote as root,
        # we need to read the results back. check=False — if sudo is missing
        # the subsequent _collect_trial_results will fail with a real error.
        subprocess.run(["sudo", "chmod", "-R", "755", str(job.job_dir)], check=False, capture_output=True)

        return _collect_trial_results(job.job_dir)

    @staticmethod
    def _parse_results(results: dict) -> dict:
        """Parse Harbor results into Marin format."""
        trials = results.get("trials", {})
        parsed: dict = {
            "trials": {},
            "aggregate": {
                "total_trials": 0,
                "successful_trials": 0,
                "mean_reward": 0.0,
                "accuracy": 0.0,
            },
        }
        total_reward = 0.0
        successful = 0
        for trial_id, trial_data in trials.items():
            reward = trial_data.get("reward", 0.0)
            total_reward += reward
            # Harbor benchmarks return 1.0 for success and 0.0 for failure; tolerate small FP error.
            if reward >= 0.99:
                successful += 1
            parsed["trials"][trial_id] = {
                "task_id": trial_id,
                "reward": reward,
                "correct": reward >= 0.99,
                "status": trial_data.get("status", "unknown"),
                "trajectory_length": trial_data.get("trajectory_length", 0),
                "trajectory_content": trial_data.get("trajectory_content"),
                "error": trial_data.get("error"),
            }
        total = len(trials)
        if total > 0:
            parsed["aggregate"]["total_trials"] = total
            parsed["aggregate"]["successful_trials"] = successful
            parsed["aggregate"]["mean_reward"] = total_reward / total
            parsed["aggregate"]["accuracy"] = successful / total
        logger.info(
            f"Results summary: {successful}/{total} successful "
            f"(accuracy: {parsed['aggregate']['accuracy']:.2%}, "
            f"mean reward: {parsed['aggregate']['mean_reward']:.3f})"
        )
        return parsed

    def _save_results(self, results: dict, harbor_model_name: str) -> None:
        """Save results to GCS (or local) and log to W&B."""
        run = self.run_config
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        results_file_name = f"results_{timestamp}.json"
        samples_file_name = f"samples_{timestamp}.jsonl"

        output_path = run.output_path
        if not is_remote_path(output_path):
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, "trajectories"), exist_ok=True)

        samples_path = os.path.join(output_path, samples_file_name)
        results_path = os.path.join(output_path, results_file_name)

        for trial_id, trial_data in results["trials"].items():
            trajectory_content = trial_data.get("trajectory_content")
            if not trajectory_content:
                continue
            try:
                ext = ".json" if trajectory_content.strip().startswith("{") else ".jsonl"
                trajectory_path = os.path.join(output_path, "trajectories", f"{trial_id}{ext}")
                with open_url(trajectory_path, "w") as dst:
                    dst.write(trajectory_content)
                results["trials"][trial_id]["trajectory_path"] = trajectory_path
                logger.info(
                    f"Saved trajectory for {trial_id} to {trajectory_path} "
                    f"(length: {trial_data.get('trajectory_length', 0)})"
                )
            except Exception as e:
                logger.warning(f"Failed to save trajectory for {trial_id}: {e}")

        trials_for_output = {
            trial_id: {k: v for k, v in dict(trial_data).items() if k not in ("trial_dir", "trajectory_content")}
            for trial_id, trial_data in results["trials"].items()
        }

        with open_url(samples_path, "w") as f:
            for trial_id, trial_data in sorted(trials_for_output.items()):
                sample = {
                    "task_id": trial_id,
                    "dataset": run.dataset,
                    "version": run.version,
                    "model": harbor_model_name,
                    "timestamp": timestamp,
                    **trial_data,
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        aggregated_results = {
            "meta": results.get("meta", {}) | {"timestamp": timestamp},
            "aggregate": results["aggregate"],
            "samples_path": samples_path,
        }
        with open_url(results_path, "w") as f:
            json.dump(aggregated_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote samples to {samples_path}")
        logger.info(f"Wrote aggregated results to {results_path}")

        self._log_to_wandb(
            aggregate=results["aggregate"],
            trials_for_output=trials_for_output,
            harbor_model_name=harbor_model_name,
        )

    def _log_to_wandb(self, *, aggregate: dict, trials_for_output: dict, harbor_model_name: str) -> None:
        run = self.run_config
        try:
            import pandas as pd
            import wandb

            wandb.init(
                project=os.environ.get("WANDB_PROJECT") or "harbor",
                name=f"{harbor_model_name}-{run.dataset}",
                tags=(run.wandb_tags or []) + ["harbor", run.dataset],
                config={
                    "model": harbor_model_name,
                    "dataset": run.dataset,
                    "version": run.version,
                    "evaluator": "harbor",
                },
            )
            wandb.log(aggregate)
            trials_df = pd.DataFrame.from_dict(trials_for_output, orient="index")
            wandb.log({"trials": wandb.Table(dataframe=trials_df)})
            wandb.finish()
            logger.info("Results logged to W&B")
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")


def _build_dataset_config(*, dataset: str, version: str, workdir: Path, task_limit: int | None):
    """Pick between local, HF-hosted, and registry datasets."""
    from harbor.models.job.config import LocalDatasetConfig, RegistryDatasetConfig
    from harbor.models.registry import RemoteRegistryInfo

    dataset_path = Path(dataset).expanduser()

    is_hf = (
        dataset.startswith("hf://")
        or dataset.startswith("hf:")
        or (version == "hf" and "/" in dataset and not dataset_path.exists())
    )
    if is_hf:
        from huggingface_hub import snapshot_download

        if dataset.startswith("hf://"):
            hf_repo_id = dataset[len("hf://") :]
        elif dataset.startswith("hf:"):
            hf_repo_id = dataset[len("hf:") :]
        else:
            hf_repo_id = dataset
        hf_cache_dir = workdir / "hf_cache"
        hf_local_dir = workdir / "hf_dataset"
        hf_local_dir.mkdir(parents=True, exist_ok=True)
        dataset_root = snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=str(hf_local_dir),
            cache_dir=str(hf_cache_dir),
            token=os.environ.get("HF_TOKEN", False),
        )
        gitattributes_path = Path(dataset_root) / ".gitattributes"
        if gitattributes_path.exists():
            gitattributes_path.unlink()
        return LocalDatasetConfig(path=Path(dataset_root), n_tasks=task_limit)

    if dataset_path.exists():
        if not dataset_path.is_dir():
            raise ValueError(f"Harbor dataset path must be a directory, got: {dataset_path}")
        return LocalDatasetConfig(path=dataset_path, n_tasks=task_limit)

    return RegistryDatasetConfig(registry=RemoteRegistryInfo(), name=dataset, version=version, n_tasks=task_limit)


def _collect_trial_results(job_dir: Path) -> dict:
    """Read per-trial result.json files from a Harbor job directory."""
    results: dict = {"trials": {}}
    for trial_dir in (d for d in job_dir.iterdir() if d.is_dir()):
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue
        trial_result = json.loads(result_file.read_text(encoding="utf-8"))
        trial_id = trial_result.get("task_name", trial_dir.name)

        # Reward lives under `verifier_result.rewards.reward`.
        verifier_result = trial_result.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        reward = rewards.get("reward", 0.0)
        if not isinstance(reward, int | float):
            reward = 0.0

        error = None
        exception_info = trial_result.get("exception_info")
        if exception_info:
            error = {
                "type": exception_info.get("exception_type"),
                "message": exception_info.get("exception_message"),
            }

        trajectory_file = trial_dir / "agent" / "trajectory.json"
        trajectory_content = None
        trajectory_length = 0
        if trajectory_file.exists():
            try:
                trajectory_content = trajectory_file.read_text(encoding="utf-8")
                trajectory_data = json.loads(trajectory_content)
                trajectory_length = len(trajectory_data.get("steps", []))
            except Exception as e:
                logger.warning(f"Failed to read trajectory for {trial_id}: {e}")

        results["trials"][trial_id] = {
            "reward": reward,
            "status": "completed" if not exception_info else "failed",
            "task_name": trial_id,
            "started_at": trial_result.get("started_at"),
            "finished_at": trial_result.get("finished_at"),
            "error": error,
            "trajectory_content": trajectory_content,
            "trajectory_length": trajectory_length,
        }
    return results
