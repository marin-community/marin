# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor dataset against an already-served model and normalize the trials.

The group launcher serves a model once and hands this runner an OpenAI endpoint; the runner points a
Harbor agent at it (``hosted_vllm/<served-name>``), runs the dataset's trials on the configured
sandbox environment, and normalizes each finished trial into the shared eval
contract: one agentic :class:`~marin.evaluation.samples.EvalSample` per task (its reward, its grading,
and a reference to the saved trajectory) plus an aggregate this module's :class:`HarborResult` reads
back for the record's metrics. Harbor's own per-trial resume is preserved by restoring completed
trials from the durable output path before the job runs.

The ``harbor`` dependency is optional and imported lazily, so importing this module never requires it.
"""

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem import StoragePath, is_remote_path, prefix_join, url_to_fs

from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborRuntimeOverlay,
    ValidatedHarborConfig,
    run_harbor_driver,
)
from marin.evaluation.records import RunStatus
from marin.evaluation.runner import EvaluationError, EvaluationOutcome
from marin.evaluation.samples import EvalSample, Grading, SampleKind, write_sample_parquet
from marin.inference.types import RunningModel

logger = logging.getLogger(__name__)

_HARBOR_WORKDIR = Path("/tmp/harbor_workdir")
_HARBOR_RESULTS_DIR = "harbor_results"
_JOB_DATASET_LENGTH = 32
_JOB_DIGEST_LENGTH = 12

# The reward at or above which a Harbor trial counts as solved (rewards are typically 0.0 / 1.0; the
# margin tolerates float noise).
SOLVED_REWARD = 0.99

_CANONICAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class HarborTrial:
    """One finished Harbor trial, normalized off its ``result.json``."""

    task_id: str
    reward: float
    status: str
    trajectory: str | None
    error: dict | None


@dataclass(frozen=True)
class HarborRunResult:
    """The aggregate of one Harbor run, and where its per-sample parquet landed."""

    dataset: str
    total_trials: int
    solved_trials: int
    failed_trials: int
    mean_reward: float
    accuracy: float
    samples_path: str | None

    def task_metrics(self) -> dict[str, dict[str, float]]:
        """Metrics keyed like the evalchemy reader: ``{dataset: {metric: value}}``."""
        return {
            self.dataset: {
                "accuracy": self.accuracy,
                "mean_reward": self.mean_reward,
                "solved": float(self.solved_trials),
                "total": float(self.total_trials),
            }
        }


def canonical_served_name(name: str) -> str:
    """A Harbor-safe served-model name (``[A-Za-z0-9._-]{1,64}``) derived from ``name``."""
    candidate = re.sub(r"[^A-Za-z0-9._-]", "_", name.strip()).strip("_") or "model"
    if len(candidate) > 64:
        candidate = f"{candidate[:55]}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"
    if not _CANONICAL_NAME_PATTERN.fullmatch(candidate):
        candidate = f"model_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
    return candidate


def _job_name(dataset: str, identity: tuple[object, ...]) -> str:
    """A deterministic Harbor job name so a re-run resumes the previous job's completed trials."""
    key = "|".join(str(value) for value in identity)
    digest = hashlib.sha256(key.encode()).hexdigest()[:_JOB_DIGEST_LENGTH]
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", dataset)[:_JOB_DATASET_LENGTH]
    return f"harbor_{safe}_{digest}"


def _job_paths(job_name: str) -> tuple[Path, Path]:
    workdir = _HARBOR_WORKDIR / job_name
    return workdir, workdir / _HARBOR_RESULTS_DIR


def _restore_completed_trials(out_path: str, job_dir: Path) -> int:
    """Download completed trials (those with ``result.json``) from ``out_path`` so Harbor skips them."""
    trials_root = prefix_join(out_path, "harbor_trials")
    if not StoragePath(trials_root).exists():
        return 0
    restored = 0
    for result_file in StoragePath(prefix_join(trials_root, "*/result.json")).glob():
        trial_dir = result_file.parent
        local = job_dir / trial_dir.name
        if (local / "result.json").exists():
            continue
        trial_dir.download_to(str(local), recursive=True)
        restored += 1
    return restored


def _read_trials(job_dir: Path) -> list[HarborTrial]:
    """Read every finished trial under ``job_dir`` off its ``result.json`` and ``trajectory.json``."""
    trials: list[HarborTrial] = []
    for trial_dir in sorted(d for d in job_dir.iterdir() if d.is_dir()):
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue
        data = json.loads(result_file.read_text())
        task_id = data.get("task_name", trial_dir.name)
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward", 0.0)
        reward = float(reward) if isinstance(reward, int | float) else 0.0
        exc = data.get("exception_info")
        error = {"type": exc.get("exception_type"), "message": exc.get("exception_message")} if exc else None
        trajectory_file = trial_dir / "agent" / "trajectory.json"
        trajectory = trajectory_file.read_text() if trajectory_file.exists() else None
        trials.append(
            HarborTrial(
                task_id=task_id,
                reward=reward,
                status="failed" if exc else "completed",
                trajectory=trajectory,
                error=error,
            )
        )
    return trials


def _sample_for(trial: HarborTrial, dataset: str, out_path: str) -> EvalSample:
    """Normalize one trial into an agentic :class:`EvalSample`, saving its trajectory alongside."""
    trajectory_uri: str | None = None
    if trial.trajectory:
        trajectory_uri = prefix_join(out_path, f"trajectories/{trial.task_id}.json")
        StoragePath(trajectory_uri).write_text(trial.trajectory)
    solved = trial.reward >= SOLVED_REWARD
    detail = json.dumps({"reward": trial.reward, "error": trial.error}, ensure_ascii=False)
    return EvalSample(
        task=dataset,
        doc_id=trial.task_id,
        kind=SampleKind.AGENTIC,
        trajectory_uri=trajectory_uri,
        grading=Grading(
            method="harbor:verifier",
            metric="reward",
            score=trial.reward,
            passed=solved,
            detail=detail,
        ),
        metrics={"reward": trial.reward},
        correct=solved,
    )


def _write_samples(trials: list[HarborTrial], dataset: str, out_path: str) -> str | None:
    """Write one agentic-sample parquet for the run; return its path (None if there were no trials)."""
    if not trials:
        return None
    samples = [_sample_for(trial, dataset, out_path) for trial in trials]
    dest = prefix_join(out_path, "samples_harbor.parquet")
    fs, _ = url_to_fs(dest)
    write_sample_parquet(fs, dest, samples)
    return dest


def _aggregate(trials: list[HarborTrial], dataset: str, samples_path: str | None) -> HarborRunResult:
    total = len(trials)
    solved = sum(1 for trial in trials if trial.reward >= SOLVED_REWARD)
    failed = sum(1 for trial in trials if trial.error is not None)
    total_reward = sum(trial.reward for trial in trials)
    return HarborRunResult(
        dataset=dataset,
        total_trials=total,
        solved_trials=solved,
        failed_trials=failed,
        mean_reward=(total_reward / total) if total else 0.0,
        accuracy=(solved / total) if total else 0.0,
        samples_path=samples_path,
    )


def _upload_trials(job_dir: Path, out_path: str) -> None:
    """Upload each finished trial directory under ``job_dir`` to ``out_path/harbor_trials`` for resume."""
    for trial_dir in (d for d in job_dir.iterdir() if d.is_dir()):
        if (trial_dir / "result.json").exists():
            target = StoragePath(prefix_join(out_path, f"harbor_trials/{trial_dir.name}"))
            target.upload_from(str(trial_dir), recursive=True)


def _run_harbor_job(
    *,
    job_name: str,
    workdir: Path,
    config: ValidatedHarborConfig,
    overlay: HarborRuntimeOverlay,
    dataset: str,
    environment: str,
    output_dir: str,
    driver_env: Mapping[str, str],
) -> HarborRunResult:
    results_dir = workdir / _HARBOR_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    job_dir = results_dir / job_name
    if is_remote_path(output_dir):
        restored = _restore_completed_trials(output_dir, job_dir)
        if restored:
            logger.info("restored %d completed Harbor trial(s) from %s", restored, output_dir)

    logger.info("starting Harbor job %s (dataset=%s env=%s)", job_name, dataset, environment)
    run_harbor_driver(config, overlay, driver_env)

    trials = _read_trials(job_dir)
    if is_remote_path(output_dir):
        _upload_trials(job_dir, output_dir)
    samples_path = _write_samples(trials, dataset, output_dir)
    result = _aggregate(trials, dataset, samples_path)
    StoragePath(prefix_join(output_dir, "harbor_result.json")).write_text(
        json.dumps(
            {
                "dataset": result.dataset,
                "total_trials": result.total_trials,
                "solved_trials": result.solved_trials,
                "failed_trials": result.failed_trials,
                "mean_reward": result.mean_reward,
                "accuracy": result.accuracy,
            },
            indent=2,
        )
    )
    logger.info(
        "Harbor %s: %d/%d solved (accuracy=%.3f mean_reward=%.3f)",
        dataset,
        result.solved_trials,
        result.total_trials,
        result.accuracy,
        result.mean_reward,
    )
    return result


def _evaluation_outcome(run: Callable[[], HarborRunResult], output_dir: str) -> EvaluationOutcome:
    try:
        result = run()
    except Exception as exc:
        raise EvaluationError(str(exc), status=RunStatus.FAILED) from exc
    if not result.total_trials:
        raise EvaluationError(
            f"Harbor eval finished with no trials under {output_dir!r}",
            status=RunStatus.FAILED,
        )
    if result.failed_trials:
        raise EvaluationError(
            f"Harbor eval finished with {result.failed_trials} of {result.total_trials} failed trials "
            f"under {output_dir!r}",
            status=RunStatus.FAILED,
        )
    return EvaluationOutcome(metrics=result.task_metrics())


@dataclass(frozen=True)
class HarborExecutor:
    """Run one normalized Harbor job policy against a served model."""

    config: ValidatedHarborConfig
    task_limit: int | None
    model_agent_kwargs: Mapping[str, object]
    secret_env_keys: tuple[str, ...] = ()

    def _run(
        self,
        model: RunningModel,
        output_dir: str,
        hf_token: str | None,
        driver_env: Mapping[str, str],
    ) -> HarborRunResult:
        dataset = self.config.record_dataset
        job_name = _job_name(
            dataset,
            (self.config.digest, model.endpoint.model, self.task_limit),
        )
        workdir, results_dir = _job_paths(job_name)
        dataset_path = materialize_harbor_dataset(
            self.config,
            workdir,
            hf_token=hf_token,
        )
        overlay = HarborRuntimeOverlay(
            job_name=job_name,
            jobs_dir=str(results_dir),
            dataset_path=str(dataset_path) if dataset_path is not None else None,
            endpoint_url=model.endpoint.base_url,
            served_model=model.endpoint.model,
            task_limit=self.task_limit,
            model_agent_kwargs=self.model_agent_kwargs,
        )
        return _run_harbor_job(
            job_name=job_name,
            workdir=workdir,
            config=self.config,
            overlay=overlay,
            dataset=dataset,
            environment=self.config.environment,
            output_dir=output_dir,
            driver_env=driver_env,
        )

    def __call__(
        self,
        model: RunningModel,
        output_dir: str,
        env_vars: Mapping[str, str],
    ) -> EvaluationOutcome:
        driver_env = {key: env_vars[key] for key in self.secret_env_keys}
        hf_token = env_vars.get("HF_TOKEN")
        if hf_token:
            driver_env["HF_TOKEN"] = hf_token
        return _evaluation_outcome(
            lambda: self._run(model, output_dir, hf_token, driver_env),
            output_dir,
        )
