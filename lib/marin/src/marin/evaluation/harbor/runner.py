# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor dataset against an already-served model and normalize the trials.

The group launcher serves a model once and hands this runner an OpenAI endpoint; the runner points a
Harbor agent at it (``hosted_vllm/<served-name>``), runs the dataset's trials on the configured
sandbox environment, and normalizes each finished trial into the shared eval
contract: one agentic :class:`~marin.evaluation.samples.EvalSample` per task (its reward, its grading,
and a reference to the saved trajectory) plus an aggregate this module's :class:`HarborResult` reads
back for the record's metrics. Harbor writes each trial's ``result.json`` and trajectory straight to
the durable output path as it finishes, so a completed trial survives a driver killed before the job
returns and Harbor's own per-trial resume reads it back from that path on the next run.

The ``harbor`` dependency is optional and imported lazily, so importing this module never requires it.
"""

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem import StoragePath, prefix_join

from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborRuntimeOverlay,
    ValidatedHarborConfig,
    run_harbor_driver,
)
from marin.evaluation.records import RunStatus
from marin.evaluation.runner import EvaluationError, EvaluationOutcome
from marin.evaluation.samples import (
    EvalSample,
    Grading,
    SampleKind,
    archive_samples_table,
    archive_steps_table,
    open_eval_archive,
    sample_to_archive_row,
    trajectory_step_rows,
)
from marin.inference.types import RunningModel

logger = logging.getLogger(__name__)

# Local scratch, used only to materialize a dataset before the isolated driver runs. Trial results
# are written straight to the remote (or local) ``output_dir``, never staged here.
_HARBOR_WORKDIR = Path("/tmp/harbor_workdir")
# Harbor writes its job tree under ``output_dir/harbor_jobs/<job_name>/<trial>/`` as trials finish.
_HARBOR_JOBS_SUBDIR = "harbor_jobs"
# Trials normalize off independent per-trial reads on the remote job tree; fan them out so a
# several-hundred-trial dataset is not a sequential round-trip per trial.
_TRIAL_READ_WORKERS = 16
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
    trial_id: str
    reward: float
    status: str
    trajectory_path: str | None
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


def _jobs_dir(output_dir: str) -> StoragePath:
    """The durable directory Harbor writes its jobs under: ``output_dir/harbor_jobs``."""
    return StoragePath.parse(output_dir) / _HARBOR_JOBS_SUBDIR


def _job_dir(output_dir: str, job_name: str) -> StoragePath:
    """The durable tree for one job: ``output_dir/harbor_jobs/<job_name>`` (Harbor appends the name)."""
    return _jobs_dir(output_dir) / job_name


def _read_trial(result_file: StoragePath) -> HarborTrial:
    """Normalize one finished trial off its ``result.json``, locating its durable trajectory."""
    trial_dir = result_file.parent
    data = json.loads(result_file.read_text())
    task_id = data.get("task_name", trial_dir.name)
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward", 0.0)
    reward = float(reward) if isinstance(reward, int | float) else 0.0
    exc = data.get("exception_info")
    error = {"type": exc.get("exception_type"), "message": exc.get("exception_message")} if exc else None
    trajectory_file = trial_dir / "agent" / "trajectory.json"
    trajectory_path = str(trajectory_file) if trajectory_file.exists() else None
    return HarborTrial(
        task_id=task_id,
        trial_id=trial_dir.name,
        reward=reward,
        status="failed" if exc else "completed",
        trajectory_path=trajectory_path,
        error=error,
    )


def _read_trials(job_dir: StoragePath) -> list[HarborTrial]:
    """Read every finished trial under ``job_dir``, one parallel per-trial read each."""
    result_files = sorted((job_dir / "*/result.json").glob(), key=lambda path: path.parent.name)
    if not result_files:
        return []
    with ThreadPoolExecutor(max_workers=min(_TRIAL_READ_WORKERS, len(result_files))) as pool:
        return list(pool.map(_read_trial, result_files))


def _sample_for(trial: HarborTrial, dataset: str, *, trajectory_uri: str | None) -> EvalSample:
    """Normalize one trial into an agentic :class:`EvalSample`, referencing its archived trajectory."""
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


def _write_archive(trials: list[HarborTrial], dataset: str, output_dir: str) -> str | None:
    """Write the run's samples, flattened steps, and raw trajectories to the finestore archive.

    Each trial's raw trajectory is stored once in the ``blobs`` table and referenced from the sample
    by a ``finestore://`` URI; its steps are flattened into the ``steps`` table for column projection.
    """
    if not trials:
        return None
    store = open_eval_archive(output_dir, writer_id="harbor")
    samples = archive_samples_table(store)
    steps = archive_steps_table(store)
    try:
        for trial in trials:
            trajectory_uri = None
            if trial.trajectory_path is not None:
                raw = StoragePath(trial.trajectory_path).read_bytes()
                trajectory_uri = store.write(
                    f"{trial.trial_id}/trajectory.json",
                    {"task": dataset, "doc_id": trial.task_id, "trial_id": trial.trial_id},
                    raw,
                )
                try:
                    trajectory = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    logger.warning("trial %s has an unparseable trajectory; skipping steps", trial.trial_id)
                else:
                    rows = trajectory_step_rows(trajectory, task=dataset, doc_id=trial.task_id, trial_id=trial.trial_id)
                    if rows:
                        steps.extend(rows)
            sample = _sample_for(trial, dataset, trajectory_uri=trajectory_uri)
            samples.append(sample_to_archive_row(sample, trial_id=trial.trial_id))
        store.seal()
    finally:
        store.close()
    return output_dir


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


def _run_harbor_job(
    *,
    job_name: str,
    config: ValidatedHarborConfig,
    overlay: HarborRuntimeOverlay,
    dataset: str,
    environment: str,
    output_dir: str,
    driver_env: Mapping[str, str],
) -> HarborRunResult:
    job_dir = _job_dir(output_dir, job_name)
    logger.info("starting Harbor job %s (dataset=%s env=%s jobs_dir=%s)", job_name, dataset, environment, job_dir)
    run_harbor_driver(config, overlay, driver_env)

    trials = _read_trials(job_dir)
    samples_path = _write_archive(trials, dataset, output_dir)
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
        workdir = _HARBOR_WORKDIR / job_name
        dataset_path = materialize_harbor_dataset(
            self.config,
            workdir,
            hf_token=hf_token,
        )
        overlay = HarborRuntimeOverlay(
            job_name=job_name,
            jobs_dir=str(_jobs_dir(output_dir)),
            dataset_path=str(dataset_path) if dataset_path is not None else None,
            endpoint_url=model.endpoint.base_url,
            served_model=model.endpoint.model,
            task_limit=self.task_limit,
            model_agent_kwargs=self.model_agent_kwargs,
        )
        return _run_harbor_job(
            job_name=job_name,
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
