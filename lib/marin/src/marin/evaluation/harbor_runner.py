# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor registry dataset against an already-served model and normalize the trials.

The group launcher serves a model once and hands this runner an OpenAI endpoint; the runner points a
Harbor agent at it (``hosted_vllm/<served-name>``), runs the dataset's trials on the configured
sandbox environment (Daytona by default), and normalizes each finished trial into the shared eval
contract: one agentic :class:`~marin.evaluation.samples.EvalSample` per task (its reward, its grading,
and a reference to the saved trajectory) plus an aggregate this module's :class:`HarborResult` reads
back for the record's metrics. Harbor's own per-trial resume is preserved by restoring completed
trials from the durable output path before the job runs.

The ``harbor`` dependency is optional and imported lazily, so importing this module never requires it.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fsspec.core import url_to_fs
from rigging.filesystem import StoragePath, is_remote_path, prefix_join

from marin.evaluation.config_artifacts import HARBOR_CONFIG
from marin.evaluation.harbor_config import HarborRuntimeBinding, ResolvedHarborPolicy
from marin.evaluation.samples import EvalSample, Grading, SampleKind, write_sample_parquet
from marin.evaluation.utils import download_from_gcs, upload_to_gcs

logger = logging.getLogger(__name__)

# Harbor is run as an external tool in an isolated uv environment (its Daytona SDK carries pre-release
# pins that do not fit the marin lock). These specs pin what that ephemeral env installs.
_HARBOR_SPEC = HARBOR_CONFIG.source_requirement
_DAYTONA_SPEC = "daytona>=0.200.1"
_DRIVER = str(Path(__file__).with_name("harbor_trial_driver.py"))

# The reward at or above which a Harbor trial counts as solved (rewards are typically 0.0 / 1.0; the
# margin tolerates float noise).
SOLVED_REWARD = 0.99

_CANONICAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class HarborEndpoint:
    """The only Harbor runner input unavailable before the endpoint is live."""

    served_model_name: str
    api_base: str


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


def _job_name(policy: ResolvedHarborPolicy, endpoint: HarborEndpoint) -> str:
    """A deterministic Harbor job name so a re-run resumes the previous job's completed trials."""
    key = f"{policy.sha256}|{endpoint.served_model_name}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", policy.dataset)[:32]
    return f"harbor_{safe}_{digest}"


def _restore_completed_trials(out_path: str, job_dir: Path) -> int:
    """Download completed trials (those with ``result.json``) from ``out_path`` so Harbor skips them."""
    trials_root = prefix_join(out_path, "harbor_trials")
    if not StoragePath(trials_root).exists():
        return 0
    restored = 0
    for result_file in StoragePath(prefix_join(trials_root, "*/result.json")).glob():
        trial_dir = os.path.dirname(str(result_file))
        local = job_dir / os.path.basename(trial_dir)
        if (local / "result.json").exists():
            continue
        download_from_gcs(trial_dir, str(local))
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
    dest = prefix_join(out_path, f"samples_{dataset}_harbor.parquet")
    fs, _ = url_to_fs(dest)
    write_sample_parquet(fs, dest, samples)
    return dest


def _aggregate(trials: list[HarborTrial], dataset: str, samples_path: str | None) -> HarborRunResult:
    total = len(trials)
    solved = sum(1 for trial in trials if trial.reward >= SOLVED_REWARD)
    total_reward = sum(trial.reward for trial in trials)
    return HarborRunResult(
        dataset=dataset,
        total_trials=total,
        solved_trials=solved,
        mean_reward=(total_reward / total) if total else 0.0,
        accuracy=(solved / total) if total else 0.0,
        samples_path=samples_path,
    )


def _run_driver(config_file: Path) -> None:
    """Run the Harbor trial driver in an isolated uv env (Harbor + Daytona, no marin project)."""
    cmd = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--prerelease=allow",
        "--with",
        _HARBOR_SPEC,
        "--with",
        _DAYTONA_SPEC,
        "python",
        _DRIVER,
        str(config_file),
    ]
    logger.info("running Harbor driver: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _upload_trials(job_dir: Path, out_path: str) -> None:
    """Upload each finished trial directory under ``job_dir`` to ``out_path/harbor_trials`` for resume."""
    for trial_dir in (d for d in job_dir.iterdir() if d.is_dir()):
        if (trial_dir / "result.json").exists():
            upload_to_gcs(str(trial_dir), prefix_join(out_path, f"harbor_trials/{trial_dir.name}"))


def run_harbor_eval(policy: ResolvedHarborPolicy, endpoint: HarborEndpoint, out_path: str) -> HarborRunResult:
    """Run one complete Harbor policy against a live endpoint and write normalized outputs.

    Policy fields never flow through this runner one by one. It binds only the endpoint and generated
    job location into an already resolved Harbor document, then executes that document in Harbor's
    isolated runtime.
    """
    job_name = _job_name(policy, endpoint)
    workdir = Path("/tmp/harbor_workdir") / job_name
    output_dir = workdir / "harbor_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = output_dir / job_name

    if is_remote_path(out_path):
        restored = _restore_completed_trials(out_path, job_dir)
        if restored:
            logger.info("restored %d completed Harbor trial(s) from %s", restored, out_path)

    resolved = policy.bind(
        HarborRuntimeBinding(
            job_name=job_name,
            jobs_dir=str(output_dir),
            served_model_name=endpoint.served_model_name,
            api_base=endpoint.api_base,
        )
    )
    driver_config = {"job_config": resolved.document}
    config_file = workdir / "driver_config.json"
    config_file.write_text(json.dumps(driver_config))
    config_file.chmod(0o600)
    StoragePath(prefix_join(out_path, "harbor_config.json")).write_text(
        json.dumps(resolved.persisted_metadata(), indent=2, sort_keys=True)
    )

    logger.info("starting Harbor job %s (dataset=%s env=%s)", job_name, policy.dataset, policy.environment)
    _run_driver(config_file)

    trials = _read_trials(job_dir)
    if is_remote_path(out_path):
        _upload_trials(job_dir, out_path)
    samples_path = _write_samples(trials, policy.dataset, out_path)
    result = _aggregate(trials, policy.dataset, samples_path)
    StoragePath(prefix_join(out_path, "harbor_result.json")).write_text(
        json.dumps(
            {
                "dataset": result.dataset,
                "total_trials": result.total_trials,
                "solved_trials": result.solved_trials,
                "mean_reward": result.mean_reward,
                "accuracy": result.accuracy,
            },
            indent=2,
        )
    )
    logger.info(
        "Harbor %s: %d/%d solved (accuracy=%.3f mean_reward=%.3f)",
        policy.dataset,
        result.solved_trials,
        result.total_trials,
        result.accuracy,
        result.mean_reward,
    )
    return result
