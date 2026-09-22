# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor dataset against an already-served model and normalize the trials.

The group launcher serves a model once and hands this runner an OpenAI endpoint; the runner points a
Harbor agent at it (``hosted_vllm/<served-name>``) and runs the dataset's trials on the configured
sandbox environment. Harbor writes each native result and normalized evaluation sample directly to
the run's FineStore archive. This module reads Harbor's durable trial results only to compute the
aggregate record metrics and coverage. Harbor also keeps its trial tree under the results root, so a
completed trial survives a driver killed before the job returns and Harbor's own per-trial resume
reads it back on the next run.

The ``harbor`` dependency is optional and imported lazily, so importing this module never requires it.
"""

import hashlib
import json
import logging
import math
import re
import statistics
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.harbor.dataset import local_harbor_dataset_path
from marin.evaluation.harbor.driver_config import (
    HarborBackendsUnavailable,
    HarborErrorTaxonomy,
    HarborRuntimeOverlay,
    ValidatedHarborConfig,
    run_harbor_driver,
)
from marin.evaluation.records import BenchmarkMetadataRef, EvalTaskRef, RunStatus, TaskCoverage
from marin.evaluation.rollouts import normalize_rollouts
from marin.evaluation.runner import EvaluationError, EvaluationOutcome
from marin.inference.iris import RemoteInferenceSession
from marin.inference.types import RunningModel

logger = logging.getLogger(__name__)

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

# Fraction of attempted trials that must be scoreable under the Harbor taxonomy. Agent failures
# remain scoreable, while infrastructure failures and passthrough failures without verifier results
# reduce completion. Below this rate, the ungraded trials make the result too uncertain to compare.
# The rate is coarse for small batches: one unscored trial in eight yields 0.875 and fails.
DEFAULT_MIN_COMPLETION_RATE = 0.9

# Error labels for ungraded trials that carry no exception of their own.
_UNKNOWN_ERROR = "unknown"
_MISSING_RESULT_ERROR = "no_result_written"
_TRIAL_RESULT_GLOB = "*/result.json"

_UNKNOWN_ERROR_PREFIX = "unknown:"

_CANONICAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class HarborTrial:
    """One finished Harbor trial, normalized off its ``result.json``.

    ``scored`` follows the taxonomy from the pinned Harbor environment. Agent failures are model
    outcomes even without a verifier result. Passthrough failures require a verifier result, and
    infrastructure or unknown failures remain ungraded.
    """

    reward: float
    scored: bool
    error: dict | None


@dataclass(frozen=True)
class HarborRunResult:
    """The aggregate of one Harbor run.

    ``attempted_trials`` is the number of trials the run set out to score after the runtime cap,
    derived from the dataset size captured during preflight rather than from result files.
    ``scored_trials`` counts outcomes accepted by the Harbor taxonomy, including agent failures
    without verifier results. Mean reward divides by this count.
    """

    dataset: str
    benchmark_trials: int
    attempted_trials: int
    scored_trials: int
    solved_trials: int
    errors: Mapping[str, int]
    mean_reward: float
    reward_stderr: float
    benchmark: BenchmarkMetadataRef

    @property
    def unscored_trials(self) -> int:
        """Attempted trials that did not produce a score-bearing outcome."""
        return max(0, self.attempted_trials - self.scored_trials)

    @property
    def completion_rate(self) -> float:
        """The fraction of attempted trials with score-bearing outcomes."""
        if self.attempted_trials <= 0:
            return 0.0
        return min(1.0, self.scored_trials / self.attempted_trials)

    def task_metrics(self) -> dict[str, dict[str, float]]:
        """Metrics keyed like the evalchemy reader: ``{dataset: {metric: value}}``."""
        metrics = {
            "mean_reward": self.mean_reward,
            "solved": float(self.solved_trials),
            "total": float(self.scored_trials),
        }
        metrics["attempted"] = float(self.attempted_trials)
        return {self.dataset: metrics}

    def canonical_task_metrics(self) -> dict[str, dict[str, float]]:
        """Canonical reward and its standard error, keyed by dataset."""
        return {self.dataset: {"reward": self.mean_reward, "reward_stderr": self.reward_stderr}}

    def task_coverage(self) -> dict[str, TaskCoverage]:
        """Coverage keyed like :meth:`task_metrics`, carrying the per-trial error distribution."""
        return {
            self.dataset: TaskCoverage(
                n_benchmark=self.benchmark_trials,
                n_attempted=self.attempted_trials,
                n_scored=self.scored_trials,
                errors=dict(self.errors),
            )
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


def _read_trial(result_file: StoragePath, taxonomy: HarborErrorTaxonomy) -> HarborTrial:
    """Normalize one Harbor result for aggregate scoring and coverage."""
    data = json.loads(result_file.read_text())
    verifier_result = data.get("verifier_result")
    rewards = (verifier_result or {}).get("rewards") or {}
    reward = rewards.get("reward", 0.0)
    reward = float(reward) if isinstance(reward, int | float) else 0.0
    exc = data.get("exception_info")
    exception_type = exc.get("exception_type") if exc else None
    error = {"type": exception_type, "message": exc.get("exception_message")} if exc else None
    if error is None:
        scored = verifier_result is not None
    elif exception_type in taxonomy.agent:
        scored = True
    elif exception_type in taxonomy.passthrough:
        scored = verifier_result is not None
    elif exception_type in taxonomy.infrastructure:
        scored = False
    elif exception_type in taxonomy.undecided:
        scored = False
    else:
        error["type"] = f"{_UNKNOWN_ERROR_PREFIX}{exception_type or _UNKNOWN_ERROR}"
        scored = False
    return HarborTrial(
        reward=reward,
        scored=scored,
        error=error,
    )


def _read_trials(job_dir: StoragePath, taxonomy: HarborErrorTaxonomy) -> list[HarborTrial]:
    result_files = sorted((job_dir / _TRIAL_RESULT_GLOB).glob(), key=lambda path: path.parent.name)
    if not result_files:
        return []
    with ThreadPoolExecutor(max_workers=min(_TRIAL_READ_WORKERS, len(result_files))) as pool:
        return list(pool.map(lambda result_file: _read_trial(result_file, taxonomy), result_files))


def _attempted_trials(job_dir: StoragePath) -> int | None:
    """The number of trials the job set out to run, from Harbor's own job-level bookkeeping.

    Harbor writes ``result.json`` (carrying ``n_total_trials``) and ``lock.json`` (carrying the
    resolved trial list) at the job root. Counting per-trial result files instead would miss every
    trial that died before writing one, so a run whose worker was preempted mid-dataset would report
    perfect coverage -- which is why an unreadable job record yields None rather than the number of
    results found: the count is unknown, and the runs where it is unknown are exactly the interrupted
    ones a found-count would certify as complete.
    """
    for name, key in (("result.json", "n_total_trials"), ("lock.json", "trials")):
        path = job_dir / name
        try:
            value = json.loads(path.read_text()).get(key)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, list) and value:
            return len(value)
    logger.warning("Harbor job %s has no readable job-level trial count; coverage is unknown", job_dir)
    return None


def _job_benchmark(job_dir: StoragePath) -> BenchmarkMetadataRef:
    """Read the evaluator-owned benchmark descriptor from Harbor's job result."""
    data = json.loads((job_dir / "result.json").read_text())
    descriptions = data.get("benchmark_metadata") or []
    if not isinstance(descriptions, list) or len(descriptions) != 1:
        raise ValueError(f"Harbor job {job_dir} did not record exactly one benchmark descriptor")
    return BenchmarkMetadataRef.model_validate(descriptions[0])


def _remove_unscored_trials(job_dir: StoragePath, taxonomy: HarborErrorTaxonomy) -> None:
    """Remove results Harbor should retry after a confirmed inference interruption."""
    for result_file in (job_dir / _TRIAL_RESULT_GLOB).glob():
        try:
            trial = _read_trial(result_file, taxonomy)
        except json.JSONDecodeError as exc:
            logger.warning(
                "removing unreadable Harbor trial result after inference interruption: %s (%s)", result_file, exc
            )
            result_file.parent.rmtree()
            continue
        error_type = (trial.error or {}).get("type", "")
        if not trial.scored and not error_type.startswith(_UNKNOWN_ERROR_PREFIX):
            result_file.parent.rmtree()


def _trial_errors(trials: list[HarborTrial], attempted: int) -> dict[str, int]:
    """Count trial errors, including errors on scored outcomes.

    Trials the job never wrote a result for are counted under :data:`_MISSING_RESULT_ERROR`.
    """
    errors: dict[str, int] = {}
    for trial in trials:
        if trial.scored and trial.error is None:
            continue
        name = (trial.error or {}).get("type") or _UNKNOWN_ERROR
        errors[name] = errors.get(name, 0) + 1
    missing = max(0, attempted - len(trials))
    if missing:
        errors[_MISSING_RESULT_ERROR] = errors.get(_MISSING_RESULT_ERROR, 0) + missing
    return errors


def _aggregate(
    trials: list[HarborTrial],
    dataset: str,
    benchmark: BenchmarkMetadataRef,
    trials_per_task: int,
) -> HarborRunResult:
    """Aggregate the graded trials, keeping the ungraded ones as coverage rather than as zeros.

    Rates divide by the graded trials. Dividing by every attempted trial, with an ungraded trial read
    back as reward 0.0, would publish the worst case as if it were the estimate; the engine recovers
    that lower bound from the coverage this records.
    """
    if benchmark.n_benchmark is None or benchmark.n_attempted is None:
        raise ValueError("Harbor benchmark metadata must report task counts")
    n_benchmark = benchmark.n_benchmark * trials_per_task
    attempted = benchmark.n_attempted * trials_per_task
    scored = [trial for trial in trials if trial.scored]
    if len(scored) > attempted:
        raise ValueError(f"Harbor scored {len(scored)} trials but intended only {attempted}")
    solved = sum(1 for trial in scored if trial.reward >= SOLVED_REWARD)
    total_reward = sum(trial.reward for trial in scored)
    reward_stderr = (
        statistics.stdev(trial.reward for trial in scored) / math.sqrt(len(scored)) if len(scored) > 1 else 0.0
    )
    return HarborRunResult(
        dataset=dataset,
        benchmark_trials=n_benchmark,
        attempted_trials=attempted,
        scored_trials=len(scored),
        solved_trials=solved,
        errors=_trial_errors(trials, attempted),
        mean_reward=(total_reward / len(scored)) if scored else 0.0,
        reward_stderr=reward_stderr,
        benchmark=benchmark,
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
    inference_session: RemoteInferenceSession,
    benchmark: BenchmarkMetadataRef,
    trials_per_task: int,
) -> HarborRunResult:
    job_dir = _job_dir(output_dir, job_name)
    logger.info("starting Harbor job %s (dataset=%s env=%s jobs_dir=%s)", job_name, dataset, environment, job_dir)
    while True:
        try:
            run_harbor_driver(config, overlay, driver_env, inference_session.backend_state)
            break
        except HarborBackendsUnavailable as exc:
            logger.warning("pausing Harbor job %s while inference recovers: %s", job_name, exc)
            _remove_unscored_trials(job_dir, config.error_taxonomy)
            inference_session.wait_until_ready()
            logger.info("inference recovered; resuming Harbor job %s", job_name)

    normalize_rollouts(output_dir, writer_id=f"marin-harbor-rollouts-{job_name}")
    trials = _read_trials(job_dir, config.error_taxonomy)
    recorded_attempted = _attempted_trials(job_dir)
    recorded_benchmark = _job_benchmark(job_dir)
    if recorded_benchmark != benchmark:
        raise ValueError("Harbor job benchmark metadata differs from preflight")
    if benchmark.n_attempted is None:
        raise ValueError("Harbor benchmark metadata did not report an attempted task count")
    n_attempted = benchmark.n_attempted * trials_per_task
    if recorded_attempted is not None and recorded_attempted > n_attempted:
        raise ValueError(f"Harbor recorded {recorded_attempted} trials but intended only {n_attempted}")
    result = _aggregate(trials, dataset, recorded_benchmark, trials_per_task)
    StoragePath(prefix_join(output_dir, "harbor_result.json")).write_text(
        json.dumps(
            {
                "dataset": result.dataset,
                "benchmark_trials": result.benchmark_trials,
                "attempted_trials": result.attempted_trials,
                "scored_trials": result.scored_trials,
                "solved_trials": result.solved_trials,
                "unscored_trials": result.unscored_trials,
                "errors": dict(result.errors),
                "mean_reward": result.mean_reward,
                "reward_stderr": result.reward_stderr,
                "benchmark_metadata": result.benchmark.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    completion = result.completion_rate
    logger.info(
        "Harbor %s: %d/%d solved of %s attempted (mean_reward=%.3f coverage=%s)",
        dataset,
        result.solved_trials,
        result.scored_trials,
        result.attempted_trials,
        result.mean_reward,
        f"{completion:.3f}",
    )
    return result


def _evaluation_outcome(
    run: Callable[[], HarborRunResult], output_dir: str, min_completion_rate: float
) -> EvaluationOutcome:
    """Accept a run that graded enough of its trials, and record how much of it happened.

    A run clearing the gate keeps its aggregate and its per-trial error distribution, so a downstream
    reader can tell the model's score apart from the infrastructure quality behind it. A run below the
    gate fails as an infrastructure failure and still records its coverage so the rejection is legible
    as counts rather than as prose. Unknown exception names reject the run as taxonomy drift.

    """
    try:
        result = run()
    except Exception as exc:
        raise EvaluationError(str(exc), status=RunStatus.FAILED) from exc
    unknown_errors = {name: count for name, count in result.errors.items() if name.startswith(_UNKNOWN_ERROR_PREFIX)}
    if unknown_errors:
        raise EvaluationError(
            f"Harbor eval encountered error names absent from its taxonomy under {output_dir!r}: "
            f"{_error_summary(unknown_errors)}",
            status=RunStatus.INFRA_FAILED,
            coverage=result.task_coverage(),
        )
    if not result.scored_trials and not result.attempted_trials:
        raise EvaluationError(
            f"Harbor eval finished with no trials under {output_dir!r}",
            status=RunStatus.FAILED,
        )
    completion = result.completion_rate
    if completion < min_completion_rate:
        raise EvaluationError(
            f"Harbor eval graded {result.scored_trials} of {result.attempted_trials} trials "
            f"({completion:.1%}), below the {min_completion_rate:.0%} gate, under "
            f"{output_dir!r}: {_error_summary(result.errors)}",
            status=RunStatus.INFRA_FAILED,
            coverage=result.task_coverage(),
        )
    elif result.errors:
        logger.warning(
            "Harbor %s admitted at %.1f%% trial completion: %s",
            result.dataset,
            completion * 100,
            _error_summary(result.errors),
        )
    return EvaluationOutcome(
        metrics=result.task_metrics(),
        canonical_metrics=result.canonical_task_metrics(),
        tasks=(EvalTaskRef(name=result.dataset, num_fewshot=None, benchmark=result.benchmark),),
        coverage=result.task_coverage(),
    )


def _error_summary(errors: Mapping[str, int]) -> str:
    """The trial-error histogram as a stable, readable string."""
    return ", ".join(f"{name}={count}" for name, count in sorted(errors.items())) or "no recorded errors"


@dataclass(frozen=True)
class HarborExecutor:
    """Run one normalized Harbor job policy against a served model."""

    config: ValidatedHarborConfig
    task_limit: int | None
    model_agent_kwargs: Mapping[str, object]
    secret_env_keys: tuple[str, ...] = ()
    min_completion_rate: float = DEFAULT_MIN_COMPLETION_RATE
    """Minimum scoreable fraction of attempted trials for accepting the run."""

    def _run(
        self,
        model: RunningModel,
        output_dir: str,
        driver_env: Mapping[str, str],
        inference_session: RemoteInferenceSession,
    ) -> HarborRunResult:
        dataset = self.config.record_dataset
        job_name = _job_name(
            dataset,
            (self.config.digest, model.endpoint.model, self.task_limit),
        )
        dataset_path = local_harbor_dataset_path(self.config)
        overlay = HarborRuntimeOverlay(
            job_name=job_name,
            jobs_dir=str(_jobs_dir(output_dir)),
            dataset_path=str(dataset_path) if dataset_path is not None else None,
            endpoint_url=model.endpoint.base_url,
            served_model=model.endpoint.model,
            task_limit=self.task_limit,
            model_agent_kwargs=self.model_agent_kwargs,
            archive_root=output_dir,
            archive_dataset=dataset,
        )
        benchmark = self.config.benchmark_for(
            self.task_limit,
            Path(dataset_path).name if dataset_path is not None else None,
        )
        return _run_harbor_job(
            job_name=job_name,
            config=self.config,
            overlay=overlay,
            dataset=dataset,
            environment=self.config.environment,
            output_dir=output_dir,
            driver_env=driver_env,
            inference_session=inference_session,
            benchmark=benchmark,
            trials_per_task=self.config.trials_per_task,
        )

    def __call__(
        self,
        session: RemoteInferenceSession,
        output_dir: str,
        env_vars: Mapping[str, str],
    ) -> EvaluationOutcome:
        """Run Harbor while supervising a managed inference dependency."""
        driver_env = {key: env_vars[key] for key in self.secret_env_keys}
        hf_token = env_vars.get("HF_TOKEN")
        if hf_token:
            driver_env["HF_TOKEN"] = hf_token
        return _evaluation_outcome(
            lambda: self._run(session.model, output_dir, driver_env, session),
            output_dir,
            self.min_completion_rate,
        )
