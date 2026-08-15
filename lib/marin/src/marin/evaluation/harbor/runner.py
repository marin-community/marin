# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor dataset against an already-served model and normalize the trials.

The group launcher serves a model once and hands this runner an OpenAI endpoint; the runner points a
Harbor agent at it (``hosted_vllm/<served-name>``), runs the dataset's trials on the configured
sandbox environment, and normalizes each finished trial into the shared eval
contract: one agentic :class:`~finestore.eval.EvalSample` per task (its reward, its grading,
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

from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborBackendsUnavailable,
    HarborRuntimeOverlay,
    ValidatedHarborConfig,
    run_harbor_driver,
)
from marin.evaluation.records import RunStatus, TaskCoverage
from marin.evaluation.runner import EvaluationError, EvaluationOutcome
from marin.inference.iris import RemoteInferenceSession
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

# The fraction of attempted trials a verifier must grade for a run to be accepted. An agentic run
# loses occasional trials to agent and verifier timeouts, and discarding an otherwise usable batch
# over one of them throws away the other 239; below this rate the run is not a usable measurement,
# since the ungraded trials could take any value and the resulting interval is too wide to compare.
# The gate is a rate, so it is coarse at small trial counts: one failure in eight is 0.875 and fails.
DEFAULT_MIN_COMPLETION_RATE = 0.9

# Error labels for ungraded trials that carry no exception of their own.
_UNKNOWN_ERROR = "unknown"
_MISSING_RESULT_ERROR = "no_result_written"

_CANONICAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class HarborTrial:
    """One finished Harbor trial, normalized off its ``result.json``.

    ``scored`` is whether the trial is a usable measurement: a verifier graded it and the trial raised
    no exception. A trial whose agent timed out part-way is often still verified, and scores zero
    because it was cut short rather than because the model was wrong -- counting that as a wrong
    answer is the imputation the coverage accounting exists to avoid, so it is an ungraded item.
    """

    task_id: str
    trial_id: str
    reward: float
    scored: bool
    status: str
    trajectory_path: str | None
    error: dict | None


@dataclass(frozen=True)
class HarborRunResult:
    """The aggregate of one Harbor run, and the root of the finestore archive it wrote.

    ``attempted_trials`` is the number of trials the run set out to score, taken from the dataset it
    launched rather than from the trial results it found: a trial that dies before writing a result
    leaves no file behind, so counting results would make the worst-affected runs look complete. It is
    ``None`` when Harbor's job bookkeeping is unreadable, which is unknown rather than complete.
    ``scored_trials`` counts the trials a verifier actually graded, and the rates divide by it -- a
    trial that errored is an ungraded item, not a wrong answer.
    """

    dataset: str
    attempted_trials: int | None
    scored_trials: int
    solved_trials: int
    errors: Mapping[str, int]
    mean_reward: float
    accuracy: float
    archive_path: str | None

    @property
    def failed_trials(self) -> int | None:
        """Attempted trials that produced no verifier grade, or None when the denominator is unknown."""
        if self.attempted_trials is None:
            return None
        return max(0, self.attempted_trials - self.scored_trials)

    @property
    def completion_rate(self) -> float | None:
        """The fraction of attempted trials a verifier graded, or None when that count is unknown."""
        if self.attempted_trials is None:
            return None
        if self.attempted_trials <= 0:
            return 0.0
        return min(1.0, self.scored_trials / self.attempted_trials)

    def task_metrics(self) -> dict[str, dict[str, float]]:
        """Metrics keyed like the evalchemy reader: ``{dataset: {metric: value}}``."""
        metrics = {
            "accuracy": self.accuracy,
            "mean_reward": self.mean_reward,
            "solved": float(self.solved_trials),
            "total": float(self.scored_trials),
        }
        if self.attempted_trials is not None:
            metrics["attempted"] = float(self.attempted_trials)
        return {self.dataset: metrics}

    def task_coverage(self) -> dict[str, TaskCoverage]:
        """Coverage keyed like :meth:`task_metrics`, carrying the per-trial error distribution."""
        return {
            self.dataset: TaskCoverage(
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


def _read_trial(result_file: StoragePath) -> HarborTrial:
    """Normalize one finished trial off its ``result.json``, locating its durable trajectory."""
    trial_dir = result_file.parent
    data = json.loads(result_file.read_text())
    task_id = data.get("task_name", trial_dir.name)
    verifier_result = data.get("verifier_result")
    rewards = (verifier_result or {}).get("rewards") or {}
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
        scored=verifier_result is not None and error is None,
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


def _remove_unscored_trials(job_dir: StoragePath) -> None:
    """Remove incomplete results so Harbor reruns them after a confirmed interruption."""
    for result_file in (job_dir / "*/result.json").glob():
        try:
            result = json.loads(result_file.read_text())
        except json.JSONDecodeError as exc:
            logger.warning(
                "removing unreadable Harbor trial result after inference interruption: %s (%s)", result_file, exc
            )
            result_file.parent.rmtree()
            continue
        if result.get("verifier_result") is None:
            result_file.parent.rmtree()


def _sample_for(trial: HarborTrial, dataset: str, *, trajectory_uri: str | None) -> EvalSample:
    """Normalize one trial into an agentic :class:`EvalSample`, referencing its archived trajectory.

    An ungraded trial is ungraded, not wrong: it carries no score and ``correct`` stays ``None``, so
    the sample browser counts it apart from the answers the model actually got wrong.
    """
    solved = trial.reward >= SOLVED_REWARD if trial.scored else None
    detail = json.dumps({"reward": trial.reward, "error": trial.error, "scored": trial.scored}, ensure_ascii=False)
    return EvalSample(
        task=dataset,
        doc_id=trial.task_id,
        kind=SampleKind.AGENTIC,
        trajectory_uri=trajectory_uri,
        grading=Grading(
            method="harbor:verifier",
            metric="reward",
            score=trial.reward if trial.scored else None,
            passed=solved,
            detail=detail,
        ),
        metrics={"reward": trial.reward} if trial.scored else {},
        correct=solved,
    )


def _write_archive(trials: list[HarborTrial], dataset: str, output_dir: str) -> str | None:
    """Write the run's samples, flattened steps, and raw trajectories to the finestore archive.

    Each trial's raw trajectory is stored once in the ``blobs`` table and referenced from the sample
    by a ``finestore://`` URI; its steps are flattened into the ``steps`` table for column projection.
    Returns the archive root, or ``None`` when there are no trials to write.
    """
    if not trials:
        return None
    store = EvaluationStore.open(output_dir, writer_id="harbor")
    try:
        for trial in trials:
            trajectory_uri = None
            if trial.trajectory_path is not None:
                stored = store.add_trajectory(
                    StoragePath(trial.trajectory_path).read_bytes(),
                    task=dataset,
                    doc_id=trial.task_id,
                    trial_id=trial.trial_id,
                )
                trajectory_uri = stored.uri
            store.add_sample(_sample_for(trial, dataset, trajectory_uri=trajectory_uri), trial_id=trial.trial_id)
        store.seal()
    finally:
        store.close()
    return output_dir


def _trial_errors(trials: list[HarborTrial], attempted: int | None) -> dict[str, int]:
    """The error-type histogram over attempted trials that produced no grade.

    Trials the job never wrote a result for are counted under :data:`_MISSING_RESULT_ERROR`: they are
    the attrition Harbor's own bookkeeping knows about but its result files cannot show. With no
    readable job record there is no such count, so only the errors the trials themselves report appear.
    """
    errors: dict[str, int] = {}
    for trial in trials:
        if trial.scored:
            continue
        name = (trial.error or {}).get("type") or _UNKNOWN_ERROR
        errors[name] = errors.get(name, 0) + 1
    missing = 0 if attempted is None else max(0, attempted - len(trials))
    if missing:
        errors[_MISSING_RESULT_ERROR] = errors.get(_MISSING_RESULT_ERROR, 0) + missing
    return errors


def _aggregate(
    trials: list[HarborTrial], dataset: str, archive_path: str | None, attempted: int | None
) -> HarborRunResult:
    """Aggregate the graded trials, keeping the ungraded ones as coverage rather than as zeros.

    Rates divide by the graded trials. Dividing by every attempted trial, with an ungraded trial read
    back as reward 0.0, would publish the worst case as if it were the estimate; the engine recovers
    that lower bound from the coverage this records.
    """
    scored = [trial for trial in trials if trial.scored]
    solved = sum(1 for trial in scored if trial.reward >= SOLVED_REWARD)
    total_reward = sum(trial.reward for trial in scored)
    return HarborRunResult(
        dataset=dataset,
        attempted_trials=None if attempted is None else max(attempted, len(trials)),
        scored_trials=len(scored),
        solved_trials=solved,
        errors=_trial_errors(trials, attempted),
        mean_reward=(total_reward / len(scored)) if scored else 0.0,
        accuracy=(solved / len(scored)) if scored else 0.0,
        archive_path=archive_path,
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
) -> HarborRunResult:
    job_dir = _job_dir(output_dir, job_name)
    logger.info("starting Harbor job %s (dataset=%s env=%s jobs_dir=%s)", job_name, dataset, environment, job_dir)
    while True:
        try:
            run_harbor_driver(config, overlay, driver_env, inference_session.backend_state)
            break
        except HarborBackendsUnavailable as exc:
            logger.warning("pausing Harbor job %s while inference recovers: %s", job_name, exc)
            _remove_unscored_trials(job_dir)
            inference_session.wait_until_ready()
            logger.info("inference recovered; resuming Harbor job %s", job_name)

    trials = _read_trials(job_dir)
    archive_path = _write_archive(trials, dataset, output_dir)
    result = _aggregate(trials, dataset, archive_path, _attempted_trials(job_dir))
    StoragePath(prefix_join(output_dir, "harbor_result.json")).write_text(
        json.dumps(
            {
                "dataset": result.dataset,
                "attempted_trials": result.attempted_trials,
                "scored_trials": result.scored_trials,
                "solved_trials": result.solved_trials,
                "failed_trials": result.failed_trials,
                "errors": dict(result.errors),
                "mean_reward": result.mean_reward,
                "accuracy": result.accuracy,
            },
            indent=2,
        )
    )
    completion = result.completion_rate
    logger.info(
        "Harbor %s: %d/%d solved of %s attempted (accuracy=%.3f mean_reward=%.3f coverage=%s)",
        dataset,
        result.solved_trials,
        result.scored_trials,
        "an unknown number of" if result.attempted_trials is None else result.attempted_trials,
        result.accuracy,
        result.mean_reward,
        "unknown" if completion is None else f"{completion:.3f}",
    )
    return result


def _evaluation_outcome(
    run: Callable[[], HarborRunResult], output_dir: str, min_completion_rate: float
) -> EvaluationOutcome:
    """Accept a run that graded enough of its trials, and record how much of it happened.

    A run clearing the gate keeps its aggregate and its per-trial error distribution, so a downstream
    reader can tell the model's score apart from the infrastructure quality behind it. A run below the
    gate fails as an infrastructure failure -- agent and verifier timeouts are not evaluation outcomes
    -- and still records its coverage so the rejection is legible as counts rather than as prose.

    A run whose attempted-trial count is unknown has no rate to gate on. It is admitted with its
    coverage left unreported, which downstream widens to "completeness unknown" rather than treating
    it as complete.
    """
    try:
        result = run()
    except Exception as exc:
        raise EvaluationError(str(exc), status=RunStatus.FAILED) from exc
    if not result.scored_trials and not result.attempted_trials:
        raise EvaluationError(
            f"Harbor eval finished with no trials under {output_dir!r}",
            status=RunStatus.FAILED,
        )
    completion = result.completion_rate
    if completion is None:
        logger.warning(
            "Harbor %s graded %d trials but recorded no attempted count; admitting without a "
            "completion gate and reporting coverage as unknown",
            result.dataset,
            result.scored_trials,
        )
    elif completion < min_completion_rate:
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
    return EvaluationOutcome(metrics=result.task_metrics(), coverage=result.task_coverage())


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
    """The fraction of attempted trials a verifier must grade for the run to be accepted."""

    def _run(
        self,
        model: RunningModel,
        output_dir: str,
        hf_token: str | None,
        driver_env: Mapping[str, str],
        inference_session: RemoteInferenceSession,
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
            inference_session=inference_session,
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
            lambda: self._run(session.model, output_dir, hf_token, driver_env, session),
            output_dir,
            self.min_completion_rate,
        )
