# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write Harbor trial outputs directly into a FineStore evaluation archive.

Harbor's job and trial directories remain operational state for resume, inspection, and retry.
FineStore is the configured evaluation-output sink: this module writes the typed result received from
the trial lifecycle, preserves trajectory and job artifacts, and appends normalized rows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from upath import UPath

from harbor.models.job.paths import (
    JOB_CONFIG_FILENAME,
    JOB_LOCK_FILENAME,
    JOB_RESULT_FILENAME,
)
from harbor.models.trajectories import ContentPart, Trajectory
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor_config.errors import ErrorCategory, error_category
from harbor_config.models.job.config import ArchiveConfig

try:
    from finestore.eval import (
        EvalSample,
        EvaluationStore,
        Grading,
        SampleKind,
        StepRecord,
    )

    _FINESTORE_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _FINESTORE_IMPORT_ERROR = error

logger = logging.getLogger(__name__)

SOLVED_REWARD = 0.99
_JSON_CONTENT_TYPE = "application/json"
_JOB_SOURCE_NAMES = (JOB_CONFIG_FILENAME, JOB_LOCK_FILENAME, JOB_RESULT_FILENAME)


def _numeric_rewards(result: TrialResult) -> dict[str, float]:
    rewards = result.verifier_result.rewards if result.verifier_result else None
    return {
        name: float(value)
        for name, value in (rewards or {}).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _is_scored(result: TrialResult) -> bool:
    exception = result.exception_info
    if exception is None:
        return result.verifier_result is not None
    category = error_category(exception.exception_type)
    if category is ErrorCategory.AGENT:
        return True
    if category is ErrorCategory.PASSTHROUGH:
        return result.verifier_result is not None
    return False


def _trial_sample(
    result: TrialResult, dataset: str, *, trajectory_uri: str | None
) -> EvalSample:
    """Normalize one finished Harbor trial into an agentic sample."""
    rewards = _numeric_rewards(result)
    reward = rewards.get("reward", 0.0)
    scored = _is_scored(result)
    solved = reward >= SOLVED_REWARD if scored else None
    exception = result.exception_info
    error = (
        {
            "type": exception.exception_type,
            "message": exception.exception_message,
            "category": error_category(exception.exception_type).value,
        }
        if exception
        else None
    )
    return EvalSample(
        task=dataset,
        doc_id=result.task_name,
        kind=SampleKind.AGENTIC,
        trajectory_uri=trajectory_uri,
        grading=Grading(
            method="harbor:verifier",
            metric="reward",
            score=reward if scored else None,
            passed=solved,
            detail=json.dumps(
                {"rewards": rewards, "error": error, "scored": scored},
                ensure_ascii=False,
            ),
        ),
        metrics=rewards if scored else {},
        correct=solved,
    )


def _content_to_text(value: str | list[ContentPart]) -> str:
    if isinstance(value, str):
        return value.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return json.dumps(
        [part.model_dump(mode="json") for part in value], ensure_ascii=True
    )


def _trajectory_step_rows(
    trajectory: Trajectory, *, task: str, doc_id: str, trial_id: str
) -> list[StepRecord]:
    """Convert a Harbor trajectory's top-level steps to normalized rows."""
    records = []
    for step in trajectory.steps:
        metrics = step.metrics
        records.append(
            StepRecord(
                task=task,
                doc_id=doc_id,
                trial_id=trial_id,
                step_id=step.step_id,
                source=step.source,
                model_name=step.model_name,
                message=_content_to_text(step.message),
                reasoning_content=_content_to_text(step.reasoning_content)
                if step.reasoning_content is not None
                else None,
                tool_calls_json=json.dumps(
                    [
                        tool_call.model_dump(mode="json")
                        for tool_call in step.tool_calls
                    ],
                    ensure_ascii=True,
                )
                if step.tool_calls is not None
                else None,
                observation_json=json.dumps(
                    step.observation.model_dump(mode="json"), ensure_ascii=True
                )
                if step.observation is not None
                else None,
                prompt_tokens=metrics.prompt_tokens if metrics is not None else None,
                completion_tokens=metrics.completion_tokens
                if metrics is not None
                else None,
                cost_usd=metrics.cost_usd if metrics is not None else None,
                prompt_token_ids=metrics.prompt_token_ids
                if metrics is not None
                else None,
                completion_token_ids=metrics.completion_token_ids
                if metrics is not None
                else None,
                logprobs=metrics.logprobs if metrics is not None else None,
            )
        )
    return records


class TrialArchive:
    """A job-scoped writer that defers ungraded attempts until the final result set."""

    def __init__(
        self,
        store: EvaluationStore,
        dataset: str,
        trials_dir: Path | UPath | None,
    ) -> None:
        self._store = store
        self._dataset = dataset
        self._trials_dir = trials_dir
        self._written_trial_ids: set[str] = set()
        self._errors: list[Exception] = []
        self._lock = threading.Lock()

    async def on_trial_end(self, event: TrialHookEvent) -> None:
        """Archive one scored trial result at the end of its run."""
        if (
            event.event != TrialEvent.END
            or event.result is None
            or not _is_scored(event.result)
        ):
            return
        await asyncio.to_thread(self._write_result_safely, event.result)

    async def finalize(self, trial_results: list[TrialResult]) -> None:
        """Complete the archive from the final trial set and job metadata."""
        await asyncio.to_thread(self._write_final_results, trial_results)
        await asyncio.to_thread(self._write_job_sources_safely)
        with self._lock:
            errors = list(self._errors)
        if errors:
            raise ExceptionGroup("FineStore evaluation archive writes failed", errors)

    def seal(self) -> None:
        self._store.seal()

    def close(self) -> None:
        self._store.close()

    def _write_final_results(self, trial_results: list[TrialResult]) -> None:
        for result in trial_results:
            with self._lock:
                written = result.trial_name in self._written_trial_ids
            if not written:
                self._write_result_safely(result)

    def _write_result_safely(self, result: TrialResult) -> None:
        try:
            self._write_result(result)
        except Exception as error:
            self._record_error(error)
            logger.warning(
                "FineStore archive write failed for trial %s; the job will fail before sealing",
                result.trial_name,
                exc_info=True,
            )

    def _write_job_sources_safely(self) -> None:
        try:
            self._write_job_sources()
        except Exception as error:
            self._record_error(error)
            logger.warning(
                "FineStore archive write failed for Harbor job metadata; the job will fail before sealing",
                exc_info=True,
            )

    def _record_error(self, error: Exception) -> None:
        with self._lock:
            self._errors.append(error)

    def _write_job_sources(self) -> None:
        if self._trials_dir is None:
            return
        for name in _JOB_SOURCE_NAMES:
            path = self._trials_dir / name
            if path.exists():
                self._store.add_source_artifact(
                    f"harbor/job/{name}",
                    path.read_bytes(),
                    content_type=_JSON_CONTENT_TYPE,
                )

    def _write_result(self, result: TrialResult) -> None:
        trajectory_uri = None
        trial_dir = result.resolve_trial_path(self._trials_dir)
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        if trajectory_path.exists():
            raw = trajectory_path.read_bytes()
            # Preserve the original JSON artifact, including surrogate escapes;
            # normalized Arrow text needs a UTF-8-safe representation.
            trajectory = Trajectory.model_validate(json.loads(raw))
            trajectory_uri = self._store.add_artifact(
                f"{result.trial_name}/trajectory.json",
                raw,
                metadata={
                    "task": self._dataset,
                    "doc_id": result.task_name,
                    "trial_id": result.trial_name,
                },
            )
            self._store.add_steps(
                _trajectory_step_rows(
                    trajectory,
                    task=self._dataset,
                    doc_id=result.task_name,
                    trial_id=result.trial_name,
                )
            )

        self._store.add_source_artifact(
            f"harbor/{result.trial_name}/result.json",
            result.model_dump_json(indent=4).encode(),
            content_type=_JSON_CONTENT_TYPE,
        )
        self._store.add_sample(
            _trial_sample(result, self._dataset, trajectory_uri=trajectory_uri),
            trial_id=result.trial_name,
        )
        with self._lock:
            self._written_trial_ids.add(result.trial_name)


_PROCESS_ARCHIVE: TrialArchive | None = None


def _initialize_archive_process(
    config: ArchiveConfig, trials_dir: Path | UPath | None
) -> None:
    global _PROCESS_ARCHIVE
    store = EvaluationStore.open(config.root, writer_id=config.writer_id)
    _PROCESS_ARCHIVE = TrialArchive(store, config.dataset, trials_dir)


def _archive_process_ready() -> None:
    if _PROCESS_ARCHIVE is None:
        raise RuntimeError("Archive process was not initialized")


def _write_process_result(result: TrialResult) -> None:
    if _PROCESS_ARCHIVE is None:
        raise RuntimeError("Archive process was not initialized")
    _PROCESS_ARCHIVE._write_result_safely(result)


def _finalize_process_archive(results: list[TrialResult]) -> None:
    if _PROCESS_ARCHIVE is None:
        raise RuntimeError("Archive process was not initialized")
    asyncio.run(_PROCESS_ARCHIVE.finalize(results))


def _seal_process_archive() -> None:
    if _PROCESS_ARCHIVE is None:
        raise RuntimeError("Archive process was not initialized")
    _PROCESS_ARCHIVE.seal()


def _close_process_archive() -> None:
    if _PROCESS_ARCHIVE is None:
        raise RuntimeError("Archive process was not initialized")
    _PROCESS_ARCHIVE.close()


class ProcessTrialArchive:
    """Stream results through a writer isolated from sandbox SDK environment changes."""

    def __init__(self, config: ArchiveConfig, trials_dir: Path | UPath | None) -> None:
        # Daytona's image uploader temporarily clears os.environ. Spawn before
        # any trials start so storage clients always inherit the original env.
        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
            initializer=_initialize_archive_process,
            initargs=(config, trials_dir),
        )
        try:
            self._executor.submit(_archive_process_ready).result()
        except BaseException:
            self._executor.shutdown()
            raise

    async def on_trial_end(self, event: TrialHookEvent) -> None:
        if (
            event.event != TrialEvent.END
            or event.result is None
            or not _is_scored(event.result)
        ):
            return
        await asyncio.wrap_future(
            self._executor.submit(_write_process_result, event.result)
        )

    async def finalize(self, trial_results: list[TrialResult]) -> None:
        await asyncio.wrap_future(
            self._executor.submit(_finalize_process_archive, trial_results)
        )

    def seal(self) -> None:
        self._executor.submit(_seal_process_archive).result()

    def close(self) -> None:
        try:
            self._executor.submit(_close_process_archive).result()
        finally:
            self._executor.shutdown()


def open_archive(
    config: ArchiveConfig | None,
    *,
    trials_dir: Path | UPath | None = None,
) -> ProcessTrialArchive | None:
    """Open the configured job archive, or return ``None`` when archiving is disabled."""
    if config is None:
        return None
    if _FINESTORE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "JobConfig.archive requires the harbor[archive] extra"
        ) from _FINESTORE_IMPORT_ERROR
    return ProcessTrialArchive(config, trials_dir)
