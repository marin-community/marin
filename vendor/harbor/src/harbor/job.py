import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from uuid import uuid4

from upath import UPath

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from harbor.metrics.base import BaseMetric
from harbor.metrics.factory import MetricFactory
from harbor.metrics.mean import Mean
from harbor.models.benchmark import BenchmarkMetadata, reward_benchmark_metadata
from harbor.models.dataset.paths import DatasetPaths
from harbor.environments.factory import EnvironmentFactory
from harbor.models.job.config import (
    DatasetConfig,
    JobConfig,
)
from harbor.models.job.lock import JobLock, build_job_lock
from harbor.models.job.paths import (
    JOB_CONFIG_FILENAME,
    JOB_LOCK_FILENAME,
    JOB_RESULT_FILENAME,
)
from harbor.models.job.reconciliation import (
    TrialPlanMismatchError,
    classify_trial_plan_mismatches,
    trial_identity_key,
)
from harbor.models.job.result import (
    ADHOC_DATASET_NAME,
    EvalsRewardsMap,
    JobResult,
    JobStats,
)
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from harbor.registry.client import RegistryClientFactory
from harbor.archive import open_archive
from harbor.telemetry.observer import (
    bind_telemetry,
    record_job_terminal,
    shutdown_telemetry,
    start_telemetry,
    trial_hook,
)
from harbor.tasks.client import TaskClient, TaskDownloadResult, TaskIdType
from harbor.trial.hooks import HookCallback, TrialEvent, TrialHookEvent
from harbor.trial.queue import TrialQueue
from harbor.utils.artifact_authority import validate_artifact_authority
from harbor.utils.artifact_writer import ArtifactUploadTimeoutError, get_artifact_writer
from harbor.utils.logger import logger
from harbor.utils.pass_at_k import compute_pass_at_k_by_evals
from harbor.utils.path_compat import safe_rmtree
from harbor_config.errors import is_resume_retryable


CANCELLED_ERROR_TYPE = "CancelledError"

# Exception types whose on-disk trial result should be RE-RUN on a fresh
# `harbor jobs resume` (a brand-new invocation of the whole job), rather than
# kept as a terminal error result. The canonical set lives in the shared
# `harbor_config.errors` taxonomy so downstream consumers (e.g. MarinSkyRL)
# inherit the same exception identifiers; this is the resume-level (whole-job
# re-invocation) retry decision ONLY and is deliberately DISTINCT from the
# in-trial retry classification in `harbor/environments/daytona/utils.py`
# (`is_sandbox_build_failure` / `_is_non_retryable`), where a SandboxBuildFailed
# is *correctly* marked non-retryable so harbor does not hammer a cap-full
# Daytona in a tight loop within a single trial attempt.
#
# The set is transient *at the job level*:
#   * CancelledError                     — interrupted (preemption / Ctrl-C /
#                                          job-level timeout); never finished.
#   * SandboxBuildFailedError            — on Daytona, almost always a cap-full
#                                          spike that has cleared by resume.
#   * ContextManagementInfrastructureError — Harbor's context-management failed
#                                          to produce a valid request; the model
#                                          never returned a real answer.

# Legacy results persisted BEFORE ContextManagementInfrastructureError existed
# carry the provider's ``BadRequestError`` type. A resume must still re-run the
# ones whose payload is the exact context-budget-exhausted signal, but ONLY
# those — a generic BadRequestError is usually a genuine misconfiguration and
# must stay terminal. Match the structured vLLM ``max_tokens=0`` payload.
_BAD_REQUEST_MAX_TOKENS_ZERO_SIGNAL = "max_tokens must be at least 1, got 0"

# Concurrency for the resume scan of on-disk trial dirs. Each trial costs ~4
# serial remote round-trips (is_dir + result exists + two reads); on a
# gs:///s3:// jobs_dir with tens of thousands of completed trials the serial
# walk dominates resume time. fsspec releases the GIL on network I/O, so a
# thread pool collapses the walk to a near-constant.
RESUME_SCAN_WORKERS = 64


class Job:
    """
    The single entrypoint for launching a set of trials.

    The main CLI command will correspond to this class.

    Handles database init and task & dataset inserts.

    Saves job-level results to the database and filesystem.

    Translates input parameters into a list of trial configs.
    """

    def __init__(
        self,
        config: JobConfig,
        *,
        _task_configs: list[TaskConfig] | None = None,
        _metrics: dict[str, list[BaseMetric]] | None = None,
        _task_download_results: dict[TaskIdType, TaskDownloadResult] | None = None,
        _benchmark_metadata: tuple[BenchmarkMetadata, ...] = (),
    ):
        """Deprecated. Use ``await Job.create(config)`` instead."""
        if _task_configs is None or _metrics is None:
            raise ValueError(
                "Instantiating Job directly is deprecated. "
                "Use `await Job.create(config)` instead."
            )

        self.config = config
        if config.archive is not None and any(
            str(root).startswith(("s3://", "s3a://"))
            for root in (config.jobs_dir, config.archive.root)
        ):
            # The archive extra provides Rigging through FineStore.
            from harbor.utils.s3_request_timeout import configure_s3_request_timeout

            configure_s3_request_timeout()
        validate_artifact_authority(
            config.jobs_dir,
            config.artifact_authority,
            operation="job startup",
        )
        self._existing_job_result = (
            JobResult.model_validate_json(self._job_result_path.read_text())
            if self._job_result_path.exists()
            else None
        )
        self.is_resuming = self._existing_job_result is not None

        self._trial_completion_lock = asyncio.Lock()
        self._job_result_snapshot_lock = threading.Lock()
        self._job_result_snapshot_futures: set[Future] = set()
        self._job_result_snapshot_timeout_count = 0
        self._first_job_result_snapshot_timeout: BaseException | None = None

        self._id = (
            uuid4()
            if self._existing_job_result is None
            else self._existing_job_result.id
        )

        self.job_dir.mkdir(parents=True, exist_ok=True)

        self._task_configs = _task_configs
        self._benchmark_metadata = _benchmark_metadata
        self._task_download_results = _task_download_results or {}
        self._init_trial_configs()
        self._metrics = _metrics
        self._job_lock: JobLock | None = None
        self._log_file_handler: logging.Handler | None = None
        self._console_handler: logging.Handler | None = None
        self._init_logger()

        self._maybe_init_existing_job()
        self._init_progress_tracking()

        self._init_remaining_trial_configs()
        self._live_rewards = self._existing_rewards.copy()

        trials_per_attempt = len(self._task_configs) * len(self.config.agents)
        self._planned_attempts = {
            trial_config.trial_name: index // trials_per_attempt + 1
            for index, trial_config in enumerate(self._trial_configs)
        }
        self._trial_queue = TrialQueue(
            n_concurrent=self.config.n_concurrent_trials,
            retry_config=self.config.retry,
            release_trial_payloads_in_memory=(
                self.config.release_trial_payloads_in_memory
            ),
            planned_attempts=self._planned_attempts,
        )
        self._trial_queue.add_hook(TrialEvent.START, self._on_trial_started)
        self._trial_queue.add_hook(TrialEvent.CANCEL, self._on_trial_cancelled)
        self._trial_queue.add_hook(TrialEvent.END, self._on_trial_completed)

    @classmethod
    async def create(cls, config: JobConfig) -> "Job":
        task_configs, benchmark_metadata = await cls._resolve_task_configs_and_metadata(
            config
        )
        EnvironmentFactory.validate_resource_policies(config.environment)
        metrics = await cls._resolve_metrics(config, task_configs)

        task_download_results = await cls._cache_tasks(task_configs)

        return cls(
            config,
            _task_configs=task_configs,
            _metrics=metrics,
            _task_download_results=task_download_results,
            _benchmark_metadata=benchmark_metadata,
        )

    def __len__(self):
        """Return the total number of trials configured for this job."""
        return len(self._trial_configs)

    @property
    def benchmark_metadata(self) -> tuple[BenchmarkMetadata, ...]:
        """Return benchmark descriptors resolved before trial execution."""
        return self._benchmark_metadata

    def add_hook(self, event: TrialEvent, callback: HookCallback) -> "Job":
        """Register a callback for a trial lifecycle event and return the job."""
        self._trial_queue.add_hook(event, callback)
        return self

    def on_trial_started(self, callback: HookCallback) -> "Job":
        """Register a callback that runs when a trial starts."""
        return self.add_hook(TrialEvent.START, callback)

    def on_environment_started(self, callback: HookCallback) -> "Job":
        """Register a callback that runs when a trial environment starts."""
        return self.add_hook(TrialEvent.ENVIRONMENT_START, callback)

    def on_agent_started(self, callback: HookCallback) -> "Job":
        """Register a callback that runs when a trial agent starts."""
        return self.add_hook(TrialEvent.AGENT_START, callback)

    def on_verification_started(self, callback: HookCallback) -> "Job":
        """Register a callback that runs when trial verification starts."""
        return self.add_hook(TrialEvent.VERIFICATION_START, callback)

    def on_verification_ended(self, callback: HookCallback) -> "Job":
        """Register a callback that runs when trial verification ends."""
        return self.add_hook(TrialEvent.VERIFICATION_END, callback)

    def on_result_persistence_started(self, callback: HookCallback) -> "Job":
        """Register a callback that runs when trial result persistence starts."""
        return self.add_hook(TrialEvent.RESULT_PERSISTENCE_START, callback)

    def on_trial_ended(self, callback: HookCallback) -> "Job":
        """Register a callback that runs when a trial ends."""
        return self.add_hook(TrialEvent.END, callback)

    def on_trial_cancelled(self, callback: HookCallback) -> "Job":
        """Register a callback that runs when a trial is cancelled."""
        return self.add_hook(TrialEvent.CANCEL, callback)

    def _init_logger(self):
        mode = "a" if self.is_resuming else "w"

        # logging.FileHandler does os.path.abspath(filename) on construction,
        # which only works for local filesystems. When jobs_dir is a remote
        # UPath (gs://, s3://, ...), skip the file handler — the console
        # handler is still attached so logs go to stdout, and external
        # aggregators (iris finelog, Cloud Logging, etc.) can pick them up.
        log_path = self._log_output_path
        protocol = getattr(log_path, "protocol", "")
        if isinstance(protocol, tuple):
            protocol = protocol[0] if protocol else ""
        if protocol in ("", "file", "local"):
            file_handler = logging.FileHandler(log_path, mode=mode)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            self._log_file_handler = file_handler
        else:
            self._log_file_handler = None

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self.config.debug else logging.INFO)

        logger.addHandler(console_handler)
        self._console_handler = console_handler

        self._logger = logger.getChild(__name__)

    def _close_logger_handlers(self) -> None:
        for handler in (self._log_file_handler, self._console_handler):
            if handler is None:
                continue
            logger.removeHandler(handler)
            handler.close()

        self._log_file_handler = None
        self._console_handler = None

    def _maybe_init_existing_job(self):
        self._existing_trial_configs: list[TrialConfig] = []
        self._existing_trial_results: list[TrialResult] = []

        self._previous_trial_results: dict[str, TrialResult] = {}
        self._existing_rewards: EvalsRewardsMap = defaultdict(dict)

        self._existing_stats = JobStats()

        if not self._job_config_path.exists():
            return

        existing_config = JobConfig.model_validate_json(
            self._job_config_path.read_text()
        )

        # A drifted JOB config on re-entry is NOT fatal on its own. The
        # authoritative resume-safety check is the per-trial reconciliation in
        # `_init_remaining_trial_configs`, which matches each on-disk trial to a
        # planned trial by *identity* (task / benchmark / dataset / agent
        # program / environment build / verifier), tolerates benign runtime
        # drift (model_alias, api_base rewritten to the live node,
        # timeout multipliers), and still raises on a genuine SEMANTIC mismatch.
        # A crude exact-equality `FileExistsError` here fired BEFORE that check
        # ran, so every preempt-resume — and the end-of-job
        # `harbor jobs start ... --export-push` re-entry, which rebuilds the
        # config from CLI args that no longer byte-match the on-disk config —
        # aborted before any trial could be reconciled or exported. Log the
        # drift and defer to the per-trial guard.
        if existing_config != self.config:
            logger.warning(
                "Job directory %s already has a config that differs from the "
                "current launch config; resuming anyway. Per-trial identity "
                "reconciliation will keep matching on-disk trials and raise on "
                "a genuine semantic mismatch.",
                self.job_dir,
            )

        trial_dirs = list(self.job_dir.iterdir())
        logger.info(
            "Resuming job %r: scanning %d on-disk trial dir(s) under %s ...",
            self.config.job_name,
            len(trial_dirs),
            self.job_dir,
        )
        scan_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=RESUME_SCAN_WORKERS) as pool:
            loaded = pool.map(self._load_existing_trial, trial_dirs)
        for trial in loaded:
            if trial is None:
                continue
            config, result = trial
            self._existing_trial_configs.append(config)
            self._existing_trial_results.append(result)
        logger.info(
            "Resume scan: loaded %d completed trial(s) in %.1fs",
            len(self._existing_trial_results),
            time.monotonic() - scan_start,
        )

        for trial_result in self._existing_trial_results:
            evals_key = JobStats.evals_key_for(trial_result)
            self._existing_rewards[evals_key][trial_result.trial_name] = (
                trial_result.verifier_result.rewards
                if trial_result.verifier_result is not None
                else None
            )
            self._previous_trial_results[trial_result.trial_name] = trial_result

        self._existing_stats = JobStats.from_trial_results(self._existing_trial_results)

    def _load_existing_trial(
        self, trial_dir: UPath
    ) -> tuple[TrialConfig, TrialResult] | None:
        """Load one on-disk trial's ``(config, result)`` for a resume.

        Returns ``None`` for entries that are not trial dirs, whose
        ``result.json`` is missing (an interrupted trial), or whose on-disk
        result is *resume-retryable* (a CancelledError / SandboxBuildFailedError
        result — see :func:`harbor_config.errors.is_resume_retryable`); the dir is removed
        so the trial is re-planned and RE-RUN on this resume instead of kept as a
        terminal result. Runs under a thread pool during the resume scan, so it
        must not mutate shared ``Job`` state (rmtree is disk I/O, not shared
        state).
        """
        if not trial_dir.is_dir():
            return None
        trial_paths = TrialPaths(trial_dir)
        if not trial_paths.result_path.exists():
            safe_rmtree(trial_paths.trial_dir)
            return None
        result = TrialResult.model_validate_json(trial_paths.result_path.read_text())
        if self._is_resume_retryable_result(result):
            # A cap-full / interrupted attempt — drop it so the trial is
            # re-planned into the remaining set and re-run on this resume,
            # exactly like an interrupted (missing-result) trial. Genuinely
            # terminal errors and valid results fall through and are KEPT.
            safe_rmtree(trial_paths.trial_dir)
            return None
        config = (
            TrialConfig.model_validate_json(trial_paths.config_path.read_text())
            if trial_paths.config_path.exists()
            else result.config
        )
        if self.config.release_trial_payloads_in_memory:
            result = result.slimmed()
        return config, result

    def _init_progress_tracking(self) -> None:
        self._running_trial_ids: set[str] = set()
        self._cancelled_trial_ids: set[str] = {
            result.trial_name
            for result in self._existing_trial_results
            if self._is_cancelled_result(result)
        }
        self._n_retries = (
            self._existing_job_result.stats.n_retries
            if self._existing_job_result is not None
            else 0
        )

    def _init_remaining_trial_configs(self):
        """Reconcile the freshly-PLANNED trial configs against the trial dirs
        already on disk (the partial progress we want to keep on a resume).

        A planned trial is matched to an existing on-disk trial in two passes:

        1. **Exact** (``TrialConfig.__eq__``) — every field equal. This is the
           common, no-drift case.
        2. **Identity-only** (:meth:`TrialConfig.matches_identity`) — same
           evaluation (task / benchmark / dataset / agent program / environment
           build / verifier) but the runtime *serving* config has drifted
           (e.g. a serving alias, an ``api_base`` rewritten to
           the live node, ``--enforce-eager``/timeout-multiplier changes). Such
           drift does NOT change what a trial *is*, so the existing trial is
           kept and the drift is logged loudly — instead of aborting the whole
           resume and re-running nothing.

        A genuinely different eval (model/benchmark/task/n_attempts/dataset
        differs — a SEMANTIC mismatch) still leaves existing trials unmatched
        after both passes, and we still raise: a resume must never silently
        reuse trials from a different evaluation.
        """
        remaining_trial_configs: list[TrialConfig] = []
        matched_existing_indices: set[int] = set()
        reconciled_trial_configs: list[TrialConfig] = []
        serving_drift_count = 0

        # Index existing configs by canonical match keys so each pass is O(n+m)
        # rather than O(planned x existing) — the quadratic scan blew up resume
        # time to hours once tens of thousands of trials had completed. The keys
        # mirror the two match relations exactly:
        #   * exact   -> TrialConfig.__eq__      (model_dump minus excluded fields)
        #   * identity-> TrialConfig.matches_identity (identity_fingerprint)
        # so equal configs hash to the same key and unequal ones do not.
        def _exact_key(cfg: TrialConfig) -> str:
            return json.dumps(
                cfg.model_dump(exclude=TrialConfig.EQUALITY_EXCLUDED_FIELDS),
                sort_keys=True,
                default=str,
            )

        existing_by_exact: dict[str, deque[int]] = defaultdict(deque)
        existing_by_identity: dict[str, deque[int]] = defaultdict(deque)
        for index, existing_trial_config in enumerate(self._existing_trial_configs):
            existing_by_exact[_exact_key(existing_trial_config)].append(index)
            existing_by_identity[trial_identity_key(existing_trial_config)].append(
                index
            )

        def _take(bucket: deque[int] | None) -> int | None:
            """Pop the lowest-index still-unmatched existing trial from *bucket*.

            Buckets are index-ordered, so this reproduces the original
            "first unmatched existing" selection. An index consumed by an
            earlier (exact) pass is skipped when its identity bucket is drained.
            """
            while bucket:
                index = bucket.popleft()
                if index not in matched_existing_indices:
                    matched_existing_indices.add(index)
                    return index
            return None

        # Pass 1: exact matches first (so an unchanged trial always binds to its
        # own exact on-disk twin before identity-only matching is considered).
        exact_matched: dict[int, int] = {}
        for plan_index, trial_config in enumerate(self._trial_configs):
            existing_index = _take(existing_by_exact.get(_exact_key(trial_config)))
            if existing_index is not None:
                exact_matched[plan_index] = existing_index

        # Pass 2: identity-only matches for any planned trial not exact-matched;
        # fall through to "remaining" (needs to run) if nothing matches at all.
        for plan_index, trial_config in enumerate(self._trial_configs):
            if plan_index in exact_matched:
                reconciled_trial_configs.append(
                    self._existing_trial_configs[exact_matched[plan_index]]
                )
                continue

            existing_index = _take(
                existing_by_identity.get(trial_identity_key(trial_config))
            )
            if existing_index is None:
                remaining_trial_configs.append(trial_config)
                reconciled_trial_configs.append(trial_config)
            else:
                serving_drift_count += 1
                self._log_serving_drift(
                    self._existing_trial_configs[existing_index], trial_config
                )
                # Keep the EXISTING (on-disk) trial config so its trial dir
                # continues to match; only the serving fields differ and those
                # are not replayed for an already-complete trial.
                reconciled_trial_configs.append(
                    self._existing_trial_configs[existing_index]
                )

        if len(matched_existing_indices) != len(self._existing_trial_configs):
            unmatched = [
                self._existing_trial_configs[i]
                for i in range(len(self._existing_trial_configs))
                if i not in matched_existing_indices
            ]
            raise TrialPlanMismatchError(
                classify_trial_plan_mismatches(unmatched, self._trial_configs),
                compatible_existing_count=len(matched_existing_indices),
            )

        if serving_drift_count:
            logger.warning(
                "Resume reconciliation: kept %d existing trial(s) across benign "
                "runtime-serving drift (model_alias / api_base / timeout "
                "scaling). Task identity is unchanged; partial progress preserved.",
                serving_drift_count,
            )

        self._trial_configs = reconciled_trial_configs
        self._remaining_trial_configs = remaining_trial_configs

    @staticmethod
    def _log_serving_drift(existing: TrialConfig, planned: TrialConfig) -> None:
        """Log the specific serving fields that drifted for one matched trial."""
        existing_agent = existing.agent
        planned_agent = planned.agent
        diffs: list[str] = []
        if existing_agent.model_alias != planned_agent.model_alias:
            diffs.append(
                f"agent.model_alias {existing_agent.model_alias!r} -> "
                f"{planned_agent.model_alias!r}"
            )
        if existing_agent.kwargs.get("api_base") != planned_agent.kwargs.get(
            "api_base"
        ):
            diffs.append(
                f"agent.kwargs.api_base "
                f"{existing_agent.kwargs.get('api_base')!r} -> "
                f"{planned_agent.kwargs.get('api_base')!r}"
            )
        if existing.timeout_multiplier != planned.timeout_multiplier:
            diffs.append(
                f"timeout_multiplier {existing.timeout_multiplier} -> "
                f"{planned.timeout_multiplier}"
            )
        if diffs:
            logger.debug(
                "Resume: serving-config drift for trial %s (kept existing): %s",
                existing.trial_name,
                "; ".join(diffs),
            )

    @staticmethod
    async def _resolve_task_configs(config: JobConfig) -> list[TaskConfig]:
        task_configs, _ = await Job._resolve_task_configs_and_metadata(config)
        return task_configs

    @staticmethod
    async def _resolve_task_configs_and_metadata(
        config: JobConfig,
    ) -> tuple[list[TaskConfig], tuple[BenchmarkMetadata, ...]]:
        task_configs: list[TaskConfig] = [
            task.model_copy(deep=True) for task in config.tasks
        ]
        metadata: list[BenchmarkMetadata] = []
        if task_configs:
            metadata.append(
                reward_benchmark_metadata(
                    ADHOC_DATASET_NAME,
                    n_benchmark=len(task_configs),
                    n_attempted=len(task_configs),
                )
            )

        for dataset in config.datasets:
            (
                dataset_tasks,
                n_benchmark,
            ) = await dataset.get_task_configs_with_pre_cap_count(
                disable_verification=config.verifier.disable
                or config.verifier.import_path is not None
            )
            task_configs.extend(dataset_tasks)
            if dataset.name is None and dataset.path is None:
                raise ValueError("dataset has no resolved name")
            task_name = dataset.name or dataset.path.name
            metadata.append(
                reward_benchmark_metadata(
                    task_name,
                    n_benchmark=n_benchmark,
                    n_attempted=len(dataset_tasks),
                )
            )

        if not task_configs:
            raise ValueError("Either datasets or tasks must be provided.")

        return task_configs, tuple(metadata)

    def _init_trial_configs(self):
        self._trial_configs = [
            TrialConfig(
                task=task_config,
                trials_dir=self.job_dir,
                artifact_authority=self.config.artifact_authority,
                agent=agent_config,
                user_agent=self.config.user_agent,
                timeout_multiplier=self.config.timeout_multiplier,
                agent_timeout_multiplier=self.config.agent_timeout_multiplier,
                verifier_timeout_multiplier=self.config.verifier_timeout_multiplier,
                agent_setup_timeout_multiplier=self.config.agent_setup_timeout_multiplier,
                environment_build_timeout_multiplier=self.config.environment_build_timeout_multiplier,
                trial_attempt_timeout_sec=self.config.trial_attempt_timeout_sec,
                trial_cleanup_grace_sec=self.config.trial_cleanup_grace_sec,
                environment=self.config.environment,
                verifier=self.config.verifier,
                artifacts=self.config.artifacts,
                extra_instruction_paths=self.config.extra_instruction_paths,
                job_id=self._id,
            )
            for _ in range(self.config.n_attempts)
            for task_config in self._task_configs
            for agent_config in self.config.agents
            # Put agent on the inside of the nested loop to potentially distribute
            # model providers and improve rate limit usage.
        ]

    @property
    def id(self):
        """The job's UUID. Stable across the run; chosen at construction
        (or recovered from `result.json` on a resume)."""
        return self._id

    @property
    def job_dir(self):
        return self.config.jobs_dir / self.config.job_name

    @property
    def _log_output_path(self):
        return self.job_dir / "job.log"

    @property
    def _job_config_path(self):
        return self.job_dir / JOB_CONFIG_FILENAME

    @property
    def _job_lock_path(self):
        return self.job_dir / JOB_LOCK_FILENAME

    @property
    def _job_result_path(self):
        return self.job_dir / JOB_RESULT_FILENAME

    @staticmethod
    def _is_cancelled_result(trial_result: TrialResult) -> bool:
        return (
            trial_result.exception_info is not None
            and trial_result.exception_info.exception_type == CANCELLED_ERROR_TYPE
        )

    @staticmethod
    def _is_resume_retryable_result(trial_result: TrialResult) -> bool:
        """True if this on-disk trial result should be RE-RUN on a fresh resume.

        The canonical retryable set lives in the shared
        :func:`harbor_config.errors.is_resume_retryable` taxonomy. Kept separate
        from :meth:`_is_cancelled_result`, which drives cancelled-specific
        progress bookkeeping (``n_cancelled_trials``); this predicate is only
        the resume-level "re-run vs keep" decision and defaults to include BOTH
        CancelledError and SandboxBuildFailedError.

        Also re-runs legacy ``BadRequestError`` results whose payload is the
        exact context-budget-exhausted signal (see
        :data:`_BAD_REQUEST_MAX_TOKENS_ZERO_SIGNAL`) — those predate
        :class:`ContextManagementInfrastructureError` but are the same failure.
        """
        info = trial_result.exception_info
        if info is None:
            return False
        if is_resume_retryable(info.exception_type):
            return True
        return (
            info.exception_type == "BadRequestError"
            and _BAD_REQUEST_MAX_TOKENS_ZERO_SIGNAL
            in f"{info.exception_message}\n{info.exception_traceback}"
        )

    def _refresh_job_progress(
        self,
        *,
        updated_at: datetime | None = None,
    ) -> None:
        updated_at = updated_at or datetime.now()
        self._job_result.updated_at = updated_at
        self._job_result.stats.update_progress_counts(
            n_total_trials=self._job_result.n_total_trials,
            n_running_trials=len(self._running_trial_ids),
            n_cancelled_trials=len(self._cancelled_trial_ids),
            n_retries=self._n_retries,
        )

    @staticmethod
    def _evals_key_for_result(trial_result: TrialResult) -> tuple[str, str]:
        dataset_name = trial_result.source or ADHOC_DATASET_NAME
        return JobStats.evals_key_for(trial_result), dataset_name

    def _refresh_metrics_for_eval(self, evals_key: str, dataset_name: str) -> None:
        rewards_list = list(self._live_rewards[evals_key].values())
        if not rewards_list:
            self._job_result.stats.evals[evals_key].metrics = []
            return
        self._job_result.stats.evals[evals_key].metrics = [
            metric.compute(rewards_list) for metric in self._metrics[dataset_name]
        ]

    def _remove_completed_attempt_for_retry(self, trial_name: str) -> None:
        previous_result = self._previous_trial_results.pop(trial_name, None)
        if previous_result is None:
            return

        evals_key, dataset_name = self._evals_key_for_result(previous_result)
        self._live_rewards[evals_key].pop(trial_name, None)
        self._job_result.stats.remove_trial(previous_result)
        if self._is_cancelled_result(previous_result):
            self._cancelled_trial_ids.discard(trial_name)
        self._n_retries += 1
        self._refresh_metrics_for_eval(evals_key, dataset_name)

    def _serialize_job_result(self) -> str:
        return self._job_result.model_dump_json(
            indent=4,
            exclude={"trial_results"},
        )

    def _queue_job_result_snapshot(self) -> None:
        result_json = self._serialize_job_result()
        future = get_artifact_writer().submit_optional(
            self._job_result_path,
            lambda result_json=result_json: result_json,
            description="job progress snapshot",
        )
        with self._job_result_snapshot_lock:
            self._job_result_snapshot_futures.add(future)
        future.add_done_callback(self._record_job_result_snapshot_outcome)

    def _record_job_result_snapshot_outcome(self, future: Future) -> None:
        error = future.exception()
        with self._job_result_snapshot_lock:
            self._job_result_snapshot_futures.discard(future)
            if isinstance(error, ArtifactUploadTimeoutError):
                self._job_result_snapshot_timeout_count += 1
                if self._first_job_result_snapshot_timeout is None:
                    self._first_job_result_snapshot_timeout = error

    async def _write_required_job_result(self) -> None:
        result_json = self._serialize_job_result()
        writer = get_artifact_writer()
        future = await writer.submit_required(
            self._job_result_path,
            lambda result_json=result_json: result_json,
            description="required job result",
        )
        await writer.await_write(future)

    async def _flush_job_result_snapshots(self) -> None:
        writer = get_artifact_writer()
        await asyncio.to_thread(
            writer.flush,
            prefix=str(self._job_result_path),
        )
        with self._job_result_snapshot_lock:
            timeout_count = self._job_result_snapshot_timeout_count
            first_timeout = self._first_job_result_snapshot_timeout
            self._job_result_snapshot_timeout_count = 0
            self._first_job_result_snapshot_timeout = None
        if timeout_count:
            logger.warning(
                "%d job progress snapshot upload(s) timed out; continuing with "
                "the required final result after stopping the timed-out uploader; "
                "first_timeout=%s",
                timeout_count,
                first_timeout,
            )

    @staticmethod
    async def _resolve_metrics(
        config: JobConfig, task_configs: list[TaskConfig]
    ) -> dict[str, list[BaseMetric]]:
        metrics: dict[str, list[BaseMetric]] = defaultdict(list)

        job_metrics = [
            MetricFactory.create_metric(metric.type, **metric.kwargs)
            for metric in config.metrics
        ]

        metrics[ADHOC_DATASET_NAME].extend(job_metrics)

        for dataset_config in config.datasets:
            await Job._resolve_dataset_metrics(dataset_config, metrics, job_metrics)

        for task_config in task_configs:
            source = task_config.source or ADHOC_DATASET_NAME
            if source not in metrics:
                metrics[source].extend(job_metrics)

        for name, metric_list in metrics.items():
            if len(metric_list) == 0:
                metrics[name].append(Mean())

        return metrics

    @staticmethod
    async def _resolve_dataset_metrics(
        dataset_config: DatasetConfig,
        metrics: dict[str, list[BaseMetric]],
        job_metrics: list[BaseMetric],
    ) -> None:
        if dataset_config.is_local():
            if dataset_config.path is None:
                raise RuntimeError(
                    "Local dataset config is missing path; this should never happen."
                )
            source = dataset_config.path.expanduser().resolve().name
            metrics[source].extend(job_metrics)
        elif dataset_config.is_hugging_face():
            if dataset_config.name is None:
                raise RuntimeError("Hugging Face dataset config is missing name.")
            metrics[dataset_config.name].extend(job_metrics)
        elif dataset_config.is_package():
            from harbor.registry.client.package import PackageDatasetClient

            if dataset_config.name is None:
                raise RuntimeError(
                    "Package dataset config is missing name; this should never happen."
                )
            client = PackageDatasetClient()
            name_string = f"{dataset_config.name}@{dataset_config.ref or 'latest'}"
            metadata = await client.get_dataset_metadata(name_string)

            # Download dataset files (e.g., metric.py)
            downloaded_files = await client.download_dataset_files(metadata)
            if DatasetPaths.METRIC_FILENAME in downloaded_files:
                from harbor.metrics.uv_script import UvScript

                metrics[dataset_config.name].append(
                    UvScript(script_path=downloaded_files[DatasetPaths.METRIC_FILENAME])
                )

            metrics[dataset_config.name].extend(
                [
                    MetricFactory.create_metric(metric.type, **metric.kwargs)
                    for metric in metadata.metrics
                ]
            )
            metrics[dataset_config.name].extend(job_metrics)
        elif dataset_config.is_registry():
            if dataset_config.name is None:
                raise RuntimeError(
                    "Registry dataset config is missing name; this should never happen."
                )
            client = RegistryClientFactory.create(
                registry_url=dataset_config.registry_url,
                registry_path=dataset_config.registry_path,
            )
            name_string = (
                f"{dataset_config.name}@{dataset_config.version}"
                if dataset_config.version
                else dataset_config.name
            )
            metadata = await client.get_dataset_metadata(name_string)
            metrics[dataset_config.name].extend(
                [
                    MetricFactory.create_metric(metric.type, **metric.kwargs)
                    for metric in metadata.metrics
                ]
            )
            metrics[dataset_config.name].extend(job_metrics)

    @staticmethod
    async def _cache_tasks(
        task_configs: list[TaskConfig],
    ) -> dict[TaskIdType, TaskDownloadResult]:
        """Pre-download git-based tasks before submitting trials."""
        remote_configs = [
            config
            for config in task_configs
            if config.is_git_task() or config.is_package_task()
        ]
        if not remote_configs:
            return {}

        overwrites = {config.overwrite for config in remote_configs}
        output_dirs = {config.download_dir for config in remote_configs}

        if len(overwrites) > 1 or len(output_dirs) > 1:
            raise ValueError(
                "overwrite and output_dir cannot be different for different trials. "
                "This should never happen."
            )

        client = TaskClient()

        task_ids = [config.get_task_id() for config in remote_configs]
        result = await client.download_tasks(
            task_ids=task_ids,
            overwrite=any(overwrites),
            output_dir=output_dirs.pop() if output_dirs else None,
        )

        return dict(zip(task_ids, result.results))

    def _init_job_lock(self) -> None:
        self._job_lock = build_job_lock(
            config=self.config,
            trial_configs=self._trial_configs,
            task_download_results=self._task_download_results,
        )

    def _write_job_lock(self) -> None:
        if self._job_lock is None:
            self._init_job_lock()
        if self._job_lock is None:
            raise RuntimeError(
                "Job lock was not initialized after _init_job_lock(); "
                "this should never happen."
            )
        if self._job_lock_path.exists():
            existing_job_lock = self._read_existing_job_lock()
            if existing_job_lock is None:
                raise ValueError(
                    f"Existing lock file at {self._job_lock_path} could not be "
                    "parsed; refusing to overwrite it."
                )
            self._job_lock.created_at = existing_job_lock.created_at
            self._job_lock.harbor = existing_job_lock.harbor
            self._job_lock.invocation = existing_job_lock.invocation
            # n_concurrent_trials is a runtime throughput knob, not a
            # replay-affecting identity field: a resume may legitimately clamp it
            # (e.g. Vista VNC remediation lowers it 32->8 via the eval sbatch's
            # config.json rewrite) without changing what any trial evaluates.
            # Carry the stored value onto the resolved lock — same rationale as
            # created_at/harbor/invocation above — so the equality check below
            # doesn't spuriously reject an otherwise-identical resume. The actual
            # runtime concurrency is governed by config.json, not this field.
            self._job_lock.n_concurrent_trials = existing_job_lock.n_concurrent_trials
            if existing_job_lock != self._job_lock:
                raise FileExistsError(
                    f"Job directory {self.job_dir} already has a lock.json that "
                    "does not match the resolved job lock."
                )
        self._job_lock_path.write_text(
            self._job_lock.model_dump_json(indent=4, exclude_none=True)
        )

    def _read_existing_job_lock(self) -> JobLock | None:
        try:
            return JobLock.model_validate_json(self._job_lock_path.read_text())
        except Exception:
            return None

    async def _on_trial_started(self, event: TrialHookEvent) -> None:
        """Internal hook to update live progress when a trial starts."""
        async with self._trial_completion_lock:
            self._remove_completed_attempt_for_retry(event.trial_id)
            self._running_trial_ids.add(event.trial_id)
            self._refresh_job_progress(updated_at=event.timestamp)
            self._queue_job_result_snapshot()

    async def _on_trial_cancelled(self, event: TrialHookEvent) -> None:
        """Internal hook to update live progress when cancellation is requested."""
        async with self._trial_completion_lock:
            self._cancelled_trial_ids.add(event.trial_id)
            self._refresh_job_progress(updated_at=event.timestamp)
            self._queue_job_result_snapshot()

    async def _on_trial_completed(self, event: TrialHookEvent) -> None:
        """Internal hook to update job stats when a trial completes."""
        if event.result is None:
            raise ValueError(
                f"Trial {event.trial_id} completed without a result. "
                "This should never happen."
            )

        trial_result = event.result
        trial_name = trial_result.trial_name

        async with self._trial_completion_lock:
            previous_result = self._previous_trial_results.get(trial_name)

            evals_key, dataset_name = self._evals_key_for_result(trial_result)

            self._live_rewards[evals_key][trial_name] = (
                trial_result.verifier_result.rewards
                if trial_result.verifier_result is not None
                else None
            )

            self._job_result.stats.update_trial(
                new_result=trial_result,
                previous_result=previous_result,
            )

            # Stats above were incremented from the full result. The retained
            # copy only needs the stats inputs for retry remove_trial, so slim
            # it when configured to keep job RSS bounded on long runs. Leave
            # `event.result` (the object propagating to later END hooks and the
            # queue) untouched here; the queue slims its own returned copy.
            if self.config.release_trial_payloads_in_memory:
                self._previous_trial_results[trial_name] = trial_result.slimmed()
            else:
                self._previous_trial_results[trial_name] = trial_result

            self._running_trial_ids.discard(trial_name)
            if previous_result is not None and self._is_cancelled_result(
                previous_result
            ):
                self._cancelled_trial_ids.discard(trial_name)
            if self._is_cancelled_result(trial_result):
                self._cancelled_trial_ids.add(trial_name)

            self._refresh_metrics_for_eval(evals_key, dataset_name)

            self._refresh_job_progress(updated_at=event.timestamp)
            self._queue_job_result_snapshot()

    async def run(self) -> JobResult:
        """Run all configured trials and return the final aggregated job result."""
        telemetry = start_telemetry(
            self.config.telemetry,
            job_uid=self._id,
            job_name=self.config.job_name,
        )
        if telemetry is not None:
            hook = trial_hook(telemetry)
            for event in TrialEvent:
                self._trial_queue.add_hook(event, hook)
        archive = open_archive(self.config.archive, trials_dir=self.job_dir)
        if archive is not None:
            self._trial_queue.add_hook(TrialEvent.END, archive.on_trial_end)
        try:
            with bind_telemetry(telemetry):
                result = await self._run_job()
                # Finalization adds existing resume results and only the ungraded
                # attempts that survived into the completed job result.
                if archive is not None:
                    await archive.finalize(result.trial_results)
                    await asyncio.to_thread(archive.seal)
        except BaseException as error:
            record_job_terminal(telemetry, error)
            raise
        else:
            record_job_terminal(telemetry)
            return result
        finally:
            if archive is not None:
                await asyncio.to_thread(archive.close)
            await asyncio.to_thread(shutdown_telemetry, telemetry)

    async def _run_job(self) -> JobResult:
        try:
            started_at = (
                self._existing_job_result.started_at
                if self._existing_job_result is not None
                else datetime.now()
            )
            self._job_result = JobResult(
                id=self._id,
                started_at=started_at,
                updated_at=started_at,
                n_total_trials=len(self._trial_configs),
                stats=JobStats.from_trial_results(
                    self._existing_trial_results,
                    n_total_trials=len(self._trial_configs),
                    n_retries=self._n_retries,
                ),
                benchmark_metadata=self.benchmark_metadata,
            )
            self._refresh_job_progress()

            self._job_config_path.write_text(self.config.model_dump_json(indent=4))
            self._init_job_lock()
            self._write_job_lock()
            await self._write_required_job_result()

            # Set up progress UI and register progress hooks
            n_remaining = len(self._remaining_trial_configs)
            n_existing = len(self._existing_trial_configs)

            loading_progress = Progress(
                SpinnerColumn(),
                MofNCompleteColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )

            if self.config.quiet:
                with loading_progress:
                    progress_task = loading_progress.add_task(
                        "Running trials...",
                        total=n_remaining + n_existing,
                        completed=n_existing,
                    )

                    trial_results = await self._run_trials_with_queue(
                        loading_progress, progress_task
                    )
            elif not loading_progress.console.is_terminal:
                self._register_log_progress_hooks()
                progress_task = loading_progress.add_task(
                    "Running trials...",
                    total=n_remaining + n_existing,
                    completed=n_existing,
                )
                trial_results = await self._run_trials_with_queue(
                    loading_progress, progress_task
                )
            else:
                running_progress = Progress(
                    SpinnerColumn(),
                    TimeElapsedColumn(),
                    TextColumn("[progress.description]{task.description}"),
                )

                with Live(
                    Group(loading_progress, running_progress), refresh_per_second=10
                ):
                    progress_task = loading_progress.add_task(
                        "Running trials...",
                        total=n_remaining + n_existing,
                        completed=n_existing,
                    )

                    trial_results = await self._run_trials_with_queue(
                        loading_progress, progress_task, running_progress
                    )

            combined_trial_results = self._existing_trial_results + trial_results

            final_rewards = defaultdict(list)

            for trial_result in combined_trial_results:
                evals_key = JobStats.evals_key_for(trial_result)

                if trial_result.verifier_result is not None:
                    final_rewards[evals_key].append(
                        trial_result.verifier_result.rewards
                    )
                else:
                    final_rewards[evals_key].append(None)

            final_stats = JobStats.from_trial_results(
                combined_trial_results,
                n_total_trials=len(self._trial_configs),
                n_retries=self._n_retries,
            )

            for evals_key, rewards in final_rewards.items():
                # Extract dataset_name from the evals_key to get the right metrics
                # The key format is either "agent__model__dataset" or "agent__dataset"
                dataset_name = evals_key.split("__")[-1]
                for metric in self._metrics[dataset_name]:
                    final_stats.evals[evals_key].metrics.append(metric.compute(rewards))

            for evals_key, pass_at_k in compute_pass_at_k_by_evals(
                combined_trial_results
            ).items():
                final_stats.evals[evals_key].pass_at_k = pass_at_k

            self._job_result.stats = final_stats
            self._job_result.trial_results = combined_trial_results

            finished_at = datetime.now()
            self._job_result.finished_at = finished_at
            self._refresh_job_progress(updated_at=finished_at)
            await self._flush_job_result_snapshots()
            await self._write_required_job_result()

            return self._job_result
        finally:
            self._close_logger_handlers()

    def _log_trial_progress(
        self,
        event: TrialHookEvent,
        status: str,
        exception_type: str | None = None,
    ) -> None:
        fields = {"trial_id": event.trial_id, "trial_status": status}
        display_status = status
        if exception_type is not None:
            fields["exception_type"] = exception_type
            display_status = f"{status} ({exception_type})"
        self._logger.info(
            "Trial %s: %s",
            event.trial_id,
            display_status,
            extra=fields,
        )

    def _register_log_progress_hooks(self) -> None:
        async def on_start(event: TrialHookEvent) -> None:
            self._log_trial_progress(event, "started")

        async def on_environment_start(event: TrialHookEvent) -> None:
            self._log_trial_progress(event, "environment started")

        async def on_agent_start(event: TrialHookEvent) -> None:
            self._log_trial_progress(event, "agent started")

        async def on_verification_start(event: TrialHookEvent) -> None:
            self._log_trial_progress(event, "verification started")

        async def on_verification_end(event: TrialHookEvent) -> None:
            self._log_trial_progress(event, "verification completed")

        async def on_result_persistence_start(event: TrialHookEvent) -> None:
            self._log_trial_progress(event, "result persistence started")

        async def on_cancel(event: TrialHookEvent) -> None:
            self._log_trial_progress(event, "cancellation requested")

        async def on_end(event: TrialHookEvent) -> None:
            if event.result is None:
                raise ValueError(
                    f"Trial {event.trial_id} completed without a result. "
                    "This should never happen."
                )
            exception = event.result.exception_info
            if exception is not None:
                self._log_trial_progress(
                    event,
                    "failed",
                    exception_type=exception.exception_type,
                )
                return
            self._log_trial_progress(event, "completed")

        self.add_hook(TrialEvent.START, on_start)
        self.add_hook(TrialEvent.ENVIRONMENT_START, on_environment_start)
        self.add_hook(TrialEvent.AGENT_START, on_agent_start)
        self.add_hook(TrialEvent.VERIFICATION_START, on_verification_start)
        self.add_hook(TrialEvent.VERIFICATION_END, on_verification_end)
        self.add_hook(
            TrialEvent.RESULT_PERSISTENCE_START,
            on_result_persistence_start,
        )
        self.add_hook(TrialEvent.CANCEL, on_cancel)
        self.add_hook(TrialEvent.END, on_end)

    async def _run_trials_with_queue(
        self,
        loading_progress: Progress,
        loading_progress_task,
        running_progress: Progress | None = None,
    ) -> list[TrialResult]:
        """Submit trials to the queue and collect results."""
        # Track per-trial progress tasks for the running_progress display.
        # Guards ensure hooks firing on retry attempts are idempotent.
        trial_progress_tasks: dict[str, TaskID] = {}
        advanced_trials: set[str] = set()

        if running_progress is not None:

            async def on_start(event: TrialHookEvent):
                if event.trial_id not in trial_progress_tasks:
                    task_id = running_progress.add_task(
                        f"{event.trial_id}: running trial...", total=None
                    )
                    trial_progress_tasks[event.trial_id] = task_id

            async def on_environment_start(event: TrialHookEvent):
                if event.trial_id in trial_progress_tasks:
                    running_progress.update(
                        trial_progress_tasks[event.trial_id],
                        description=f"{event.trial_id}: starting environment...",
                    )

            async def on_agent_start(event: TrialHookEvent):
                if event.trial_id in trial_progress_tasks:
                    running_progress.update(
                        trial_progress_tasks[event.trial_id],
                        description=f"{event.trial_id}: running agent...",
                    )

            async def on_verification_start(event: TrialHookEvent):
                if event.trial_id in trial_progress_tasks:
                    running_progress.update(
                        trial_progress_tasks[event.trial_id],
                        description=f"{event.trial_id}: running verifier...",
                    )

            async def on_result_upload(event: TrialHookEvent):
                if event.trial_id in trial_progress_tasks:
                    running_progress.update(
                        trial_progress_tasks[event.trial_id],
                        description=f"{event.trial_id}: uploading results...",
                    )

            async def on_cancel(event: TrialHookEvent):
                if event.trial_id in trial_progress_tasks:
                    running_progress.update(
                        trial_progress_tasks[event.trial_id],
                        description=f"{event.trial_id}: canceling trial; "
                        "this may take up to a minute...",
                    )

            async def on_end_progress(event: TrialHookEvent):
                if event.trial_id in trial_progress_tasks:
                    running_progress.remove_task(
                        trial_progress_tasks.pop(event.trial_id)
                    )
                if event.trial_id not in advanced_trials:
                    advanced_trials.add(event.trial_id)
                    loading_progress.advance(loading_progress_task)
                    self._update_metric_display(
                        event, loading_progress, loading_progress_task
                    )

            self.add_hook(TrialEvent.START, on_start)
            self.add_hook(TrialEvent.ENVIRONMENT_START, on_environment_start)
            self.add_hook(TrialEvent.AGENT_START, on_agent_start)
            self.add_hook(TrialEvent.VERIFICATION_START, on_verification_start)
            self.add_hook(TrialEvent.VERIFICATION_END, on_result_upload)
            self.add_hook(
                TrialEvent.RESULT_PERSISTENCE_START,
                on_result_upload,
            )
            self.add_hook(TrialEvent.CANCEL, on_cancel)
            self.add_hook(TrialEvent.END, on_end_progress)
        else:

            async def on_end_quiet(event: TrialHookEvent):
                if event.trial_id not in advanced_trials:
                    advanced_trials.add(event.trial_id)
                    loading_progress.advance(loading_progress_task)
                    self._update_metric_display(
                        event, loading_progress, loading_progress_task
                    )

            self.add_hook(TrialEvent.END, on_end_quiet)

        coros = self._trial_queue.submit_batch(self._remaining_trial_configs)

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(coro) for coro in coros]

        return [t.result() for t in tasks]

    def _update_metric_display(
        self, event: TrialHookEvent, loading_progress, loading_progress_task
    ):
        """Update the progress bar description with the latest metric value."""
        if not self._metrics or event.result is None:
            return
        dataset_name = event.config.task.source or ADHOC_DATASET_NAME
        if dataset_name not in self._metrics:
            return

        rewards = list(
            self._live_rewards.get(
                JobStats.evals_key_for(event.result),
                {},
            ).values()
        )

        if rewards:
            metric_result = self._metrics[dataset_name][0].compute(rewards)
            first_metric_name, first_metric_value = next(iter(metric_result.items()))
            loading_progress.update(
                loading_progress_task,
                description=f"{first_metric_name.title()}: {first_metric_value:.3f}",
            )
