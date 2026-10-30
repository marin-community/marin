import warnings
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)
from upath import UPath

from harbor_config.models.metric.config import MetricConfig
from harbor_config.models.task.id import GitTaskId, LocalTaskId, PackageTaskId
from harbor_config.models.trial.config import (
    AgentConfig,
    ArtifactAuthorityConfig,
    ArtifactConfig,
    DEFAULT_TRIAL_CLEANUP_GRACE_SEC,
    EnvironmentConfig,
    TaskConfig,
    UserAgentConfig,
    VerifierConfig,
)

HF_DATASET_PREFIX = "hf://"


class DatasetConfig(BaseModel):
    path: Path | None = None
    name: str | None = None
    version: str | None = None
    ref: str | None = None
    registry_url: str | None = None
    registry_path: Path | None = None
    overwrite: bool = Field(
        default=False, description="Whether to overwrite cached remote tasks."
    )
    download_dir: Path | None = Field(
        default=None, description="The directory to cache remote tasks to."
    )
    task_names: list[str] | None = Field(
        default=None,
        description="Tasks to include from the dataset. Name can be a glob pattern.",
    )
    exclude_task_names: list[str] | None = Field(
        default=None,
        description="Tasks to exclude from the dataset. Name can be a glob pattern.",
    )
    n_tasks: int | None = Field(
        default=None,
        description="Maximum number of tasks to include from this dataset. Applied after task_names/exclude_task_names filtering.",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_registry_field(cls, data):
        """Migrate legacy nested 'registry' config to flat fields.

        .. deprecated::
            Use ``registry_url`` / ``registry_path`` instead of ``registry``.
        """
        if isinstance(data, dict) and "registry" in data:
            warnings.warn(
                "The nested 'registry' key in dataset config is deprecated. "
                "Use 'registry_url' or 'registry_path' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            registry = data.pop("registry")
            if isinstance(registry, dict):
                if "url" in registry:
                    data.setdefault("registry_url", registry["url"])
                if "path" in registry:
                    data.setdefault("registry_path", registry["path"])
        return data

    @model_validator(mode="after")
    def validate_dataset_source(self):
        if self.path is None and self.name is None:
            raise ValueError("Either 'path' or 'name' must be set.")
        if self.path is not None and self.name is not None:
            raise ValueError("Cannot set both 'path' and 'name'.")
        if self.version is not None and self.ref is not None:
            raise ValueError("Cannot set both 'version' and 'ref'.")
        if self.is_hugging_face():
            repository = self.name.removeprefix(HF_DATASET_PREFIX)
            parts = repository.split("/")
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    "Hugging Face dataset name must be hf://org/repository."
                )
            if self.version is not None:
                raise ValueError("Hugging Face datasets use 'ref', not 'version'.")
        return self

    def is_local(self) -> bool:
        return self.path is not None

    def is_hugging_face(self) -> bool:
        return self.name is not None and self.name.startswith(HF_DATASET_PREFIX)

    def is_package(self) -> bool:
        return self.name is not None and "/" in self.name and not self.is_hugging_face()

    def is_registry(self) -> bool:
        return self.name is not None and "/" not in self.name

    def _filter_task_ids(
        self, task_ids: list[LocalTaskId | GitTaskId | PackageTaskId]
    ) -> list[LocalTaskId | GitTaskId | PackageTaskId]:
        filtered_ids = task_ids
        if self.task_names:
            filtered_ids = [
                task_id
                for task_id in filtered_ids
                if any(
                    fnmatch(task_id.get_name(), pattern_id)
                    for pattern_id in self.task_names
                )
            ]
            if not filtered_ids:
                available = sorted(tid.get_name() for tid in task_ids)
                raise ValueError(
                    f"No tasks matched the filter(s) {self.task_names}. "
                    f"There are {len(available)} tasks available in this dataset. "
                    f"Example task names: {available[:5]}"
                )

        if self.exclude_task_names:
            filtered_ids = [
                task_id
                for task_id in filtered_ids
                if not any(
                    fnmatch(task_id.get_name(), pattern_id)
                    for pattern_id in self.exclude_task_names
                )
            ]

        if self.n_tasks is not None:
            filtered_ids = filtered_ids[: self.n_tasks]

        return filtered_ids

    async def get_task_configs(
        self, disable_verification: bool = False
    ) -> list[TaskConfig]:
        if self.is_local():
            return await self._get_local_task_configs(disable_verification)
        elif self.is_hugging_face():
            return await self._get_hugging_face_task_configs(disable_verification)
        elif self.is_package():
            return await self._get_package_task_configs()
        else:
            return await self._get_registry_task_configs()

    async def get_task_configs_with_pre_cap_count(
        self, disable_verification: bool = False
    ) -> tuple[list[TaskConfig], int]:
        """Resolve filtered tasks and return their count before ``n_tasks``."""
        uncapped = self.model_copy(deep=True, update={"n_tasks": None})
        task_configs = await uncapped.get_task_configs(disable_verification)
        if self.is_package() or self.is_hugging_face():
            self.ref = uncapped.ref
        n_benchmark = len(task_configs)
        if self.n_tasks is not None:
            task_configs = task_configs[: self.n_tasks]
        return task_configs, n_benchmark

    async def _get_local_task_configs(
        self, disable_verification: bool
    ) -> list[TaskConfig]:
        assert self.path is not None
        return self._task_configs_from_path(
            self.path,
            self.path.expanduser().resolve().name,
            disable_verification,
        )

    async def _get_hugging_face_task_configs(
        self, disable_verification: bool
    ) -> list[TaskConfig]:
        from harbor.hugging_face_dataset import download_dataset_snapshot

        assert self.name is not None
        snapshot = await download_dataset_snapshot(
            repo_id=self.name.removeprefix(HF_DATASET_PREFIX),
            revision=self.ref,
            cache_dir=self.download_dir,
            overwrite=self.overwrite,
        )
        self.ref = snapshot.name
        return self._task_configs_from_path(snapshot, self.name, disable_verification)

    def _task_configs_from_path(
        self, path: Path, source: str, disable_verification: bool
    ) -> list[TaskConfig]:
        from harbor.models.task.task import Task as TaskModel

        task_ids: list[LocalTaskId | GitTaskId | PackageTaskId] = [
            LocalTaskId(path=task_path)
            for task_path in path.iterdir()
            if TaskModel.is_valid_dir(
                task_path,
                disable_verification=disable_verification,
            )
        ]
        return [
            TaskConfig(
                path=task_id.path,
                source=source,
            )
            for task_id in self._filter_task_ids(task_ids)
            if isinstance(task_id, LocalTaskId)
        ]

    async def _get_registry_task_configs(self) -> list[TaskConfig]:
        from harbor.registry.client import RegistryClientFactory

        assert self.name is not None
        client = RegistryClientFactory.create(
            registry_url=self.registry_url,
            registry_path=self.registry_path,
        )

        name_string = f"{self.name}@{self.version}" if self.version else self.name
        metadata = await client.get_dataset_metadata(name_string)

        result: list[TaskConfig] = []
        for task_id in self._filter_task_ids(metadata.task_ids):
            match task_id:
                case GitTaskId():
                    result.append(
                        TaskConfig(
                            path=task_id.path,
                            git_url=task_id.git_url,
                            git_commit_id=task_id.git_commit_id,
                            overwrite=self.overwrite,
                            download_dir=self.download_dir,
                            source=self.name,
                        )
                    )
                case LocalTaskId():
                    result.append(
                        TaskConfig(
                            path=task_id.path,
                            overwrite=self.overwrite,
                            download_dir=self.download_dir,
                            source=self.name,
                        )
                    )
        return result

    async def _get_package_task_configs(self) -> list[TaskConfig]:
        from harbor.registry.client.package import PackageDatasetClient

        assert self.name is not None
        client = PackageDatasetClient()

        name_string = f"{self.name}@{self.ref or 'latest'}"
        metadata = await client.get_dataset_metadata(name_string)

        # Update ref to be the content hash for config version tracking
        self.ref = metadata.version

        return [
            TaskConfig(
                name=f"{task_id.org}/{task_id.name}",
                ref=task_id.ref,
                overwrite=self.overwrite,
                download_dir=self.download_dir,
                source=self.name,
            )
            for task_id in self._filter_task_ids(metadata.task_ids)
            if isinstance(task_id, PackageTaskId)
        ]


class RetryConfig(BaseModel):
    max_retries: int = Field(
        default=0, description="Maximum number of retry attempts", ge=0
    )
    include_exceptions: set[str] | None = Field(
        default=None,
        description="Exception types to retry on. If None, retries all exceptions.",
    )
    exclude_exceptions: set[str] | None = Field(
        default_factory=lambda: {
            "AgentTimeoutError",
            "VerifierTimeoutError",
            "RewardFileNotFoundError",
            "RewardFileEmptyError",
            "VerifierOutputParseError",
        },
        description="Exception types to NOT retry on. Takes precedence over "
        "include_exceptions.",
    )
    wait_multiplier: float = Field(
        default=1.0, description="Multiplier for exponential backoff wait time"
    )
    min_wait_sec: float = Field(
        default=1.0, description="Minimum wait time in seconds between retries"
    )
    max_wait_sec: float = Field(
        default=60.0, description="Maximum wait time in seconds between retries"
    )


_TELEMETRY_FIELD_MAX_LENGTH = 4096


class TelemetryConfig(BaseModel):
    """Explicit Finelog export and stable run identity."""

    endpoint: SecretStr = Field(min_length=1, max_length=_TELEMETRY_FIELD_MAX_LENGTH)
    root_run_uid: str = Field(min_length=1, max_length=_TELEMETRY_FIELD_MAX_LENGTH)
    execution_uid: str = Field(min_length=1, max_length=_TELEMETRY_FIELD_MAX_LENGTH)
    serving_job_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=_TELEMETRY_FIELD_MAX_LENGTH,
    )


class ArchiveConfig(BaseModel):
    """A finestore eval archive to write each finished trial into.

    ``root`` is the archive's URL prefix (the run's results directory) and
    ``dataset`` is the label the archive records as a sample's ``task`` (the
    dataset the trials belong to). Present only when a caller opts in to
    dashboard-ready archive output; absent for a plain Harbor run.
    """

    root: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    writer_id: str = "harbor"


class JobConfig(BaseModel):
    # If replay-affecting fields are added or changed here, update JobLock in
    # harbor.models.job.lock so lock.json records the same resolved run input.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_name: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    )
    jobs_dir: UPath = Field(default_factory=lambda: UPath("jobs"))
    artifact_authority: ArtifactAuthorityConfig | None = None
    n_attempts: int = 1
    timeout_multiplier: float = 1.0
    agent_timeout_multiplier: float | None = None
    verifier_timeout_multiplier: float | None = None
    agent_setup_timeout_multiplier: float | None = None
    environment_build_timeout_multiplier: float | None = None
    trial_attempt_timeout_sec: float | None = Field(default=None, gt=0)
    trial_cleanup_grace_sec: float = Field(
        default=DEFAULT_TRIAL_CLEANUP_GRACE_SEC,
        gt=0,
        description=(
            "Seconds to wait for timed-out phase or attempt cancellation before "
            "quarantining unfinished work."
        ),
    )
    debug: bool = Field(default=False, description="Enable debug logging")
    n_concurrent_trials: int = 4
    quiet: bool = Field(default=False, description="Suppress trial progress displays")
    release_trial_payloads_in_memory: bool = Field(
        default=False,
        description=(
            "Drop heavy per-trial payloads (agent rollout_details/metadata, "
            "verifier output, exception diagnostics, and per-step results) from "
            "the in-memory copies a job retains for aggregation, including on "
            "resume. The full result is always persisted to disk first, so "
            "nothing recoverable is lost; only stats inputs (rewards, exception "
            "type, agent_info, token-cost scalars) are kept in RAM. Enable for "
            "long, memory-heavy datagen runs to bound worker RSS. Leave False "
            "for RL, which reads rollout_details from the returned results."
        ),
    )
    telemetry: TelemetryConfig | None = Field(default=None, exclude=True, repr=False)
    archive: ArchiveConfig | None = Field(default=None, repr=False)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    metrics: list[MetricConfig] = Field(default_factory=list)
    agents: list[AgentConfig] = Field(default_factory=lambda: [AgentConfig()])
    user_agent: UserAgentConfig | None = Field(
        default=None,
        description="Optional simulated-user agent and bridge applied to every trial.",
    )
    datasets: list[DatasetConfig] = Field(default_factory=list)
    tasks: list[TaskConfig] = Field(default_factory=list)
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)
    extra_instruction_paths: list[Path] = Field(default_factory=list)

    @field_validator("jobs_dir", mode="before")
    @classmethod
    def _coerce_jobs_dir(cls, v):
        """Accept ``str`` / ``Path`` / ``UPath`` and normalize.

        A ``gs://`` URI routes through fsspec; a local path behaves like
        ``pathlib.Path``.

        Returns a ``str`` rather than a ``UPath`` so that pydantic's chain
        validator for the ``UPath`` field type (``chain[str, UPath(...)]``)
        gets a string input and builds the final ``UPath`` itself. Returning
        a ``UPath`` (or a UPath subclass like ``GCSPath``) directly from
        this ``mode="before"`` validator triggers a
        ``ValidationError: Input should be a valid string`` on resume,
        because pydantic's chain still expects str for the first step —
        encountered on every preempt-retry of a job with a remote
        ``jobs_dir`` (e.g. ``gs://``).
        """
        if isinstance(v, str):
            return v
        if isinstance(v, (Path, UPath)) or hasattr(v, "__fspath__"):
            return str(v)
        return v

    @field_serializer("jobs_dir")
    def _serialize_jobs_dir(self, v: UPath) -> str:
        return str(v)

    @model_validator(mode="before")
    @classmethod
    def _migrate_orchestrator_config(cls, data):
        """Migrate legacy 'orchestrator' config key to top-level fields.

        .. deprecated::
            Use top-level ``n_concurrent_trials``, ``quiet``, and ``retry`` instead.
        """
        if isinstance(data, dict) and "orchestrator" in data:
            warnings.warn(
                "The 'orchestrator' config key is deprecated. "
                "Use top-level 'n_concurrent_trials', 'quiet', and 'retry' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            orch = data.pop("orchestrator")
            if isinstance(orch, dict):
                data.setdefault(
                    "n_concurrent_trials", orch.get("n_concurrent_trials", 4)
                )
                if "quiet" in orch:
                    data.setdefault("quiet", orch["quiet"])
                if "retry" in orch:
                    data.setdefault("retry", orch["retry"])
        return data

    def __eq__(self, other):
        if not isinstance(other, JobConfig):
            return NotImplemented

        # Exclude non-replay identity/logging fields from equality comparison.
        exclude = {"job_name", "debug", "telemetry", "archive"}
        return self.model_dump(exclude=exclude) == other.model_dump(exclude=exclude)
