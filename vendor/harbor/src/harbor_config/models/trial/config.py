import warnings
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, NotRequired, TypedDict
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from shortuuid import ShortUUID
from upath import UPath

from harbor_config.env import templatize_sensitive_env
from harbor_config.model_name import split_provider_model_name
from harbor_config.models.agent.name import AgentName
from harbor_config.models.bridge import BridgeConfig
from harbor_config.models.environment_type import EnvironmentType
from harbor_config.models.task.config import ArtifactConfig, MCPServerConfig, TpuSpec
from harbor_config.models.task.id import GitTaskId, LocalTaskId, PackageTaskId

DEFAULT_TRIAL_CLEANUP_GRACE_SEC = 30.0


class ArtifactAuthorityConfig(BaseModel):
    """Expected launcher authority for a protected local artifact root."""

    authority_id: str = Field(min_length=1)
    record_path: Path
    mode: Literal["require", "warn"] = "require"


class ServiceVolumeBind(TypedDict):
    create_host_path: NotRequired[Literal[False]]


class ServiceVolumeVolume(TypedDict):
    subpath: NotRequired[str]


class ServiceVolumeImage(TypedDict):
    subpath: NotRequired[str]


class ServiceVolumeConfig(TypedDict):
    type: Literal["bind", "volume", "image"]
    source: str
    target: str
    read_only: NotRequired[Literal[True]]
    bind: NotRequired[ServiceVolumeBind]
    volume: NotRequired[ServiceVolumeVolume]
    image: NotRequired[ServiceVolumeImage]


class ResourceMode(str, Enum):
    AUTO = "auto"
    LIMIT = "limit"
    REQUEST = "request"
    GUARANTEE = "guarantee"
    IGNORE = "ignore"


class ResolvedStepTimeouts(BaseModel):
    """Effective agent and verifier deadlines for one task step."""

    name: str
    agent: float | None
    verifier: float | None


class ResolvedTimeouts(BaseModel):
    """Effective phase deadlines persisted with a trial configuration."""

    agent: float | None
    verifier: float
    agent_setup: float
    environment_build: float
    trial_attempt: float | None = None
    trial_cleanup_grace: float | None = None
    steps: list[ResolvedStepTimeouts] = Field(default_factory=list)


class AgentConfig(BaseModel):
    name: str | None = None
    import_path: str | None = None
    model_name: str | None = Field(
        default=None,
        description="Immutable model identity recorded in results and used for resume.",
    )
    model_alias: str | None = Field(
        default=None,
        description="Mutable serving name used to route model requests for this run.",
    )
    skills: list[Path] = Field(default_factory=list)
    mode: Literal["container", "local"] = "container"
    override_timeout_sec: float | None = None
    override_setup_timeout_sec: float | None = None
    max_timeout_sec: float | None = Field(
        default=None,
        description=(
            "Hard ceiling on the final agent timeout after applying the effective "
            "timeout multiplier."
        ),
    )
    kwargs: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    # When True (default; preserves current behavior) the host-side agent logs are
    # re-uploaded BACK into a non-mounted agent environment after the agent phase.
    # This push-back is best-effort and is NOT consumed downstream: the verifier
    # reads only the verifier dir, artifact collection uses the artifacts dir, and
    # the trace/rollout bundle reads the authoritative HOST copy (already downloaded
    # via _download_agent_logs). Set False to skip the round-trip and avoid its
    # retry-churn on high-latency (object-store) environments. Read by the Trial
    # (Trial._upload_agent_logs); NOT forwarded to the agent constructor. Mounted
    # environments ignore this (they use a bind mount, no round-trip).
    upload_agent_logs: bool = True

    @field_serializer("env")
    @classmethod
    def _serialize_env(cls, env: dict[str, str]) -> dict[str, str]:
        return templatize_sensitive_env(env)

    @model_validator(mode="after")
    def require_model_name_for_alias(self):
        if self.model_alias is not None and self.model_name is None:
            raise ValueError("model_alias requires an immutable model_name")
        return self

    @property
    def model_name_without_provider(self) -> str | None:
        """Return the immutable model name with its provider prefix removed."""
        if self.model_name is None:
            return None
        _, model_name = split_provider_model_name(self.model_name)
        return model_name

    @field_serializer("kwargs")
    @classmethod
    def _serialize_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Replace non-serializable kwargs on dump.

        Some callers (e.g. SkyRL's terminal_bench generator with PRM
        `teacher_hint`) stash a Python callable in kwargs (e.g. a
        ``turn_callback`` returned by ``prm_instance.as_turn_callback()``).
        Pydantic v2's default serializer raises
        ``PydanticSerializationError: Unable to serialize unknown type:
        <class 'function'>`` when it tries to dump such values to JSON.

        Replace any non-JSON-serializable value with a string placeholder so
        ``model_dump()`` and JSON dumps succeed. Original values remain on the
        live in-memory instance for the agent factory.
        """
        import json

        def _safe(v: Any) -> Any:
            try:
                json.dumps(v)
                return v
            except (TypeError, ValueError):
                return f"<non-serializable {type(v).__name__}>"

        return {k: _safe(v) for k, v in kwargs.items()}

    @model_validator(mode="after")
    def set_default_name(self):
        if self.name is None and self.import_path is None:
            self.name = AgentName.ORACLE.value
        return self


class UserAgentConfig(AgentConfig):
    user_persona_path: Path | None = None
    user_prompt_template_path: Path | None = None
    bridge: BridgeConfig


class EnvironmentConfig(BaseModel):
    type: EnvironmentType | None = None
    import_path: str | None = None
    force_build: bool = False
    delete: bool = True
    cpu_enforcement_policy: ResourceMode = ResourceMode.AUTO
    memory_enforcement_policy: ResourceMode = ResourceMode.AUTO
    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None
    override_gpus: int | None = None
    override_tpu: TpuSpec | None = None
    suppress_override_warnings: bool = Field(
        default=False,
        exclude=True,
        description="Deprecated; has no effect.",
    )
    mounts: list[ServiceVolumeConfig] | None = None
    extra_docker_compose: list[Path] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_mounts_json(cls, data: Any) -> Any:
        """Accept the legacy ``mounts_json`` input key as an alias for ``mounts``."""
        if isinstance(data, dict) and "mounts_json" in data:
            legacy = data.pop("mounts_json")
            if "mounts" not in data:
                warnings.warn(
                    "EnvironmentConfig.mounts_json is deprecated; "
                    "use 'mounts' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                data["mounts"] = legacy
        if isinstance(data, dict) and "suppress_override_warnings" in data:
            warnings.warn(
                "EnvironmentConfig.suppress_override_warnings is deprecated and "
                "has no effect; resource override warnings are no longer emitted.",
                DeprecationWarning,
                stacklevel=2,
            )
        return data

    @field_validator(
        "cpu_enforcement_policy",
        "memory_enforcement_policy",
        mode="before",
    )
    @classmethod
    def _normalize_resource_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @property
    def mounts_json(self) -> list[ServiceVolumeConfig] | None:
        """Deprecated alias for :attr:`mounts`. Will be removed in a future release."""
        warnings.warn(
            "EnvironmentConfig.mounts_json is deprecated; use 'mounts' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mounts

    @mounts_json.setter
    def mounts_json(self, value: list[ServiceVolumeConfig] | None) -> None:
        warnings.warn(
            "EnvironmentConfig.mounts_json is deprecated; use 'mounts' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.mounts = value

    @field_serializer("env")
    @classmethod
    def _serialize_env(cls, env: dict[str, str]) -> dict[str, str]:
        return templatize_sensitive_env(env)

    @field_validator("env", mode="before")
    @classmethod
    def _env_list_to_dict(cls, v: list[str] | dict[str, str]) -> dict[str, str]:
        """Accept legacy YAML list of KEY=VALUE strings or the canonical dict."""
        if isinstance(v, dict):
            return v
        if isinstance(v, list):
            warnings.warn(
                "List-style 'environment.env' is deprecated. "
                "Use a mapping of env var names to values instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            out: dict[str, str] = {}
            for item in v:
                if "=" not in item:
                    raise ValueError(
                        f"Invalid env var format: {item!r}. Expected KEY=VALUE"
                    )
                key, value = item.split("=", 1)
                out[key.strip()] = value.strip()
            return out
        return v

    @model_validator(mode="after")
    def set_default_type(self):
        if self.type is None and self.import_path is None:
            self.type = EnvironmentType.DOCKER
        return self


class VerifierConfig(BaseModel):
    override_timeout_sec: float | None = None
    max_timeout_sec: float | None = Field(
        default=None,
        description=(
            "Hard ceiling on the final verifier timeout after applying the "
            "effective timeout multiplier."
        ),
    )
    env: dict[str, str] = Field(default_factory=dict)
    import_path: str | None = Field(default=None, exclude_if=lambda v: v is None)
    kwargs: dict[str, Any] = Field(default_factory=dict, exclude_if=lambda v: not v)
    disable: bool = False

    @field_serializer("env")
    @classmethod
    def _serialize_env(cls, env: dict[str, str]) -> dict[str, str]:
        return templatize_sensitive_env(env)

    def identity_fingerprint(self) -> dict[str, Any]:
        """Compare judge hosting service and settings without runtime URL routing."""
        dump = self.model_dump()
        base_url = self.env.get("OPENAI_BASE_URL")
        if base_url and self.env.get("MODEL_NAME"):
            url = urlsplit(base_url)
            if url.hostname:
                dump["env"]["OPENAI_BASE_URL"] = (url.scheme.lower(), url.hostname)
        return dump


class TaskConfig(BaseModel):
    path: Path | None = None
    git_url: str | None = None
    git_commit_id: str | None = None
    name: str | None = None  # org/name format (e.g. "harbor/hello-world")
    ref: str | None = (
        None  # tag, revision, or digest (e.g. "latest", "3", "sha256:...")
    )
    overwrite: bool = False
    download_dir: Path | None = None
    source: str | None = None

    @model_validator(mode="after")
    def validate_task_source(self):
        has_path = self.path is not None
        has_package = self.name is not None

        if not has_path and not has_package:
            raise ValueError("Either 'path' or 'name' must be set.")

        if has_path and has_package:
            raise ValueError("Cannot set both 'path' and 'name'.")

        if self.ref is not None and not has_package:
            raise ValueError("'ref' requires 'name' to be set.")

        if self.git_commit_id is not None and self.git_url is None:
            raise ValueError("'git_commit_id' requires 'git_url' to be set.")

        return self

    def is_git_task(self) -> bool:
        """Check if this is a Git-based task."""
        return self.git_url is not None

    def is_package_task(self) -> bool:
        """Check if this is a package-based task."""
        return self.name is not None

    def get_task_id(self) -> LocalTaskId | GitTaskId | PackageTaskId:
        """Get the appropriate TaskId based on configuration."""
        if self.is_package_task():
            assert self.name is not None
            org, name = self.name.split("/", 1)
            return PackageTaskId(org=org, name=name, ref=self.ref)
        if self.is_git_task():
            if self.git_url is None or self.path is None:
                raise ValueError("git_url and path must be set for a git task.")

            return GitTaskId(
                git_url=self.git_url, git_commit_id=self.git_commit_id, path=self.path
            )
        if self.path is None:
            raise ValueError("path must be set for a local task.")
        return LocalTaskId(path=self.path)

    def get_local_path(self) -> Path:
        return self.get_task_id().get_local_path()


class TrialConfig(BaseModel):
    # If replay-affecting fields are added or changed here, update TrialLock in
    # harbor.models.job.lock so lock.json records the same resolved run input.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    EQUALITY_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"trial_name", "job_id", "resolved_timeouts", "artifact_authority"}
    )

    task: TaskConfig
    trial_name: str = ""
    trials_dir: UPath = Field(default_factory=lambda: UPath("trials"))
    artifact_authority: ArtifactAuthorityConfig | None = None
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
    resolved_timeouts: ResolvedTimeouts | None = None
    agent: AgentConfig = Field(default_factory=AgentConfig)
    user_agent: UserAgentConfig | None = Field(
        default=None,
        description=(
            "Optional simulated-user agent that communicates with the primary "
            "agent through its configured bridge."
        ),
    )
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)
    extra_instruction_paths: list[Path] = Field(default_factory=list)
    job_id: UUID | None = None

    @field_validator("trials_dir", mode="before")
    @classmethod
    def _coerce_trials_dir(cls, v):
        """Accept ``str`` / ``Path`` / ``UPath`` and normalize.

        Returns a ``str`` rather than a ``UPath`` so pydantic's chain
        validator for the ``UPath`` field type (``chain[str, UPath(...)]``)
        gets a string and builds the final ``UPath`` itself. Returning a
        ``UPath`` subclass (e.g. ``GCSPath``) directly from this
        ``mode="before"`` validator triggers
        ``ValidationError: Input should be a valid string`` when harbor
        deserializes a remote (``gs://``) trial config on resume — the same
        bug fixed for ``JobConfig.jobs_dir``. Hit on every preempt-resume of
        a job whose trials_dir is remote: ``_maybe_init_existing_job`` calls
        ``TrialConfig.model_validate_json`` per existing trial and crashes.
        """
        if isinstance(v, str):
            return v
        if isinstance(v, (Path, UPath)) or hasattr(v, "__fspath__"):
            return str(v)
        return v

    @field_serializer("trials_dir")
    def _serialize_trials_dir(self, v: UPath) -> str:
        return str(v)

    def __eq__(self, other):
        if not isinstance(other, TrialConfig):
            return NotImplemented

        return self.model_dump(exclude=self.EQUALITY_EXCLUDED_FIELDS) == (
            other.model_dump(exclude=self.EQUALITY_EXCLUDED_FIELDS)
        )

    # Agent-config keys that describe HOW a trial is SERVED at runtime (which
    # served-model alias / endpoint / sampling-transport knobs the agent dials)
    # rather than WHAT the trial IS. They drift across a resume that lands on a
    # new serve node or after a serve-geometry change (e.g. a `--served-model-name`
    # alias, an `api_base` rewrite to the live MASTER_ADDR) WITHOUT changing the
    # task, the prompt, the verifier, or the trial's identity. The resume
    # reconciliation must ignore these so partial progress survives benign drift.
    _SERVING_AGENT_FIELDS: ClassVar[set[str]] = {
        "model_alias",
        "kwargs",
        "env",
        "override_timeout_sec",
        "override_setup_timeout_sec",
        "max_timeout_sec",
    }
    # Top-level TrialConfig fields that record runtime serving, artifact wiring,
    # or derived execution metadata and do not change a trial's identity.
    _SERVING_TOP_FIELDS: ClassVar[set[str]] = {
        "trial_name",
        "job_id",
        "trials_dir",
        "timeout_multiplier",
        "agent_timeout_multiplier",
        "verifier_timeout_multiplier",
        "agent_setup_timeout_multiplier",
        "environment_build_timeout_multiplier",
        "resolved_timeouts",
        "artifacts",
        "artifact_authority",
    }
    # Environment ``kwargs`` keys that are cache-acquisition / provisioning
    # knobs: they change HOW a sandbox is provisioned and cached, not WHAT a
    # trial evaluates. The Daytona snapshot trio (``auto_snapshot`` and the
    # explicit/global snapshot switches) is the motivating case — trials
    # launched before ``auto_snapshot`` was enabled must still resume (be kept,
    # not re-run) once the flag is turned on. The lock equality key reuses this
    # same set (see ``_environment_equality_key`` in harbor.models.job.lock) so
    # the two identity checks cannot drift.
    _PROVISIONING_ENV_KWARG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"auto_snapshot", "snapshot_template_name", "assume_global_snapshot"}
    )

    def identity_fingerprint(self) -> dict[str, Any]:
        """A dump of only the TASK-IDENTITY fields — what a trial *is*.

        Two trials with the same fingerprint run the *same* evaluation (same
        task / benchmark / dataset, same agent program + model identity, same
        environment build, same verifier), even if served via a different
        endpoint alias or with different timeout scaling. Used by the resume
        reconciliation to keep partial progress across benign serving-config
        drift while still rejecting a genuinely different eval.
        """
        dump = self.model_dump(exclude=self._SERVING_TOP_FIELDS)
        dump["verifier"] = self.verifier.identity_fingerprint()
        task = dump.get("task")
        if isinstance(task, dict) and task.get("source") == "hf_dataset":
            # Harbor materializes a pinned Hugging Face dataset into a fresh
            # temporary directory for each job invocation. Its task directory
            # name is stable; the parent workdir is not part of evaluation
            # identity.
            task_path = task.get("path")
            if task_path is not None:
                task["path"] = Path(task_path).name
        agent = dump.get("agent")
        if isinstance(agent, dict):
            for key in self._SERVING_AGENT_FIELDS:
                agent.pop(key, None)
        environment = dump.get("environment")
        if isinstance(environment, dict):
            env_kwargs = environment.get("kwargs")
            if isinstance(env_kwargs, dict):
                for key in self._PROVISIONING_ENV_KWARG_KEYS:
                    env_kwargs.pop(key, None)
        return dump

    def matches_identity(self, other: "TrialConfig") -> bool:
        """True iff `other` is the SAME evaluation (ignoring serving drift)."""
        return self.identity_fingerprint() == other.identity_fingerprint()

    @model_validator(mode="after")
    def set_default_trial_name(self):
        if not self.trial_name:
            self.trial_name = self.generate_trial_name()
        return self

    @model_validator(mode="after")
    def _validate_user_agent_mode(self):
        if self.user_agent is not None and self.user_agent.mode != "container":
            raise ValueError("user_agent requires container mode")
        return self

    def generate_trial_name(self):
        task_id = self.task.get_task_id()
        task_name = task_id.get_name().split("/")[-1]
        return f"{task_name[:32].rstrip('_-')}__{ShortUUID().random(length=7)}"
