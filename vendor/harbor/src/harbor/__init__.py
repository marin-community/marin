import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Core classes
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment, ExecResult
    from harbor.job import Job

    # Agent models
    from harbor.models.agent.context import AgentContext
    from harbor.models.agent.name import AgentName
    from harbor.bridges.base import BaseBridge
    from harbor.models.bridge import BridgeConfig, BridgeKind
    from harbor.models.dataset_item import DownloadedDatasetItem
    from harbor.models.benchmark import BenchmarkMetadata, MetricKind, MetricMetadata

    # Enum types
    from harbor.models.environment_type import EnvironmentType

    # Job models
    from harbor.models.job.config import (
        DatasetConfig,
        JobConfig,
        RetryConfig,
    )
    from harbor.models.job.result import AgentDatasetStats, JobResult, JobStats

    # Metric models
    from harbor.models.metric.config import MetricConfig
    from harbor.models.metric.type import MetricType
    from harbor.models.metric.usage_info import UsageInfo

    # Registry models
    from harbor.models.registry import (
        DatasetSpec,
        LocalRegistryInfo,
        Registry,
        RegistryTaskId,
        RemoteRegistryInfo,
    )

    # Task models
    from harbor.models.task.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        VerifierConfig,
    )
    from harbor.models.task.id import GitTaskId, LocalTaskId
    from harbor.models.task.paths import TaskPaths
    from harbor.models.task.task import Task
    from harbor.models.trial.config import (
        AgentConfig as TrialAgentConfig,
    )
    from harbor.models.trial.config import (
        EnvironmentConfig as TrialEnvironmentConfig,
    )
    from harbor.models.trial.config import (
        TaskConfig as TrialTaskConfig,
    )

    # Trial models
    from harbor.models.trial.config import (
        TrialConfig,
        UserAgentConfig,
    )
    from harbor.models.trial.config import (
        VerifierConfig as TrialVerifierConfig,
    )
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.trial.result import (
        AgentInfo,
        ExceptionInfo,
        ModelInfo,
        TimingInfo,
        TrialResult,
    )

    # Verifier models
    from harbor.models.verifier.result import VerifierResult
    from harbor.trial.hooks import TrialEvent, TrialHookEvent
    from harbor.trial.queue import TrialQueue
    from harbor.trial.trial import Trial
    from harbor.verifier.base import BaseVerifier
    from harbor.verifier.verifier import Verifier

__version__ = importlib.metadata.version("harbor")


def _pin_s3_addressing_style() -> None:
    """Pin virtual-hosted (hostname-based) S3 addressing for every s3fs filesystem
    harbor creates, so trace/agent-log writes to the CoreWeave/marin object store
    (``s3://marin-us-east-02a/.../trace_jobs/.../agent``) don't get rejected with
    ``PathStyleRequestNotAllowed`` and flood the RL job's log.

    WHY here (not per-call): harbor addresses remote artifact stores exclusively
    through bare ``upath.UPath("s3://...")`` (jobs_dir/trials_dir/agent_dir), whose
    underlying ``s3fs.S3FileSystem`` is built with DEFAULT storage options — and
    s3fs defaults to path-style addressing for a custom ``endpoint_url``, which the
    CoreWeave store forbids. There is NO per-call ``storage_options`` to thread, and
    botocore has NO env var for ``addressing_style`` (it's read only from the shared
    AWS config file's ``s3`` section). So we set the fsspec-global default
    ``config_kwargs`` for the ``s3`` protocol ONCE at package import, BEFORE the first
    ``S3FileSystem`` is instantiated; every UPath-driven s3fs client then resolves
    ``addressing_style=virtual``. This also repairs the broken ``FSSPEC_S3_CONFIG_KWARGS``
    env form (fsspec stores that suffix-form value as a raw STRING, which s3fs then
    fails on with ``'str' object has no attribute 'copy'``) by replacing it with a
    proper dict. Env-overridable via ``OT_AGENT_S3_ADDRESSING_STYLE`` (default
    ``virtual``; set ``path``/``auto`` for a path-style store such as MinIO, or empty
    to leave botocore's default untouched). Best-effort; never raises at import."""
    import os

    style = os.environ.get("OT_AGENT_S3_ADDRESSING_STYLE", "virtual").strip()
    if not style:
        return
    try:
        import fsspec.config

        s3_conf = fsspec.config.conf.setdefault("s3", {})
        config_kwargs = s3_conf.get("config_kwargs")
        # A prior FSSPEC_S3_CONFIG_KWARGS env leaves this a STRING (broken); replace it.
        if not isinstance(config_kwargs, dict):
            config_kwargs = {}
            s3_conf["config_kwargs"] = config_kwargs
        s3_section = config_kwargs.get("s3")
        if not isinstance(s3_section, dict):
            s3_section = {}
            config_kwargs["s3"] = s3_section
        s3_section["addressing_style"] = style
    except Exception:  # noqa: BLE001 - never break import over a best-effort default
        pass


def _route_s3_endpoint_url_to_client_kwargs() -> None:
    """Move a top-level ``endpoint_url`` in the fsspec ``s3`` config into ``client_kwargs``.

    fsspec loads ``FSSPEC_S3`` (e.g. the CoreWeave/iris-injected
    ``{"endpoint_url": "http://cwlota.com", "client_kwargs": {"region_name": "auto"}}``) into
    ``fsspec.config.conf["s3"]`` with ``endpoint_url`` at the TOP level. Old s3fs forwards every
    unrecognised top-level kwarg to ``botocore.session.Session(**kwargs)``, which does not accept
    ``endpoint_url`` -> ``TypeError: Session.__init__() got an unexpected keyword argument
    'endpoint_url'`` on the first ``UPath("s3://...").exists()``. Its correct home is
    ``client_kwargs`` (forwarded to ``session.create_client``, which DOES accept ``endpoint_url``),
    where ``region_name`` already lives. Relocating it here — the same import-time chokepoint as the
    addressing-style pin, since harbor addresses s3 through bare ``UPath`` with no per-call
    storage_options — makes s3 access work on any s3fs version. Best-effort; never raises at import."""
    try:
        import fsspec.config

        s3_conf = fsspec.config.conf.get("s3")
        if not isinstance(s3_conf, dict) or "endpoint_url" not in s3_conf:
            return
        client_kwargs = s3_conf.get("client_kwargs")
        if not isinstance(client_kwargs, dict):
            client_kwargs = {}
            s3_conf["client_kwargs"] = client_kwargs
        # Drop the stray top-level key; keep an explicit client_kwargs endpoint_url if one exists.
        client_kwargs.setdefault("endpoint_url", s3_conf.pop("endpoint_url"))
    except Exception:  # noqa: BLE001 - never break import over a best-effort default
        pass


_pin_s3_addressing_style()
_route_s3_endpoint_url_to_client_kwargs()


# Lazy imports to avoid loading heavy dependencies at package import time
_LAZY_IMPORTS = {
    # Core classes
    "Job": ("harbor.job", "Job"),
    "Trial": ("harbor.trial.trial", "Trial"),
    "Task": ("harbor.models.task.task", "Task"),
    "BaseAgent": ("harbor.agents.base", "BaseAgent"),
    "BaseBridge": ("harbor.bridges.base", "BaseBridge"),
    "BaseEnvironment": ("harbor.environments.base", "BaseEnvironment"),
    "ExecResult": ("harbor.environments.base", "ExecResult"),
    "BaseVerifier": ("harbor.verifier.base", "BaseVerifier"),
    "Verifier": ("harbor.verifier.verifier", "Verifier"),
    "TrialQueue": ("harbor.trial.queue", "TrialQueue"),
    # Job models
    "JobConfig": ("harbor.models.job.config", "JobConfig"),
    "RetryConfig": ("harbor.models.job.config", "RetryConfig"),
    "DatasetConfig": ("harbor.models.job.config", "DatasetConfig"),
    "LocalDatasetConfig": (
        "harbor.models.job.config",
        "DatasetConfig",
    ),  # deprecated alias
    "RegistryDatasetConfig": (
        "harbor.models.job.config",
        "DatasetConfig",
    ),  # deprecated alias
    "JobResult": ("harbor.models.job.result", "JobResult"),
    "JobStats": ("harbor.models.job.result", "JobStats"),
    "AgentDatasetStats": ("harbor.models.job.result", "AgentDatasetStats"),
    "BenchmarkMetadata": ("harbor.models.benchmark", "BenchmarkMetadata"),
    "MetricMetadata": ("harbor.models.benchmark", "MetricMetadata"),
    "MetricKind": ("harbor.models.benchmark", "MetricKind"),
    # Trial models
    "TrialConfig": ("harbor.models.trial.config", "TrialConfig"),
    "UserAgentConfig": ("harbor.models.trial.config", "UserAgentConfig"),
    "BridgeConfig": ("harbor.models.bridge", "BridgeConfig"),
    "BridgeKind": ("harbor.models.bridge", "BridgeKind"),
    "TrialAgentConfig": ("harbor.models.trial.config", "AgentConfig"),
    "TrialEnvironmentConfig": ("harbor.models.trial.config", "EnvironmentConfig"),
    "TrialVerifierConfig": ("harbor.models.trial.config", "VerifierConfig"),
    "TrialTaskConfig": ("harbor.models.trial.config", "TaskConfig"),
    "TrialResult": ("harbor.models.trial.result", "TrialResult"),
    "TimingInfo": ("harbor.models.trial.result", "TimingInfo"),
    "ExceptionInfo": ("harbor.models.trial.result", "ExceptionInfo"),
    "ModelInfo": ("harbor.models.trial.result", "ModelInfo"),
    "AgentInfo": ("harbor.models.trial.result", "AgentInfo"),
    "TrialPaths": ("harbor.models.trial.paths", "TrialPaths"),
    # Task models
    "TaskConfig": ("harbor.models.task.config", "TaskConfig"),
    "EnvironmentConfig": ("harbor.models.task.config", "EnvironmentConfig"),
    "AgentConfig": ("harbor.models.task.config", "AgentConfig"),
    "VerifierConfig": ("harbor.models.task.config", "VerifierConfig"),
    "TaskPaths": ("harbor.models.task.paths", "TaskPaths"),
    "LocalTaskId": ("harbor.models.task.id", "LocalTaskId"),
    "GitTaskId": ("harbor.models.task.id", "GitTaskId"),
    # Agent models
    "AgentContext": ("harbor.models.agent.context", "AgentContext"),
    "AgentName": ("harbor.models.agent.name", "AgentName"),
    # Verifier models
    "VerifierResult": ("harbor.models.verifier.result", "VerifierResult"),
    # Enum types
    "EnvironmentType": ("harbor.models.environment_type", "EnvironmentType"),
    "TrialEvent": ("harbor.trial.hooks", "TrialEvent"),
    "TrialHookEvent": ("harbor.trial.hooks", "TrialHookEvent"),
    # Registry models
    "LocalRegistryInfo": ("harbor.models.registry", "LocalRegistryInfo"),
    "RemoteRegistryInfo": ("harbor.models.registry", "RemoteRegistryInfo"),
    "RegistryTaskId": ("harbor.models.registry", "RegistryTaskId"),
    "DatasetSpec": ("harbor.models.registry", "DatasetSpec"),
    "Registry": ("harbor.models.registry", "Registry"),
    "DownloadedDatasetItem": ("harbor.models.dataset_item", "DownloadedDatasetItem"),
    # Metric models
    "MetricConfig": ("harbor.models.metric.config", "MetricConfig"),
    "MetricType": ("harbor.models.metric.type", "MetricType"),
    "UsageInfo": ("harbor.models.metric.usage_info", "UsageInfo"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core classes
    "Job",
    "Trial",
    "Task",
    "BaseAgent",
    "BaseBridge",
    "BaseEnvironment",
    "ExecResult",
    "BaseVerifier",
    "Verifier",
    "TrialQueue",
    # Job models
    "JobConfig",
    "RetryConfig",
    "DatasetConfig",
    "LocalDatasetConfig",  # deprecated alias
    "RegistryDatasetConfig",  # deprecated alias
    "JobResult",
    "JobStats",
    "AgentDatasetStats",
    "BenchmarkMetadata",
    "MetricMetadata",
    "MetricKind",
    # Trial models
    "TrialConfig",
    "UserAgentConfig",
    "BridgeConfig",
    "BridgeKind",
    "TrialAgentConfig",
    "TrialEnvironmentConfig",
    "TrialVerifierConfig",
    "TrialTaskConfig",
    "TrialResult",
    "TimingInfo",
    "ExceptionInfo",
    "ModelInfo",
    "AgentInfo",
    "TrialPaths",
    # Task models
    "TaskConfig",
    "EnvironmentConfig",
    "AgentConfig",
    "VerifierConfig",
    "TaskPaths",
    "LocalTaskId",
    "GitTaskId",
    # Agent models
    "AgentContext",
    "AgentName",
    # Verifier models
    "VerifierResult",
    # Enum types
    "EnvironmentType",
    "TrialEvent",
    "TrialHookEvent",
    # Registry models
    "LocalRegistryInfo",
    "RemoteRegistryInfo",
    "RegistryTaskId",
    "DatasetSpec",
    "Registry",
    "DownloadedDatasetItem",
    # Metric models
    "MetricConfig",
    "MetricType",
    "UsageInfo",
]
