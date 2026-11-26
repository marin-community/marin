# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight inference serving configuration.

These dataclasses are safe to construct in CPU coordinators and CLI processes.
Accelerator-heavy serving implementations translate them inside worker jobs.
"""

import json
import re
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from fray.types import CpuConfig, EnvironmentConfig, GpuConfig, ResourceConfig, TpuConfig
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from marin.external_dependencies import VllmGpuRelease
from marin.inference.vllm_release_schema import CUDA_TOOLCHAIN_VERSION_BY_BACKEND

# Worker and isolated vLLM environments use one Python version so cloudpickle
# and the launched callable stay compatible.
WORKER_PYTHON_VERSION = "3.12"
# Stock CUDA vLLM runs in an isolated uv-tool environment and does not
# participate in Marin's workspace dependency resolution.
DEFAULT_CUDA_VLLM_VERSION = "0.25.1"
VLLM_METRIC_PREFIX = "vllm:"
VLLM_DATA_PARALLEL_SIZE_FLAGS = ("--data-parallel-size", "--data_parallel_size", "-dp")
VLLM_TOPOLOGY_ALIASES = {
    "-tp": "--tensor-parallel-size",
    "-pp": "--pipeline-parallel-size",
    "-dp": "--data-parallel-size",
    "-dpl": "--data-parallel-size-local",
    "-dpn": "--data-parallel-rank",
    "-dpr": "--data-parallel-start-rank",
    "-dpa": "--data-parallel-address",
    "-dpp": "--data-parallel-rpc-port",
    "-dpb": "--data-parallel-backend",
    "-n": "--nnodes",
    "-dph": "--data-parallel-hybrid-lb",
    "-dpe": "--data-parallel-external-lb",
    "-dpm": "--data-parallel-multi-port-external-lb",
}
VLLM_TOPOLOGY_OPTIONS = frozenset(
    {
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        *VLLM_DATA_PARALLEL_SIZE_FLAGS,
        "--data-parallel-size-local",
        "--data-parallel-rank",
        "--data-parallel-start-rank",
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--data-parallel-backend",
        "--data-parallel-hybrid-lb",
        "--data-parallel-external-lb",
        "--data-parallel-multi-port-external-lb",
        "--nnodes",
        "--node-rank",
        "--master-addr",
        "--master-port",
        "--device-ids",
        "--headless",
    }
)


def resolve_tokenizer_revision(
    revision: str | None,
    tokenizer: str | None,
    tokenizer_revision: str | None,
) -> str | None:
    """Resolve the revision owned by an explicit or model-default tokenizer."""
    if tokenizer_revision is not None:
        return tokenizer_revision
    if tokenizer is None:
        return revision
    return None


def normalize_vllm_option(option: str) -> str:
    """Normalize known topology aliases and long-option underscores."""
    if option.startswith("--"):
        option = option.replace("_", "-")
    return VLLM_TOPOLOGY_ALIASES.get(option, option)


def vllm_option_flags(option: str) -> tuple[str, ...]:
    """Return a canonical option and its known short aliases."""
    return (option, *(alias for alias, canonical in VLLM_TOPOLOGY_ALIASES.items() if canonical == option))


def validate_serving_geometry_args(args: tuple[str, ...]) -> None:
    """Reject flags owned by Iris serving geometry."""
    for arg in args:
        option = normalize_vllm_option(arg.partition("=")[0])
        if option.startswith("-n") and option[2:].lstrip("-").isdigit():
            option = "--nnodes"
        if option.startswith("--no-"):
            option = "--" + option.removeprefix("--no-")
        if option in VLLM_TOPOLOGY_OPTIONS:
            raise ValueError(f"serving geometry owns {option}; remove it from extra args")


@dataclass(frozen=True, kw_only=True)
class ServingGeometry:
    """GPU layout for one Iris serving instance.

    Attributes:
        tensor_parallel_size: GPUs per replica per pipeline stage, within one task.
        data_parallel_size: Global replicas, also the router target count.
        pipeline_parallel_size: Stages per replica, across tasks.
        gpus_per_task: GPUs per Iris task, on one host.

    Local DP = GPUs/task / TP; total GPUs = TP * DP * PP;
    tasks = total GPUs / GPUs/task. TP must divide GPUs/task;
    local DP must divide global DP.

    With PP>1, the launcher requires all DP ranks on each stage's task.
    """

    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    gpus_per_task: int

    def __post_init__(self) -> None:
        if min(self.tensor_parallel_size, self.data_parallel_size, self.pipeline_parallel_size, self.gpus_per_task) < 1:
            raise ValueError("parallel sizes and gpus_per_task must be positive")
        if self.gpus_per_task % self.tensor_parallel_size:
            raise ValueError("gpus_per_task must be divisible by tensor_parallel_size")
        local_dp = self.local_data_parallel_size
        if self.data_parallel_size % local_dp:
            raise ValueError("local data parallel size must evenly divide data_parallel_size")
        if self.pipeline_parallel_size > 1 and local_dp != self.data_parallel_size:
            raise ValueError("pipeline parallelism requires all data-parallel ranks on each host")

    @property
    def local_data_parallel_size(self) -> int:
        return self.gpus_per_task // self.tensor_parallel_size

    @property
    def task_count(self) -> int:
        return self.total_gpus // self.gpus_per_task

    @property
    def total_gpus(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size * self.pipeline_parallel_size

    @property
    def label(self) -> str:
        return f"PP{self.pipeline_parallel_size} TP{self.tensor_parallel_size} DP{self.data_parallel_size}"


@dataclass(frozen=True)
class EffectiveServing:
    """Resolved serving settings reported by a running inference endpoint."""

    tensor_parallel_size: int | None
    data_parallel_size: int | None
    pipeline_parallel_size: int
    task_count: int
    max_model_len: int | None


# This standard set includes families consumed by Marin's inference dashboards or needed for
# basic serving diagnosis: request volume, latency, scheduler pressure, KV-cache use, and
# preemption. Selection is made per complete Prometheus family and was sized against Marin's
# pinned GPU and TPU vLLM definitions using representative label cardinalities. Add a family
# only for a concrete consumer, and recheck that the complete selected scrape fits the limit.
STANDARD_VLLM_METRIC_FAMILIES = frozenset(
    {
        "vllm:e2e_request_latency_seconds",
        "vllm:generation_tokens",
        "vllm:inter_token_latency_seconds",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prompt_tokens",
        "vllm:request_queue_time_seconds",
        "vllm:request_success",
        "vllm:request_time_per_output_token_seconds",
        "vllm:time_to_first_token_seconds",
    }
)


def _normalize_vllm_metric_families(families: object, *, source: str) -> frozenset[str]:
    if not isinstance(families, (list, tuple, frozenset)) or not all(isinstance(family, str) for family in families):
        raise ValueError(f"Invalid vLLM metrics config {source}: every family must be a string")
    invalid = [family for family in families if not family.startswith(VLLM_METRIC_PREFIX)]
    if invalid:
        raise ValueError(f"Invalid vLLM metrics config {source}: every family must start with 'vllm:'")
    return frozenset(families)


def _parse_vllm_metric_families(contents: bytes, *, source: str) -> frozenset[str]:
    try:
        document = tomllib.loads(contents.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid vLLM metrics config {source}: {exc}") from exc
    if set(document) != {"families"} or not isinstance(document["families"], list):
        raise ValueError(f"Invalid vLLM metrics config {source}: expected one 'families' array")
    return _normalize_vllm_metric_families(document["families"], source=source)


def load_vllm_metric_family_additions(path: Path | None) -> frozenset[str]:
    """Load optional additions for the standard contract."""
    if path is None:
        return frozenset()
    return _parse_vllm_metric_families(StoragePath(str(path)).read_bytes(), source=str(path))


class VllmLauncherType(StrEnum):
    PREINSTALLED = "preinstalled"
    CUDA = "cuda"
    TPU = "tpu"


UPSTREAM_VLLM_WHEEL_INDEX_BASE = "https://wheels.vllm.ai"
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class UpstreamVllmNightly:
    """An upstream vLLM main commit whose per-commit wheel index supplies the pinned ``vllm_version``.

    ``torch_backend`` is the CUDA build the commit's wheels were compiled against, for example
    ``cu132``; it selects the matching PyTorch index. ``cuda_toolchain_version`` pins the CUDA
    compiler packages to the toolkit that PyTorch build requires, for example ``13.2.78``.
    """

    commit: str
    torch_backend: str
    cuda_toolchain_version: str

    def __post_init__(self) -> None:
        if not _GIT_COMMIT.fullmatch(self.commit):
            raise ValueError(f"vLLM nightly commit must be a full 40-character lowercase SHA, got {self.commit!r}")
        if self.torch_backend not in CUDA_TOOLCHAIN_VERSION_BY_BACKEND:
            raise ValueError(
                f"vLLM nightly torch_backend must be one of {sorted(CUDA_TOOLCHAIN_VERSION_BY_BACKEND)}, "
                f"got {self.torch_backend!r}"
            )
        cuda_minor = CUDA_TOOLCHAIN_VERSION_BY_BACKEND[self.torch_backend].rsplit(".", 1)[0]
        if not self.cuda_toolchain_version.startswith(f"{cuda_minor}."):
            raise ValueError(
                f"vLLM nightly cuda_toolchain_version {self.cuda_toolchain_version!r} does not match "
                f"{self.torch_backend} (CUDA {cuda_minor})"
            )

    @property
    def index_url(self) -> str:
        return f"{UPSTREAM_VLLM_WHEEL_INDEX_BASE}/{self.commit}/"


class VllmSource(StrEnum):
    UPSTREAM = "upstream"
    MARIN_FORK = "marin_fork"


class VllmCompilationCacheMode(StrEnum):
    MANAGED = "managed"
    CALLER_MANAGED = "caller_managed"


@dataclass(frozen=True)
class ResolvedModelLocator:
    """An immutable model source resolved before inference is launched."""

    uri: str
    identity: str

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("resolved model URI must not be empty")
        if not self.identity:
            raise ValueError("resolved model identity must not be empty")


class SpeculativeMethod(StrEnum):
    """Speculative decoding methods supported by Marin's vLLM launcher."""

    EAGLE3 = "eagle3"


class ObjectStoreLoadMode(StrEnum):
    """How an inference worker makes object-store weights available to its engine."""

    STREAM = "stream"
    STAGE_LOCAL = "stage_local"


@dataclass(frozen=True)
class SpeculativeServingConfig:
    """A resolved draft model and its vLLM speculative-decoding policy."""

    method: SpeculativeMethod
    model: ResolvedModelLocator
    num_speculative_tokens: int

    def __post_init__(self) -> None:
        if self.num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be positive")

    def vllm_argument(self) -> str:
        """Return the JSON value accepted by vLLM's ``--speculative-config``."""
        return json.dumps(
            {
                "method": self.method.value,
                "model": self.model.uri,
                "num_speculative_tokens": self.num_speculative_tokens,
            },
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class ServedModelConfig:
    """Model and tokenizer inputs for local or Iris-backed serving.

    ``revision`` pins ``weights``. ``tokenizer_revision`` pins an explicitly
    configured ``tokenizer``. When ``tokenizer`` is omitted, it defaults to the
    original weights repository and may inherit its revision.

    Omitted ``tensor_parallel_size`` uses geometry, or is inferred from the model
    and chips. An explicit value must match geometry.
    """

    weights: str
    revision: str | None = None
    api_model: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    tensor_parallel_size: int | None = None
    chat_template_content: str | None = None
    object_store_load_mode: ObjectStoreLoadMode = ObjectStoreLoadMode.STREAM

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights must not be empty")
        if self.api_model == "":
            raise ValueError("api_model must not be empty")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.tensor_parallel_size is not None and self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")

    @property
    def model_id(self) -> str:
        """Model identifier accepted by the served OpenAI endpoint."""
        return self.api_model or self.weights

    @property
    def effective_tokenizer_revision(self) -> str | None:
        """Revision owned by the configured or default tokenizer repository."""
        return resolve_tokenizer_revision(self.revision, self.tokenizer, self.tokenizer_revision)


@dataclass
class InferenceModelConfig:
    """Model and engine arguments consumed by an in-process vLLM server."""

    name: str
    path: str | None
    engine_kwargs: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class VllmRouterConfig:
    """Consistent hashing with no HTTP retries.

    Missing session IDs use the upstream fallback; identical bodies may hit a busy rank:
    https://github.com/vllm-project/router/blob/v0.1.16/docs/load_balancing/README.md#consistent-hash
    """

    request_timeout_seconds: int = 14400

    def __post_init__(self) -> None:
        if self.request_timeout_seconds < 1:
            raise ValueError("router request timeout must be positive")


@dataclass(frozen=True)
class VllmEngineConfig:
    """vLLM runtime and native arguments.

    Iris geometry owns topology in multi-task and routed deployments.
    Single-task serving without a router allows native overrides.
    vLLM YAML ``--config`` is unsupported.
    """

    launcher: VllmLauncherType = VllmLauncherType.PREINSTALLED
    source: VllmSource = VllmSource.UPSTREAM
    version: str | None = None
    nightly: UpstreamVllmNightly | None = None
    release: VllmGpuRelease | None = None
    extra_requirements: tuple[str, ...] = ()
    compilation_cache: VllmCompilationCacheMode = VllmCompilationCacheMode.MANAGED
    startup_timeout_seconds: int = 1800
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    extra_args: tuple[str, ...] = ()
    extra_metric_families: frozenset[str] = frozenset()
    speculative: SpeculativeServingConfig | None = None
    router: VllmRouterConfig | None = None

    def __post_init__(self) -> None:
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs is not None and self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.source is VllmSource.MARIN_FORK and self.launcher is not VllmLauncherType.CUDA:
            raise ValueError("the Marin vLLM fork source requires the CUDA launcher")
        if self.release is not None and self.source is not VllmSource.MARIN_FORK:
            raise ValueError("a vLLM release pin requires the Marin CUDA fork")
        if self.nightly is not None and (self.source is not VllmSource.UPSTREAM or self.version is None):
            raise ValueError("a vLLM nightly pin requires upstream vLLM and an exact version")
        if self.router is not None and self.launcher is VllmLauncherType.TPU:
            raise ValueError("session routing is supported by CUDA vLLM")
        extra_metric_families = _normalize_vllm_metric_families(
            self.extra_metric_families,
            source="VllmEngineConfig.extra_metric_families",
        )
        object.__setattr__(self, "extra_metric_families", extra_metric_families)


@dataclass(frozen=True)
class LevanterEngineConfig:
    max_seqs: int = 16
    page_size: int = 128
    hbm_utilization: float = 0.8

    def __post_init__(self) -> None:
        if self.max_seqs <= 0:
            raise ValueError("max_seqs must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if not 0 < self.hbm_utilization <= 1:
            raise ValueError("hbm_utilization must be in (0, 1]")


InferenceEngineConfig = VllmEngineConfig | LevanterEngineConfig


def uses_native_vllm_group(geometry: ServingGeometry | None) -> bool:
    """Coordinate native vLLM when geometry spans several Iris tasks."""
    return geometry is not None and geometry.task_count > 1


@dataclass(frozen=True)
class InferenceProxyConfig:
    port: int = 0
    request_timeout_seconds: float = 300.0
    readiness_timeout_seconds: float = 300.0
    max_pending_requests: int = 256
    response_fetch_batch_size: int = 64
    ignored_request_fields: tuple[str, ...] = ()
    server_start_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.port < 0:
            raise ValueError("port must not be negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.readiness_timeout_seconds <= 0:
            raise ValueError("readiness_timeout_seconds must be positive")
        if self.max_pending_requests <= 0:
            raise ValueError("max_pending_requests must be positive")
        if self.response_fetch_batch_size <= 0:
            raise ValueError("response_fetch_batch_size must be positive")
        if self.server_start_timeout_seconds <= 0:
            raise ValueError("server_start_timeout_seconds must be positive")


@dataclass(frozen=True)
class InferenceWorkerConfig:
    max_in_flight: int = 16
    request_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")


@dataclass(frozen=True)
class BrokerConfig:
    worker: InferenceWorkerConfig = field(default_factory=InferenceWorkerConfig)
    proxy: InferenceProxyConfig = field(default_factory=InferenceProxyConfig)
    request_lease_timeout_seconds: float = 240.0
    broker_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig.with_cpu(
            cpu=2,
            ram="8g",
            disk="20g",
            preemptible=False,
        )
    )
    broker_ready_timeout_seconds: float = 900.0
    max_retries_failure: int = 1
    max_retries_preemption: int = 10

    def __post_init__(self) -> None:
        worker_timeout = self.worker.request_timeout_seconds
        lease_timeout = self.request_lease_timeout_seconds
        proxy_timeout = self.proxy.request_timeout_seconds
        if not 0 < worker_timeout < lease_timeout < proxy_timeout:
            raise ValueError(
                "Brokered inference timeouts must satisfy "
                "0 < worker.request_timeout_seconds < request_lease_timeout_seconds "
                "< proxy.request_timeout_seconds; "
                f"got worker={worker_timeout:.1f}s lease={lease_timeout:.1f}s proxy={proxy_timeout:.1f}s."
            )
        if self.proxy.readiness_timeout_seconds <= 0:
            raise ValueError("proxy.readiness_timeout_seconds must be positive")
        if self.broker_ready_timeout_seconds <= 0:
            raise ValueError("broker_ready_timeout_seconds must be positive")
        if self.broker_resources.preemptible:
            raise ValueError("broker_resources must be non-preemptible")
        if self.max_retries_failure < 0 or self.max_retries_preemption < 0:
            raise ValueError("broker retry counts must not be negative")


@dataclass(frozen=True)
class IrisConfig:
    """Iris placement and environment for one serving instance."""

    worker_resources: ResourceConfig
    worker_environment: EnvironmentConfig
    task_timeout: Duration | None = None
    """Deadline for each preparation/serving task and the preparation batch wait."""
    serving_geometry: ServingGeometry | None = None
    cache_ttl_days: int = 14
    endpoint_ready_timeout_seconds: float = 1800.0
    endpoint_health_timeout_seconds: float = 1800.0
    priority: int = 0
    max_retries_failure: int = 1
    max_retries_preemption: int = 10

    def __post_init__(self) -> None:
        if self.cache_ttl_days < 0:
            raise ValueError("cache_ttl_days must not be negative")
        if self.endpoint_ready_timeout_seconds <= 0:
            raise ValueError("endpoint_ready_timeout_seconds must be positive")
        if self.endpoint_health_timeout_seconds <= 0:
            raise ValueError("endpoint_health_timeout_seconds must be positive")
        if self.max_retries_failure < 0 or self.max_retries_preemption < 0:
            raise ValueError("worker retry counts must not be negative")
        # Lazy artifact fingerprinting substitutes a symbolic runtime-resource
        # marker before concrete resources are restored at execution time.
        if not isinstance(self.worker_resources, ResourceConfig):
            return
        geometry = self.serving_geometry
        if geometry is None and self.worker_resources.replicas != (
            self.worker_resources.device.vm_count() if isinstance(self.worker_resources.device, TpuConfig) else 1
        ):
            raise ValueError("Each inference instance must use exactly one Iris task")
        device = self.worker_resources.device
        if geometry is not None:
            if self.worker_resources.replicas != geometry.task_count:
                raise ValueError("worker resource replicas must equal serving geometry task_count")
            if not isinstance(device, GpuConfig) or device.count != geometry.gpus_per_task:
                raise ValueError("serving geometry requires matching GPU resources per task")
        if isinstance(device, CpuConfig):
            raise ValueError("Inference workers require an accelerator")


@dataclass(frozen=True)
class RemoteInferenceConfig:
    """Remote serving; ``instances`` counts independent servers.

    Multi-task geometry uses one coordinated vLLM group and stream loading.
    Session routing requires ``instances=1``, ``broker=None``.
    """

    model: ServedModelConfig
    engine: InferenceEngineConfig
    iris: IrisConfig
    instances: int = 1
    broker: BrokerConfig | None = None
    capability_origin: str | None = None

    def __post_init__(self) -> None:
        if self.instances <= 0:
            raise ValueError("instances must be positive")
        resources = self.iris.worker_resources
        device = resources.device if isinstance(resources, ResourceConfig) else None
        if isinstance(device, TpuConfig) and device.vm_count() > 1:
            if not isinstance(self.engine, VllmEngineConfig) or self.engine.launcher is not VllmLauncherType.TPU:
                raise ValueError("multi-host TPU serving requires the isolated TPU vLLM backend")
            if self.instances != 1 or self.broker is not None:
                raise ValueError("multi-host TPU serving requires one instance and no broker")
            if self.model.object_store_load_mode is not ObjectStoreLoadMode.STREAM:
                raise ValueError("multi-host TPU serving requires shared weights with stream loading")
        geometry = self.iris.serving_geometry
        session_routing = isinstance(self.engine, VllmEngineConfig) and self.engine.router is not None
        if (
            geometry is not None
            and self.model.tensor_parallel_size is not None
            and self.model.tensor_parallel_size != geometry.tensor_parallel_size
        ):
            raise ValueError("model tensor_parallel_size must match serving geometry")
        native_group = uses_native_vllm_group(geometry)
        if native_group or session_routing:
            if not isinstance(self.engine, VllmEngineConfig):
                raise ValueError("native GPU serving requires the vLLM backend")
            if geometry is not None:
                validate_serving_geometry_args(self.engine.extra_args)
        else:
            return
        if native_group and self.model.object_store_load_mode is not ObjectStoreLoadMode.STREAM:
            raise ValueError("multi-task GPU serving requires shared weights with stream loading")
        if self.instances != 1 or self.broker is not None:
            raise ValueError("native GPU serving requires one instance and no broker")
