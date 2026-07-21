# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight inference serving configuration.

These dataclasses are safe to construct in CPU coordinators and CLI processes.
Accelerator-heavy serving implementations translate them inside worker jobs.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from fray.types import CpuConfig, EnvironmentConfig, ResourceConfig, TpuConfig

# Worker and isolated vLLM environments use one Python version so cloudpickle
# and the launched callable stay compatible.
WORKER_PYTHON_VERSION = "3.12"
# Stock CUDA vLLM runs in an isolated uv-tool environment and does not
# participate in Marin's workspace dependency resolution.
DEFAULT_CUDA_VLLM_VERSION = "0.25.1"
TPU_VLLM_WORKER_EXTRAS = ("tpu", "vllm")


class VllmLauncherType(StrEnum):
    WORKSPACE = "workspace"
    CUDA = "cuda"
    TPU = "tpu"


class VllmSource(StrEnum):
    UPSTREAM = "upstream"
    MARIN_FORK = "marin_fork"


@dataclass(frozen=True)
class ServedModelConfig:
    model: str
    model_path: str | None = None
    tokenizer: str | None = None
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    tensor_parallel_size: int | None = None
    chat_template_content: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.tensor_parallel_size is not None and self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")


@dataclass(frozen=True)
class VllmEngineConfig:
    launcher: VllmLauncherType = VllmLauncherType.WORKSPACE
    source: VllmSource = VllmSource.UPSTREAM
    version: str | None = None
    startup_timeout_seconds: int = 1800
    max_num_batched_tokens: int | None = None
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.source is VllmSource.MARIN_FORK and self.launcher is not VllmLauncherType.CUDA:
            raise ValueError("the Marin vLLM fork source requires the CUDA launcher")


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
    """Iris placement and environment for one remote inference instance."""

    worker_resources: ResourceConfig
    worker_environment: EnvironmentConfig
    cache_ttl_days: int = 14
    endpoint_ready_timeout_seconds: float = 1800.0
    priority: int = 0
    max_retries_failure: int = 1
    max_retries_preemption: int = 10

    def __post_init__(self) -> None:
        if self.cache_ttl_days < 0:
            raise ValueError("cache_ttl_days must not be negative")
        if self.endpoint_ready_timeout_seconds <= 0:
            raise ValueError("endpoint_ready_timeout_seconds must be positive")
        if self.max_retries_failure < 0 or self.max_retries_preemption < 0:
            raise ValueError("worker retry counts must not be negative")
        # Lazy artifact fingerprinting substitutes a symbolic runtime-resource
        # marker before concrete resources are restored at execution time.
        if not isinstance(self.worker_resources, ResourceConfig):
            return
        if self.worker_resources.replicas != 1:
            raise ValueError("Each inference instance must use exactly one Iris task")
        device = self.worker_resources.device
        if isinstance(device, CpuConfig):
            raise ValueError("Inference workers require an accelerator")
        if isinstance(device, TpuConfig) and device.vm_count() != 1:
            raise ValueError(f"Inference instances require a single-host TPU; got {device.variant}")
