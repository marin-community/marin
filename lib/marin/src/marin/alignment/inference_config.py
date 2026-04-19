# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inference backend configuration for the alignment pipeline.

Provides a unified type for specifying how to run inference — either via the
OpenAI API or via a local vLLM instance. Both are accepted anywhere a model is
needed (teacher, rejected, ideation, judge, etc.).

Usage:
    # OpenAI API model
    teacher = OpenAIConfig(model="gpt-4.1")

    # Local model via vLLM
    teacher = VLLMConfig(model="/path/to/checkpoint", tensor_parallel_size=4)

    # Both are InferenceConfig and accepted interchangeably
    config = ResponseGenConfig(model=teacher, ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from fray.v2.types import ResourceConfig


@dataclass(frozen=True)
class InferenceConfig:
    """Base class for inference backend configuration.

    Subclass this to add backend-specific parameters. The `model` field
    identifies what to run (API model ID or local checkpoint path).
    """

    model: str

    @property
    def is_api(self) -> bool:
        return isinstance(self, OpenAIConfig)

    @property
    def is_local(self) -> bool:
        return isinstance(self, VLLMConfig)

    @property
    def resources(self) -> ResourceConfig:
        """Default resources for this backend."""
        raise NotImplementedError

    @property
    def pip_dependency_groups(self) -> list[str]:
        """Dependency groups required to execute this backend."""
        raise NotImplementedError

    @property
    def pip_packages(self) -> list[str]:
        """Additional pip packages required to execute this backend."""
        return []


@dataclass(frozen=True)
class OpenAIConfig(InferenceConfig):
    """Configuration for API-based inference via the OpenAI chat completions API.

    Args:
        model: OpenAI model identifier (for example, ``gpt-4.1``).
        num_retries: Number of retries on transient failures.
        workers: Number of parallel API calls.
    """

    num_retries: int = 10
    workers: int = 64

    @property
    def resources(self) -> ResourceConfig:
        return ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g")

    @property
    def pip_dependency_groups(self) -> list[str]:
        return ["cpu"]

    @property
    def pip_packages(self) -> list[str]:
        return []


@dataclass(frozen=True)
class VLLMConfig(InferenceConfig):
    """Configuration for local inference via vLLM.

    Args:
        model: Path to model checkpoint or HuggingFace model ID for local loading.
            Remote object-store URIs like `gs://...` are also supported.
        tokenizer: Optional tokenizer path or Hugging Face tokenizer ID. When
            unset, vLLM and local prompt rendering use `model`.
        hf_overrides: Optional Hugging Face config overrides forwarded to vLLM.
            This is useful when a checkpoint layout omits architecture metadata
            that vLLM needs for model resolution.
        tensor_parallel_size: Number of GPUs/TPUs for tensor parallelism.
        max_model_len: Maximum sequence length for vLLM.
        gpu_memory_utilization: Fraction of GPU memory to use.
        load_format: Optional vLLM load format override. If unset, remote object-store
            model URIs default to `runai_streamer`.
        gpu_type: Optional GPU type for resource allocation (for example,
            ``"H100"``). When set, GPU resources are requested and Docker mode
            is the default serving mode.
        gpu_count: Number of GPUs to request when ``gpu_type`` is set.
        tpu_type: TPU type for resource allocation (e.g. "v6e-8"). Used when
            ``gpu_type`` is unset.
        cpu: Host CPU request for the worker.
        disk: Ephemeral disk request for TPU workers (e.g. "200g").
        ram: Memory request for TPU workers (e.g. "256g").
        preemptible: Whether the remote worker can be preempted.
        serve_mode: Optional vLLM serving mode override. When unset, TPU-backed
            configs default to ``"native"`` and GPU-backed configs default to
            ``"docker"``.
        docker_image: Optional Docker image override for Docker-backed vLLM.
        additional_config: Optional vLLM TPU additional-config payload.
        model_impl_type: Optional TPU model implementation selection. This is
            forwarded as ``MODEL_IMPL_TYPE`` for native/docker TPU startup.
        native_stderr_mode: Native-server stderr handling mode. ``"file"``
            keeps stderr in the native log files only. ``"tee"`` also mirrors
            stderr into the parent worker logs for easier debugging.
        extra_pip_packages: Additional pip packages to install in remote jobs
            before launching the model.
    """

    tokenizer: str | None = None
    hf_overrides: dict[str, Any] | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    load_format: str | None = None
    gpu_type: str | None = None
    gpu_count: int = 1
    tpu_type: str | None = "v6e-8"
    cpu: int = 32
    disk: str = "50g"
    ram: str = "128g"
    preemptible: bool = True
    serve_mode: Literal["native", "docker"] | None = None
    docker_image: str | None = None
    additional_config: dict[str, Any] | None = None
    model_impl_type: Literal["auto", "vllm", "flax_nnx"] | None = None
    native_stderr_mode: Literal["file", "tee"] = "file"
    extra_pip_packages: tuple[str, ...] = ()

    @property
    def resources(self) -> ResourceConfig:
        if self.gpu_type is not None:
            return ResourceConfig.with_gpu(
                self.gpu_type,
                count=self.gpu_count,
                cpu=self.cpu,
                disk=self.disk,
                ram=self.ram,
                preemptible=self.preemptible,
            )
        if self.tpu_type is None:
            raise ValueError("VLLMConfig requires either gpu_type or tpu_type.")
        return ResourceConfig.with_tpu(
            self.tpu_type,
            cpu=self.cpu,
            disk=self.disk,
            ram=self.ram,
            preemptible=self.preemptible,
        )

    @property
    def pip_dependency_groups(self) -> list[str]:
        if self.gpu_type is not None:
            return ["gpu"]
        return ["vllm", "tpu"]

    @property
    def pip_packages(self) -> list[str]:
        return list(self.extra_pip_packages)

    @property
    def resolved_serve_mode(self) -> Literal["native", "docker"]:
        if self.serve_mode is not None:
            return self.serve_mode
        if self.gpu_type is not None:
            return "docker"
        return "native"

    def __hash__(self) -> int:
        overrides_json = None
        if self.hf_overrides is not None:
            overrides_json = json.dumps(self.hf_overrides, sort_keys=True)
        additional_config_json = None
        if self.additional_config is not None:
            additional_config_json = json.dumps(self.additional_config, sort_keys=True)
        return hash(
            (
                self.model,
                self.tokenizer,
                overrides_json,
                self.tensor_parallel_size,
                self.max_model_len,
                self.gpu_memory_utilization,
                self.load_format,
                self.gpu_type,
                self.gpu_count,
                self.tpu_type,
                self.cpu,
                self.disk,
                self.ram,
                self.preemptible,
                self.serve_mode,
                self.docker_image,
                additional_config_json,
                self.model_impl_type,
                self.extra_pip_packages,
            )
        )
