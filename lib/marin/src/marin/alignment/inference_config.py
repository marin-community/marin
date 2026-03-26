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

from dataclasses import dataclass

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


@dataclass(frozen=True)
class VLLMConfig(InferenceConfig):
    """Configuration for local inference via vLLM.

    Args:
        model: Path to model checkpoint or HuggingFace model ID for local loading.
            Remote object-store URIs like `gs://...` are also supported.
        tensor_parallel_size: Number of GPUs/TPUs for tensor parallelism.
        max_model_len: Maximum sequence length for vLLM.
        gpu_memory_utilization: Fraction of GPU memory to use.
        load_format: Optional vLLM load format override. If unset, remote object-store
            model URIs default to `runai_streamer`.
        tpu_type: TPU type for resource allocation (e.g. "v6e-8").
        disk: Ephemeral disk request for TPU workers (e.g. "200g").
        ram: Memory request for TPU workers (e.g. "256g").
    """

    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    load_format: str | None = None
    tpu_type: str = "v6e-8"
    disk: str = "50g"
    ram: str = "128g"

    @property
    def resources(self) -> ResourceConfig:
        return ResourceConfig.with_tpu(self.tpu_type, disk=self.disk, ram=self.ram)
