# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inference backend configuration for the alignment pipeline.

Provides a unified type for specifying how to run inference — either via
API calls (litellm) or via a local vLLM instance. Both are accepted
anywhere a model is needed (teacher, rejected, ideation, judge, etc.).

Usage:
    # API model (OpenAI, Anthropic, etc. via litellm)
    teacher = LiteLLMConfig(model="openai/gpt-4.1")

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
        return isinstance(self, LiteLLMConfig)

    @property
    def is_local(self) -> bool:
        return isinstance(self, VLLMConfig)

    @property
    def resources(self) -> ResourceConfig:
        """Default resources for this backend."""
        raise NotImplementedError


@dataclass(frozen=True)
class LiteLLMConfig(InferenceConfig):
    """Configuration for API-based inference via litellm.

    Supports any provider litellm supports: OpenAI, Anthropic, Google, etc.

    Args:
        model: litellm model identifier (e.g. "openai/gpt-4.1", "anthropic/claude-sonnet-4-20250514").
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
        tensor_parallel_size: Number of GPUs/TPUs for tensor parallelism.
        max_model_len: Maximum sequence length for vLLM.
        gpu_memory_utilization: Fraction of GPU memory to use.
        tpu_type: TPU type for resource allocation (e.g. "v6e-8").
    """

    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tpu_type: str = "v6e-8"

    @property
    def resources(self) -> ResourceConfig:
        return ResourceConfig.with_tpu(self.tpu_type)
