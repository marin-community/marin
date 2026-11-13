# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch engine options and entrypoint names accepted by authored recipes."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class RLEntrypoint(StrEnum):
    """Execution modes supported by Iris RL configurations."""

    GENERATE = "generate"
    STANDARD = "standard"
    TASKCOMPENDIUM = "taskcompendium"
    TERMINAL_BENCH = "terminal_bench"
    TERMINAL_BENCH_GENERATE = "terminal_bench_generate"


# Source: skyrl_train/inference_engines/ray_wrapped_inference_engine.py
SKYRL_INTERNAL_ENGINE_KWARGS = frozenset(
    {
        # Hardcoded values
        "trust_remote_code",  # Always True
        "worker_extension_cls",  # vLLM SkyRL extension path
        "data_parallel_backend",  # Hardcoded "mp"
        "max_logprobs",  # Hardcoded 1
        # Calculated from config/environment
        "distributed_executor_backend",  # Calculated from TP size ("uni" or "ray")
        "enforce_eager",  # Set from generator.enforce_eager config
        "tensor_parallel_size",  # Set from generator config
        "data_parallel_size",  # Set from generator config
        "seed",  # Set from config
        "enable_prefix_caching",  # Set from generator config
        "dtype",  # Set from generator.model_dtype
        "gpu_memory_utilization",  # Set from generator config
        "max_num_batched_tokens",  # Set from generator config
        "max_num_seqs",  # Set from generator config
        "enable_sleep_mode",  # Set from trainer.placement.colocate_all
        "vllm_v1_disable_multiproc",  # Set from generator config
        # Ray internal management
        "bundle_indices",  # Calculated from parallelism config
        "num_gpus",  # Ray resource allocation
        "noset_visible_devices",  # Ray CUDA_VISIBLE_DEVICES handling
        "attention_backend",  # Set from generator.vllm_attention_backend
        "tokenizer",  # Set from trainer.policy.model.tokenizer_path
        "skip_tokenizer_init",
        "speculative_config",  # Derived from generator.speculative_decoding
    }
)


def validate_engine_init_kwargs(engine_init_kwargs: Mapping[str, Any]) -> None:
    """Reject internally supplied options that cause duplicate-keyword errors at engine startup."""
    forbidden = engine_init_kwargs.keys() & SKYRL_INTERNAL_ENGINE_KWARGS
    if forbidden:
        raise ValueError(
            f"generator.engine_init_kwargs: SkyRL sets these keys internally: {', '.join(sorted(forbidden))}"
        )


def validate_tp_divides_heads(tensor_parallel_size: int, num_attention_heads: int | None) -> None:
    """Require TP to divide the declared head count so vLLM does not stall at engine initialization."""
    if num_attention_heads and num_attention_heads % tensor_parallel_size:
        raise ValueError(
            f"generator.inference_engine_tensor_parallel_size={tensor_parallel_size} does not "
            f"divide model_num_attention_heads={num_attention_heads}"
        )
