"""Launch engine options and entrypoint names accepted by authored recipes."""

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any


class RLEntrypoint(StrEnum):
    """Execution modes supported by Iris RL configurations."""

    GENERATE = "generate"
    MINI_SWE = "mini_swe"
    STANDARD = "standard"
    TERMINAL_BENCH = "terminal_bench"
    TERMINAL_BENCH_GENERATE = "terminal_bench_generate"


RL_ENTRYPOINTS = MappingProxyType(
    {
        RLEntrypoint.GENERATE: "skyrl_train.entrypoints.main_generate",
        RLEntrypoint.MINI_SWE: "skyrl_train.entrypoints.mini_swe",
        RLEntrypoint.STANDARD: "skyrl_train.entrypoints.main_base",
        RLEntrypoint.TERMINAL_BENCH: "skyrl_train.entrypoints.terminal_bench",
        RLEntrypoint.TERMINAL_BENCH_GENERATE: "skyrl_train.entrypoints.terminal_bench_generate",
    }
)

SKYRL_INTERNAL_ENGINE_KWARGS = frozenset(
    {
        "trust_remote_code",
        "worker_extension_cls",
        "data_parallel_backend",
        "max_logprobs",
        "distributed_executor_backend",
        "enforce_eager",
        "tensor_parallel_size",
        "data_parallel_size",
        "seed",
        "enable_prefix_caching",
        "dtype",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_sleep_mode",
        "vllm_v1_disable_multiproc",
        "bundle_indices",
        "num_gpus",
        "noset_visible_devices",
        "model_path",
        "tp_size",
        "mem_fraction_static",
        "random_seed",
        "disable_radix_cache",
        "max_prefill_tokens",
        "max_running_requests",
        "mm_attention_backend",
        "attention_backend",
        "enable_memory_saver",
        "tokenizer",
        "custom_weight_loader",
        "skip_tokenizer_init",
        "speculative_config",
    }
)


def validate_engine_init_kwargs(engine_init_kwargs: Mapping[str, Any], config_path: Path | None = None) -> None:
    """Reject engine keyword arguments supplied internally by SkyRL."""
    forbidden = engine_init_kwargs.keys() & SKYRL_INTERNAL_ENGINE_KWARGS
    if forbidden:
        context = f" in {config_path}" if config_path is not None else ""
        raise ValueError(
            f"generator.engine_init_kwargs{context}: SkyRL sets these keys internally: {', '.join(sorted(forbidden))}"
        )


def validate_tp_divides_heads(
    tensor_parallel_size: int, num_attention_heads: int | None, config_path: Path | None = None
) -> None:
    """Require inference tensor parallelism to divide the declared attention-head count."""
    if num_attention_heads and num_attention_heads % tensor_parallel_size:
        context = f" in {config_path}" if config_path is not None else ""
        raise ValueError(
            f"generator.inference_engine_tensor_parallel_size={tensor_parallel_size} does not "
            f"divide model_num_attention_heads={num_attention_heads}{context}"
        )
