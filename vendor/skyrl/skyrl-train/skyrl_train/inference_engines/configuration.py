"""Shared configuration assembly for owned inference-engine roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy


@dataclass(frozen=True)
class InferenceEngineRoleConfig:
    """Role-specific inputs layered over the shared generator configuration."""

    pretrain: str
    backend: str
    num_inference_engines: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    expert_parallel_size: int
    decode_context_parallel_size: int
    shared_pg: Any
    inference_engine_enable_sleep: bool
    max_logprobs: int = 1


def inference_engine_kwargs_from_config(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    role: InferenceEngineRoleConfig,
    *,
    engine_init_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Combine shared generator settings with explicit role geometry and ownership."""
    kwargs = {
        "num_inference_engines": role.num_inference_engines,
        "tensor_parallel_size": role.tensor_parallel_size,
        "pipeline_parallel_size": role.pipeline_parallel_size,
        "data_parallel_size": role.data_parallel_size,
        "expert_parallel_size": role.expert_parallel_size,
        "decode_context_parallel_size": role.decode_context_parallel_size,
        "model_dtype": cfg.generator.model_dtype,
        "pretrain": role.pretrain,
        "seed": cfg.trainer.seed,
        "vllm_v1_disable_multiproc": cfg.generator.vllm_v1_disable_multiproc,
        "enable_prefix_caching": cfg.generator.enable_prefix_caching,
        "enforce_eager": cfg.generator.enforce_eager,
        "shared_pg": role.shared_pg,
        "engine_init_timeout_seconds": cfg.generator.engine_init_timeout_seconds,
        "gpu_memory_utilization": cfg.generator.gpu_memory_utilization,
        "inference_engine_enable_sleep": role.inference_engine_enable_sleep,
        "max_num_batched_tokens": cfg.generator.max_num_batched_tokens,
        "max_num_seqs": cfg.generator.max_num_seqs,
        "tokenizer": tokenizer,
        "backend": role.backend,
        "vllm_attention_backend": cfg.generator.get("vllm_attention_backend", None),
        "engine_init_kwargs": engine_init_kwargs,
        "max_logprobs": role.max_logprobs,
        "mp_backend": cfg.generator.get("inference_engine_mp_backend", False),
        "placement_group_timeout_seconds": int(cfg.trainer.distributed.placement_group_timeout_seconds),
        "weight_sync_pause_policy": resolve_weight_sync_pause_policy(cfg.generator),
    }
    if (rope_scaling := cfg.generator.get("rope_scaling", None)) is not None:
        kwargs["rope_scaling"] = OmegaConf.to_container(rope_scaling, resolve=True)
    if (rope_theta := cfg.generator.get("rope_theta", None)) is not None:
        kwargs["rope_theta"] = rope_theta
    return kwargs
