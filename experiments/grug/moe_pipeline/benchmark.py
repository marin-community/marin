# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment-configured benchmark for the Grug MoE pipeline trainer."""

import math
import os
from collections.abc import Mapping

from experiments.grug.moe_pipeline.train import GrugPipelineTrainConfig, PipelineSchedule, _run_grug_local

_ATTENTION_HEAD_DIM = 128


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name)
    return default if not value else int(value)


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    value = environ.get(name)
    return default if not value else float(value)


def _default_num_kv_heads(num_heads: int) -> int:
    for candidate in range(num_heads // 4, 0, -1):
        if num_heads % candidate == 0:
            return candidate
    return 1


def _resolve_benchmark_config(environ: Mapping[str, str]) -> GrugPipelineTrainConfig:
    stages = _env_int(environ, "PIPELINE_STAGES", 4)
    physical_stages = _env_int(environ, "PIPELINE_PHYSICAL_STAGES", stages)
    seq_len = _env_int(environ, "PIPELINE_SEQ_LEN", 4096)
    hidden_dim = _env_int(environ, "PIPELINE_HIDDEN_DIM", 2560)
    warmup_steps = _env_int(environ, "PIPELINE_WARMUP_STEPS", 1)
    profile_start_step = _env_int(environ, "PIPELINE_PROFILE_START_STEP", warmup_steps)
    profile_steps = _env_int(environ, "PIPELINE_PROFILE_STEPS", 0)
    profile_run_id = environ.get("PIPELINE_PROFILE_RUN_ID")
    steps = _env_int(environ, "PIPELINE_STEPS", 4)
    if profile_steps > 0:
        if not profile_run_id:
            raise ValueError("PIPELINE_PROFILE_RUN_ID is required when profiling")
        if profile_start_step + profile_steps > steps:
            raise ValueError("pipeline profile window must fit within PIPELINE_STEPS")

    layer_counts_value = environ.get("PIPELINE_LAYERS_PER_STAGE")
    layer_counts = None if not layer_counts_value else tuple(int(value) for value in layer_counts_value.split(","))
    memory_threshold_value = environ.get("PIPELINE_RESHARD_THRESHOLD_BYTES")
    memory_threshold = None if not memory_threshold_value else int(memory_threshold_value)
    if hidden_dim % _ATTENTION_HEAD_DIM != 0:
        raise ValueError(f"PIPELINE_HIDDEN_DIM must be divisible by {_ATTENTION_HEAD_DIM}, got {hidden_dim}")
    num_heads = hidden_dim // _ATTENTION_HEAD_DIM
    num_kv_heads = _default_num_kv_heads(num_heads)
    return GrugPipelineTrainConfig(
        stages=stages,
        physical_stages=physical_stages,
        microbatches=_env_int(environ, "PIPELINE_MICROBATCHES", 8),
        batch_size=_env_int(environ, "PIPELINE_BATCH", 256),
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        intermediate_dim=_env_int(environ, "PIPELINE_INTERMEDIATE_DIM", math.ceil(hidden_dim / 256) * 128),
        shared_expert_intermediate_dim=_env_int(
            environ,
            "PIPELINE_SHARED_EXPERT_INTERMEDIATE_DIM",
            hidden_dim,
        ),
        num_layers=_env_int(environ, "PIPELINE_LAYERS", 24),
        num_experts=_env_int(environ, "PIPELINE_EXPERTS", 256),
        top_k=_env_int(environ, "PIPELINE_TOP_K", 4),
        expert_axis_size=_env_int(environ, "PIPELINE_EXPERT_AXIS", 8),
        vocab_size=_env_int(environ, "PIPELINE_VOCAB_SIZE", 128_256),
        num_heads=_env_int(environ, "PIPELINE_HEADS", num_heads),
        num_kv_heads=_env_int(environ, "PIPELINE_KV_HEADS", num_kv_heads),
        sliding_window=_env_int(environ, "PIPELINE_SLIDING_WINDOW", 2048),
        qk_mult=_env_float(environ, "PIPELINE_QK_MULT", 1.3),
        steps=steps,
        warmup_steps=warmup_steps,
        profile_start_step=profile_start_step,
        profile_steps=profile_steps,
        profile_run_id=profile_run_id,
        layer_counts=layer_counts,
        memory_threshold=memory_threshold,
        mp_policy_string=environ.get("PIPELINE_MP", "params=bfloat16,compute=bfloat16,output=bfloat16"),
        remat_mode=environ.get("PIPELINE_REMAT", "recompute_all"),
        attention_implementation=environ.get("PIPELINE_ATTENTION", "gpu_fa4_cute"),
        moe_implementation=environ.get("PIPELINE_MOE", "ring"),
        schedule=PipelineSchedule(environ.get("PIPELINE_SCHEDULE", PipelineSchedule.ZERO_BUBBLE)),
    )


def main() -> None:
    config = _resolve_benchmark_config(os.environ)
    _run_grug_local(config)


if __name__ == "__main__":
    main()
