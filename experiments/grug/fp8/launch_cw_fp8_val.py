# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GFP8-035: single-H100 MoE training validation -- does all-E4M3 f8 hold over a trajectory?

Prior f8 evidence is benchmark-only (standalone grouped-GEMM throughput) plus a single-step
synthetic-cotangent numerics check (GFP8-023). This trains a small MoE end-to-end on one H100
and compares two arms that differ *only* in the expert grouped-GEMM kernel:

    MOE_IMPL=scatter      bf16 ragged_dot (baseline)
    MOE_IMPL=scatter_f8   all-E4M3, current/per-step per-tensor scaling (the f8 arm)

Everything else -- seed, data order, optimizer, model shape, mesh -- is identical, so any
divergence in the loss curve is attributable to E4M3 quantization of the expert forward and
gradients. Current/per-step scaling is the *best case* for E4M3 (an ideal per-tensor scale
recomputed every step); if loss drifts or NaNs even here, coarser scaling cannot rescue it.

Vehicle: 1 GPU, expert_axis=1, replica_axis=1 -> the local ``scatter`` backend (no EP
shard_map, identical per-expert GEMM numerics to the production EP path).

Env knobs (all optional):

    MOE_IMPL        scatter (default) | scatter_f8
    FP8_HIDDEN_DIM  hidden width (default 512; multiple of head_dim=128)
    FP8_NUM_LAYERS  number of MoE blocks (default 6)
    FP8_NUM_EXPERTS routed experts (default 8)
    FP8_TOP_K       experts per token (default 2)
    FP8_BATCH       global batch in sequences (default 16)
    FP8_SEQ_LEN     sequence length (default 1024)
    FP8_STEPS       training steps (default 3000)
    FP8_TRACKER     wandb (default) | json_logger
    RUN_ID          unique run identifier (default: MOE_IMPL + timestamp)
"""

import datetime
import os

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial, slimpajama_6b_data
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.llama import llama3_tokenizer_vocab_size

HEAD_DIM = 128
VOCAB_SIZE = llama3_tokenizer_vocab_size
OUTPUT_SUBDIR = "experiments/grug-moe-fp8-val"

# Identical across both arms so the only difference is the expert GEMM kernel.
VAL_OPTIMIZER = AdamConfig(
    learning_rate=6e-4,
    weight_decay=0.1,
    lr_schedule="cosine",
    warmup=100,
    min_lr_ratio=0.1,
)
VAL_TRAINER_DEFAULTS = dict(z_loss_weight=1e-4, ema_beta=None, log_every=1)

_VALID_MOE_IMPLS = ("scatter", "scatter_f8")


def build_val_model() -> GrugModelConfig:
    moe_impl = os.environ.get("MOE_IMPL", "scatter")
    if moe_impl not in _VALID_MOE_IMPLS:
        raise ValueError(f"MOE_IMPL={moe_impl!r} must be one of {_VALID_MOE_IMPLS}")
    hidden_dim = env_int("FP8_HIDDEN_DIM", 512)
    if hidden_dim % HEAD_DIM != 0:
        raise ValueError(f"FP8_HIDDEN_DIM={hidden_dim} must be a multiple of head_dim={HEAD_DIM}")
    num_heads = hidden_dim // HEAD_DIM
    num_kv_heads = max(1, num_heads // 4)
    while num_heads % num_kv_heads != 0:
        num_kv_heads -= 1
    intermediate_dim = hidden_dim // 2
    seq_len = env_int("FP8_SEQ_LEN", 1024)
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        num_layers=env_int("FP8_NUM_LAYERS", 6),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=intermediate_dim,
        num_experts=env_int("FP8_NUM_EXPERTS", 8),
        num_experts_per_token=env_int("FP8_TOP_K", 2),
        max_seq_len=seq_len,
        sliding_window=seq_len,
        moe_implementation=moe_impl,
    )


def build_val_step() -> ExecutorStep:
    moe_impl = os.environ.get("MOE_IMPL", "scatter")
    run_id = os.environ.get("RUN_ID") or f"{datetime.datetime.now(datetime.UTC):%Y%m%d-%H%M%S}"
    steps = env_int("FP8_STEPS", 3000)
    batch_size = env_int("FP8_BATCH", 16)

    # One H100, pure local scatter: no replication, no expert parallelism.
    resources = ResourceConfig.with_gpu("H100", count=1, cpu=16, ram="128g", disk="128g", replicas=1)

    if os.environ.get("FP8_TRACKER", "wandb").lower() == "wandb":
        tracker = WandbConfig(
            entity=os.environ.get("WANDB_ENTITY") or None,
            project=os.environ.get("WANDB_PROJECT", "marin_moe"),
            tags=["grug", "moe", "fp8", "gfp8-035", moe_impl],
            group="grug-moe-fp8-val",
            name=None,
            replicate_path=this_output_path(),
        )
    else:
        tracker = JsonLoggerConfig(logger_name="grug_moe_fp8_val.metrics")

    grug_trainer = GrugTrainerConfig(expert_axis_size=1, replica_axis_size=1, **VAL_TRAINER_DEFAULTS)

    model = build_val_model()
    name = f"fp8val-{moe_impl}-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}"
    return ExecutorStep(
        name=f"{OUTPUT_SUBDIR}/{name}-{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=slimpajama_6b_data(),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(resources),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned(os.environ.get("FP8_MP", "params=float32,compute=bfloat16,output=bfloat16")),
            tracker=tracker,
            optimizer=versioned(VAL_OPTIMIZER),
            grug_trainer=versioned(grug_trainer),
            eval=None,
        ),
    )


fp8_val_step = build_val_step()


def main():
    executor_main(steps=[fp8_val_step])


if __name__ == "__main__":
    main()
