# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave H100 speed/profiling launcher for the May d=2560 Grug MoE recipe.

This matches the model shape from issue #6044 while using the current
CoreWeave/R2 launch path. Defaults are for a fast profiling run, not a full
10T-token training run:

    MAY_GPU_REPLICAS=32      32 gd-8xh100ib nodes, 256 H100s
    MAY_EXPERT_AXIS=8        expert parallelism inside each NVLink node
    MAY_REPLICA_AXIS=1       FSDP over the whole data axis
    MAY_BATCH=256            seq=4096 context; raise only after profiling memory
    MAY_STEPS=50             throughput/profiling length
    MAY_CHECKPOINTS=local    avoid object-store checkpoint stalls
    MAY_MP=params=float32,compute=bfloat16,output=bfloat16

The default parameter policy keeps one sharded fp32 parameter tree plus sharded
optimizer state. A persistent bf16-params + fp32-master split is not currently a
Grug train-state mode; use ``MAY_MP=params=bfloat16,...`` only for a risky
throughput experiment that gives up the fp32 master copy.
"""

import dataclasses
import datetime
import os
from typing import cast

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic import MoeAdamHHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    env_int,
    run_grug_moe_trial,
    slimpajama_6b_data,
)
from experiments.grug.moe.model import GrugModelConfig, RematMode
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

GPUS_PER_NODE = 8
DEFAULT_HIDDEN_DIM = 2560
DEFAULT_SEQ_LEN = 4096
DEFAULT_BATCH = 256
DEFAULT_STEPS = 50
DEFAULT_TOTAL_TOKENS = 1.0e13
DEFAULT_WARMUP_FRACTION = 0.01

MAY_HEURISTIC = MoeAdamHHeuristic(
    lr_coeff=0.06602,
    lr_tokens_exp=-0.395,
    lr_dim_exp=-0.150,
)


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key}={raw!r} must be a boolean")


def env_optional_int(key: str) -> int | None:
    raw = os.environ.get(key, "")
    return int(raw) if raw else None


def build_may_model() -> GrugModelConfig:
    hidden_dim = env_int("MAY_HIDDEN_DIM", DEFAULT_HIDDEN_DIM)
    seq_len = env_int("MAY_SEQ_LEN", DEFAULT_SEQ_LEN)
    remat_mode = os.environ.get("MAY_REMAT", "save_moe")
    if remat_mode not in ("recompute_all", "save_moe"):
        raise ValueError(f"MAY_REMAT={remat_mode!r} must be 'recompute_all' or 'save_moe'")
    attention_implementation = os.environ.get("MAY_ATTENTION_IMPLEMENTATION", "gpu_fa4_cute")

    model = MAY_HEURISTIC.build_model_config(hidden_dim, seq_len=seq_len)
    return dataclasses.replace(
        model,
        num_experts=env_int("MAY_NUM_EXPERTS", 256),
        num_experts_per_token=env_int("MAY_TOP_K", 4),
        router_z_loss_coef=0.0,
        routing_renorm_sum=env_float("MAY_ROUTING_RENORM_SUM", 2.5),
        use_half_rope=True,
        use_pko=True,
        pko_on_last_layer=True,
        moe_implementation=cast(str, os.environ.get("MAY_MOE_IMPLEMENTATION", "ring")),
        attention_implementation=cast(str, attention_implementation or None),
        remat_mode=cast(RematMode, remat_mode),
    )


def build_may_optimizer(*, batch_size: int, seq_len: int) -> GrugMoeMuonHConfig:
    total_tokens = env_float("MAY_TOTAL_TOKENS", DEFAULT_TOTAL_TOKENS)
    hidden_dim = env_int("MAY_HIDDEN_DIM", DEFAULT_HIDDEN_DIM)
    base_optimizer = MAY_HEURISTIC.build_optimizer_config(batch_size, total_tokens, hidden_dim, seq_len=seq_len)
    return GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=env_float("MAY_WARMUP_FRACTION", DEFAULT_WARMUP_FRACTION),
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )


def build_data():
    data = os.environ.get("MAY_DATA", "slimpajama").lower()
    if data == "slimpajama":
        return slimpajama_6b_data()
    if data == "nemotron":
        return NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
    raise ValueError(f"MAY_DATA={data!r} must be 'slimpajama' or 'nemotron'")


def build_tracker(run_id: str):
    if os.environ.get("MAY_TRACKER", "json_logger").lower() == "wandb":
        return WandbConfig(
            entity=os.environ.get("WANDB_ENTITY") or "marin-community",
            project=os.environ.get("WANDB_PROJECT", "marin_moe"),
            tags=["grug", "moe", "may", "cw", "h100", f"d{env_int('MAY_HIDDEN_DIM', DEFAULT_HIDDEN_DIM)}"],
            group=os.environ.get("MAY_WANDB_GROUP", "grug-moe-cw-may-d2560"),
            name=run_id,
            replicate_path=this_output_path(),
        )
    return JsonLoggerConfig(logger_name=os.environ.get("MAY_JSON_LOGGER", "grug_moe_cw_may.metrics"))


def build_checkpointer(run_id: str) -> CheckpointerConfig | None:
    checkpoint_mode = os.environ.get("MAY_CHECKPOINTS", "local").lower()
    if checkpoint_mode == "local":
        return CheckpointerConfig(
            base_path=f"/tmp/grug-may-d2560-ckpt/{run_id}",
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=None,
        )
    if checkpoint_mode == "s3":
        return None
    raise ValueError(f"MAY_CHECKPOINTS={checkpoint_mode!r} must be 'local' or 's3'")


def build_eval() -> GrugEvalConfig | None:
    if os.environ.get("MAY_EVAL", "").lower() not in ("1", "true", "yes"):
        return None
    return GrugEvalConfig(
        eval_batch_size=env_int("MAY_EVAL_BATCH", 512),
        steps_per_eval=env_int("MAY_EVAL_INTERVAL", 1000),
        max_eval_batches=env_int("MAY_MAX_EVAL_BATCHES", 8),
        eval_current=True,
        eval_ema=False,
    )


def build_may_step() -> ExecutorStep:
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime(
        "cw-may-d2560-%Y%m%d-%H%M%S"
    )

    replicas = env_int("MAY_GPU_REPLICAS", 32)
    expert_axis = env_int("MAY_EXPERT_AXIS", 8)
    replica_axis = env_int("MAY_REPLICA_AXIS", 1)
    batch_size = env_int("MAY_BATCH", DEFAULT_BATCH)
    steps = env_int("MAY_STEPS", DEFAULT_STEPS)
    profiler_steps = env_int("MAY_PROFILER_STEPS", 0)

    model = build_may_model()
    if model.num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by MAY_EXPERT_AXIS={expert_axis}")

    data_axis = (replicas * GPUS_PER_NODE) // (replica_axis * expert_axis)
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"MAY_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu("H100", count=GPUS_PER_NODE, cpu=32, ram="256g", disk="256g", replicas=replicas)
    grug_trainer = GrugTrainerConfig(
        expert_axis_size=expert_axis,
        replica_axis_size=replica_axis,
        z_loss_weight=0.0,
        ema_beta=None,
        log_every=1,
    )
    profiler = ProfilerConfig(
        enabled=profiler_steps > 0,
        start_step=env_int("MAY_PROFILER_START", 8),
        num_steps=profiler_steps,
        perfetto_link=env_bool("MAY_PROFILER_PERFETTO_LINK", False),
        profile_options=ProfileOptionsConfig(
            host_tracer_level=env_optional_int("MAY_PROFILER_HOST_TRACER_LEVEL"),
            python_tracer_level=env_optional_int("MAY_PROFILER_PYTHON_TRACER_LEVEL"),
            device_tracer_level=env_optional_int("MAY_PROFILER_DEVICE_TRACER_LEVEL"),
            enable_hlo_proto=env_bool("MAY_PROFILER_ENABLE_HLO_PROTO", profiler_steps > 0),
            include_dataset_ops=env_bool("MAY_PROFILER_INCLUDE_DATASET_OPS", False),
        ),
    )
    eval_cfg = build_eval()

    name = f"grug-moe-cw-may-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}-r{replicas}"
    return ExecutorStep(
        name=f"{name}-{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=build_data(),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(resources),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned(os.environ.get("MAY_MP", "params=float32,compute=bfloat16,output=bfloat16")),
            tracker=build_tracker(run_id),
            optimizer=versioned(build_may_optimizer(batch_size=batch_size, seq_len=model.max_seq_len)),
            grug_trainer=versioned(grug_trainer),
            eval=versioned(eval_cfg) if eval_cfg is not None else None,
            profiler=profiler,
            checkpointer=build_checkpointer(run_id),
        ),
    )


may_d2560_step = build_may_step()


def main() -> None:
    executor_main(steps=[may_d2560_step])


if __name__ == "__main__":
    main()
