# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimalist dense-transformer scale/smoke run for the CoreWeave H100 cluster.

Same CoreWeave plumbing as ``launch_cw_scale.py`` (SCALE_* env knobs, FSDP mesh,
SlimPajama block-shuffle), but trains the bare ``model_basic`` dense transformer
via ``train_basic`` instead of the MoE model. Full multi-head attention
(``num_kv_heads == num_heads``), a plain GELU MLP, and no norms.

Env knobs (all optional):

    SCALE_GPU_REPLICAS   number of 8xH100 nodes (default 1 -> 8 GPUs)
    SCALE_EXPERT_AXIS    intra-node shard axis (default 8; dense model shards
                         batch, not experts, over it)
    SCALE_REPLICA_AXIS   cross-node replication; 1 = pure FSDP (default 1)
    SCALE_HIDDEN_DIM / SCALE_NUM_LAYERS / SCALE_NUM_HEADS / SCALE_INTERMEDIATE_DIM
    SCALE_BATCH / SCALE_SEQ_LEN / SCALE_STEPS
    SCALE_TRACKER        wandb | json_logger (default json_logger)
    SCALE_PROFILER_STEPS / SCALE_PROFILER_START
    SCALE_CHECKPOINTS    s3 (default) | local
    RUN_ID               unique run identifier
"""

import dataclasses
import datetime
import os
from typing import cast

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.text import BlockShuffleConfig
from levanter.optim import GrugMuonConfig, OptimizerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, slimpajama_6b_dataset
from experiments.grug.moe.launch_cw_scale import (
    DEFAULT_BATCH,
    DEFAULT_SEQ_LEN,
    GPUS_PER_NODE,
    HEAD_DIM,
    OUTPUT_SUBDIR,
    SCALE_OPTIMIZER,
    SCALE_TRAINER_DEFAULTS,
    VOCAB_SIZE,
)
from experiments.grug.moe.model import GrugModelConfig, RematMode
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugRunConfig, GrugTrainerConfig
from experiments.grug.moe.train_basic import run_grug_basic
from experiments.grug.moe.train_mid import run_grug_mid

_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


def build_basic_model() -> GrugModelConfig:
    """Minimalist dense transformer (overridable via SCALE_* env vars). Full MHA."""
    hidden_dim = env_int("SCALE_HIDDEN_DIM", 768)
    if hidden_dim % HEAD_DIM != 0:
        raise ValueError(f"SCALE_HIDDEN_DIM={hidden_dim} must be a multiple of head_dim={HEAD_DIM}")
    num_heads = env_int("SCALE_NUM_HEADS", hidden_dim // HEAD_DIM)
    intermediate_dim = env_int("SCALE_INTERMEDIATE_DIM", hidden_dim * 4)
    seq_len = env_int("SCALE_SEQ_LEN", DEFAULT_SEQ_LEN)
    remat_mode = os.environ.get("SCALE_REMAT", "recompute_all")
    if remat_mode not in ("recompute_all", "save_moe", "none"):
        raise ValueError(f"SCALE_REMAT={remat_mode!r} must be 'recompute_all', 'save_moe', or 'none'")
    attn_impl = os.environ.get("SCALE_ATTN_IMPL") or None
    if attn_impl not in (None, "reference", "gpu_fa4_cute", "gpu_fa4_thd", "tpu_splash"):
        raise ValueError(f"SCALE_ATTN_IMPL={attn_impl!r} is not a known GrugAttentionImplementation")
    # MoE dispatch backend for the leading MoE layers; None -> grug default ("ring").
    # On single-node GPU (no expert parallelism) the local backends are the right choice:
    # "scatter" (grouped GMM + scatter-add) or "sonic" (Triton gather/combine).
    moe_impl = os.environ.get("SCALE_MOE_IMPL") or None
    if moe_impl not in (None, "ring", "ragged_all_to_all", "deepep", "scatter", "sonic"):
        raise ValueError(f"SCALE_MOE_IMPL={moe_impl!r} is not a known MoeImplementation")
    # SCALE_ZERO_INIT zeros every weight (std=0): keeps the norm-free deep model finite
    # for throughput probes where we only measure fit/MFU, not learning.
    initializer_std = 0.0 if os.environ.get("SCALE_ZERO_INIT") == "1" else 0.02
    return GrugModelConfig(
        initializer_std=initializer_std,
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        num_layers=env_int("SCALE_NUM_LAYERS", 12),
        num_heads=num_heads,
        num_kv_heads=num_heads,  # full multi-head attention
        head_dim=HEAD_DIM,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=0,  # dense model: no shared expert
        # MoE knobs used only by the leading SCALE_NUM_MOE_LAYERS layers; 1/1 keeps dense.
        num_experts=env_int("SCALE_NUM_EXPERTS", 1),
        num_experts_per_token=env_int("SCALE_EXPERTS_PER_TOKEN", 1),
        moe_implementation=moe_impl,
        max_seq_len=seq_len,
        sliding_window=seq_len,
        remat_mode=cast(RematMode, remat_mode),
        attention_implementation=attn_impl,
    )


def run_grug_basic_trial(config: GrugMoeLaunchConfig) -> None:
    """Map launch knobs onto a Levanter trainer and dispatch the dense run."""
    initialize_from = latest_checkpoint_path(config.init_from) if config.init_from is not None else None
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=(
            dataclasses.replace(config.tracker, name=config.run_id)
            if isinstance(config.tracker, WandbConfig)
            else config.tracker
        ),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=initialize_from,
        checkpointer=config.checkpointer
        or resolve_checkpointer_output_path(CheckpointerConfig(save_interval=None, keep=None), config.output_path),
    )
    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    # SCALE_MODEL selects the model/training variant: "basic" (dense / leading- or
    # interleaved-MoE model_basic) or "mid" (every layer = shared expert + routed MoE).
    run_fn = run_grug_mid if os.environ.get("SCALE_MODEL", "basic").lower() == "mid" else run_grug_basic
    run_fn(
        GrugRunConfig(
            model=config.model,
            data=config.data,
            resources=config.resources,
            optimizer=config.optimizer,
            trainer=grug_trainer,
            eval=config.eval,
            processes_per_task=config.processes_per_task,
        )
    )


def build_basic_checkpoint(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Assemble the CoreWeave dense-transformer run as a lazy checkpoint from SCALE_* env."""
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")

    replicas = env_int("SCALE_GPU_REPLICAS", 1)
    expert_axis = env_int("SCALE_EXPERT_AXIS", 8)
    replica_axis = env_int("SCALE_REPLICA_AXIS", 1)
    batch_size = env_int("SCALE_BATCH", DEFAULT_BATCH)
    steps = env_int("SCALE_STEPS", 50)
    processes_per_task = env_int("SCALE_PROCESSES_PER_TASK", 1)
    profiler_steps = env_int("SCALE_PROFILER_STEPS", 0)
    profiler = ProfilerConfig(
        enabled=profiler_steps > 0,
        start_step=env_int("SCALE_PROFILER_START", 8),
        num_steps=profiler_steps,
        perfetto_link=os.environ.get("SCALE_PROFILER_LINK", "") == "1",
    )

    # LR override for the throughput/profile run (the no-norm model needs a tiny LR
    # to stay finite long enough to reach the profiler window).
    lr = float(os.environ.get("SCALE_LR") or SCALE_OPTIMIZER.learning_rate)
    optimizer: OptimizerConfig
    opt_name = os.environ.get("SCALE_OPTIMIZER", "adam").lower()
    if opt_name in ("muon", "grug_muon"):
        # Muon (one momentum buffer) for matrix/expert weights + AdamW for embed/head:
        # ~8 B/param vs Adam's 12, the memory lever for large-expert-count MoE.
        optimizer = GrugMuonConfig(learning_rate=lr, adam_lr=lr)
    elif opt_name in ("muonh", "grug_moe_muonh"):
        # May Recipe MuonH: Newton-Schulz + norm-preserving ("H") step on the matrix
        # groups (attn / experts / shared / gated), AdamH on lm_head, Adam on the rest.
        optimizer = GrugMoeMuonHConfig(learning_rate=lr, adam_lr=lr)
    elif opt_name in ("adamh", "grug_moe_adamh"):
        optimizer = GrugMoeAdamHConfig(learning_rate=lr, adam_lr=lr)
    else:
        optimizer = dataclasses.replace(SCALE_OPTIMIZER, learning_rate=lr)

    checkpoint_mode = os.environ.get("SCALE_CHECKPOINTS", "s3").lower()
    if checkpoint_mode in ("local", "none"):
        # "none" carries the same cheap /tmp config, but train_basic skips every save
        # (see SCALE_CHECKPOINTS handling there) so the probe writes no checkpoint.
        checkpointer = CheckpointerConfig(
            base_path=f"/tmp/grug-basic-ckpt/{run_id}",
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=None,
        )
    elif checkpoint_mode == "s3":
        checkpointer = None
    else:
        raise ValueError(f"SCALE_CHECKPOINTS={checkpoint_mode!r} must be 's3', 'local', or 'none'")

    model = build_basic_model()

    # Batch is sharded over (replica_dcn, data, expert); data absorbs the rest.
    data_axis = (replicas * GPUS_PER_NODE) // (replica_axis * expert_axis)
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"SCALE_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu("H100", count=GPUS_PER_NODE, cpu=32, ram="256g", disk="256g", replicas=replicas)

    use_wandb = os.environ.get("SCALE_TRACKER", "json_logger").lower() == "wandb"
    json_logger_name = os.environ.get("SCALE_JSON_LOGGER", "grug_basic_scale.metrics")
    wandb_entity = os.environ.get("WANDB_ENTITY") or None
    wandb_project = os.environ.get("WANDB_PROJECT", "marin_moe")

    grug_trainer = GrugTrainerConfig(
        expert_axis_size=expert_axis,
        replica_axis_size=replica_axis,
        **SCALE_TRAINER_DEFAULTS,
    )

    mp = os.environ.get("SCALE_MP", "params=float32,compute=bfloat16,output=bfloat16")
    name = f"grug-basic-cw-d{model.hidden_dim}-L{model.num_layers}-r{replicas}"
    slim = slimpajama_6b_dataset()

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        if use_wandb:
            tracker = WandbConfig(
                entity=wandb_entity,
                project=wandb_project,
                tags=["grug", "basic", "cw", "h100", "scale"],
                group="grug-basic-cw-scale",
                name=None,
                replicate_path=ctx.output_path,
            )
        else:
            tracker = JsonLoggerConfig(logger_name=json_logger_name)
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp=mp,
            tracker=tracker,
            optimizer=optimizer,
            grug_trainer=grug_trainer,
            processes_per_task=processes_per_task,
            eval=None,
            profiler=profiler,
            checkpointer=checkpointer,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{OUTPUT_SUBDIR}/{name}-{run_id}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_basic_trial,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    StepRunner().run([build_basic_checkpoint().lower()])
