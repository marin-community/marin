# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP8 loss-curve validation for PR #7079 on the cw-us-east-02a H100 cluster.

Two arms, launched from different checkouts of this same file:

- **bf16 control** (from ``main``): row-13 Grug MoE, bf16 everywhere.
- **FP8 test** (from ``fp8-moe-mlp-comms`` merged with main, ``FP8VAL_FP8=1``):
  identical config plus ``GrugFp8Config(wire=True, dense=True)`` — FP8 expert
  grouped GEMMs, FP8 EP dispatch/combine wire, and FP8 dense GEMMs for the
  attention q/k/v/o and shared-expert projections.

Model is the flagship "row 13" shape (d2560, 26 layers, 64 experts top-4, MHA
20x128, seq 4096) with the production MuonH recipe run through a full LR
schedule (warmup 10%, cooldown to 0) so precision effects in the cooldown tail
are exercised (see #6486). Data is the SlimPajama-6B llama3-tokenized cache
already materialized under this cluster's MARIN_PREFIX — no cross-region reads.

Each arm runs on 4 nodes / 32 H100s: experts sharded 8-way over intra-node
NVLink, FSDP over the 4-way cross-node ``data`` axis.

Env knobs:

    FP8VAL_FP8           1 enables FP8 (requires the PR branch); default off
    FP8VAL_STEPS         training steps (default 11000 ~= 5.8B tokens)
    FP8VAL_BATCH         global batch in sequences (default 128)
    FP8VAL_GPU_REPLICAS  8xH100 nodes per arm (default 4)
    FP8VAL_EXPERT_AXIS   expert-parallel axis size (default 8)
    RUN_ID               unique run identifier (also the W&B run name)
"""

import datetime
import os

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial, slimpajama_6b_dataset
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.llama import llama3_tokenizer_vocab_size

GPUS_PER_NODE = 8
HIDDEN_DIM = 2560
SEQ_LEN = 4096

# Row-13 production MuonH recipe; min_lr_ratio=0 cools all the way down so the
# tail of the schedule (where subtle precision gaps surface, per #6486) is part
# of the comparison.
VAL_OPTIMIZER = GrugMoeMuonHConfig(learning_rate=1e-3, adam_lr=1e-4, min_lr_ratio=0.0, warmup=0.1)

OUTPUT_SUBDIR = "experiments/grug-moe-fp8val"

_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")

_FP8_ENABLED = os.environ.get("FP8VAL_FP8", "") == "1"


def build_val_model() -> GrugModelConfig:
    """Row-13 architecture: d2560 / 26L / 64 experts top-4 / MHA 20x128 / seq 4096."""
    kwargs = {}
    if _FP8_ENABLED:
        # Branch-only import, guarded so the bf16 control arm can run this
        # launcher from a `main` checkout where GrugFp8Config does not exist.
        from experiments.grug.moe.model import GrugFp8Config  # noqa: PLC0415

        kwargs["fp8"] = GrugFp8Config(wire=True, dense=True)
    return GrugModelConfig(
        vocab_size=llama3_tokenizer_vocab_size,
        hidden_dim=HIDDEN_DIM,
        num_layers=26,
        num_heads=20,
        num_kv_heads=20,
        head_dim=128,
        intermediate_dim=HIDDEN_DIM // 2,
        shared_expert_intermediate_dim=HIDDEN_DIM // 2,
        num_experts=64,
        num_experts_per_token=4,
        max_seq_len=SEQ_LEN,
        sliding_window=2048,
        initializer_std=0.5 / (HIDDEN_DIM**0.5),
        qk_mult=1.3,
        attention_implementation="gpu_fa4_cute",
        remat_mode="recompute_all",
        **kwargs,
    )


def build_val_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    arm = "fp8" if _FP8_ENABLED else "bf16"
    default_run_id = f"fp8val-{arm}-{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d-%H%M')}"
    run_id = os.environ.get("RUN_ID") or default_run_id

    replicas = env_int("FP8VAL_GPU_REPLICAS", 4)
    expert_axis = env_int("FP8VAL_EXPERT_AXIS", 8)
    batch_size = env_int("FP8VAL_BATCH", 128)
    steps = env_int("FP8VAL_STEPS", 11000)

    model = build_val_model()
    data_axis = (replicas * GPUS_PER_NODE) // expert_axis
    batch_shards = data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"FP8VAL_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu("H100", count=GPUS_PER_NODE, cpu=32, ram="256g", disk="256g", replicas=replicas)
    slim = slimpajama_6b_dataset()

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin_moe",
                tags=["fp8-loss-val", "pr7079", "grug", "moe", "h100", arm],
                group="fp8-loss-val-7079",
                name=None,
            ),
            optimizer=VAL_OPTIMIZER,
            grug_trainer=GrugTrainerConfig(
                expert_axis_size=expert_axis,
                replica_axis_size=1,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            ),
            eval=None,
        )

    step_name = f"{OUTPUT_SUBDIR}/fp8val-{arm}-{run_id}"
    version = resolve_version(step_name, version)
    return ArtifactStep(
        name=user_namespaced_name(step_name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    experiment_main(build_val_checkpoint)()
