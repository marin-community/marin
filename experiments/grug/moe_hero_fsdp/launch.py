# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hardcoded one-rack GB200 launcher for the FSDP MoE hero configuration."""

import dataclasses
import math
import os

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.grug.attention import RotaryConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe_hero_fsdp.model import GrugModelConfig
from experiments.grug.moe_hero_fsdp.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_fsdp.train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.llama import llama3_tokenizer

HERO_RUN_ID = "moe-hero-fsdp"
HERO_STEPS = 25
HERO_BATCH_SIZE = 1152
HERO_PROCESSES_PER_TASK = 1
HERO_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
HERO_TOKENS = float(HERO_STEPS * HERO_BATCH_SIZE * 4096)

HERO_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=6144,
    intermediate_dim=3072,
    shared_expert_intermediate_dim=6144,
    num_shared_experts=2,
    num_experts=128,
    num_experts_per_token=4,
    num_layers=48,
    num_heads=48,
    num_kv_heads=12,
    local_kv_heads=12,
    global_kv_heads=6,
    head_dim=128,
    max_seq_len=4096,
    sliding_window=512,
    global_every=6,
    capacity_factor=1.0,
    layer_norm_eps=1e-5,
    initializer_std=0.5 / math.sqrt(6144),
    qk_mult=1.3,
    xsa=True,
    router_z_loss_coef=0.0,
    disable_pko=True,
    disable_long_rope=True,
    attention_implementation="gpu_fa4_cute",
    moe_implementation="sonic_cute",
    expert_chunks=4,
    report_capacity_overflow=True,
    remat_mode="recompute_all",
    rope=RotaryConfig(theta=10_000.0, scaling_factor=None),
    rope_fused=True,
    use_array_stacked_blocks=True,
)


def _build_hero_optimizer() -> GrugMoeMuonHConfig:
    derived = MoeHeuristic(min_lr_ratio=0.05, max_learning_rate=0.05).build_optimizer_config(
        batch_size=HERO_BATCH_SIZE,
        tokens=HERO_TOKENS,
        hidden_dim=HERO_MODEL.hidden_dim,
        seq_len=HERO_MODEL.max_seq_len,
    )
    return GrugMoeMuonHConfig(
        learning_rate=derived.learning_rate,
        weight_decay=0.1,
        min_lr_ratio=derived.min_lr_ratio,
        warmup=derived.warmup,
        decay=derived.decay,
        rewarmup=0.0,
        cooldown=None,
        cycle_length=None,
        cycles=None,
        lr_schedule="linear",
        haps=None,
        weight_decay_modules=None,
        default_weight_decay_mask=None,
        adam_lr=derived.adam_lr,
        momentum=0.95,
        nesterov=True,
        backend_steps=5,
        beta1=derived.beta1,
        beta2=derived.beta2,
        epsilon=derived.epsilon,
        muon_epsilon=1e-8,
        max_grad_norm=None,
        coefficient_type="quintic",
    )


HERO_OPTIMIZER = _build_hero_optimizer()

HERO_GRUG_TRAINER = GrugTrainerConfig(
    data_seed=None,
    log_every=1,
    ema_beta=None,
    z_loss_weight=1e-4,
    offload_opt_state=True,
    expert_axis_size=1,
    replica_axis_size=1,
    sharding_dump_path=None,
)

HERO_TRAIN_RESOURCES = ResourceConfig.with_gpu(
    "GB200",
    count=4,
    cpu=32,
    ram="256g",
    disk="256g",
    replicas=16,
)

_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


def _slimpajama_6b_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
    )


def build_hero_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build the fixed 64-GPU FSDP hero run."""
    run_id = os.environ.get("RUN_ID") or HERO_RUN_ID
    name = "grug/moe-hero-fsdp"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=HERO_BATCH_SIZE,
            num_train_steps=HERO_STEPS,
            profiler=ProfilerConfig(enabled=False, start_step=8, num_steps=0),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["grug", "moe", "hero", "fsdp", "gb200"],
                group="moe-hero-fsdp",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=20),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=CheckpointerConfig(
                base_path=f"{ctx.output_path}/checkpoints",
                temporary_base_path=None,
                save_interval=None,
                keep=None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        return GrugRunConfig(
            model=HERO_MODEL,
            data=mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE),
            resources=ctx.runtime_arg("train_resources"),
            optimizer=HERO_OPTIMIZER,
            trainer=dataclasses.replace(HERO_GRUG_TRAINER, trainer=trainer),
            eval=None,
            processes_per_task=HERO_PROCESSES_PER_TASK,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": HERO_TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    experiment_main(build_hero_checkpoint)()
