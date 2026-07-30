# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hardcoded one-rack d5120 EP64 launcher with 256 routed experts and top-8 routing."""

import dataclasses
import datetime
import os

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.grug.attention import RotaryConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size

HERO_STEPS = 350
HERO_BATCH_SIZE = 1024
HERO_PROCESSES_PER_TASK = 1
HERO_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"

HERO_MODEL = GrugModelConfig(
    vocab_size=llama3_tokenizer_vocab_size,
    hidden_dim=5120,
    intermediate_dim=1280,
    shared_expert_intermediate_dim=5120,
    num_shared_experts=1,
    num_experts=256,
    num_experts_per_token=8,
    num_layers=48,
    num_heads=40,
    num_kv_heads=10,
    head_dim=128,
    max_seq_len=4096,
    sliding_window=2048,
    capacity_factor=1.0625,
    layer_norm_eps=1e-5,
    initializer_std=0.006987712429686843,
    qk_mult=1.3,
    router_z_loss_coef=0.0,
    disable_pko=True,
    disable_long_rope=True,
    attention_implementation="gpu_fa4_cute",
    moe_implementation="ragged_all_to_all",
    expert_chunks=1,
    report_capacity_overflow=True,
    remat_mode="recompute_all",
    rope=RotaryConfig(theta=10_000.0, scaling_factor=None),
    use_array_stacked_blocks=True,
)

HERO_OPTIMIZER = GrugMoeMuonHConfig(
    learning_rate=0.038956464533085024,
    min_lr_ratio=0.05,
    warmup=0.01,
    decay=None,
    rewarmup=0.0,
    cooldown=None,
    cycle_length=None,
    cycles=None,
    lr_schedule="linear",
    haps=None,
    adam_lr=0.008989953353788853,
    momentum=0.95,
    nesterov=True,
    backend_steps=5,
    beta1=0.9062,
    beta2=0.9684910757595268,
    epsilon=1.810213843721233e-16,
    muon_epsilon=1e-8,
    max_grad_norm=None,
    coefficient_type="quintic",
)

HERO_GRUG_TRAINER = GrugTrainerConfig(
    data_seed=None,
    log_every=1,
    ema_beta=None,
    z_loss_weight=1e-4,
    offload_opt_state=False,
    expert_axis_size=64,
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
    """Build the fixed one-rack EP64 throughput run."""
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("hero-ep-%Y%m%d-%H%M%S")
    name = f"experiments/grug-moe-hero-ep/d5120-L48-e256-{run_id}"
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
            tracker=JsonLoggerConfig(logger_name=f"{run_id}.metrics"),
            watch=WatchConfig(interval=0),
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
