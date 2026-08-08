# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the segmented and native-SM100 FA4 paths on a 1B-active FSDP Grug model."""

import dataclasses
import math

import click
from fray.cluster import ResourceConfig
from levanter.tracker.telemetry import TelemetryConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem import marin_temp_bucket, prefix_join

from experiments.grug.moe_hero_fsdp.heuristic import MoeHeuristic
from experiments.grug.moe_hero_fsdp.launch import (
    DEFAULT_WANDB_PROJECT,
    HERO_GPUS_PER_TASK,
    HeroThroughputResult,
    _hero_run_config,
    _slimpajama_6b_dataset,
)
from experiments.grug.moe_hero_fsdp.model import GrugModelConfig
from experiments.grug.moe_hero_fsdp.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_fsdp.train import GrugRunConfig, GrugRunMode, GrugTrainerConfig, run_grug

ACTIVE_PARAMETER_COUNT = 869_793_792
BENCHMARK_BATCH_SIZE = 512
BENCHMARK_NODES = 4
BENCHMARK_STEPS = 20
BENCHMARK_SEQUENCE_LENGTH = 4096
BENCHMARK_OUTPUT_TTL_DAYS = 30
PROFILE_START_STEP = 10
PROFILE_NUM_STEPS = 3
WANDB_GROUP = "fa4-sm100-1b"
THD_MAX_SEGMENTS = 32
SUPPORTED_ATTENTION_IMPLEMENTATIONS = ("gpu_fa4_cute", "gpu_fa4_thd")


def build_fa4_benchmark_configs(
    *, attention_implementation: str, num_train_steps: int = BENCHMARK_STEPS
) -> tuple[GrugModelConfig, GrugMoeMuonHConfig]:
    """Build the 15.44B-total, 869.8M-active model used for the FA4 comparison."""
    if attention_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"attention_implementation must be one of {SUPPORTED_ATTENTION_IMPLEMENTATIONS}, "
            f"got {attention_implementation!r}"
        )

    hidden_dim = 2048
    model = GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=hidden_dim,
        intermediate_dim=1024,
        shared_expert_intermediate_dim=1024,
        num_shared_experts=2,
        num_experts=128,
        num_experts_per_token=4,
        num_layers=18,
        num_heads=16,
        num_kv_heads=4,
        local_kv_heads=4,
        global_kv_heads=2,
        head_dim=128,
        max_seq_len=BENCHMARK_SEQUENCE_LENGTH,
        sliding_window=512,
        global_every=6,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(hidden_dim),
        qk_mult=1.3,
        sconv=True,
        attention_implementation=attention_implementation,
        # slimpajama-6b/2026.06.28 at sequence length 4096 holds at most 20 EOS-delimited
        # documents per sequence (441k windows sampled; p99.99 is 17), so 32 clears the
        # measured tail without padding cu_seqlens further than the native kernel needs.
        thd_max_segments=THD_MAX_SEGMENTS,
        moe_implementation="sonic_cute",
        expert_chunks=4,
        report_capacity_overflow=True,
        rope_fused=True,
    )
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_train_steps,
        batch_size=BENCHMARK_BATCH_SIZE,
        hidden_dim=model.hidden_dim,
        seq_len=model.max_seq_len,
    )
    return model, optimizer


def build_fa4_benchmark_run(
    *,
    run_id: str,
    attention_implementation: str,
    profile: bool,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build one disposable 16-GPU arm with an optional steady-state XPlane capture."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    model, optimizer = build_fa4_benchmark_configs(attention_implementation=attention_implementation)
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=False,
        save_checkpoints=False,
        expert_axis_size=1,
        replica_axis_size=1,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_TASK,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=BENCHMARK_NODES,
    )
    name = f"grug/fa4-sm100-1b/{run_id}"
    version = resolve_version(name, version)
    step_name = user_namespaced_name(name, version)
    output_path = marin_temp_bucket(
        ttl_days=BENCHMARK_OUTPUT_TTL_DAYS,
        prefix=prefix_join(step_name, version),
    )
    slim = _slimpajama_6b_dataset()

    def build_config(ctx: StepContext) -> GrugRunConfig:
        config = _hero_run_config(
            ctx=ctx,
            run_id=run_id,
            batch_size=BENCHMARK_BATCH_SIZE,
            num_steps=BENCHMARK_STEPS,
            model=model,
            optimizer=optimizer,
            grug_trainer=grug_trainer,
            wandb_project=DEFAULT_WANDB_PROJECT,
            slim=slim,
            run_mode=GrugRunMode.DEFAULT,
            profile_start_step=PROFILE_START_STEP if profile else None,
            profile_num_steps=PROFILE_NUM_STEPS,
        )
        tracker = (
            WandbConfig(
                entity="marin-community",
                project=DEFAULT_WANDB_PROJECT,
                tags=["grug", "moe", "fsdp", "gb200", "fa4-sm100", "1b-active"],
                group=WANDB_GROUP,
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            TelemetryConfig(),
        )
        trainer = dataclasses.replace(config.trainer.trainer, tracker=tracker)
        return dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, trainer=trainer))

    return ArtifactStep(
        name=step_name,
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
        override_path=output_path,
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--attention-implementation",
    type=click.Choice(SUPPORTED_ATTENTION_IMPLEMENTATIONS),
    required=True,
)
@click.option(
    "--profile/--no-profile",
    default=False,
    show_default=True,
    help=f"Capture XPlane for steps {PROFILE_START_STEP}-{PROFILE_START_STEP + PROFILE_NUM_STEPS - 1}.",
)
@build_options
def main(run_id: str, attention_implementation: str, profile: bool) -> ArtifactStep[HeroThroughputResult]:
    return build_fa4_benchmark_run(
        run_id=run_id,
        attention_implementation=attention_implementation,
        profile=profile,
    )


if __name__ == "__main__":
    main()
