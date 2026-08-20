# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack d6144 checkpoint and restore diagnostic."""

import os
from dataclasses import dataclass

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_ep.heuristic import build_hero_configs
from experiments.grug.moe_hero_ep.launch_mfu_test import (
    DEFAULT_WANDB_PROJECT,
    HERO_EP_BATCH_SIZE,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MIXED_PRECISION,
    HeroThroughputResult,
)
from experiments.grug.moe_hero_ep.launch_scaling_ladder import TENSORSTORE_CACHE_BYTES, _ladder_model
from experiments.grug.moe_hero_ep.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    MasterParamMode,
    TrainingDataMode,
    WatchMode,
    run_grug,
)

REFERENCE_RACKS = 11
REFERENCE_SCHEDULE_STEPS = 390_251
ONE_RACK_TASKS = HERO_EP_NODES
TASK_MEMORY = "850g"
TEMP_TTL_DAYS = 1
CHECKPOINT_RESTORE_GROUP = "moe-hero-ep-checkpoint-restore"


@dataclass(frozen=True)
class CheckpointRestorePhaseConfig:
    grug: GrugRunConfig
    abort_after_completion: bool


def run_checkpoint_restore_phase(config: CheckpointRestorePhaseConfig) -> None:
    """Run one bounded training phase and optionally abort its coordinator."""
    run_grug(config.grug)
    if config.abort_after_completion:
        raise RuntimeError(
            f"Intentional coordinator abort after checkpointed step {config.grug.stop_after_steps}."
        )


def build_checkpoint_restore_phase(
    *,
    run_id: str,
    owner: str,
    stop_after_steps: int,
    abort_after_completion: bool,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build one phase of the isolated one-rack checkpoint/restore diagnostic."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not owner.strip() or "/" in owner or ".." in owner:
        raise ValueError(f"owner must be one path segment, got {owner!r}")
    if stop_after_steps <= 0 or stop_after_steps > REFERENCE_SCHEDULE_STEPS:
        raise ValueError(
            f"stop_after_steps must be in [1, {REFERENCE_SCHEDULE_STEPS}], got {stop_after_steps}"
        )

    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    output_path = marin_temp_bucket(
        ttl_days=TEMP_TTL_DAYS,
        prefix=f"users/{owner}/hero-checkpoint-restore/{run_id}",
    )
    model = _ladder_model("d6144")
    reference_batch_size = HERO_EP_BATCH_SIZE * REFERENCE_RACKS
    _, optimizer = build_hero_configs(
        num_train_steps=REFERENCE_SCHEDULE_STEPS,
        batch_size=reference_batch_size,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=120,
        ram=TASK_MEMORY,
        disk="1t",
        replicas=ONE_RACK_TASKS,
    )
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT

    def build_config(ctx: StepContext) -> CheckpointRestorePhaseConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=HERO_EP_BATCH_SIZE,
            num_train_steps=REFERENCE_SCHEDULE_STEPS,
            profiler=ProfilerConfig(enabled=False),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "hero",
                    "ep",
                    "checkpoint-restore",
                    "synthetic-data",
                    "shape-d6144",
                    "racks-1",
                    "gb200",
                    "issue-8492",
                ],
                group=CHECKPOINT_RESTORE_GROUP,
                name=run_id,
                replicate_path=ctx.output_path,
                resume="allow",
            ),
            watch=WatchConfig(interval=10),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=CheckpointerConfig(
                base_path=prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=None,
                keep=None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=False,
                keep_last_temporary_checkpoints=0,
            ),
        )
        grug = GrugRunConfig(
            model=model,
            data=LmDataConfig(
                tokenizer="passthrough",
                vocab_size=model.vocab_size,
                auto_build_caches=False,
            ),
            resources=ctx.runtime_arg("train_resources"),
            tensorstore_cache_bytes=TENSORSTORE_CACHE_BYTES,
            optimizer=optimizer,
            trainer=GrugTrainerConfig(
                trainer=trainer,
                data_seed=None,
                log_every=1,
                ema_beta=None,
                z_loss_weight=1e-4,
                offload_opt_state=True,
                master_param_mode=MasterParamMode.FP32_PINNED_HOST,
                training_data_mode=TrainingDataMode.SYNTHETIC,
                watch_mode=WatchMode.INLINE,
                save_checkpoints=True,
                expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
                replica_axis_size=1,
                sharding_dump_path=None,
            ),
            eval=None,
            stop_after_steps=stop_after_steps,
            processes_per_task=HERO_GPUS_PER_NODE,
            max_retries_failure=1,
            max_task_failures=1,
        )
        return CheckpointRestorePhaseConfig(
            grug=grug,
            abort_after_completion=abort_after_completion,
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_checkpoint_restore_phase,
        build_config=build_config,
        runtime_args={"train_resources": train_resources},
        override_path=output_path,
    )


@click.command()
@click.option("--run-id", required=True, help="Shared run identifier for both phases and W&B.")
@click.option("--owner", required=True, help="Owner segment below the one-day temporary prefix.")
@click.option("--stop-after-steps", required=True, type=click.IntRange(min=1))
@click.option(
    "--abort-after-completion/--no-abort-after-completion",
    default=False,
    show_default=True,
    help="Fail the coordinator after the bounded GPU child commits its final checkpoint.",
)
@build_options
def main(
    run_id: str,
    owner: str,
    stop_after_steps: int,
    abort_after_completion: bool,
) -> ArtifactStep[HeroThroughputResult]:
    return build_checkpoint_restore_phase(
        run_id=run_id,
        owner=owner,
        stop_after_steps=stop_after_steps,
        abort_after_completion=abort_after_completion,
    )


if __name__ == "__main__":
    main()
