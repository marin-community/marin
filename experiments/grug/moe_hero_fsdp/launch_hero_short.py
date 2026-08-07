# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short hero run: the full FSDP hero shape, capped at 2k steps, on a 1T-token LR schedule.

Mirrors ``launch.py``'s hero settings (d6144 shape, GB200 racks, MuonH) on the datakit stage-1
(phase_0) data mix, but caps the trainer at ``SHORT_STEPS`` while building the LR schedule over the
1e12-token horizon instead of the
run length. ``run_grug`` builds the schedule with ``optimizer.build(trainer.num_train_steps)``, so a
plain 2k-step run would compress warmup+decay into 2k steps; here the schedule length is pinned to the
1T horizon and the 2k steps execute only its early portion -- LR magnitude and shape as if 1T tokens.

Warmup is the base 1% fraction of the *schedule* horizon (~schedule_steps/100), so at 1 rack the 2k
steps sit inside the warmup ramp; scale ``--dp-racks`` up (larger batch -> shorter horizon -> shorter
warmup) to reach peak LR within the run.
"""

import dataclasses
import os

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent
from levanter.tracker.telemetry import TelemetryConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.datakit.source_key import datakit_source_path
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem import prefix_join

from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config
from experiments.grug.moe_hero_fsdp.heuristic import build_hero_configs
from experiments.grug.moe_hero_fsdp.launch import (
    DEFAULT_WANDB_PROJECT,
    HERO_CHECKPOINT_INTERVAL,
    HERO_FSDP_BATCH_SIZE,
    HERO_MIXED_PRECISION,
    HERO_NODES_PER_RACK,
    HERO_PROCESS_STALL_TIMEOUT,
    HERO_PROCESSES_PER_TASK,
    HERO_STALL_DIAGNOSTIC_TIMEOUT,
    HERO_TRAIN_STEP_TIMEOUT,
    HeroThroughputResult,
)
from experiments.grug.moe_hero_fsdp.train import GrugRunConfig, GrugTrainerConfig, run_grug

SHORT_STEPS = 2000
SCHEDULE_TOKENS = 1_000_000_000_000  # 1T: LR-schedule horizon, decoupled from the 2k-step run
SEQ_LEN = 4096  # hero max_seq_len (asserted against the built model below)


def _root_component(component: DatasetComponent | ConcatDatasetComponent) -> DatasetComponent | ConcatDatasetComponent:
    """Root a datakit component's relative cache_dir against MARIN_PREFIX; absolute paths pass through."""
    if isinstance(component, ConcatDatasetComponent):
        return dataclasses.replace(component, children={n: _root_component(c) for n, c in component.children.items()})
    return dataclasses.replace(component, cache_dir=datakit_source_path(component.cache_dir))


def build_hero_short_run(
    *, run_id: str, dp_racks: int, save_checkpoints: bool = False, version: str | None = None
) -> ArtifactStep[HeroThroughputResult]:
    """The hero shape for ``SHORT_STEPS`` steps with a 1T-token LR schedule, on ``dp_racks`` GB200 racks."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if dp_racks <= 0:
        raise ValueError(f"dp_racks must be positive, got {dp_racks}")

    batch_size = dp_racks * HERO_FSDP_BATCH_SIZE
    schedule_steps = round(SCHEDULE_TOKENS / (batch_size * SEQ_LEN))
    # LR magnitude (heuristic tokens = schedule_steps * batch * seq ~= 1e12) and schedule length both
    # come from the 1T horizon; the trainer below runs only SHORT_STEPS of it.
    model, base_optimizer = build_hero_configs(num_train_steps=schedule_steps, batch_size=batch_size)
    assert model.max_seq_len == SEQ_LEN, f"SEQ_LEN {SEQ_LEN} != hero max_seq_len {model.max_seq_len}"
    # Pin the LR schedule to the 1T horizon; run_grug otherwise builds it over trainer.num_train_steps.
    optimizer = dataclasses.replace(base_optimizer, schedule_num_train_steps_override=schedule_steps)

    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=True,
        save_checkpoints=save_checkpoints,
        expert_axis_size=1,
        replica_axis_size=dp_racks,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200", count=4, cpu=120, ram="850g", disk="1t", replicas=HERO_NODES_PER_RACK * dp_racks
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=SHORT_STEPS,
            profiler=ProfilerConfig(enabled=False, start_step=8, num_steps=0),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=(
                WandbConfig(
                    entity="marin-community",
                    project=os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT,
                    tags=["grug", "moe", "hero", "fsdp", "gb200", "short"],
                    group="moe-hero-fsdp-short",
                    name=run_id,
                    replicate_path=ctx.output_path,
                ),
                TelemetryConfig(),
            ),
            watch=WatchConfig(interval=20),
            progress_watchdog=ProgressWatchdogConfig(
                step_timeout=HERO_TRAIN_STEP_TIMEOUT,
                process_timeout=HERO_PROCESS_STALL_TIMEOUT,
                diagnostic_timeout=HERO_STALL_DIAGNOSTIC_TIMEOUT,
            ),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=CheckpointerConfig(
                base_path=prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=HERO_CHECKPOINT_INTERVAL,
                keep=None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        # Datakit two-phase mixture pinned to stage 1 (phase_0): drop the phase_1 transition so the
        # short run trains on a constant stage-1 mix. Relative bucket cache_dirs root against MARIN_PREFIX.
        data = _datakit_data_config(
            total_steps=SHORT_STEPS,
            batch_size=batch_size,
            max_seq_len=SEQ_LEN,
            enable_simulated_epoching=False,
            val_components={},
        )
        data = dataclasses.replace(data, train_weights=[data.train_weights[0]])
        data = dataclasses.replace(data, components={n: _root_component(c) for n, c in data.components.items()})
        return GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=None,
            processes_per_task=HERO_PROCESSES_PER_TASK,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option("--dp-racks", type=click.IntRange(min=1), required=True, help="Data-parallel NVL72 rack count.")
@click.option(
    "--save-checkpoints/--no-save-checkpoints",
    default=False,
    show_default=True,
    help="Write the (multi-TiB) final hero checkpoint; off by default for a short run.",
)
@build_options
def main(run_id: str, dp_racks: int, save_checkpoints: bool) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_short_run(run_id=run_id, dp_racks=dp_racks, save_checkpoints=save_checkpoints)


if __name__ == "__main__":
    main()
