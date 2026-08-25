# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H100 scaling ladder for the Grug MoE EP hero recipe.

The three rungs keep the existing hero model, data, optimizer, and 791-token-per-active-parameter
scaling rules while mapping the expert and replica axes onto Hopper nodes. Every rung uses a global
sequence batch of 1024.

    size   H100 GPUs  EP per task  global batch  steps  tokens
    d384       4           4           1024        1520   6.4B
    d512       8           8           1024        3930    16B
    d768      16           8           1024       11420    48B

Use ``--batch-size`` for one-off batch comparisons. The step count and compute-scaled optimizer are
recomputed from the override unless ``--num-steps`` is also set.
"""

import dataclasses
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import data_local_temporary_checkpoint_base_path, temporary_checkpoint_base_path
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.checkpointing import RESTORE_BARRIER_TIMEOUT
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    HARRIER_MIX_2026_08_18_STORE,
    HARRIER_MIX_2026_08_18_TAG,
    harrier_mix_2026_08_18_data_config,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.launch_mfu_test import (
    DEFAULT_WANDB_PROJECT,
    HERO_MIXED_PRECISION,
    HeroThroughputResult,
    _validation_datasets,
)
from experiments.grug.moe_hero_ep.launch_scaling_ladder import TENSORSTORE_CACHE_BYTES, TOKENS_PER_ACTIVE_PARAM
from experiments.grug.moe_hero_ep.small_scale_abl_launch import (
    _EP_CAPACITY_FACTOR,
    SEQ_LEN,
    SmallShape,
    _active_params,
    _small_model,
)
from experiments.grug.moe_hero_ep.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    MasterParamMode,
    WatchMode,
    _compute_flops,
    run_grug,
)
from experiments.marin_tokenizer import marin_tokenizer

H100_LADDER_SIZES = ("d384", "d512", "d768")
GLOBAL_BATCH_SIZE = 1024
QB_HIST_BINS = 10_000
WATCH_INTERVAL = 10
RESUME_SAVE_INTERVAL = timedelta(hours=1)
STEP_TIMEOUT = timedelta(minutes=15)
PROCESS_STALL_TIMEOUT = timedelta(hours=1)
STARTUP_TIMEOUT = timedelta(seconds=2 * RESTORE_BARRIER_TIMEOUT)
MAX_RETRIES_FAILURE = 3
MAX_TASK_FAILURES = 3


@dataclasses.dataclass(frozen=True)
class H100LadderRung:
    shape: SmallShape
    gpus_per_task: int
    task_count: int

    @property
    def global_device_count(self) -> int:
        return self.gpus_per_task * self.task_count


def _h100_ladder_rung(size: str) -> H100LadderRung:
    if size == "d384":
        return H100LadderRung(SmallShape(384, 4, 3, 1, 1), gpus_per_task=4, task_count=1)
    if size == "d512":
        return H100LadderRung(SmallShape(512, 6, 4, 1, 1), gpus_per_task=8, task_count=1)
    if size == "d768":
        return H100LadderRung(SmallShape(768, 8, 6, 1, 1), gpus_per_task=8, task_count=2)
    raise ValueError(f"size must be one of {list(H100_LADDER_SIZES)}, got {size!r}")


def _h100_ladder_model(rung: H100LadderRung):
    return _small_model(
        rung.shape,
        _EP_CAPACITY_FACTOR,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="fixed_pooled_wave_all_to_all",
        expert_chunks=1,
        seq_len=SEQ_LEN,
        num_experts=384,
        num_experts_per_token=8,
        intermediate_dim=None,
        latent_dim=None,
        pooled_transport_capacity_factor=_EP_CAPACITY_FACTOR,
        num_expert_waves=3,
        qb_use_histogram=True,
        qb_hist_bins=QB_HIST_BINS,
    )


def build_h100_ladder_run(
    *,
    run_id: str,
    size: str,
    num_steps: int | None = None,
    batch_size: int = GLOBAL_BATCH_SIZE,
    checkpoint_every: int | None = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build one H100 scaling-ladder rung.

    The default step budget trains for ``TOKENS_PER_ACTIVE_PARAM`` tokens per active parameter.
    Narrow-rung evaluation runs every 5% of training. Permanent checkpoints default to the final
    step, with one rolling hourly checkpoint on region-local temporary storage for recovery.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not wandb_project.strip():
        raise ValueError("wandb_project must not be empty")
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be positive, got {checkpoint_every}")

    rung = _h100_ladder_rung(size)
    model = _h100_ladder_model(rung)
    training_tokens = TOKENS_PER_ACTIVE_PARAM * _active_params(model)
    if batch_size <= 0 or batch_size % rung.global_device_count != 0:
        raise ValueError(f"batch_size must be positive and divisible by {rung.global_device_count}, got {batch_size}")

    global_tokens_per_step = batch_size * SEQ_LEN
    if num_steps is None:
        num_steps = max(1, round(training_tokens / global_tokens_per_step))
    elif num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    flops_per_example, _ = _compute_flops(model_config=model)
    run_flops = flops_per_example * batch_size * num_steps
    steps_per_eval = max(1, round(num_steps / 20))
    keep_permanent = None if checkpoint_every is None else [{"every": checkpoint_every}]
    optimizer = dataclasses.replace(
        MoeHeuristic().build_optimizer_config(
            num_train_steps=num_steps,
            batch_size=batch_size,
            hidden_dim=model.hidden_dim,
            seq_len=SEQ_LEN,
        ),
        use_syrk=True,
    )
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=True,
        master_param_mode=MasterParamMode.FP32_PINNED_HOST,
        watch_mode=WatchMode.INLINE,
        save_checkpoints=True,
        expert_axis_size=rung.gpus_per_task,
        replica_axis_size=rung.task_count,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "H100",
        count=rung.gpus_per_task,
        cpu=32,
        ram="600g",
        disk="900g",
        replicas=rung.task_count,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    validation = [*_validation_datasets(), *uncheatable_datasets(tokenizer=marin_tokenizer).values()]

    def build_config(ctx: StepContext) -> GrugRunConfig:
        permanent_checkpoint_path = prefix_join(ctx.output_path, "checkpoints")
        temporary_checkpoint_path = temporary_checkpoint_base_path(ctx.output_path)
        data_local_checkpoint_path = data_local_temporary_checkpoint_base_path(ctx.output_path)
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=num_steps,
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
                    "scaling-ladder",
                    "h100",
                    f"shape-{size}",
                    f"h100-nodes-{rung.task_count}",
                    f"ep{rung.gpus_per_task}",
                    HARRIER_MIX_2026_08_18_TAG,
                ],
                group="moe-hero-ep-h100-scaling-ladder",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=WATCH_INTERVAL),
            progress_watchdog=ProgressWatchdogConfig(
                step_timeout=STEP_TIMEOUT,
                process_timeout=PROCESS_STALL_TIMEOUT,
                startup_timeout=STARTUP_TIMEOUT,
            ),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            load_checkpoint_path=[
                permanent_checkpoint_path,
                temporary_checkpoint_path,
                data_local_checkpoint_path,
            ],
            checkpointer=CheckpointerConfig(
                base_path=permanent_checkpoint_path,
                temporary_base_path=temporary_checkpoint_path,
                save_interval=RESUME_SAVE_INTERVAL,
                keep=keep_permanent,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        return GrugRunConfig(
            model=model,
            data=harrier_mix_2026_08_18_data_config(
                ctx=ctx,
                total_steps=num_steps,
                batch_size=batch_size,
                max_seq_len=model.max_seq_len,
                experiment_flops=run_flops,
                validation=validation,
            ),
            resources=ctx.runtime_arg("train_resources"),
            tensorstore_cache_bytes=TENSORSTORE_CACHE_BYTES,
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=GrugEvalConfig(
                steps_per_eval=steps_per_eval,
                eval_batch_size=rung.global_device_count,
                eval_ema=False,
                compute_bpb=True,
                dropless_eval=True,
                dropless_eval_moe_implementation="sonic",
            ),
            stop_after_steps=num_steps,
            processes_per_task=rung.gpus_per_task,
            max_retries_failure=MAX_RETRIES_FAILURE,
            max_task_failures=MAX_TASK_FAILURES,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(HARRIER_MIX_2026_08_18_STORE, *validation),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option("--size", required=True, type=click.Choice(H100_LADDER_SIZES), help="H100 ladder rung width.")
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Training steps. Default trains 791 tokens per active parameter at the rung's batch.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=GLOBAL_BATCH_SIZE,
    show_default=True,
    help="Global sequence batch.",
)
@click.option(
    "--checkpoint-every",
    type=click.IntRange(min=1),
    default=None,
    help="Keep a permanent checkpoint every N steps. Default keeps only the final checkpoint.",
)
@click.option(
    "--wandb-project",
    envvar="WANDB_PROJECT",
    default=DEFAULT_WANDB_PROJECT,
    show_default=True,
    help="W&B project for the run.",
)
@build_options
def main(
    run_id: str,
    size: str,
    num_steps: int | None,
    batch_size: int,
    checkpoint_every: int | None,
    wandb_project: str,
) -> ArtifactStep[HeroThroughputResult]:
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        num_steps=num_steps,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        wandb_project=wandb_project,
    )


if __name__ == "__main__":
    main()
