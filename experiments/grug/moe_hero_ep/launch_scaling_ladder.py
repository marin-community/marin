# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hero-shape scaling ladder: one recipe, five widths.

Every rung trains the *same* EP hero recipe -- 384 routed experts, top-8, hidden/2-wide experts in a
hidden/2 latent, ragged all-to-all transport, the Harrier 2026.08.18 two-phase mixture on the
Marin tokenizer, offloaded MuonH state, the QB histogram estimator, and a dropless held-out eval --
and differs only in width and the rack count it spans. Behaviour is uniform across the ladder so a
rung predicts the d6144 hero. ``d6144`` is the hero itself.

    size   racks  batch    steps  eval        checkpoints  tokens  active  total   FLOPs
    d768     1     1024    11420  every 5%    final only     48B     61M    1.6B    5.5e19
    d1024    2     2048    15276  every 5%    final only    128B    162M    4.0B    2.7e20
    d1536    6     6144    15128  every 5%    final only    381B    481M   11.5B    1.8e21
    d2048   11    11264    20072  every 5%    final only    926B    1.2B   27.7B    9.2e21
    d6144   11    11264   390251  every 3000  every 6k       18T     23B     535B    2.7e24

Train batch is 1024 x racks (constant per-rack load); eval batch is 64 x racks (one sequence per
device). Tokens/steps hold 791 tokens per active parameter (18T at d6144); FLOPs are the levanter
analytic estimate (forward+backward, including attention and the latent-MoE correction).
"""

import dataclasses
import os
from datetime import timedelta

import click
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import (
    data_local_temporary_checkpoint_base_path,
    temporary_checkpoint_base_path,
)
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.checkpointing import RESTORE_BARRIER_TIMEOUT
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    HARRIER_MIX_2026_08_18_STORE,
    HARRIER_MIX_2026_08_18_TAG,
    harrier_mix_2026_08_18_data_config,
)
from experiments.grug.moe_hero_ep.hero_recipe import (
    DEFAULT_WANDB_PROJECT,
    HERO_EP_BATCH_SIZE,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MODEL_CONFIG,
    HERO_NODE_CPU,
    HERO_NODE_DISK,
    HERO_NODE_RAM,
    HERO_PROCESSES_PER_TASK,
    HERO_QB_HIST_BINS,
    HERO_TENSORSTORE_CACHE_BYTES,
    HERO_WATCH_INTERVAL,
    HeroThroughputResult,
    hero_grug_trainer_config,
    hero_trainer_config,
    validation_datasets,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic, build_hero_configs
from experiments.grug.moe_hero_ep.small_scale_abl_launch import (
    _EP_CAPACITY_FACTOR,
    SEQ_LEN,
    SMALL_SHAPES,
    _active_params,
    _small_model,
)
from experiments.grug.moe_hero_ep.train import (
    GrugEvalConfig,
    GrugRunConfig,
    TrainingDataMode,
    WatchMode,
    _compute_flops,
    run_grug,
)
from experiments.marin_tokenizer import marin_tokenizer

# Deadlines for the progress watchdog. A stalled process exits so the scheduler can replace the
# gang rather than leaving every rank blocked on it.
HERO_STEP_TIMEOUT = timedelta(minutes=15)
HERO_PROCESS_STALL_TIMEOUT = timedelta(hours=1)
# Twice the restore barrier, which keeps a barrier expiry ahead of this deadline: the barrier
# names the ranks that never arrived, while this one only reports that nothing progressed.
HERO_STARTUP_TIMEOUT = timedelta(seconds=2 * RESTORE_BARRIER_TIMEOUT)

LADDER_RACKS: dict[str, int] = {"d768": 1, "d1024": 2, "d1536": 6, "d2048": 11, "d6144": 11}
# Each rung uses the rack count that holds its batch. d6144 uses the shared hero recipe. Narrower
# rungs use `_small_model` with the same routing geometry.
# 791 tokens per active parameter sets the step budget: it lands the d6144 hero at 18T tokens and
# scales every narrower rung by the same ratio.
TOKENS_PER_ACTIVE_PARAM = 791
# A crash costs at most this much training time. A hero checkpoint is several TB, thus a shorter
# interval would spend a large part of the run inside a checkpoint write.
RESUME_SAVE_INTERVAL = timedelta(hours=1)
# A rung runs up to 176 tasks for hundreds of GPU-days, where a hardware fault or a host
# out-of-memory on one task is routine. A rung resumes from its newest checkpoint, thus a retry
# continues the run instead of repeating it. Retry deeply so one bad task does not end a rung.
# The two counters are separate gates and the job fails when either one trips.
LADDER_MAX_RETRIES_FAILURE = 1000
LADDER_MAX_TASK_FAILURES = 1000


def _ladder_model(size: str):
    """The GrugModelConfig for ``size`` at the hero routing geometry with the QB histogram estimator."""
    if size == "d6144":
        return HERO_MODEL_CONFIG
    shape = SMALL_SHAPES[size]
    return _small_model(
        shape,
        _EP_CAPACITY_FACTOR,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ragged_all_to_all",
        expert_chunks=1,
        seq_len=SEQ_LEN,
        num_experts=384,
        num_experts_per_token=8,
        intermediate_dim=None,
        latent_dim=None,
        qb_use_histogram=True,
        qb_hist_bins=HERO_QB_HIST_BINS,
    )


def build_ladder_run(
    *,
    run_id: str,
    size: str,
    num_steps: int | None = None,
    checkpoint_every: int | None = None,
    gate_router_weight_decay: float = 0.0,
    source_checkpoint_path: str | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One scaling-ladder rung at width ``size`` on ``LADDER_RACKS[size]`` GB200 racks.

    ``num_steps`` defaults to the steps needed to train ``TOKENS_PER_ACTIVE_PARAM`` tokens per active
    parameter at the rung's (rack-scaled) batch. Every eval scores the held-out set both as-trained
    and dropless. The narrow rungs eval every 5% of the run and keep only the forced final
    checkpoint; the d6144 hero evals every 3000 steps and keeps a permanent checkpoint every 6000.
    ``checkpoint_every`` overrides that cadence for any rung. A rolling temporary checkpoint every
    ``RESUME_SAVE_INTERVAL`` on region-local storage covers a crash or a preemption, and a rung
    resumes from the newest checkpoint it finds.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if size not in LADDER_RACKS:
        raise ValueError(f"size must be one of {sorted(LADDER_RACKS)}, got {size!r}")

    dp_racks = LADDER_RACKS[size]
    # Weak scaling holds per-rack token load constant; eval is one sequence per device.
    batch_size = HERO_EP_BATCH_SIZE * dp_racks
    eval_batch_size = HERO_EP_EXPERT_AXIS_SIZE * dp_racks
    global_tokens_per_step = batch_size * SEQ_LEN

    model = _ladder_model(size)
    if num_steps is None:
        num_steps = max(1, round(TOKENS_PER_ACTIVE_PARAM * _active_params(model) / global_tokens_per_step))
    elif num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    flops_per_example, _ = _compute_flops(model_config=model)
    run_flops = flops_per_example * batch_size * num_steps

    # The narrow rungs are short: eval every 5% of the run and keep only the forced final checkpoint.
    # The d6144 hero is long: eval every 3000 steps and keep a permanent checkpoint every 6000.
    # `keep_permanent=None` still writes the final checkpoint; restore is not used (see run_grug).
    if size == "d6144":
        steps_per_eval = 3000
        keep_permanent: list[dict[str, int]] | None = [{"every": 6000}]
    else:
        steps_per_eval = max(1, round(num_steps / 20))
        keep_permanent = None
    if checkpoint_every is not None:
        keep_permanent = [{"every": checkpoint_every}]

    # The optimizer's LR/epsilon are compute-scaled from the token budget and width; the hero builder
    # already does this at d6144, so reuse it there and the shared MoeHeuristic at the narrow rungs.
    if size == "d6144":
        _, optimizer = build_hero_configs(num_train_steps=num_steps, batch_size=batch_size)
    else:
        optimizer = dataclasses.replace(
            MoeHeuristic().build_optimizer_config(
                num_train_steps=num_steps,
                batch_size=batch_size,
                hidden_dim=model.hidden_dim,
                seq_len=SEQ_LEN,
            ),
            use_syrk=True,  # GB200 SM100 symmetric GEMM for MuonH Newton-Schulz
        )
    optimizer = dataclasses.replace(optimizer, gate_router_weight_decay=gate_router_weight_decay)

    # Uniform hero trainer: expert-parallel within each rack, replicated across racks, MuonH state
    # offloaded to FP32 pinned host.
    grug_trainer = hero_grug_trainer_config(
        replica_axis_size=dp_racks,
        training_data_mode=TrainingDataMode.MIXTURE,
        watch_mode=WatchMode.INLINE,
        save_checkpoints=True,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=HERO_NODE_CPU,
        ram=HERO_NODE_RAM,
        disk=HERO_NODE_DISK,
        replicas=HERO_EP_NODES * dp_racks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    validation = [*validation_datasets(), *uncheatable_datasets(tokenizer=marin_tokenizer).values()]
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT

    def build_config(ctx: StepContext) -> GrugRunConfig:
        permanent_checkpoint_path = prefix_join(ctx.output_path, "checkpoints")
        temporary_checkpoint_path = temporary_checkpoint_base_path(ctx.output_path)
        data_local_checkpoint_path = data_local_temporary_checkpoint_base_path(ctx.output_path)
        trainer = hero_trainer_config(
            run_id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=num_steps,
            profiler=ProfilerConfig(enabled=False),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "hero",
                    "ep",
                    "scaling-ladder",
                    f"shape-{size}",
                    f"racks-{dp_racks}",
                    "gb200",
                    HARRIER_MIX_2026_08_18_TAG,
                ],
                group="moe-hero-ep-scaling-ladder",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=HERO_WATCH_INTERVAL),
            progress_watchdog=ProgressWatchdogConfig(
                step_timeout=HERO_STEP_TIMEOUT,
                process_timeout=HERO_PROCESS_STALL_TIMEOUT,
                startup_timeout=HERO_STARTUP_TIMEOUT,
            ),
            # Existing 02A temporaries remain valid resume candidates for this lineage. A
            # ``source_checkpoint_path`` (a pinned checkpoint from another run) goes last: a cold
            # start finds only it and forks from it, while every retry prefers this run's own newer
            # checkpoint from the paths above.
            load_checkpoint_path=[
                permanent_checkpoint_path,
                temporary_checkpoint_path,
                data_local_checkpoint_path,
                *([source_checkpoint_path] if source_checkpoint_path else []),
            ],
            # load_checkpoint stays None: the trainer resumes from the newest checkpoint that
            # exists, so a retry after a hardware or memory fault continues the run.
            checkpointer=CheckpointerConfig(
                base_path=permanent_checkpoint_path,
                # Rolling resume checkpoints go to region-local temp storage with a lifecycle TTL.
                # The durable output root keeps only the permanent milestones and the final one.
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
            tensorstore_cache_bytes=HERO_TENSORSTORE_CACHE_BYTES,
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=GrugEvalConfig(
                steps_per_eval=steps_per_eval,
                eval_batch_size=eval_batch_size,
                eval_ema=False,
                compute_bpb=True,
                dropless_eval=True,
                # The hero is the run whose full loss curve we report, so give it a baseline point
                # at the start of the curve.
                eval_at_first_step=size == "d6144",
            ),
            stop_after_steps=num_steps,
            processes_per_task=HERO_PROCESSES_PER_TASK,
            max_retries_failure=LADDER_MAX_RETRIES_FAILURE,
            max_task_failures=LADDER_MAX_TASK_FAILURES,
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
@click.option("--size", required=True, type=click.Choice(sorted(LADDER_RACKS)), help="Ladder rung width.")
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Training steps. Default trains 791 tokens per active parameter at the rung's batch.",
)
@click.option(
    "--checkpoint-every",
    type=click.IntRange(min=1),
    default=None,
    help="Keep a permanent checkpoint every N steps on the durable output root. Default follows "
    "the rung (6000 at d6144, final only elsewhere). Resume uses the rolling temporary checkpoint "
    "and is not affected by this option.",
)
@click.option(
    "--gate-router-weight-decay",
    type=click.FloatRange(min=0.0),
    default=0.0,
    show_default=True,
    help="Decoupled weight decay on attn_gate and the router weight, annealed linearly to 0 over "
    "num_train_steps. 0 disables it.",
)
@click.option(
    "--source-checkpoint-path",
    default=None,
    help="Fork from this pinned checkpoint (e.g. another run's step directory) under this run's own "
    "id: a cold start restores it (full state, including step); retries use this run's own checkpoints.",
)
@build_options
def main(
    run_id: str,
    size: str,
    num_steps: int | None,
    checkpoint_every: int | None,
    gate_router_weight_decay: float,
    source_checkpoint_path: str | None,
) -> ArtifactStep[HeroThroughputResult]:
    return build_ladder_run(
        run_id=run_id,
        size=size,
        num_steps=num_steps,
        checkpoint_every=checkpoint_every,
        gate_router_weight_decay=gate_router_weight_decay,
        source_checkpoint_path=source_checkpoint_path,
    )


if __name__ == "__main__":
    main()
