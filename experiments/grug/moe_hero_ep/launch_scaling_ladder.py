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
    d6144   11     2816   390251  every 3000  every 6k       18T     23B     535B    2.7e24

At 4K, train batch is 1024 x racks; longer contexts shrink it to preserve tokens per step (the d6144
hero trained at 4K with batch 11264 until step 257553). Eval batch is one sequence per batch shard:
64 x racks, or (64 / CP) x racks with CP-way context parallelism. Tokens/steps hold 791 tokens per
active parameter (18T at d6144); FLOPs are the levanter analytic estimate (forward+backward, including
attention and the latent-MoE correction), with the d6144 run counted at 4K throughout.

Changelog:
    2026-09-02 (#8818, PR #8833): decoupled weight decay on the attn_gate and router weights is on by
        default (0.02, annealed linearly to 0 over training and read from the Adam step count so it
        resumes at the right step); pass ``--gate-router-weight-decay 0`` to opt out.
        hero-wd-gate-router-p02-step58k forks hero-12d8b6f0-dee637 at step 58014 on the pooled-wave transport
        (see ``trigger_hero.sh``).
    2026-09-09 (#8870): the ragged all-to-all transport with fp32 weights on device returns as the
        default after the pooled-wave fallback of 2026-09-03 (#8884).
        hero-ragged_a2a-ep-step81k forks hero-wd-gate-router-p02-step58k at step 81716 (see ``trigger_hero.sh``).
        hero-ragged_a2a-nccl2307-ep-step81k restarts that fork on the NCCL 2.30.7 PJRT wheel (#9062).
    2026-10-14 (#9615): the hero trains at 16K context with qk_mult 1.48 from step 257553.
        hero-ctx16k-step257k forks hero-mfu30-step231k at permanent step 257553 (see ``trigger_hero.sh``).
"""

import dataclasses
import itertools
import os
from collections.abc import Sequence
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
from marin.training.training import checkpoint_resume_paths, temporary_checkpoint_base_path
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.checkpointing import RESTORE_BARRIER_TIMEOUT
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    HARRIER_MIX_2026_08_18_STORE,
    HARRIER_MIX_2026_08_18_TAG,
    SIMULATED_EPOCHING_MAX_FLOPS,
    HarrierContextPhase,
    harrier_mix_2026_08_18_data_config,
)
from experiments.grug.moe_hero_ep.hero_recipe import (
    DEFAULT_WANDB_PROJECT,
    HERO_BALANCE_ATTENTION_ROWS,
    HERO_CONTEXT_AXIS_SIZE,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_INITIAL_SEQ_LEN,
    HERO_MODEL_CONFIG,
    HERO_NODE_CPU,
    HERO_NODE_DISK,
    HERO_NODE_RAM,
    HERO_PROCESSES_PER_TASK,
    HERO_QB_HIST_BINS,
    HERO_TENSORSTORE_CACHE_BYTES,
    HERO_TOKENS_PER_RACK,
    HERO_WATCH_INTERVAL,
    HeroThroughputResult,
    hero_expert_axis_size,
    hero_grug_trainer_config,
    hero_rack_batch_size,
    hero_trainer_config,
    validate_model_split,
    validated_batch_axis_size,
    validation_datasets,
    with_transport_remat_mode,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.small_scale_abl_launch import (
    _EP_CAPACITY_FACTOR,
    SEQ_LEN,
    SMALL_SHAPES,
    _active_params,
    _small_model,
)
from experiments.grug.moe_hero_ep.train import (
    FlopsBaseline,
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

# The context lengths the d6144 hero trained at before its current one, oldest first, each with the
# step at which it ended. A context-switch deploy PR appends one entry and sets the baseline below.
HERO_CONTEXT_PHASES: tuple[HarrierContextPhase, ...] = (HarrierContextPhase(end_step=257_553, seq_len=4096),)
# The parent's cumulative training FLOPs after HERO_CONTEXT_PHASES[-1].end_step updates.
HERO_FLOPS_BASELINE_TOTAL: float | None = 1.7792877262736668e24


def _hero_flops_baseline(
    context_phases: Sequence[HarrierContextPhase], flops_baseline_total: float | None, seq_len: int
) -> FlopsBaseline | None:
    """Check the hero's context history against ``seq_len`` and return its FLOPs baseline.

    The history must start at 4K and change context length at every entry, including the current one.
    """
    seq_lens = [phase.seq_len for phase in context_phases] + [seq_len]
    if seq_lens[0] != HERO_INITIAL_SEQ_LEN or any(a == b for a, b in itertools.pairwise(seq_lens)):
        raise ValueError(f"The hero's context lengths must start at 4K and change at every phase, got {seq_lens}")
    if not context_phases or flops_baseline_total is None:
        if context_phases or flops_baseline_total is not None:
            raise ValueError("The FLOPs baseline total must be set exactly when the context history is non-empty")
        return None
    return FlopsBaseline(context_phases[-1].end_step, flops_baseline_total)


LADDER_RACKS: dict[str, int] = {"d768": 1, "d1024": 2, "d1536": 6, "d2048": 11, "d6144": 11}
# Each rung uses the rack count that holds its batch. d6144 uses the shared hero recipe. Narrower
# rungs use `_small_model` with the same routing geometry.
# 791 tokens per active parameter sets the step budget: it lands the d6144 hero at 18T tokens and
# scales every narrower rung by the same ratio.
TOKENS_PER_ACTIVE_PARAM = 791
# A crash costs at most this much training time. A hero checkpoint is several TB, thus a shorter
# interval would spend a large part of the run inside a checkpoint write.
RESUME_SAVE_INTERVAL = timedelta(hours=1)
# Rolling resume checkpoints expire this many days after they are written. The live run's newest one
# is at most RESUME_SAVE_INTERVAL old, so the TTL only deletes checkpoints that replaced runs leave
# behind. Each is several TB in a zone with a 100 TiB quota, which the 14-day default let fill
# (#8506, 2026-09-23). A run stalled this long without saving resumes from its newest permanent
# checkpoint instead.
RESUME_CHECKPOINT_TTL_DAYS = 3
HERO_RESUME_CHECKPOINT_PREFIX = "s3://hero-checkpoints"
# A rung runs up to 176 tasks for hundreds of GPU-days, where a hardware fault or a host
# out-of-memory on one task is routine. A rung resumes from its newest checkpoint, thus a retry
# continues the run instead of repeating it. Retry deeply so one bad task does not end a rung.
# The two counters are separate gates and the job fails when either one trips.
LADDER_MAX_RETRIES_FAILURE = 1000
LADDER_MAX_TASK_FAILURES = 1000


def _ladder_model(size: str):
    """The GrugModelConfig for ``size`` at the hero routing geometry with the QB histogram estimator."""
    if size == "d6144":
        # Only the hero rung is measured with the layer-carry offload.
        return with_transport_remat_mode(HERO_MODEL_CONFIG)
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
    qk_mult: float | None = None,
    qk_mult_long_scale: float | None = None,
    num_steps: int | None = None,
    checkpoint_every: int | None = None,
    eval_every: int | None = None,
    gate_router_weight_decay: float = 0.02,
    version: str | None = None,
    initialize_from_checkpoint: str | None = None,
    context_phases: Sequence[HarrierContextPhase] | None = None,
    flops_baseline_total: float | None = None,
    context_axis_size: int | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One scaling-ladder rung at width ``size`` on ``LADDER_RACKS[size]`` GB200 racks.

    ``num_steps`` defaults to the steps needed to train ``TOKENS_PER_ACTIVE_PARAM`` tokens per active
    parameter at the rung's (rack-scaled) batch. Every eval scores the held-out set dropless. The
    narrow rungs eval every 5% of the run and keep only the forced final
    checkpoint; the d6144 hero evals every 3000 steps and keeps a permanent checkpoint every 6000.
    ``checkpoint_every`` and ``eval_every`` override those cadences for any rung. A rolling temporary checkpoint every
    ``RESUME_SAVE_INTERVAL`` on region-local storage covers a crash or a preemption, and a rung
    resumes from the newest checkpoint it finds. ``initialize_from_checkpoint`` is another run's
    checkpoint directory, added as the resume fallback so a relaunch under a new run id continues
    that lineage's full state from exactly that step while writing only to its own tree.

    ``gate_router_weight_decay`` is on by default (see ``GrugMoeMuonHConfig``): the recipe decays the
    attn_gate and router weights, and because the decay reads the Adam step count it also applies at
    the right point when a run resumes an existing checkpoint. Pass 0 to opt out.

    ``qk_mult`` and ``qk_mult_long_scale`` override the rung's model config when set; the hero's own
    values, including its context length and capacity factor, live in ``HERO_MODEL``.

    ``context_phases`` is the run's context history and ``flops_baseline_total`` its cumulative training
    FLOPs at the last switch. Each data bucket resumes at its next shuffle window (see
    ``LmDataConfig.prior_context_phases``). ``None`` selects the checked-in history: ``HERO_CONTEXT_PHASES``
    and ``HERO_FLOPS_BASELINE_TOTAL`` for d6144, and none for the narrow rungs, which train at their context
    length from their start. Pass both to build another history.

    ``context_axis_size`` splits each sequence over that many devices, and each rack's expert axis shrinks to
    ``HERO_EP_EXPERT_AXIS_SIZE // context_axis_size`` to keep the rack full. ``None`` selects the checked-in
    ``HERO_CONTEXT_AXIS_SIZE`` (with ``HERO_BALANCE_ATTENTION_ROWS``) for d6144, and no context parallelism for
    the narrow rungs.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if size not in LADDER_RACKS:
        raise ValueError(f"size must be one of {sorted(LADDER_RACKS)}, got {size!r}")

    hero = size == "d6144"
    if context_phases is None:
        if flops_baseline_total is not None:
            raise ValueError("flops_baseline_total requires an explicit context_phases")
        context_phases = HERO_CONTEXT_PHASES if hero else ()
        flops_baseline_total = HERO_FLOPS_BASELINE_TOTAL if hero else None
    if context_axis_size is None:
        context_axis_size = HERO_CONTEXT_AXIS_SIZE if hero else 1
    balance_attention_rows = HERO_BALANCE_ATTENTION_ROWS if hero else False
    expert_axis_size = hero_expert_axis_size(context_axis_size)

    model = _ladder_model(size)
    if qk_mult is not None:
        model = dataclasses.replace(model, qk_mult=qk_mult)
    if qk_mult_long_scale is not None:
        model = dataclasses.replace(model, qk_mult_long_scale=qk_mult_long_scale)
    seq_len = model.max_seq_len
    flops_baseline = _hero_flops_baseline(context_phases, flops_baseline_total, seq_len)
    validate_model_split(model, context_axis_size=context_axis_size, expert_axis_size=expert_axis_size)

    dp_racks = LADDER_RACKS[size]
    # Weak scaling holds per-rack token load constant.
    global_tokens_per_step = HERO_TOKENS_PER_RACK * dp_racks
    batch_size = hero_rack_batch_size(seq_len) * dp_racks
    batch_axes = validated_batch_axis_size(
        device_count=HERO_EP_EXPERT_AXIS_SIZE * dp_racks,
        dp_racks=dp_racks,
        batch_size=batch_size,
        context_axis_size=context_axis_size,
        expert_axis_size=expert_axis_size,
    )
    # Eval is one sequence per batch shard, so each device scores max_seq_len / context_axis_size tokens per
    # eval batch at every context length. The README's "Scaling ladder" section explains the choice.
    eval_batch_size = batch_axes

    if context_phases and num_steps is not None:
        # The prior phases' data stages are rebuilt from the schedule length, so it must stay the run's own.
        raise ValueError("A run with a context history keeps its schedule; do not pass num_steps")
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
        # Permanent checkpoint every 6000 steps, plus a one-off at step 55000 for the post-handoff
        # weight-decay comparison (#8818). Interval keeps are modular within each `until` range, so
        # the 6000 cadence brackets the pinned 55000 range on both sides.
        keep_permanent: list[dict[str, int | None]] | None = [
            {"until": 54000, "every": 6000},
            {"until": 55000, "every": 55000},
            {"until": None, "every": 6000},
        ]
    else:
        steps_per_eval = max(1, round(num_steps / 20))
        keep_permanent = None
    if checkpoint_every is not None:
        keep_permanent = [{"every": checkpoint_every}]
    if eval_every is not None:
        steps_per_eval = eval_every

    # The optimizer's LR/epsilon are compute-scaled from the tokens per step, token budget, and width.
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_steps,
        batch_size=batch_size,
        hidden_dim=model.hidden_dim,
        seq_len=seq_len,
    )
    optimizer = dataclasses.replace(optimizer, gate_router_weight_decay=gate_router_weight_decay)

    # Uniform hero trainer: expert- and context-parallel within each rack, replicated across racks, MuonH
    # state offloaded to FP32 pinned host.
    grug_trainer = hero_grug_trainer_config(
        replica_axis_size=dp_racks,
        training_data_mode=TrainingDataMode.MIXTURE,
        watch_mode=WatchMode.INLINE,
        save_checkpoints=True,
        context_axis_size=context_axis_size,
        expert_axis_size=expert_axis_size,
        balance_attention_rows=balance_attention_rows,
        gc_interval=100,
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
        temporary_checkpoint_path = temporary_checkpoint_base_path(
            ctx.output_path,
            ttl_days=RESUME_CHECKPOINT_TTL_DAYS,
            temp_prefix=HERO_RESUME_CHECKPOINT_PREFIX if size == "d6144" else None,
        )
        load_checkpoint_path = checkpoint_resume_paths(
            ctx.output_path,
            ttl_days=RESUME_CHECKPOINT_TTL_DAYS,
            temp_prefix=HERO_RESUME_CHECKPOINT_PREFIX if size == "d6144" else None,
        )
        if initialize_from_checkpoint is not None:
            load_checkpoint_path.append(initialize_from_checkpoint)
        trainer = hero_trainer_config(
            run_id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=num_steps,
            profiler=ProfilerConfig(enabled=False),
            tracker=WandbConfig(
                save_code=False,
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
            load_checkpoint_path=load_checkpoint_path,
            # load_checkpoint stays None: the trainer resumes from the newest checkpoint that
            # exists, so a retry after a hardware or memory fault continues the run. Continuing
            # another run requires a checkpoint, so a wrong path fails instead of starting fresh.
            load_checkpoint=True if initialize_from_checkpoint is not None else None,
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
            context_axis_size=context_axis_size,
        )
        return GrugRunConfig(
            model=model,
            flops_baseline=flops_baseline,
            data=harrier_mix_2026_08_18_data_config(
                ctx=ctx,
                total_steps=num_steps,
                batch_size=batch_size,
                max_seq_len=model.max_seq_len,
                enable_simulated_epoching=run_flops <= SIMULATED_EPOCHING_MAX_FLOPS,
                validation=validation,
                prior_context_phases=context_phases,
            ),
            resources=ctx.runtime_arg("train_resources"),
            tensorstore_cache_bytes=HERO_TENSORSTORE_CACHE_BYTES,
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=GrugEvalConfig(
                steps_per_eval=steps_per_eval,
                eval_batch_size=eval_batch_size,
                # The capacity-limited eval breaks the ragged train step at d6144 (#8861). The
                # dropless eval is the reported metric.
                eval_current=False,
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
@click.option(
    "--qk-mult",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Override the query/key attention multiplier on all layers.",
)
@click.option(
    "--qk-mult-long-scale",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Override the extra query/key multiplier on the global (full-causal) layers only.",
)
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
    "--eval-every",
    type=click.IntRange(min=1),
    default=None,
    help="Eval every N steps. Default follows the rung (3000 at d6144, every 5% elsewhere).",
)
@click.option(
    "--gate-router-weight-decay",
    type=click.FloatRange(min=0.0),
    default=0.02,
    show_default=True,
    help="Decoupled weight decay on attn_gate and the router weight, annealed linearly to 0 over "
    "training. Defaults on for the hero recipe; pass 0 to opt out.",
)
@click.option(
    "--initialize-from-checkpoint",
    default=None,
    help="Checkpoint directory of another run to resume from under a new --run-id; this run writes only "
    "to its own tree and later restarts prefer its own, newer checkpoints.",
)
@build_options
def main(
    run_id: str,
    size: str,
    qk_mult: float | None,
    qk_mult_long_scale: float | None,
    num_steps: int | None,
    checkpoint_every: int | None,
    eval_every: int | None,
    gate_router_weight_decay: float,
    initialize_from_checkpoint: str | None,
) -> ArtifactStep[HeroThroughputResult]:
    return build_ladder_run(
        run_id=run_id,
        size=size,
        qk_mult=qk_mult,
        qk_mult_long_scale=qk_mult_long_scale,
        num_steps=num_steps,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        gate_router_weight_decay=gate_router_weight_decay,
        initialize_from_checkpoint=initialize_from_checkpoint,
    )


if __name__ == "__main__":
    main()
