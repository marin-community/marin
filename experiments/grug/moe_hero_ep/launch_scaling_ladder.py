# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hero-shape scaling ladders on GB200 and H100.

Every rung trains the *same* EP hero recipe -- 384 routed experts, top-8, hidden/2-wide experts in a
hidden/2 latent, pooled-wave transport, the Harrier 2026.08.18 two-phase mixture on the Marin
tokenizer, offloaded MuonH state on FP32 pinned-host master params, the QB histogram estimator, and
a dropless held-out eval -- and differs only in width and the rack count it spans. Behaviour is
uniform across the ladder so a rung predicts the d6144 hero. ``d6144`` is the hero itself.

    size   racks  batch    steps  eval        checkpoints  tokens  active  total   FLOPs
    d768     1     1024    11420  every 5%    final only     48B     61M    1.6B    5.5e19
    d1024    2     2048    15276  every 5%    final only    128B    162M    4.0B    2.7e20
    d1536    6     6144    15128  every 5%    final only    381B    481M   11.5B    1.8e21
    d2048   11    11264    20072  every 5%    final only    926B    1.2B   27.7B    9.2e21
    d6144   11    11264   390251  every 3000  every 6k       18T     23B     535B    2.7e24

Train batch is 1024 x racks (constant per-rack load); eval batch is 64 x racks (one sequence per
device). Tokens/steps hold 791 tokens per active parameter (18T at d6144); FLOPs are the levanter
analytic estimate (forward+backward, including attention and the latent-MoE correction).

The H100 ladder extends the same recipe downward. Its global token batch is the largest power-of-two
sequence batch no greater than ``training_tokens ** 0.6``. d384 uses EP4 because EP8 underfills the
devices at this width, d512 uses EP8, and d768 uses two data-parallel EP8 replicas to stay
comfortably below a 12-hour wall-time target. The hardware mapping changes only the expert/replica
topology, accelerator kernels, and per-device eval batch:

    size   H100 GPUs  EP per task  global batch  steps  tokens
    d384       4           4            128       12162   6.4B
    d512       8           8            256       15721    16B
    d768      16           8            512       22840    48B
"""

import dataclasses
import os
from datetime import timedelta
from enum import StrEnum

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.grug.grug_moe import MoeImplementation
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
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
from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL, MoeHeuristic, build_hero_configs
from experiments.grug.moe_hero_ep.launch_mfu_test import (
    DEFAULT_WANDB_PROJECT,
    HERO_EP_BATCH_SIZE,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MIXED_PRECISION,
    HeroThroughputResult,
    _validation_datasets,
)
from experiments.grug.moe_hero_ep.model import QbEstimator
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
    GrugTrainerConfig,
    MasterParamMode,
    WatchMode,
    _compute_flops,
    run_grug,
)
from experiments.marin_tokenizer import marin_tokenizer

# Ladder rungs, each pinned to the rack count that holds its batch. d6144 is the hero, reusing
# HERO_MODEL; the narrower rungs reuse the ablation's `_small_model` at the same hero routing geometry.
# Deadlines for the progress watchdog. A stalled process exits so the scheduler can replace the
# gang rather than leaving every rank blocked on it.
HERO_STEP_TIMEOUT = timedelta(minutes=15)
HERO_PROCESS_STALL_TIMEOUT = timedelta(hours=1)
# Twice the restore barrier, which keeps a barrier expiry ahead of this deadline: the barrier
# names the ranks that never arrived, while this one only reports that nothing progressed.
HERO_STARTUP_TIMEOUT = timedelta(seconds=2 * RESTORE_BARRIER_TIMEOUT)

LADDER_RACKS: dict[str, int] = {"d768": 1, "d1024": 2, "d1536": 6, "d2048": 11, "d6144": 11}
H100_LADDER_NODES: dict[str, int] = {"d384": 1, "d512": 1, "d768": 2}
H100_LADDER_GPUS_PER_TASK: dict[str, int] = {"d384": 4, "d512": 8, "d768": 8}
QB_HIST_BINS = 10_000
# Gradient and parameter norm logs every 10 steps on every rung.
WATCH_INTERVAL = 10
# 791 tokens per active parameter sets the step budget: it lands the d6144 hero at 18T tokens and
# scales every narrower rung by the same ratio.
TOKENS_PER_ACTIVE_PARAM = 791
# The decoded-chunk cache is process-local. A two-tray loader benchmark at the hero's global batch
# and sequence length found 1 GB retained 18.6x throughput headroom while bounding the cache near
# 0.923 GiB per process; 125 GB allowed native RSS to grow until the cache filled.
TENSORSTORE_CACHE_BYTES = 1_000_000_000
# A crash costs at most this much training time. A hero checkpoint is several TB, thus a shorter
# interval would spend a large part of the run inside a checkpoint write.
RESUME_SAVE_INTERVAL = timedelta(hours=1)
# A rung runs up to 176 tasks for hundreds of GPU-days, where a hardware fault or a host
# out-of-memory on one task is routine. A rung resumes from its newest checkpoint, thus a retry
# continues the run instead of repeating it. Retry deeply so one bad task does not end a rung.
# The two counters are separate gates and the job fails when either one trips.
LADDER_MAX_RETRIES_FAILURE = 1000
LADDER_MAX_TASK_FAILURES = 1000
# H100 rungs are short diagnostics without the hero's multi-day exposure to routine hardware faults.
# Keep retries bounded so a deterministic numerical failure cannot turn into a persistent restart loop.
H100_MAX_RETRIES_FAILURE = 3
H100_MAX_TASK_FAILURES = 3


class LadderTarget(StrEnum):
    GB200_RACK = "gb200-rack"
    H100 = "h100"


@dataclasses.dataclass(frozen=True)
class LadderExecution:
    accelerator: str
    gpus_per_task: int
    task_count: int
    cpu: int
    ram: str
    disk: str
    expert_axis_size: int
    replica_axis_size: int
    eval_batch_size: int
    dropless_eval_moe_implementation: MoeImplementation
    max_retries_failure: int
    max_task_failures: int
    tags: tuple[str, ...]


def _ladder_execution(size: str, target: LadderTarget) -> LadderExecution:
    if target == LadderTarget.GB200_RACK:
        if size not in LADDER_RACKS:
            raise ValueError(f"size must be one of {sorted(LADDER_RACKS)} for {target}, got {size!r}")
        dp_racks = LADDER_RACKS[size]
        return LadderExecution(
            accelerator="GB200",
            gpus_per_task=HERO_GPUS_PER_NODE,
            task_count=HERO_EP_NODES * dp_racks,
            cpu=120,
            ram="890g",
            disk="1t",
            expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
            replica_axis_size=dp_racks,
            eval_batch_size=HERO_EP_EXPERT_AXIS_SIZE * dp_racks,
            dropless_eval_moe_implementation="sonic_cute",
            max_retries_failure=LADDER_MAX_RETRIES_FAILURE,
            max_task_failures=LADDER_MAX_TASK_FAILURES,
            tags=(f"racks-{dp_racks}", "gb200"),
        )

    if size not in H100_LADDER_NODES:
        raise ValueError(f"size must be one of {sorted(H100_LADDER_NODES)} for {target}, got {size!r}")
    nodes = H100_LADDER_NODES[size]
    gpus_per_task = H100_LADDER_GPUS_PER_TASK[size]
    return LadderExecution(
        accelerator="H100",
        gpus_per_task=gpus_per_task,
        task_count=nodes,
        cpu=32,
        ram="600g",
        disk="900g",
        expert_axis_size=gpus_per_task,
        replica_axis_size=nodes,
        eval_batch_size=gpus_per_task * nodes,
        dropless_eval_moe_implementation="sonic",
        max_retries_failure=H100_MAX_RETRIES_FAILURE,
        max_task_failures=H100_MAX_TASK_FAILURES,
        tags=(f"h100-nodes-{nodes}", "h100", f"ep{gpus_per_task}"),
    )


def _h100_ladder_batch_size(training_tokens: int, global_device_count: int) -> int:
    """Largest power-of-two sequence batch within the token-budget scaling ceiling."""
    max_sequences = int(training_tokens**0.6) // SEQ_LEN
    if max_sequences < global_device_count:
        raise ValueError(f"token batch ceiling permits {max_sequences} sequences for {global_device_count} devices")
    batch_size = 1 << (max_sequences.bit_length() - 1)
    assert batch_size % global_device_count == 0
    assert batch_size * SEQ_LEN <= training_tokens**0.6
    return batch_size


def _ladder_model(size: str):
    """The GrugModelConfig for ``size`` at the hero routing geometry with the QB histogram estimator."""
    if size == "d6144":
        return dataclasses.replace(HERO_MODEL, qb_estimator=QbEstimator.HIST, qb_hist_bins=QB_HIST_BINS)
    shape = SMALL_SHAPES[size]
    return _small_model(
        shape,
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


def build_ladder_run(
    *,
    run_id: str,
    size: str,
    target: LadderTarget = LadderTarget.GB200_RACK,
    num_steps: int | None = None,
    batch_size: int | None = None,
    checkpoint_every: int | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One scaling-ladder rung at width ``size`` on the selected hardware target.

    ``num_steps`` defaults to the steps needed to train ``TOKENS_PER_ACTIVE_PARAM`` tokens per active
    parameter at the rung's batch. ``batch_size`` overrides the hardware target's canonical batch
    for a one-off comparison. Every eval scores the held-out set both as-trained and dropless. The
    narrow rungs eval every 5% of the run and keep only the forced final checkpoint; the d6144 hero
    evals every 3000 steps and keeps a permanent checkpoint every 6000. ``checkpoint_every``
    overrides that cadence for any rung. A rolling temporary checkpoint every
    ``RESUME_SAVE_INTERVAL`` on region-local storage covers a crash or a preemption, and a rung
    resumes from the newest checkpoint it finds.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    execution = _ladder_execution(size, target)
    model = _ladder_model(size)
    training_tokens = TOKENS_PER_ACTIVE_PARAM * _active_params(model)
    global_device_count = execution.gpus_per_task * execution.task_count
    if batch_size is None and target == LadderTarget.H100:
        batch_size = _h100_ladder_batch_size(
            training_tokens,
            global_device_count=global_device_count,
        )
    elif batch_size is None:
        batch_size = HERO_EP_BATCH_SIZE * execution.replica_axis_size
    elif batch_size <= 0 or batch_size % global_device_count != 0:
        raise ValueError(f"batch_size must be positive and divisible by {global_device_count}, got {batch_size}")
    eval_batch_size = execution.eval_batch_size
    global_tokens_per_step = batch_size * SEQ_LEN

    if num_steps is None:
        num_steps = max(1, round(training_tokens / global_tokens_per_step))
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
            use_syrk=True,  # SM90/SM100 symmetric GEMM for MuonH Newton-Schulz
        )

    # Uniform hero trainer: expert-parallel within each topology group, replicated across groups,
    # with MuonH state offloaded to FP32 pinned host so pooled all-to-all buffers retain HBM.
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=True,
        master_param_mode=MasterParamMode.FP32_PINNED_HOST,
        watch_mode=WatchMode.INLINE,
        save_checkpoints=True,
        expert_axis_size=execution.expert_axis_size,
        replica_axis_size=execution.replica_axis_size,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        execution.accelerator,
        count=execution.gpus_per_task,
        cpu=execution.cpu,
        ram=execution.ram,
        disk=execution.disk,
        replicas=execution.task_count,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    validation = [*_validation_datasets(), *uncheatable_datasets(tokenizer=marin_tokenizer).values()]
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT

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
                    f"shape-{size}",
                    *execution.tags,
                    HARRIER_MIX_2026_08_18_TAG,
                ],
                group="moe-hero-ep-scaling-ladder",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=WATCH_INTERVAL),
            progress_watchdog=ProgressWatchdogConfig(
                step_timeout=HERO_STEP_TIMEOUT,
                process_timeout=HERO_PROCESS_STALL_TIMEOUT,
                startup_timeout=HERO_STARTUP_TIMEOUT,
            ),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            # Existing 02A temporaries remain valid resume candidates for this lineage.
            load_checkpoint_path=[
                permanent_checkpoint_path,
                temporary_checkpoint_path,
                data_local_checkpoint_path,
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
            tensorstore_cache_bytes=TENSORSTORE_CACHE_BYTES,
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=GrugEvalConfig(
                steps_per_eval=steps_per_eval,
                eval_batch_size=eval_batch_size,
                eval_ema=False,
                compute_bpb=True,
                dropless_eval=True,
                dropless_eval_moe_implementation=execution.dropless_eval_moe_implementation,
                # The hero is the run whose full loss curve we report, so give it a baseline point
                # at the start of the curve.
                eval_at_first_step=size == "d6144",
            ),
            stop_after_steps=num_steps,
            processes_per_task=execution.gpus_per_task,
            max_retries_failure=execution.max_retries_failure,
            max_task_failures=execution.max_task_failures,
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
    "--size",
    required=True,
    type=click.Choice(sorted(set(LADDER_RACKS) | set(H100_LADDER_NODES))),
    help="Ladder rung width; availability depends on --target.",
)
@click.option(
    "--target",
    type=click.Choice([target.value for target in LadderTarget]),
    default=LadderTarget.GB200_RACK.value,
    show_default=True,
    help="Canonical GB200 racks or the H100 EP4/EP8 ladder mapping.",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Training steps. Default trains 791 tokens per active parameter at the rung's batch.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Global sequence batch override. Default follows the selected hardware target.",
)
@click.option(
    "--checkpoint-every",
    type=click.IntRange(min=1),
    default=None,
    help="Keep a permanent checkpoint every N steps on the durable output root. Default follows "
    "the rung (6000 at d6144, final only elsewhere). Resume uses the rolling temporary checkpoint "
    "and is not affected by this option.",
)
@build_options
def main(
    run_id: str,
    size: str,
    target: str,
    num_steps: int | None,
    batch_size: int | None,
    checkpoint_every: int | None,
) -> ArtifactStep[HeroThroughputResult]:
    return build_ladder_run(
        run_id=run_id,
        size=size,
        target=LadderTarget(target),
        num_steps=num_steps,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
    )


if __name__ == "__main__":
    main()
