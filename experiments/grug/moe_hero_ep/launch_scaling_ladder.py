# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hero-shape scaling ladder: one recipe, five widths.

Every rung trains the *same* EP hero recipe -- 384 routed experts, top-8, hidden/2-wide experts in a
hidden/2 latent, pooled-wave transport, SlimPajama/llama3 data, offloaded MuonH state on FP32
pinned-host master params, the QB histogram estimator, and a dropless held-out eval -- and differs
only in width and the rack count it spans. Behaviour is uniform across the ladder so a rung predicts
the d6144 hero. ``d6144`` is the hero itself.

    size   racks  batch  eval          checkpoints  tokens  active  total   FLOPs
    d768     1     1024  every 5%      final only     45B     61M    1.6B    5.2e19
    d1024    2     2048  every 5%      final only    122B    162M    4.0B    2.6e20
    d1536    6     6144  every 5%      final only    361B    481M   11.5B    1.7e21
    d2048   11    11264  every 5%      final only    878B    1.2B   27.7B    8.7e21
    d6144   11    11264  every 3000    every 6k      17.1T    23B     535B    2.6e24

Train batch is 1024 x racks (constant per-rack load); eval batch is 64 x racks (one sequence per
device). Tokens/steps assume the default 750 tokens per active parameter; FLOPs are the levanter
analytic estimate (forward+backward, including attention and the latent-MoE correction).
"""

import dataclasses
import os

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL, MoeHeuristic, build_hero_configs
from experiments.grug.moe_hero_ep.launch_mfu_test import (
    _SLIMPAJAMA_SHUFFLE,
    DEFAULT_WANDB_PROJECT,
    HERO_EP_BATCH_SIZE,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MIXED_PRECISION,
    HeroThroughputResult,
    _slimpajama_6b_dataset,
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
    run_grug,
)

# Ladder rungs, each pinned to the rack count that holds its batch. d6144 is the hero, reusing
# HERO_MODEL; the narrower rungs reuse the ablation's `_small_model` at the same hero routing geometry.
LADDER_RACKS: dict[str, int] = {"d768": 1, "d1024": 2, "d1536": 6, "d2048": 11, "d6144": 11}
QB_HIST_BINS = 10_000
# 750 tokens per active parameter sets the step budget, matching the hero LR-sweep token budget.
TOKENS_PER_ACTIVE_PARAM = 750


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
    num_steps: int | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One scaling-ladder rung at width ``size`` on ``LADDER_RACKS[size]`` GB200 racks.

    ``num_steps`` defaults to the steps needed to train ``TOKENS_PER_ACTIVE_PARAM`` tokens per active
    parameter at the rung's (rack-scaled) batch. Every eval scores the held-out set both as-trained
    and dropless. The narrow rungs eval every 5% of the run and keep only the forced final
    checkpoint; the d6144 hero evals every 3000 steps and keeps a permanent checkpoint every 6000.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if size not in LADDER_RACKS:
        raise ValueError(f"size must be one of {sorted(LADDER_RACKS)}, got {size!r}")

    dp_racks = LADDER_RACKS[size]
    # Weak scaling: batch grows with racks so per-rack token load (and the pooled-wave drop dynamics)
    # stays constant across rungs. Eval batch is one sequence per device (64 per rack).
    batch_size = HERO_EP_BATCH_SIZE * dp_racks
    eval_batch_size = HERO_EP_EXPERT_AXIS_SIZE * dp_racks
    global_tokens_per_step = batch_size * SEQ_LEN

    model = _ladder_model(size)
    if num_steps is None:
        num_steps = max(1, round(TOKENS_PER_ACTIVE_PARAM * _active_params(model) / global_tokens_per_step))
    elif num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    # The narrow rungs are short: eval every 5% of the run and keep only the forced final checkpoint.
    # The d6144 hero is long: eval every 3000 steps and keep a permanent checkpoint every 6000.
    # `keep_permanent=None` still writes the final checkpoint; restore is not used (see run_grug).
    if size == "d6144":
        steps_per_eval = 3000
        keep_permanent: list[dict[str, int]] | None = [{"every": 6000}]
    else:
        steps_per_eval = max(1, round(num_steps / 20))
        keep_permanent = None

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

    # Uniform hero trainer: expert-parallel within each rack, replicated across racks, MuonH state
    # offloaded to FP32 pinned host so the pooled all-to-all buffers keep their HBM.
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=True,
        master_param_mode=MasterParamMode.FP32_PINNED_HOST,
        watch_mode=WatchMode.INLINE,
        save_checkpoints=True,
        expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
        replica_axis_size=dp_racks,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=HERO_EP_NODES * dp_racks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()
    validation = _validation_datasets()
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT

    def build_config(ctx: StepContext) -> GrugRunConfig:
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
                tags=["grug", "moe", "hero", "ep", "scaling-ladder", f"shape-{size}", f"racks-{dp_racks}", "gb200"],
                group="moe-hero-ep-scaling-ladder",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=0),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            # No time-based temporary checkpoints -- an offloaded run cannot restore from one (its
            # pinned-host master/opt state comes back device-kind and mismatches the jitted step), so
            # they would only cost storage. Keep just the permanent step-interval checkpoints below
            # plus the forced final one.
            checkpointer=CheckpointerConfig(
                base_path=prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=None,
                keep=keep_permanent,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        return GrugRunConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, validation=validation, shuffle=_SLIMPAJAMA_SHUFFLE),
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=GrugEvalConfig(
                steps_per_eval=steps_per_eval,
                eval_batch_size=eval_batch_size,
                eval_ema=False,
                compute_bpb=True,
                dropless_eval=True,
            ),
            stop_after_steps=num_steps,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim, *validation),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option("--size", required=True, type=click.Choice(sorted(LADDER_RACKS)), help="Ladder rung width.")
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Training steps. Default trains 750 tokens per active parameter at the rung's batch.",
)
@build_options
def main(run_id: str, size: str, num_steps: int | None) -> ArtifactStep[HeroThroughputResult]:
    return build_ladder_run(run_id=run_id, size=size, num_steps=num_steps)


if __name__ == "__main__":
    main()
