# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small-scale hero-shape ablation: d768 / d1024 / d1280 / d1536.

Downsized copies of the ``moe_hero_ep`` hero shape (hidden-wide routed experts, top-4, two shared,
SConv, hybrid GQA, ``fixed_all_to_all`` EP MoE) at the small sweep widths, each trained to 750 tokens
per active parameter (the EP sweep budget). The architecture, data (datakit two-phase mixture),
evals (paloma + uncheatable every 1k), and per-size step counts match the Aug hero LR sweep grid
(issue #7856) so these one-rack EP runs are comparable to the FSDP sweep points; only the
width/depth/head split and the token budget shrink relative to the d6144 shape.

Each ``--size`` submits one job on the fleet ``--target`` names: a GB200 rack (EP64), 8 H100 nodes
(EP64), or 2 H100 nodes (EP16). The expert axis spans the fleet, and it sets cell size, so the
target is not just a hardware choice -- see the ``TARGETS`` comments.
"""

import dataclasses
import math
import os
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.datakit.source_key import datakit_source_path
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem import prefix_join

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config, _val_component
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.launch import (
    DEFAULT_WANDB_PROJECT,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MIXED_PRECISION,
    HeroThroughputResult,
)
from experiments.grug.moe_hero_ep.model import GrugModelConfig, QbEstimator
from experiments.grug.moe_hero_ep.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.marin_tokenizer import marin_tokenizer

# The EP sweep (issue #8062) settings: 4096-token sequences at batch 1024, 2048 sliding window with
# global attention every 4th layer plus the final layer, and the 750x token budget per size.
SEQ_LEN = 4096
SMALL_BATCH_SIZE = 1024
# Tokens per step are fixed at ~4M (batch 1024 x seq 4096) to approximate the per-shard token-dropping
# dynamics under the fixed_all_to_all EP MoE; a sequence-length sweep holds this constant and moves only
# the context length.
TOKENS_PER_STEP = SMALL_BATCH_SIZE * SEQ_LEN
SLIDING_WINDOW = 2048
GLOBAL_EVERY = 4
EVAL_BATCH_SIZE = 256
# These runs are hours long, so they checkpoint: the trainer restores from the latest committed
# checkpoint, and an interrupted run would otherwise restart at step 0. A d1280 checkpoint is about
# 38 GB, against 2.7 TiB at the d6144 hero shape.
CHECKPOINT_INTERVAL = timedelta(minutes=30)

# Paloma + uncheatable held-out sets (marin_tokenizer), added as zero-train-weight datakit components
# so they surface as tagged eval sets -- matching the FSDP sweep.
_VALIDATION = [
    *paloma_datasets(tokenizer=marin_tokenizer).values(),
    *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
]


@dataclasses.dataclass(frozen=True)
class Target:
    """Accelerator fleet that one run occupies, and the expert axis it spans."""

    accelerator: str
    gpus_per_node: int
    nodes: int
    cpu: int
    ram: str
    disk: str
    attention_implementation: str
    use_syrk: bool

    @property
    def expert_axis_size(self) -> int:
        return self.gpus_per_node * self.nodes


# These models are small enough to hold a whole rack's worth of experts on one node, so the H100
# target keeps the all-to-all inside a single NVLink domain instead of crossing InfiniBand.
#
# Kernel availability follows the accelerator, in two places. `gpu_fa4_cute` is Blackwell-only: its
# MMA op accepts sm_100/sm_103/sm_110 and rejects H100's sm_90a outright, and `gpu_fa4_thd` needs
# fixed-shape THD segment metadata this model does not supply, so Hopper falls back to reference
# attention. MuonH's `use_syrk` likewise routes the 4D expert-stack Newton-Schulz through QuACK's
# SM100 symmetric GEMM, so Hopper takes the plain vmapped path instead.
TARGETS: dict[str, Target] = {
    "gb200-rack": Target("GB200", HERO_GPUS_PER_NODE, HERO_EP_NODES, 120, "850g", "1t", "gpu_fa4_cute", True),
    # 8 nodes, not 1: capacity is per (sender shard, expert) cell, so the shard count sets how
    # readily cells overflow. EP8 would give 4,096-row cells against 512 at EP64 and would drop far
    # less on the same routing, which is not the behavior these runs are meant to reproduce.
    # 32 CPU and 600g, not 120 and 1900g: an H100 node allocates 127 CPU and about 2 TB, so the
    # larger request demands an effectively empty node and Kueue rejects the whole 8-pod gang
    # ("excluded: resource cpu: 39, resource memory: 25" of 65 nodes). Host memory here holds the
    # loader and checkpoint staging -- a d1280 checkpoint is about 38 GB -- so 600g keeps a wide
    # margin, and the trainer is GPU-bound at these capacity factors.
    "h100-8node": Target("H100", 8, 8, 32, "600g", "900g", "reference", False),
    # 2 nodes = EP16, which reproduces the d6144 hero's per-shard routing statistics exactly.
    # Cell capacity is `cf * (tokens_per_step / shards) * top_k / num_experts`, so the shard count --
    # not the expert count -- is what sets cell size. At EP16 with the grid's 1,048,576 tokens per
    # step, each shard carries the hero's 65,536 tokens and 2,048-row cells. Pair this with
    # `--seq-len 4096` (the batch widens to 256) and each shard also holds the hero's 16 documents,
    # instead of the 2 that EP64 gives. Two documents per shard is why the EP64 ablation drops ~7%
    # at cf 2.5 where the hero drops 0.27%: per-cell load is a sum over correlated document blocks,
    # so the effective sample size is the document count, not the token count.
    "h100-2node": Target("H100", 8, 2, 32, "600g", "900g", "reference", False),
}


@dataclasses.dataclass(frozen=True)
class Flavor:
    """How the MoE layer shards, and what that implies for routing capacity.

    ``ep`` spans the fleet with expert parallelism and drops assignments above each fixed
    (sender shard, expert) cell. The FSDP arms keep one expert axis, so every device holds the whole
    bank and the local `sonic_cute` kernel runs the experts: ``fsdp-nodrop`` at one chunk computes
    every assignment (dropless), and ``fsdp-chunk4`` splits into four chunks to match the FSDP hero's
    minor-dropping reference. Both use the same kernel; only the chunk count (drop rate) differs.
    """

    expert_axis_size: int | None  # None spans the fleet
    moe_implementation: str
    expert_chunks: int


FLAVORS: dict[str, Flavor] = {
    "ep": Flavor(None, "fixed_all_to_all", 1),
    # The dropless FSDP arm is `sonic_cute` at one chunk -- "1 computes every assignment" per the FSDP
    # hero -- so it matches `fsdp-chunk4`'s kernel and only the chunk count (drop rate) differs. The
    # `scatter` grouped-GMM path mis-routes this QB/sigmoid-combine model (loss ~1.1 above chunk4).
    "fsdp-nodrop": Flavor(1, "sonic_cute", 1),
    "fsdp-chunk4": Flavor(1, "sonic_cute", 4),
}


@dataclasses.dataclass(frozen=True)
class SmallShape:
    hidden_dim: int
    num_layers: int
    num_heads: int
    local_kv_heads: int
    global_kv_heads: int


# hidden/depth/head split from the #7856 sweep grid: heads = hidden/128, KV split local = heads//4 and
# global = heads//8 (floored at 1), depth following the 8/12/14/16/22 progression. The step count is
# derived from the active-param count (see `_active_params`), not carried here.
SMALL_SHAPES: dict[str, SmallShape] = {
    "d768": SmallShape(768, 8, 6, 1, 1),
    "d1024": SmallShape(1024, 12, 8, 2, 1),
    "d1280": SmallShape(1280, 14, 10, 2, 1),
    "d1536": SmallShape(1536, 16, 12, 3, 1),
    "d2048": SmallShape(2048, 22, 16, 4, 2),
}


def _small_model(
    shape: SmallShape,
    capacity_factor: float,
    attention_implementation: str,
    moe_implementation: str,
    expert_chunks: int,
    seq_len: int,
    num_experts: int,
    num_experts_per_token: int,
    intermediate_dim: int | None,
    latent_dim: int | None,
    qb_use_histogram: bool = False,
    qb_hist_bins: int = 1000,
) -> GrugModelConfig:
    """The hero shape (moe_hero_ep ``HERO_MODEL``) downsized to this width.

    Every routing/MoE/attention-kernel field is kept from the d6144 hero shape; only the width,
    depth, head split, and intermediate width shrink. ``fixed_all_to_all`` is the EP MoE backend, so
    ``num_experts`` (192, the hero bank) still divides the EP64 expert axis.
    """
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=shape.hidden_dim,
        # Routed experts default hidden-wide, matching the EP hero; only the shared expert is hidden/2.
        intermediate_dim=intermediate_dim if intermediate_dim is not None else shape.hidden_dim,
        shared_expert_intermediate_dim=shape.hidden_dim // 2,
        num_shared_experts=2,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        # Round depth up to even here (not in GrugModelConfig) so global_every scheduling and the
        # last-layer-global rule land cleanly, without rewriting odd-depth configs elsewhere (e.g. HF).
        num_layers=shape.num_layers + shape.num_layers % 2,
        num_heads=shape.num_heads,
        num_kv_heads=max(shape.local_kv_heads, shape.global_kv_heads),
        local_kv_heads=shape.local_kv_heads,
        global_kv_heads=shape.global_kv_heads,
        head_dim=128,
        max_seq_len=seq_len,
        sliding_window=SLIDING_WINDOW,
        global_every=GLOBAL_EVERY,
        capacity_factor=capacity_factor,
        initializer_std=0.5 / math.sqrt(shape.hidden_dim),
        qk_mult=1.3,
        sconv=True,
        attention_implementation=attention_implementation,
        moe_implementation=moe_implementation,
        expert_chunks=expert_chunks,
        # Routed experts run in a latent space half the hidden width, matching the EP hero arm.
        latent_dim=latent_dim if latent_dim is not None else shape.hidden_dim // 2,
        qb_estimator=QbEstimator.HIST if qb_use_histogram else QbEstimator.TOPK,
        qb_hist_bins=qb_hist_bins,
        report_capacity_overflow=True,
        rope_fused=True,
    )


def _active_params(cfg: GrugModelConfig) -> int:
    """Active (non-embedding) parameters per token: attention, router, the top-k routed experts, the
    LatentMoE down/up projections, and the shared experts, summed over layers."""
    d = cfg.hidden_dim
    expert_width = cfg.latent_dim if cfg.latent_dim is not None else d
    per_expert = 3 * expert_width * cfg.intermediate_dim  # gated MLP: gate + up + down
    routed = cfg.num_experts_per_token * per_expert
    latent_proj = 0 if cfg.latent_dim is None else 2 * d * cfg.latent_dim
    shared = cfg.num_shared_experts * 3 * d * cfg.shared_expert_intermediate_dim
    attn = 2 * d * cfg.num_heads * cfg.head_dim + 2 * d * cfg.num_kv_heads * cfg.head_dim  # Q,O and K,V
    router = d * cfg.num_experts
    return cfg.num_layers * (attn + router + routed + latent_proj + shared)


def _root_component(component: DatasetComponent | ConcatDatasetComponent) -> DatasetComponent | ConcatDatasetComponent:
    """Root a datakit component's relative cache_dir against MARIN_PREFIX (recursing into concat
    children); absolute paths -- e.g. the paloma/uncheatable caches -- pass through unchanged."""
    if isinstance(component, ConcatDatasetComponent):
        return dataclasses.replace(component, children={n: _root_component(c) for n, c in component.children.items()})
    return dataclasses.replace(component, cache_dir=datakit_source_path(component.cache_dir))


def build_small_run(
    *,
    run_id: str,
    size: str,
    target: str = "gb200-rack",
    flavor: str = "ep",
    capacity_factor: float = 1.33,
    seq_len: int = SEQ_LEN,
    tokens_per_step: int = TOKENS_PER_STEP,
    num_experts: int = 192,
    num_experts_per_token: int = 4,
    intermediate_dim: int | None = None,
    latent_dim: int | None = None,
    qb_use_histogram: bool = False,
    qb_hist_bins: int = 1000,
    tokens_per_active_param: int = 750,
    num_train_steps_override: int | None = None,
    watch_interval: int = 10,
    dp_racks: int = 1,
    steps_per_eval: int = 1000,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One expert-parallel run of the downsized hero shape ``size``.

    ``tokens_per_active_param`` (default 750) sets the step budget directly: ``num_steps`` is the steps
    needed to train that many tokens per active parameter at the fixed tokens per step, from the model's
    ``_active_params`` count. ``watch_interval`` controls gradient and parameter norm logs.
    The expert overrides let a rung reproduce the hero's routing geometry: cell load is
    ``tokens_per_shard * top-k / experts``, which depends on ``num_experts`` and
    ``num_experts_per_token`` but not on the model width, so a narrow rung can carry the hero's
    exact drop dynamics at a fraction of the step time.

    ``dp_racks`` replicates the run across that many racks (data-parallel over the ``replica`` axis,
    expert-parallel within each rack), which the widest ladder rung needs to hold its batch.
    ``steps_per_eval`` sets both the eval cadence and the permanent-checkpoint cadence, so a
    checkpoint-reload eval job finds a saved state at every eval step.
    """
    if tokens_per_active_param <= 0:
        raise ValueError(f"tokens_per_active_param must be positive, got {tokens_per_active_param}")
    if dp_racks <= 0:
        raise ValueError(f"dp_racks must be positive, got {dp_racks}")
    if steps_per_eval <= 0:
        raise ValueError(f"steps_per_eval must be positive, got {steps_per_eval}")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if size not in SMALL_SHAPES:
        raise ValueError(f"size must be one of {sorted(SMALL_SHAPES)}, got {size!r}")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {sorted(TARGETS)}, got {target!r}")
    if flavor not in FLAVORS:
        raise ValueError(f"flavor must be one of {sorted(FLAVORS)}, got {flavor!r}")
    if tokens_per_step % seq_len != 0:
        raise ValueError(f"seq_len={seq_len} must divide the {tokens_per_step}-token step budget")

    shape = SMALL_SHAPES[size]
    fleet = TARGETS[target]
    sharding = FLAVORS[flavor]
    # `tokens_per_step` is the per-rack (per expert mesh) token load; it stays fixed so the
    # fixed-all-to-all drop dynamics are constant across sizes. The global batch scales with the
    # rack count, so a wider rung on more racks keeps the same per-rack load as a one-rack rung.
    global_tokens_per_step = tokens_per_step * dp_racks
    batch_size = global_tokens_per_step // seq_len
    expert_axis_size = fleet.expert_axis_size if sharding.expert_axis_size is None else sharding.expert_axis_size
    model = _small_model(
        shape,
        capacity_factor,
        fleet.attention_implementation,
        sharding.moe_implementation,
        sharding.expert_chunks,
        seq_len,
        num_experts,
        num_experts_per_token,
        intermediate_dim,
        latent_dim,
        qb_use_histogram,
        qb_hist_bins,
    )
    # Train to `tokens_per_active_param` tokens per active parameter, at the global tokens per step.
    if num_train_steps_override is not None:
        num_steps = num_train_steps_override
    else:
        num_steps = max(1, round(tokens_per_active_param * _active_params(model) / global_tokens_per_step))
    optimizer = dataclasses.replace(
        MoeHeuristic().build_optimizer_config(
            num_train_steps=num_steps,
            batch_size=batch_size,
            hidden_dim=model.hidden_dim,
            seq_len=seq_len,
        ),
        use_syrk=fleet.use_syrk,
    )
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=False,  # small models fit HBM; host offload destabilized small runs
        save_checkpoints=True,
        expert_axis_size=expert_axis_size,
        replica_axis_size=dp_racks,
        sharding_dump_path=None,
    )
    if model.num_experts % expert_axis_size != 0:
        raise ValueError(f"num_experts={model.num_experts} must divide the expert axis {expert_axis_size}")
    train_resources = ResourceConfig.with_gpu(
        fleet.accelerator,
        count=fleet.gpus_per_node,
        cpu=fleet.cpu,
        ram=fleet.ram,
        disk=fleet.disk,
        replicas=fleet.nodes * dp_racks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

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
                project=os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT,
                tags=[
                    "grug",
                    "moe",
                    "hero",
                    "ep",
                    "small-abl",
                    f"shape-{size}",
                    f"capacity-{capacity_factor:g}",
                    f"seq{seq_len}",
                    f"tok{tokens_per_step // 1024}k",
                    f"watch{watch_interval}",
                    flavor,
                    target,
                    "MHEP",
                ],
                group="moe-hero-ep-small-abl",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=watch_interval),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=CheckpointerConfig(
                base_path=prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=CHECKPOINT_INTERVAL,
                keep=[{"every": steps_per_eval}],
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        # Datakit two-phase mixture, marin_prefix-rooted (relative bucket paths resolve against the
        # cluster's region-local prefix). Paloma + uncheatable ride in as zero-train-weight components
        # so they surface as tagged eval sets.
        if ctx.is_fingerprint:
            val_components = {v.name: _val_component(ctx.artifact_path(v)) for v in _VALIDATION}
        else:
            val_components = {v.name: ctx.resolved(v).as_component() for v in _VALIDATION}
        data = _datakit_data_config(
            total_steps=num_steps,
            batch_size=batch_size,
            max_seq_len=seq_len,
            enable_simulated_epoching=False,
            val_components=val_components,
        )
        data = dataclasses.replace(
            data, components={name: _root_component(component) for name, component in data.components.items()}
        )
        return GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=GrugEvalConfig(
                eval_batch_size=EVAL_BATCH_SIZE,
                steps_per_eval=steps_per_eval,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                # EP rungs also log an `eval_dropless` macro loss (dropless local backend on an
                # expert-collapsed mesh); a no-op for the FSDP flavors, which already run dropless.
                dropless_eval=True,
            ),
            # One process per GPU on every target: the H100 nodes hold 8 GPUs, not the
            # GB200 hero's 4, so the count follows the fleet rather than the hero constant.
            processes_per_task=fleet.gpus_per_node,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=tuple(_VALIDATION),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--size",
    type=click.Choice(sorted(SMALL_SHAPES)),
    required=True,
    help="Downsized hero shape to run on one EP64 rack.",
)
@click.option(
    "--target",
    type=click.Choice(sorted(TARGETS)),
    default="gb200-rack",
    show_default=True,
    help="Accelerator fleet for the run. The expert axis spans every GPU it holds.",
)
@click.option(
    "--flavor",
    type=click.Choice(sorted(FLAVORS)),
    default="ep",
    show_default=True,
    help="MoE sharding: expert-parallel, or FSDP with no routing capacity.",
)
@click.option(
    "--seq-len",
    type=click.IntRange(min=1),
    default=SEQ_LEN,
    show_default=True,
    help="Sequence length. The batch widens to hold tokens per step constant.",
)
@click.option(
    "--tokens-per-step",
    type=click.IntRange(min=1),
    default=TOKENS_PER_STEP,
    show_default=True,
    help="Per-rack tokens per step; the global batch scales with --dp-racks. Holds the drop dynamics.",
)
@click.option(
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    # autoresearch: titrating capacity toward the 2% drop budget from the 2.5 baseline
    # (0.63% drops); 2.0 measured 1.32%, and histogram QB pulled it back to 1.25%.
    # Hero arm is 1.33, which runs ~5-6% drops and violates the budget.
    default=1.8,
    show_default=True,
    help="Fixed all-to-all capacity factor (EP hero arm is 1.33). Higher drops fewer and pads more.",
)
@click.option("--num-experts", type=click.IntRange(min=1), default=192, help="Routed expert count (hero bank).")
@click.option("--num-experts-per-token", type=click.IntRange(min=1), default=4, help="Routed experts per token.")
@click.option(
    "--intermediate-dim",
    type=click.IntRange(min=1),
    default=None,
    help="Routed expert MLP width. Defaults to hidden_dim (hidden-wide), matching the EP hero.",
)
@click.option(
    "--latent-dim",
    type=click.IntRange(min=1),
    default=None,
    help="LatentMoE routed-expert width. Defaults to hidden_dim // 2 (the EP hero arm).",
)
@click.option(
    "--qb-histogram/--no-qb-histogram",
    # autoresearch: the #8062 ladder's ep64 arms ran histogram QB; it measured lower
    # drops at neutral loss, which buys capacity-factor headroom under the drop budget.
    default=True,
    show_default=True,
    help="Estimate the QB quantile with the histogram estimator instead of the top-k mean (the hero default).",
)
@click.option(
    "--qb-hist-bins",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="Histogram bin count for the QB quantile estimator.",
)
@click.option(
    "--tokens-per-active-param",
    type=click.IntRange(min=1),
    default=750,
    show_default=True,
    help="Token budget per active parameter, sizing the step count (issue #8062 specifies 750).",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Run exactly this many steps, bypassing --tokens-per-active-param (benchmark runs).",
)
@click.option(
    "--watch-interval",
    type=click.IntRange(min=0),
    default=10,
    show_default=True,
    help="Steps between gradient and parameter norm logs. Zero disables norm logs.",
)
@click.option(
    "--dp-racks",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Replicate the run across this many racks. The batch scales with the fleet, not the rack.",
)
@click.option(
    "--steps-per-eval",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="Eval and permanent-checkpoint cadence in steps.",
)
@build_options
def main(
    run_id: str,
    size: str,
    target: str,
    flavor: str,
    seq_len: int,
    tokens_per_step: int,
    capacity_factor: float,
    num_experts: int,
    num_experts_per_token: int,
    intermediate_dim: int | None,
    latent_dim: int | None,
    qb_histogram: bool,
    qb_hist_bins: int,
    tokens_per_active_param: int,
    num_steps: int | None,
    watch_interval: int,
    dp_racks: int,
    steps_per_eval: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_small_run(
        run_id=run_id,
        size=size,
        target=target,
        flavor=flavor,
        seq_len=seq_len,
        tokens_per_step=tokens_per_step,
        capacity_factor=capacity_factor,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        qb_use_histogram=qb_histogram,
        qb_hist_bins=qb_hist_bins,
        tokens_per_active_param=tokens_per_active_param,
        num_train_steps_override=num_steps,
        watch_interval=watch_interval,
        dp_racks=dp_racks,
        steps_per_eval=steps_per_eval,
    )


if __name__ == "__main__":
    main()
