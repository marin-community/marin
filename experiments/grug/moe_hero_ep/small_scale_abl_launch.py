# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small-scale hero-shape ablation from d768 through d2048.

Downsized copies of the ``moe_hero_ep`` hero shape (128 experts, top-4, two shared, SConv,
hybrid GQA, and a selectable MoE backend) at five sweep widths, each trained to its 60x token
budget (60 tokens per active parameter). The architecture, data (datakit two-phase mixture), evals
(paloma + uncheatable every 1k), and per-size step counts match the issue #7856 sweep grid so these
runs are comparable to the FSDP sweep points; only the width/depth/head split and token budget
shrink relative to the d6144 shape.

Each ``--size`` submits one job on the fleet ``--target`` names: one, two, or four GB200 nodes
(EP4/EP8/EP16), a GB200 rack (EP64), 8 H100 nodes (EP64), or 2 H100 nodes (EP16). The expert axis
spans the fleet, and it sets cell size, so the target is not just a hardware choice -- see the
``TARGETS`` comments.
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
from experiments.grug.moe_hero_ep.jax_runtime import jax_nightly_pip_packages
from experiments.grug.moe_hero_ep.launch import (
    DEFAULT_WANDB_PROJECT,
    FLAVORS,
    FOUR_NODE_EP_WORKER_CPU,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MIXED_PRECISION,
    HERO_WORKER_CPU,
    HERO_WORKER_RAM,
    HeroThroughputResult,
)
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.marin_tokenizer import marin_tokenizer

# Shared with the FSDP sweep grid (issue #7856): 8192-token sequences, 512 sliding window with global
# attention every 4th layer + the final layer, and the 60x token budget's step count per size.
SEQ_LEN = 8192
SMALL_BATCH_SIZE = 128
# Sequence-length sweeps hold tokens per step constant, so the token budget and the per-shard
# routing capacity stay fixed and only the context length moves.
TOKENS_PER_STEP = SMALL_BATCH_SIZE * SEQ_LEN
SLIDING_WINDOW = 512
GLOBAL_EVERY = 4
EVAL_BATCH_SIZE = 256
BASELINE_TOKENS_PER_ACTIVE_PARAM = 60
GB200_SCREEN_WORKER_RAM = "128g"
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
    # Correctness probe for process-per-GPU routing. Pair 262,144 tokens/step with 12 experts to
    # preserve the EP16 E48 proxy's 65,536 tokens and three experts per rank while using one node.
    "gb200-1node": Target(
        "GB200",
        HERO_GPUS_PER_NODE,
        1,
        FOUR_NODE_EP_WORKER_CPU,
        GB200_SCREEN_WORKER_RAM,
        "1t",
        "gpu_fa4_cute",
        True,
    ),
    # Cross-host correctness probe small enough to fit domains that cannot admit an EP16 gang.
    # Pair 524,288 tokens/step with 24 experts for the same per-rank geometry as the EP4/E12 and
    # EP16/E48 probes.
    "gb200-2node": Target(
        "GB200",
        HERO_GPUS_PER_NODE,
        2,
        FOUR_NODE_EP_WORKER_CPU,
        GB200_SCREEN_WORKER_RAM,
        "1t",
        "gpu_fa4_cute",
        True,
    ),
    # EP16 keeps cross-node P2P in the experiment while fitting capacity gaps too small for a rack.
    # At 1,048,576 tokens per step it has the d6144 hero's 65,536 tokens per sender shard, so E192
    # top-4 runs reproduce the hero's routing-cell load. It is a relative transport-knob screen:
    # 15 peers per rank cannot reproduce EP64's 63-peer FIFO pressure.
    "gb200-4node": Target(
        "GB200",
        HERO_GPUS_PER_NODE,
        4,
        FOUR_NODE_EP_WORKER_CPU,
        GB200_SCREEN_WORKER_RAM,
        "1t",
        "gpu_fa4_cute",
        True,
    ),
    "gb200-rack": Target(
        "GB200", HERO_GPUS_PER_NODE, HERO_EP_NODES, HERO_WORKER_CPU, HERO_WORKER_RAM, "1t", "gpu_fa4_cute", True
    ),
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
class SmallShape:
    hidden_dim: int
    num_layers: int
    num_heads: int
    local_kv_heads: int
    global_kv_heads: int
    num_steps: int  # 60x token budget = round(60 * active_params / (batch * seq_len))


# Hidden/depth/head split and the 60x step count come from the sweep grid in
# https://github.com/marin-community/marin/issues/7856.
SMALL_SHAPES: dict[str, SmallShape] = {
    "d768": SmallShape(768, 8, 6, 1, 1, num_steps=3105),
    "d1024": SmallShape(1024, 12, 8, 2, 1, num_steps=8325),
    "d1280": SmallShape(1280, 14, 10, 2, 1, num_steps=15019),
    # Extrapolated rung, not from #7856: head count stays hidden/128, depth continues 8/12/14 -> 16,
    # and the KV split follows the rule the other three obey (local = heads//4, global = heads//8,
    # floored at 1), which also reproduces the d6144 hero's 12/6 on 48 heads. The 60x step count is
    # not a free choice -- active parameters scale as layers x hidden^2, and that model reproduces
    # the three measured step counts to within 0.5%, giving 24,840 here.
    "d1536": SmallShape(1536, 16, 12, 3, 1, num_steps=24840),
    # Issue #8062's fourth rung. Heads stay hidden/128, the KV split follows local = heads//4 and
    # global = heads//8, and depth continues the 8/12/14/16 progression to 22. The 60x step count
    # comes from the same layers x hidden^2 active-parameter model that reproduces the other four.
    "d2048": SmallShape(2048, 22, 16, 4, 2, num_steps=60640),
}


def _small_model(
    shape: SmallShape,
    capacity_factor: float,
    attention_implementation: str,
    moe_implementation: str,
    seq_len: int,
    num_experts: int,
    num_experts_per_token: int,
    intermediate_dim: int | None,
    latent_dim: int | None,
    ragged_all_to_all_splits_per_peer: int = 1,
) -> GrugModelConfig:
    """The hero shape (moe_hero_ep ``HERO_MODEL``) downsized to this width.

    Every routing and attention-kernel field is kept from the d6144 hero shape; only the width,
    depth, head split, intermediate width, expert count, and caller-selected MoE backend can change.
    """
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=shape.hidden_dim,
        intermediate_dim=intermediate_dim if intermediate_dim is not None else shape.hidden_dim // 2,
        shared_expert_intermediate_dim=shape.hidden_dim // 2,
        num_shared_experts=2,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        num_layers=shape.num_layers,
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
        latent_dim=latent_dim,
        ragged_all_to_all_splits_per_peer=ragged_all_to_all_splits_per_peer,
        report_capacity_overflow=True,
        rope_fused=True,
    )


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
    capacity_factor: float = 1.0,
    seq_len: int = SEQ_LEN,
    tokens_per_step: int = TOKENS_PER_STEP,
    num_experts: int = 128,
    num_experts_per_token: int = 4,
    intermediate_dim: int | None = None,
    latent_dim: int | None = None,
    ragged_all_to_all_splits_per_peer: int = 1,
    tokens_per_active_param: int = BASELINE_TOKENS_PER_ACTIVE_PARAM,
    watch_interval: int = 0,
    jax_nightly_version: str | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One expert-parallel run of the downsized hero shape ``size``.

    ``tokens_per_active_param`` scales the step budget. The shapes carry a 60x count; issue #8062
    specifies 750x for the EP/FSDP ladder, which is what makes these rungs comparable to the hero.
    ``watch_interval`` controls gradient and parameter norm logs. Zero disables these logs.
    The expert overrides let a rung reproduce the hero's routing geometry: cell load is
    ``tokens_per_shard * top-k / experts``, which depends on ``num_experts`` and
    ``num_experts_per_token`` but not on the model width, so a narrow rung can carry the hero's
    exact drop dynamics at a fraction of the step time.
    """
    if tokens_per_active_param <= 0:
        raise ValueError(f"tokens_per_active_param must be positive, got {tokens_per_active_param}")
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
    # Tokens per step stay fixed, so a shorter context trains on the same data with a wider batch.
    batch_size = tokens_per_step // seq_len
    # The 60x token budget is what the step count encodes, so a wider step needs proportionally
    # fewer of them. A wider step also deepens each routing cell, which is what sets the drop rate:
    # capacity is ceil(factor * tokens_per_shard * top-k / experts).
    num_steps = max(
        1,
        round(
            shape.num_steps
            * TOKENS_PER_STEP
            / tokens_per_step
            * tokens_per_active_param
            / BASELINE_TOKENS_PER_ACTIVE_PARAM
        ),
    )
    expert_axis_size = fleet.expert_axis_size if sharding.expert_axis_size is None else sharding.expert_axis_size
    model = _small_model(
        shape,
        capacity_factor,
        fleet.attention_implementation,
        sharding.moe_implementation,
        seq_len,
        num_experts,
        num_experts_per_token,
        intermediate_dim,
        latent_dim,
        ragged_all_to_all_splits_per_peer,
    )
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
        replica_axis_size=1,
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
        replicas=fleet.nodes,
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
                    f"ragged-splits-{ragged_all_to_all_splits_per_peer}",
                    f"jax-{jax_nightly_version or 'stable'}",
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
                keep=None,
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
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
            processes_per_task=fleet.gpus_per_node,
            worker_pip_packages=jax_nightly_pip_packages(jax_nightly_version),
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
    help="Downsized hero shape to run on the selected accelerator fleet.",
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
    help="Tokens per optimizer step. Widens the batch and shortens the run to hold the token budget.",
)
@click.option(
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
    help="Fixed all-to-all capacity factor. Higher values drop fewer assignments and pad more.",
)
@click.option("--num-experts", type=click.IntRange(min=1), default=128, help="Routed expert count.")
@click.option("--num-experts-per-token", type=click.IntRange(min=1), default=4, help="Routed experts per token.")
@click.option(
    "--intermediate-dim",
    type=click.IntRange(min=1),
    default=None,
    help="Expert width. Defaults to hidden_dim // 2.",
)
@click.option(
    "--latent-dim",
    type=click.IntRange(min=1),
    default=None,
    help="LatentMoE: run routed experts at this width. Divides all-to-all traffic by hidden/latent.",
)
@click.option(
    "--ragged-all-to-all-splits-per-peer",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Split each peer transfer into this many ragged updates so large slices use more GPU blocks.",
)
@click.option(
    "--tokens-per-active-param",
    type=click.IntRange(min=1),
    default=BASELINE_TOKENS_PER_ACTIVE_PARAM,
    help="Token budget per active parameter. The shapes carry 60; issue #8062 specifies 750.",
)
@click.option(
    "--watch-interval",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Steps between gradient and parameter norm logs. Zero disables norm logs.",
)
@click.option(
    "--jax-nightly-version",
    default=None,
    help="Install this exact JAX nightly on workers after the locked GPU environment sync.",
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
    ragged_all_to_all_splits_per_peer: int,
    tokens_per_active_param: int,
    watch_interval: int,
    jax_nightly_version: str | None,
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
        ragged_all_to_all_splits_per_peer=ragged_all_to_all_splits_per_peer,
        tokens_per_active_param=tokens_per_active_param,
        watch_interval=watch_interval,
        jax_nightly_version=jax_nightly_version,
    )


if __name__ == "__main__":
    main()
