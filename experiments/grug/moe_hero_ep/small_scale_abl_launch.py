# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small-scale hero-shape ablation: d768 / d1024 / d1280 on one EP64 rack each.

Downsized copies of the ``moe_hero_ep`` hero shape (128 experts, top-4, two shared, SConv,
hybrid GQA, ``fixed_all_to_all`` EP MoE) at the three small sweep widths, each trained to its 60x
token budget (60 tokens per active parameter). The architecture, data (datakit two-phase mixture),
evals (paloma + uncheatable every 1k), and per-size step counts match the Aug hero LR sweep grid
(issue #7856) so these one-rack EP runs are comparable to the FSDP sweep points; only the
width/depth/head split and the token budget shrink relative to the d6144 shape.

Each ``--size`` submits one 1-rack (16 x GB200x4 = EP64) job.
"""

import dataclasses
import math

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.datakit.source_key import datakit_source_path
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config, _val_component
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.launch import (
    DEFAULT_WANDB_PROJECT,
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MIXED_PRECISION,
    HERO_PROCESSES_PER_TASK,
    HeroThroughputResult,
)
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.marin_tokenizer import marin_tokenizer

# Shared with the FSDP sweep grid (issue #7856): 8192-token sequences, 512 sliding window with global
# attention every 4th layer + the final layer, and the 60x token budget's step count per size.
SMALL_BATCH_SIZE = 128
SEQ_LEN = 8192
SLIDING_WINDOW = 512
GLOBAL_EVERY = 4
EVAL_BATCH_SIZE = 256

# Paloma + uncheatable held-out sets (marin_tokenizer), added as zero-train-weight datakit components
# so they surface as tagged eval sets -- matching the FSDP sweep.
_VALIDATION = [
    *paloma_datasets(tokenizer=marin_tokenizer).values(),
    *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
]


@dataclasses.dataclass(frozen=True)
class SmallShape:
    hidden_dim: int
    num_layers: int
    num_heads: int
    local_kv_heads: int
    global_kv_heads: int
    num_steps: int  # 60x token budget = round(60 * active_params / (batch * seq_len))


# hidden/depth/head split and the 60x step count come straight from the sweep grid in #7856.
SMALL_SHAPES: dict[str, SmallShape] = {
    "d768": SmallShape(768, 8, 6, 1, 1, num_steps=3105),
    "d1024": SmallShape(1024, 12, 8, 2, 1, num_steps=8325),
    "d1280": SmallShape(1280, 14, 10, 2, 1, num_steps=15019),
}


def _small_model(shape: SmallShape) -> GrugModelConfig:
    """The hero shape (moe_hero_ep ``HERO_MODEL``) downsized to this width.

    Every routing/MoE/attention-kernel field is kept from the d6144 hero shape; only the width,
    depth, head split, and intermediate width shrink. ``fixed_all_to_all`` is the EP MoE backend, so
    ``num_experts`` (128) still divides the EP64 expert axis.
    """
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=shape.hidden_dim,
        intermediate_dim=shape.hidden_dim // 2,
        shared_expert_intermediate_dim=shape.hidden_dim // 2,
        num_shared_experts=2,
        num_experts=128,
        num_experts_per_token=4,
        num_layers=shape.num_layers,
        num_heads=shape.num_heads,
        num_kv_heads=max(shape.local_kv_heads, shape.global_kv_heads),
        local_kv_heads=shape.local_kv_heads,
        global_kv_heads=shape.global_kv_heads,
        head_dim=128,
        max_seq_len=SEQ_LEN,
        sliding_window=SLIDING_WINDOW,
        global_every=GLOBAL_EVERY,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(shape.hidden_dim),
        qk_mult=1.3,
        sconv=True,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="fixed_all_to_all",
        expert_chunks=1,
        report_capacity_overflow=True,
        rope_fused=True,
    )


def _root_component(component: DatasetComponent | ConcatDatasetComponent) -> DatasetComponent | ConcatDatasetComponent:
    """Root a datakit component's relative cache_dir against MARIN_PREFIX (recursing into concat
    children); absolute paths -- e.g. the paloma/uncheatable caches -- pass through unchanged."""
    if isinstance(component, ConcatDatasetComponent):
        return dataclasses.replace(component, children={n: _root_component(c) for n, c in component.children.items()})
    return dataclasses.replace(component, cache_dir=datakit_source_path(component.cache_dir))


def build_small_run(*, run_id: str, size: str, version: str | None = None) -> ArtifactStep[HeroThroughputResult]:
    """One 1-rack EP64 run of the downsized hero shape ``size`` at its 60x token budget."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if size not in SMALL_SHAPES:
        raise ValueError(f"size must be one of {sorted(SMALL_SHAPES)}, got {size!r}")

    shape = SMALL_SHAPES[size]
    model = _small_model(shape)
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=shape.num_steps,
        batch_size=SMALL_BATCH_SIZE,
        hidden_dim=model.hidden_dim,
        seq_len=SEQ_LEN,
    )
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=False,  # small models fit HBM; host offload destabilized small runs
        expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
        replica_axis_size=1,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=HERO_EP_NODES,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=SMALL_BATCH_SIZE,
            num_train_steps=shape.num_steps,
            profiler=ProfilerConfig(enabled=False),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project=DEFAULT_WANDB_PROJECT,
                tags=["grug", "moe", "hero", "ep", "small-abl", f"shape-{size}", "gb200", "MHEP"],
                group="moe-hero-ep-small-abl",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=0),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
        )
        # Datakit two-phase mixture, marin_prefix-rooted (relative bucket paths resolve against the
        # cluster's region-local prefix). Paloma + uncheatable ride in as zero-train-weight components
        # so they surface as tagged eval sets.
        if ctx.is_fingerprint:
            val_components = {v.name: _val_component(ctx.artifact_path(v)) for v in _VALIDATION}
        else:
            val_components = {v.name: ctx.resolved(v).as_component() for v in _VALIDATION}
        data = _datakit_data_config(
            total_steps=shape.num_steps,
            batch_size=SMALL_BATCH_SIZE,
            max_seq_len=SEQ_LEN,
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
            processes_per_task=HERO_PROCESSES_PER_TASK,
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
@build_options
def main(run_id: str, size: str) -> ArtifactStep[HeroThroughputResult]:
    return build_small_run(run_id=run_id, size=size)


if __name__ == "__main__":
    main()
