# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parameterized launcher for the Aug/Hero model-size sweep (see ``sweep.md``).

Runs the hero FSDP MoE model at a range of widths on B200/GB200. Each size fixes width, depth, the
local/global KV split, batch, and node count; the attention schedule is shared: a 512-token sliding
window with global attention on every 4th layer plus the final layer (``_long_layer_schedule``).

Selected by env at submit time:
    SWEEP_SIZE        one of d512, d768, d1024, d1280, d1536, d2048  (required)
    SWEEP_TOKEN_MULT  total tokens = this x active params            (default 60)
    SWEEP_LR_MULT     multiplier on the May-Recipe recommended LR    (default 1.0)
    RUN_ID            run name (default aug-hero-{size}-{mult}x-lr{lr})
"""

import dataclasses
import datetime
import math
import os
from dataclasses import dataclass

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
from marin.experiment.cli import experiment_main
from marin.experiment.namespacing import user_namespaced_name

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.launch_datakit_moe_mix import (
    _datakit_data_config,
    _val_component,
)
from experiments.grug.moe_hero_fsdp.heuristic import MoeHeuristic
from experiments.grug.moe_hero_fsdp.launch import HERO_GRUG_TRAINER, HERO_MIXED_PRECISION, HeroThroughputResult
from experiments.grug.moe_hero_fsdp.model import GrugModelConfig
from experiments.grug.moe_hero_fsdp.train import GrugEvalConfig, GrugRunConfig, run_grug
from experiments.marin_tokenizer import marin_tokenizer

VOCAB_SIZE = 128_256
HEAD_DIM = 128
SEQ_LEN = 8192
SLIDING_WINDOW = 512
GLOBAL_EVERY = 4
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOKEN = 4
NUM_SHARED_EXPERTS = 2
GPUS_PER_NODE = 4  # B200/GB200 node unit used for the sweep
EVAL_BATCH_SIZE = 256
CHECKPOINT_INTERVAL = datetime.timedelta(minutes=30)  # temporary (time-policy) checkpoints for resume

# Paloma + uncheatable held-out validation sets (tagged eval), tokenized with the datakit tokenizer.
_VALIDATION = [
    *paloma_datasets(tokenizer=marin_tokenizer).values(),
    *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
]


@dataclass(frozen=True)
class SweepSize:
    name: str
    hidden_dim: int
    num_layers: int
    batch_size: int
    nodes: int  # GPUS_PER_NODE GPUs each


SWEEP_SIZES: dict[str, SweepSize] = {
    s.name: s
    for s in (
        SweepSize("d512", 512, 6, 64, 1),
        SweepSize("d768", 768, 8, 128, 1),
        SweepSize("d1024", 1024, 12, 128, 1),
        SweepSize("d1280", 1280, 14, 128, 2),
        SweepSize("d1536", 1536, 16, 256, 4),
        SweepSize("d2048", 2048, 18, 512, 4),
    )
}


def _kv_heads(hidden_dim: int) -> tuple[int, int]:
    """Heterogeneous GQA that floors the KV ops: local layers keep 1/4 of the query heads, global
    layers 1/8 (floored, min 1). Global always divides local at these widths, so max(local, global)
    is the clean stored count."""
    num_heads = hidden_dim // HEAD_DIM
    return max(1, num_heads // 4), max(1, num_heads // 8)


def _global_layer_count(num_layers: int) -> int:
    """Global layers = every GLOBAL_EVERY-th layer, plus the final layer (matches _long_layer_schedule)."""
    return len({index for index in range(1, num_layers + 1) if index % GLOBAL_EVERY == 0} | {num_layers})


def active_params(size: SweepSize) -> int:
    """Per-token params excluding embedding and lm_head: attention (per-layer KV split), router,
    top-k routed experts, and the shared experts. Used to set the token budget."""
    d, layers = size.hidden_dim, size.num_layers
    intermediate = d // 2
    local_kv, global_kv = _kv_heads(d)
    num_global = _global_layer_count(layers)
    num_local = layers - num_global
    qo = layers * 2 * d * d
    kv = num_local * (2 * d * local_kv * HEAD_DIM) + num_global * (2 * d * global_kv * HEAD_DIM)
    router = layers * d * NUM_EXPERTS
    routed = layers * NUM_EXPERTS_PER_TOKEN * 3 * d * intermediate
    shared = layers * NUM_SHARED_EXPERTS * 3 * d * intermediate
    return qo + kv + router + routed + shared


def build_sweep_configs(size: SweepSize, *, num_train_steps: int, lr_mult: float):
    """The hero model at this sweep width plus its LR-scaled MuonH optimizer."""
    local_kv, global_kv = _kv_heads(size.hidden_dim)
    model = GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=size.hidden_dim,
        intermediate_dim=size.hidden_dim // 2,
        shared_expert_intermediate_dim=size.hidden_dim // 2,
        num_shared_experts=NUM_SHARED_EXPERTS,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        num_layers=size.num_layers,
        num_heads=size.hidden_dim // HEAD_DIM,
        num_kv_heads=max(local_kv, global_kv),
        local_kv_heads=local_kv,
        global_kv_heads=global_kv,
        head_dim=HEAD_DIM,
        max_seq_len=SEQ_LEN,
        sliding_window=SLIDING_WINDOW,
        global_every=GLOBAL_EVERY,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(size.hidden_dim),
        qk_mult=1.3,
        sconv=True,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="sonic_cute",
        expert_chunks=1,
        report_capacity_overflow=True,
        rope_fused=True,
    )
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_train_steps,
        batch_size=size.batch_size,
        hidden_dim=size.hidden_dim,
        seq_len=SEQ_LEN,
    )
    optimizer = dataclasses.replace(
        optimizer,
        learning_rate=optimizer.learning_rate * lr_mult,
        adam_lr=optimizer.adam_lr * lr_mult,
    )
    return model, optimizer


def _root_component(component: DatasetComponent | ConcatDatasetComponent) -> DatasetComponent | ConcatDatasetComponent:
    """Root a datakit component's relative cache_dir against MARIN_PREFIX (recursing into concat
    children). Absolute paths -- e.g. the paloma/uncheatable caches -- pass through unchanged."""
    if isinstance(component, ConcatDatasetComponent):
        return dataclasses.replace(component, children={n: _root_component(c) for n, c in component.children.items()})
    if component.cache_dir is not None:
        return dataclasses.replace(component, cache_dir=datakit_source_path(component.cache_dir))
    return component


def build_sweep_run(*, version: str | None = None) -> ArtifactStep[HeroThroughputResult]:
    """Build one sweep run selected by SWEEP_SIZE / SWEEP_TOKEN_MULT / SWEEP_LR_MULT."""
    size = SWEEP_SIZES[os.environ["SWEEP_SIZE"]]
    token_mult = float(os.environ.get("SWEEP_TOKEN_MULT", "60"))
    lr_mult = float(os.environ.get("SWEEP_LR_MULT", "1.0"))
    steps = round(token_mult * active_params(size) / (size.batch_size * SEQ_LEN))

    run_id = os.environ.get("RUN_ID") or f"aug-hero-{size.name}-{int(token_mult)}x-lr{lr_mult:g}"
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    model, optimizer = build_sweep_configs(size, num_train_steps=steps, lr_mult=lr_mult)

    # GPU fleet is env-overridable so the same launcher can target GB200 (default) or H100 for
    # ablations. SWEEP_GPUS_PER_NODE/SWEEP_NODES let the caller size the mesh (e.g. 8x8 = 64 H100).
    resources = ResourceConfig.with_gpu(
        os.environ.get("SWEEP_GPU_TYPE", "GB200"),
        count=int(os.environ.get("SWEEP_GPUS_PER_NODE", str(GPUS_PER_NODE))),
        cpu=32,
        ram="256g",
        disk="256g",
        replicas=int(os.environ.get("SWEEP_NODES", str(size.nodes))),
    )

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=size.batch_size,
            num_train_steps=steps,
            profiler=ProfilerConfig(enabled=False, start_step=8, num_steps=0),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["grug", "moe", "hero", "sweep", size.name],
                group="aug-hero-sweep",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=20),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=CheckpointerConfig(
                base_path=f"{ctx.output_path}/checkpoints",
                temporary_base_path=f"{ctx.output_path}/checkpoints",
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
            total_steps=steps,
            batch_size=size.batch_size,
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
            # offload_opt_state is a d6144-specific Grace-Blackwell host-offload; disable it for the
            # small sweep models (their optimizer state trivially fits in HBM, and the pinned-host
            # arena + cudaFreeAsync path was destabilizing these runs).
            trainer=dataclasses.replace(HERO_GRUG_TRAINER, trainer=trainer, offload_opt_state=False),
            eval=GrugEvalConfig(
                eval_batch_size=EVAL_BATCH_SIZE,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
            processes_per_task=1,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=tuple(_VALIDATION),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    experiment_main(build_sweep_run)()
