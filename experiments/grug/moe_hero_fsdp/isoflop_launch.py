# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aug hero iso-FLOP sweep on H100 (issue #8003).

Six model widths (d512-d2048) x six compute budgets (1e18-3e20 FLOPs, EXCLUDING embed/lm_head), so
the loss-vs-width curve at a fixed FLOP count reveals the compute-optimal width. Same architecture as
the Aug hero LR sweep (128 experts / top-4 / 2 shared, SConv, hybrid GQA, seq 8192, global-every-4,
sliding-window 512) with the refit LR heuristic (heuristic.py). Tokens = budget / flops_per_token_excl;
batch is tied to model width; node count scales with the budget for wall-time. datakit two-phase data +
paloma/uncheatable evals every 1k, matching the sweep. H100 recipe: ring MoE, use_syrk off, PGLE off.

Each ``--budget --size`` submits one job. Only the sensible iso-FLOP band is defined (24 cells).
"""

import dataclasses
import datetime
import math

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
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config, _val_component
from experiments.grug.moe_hero_fsdp.heuristic import MoeHeuristic
from experiments.grug.moe_hero_fsdp.model import GrugModelConfig
from experiments.grug.moe_hero_fsdp.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.marin_tokenizer import marin_tokenizer

VOCAB_SIZE = 128_256
HEAD_DIM = 128
SEQ_LEN = 8192
SLIDING_WINDOW = 512
GLOBAL_EVERY = 4
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOKEN = 4
NUM_SHARED_EXPERTS = 2
GPUS_PER_NODE = 8  # H100 node
EVAL_BATCH_SIZE = 256
CHECKPOINT_INTERVAL = datetime.timedelta(minutes=30)

# Paloma + uncheatable held-out sets (marin_tokenizer), added as zero-train-weight datakit components.
_VALIDATION = [
    *paloma_datasets(tokenizer=marin_tokenizer).values(),
    *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
]


@dataclasses.dataclass(frozen=True)
class IsoflopSize:
    hidden_dim: int
    num_layers: int
    num_heads: int
    local_kv_heads: int
    global_kv_heads: int


# hidden/depth/head split matches the LR sweep grid (#7856).
ISOFLOP_SIZES: dict[str, IsoflopSize] = {
    "d512": IsoflopSize(512, 6, 4, 1, 1),
    "d768": IsoflopSize(768, 8, 6, 1, 1),
    "d1024": IsoflopSize(1024, 12, 8, 2, 1),
    "d1280": IsoflopSize(1280, 14, 10, 2, 1),
    "d1536": IsoflopSize(1536, 16, 12, 3, 1),
    "d2048": IsoflopSize(2048, 18, 16, 4, 2),
}

BUDGETS: dict[str, float] = {"1e18": 1e18, "3e18": 3e18, "1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20}

# Iso-FLOP band: per-cell batch (tied to width, with the low-/high-step adjustments) and per-budget
# node count. Only the ~24 sensible cells are defined; degenerate under/over-trained corners are omitted.
NODES_BY_BUDGET: dict[str, int] = {"1e18": 1, "3e18": 1, "1e19": 1, "3e19": 2, "1e20": 4, "3e20": 8}
GRID_BATCH: dict[tuple[str, str], int] = {
    ("1e18", "d512"): 32,
    ("1e18", "d768"): 32,
    ("1e18", "d1024"): 32,
    ("3e18", "d512"): 64,
    ("3e18", "d768"): 32,
    ("3e18", "d1024"): 32,
    ("3e18", "d1280"): 32,
    ("1e19", "d512"): 64,
    ("1e19", "d768"): 128,
    ("1e19", "d1024"): 128,
    ("1e19", "d1280"): 64,
    ("1e19", "d1536"): 128,
    ("3e19", "d768"): 128,
    ("3e19", "d1024"): 128,
    ("3e19", "d1280"): 128,
    ("3e19", "d1536"): 128,
    ("1e20", "d1024"): 128,
    ("1e20", "d1280"): 128,
    ("1e20", "d1536"): 256,
    ("1e20", "d2048"): 256,
    ("3e20", "d1024"): 256,
    ("3e20", "d1280"): 128,
    ("3e20", "d1536"): 256,
    ("3e20", "d2048"): 512,
}


def _num_global_layers(num_layers: int) -> int:
    return len({i for i in range(1, num_layers + 1) if i % GLOBAL_EVERY == 0} | {num_layers})


def flops_per_token_excl(size: IsoflopSize) -> float:
    """Forward+backward FLOPs per token excluding embed/lm_head (mixed local/global attention)."""
    h, ell = size.hidden_dim, size.num_layers
    inter = h // 2
    stored_kv = max(size.local_kv_heads, size.global_kv_heads)
    num_global = _num_global_layers(ell)
    num_local = ell - num_global

    def keys(window: int) -> float:  # avg causal keys attended with a sliding window over the sequence
        return (
            (window * (window + 1) / 2 + (SEQ_LEN - window) * window) / SEQ_LEN
            if window < SEQ_LEN
            else (SEQ_LEN + 1) / 2
        )

    mlp = 2 * 3 * h * inter * NUM_EXPERTS_PER_TOKEN + 2 * 3 * h * inter * NUM_SHARED_EXPERTS + 2 * h * NUM_EXPERTS
    qkv = 2 * h * (size.num_heads * HEAD_DIM + 2 * stored_kv * HEAD_DIM)
    o_proj = 2 * h * h
    attn = num_local * 4 * h * keys(SLIDING_WINDOW) + num_global * 4 * h * keys(SEQ_LEN)
    return 3 * (ell * (mlp + qkv + o_proj) + attn)


def _build_model(size: IsoflopSize) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=size.hidden_dim,
        intermediate_dim=size.hidden_dim // 2,
        shared_expert_intermediate_dim=size.hidden_dim // 2,
        num_shared_experts=NUM_SHARED_EXPERTS,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        num_layers=size.num_layers,
        num_heads=size.num_heads,
        num_kv_heads=max(size.local_kv_heads, size.global_kv_heads),
        local_kv_heads=size.local_kv_heads,
        global_kv_heads=size.global_kv_heads,
        head_dim=HEAD_DIM,
        max_seq_len=SEQ_LEN,
        sliding_window=SLIDING_WINDOW,
        global_every=GLOBAL_EVERY,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(size.hidden_dim),
        qk_mult=1.3,
        sconv=True,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",  # H100: portable EP backend (sonic_cute is SM100-only)
        expert_chunks=1,
        report_capacity_overflow=True,
        rope_fused=True,
    )


def _root_component(component: DatasetComponent | ConcatDatasetComponent) -> DatasetComponent | ConcatDatasetComponent:
    """Root a datakit component's relative cache_dir against MARIN_PREFIX; absolute paths pass through."""
    if isinstance(component, ConcatDatasetComponent):
        return dataclasses.replace(component, children={n: _root_component(c) for n, c in component.children.items()})
    return dataclasses.replace(component, cache_dir=datakit_source_path(component.cache_dir))


class HeroThroughputResult(Artifact):
    """Metrics-only iso-FLOP run result (temporary checkpoints only)."""


def num_steps_for(budget_label: str, size_name: str) -> int:
    tokens = BUDGETS[budget_label] / flops_per_token_excl(ISOFLOP_SIZES[size_name])
    batch = GRID_BATCH[(budget_label, size_name)]
    return max(1, round(tokens / (batch * SEQ_LEN)))


def build_isoflop_run(
    *, run_id: str, budget: str, size: str, lr_factor: float = 1.0, version: str | None = None
) -> ArtifactStep[HeroThroughputResult]:
    if (budget, size) not in GRID_BATCH:
        raise ValueError(f"cell (budget={budget}, size={size}) is not in the iso-FLOP band; valid: {sorted(GRID_BATCH)}")
    shape = ISOFLOP_SIZES[size]
    batch = GRID_BATCH[(budget, size)]
    nodes = NODES_BY_BUDGET[budget]
    steps = num_steps_for(budget, size)

    model = _build_model(shape)
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=steps, batch_size=batch, hidden_dim=shape.hidden_dim, seq_len=SEQ_LEN
    )
    optimizer = dataclasses.replace(optimizer, use_syrk=False)  # H100: portable Newton-Schulz (no SM100 syrk)
    if lr_factor != 1.0:
        # Uniformly scale muonh + adam LR (preserves the muonh:adam ratio) for the LR-sensitivity sweep.
        optimizer = dataclasses.replace(
            optimizer,
            learning_rate=optimizer.learning_rate * lr_factor,
            adam_lr=optimizer.adam_lr * lr_factor,
        )
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=False,
        expert_axis_size=1,
        replica_axis_size=1,
        sharding_dump_path=None,
    )
    resources = ResourceConfig.with_gpu("H100", count=GPUS_PER_NODE, cpu=32, ram="256g", disk="256g", replicas=nodes)
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=batch,
            num_train_steps=steps,
            profiler=ProfilerConfig(enabled=False),
            mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["grug", "moe", "aug", "isoflop", f"budget-{budget}", f"shape-{size}", "h100"],
                group="aug-hero-isoflop",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(
                watch_targets=["grads", "params", "opt_state", "updates"],
                include_norms=True,
                include_per_parameter_norms=True,
                include_histograms=True,
                split_scan_layers=False,  # stacked-layer unstack OOMs at step ~10
                interval=1,
            ),
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
        if ctx.is_fingerprint:
            val_components = {v.name: _val_component(ctx.artifact_path(v)) for v in _VALIDATION}
        else:
            val_components = {v.name: ctx.resolved(v).as_component() for v in _VALIDATION}
        data = _datakit_data_config(
            total_steps=steps,
            batch_size=batch,
            max_seq_len=SEQ_LEN,
            enable_simulated_epoching=False,
            val_components=val_components,
        )
        data = dataclasses.replace(data, components={n: _root_component(c) for n, c in data.components.items()})
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


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--budget", type=click.Choice(list(BUDGETS)), required=True, help="Compute budget (FLOPs excl embed/lm_head)."
)
@click.option("--size", type=click.Choice(list(ISOFLOP_SIZES)), required=True, help="Model width.")
@click.option("--lr-factor", type=float, default=1.0, show_default=True, help="Uniformly scale muonh+adam LR.")
@build_options
def main(run_id: str, budget: str, size: str, lr_factor: float) -> ArtifactStep[HeroThroughputResult]:
    return build_isoflop_run(run_id=run_id, budget=budget, size=size, lr_factor=lr_factor)


if __name__ == "__main__":
    main()
