# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FSDP ablation harness: gate 1 (d768/d1024/d1280) + gate 2 (d1536/d2048), one job per size.

Downsized copies of the FSDP hero shape (128 experts / top-4 / two shared, SConv, hybrid GQA, seq
8192, sliding-window 2048, global-every-4), each trained to its 60x token budget (60 tokens per
active parameter). Gate 1 is the quick effective-speedup check (small widths, batch 128, 1 node);
gate 2 extends to the scaling-law projection (larger widths, batch 256, 2 nodes). Depth is derived
(aspect-ratio rule rounded up to the
nearest even count) and the LR from ``MoeHeuristic``, so a size is fully specified by its width.
datakit two-phase data + paloma/uncheatable evals every 1k match the iso-FLOP/LR sweeps, so
these one-node runs are directly comparable. ``--accelerator`` picks the H100 recipe (ring MoE, no
syrk; disable PGLE at launch via ``-e JAX_ENABLE_PGLE 0``) or the B200/SM100 recipe (sonic_cute MoE +
syrk Newton-Schulz).

This is the shared harness for hero ablations (e.g. QB mechanisms, #8033); the ablated
knob is added as a CLI option + config override as each variant lands. ``--size`` submits one job.
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
SLIDING_WINDOW = 2048
GLOBAL_EVERY = 4
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOKEN = 4
NUM_SHARED_EXPERTS = 2
TOKENS_PER_ACTIVE_PARAM = 60  # 60x token budget
EVAL_BATCH_SIZE = 256
CHECKPOINT_INTERVAL = datetime.timedelta(minutes=30)

# Paloma + uncheatable held-out sets (marin_tokenizer), added as zero-train-weight datakit components.
_VALIDATION = [
    *paloma_datasets(tokenizer=marin_tokenizer).values(),
    *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
]


# Ablation widths grouped by gate (moe/agent.md gates); depth is derived (even-round), only width is
# pinned. Gate 1 = quick effective-speedup check (small widths, 1 node); gate 2 = scaling-law
# projection (larger widths, more batch + nodes).
@dataclasses.dataclass(frozen=True)
class Gate:
    widths: dict[str, int]
    batch_size: int
    nodes: int


GATES: dict[str, Gate] = {
    "gate1": Gate(widths={"d768": 768, "d1024": 1024, "d1280": 1280}, batch_size=128, nodes=1),
    "gate2": Gate(widths={"d1536": 1536, "d2048": 2048}, batch_size=256, nodes=2),
}
WIDTHS: dict[str, int] = {size: w for gate in GATES.values() for size, w in gate.widths.items()}
SIZE_GATE: dict[str, str] = {size: name for name, gate in GATES.items() for size in gate.widths}


@dataclasses.dataclass(frozen=True)
class Accelerator:
    gpu: str  # iris resource type
    gpus_per_node: int
    moe_implementation: str
    use_syrk: bool


# H100: portable ring MoE, no SM100 syrk (PGLE must be disabled via the launch env: -e JAX_ENABLE_PGLE 0).
# B200 (iris resource "GB200", SM100): sonic_cute grouped-GEMM MoE + syrk Newton-Schulz.
ACCELERATORS: dict[str, Accelerator] = {
    "H100": Accelerator(gpu="H100", gpus_per_node=8, moe_implementation="ring", use_syrk=False),
    "B200": Accelerator(gpu="GB200", gpus_per_node=4, moe_implementation="sonic_cute", use_syrk=True),
}


def _kv_heads(num_heads: int) -> tuple[int, int]:
    """Heterogeneous GQA: local layers keep 1/4 of the query heads, global layers 1/8 (floored, min 1)."""
    return max(1, num_heads // 4), max(1, num_heads // 8)


def _num_global_layers(num_layers: int) -> int:
    return len({i for i in range(1, num_layers + 1) if i % GLOBAL_EVERY == 0} | {num_layers})


def _active_params(hidden_dim: int) -> int:
    """Per-token params excluding embed/lm_head: attention, router, top-k routed + shared experts."""
    ell = round(hidden_dim / (64 + 4 * math.log2(hidden_dim) - 9))
    ell += ell % 2  # aspect-ratio depth, rounded up to nearest even
    inter = hidden_dim // 2
    num_heads = hidden_dim // HEAD_DIM
    local_kv, global_kv = _kv_heads(num_heads)
    num_global = _num_global_layers(ell)
    num_local = ell - num_global
    qo = ell * 2 * hidden_dim * hidden_dim
    kv = num_local * (2 * hidden_dim * local_kv * HEAD_DIM) + num_global * (2 * hidden_dim * global_kv * HEAD_DIM)
    router = ell * hidden_dim * NUM_EXPERTS
    routed = ell * NUM_EXPERTS_PER_TOKEN * 3 * hidden_dim * inter
    shared = ell * NUM_SHARED_EXPERTS * 3 * hidden_dim * inter
    return qo + kv + router + routed + shared


def num_steps_for(hidden_dim: int, batch_size: int) -> int:
    tokens = TOKENS_PER_ACTIVE_PARAM * _active_params(hidden_dim)
    return max(1, round(tokens / (batch_size * SEQ_LEN)))


def _build_model(hidden_dim: int, moe_implementation: str) -> GrugModelConfig:
    """The FSDP hero shape downsized to this width; depth from the even-round depth heuristic."""
    num_heads = hidden_dim // HEAD_DIM
    local_kv, global_kv = _kv_heads(num_heads)
    num_layers = round(hidden_dim / (64 + 4 * math.log2(hidden_dim) - 9))
    num_layers += num_layers % 2  # aspect-ratio depth, rounded up to nearest even
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim // 2,
        shared_expert_intermediate_dim=hidden_dim // 2,
        num_shared_experts=NUM_SHARED_EXPERTS,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=max(local_kv, global_kv),
        local_kv_heads=local_kv,
        global_kv_heads=global_kv,
        head_dim=HEAD_DIM,
        max_seq_len=SEQ_LEN,
        sliding_window=SLIDING_WINDOW,
        global_every=GLOBAL_EVERY,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(hidden_dim),
        qk_mult=1.3,
        sconv=True,
        attention_implementation="gpu_fa4_cute",
        moe_implementation=moe_implementation,
        expert_chunks=1,
        report_capacity_overflow=True,
        rope_fused=True,
    )


def _root_component(component: DatasetComponent | ConcatDatasetComponent) -> DatasetComponent | ConcatDatasetComponent:
    """Root a datakit component's relative cache_dir against MARIN_PREFIX; absolute paths pass through."""
    if isinstance(component, ConcatDatasetComponent):
        return dataclasses.replace(component, children={n: _root_component(c) for n, c in component.children.items()})
    return dataclasses.replace(component, cache_dir=datakit_source_path(component.cache_dir))


class AblationResult(Artifact):
    """Metrics-only ablation result (temporary checkpoints only)."""


def build_ablation_run(
    *, run_id: str, size: str, accelerator: str = "H100", version: str | None = None
) -> ArtifactStep[AblationResult]:
    if size not in WIDTHS:
        raise ValueError(f"size must be one of {sorted(WIDTHS)}, got {size!r}")
    if accelerator not in ACCELERATORS:
        raise ValueError(f"accelerator must be one of {sorted(ACCELERATORS)}, got {accelerator!r}")
    acc = ACCELERATORS[accelerator]
    gate = GATES[SIZE_GATE[size]]
    batch = gate.batch_size
    hidden_dim = WIDTHS[size]
    steps = num_steps_for(hidden_dim, batch)

    model = _build_model(hidden_dim, acc.moe_implementation)
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=steps, batch_size=batch, hidden_dim=hidden_dim, seq_len=SEQ_LEN
    )
    optimizer = dataclasses.replace(optimizer, use_syrk=acc.use_syrk)
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
    resources = ResourceConfig.with_gpu(
        acc.gpu, count=acc.gpus_per_node, cpu=32, ram="256g", disk="256g", replicas=gate.nodes
    )
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
                tags=["grug", "moe", "hero", "fsdp", "abl", SIZE_GATE[size], f"shape-{size}", accelerator.lower()],
                group="moe-hero-fsdp-abl",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(
                watch_targets=["grads", "params", "opt_state", "updates"],
                include_norms=True,
                include_per_parameter_norms=True,
                include_histograms=False,  # histogram ravel hits ShardingTypeError on 2D-sharded params
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
        artifact_type=AblationResult,
        run=run_grug,
        build_config=build_config,
        deps=tuple(_VALIDATION),
        runtime_args={"train_resources": resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option("--size", type=click.Choice(sorted(WIDTHS)), required=True, help="Ablation width.")
@click.option(
    "--accelerator",
    type=click.Choice(sorted(ACCELERATORS)),
    default="H100",
    show_default=True,
    help="Target accelerator: H100 (ring MoE, no syrk) or B200 (sonic_cute MoE + syrk).",
)
@build_options
def main(run_id: str, size: str, accelerator: str) -> ArtifactStep[AblationResult]:
    return build_ablation_run(run_id=run_id, size=size, accelerator=accelerator)


if __name__ == "__main__":
    main()
