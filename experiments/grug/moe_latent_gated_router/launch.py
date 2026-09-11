# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched TPU ablations of the EP hero's latent feature and router input."""

import dataclasses
import math
from dataclasses import dataclass
from datetime import timedelta

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
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.launch import _NEMOTRON_WEIGHTS, _PROOFPILE_WEIGHT, _STARCODER_WEIGHT
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_latent_gated_router.model import GrugModelConfig, LatentRouting, QbEstimator
from experiments.grug.moe_latent_gated_router.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _compute_flops,
    run_grug,
)
from experiments.llama import llama3_tokenizer

ISSUE = 9110
SEQ_LEN = 4096


@dataclass(frozen=True)
class Point:
    hidden_dim: int
    layers: int
    batch_size: int
    budget: float


POINTS = {
    512: Point(512, 6, 32, 3.82e17),
    768: Point(768, 8, 64, 2.81e18),
    1024: Point(1024, 12, 128, 1.16e19),
    1280: Point(1280, 14, 256, 3.46e19),
}


def model_config(point: Point, routing: LatentRouting) -> GrugModelConfig:
    """Scale the hero architecture onto the Agent MoE width/depth grid."""
    dim = point.hidden_dim
    heads = dim // 128
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=dim,
        intermediate_dim=dim // 2,
        shared_expert_intermediate_dim=dim // 2,
        num_shared_experts=2,
        num_experts=384,
        num_experts_per_token=8,
        latent_dim=dim // 2,
        latent_routing=routing,
        num_layers=point.layers,
        num_heads=heads,
        num_kv_heads=max(1, heads // 4),
        local_kv_heads=max(1, heads // 4),
        global_kv_heads=max(1, heads // 8),
        head_dim=128,
        max_seq_len=SEQ_LEN,
        sliding_window=2048,
        global_every=4,
        initializer_std=0.5 / math.sqrt(dim),
        qk_mult=1.3,
        sconv=True,
        rope_fused=False,
        attention_implementation="tpu_splash",
        moe_implementation="ring",
        capacity_factor=1.15,
        qb_estimator=QbEstimator.HIST,
        qb_hist_bins=10_000,
        report_capacity_overflow=True,
    )


def training_steps(point: Point) -> int:
    baseline = model_config(point, LatentRouting.FULL_WIDTH_RMS)
    _, summary = _compute_flops(model_config=baseline)
    # Agent MoE budgets count forward+backward non-embedding matmuls.
    non_embedding_flops = summary["throughput/flops_per_token_analytic"] - 2 * baseline.hidden_dim * baseline.vocab_size
    return max(1, round(point.budget / (3 * non_embedding_flops * point.batch_size * SEQ_LEN)))


def trial(
    point: Point,
    routing: LatentRouting,
    *,
    run_id: str,
    stop_after_steps: int | None = None,
    version: str | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one arm; both arms use the baseline's token count and optimizer schedule."""
    model = model_config(point, routing)
    steps = training_steps(point)
    optimizer = dataclasses.replace(
        MoeHeuristic().build_optimizer_config(
            num_train_steps=steps, batch_size=point.batch_size, hidden_dim=point.hidden_dim, seq_len=SEQ_LEN
        ),
        use_syrk=False,
    )
    nem = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    ]
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=point.batch_size,
            num_train_steps=steps,
            profiler=ProfilerConfig(enabled=False),
            mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                name=run_id,
                group="moe-latent-gated-router-9110",
                tags=["MOE-LGR", "issue-9110", routing.value, f"d{point.hidden_dim}"],
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=100),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=resolve_checkpointer_output_path(
                CheckpointerConfig(save_interval=timedelta(minutes=15), keep=None), ctx.output_path
            ),
        )
        return GrugRunConfig(
            model=model,
            data=mixture(ctx, train, validation=validation),
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=GrugTrainerConfig(
                trainer=trainer,
                data_seed=1,
                ema_beta=None,
                log_every=1,
                z_loss_weight=1e-4,
                expert_axis_size=1,
                replica_axis_size=1,
                save_checkpoints=True,
            ),
            eval=GrugEvalConfig(
                eval_batch_size=point.batch_size,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
            stop_after_steps=stop_after_steps,
            max_retries_failure=2,
            max_task_failures=2,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": ResourceConfig.with_tpu("v5p-8")},
    )


@click.command()
@click.option("--dim", type=click.Choice([str(dim) for dim in POINTS]), required=True)
@click.option("--arm", type=click.Choice([arm.value for arm in LatentRouting]), required=True)
@click.option("--run-id", required=True)
@click.option("--stop-after-steps", type=click.IntRange(min=1), default=None)
@build_options
def main(dim: str, arm: str, run_id: str, stop_after_steps: int | None):
    return trial(POINTS[int(dim)], LatentRouting(arm), run_id=run_id, stop_after_steps=stop_after_steps)


if __name__ == "__main__":
    main()
