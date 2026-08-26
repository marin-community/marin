# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the #7856 d512 LR sweep on TPU with a constant post-warmup LR.

The historical sweep used a 1% warmup followed by a linear decay to 5% of the
peak learning rate. This launcher preserves its d512 model, datakit mixture,
batch, seed, token budgets, peak-LR multipliers, and evaluation cadence while
changing the post-warmup schedule to constant. Training stays in us-central2 on
a v4-8 so the datakit store is read in-region.
"""

import dataclasses
import math
from collections.abc import Sequence
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
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.launch_datakit_moe_mix import (
    _VALIDATION,
    ENABLE_SIMULATED_EPOCHING,
    _datakit_data_config,
    _val_component,
)
from experiments.grug.moe_hero_fsdp_constant_lr_tpu.heuristic import MoeHeuristic
from experiments.grug.moe_hero_fsdp_constant_lr_tpu.model import GrugModelConfig
from experiments.grug.moe_hero_fsdp_constant_lr_tpu.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    run_grug,
)

EXPERIMENT_PREFIX = "AUG-LRC-TPU"
EXPERIMENT_VERSION = "2026.08.26"
WANDB_PROJECT = "marin_moe"
WANDB_GROUP = "issue-7856-d512-constant-lr-tpu"
TPU_STORE_PREFIX = "gs://marin-us-central2/datakit/store_8ac06c74"

D512_HIDDEN_DIM = 512
D512_BATCH_SIZE = 64
D512_SEQUENCE_LENGTH = 8192
D512_TOKEN_MULTIPLES = (30, 60, 150, 300, 600)
D512_LR_MULTIPLIERS = (0.7, 0.85, 1.0, 1.2, 1.4)
D512_STEPS = {
    30: 1_058,
    60: 2_115,
    150: 5_288,
    300: 10_575,
    600: 21_150,
}

MAX_CONCURRENT_RUNS = 5
CHECKPOINT_INTERVAL = timedelta(minutes=30)
TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-8", zone="us-central2-b")


@dataclass(frozen=True)
class D512ConstantLrPoint:
    """One d512 token-budget and peak-LR cell from issue #7856."""

    experiment_id: str
    token_multiple: int
    lr_multiplier: float
    num_train_steps: int

    @property
    def run_id(self) -> str:
        return f"{self.experiment_id}-d512-{self.token_multiple}x-lr{self.lr_multiplier:g}"


D512_CONSTANT_LR_POINTS = tuple(
    D512ConstantLrPoint(
        experiment_id=f"{EXPERIMENT_PREFIX}-{index:03d}",
        token_multiple=token_multiple,
        lr_multiplier=lr_multiplier,
        num_train_steps=D512_STEPS[token_multiple],
    )
    for index, (token_multiple, lr_multiplier) in enumerate(
        (
            (token_multiple, lr_multiplier)
            for token_multiple in D512_TOKEN_MULTIPLES
            for lr_multiplier in D512_LR_MULTIPLIERS
        ),
        start=1,
    )
)


def d512_model_config() -> GrugModelConfig:
    """Return the d512 architecture materialized by the completed #7856 runs."""
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=D512_HIDDEN_DIM,
        intermediate_dim=256,
        shared_expert_intermediate_dim=256,
        num_shared_experts=2,
        num_experts=128,
        num_experts_per_token=4,
        num_layers=6,
        num_heads=4,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=128,
        max_seq_len=D512_SEQUENCE_LENGTH,
        sliding_window=512,
        global_every=4,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(D512_HIDDEN_DIM),
        qk_mult=1.3,
        sconv=True,
        sconv_sites=("k", "v", "attn", "mlp"),
        attention_implementation=None,
        moe_implementation=None,
        expert_chunks=1,
        report_capacity_overflow=True,
        rope_fused=True,
    )


def constant_lr_optimizer(point: D512ConstantLrPoint):
    """Build the #7856 optimizer at ``point`` with constant post-warmup LR."""
    optimizer = MoeHeuristic(lr_schedule="constant").build_optimizer_config(
        num_train_steps=point.num_train_steps,
        batch_size=D512_BATCH_SIZE,
        hidden_dim=D512_HIDDEN_DIM,
        seq_len=D512_SEQUENCE_LENGTH,
    )
    return dataclasses.replace(
        optimizer,
        learning_rate=optimizer.learning_rate * point.lr_multiplier,
        adam_lr=optimizer.adam_lr * point.lr_multiplier,
        # QuACK's SYRK path is an SM100 implementation. The vmap fallback is the
        # equivalent portable MuonH update used on TPU.
        use_syrk=False,
    )


def select_d512_points(
    *,
    token_multiples: Sequence[int] = (),
    lr_multipliers: Sequence[float] = (),
) -> tuple[D512ConstantLrPoint, ...]:
    """Select an exact subset of the matrix; empty filters select every value."""
    selected = tuple(
        point
        for point in D512_CONSTANT_LR_POINTS
        if (not token_multiples or point.token_multiple in token_multiples)
        and (not lr_multipliers or point.lr_multiplier in lr_multipliers)
    )
    if not selected:
        raise ValueError("the d512 constant-LR selection is empty")
    return selected


def build_d512_constant_lr_run(
    point: D512ConstantLrPoint,
    *,
    version: str = EXPERIMENT_VERSION,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one TPU checkpoint cell from the d512 constant-LR matrix."""
    model = d512_model_config()
    optimizer = constant_lr_optimizer(point)
    name = f"grug/{point.run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> GrugRunConfig:
        if ctx.is_fingerprint:
            val_components = {dataset.name: _val_component(ctx.artifact_path(dataset)) for dataset in _VALIDATION}
        else:
            val_components = {dataset.name: ctx.resolved(dataset).as_component() for dataset in _VALIDATION}

        data = _datakit_data_config(
            store_prefix=TPU_STORE_PREFIX,
            total_steps=point.num_train_steps,
            batch_size=D512_BATCH_SIZE,
            max_seq_len=D512_SEQUENCE_LENGTH,
            enable_simulated_epoching=ENABLE_SIMULATED_EPOCHING,
            val_components=val_components,
        )
        trainer = TrainerConfig(
            id=point.run_id,
            seed=0,
            train_batch_size=D512_BATCH_SIZE,
            num_train_steps=point.num_train_steps,
            profiler=ProfilerConfig(enabled=False),
            mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project=WANDB_PROJECT,
                tags=[
                    "grug",
                    "moe",
                    "issue-7856",
                    EXPERIMENT_PREFIX,
                    "d512",
                    "constant-lr",
                    "tpu-v4-8",
                ],
                group=WANDB_GROUP,
                name=point.run_id,
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
        return GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=GrugTrainerConfig(
                trainer=trainer,
                data_seed=None,
                log_every=1,
                ema_beta=None,
                z_loss_weight=1e-4,
                offload_opt_state=False,
                expert_axis_size=1,
                replica_axis_size=1,
                sharding_dump_path=None,
            ),
            eval=GrugEvalConfig(
                eval_batch_size=256,
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
        artifact_type=LevanterCheckpoint,
        run=run_grug,
        build_config=build_config,
        deps=tuple(_VALIDATION),
        runtime_args={"train_resources": TRAIN_RESOURCES},
    )


@click.command()
@click.option(
    "--token-multiple",
    "token_multiples",
    multiple=True,
    type=click.Choice([str(value) for value in D512_TOKEN_MULTIPLES]),
    help="Select one or more token budgets. Omit to select all five.",
)
@click.option(
    "--lr-multiplier",
    "lr_multipliers",
    multiple=True,
    type=click.Choice([f"{value:g}" for value in D512_LR_MULTIPLIERS]),
    help="Select one or more peak-LR multipliers. Omit to select all five.",
)
@click.option(
    "--version",
    default=EXPERIMENT_VERSION,
    show_default=True,
    help="Artifact version shared by the matrix and exact retries.",
)
@click.option(
    "--max-concurrent",
    type=click.IntRange(min=1),
    default=MAX_CONCURRENT_RUNS,
    show_default=True,
    help="Maximum TPU cells materialized concurrently by this parent.",
)
def main(
    token_multiples: tuple[str, ...],
    lr_multipliers: tuple[str, ...],
    version: str,
    max_concurrent: int,
) -> None:
    """Materialize the selected d512 constant-LR TPU cells."""
    points = select_d512_points(
        token_multiples=tuple(int(value) for value in token_multiples),
        lr_multipliers=tuple(float(value) for value in lr_multipliers),
    )
    StepRunner().run(
        [build_d512_constant_lr_run(point, version=version).lower() for point in points],
        max_concurrent=min(max_concurrent, len(points)),
    )


if __name__ == "__main__":
    main()
