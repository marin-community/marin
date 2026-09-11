# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the one-layer dense d512 constant-LR scaling sweep with MuonH."""

import dataclasses
from dataclasses import dataclass
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.analysis.backward_flow import BackwardFlowConfig
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

from experiments.grug.dense_one_layer_sgdh.launch import dense_model_config
from experiments.grug.dense_one_layer_sgdh.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.grug.moe.launch_datakit_moe_mix import (
    _VALIDATION,
    ENABLE_SIMULATED_EPOCHING,
    _datakit_data_config,
    _val_component,
)
from experiments.grug.moe_hero_fsdp_constant_lr_tpu.heuristic import MoeHeuristic
from experiments.grug.moe_hero_fsdp_constant_lr_tpu.optimizer import GrugMoeMuonHConfig

EXPERIMENT_PREFIX = "AUG-LRC-1L-DENSE-MUONH"
EXPERIMENT_VERSION = "2026.09.11"
WANDB_PROJECT = "marin_moe"
WANDB_GROUP = "issue-7856-d512-constant-lr-one-layer-dense-muonh-tpu"
TPU_STORE_PREFIX = "gs://marin-us-central2/datakit/store_8ac06c74"

D512_HIDDEN_DIM = 512
D512_BATCH_SIZE = 64
D512_SEQUENCE_LENGTH = 8192
D512_TOKEN_MULTIPLES = (30, 60, 150, 300, 600)
D512_LR_MULTIPLIERS = (0.1, 0.2, 0.32, 0.45, 0.7)
D512_STEPS = {30: 1_058, 60: 2_115, 150: 5_288, 300: 10_575, 600: 21_150}

MAX_CONCURRENT_RUNS = 5
CHECKPOINT_INTERVAL = timedelta(minutes=30)
TRAIN_RESOURCES = ResourceConfig.with_tpu(
    "v4-8",
    ram="190g",
    regions=("us-central2",),
    zone="us-central2-b",
)


@dataclass(frozen=True)
class DenseMuonHPoint:
    experiment_id: str
    token_multiple: int
    lr_multiplier: float
    num_train_steps: int

    @property
    def run_id(self) -> str:
        return f"{self.experiment_id}-d512-{self.token_multiple}x-lr{self.lr_multiplier:g}"


@dataclass(frozen=True)
class DenseMuonHExperiment:
    experiment_prefix: str
    experiment_version: str
    wandb_group: str
    lr_schedule: str
    schedule_tag: str
    min_lr_ratio: float = 0.05


CONSTANT_LR_EXPERIMENT = DenseMuonHExperiment(
    experiment_prefix=EXPERIMENT_PREFIX,
    experiment_version=EXPERIMENT_VERSION,
    wandb_group=WANDB_GROUP,
    lr_schedule="constant",
    schedule_tag="constant-lr",
)


SWEEP_POINTS = tuple(
    DenseMuonHPoint(
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


def dense_muonh_optimizer(
    point: DenseMuonHPoint,
    experiment: DenseMuonHExperiment = CONSTANT_LR_EXPERIMENT,
) -> GrugMoeMuonHConfig:
    """Build the matched MuonH optimizer for one dense sweep cell."""
    reference = MoeHeuristic(lr_schedule=experiment.lr_schedule).build_optimizer_config(
        num_train_steps=point.num_train_steps,
        batch_size=D512_BATCH_SIZE,
        hidden_dim=D512_HIDDEN_DIM,
        seq_len=D512_SEQUENCE_LENGTH,
    )
    return dataclasses.replace(
        reference,
        learning_rate=reference.learning_rate * point.lr_multiplier,
        adam_lr=reference.adam_lr * point.lr_multiplier,
        min_lr_ratio=experiment.min_lr_ratio,
    )


def build_dense_muonh_run(
    point: DenseMuonHPoint,
    *,
    experiment: DenseMuonHExperiment = CONSTANT_LR_EXPERIMENT,
    version: str = EXPERIMENT_VERSION,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one cell of the one-layer dense MuonH sweep."""
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
                    "dense",
                    "issue-7856",
                    experiment.experiment_prefix,
                    "d512",
                    "one-layer",
                    "muonh",
                    experiment.schedule_tag,
                    "tpu-v4-8",
                ],
                group=experiment.wandb_group,
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
            model=dense_model_config(),
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            optimizer=dense_muonh_optimizer(point, experiment),
            trainer=GrugTrainerConfig(
                trainer=trainer,
                data_seed=None,
                log_every=1,
                ema_beta=None,
                z_loss_weight=1e-4,
                backward_flow=BackwardFlowConfig(interval=0),
            ),
            eval=GrugEvalConfig(
                eval_batch_size=256,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
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
@click.option("--version", default=EXPERIMENT_VERSION, show_default=True)
@click.option("--max-concurrent", type=click.IntRange(min=1), default=MAX_CONCURRENT_RUNS, show_default=True)
def main(version: str, max_concurrent: int) -> None:
    """Materialize the 25-cell one-layer dense MuonH LR sweep."""
    StepRunner().run(
        [
            build_dense_muonh_run(point, experiment=CONSTANT_LR_EXPERIMENT, version=version).lower()
            for point in SWEEP_POINTS
        ],
        max_concurrent=min(max_concurrent, len(SWEEP_POINTS)),
    )


if __name__ == "__main__":
    main()
