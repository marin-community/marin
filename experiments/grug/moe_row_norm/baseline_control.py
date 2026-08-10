# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched canonical-MoE control for the factorized row-norm experiment."""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer

_TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8")
_SEQ_LEN = 8192
_BATCH_SIZE = 16
_NUM_STEPS = 10_980
_HIDDEN_DIM = 512
_RUN_ID = "MOE-ROW-NORM-CTRL-001-d512"
_WANDB_GROUP = "MOE-ROW-NORM-gate1-issue-8131"

_NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
_STARCODER_WEIGHT = 0.25
_PROOFPILE_WEIGHT = 0.055


def baseline_control_recipe() -> tuple[GrugModelConfig, GrugMoeMuonHConfig]:
    """Return the canonical model and optimizer for the matched d512 control."""
    heuristic = MoeHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(_HIDDEN_DIM, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    tokens = float(_NUM_STEPS * _BATCH_SIZE * _SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(
        _BATCH_SIZE,
        tokens,
        _HIDDEN_DIM,
        seq_len=_SEQ_LEN,
    )
    return model, optimizer


def baseline_control(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build the canonical d512 baseline matched to the row-norm Gate-1 cell."""
    name = "grug/moe_row_norm_baseline_control_d512"
    version = resolve_version(name, version)
    model, optimizer = baseline_control_recipe()
    nem = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    ]

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            run_id=_RUN_ID,
            resources=ctx.runtime_arg("train_resources"),
            steps=_NUM_STEPS,
            batch_size=_BATCH_SIZE,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=["MOE-ROW-NORM", "issue-8131", "gate1", "baseline-control", "d512"],
                group=_WANDB_GROUP,
                name=None,
            ),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1),
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
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    experiment_main(baseline_control)()
