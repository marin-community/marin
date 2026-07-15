# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""D512 Grug MoE run with paired MRCR context perplexity evaluation."""

import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.mrcr import mrcr_datasets, mrcr_loss_contrasts
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer

EXPERIMENT_ID = "MOE-MRCR-001"
BUDGET = 3.82e17
HIDDEN_DIM = 512
TARGET_STEPS = 10_980
TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8", zone="us-east5-a")
NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
STARCODER_WEIGHT = 0.25
PROOFPILE_WEIGHT = 0.055


def mrcr_d512(*, version: str = "2026.07.14") -> ArtifactStep[LevanterCheckpoint]:
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=BUDGET,
        hidden_dim=HIDDEN_DIM,
        target_steps=TARGET_STEPS,
    )
    run_id = os.environ.get("GRUG_RUN_ID", f"{EXPERIMENT_ID}-d512")
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nemotron[split]: weight for split, weight in NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = STARCODER_WEIGHT
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = PROOFPILE_WEIGHT
    mrcr = mrcr_datasets(tokenizer=llama3_tokenizer)
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
        *mrcr.values(),
    ]

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["MOE-MRCR", "issue-7181", "mrcr"],
                group="moe-mrcr-context-ppl",
                name=None,
            ),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
            eval=GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                loss_contrasts=mrcr_loss_contrasts(),
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(f"grug/{run_id.lower()}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    StepRunner().run([mrcr_d512().lower()])
