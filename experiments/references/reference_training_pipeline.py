# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference: Single-run pretraining → midtraining → SFT pipeline.

Demonstrates that pretrain, midtrain, and SFT are all data mixing phases.
One Grug training run covers the full pipeline by varying the mixture weights:

  1. Pretrain (steps 0-40k): DCLM baseline
  2. Midtrain (steps 40k-50k): blend DCLM + Dolmino math
  3. SFT (steps 50k-52k): instruction/chat data

This file uses a static mixture that approximates the time-averaged dataset
composition across all three phases. A run with step-varying weights builds
LmDataConfig directly with train_weights as a list of (start_seq_idx, weights_dict)
tuples. An SFT chat dataset (e.g. SmolTalk) adds a third component tokenized
with ChatLmDatasetFormat via a custom Dataset handle.
"""

from fray.cluster import ResourceConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.dclm import dclm_datasets
from experiments.datasets.dolmino import dolmino_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.base.launch import GrugBaseLaunchConfig, run_grug_base_trial
from experiments.grug.base.model import GrugModelConfig
from experiments.marin_tokenizer import marin_tokenizer

# --- Schedule ---
PRETRAIN_STEPS = 40_000
MIDTRAIN_STEPS = 10_000
SFT_STEPS = 2_000
TOTAL_STEPS = PRETRAIN_STEPS + MIDTRAIN_STEPS + SFT_STEPS

# Resource for TPU dispatch: a run-arg, not part of checkpoint identity.
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-8")

# --- Model: 600M Grug ---
_MODEL = GrugModelConfig(
    vocab_size=128_256,
    max_seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
)


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """600M Grug reference pipeline as a lazy checkpoint, every decision stated inline."""
    name = "references/reference-pipeline"
    version = resolve_version(name, version)
    dclm = dclm_datasets(tokenizer=marin_tokenizer)["dclm_baseline"]
    dolmino_math = dolmino_datasets(tokenizer=marin_tokenizer)["dolmino/math/metamath-owmfilter"]
    validation = [
        *paloma_datasets(tokenizer=marin_tokenizer).values(),
        *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
    ]

    # Mixture weights that approximate the time-averaged composition:
    # pretrain (40k steps) at dclm=1.0; midtrain (10k steps) at dclm=0.7, dolmino=0.3.
    train = {dclm: 1.0, dolmino_math: 0.06}

    def build_config(ctx: StepContext) -> GrugBaseLaunchConfig:
        return GrugBaseLaunchConfig(
            model=_MODEL,
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            run_id="reference-pipeline",
            resources=ctx.runtime_arg("train_resources"),
            steps=TOTAL_STEPS,
            batch_size=256,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin",
                tags=["reference", "pipeline"],
                group="reference-pipeline",
                name=None,
            ),
            optimizer=AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                warmup=0.05,
                decay=0.2,
            ),
            steps_per_eval=500,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_base_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    experiment_main(build)()
