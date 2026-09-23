# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-update Snowball SFT comparison followed by FinanceBench.

The two arms reuse the oracle-verified weak-specification datasets from the
generation diagnostic. They differ only in whether GLM received the pinned finance
curriculum section while generating those examples. Datakit validates, renders,
normalizes, tokenizes, and packs each Parquet dataset before one matched Snowball
optimizer update. Model staging, checkpoints, and HF exports live in the CoreWeave
region's lifecycle-managed temporary bucket.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import click
from fray.cluster import ResourceConfig
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.evaluation.model_config import ModelConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.models import models
from experiments.evaluation.pipeline import EvaluationResult, ProducedEvaluationModel, eval_step
from experiments.models import ModelConfig as DownloadModelConfig
from experiments.models import download_model
from experiments.post_training.curriculum_sft.ablation.dataset import (
    DEFAULT_GENERATION_URI,
    STORE_RELATIVE_PATH,
    AblationDataset,
    AblationStore,
    dataset_step,
    store_step,
)
from experiments.post_training.curriculum_sft.ablation.matrix import (
    AblationCell,
    CurriculumCondition,
    GenerationSpec,
)
from experiments.sft.launcher import PreparedModel, SFTSpec

SNOWBALL_TOKENIZER = "marin-community/marin-tokenizer"
SNOWBALL_EOT_TOKEN_ID = 128001
SNOWBALL_END_OF_MESSAGE_TOKEN_ID = 128009
SNOWBALL_EOS_TOKEN_IDS = (SNOWBALL_EOT_TOKEN_ID, SNOWBALL_END_OF_MESSAGE_TOKEN_ID)

DATA_VERSION = "2026.09.22.3"
FINANCEBENCH_CONFIG = Path("experiments/evaluation/configs/evalchemy/financebench.yaml")
COREWEAVE_CLUSTER = "cw-rno2a"
COREWEAVE_PREFIX = "s3://marin-us-east-02a/marin"
TEMP_TTL_DAYS = 7
SNOWBALL_EVALUATION_MODEL = "snowball-datakit-sft-2026-09-20"

TRAIN_STEPS = 1
TRAIN_BATCH_SIZE = 64
TRAIN_SEQUENCE_LENGTH = 4096
DATA_AXIS_SIZE = 8
EXPERT_AXIS_SIZE = 8
_TRAIN_RESOURCES = "train_resources"


def _evaluation_model(name: str) -> ModelConfig:
    return dataclasses.replace(
        _base_model(),
        name=name,
        location="artifact://pending",
        tokenizer=SNOWBALL_TOKENIZER,
    )


def _base_model() -> ModelConfig:
    model = models()[SNOWBALL_EVALUATION_MODEL]
    if model.revision is None:
        raise ValueError(f"{SNOWBALL_EVALUATION_MODEL} must pin an immutable revision")
    return model


def _training_resources() -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "H100",
        count=8,
        cpu=32,
        ram="512g",
        disk="256g",
        replicas=8,
        preemptible=False,
    )


def _staged_model() -> ArtifactStep[LevanterCheckpoint]:
    model = _base_model()
    step = download_model(DownloadModelConfig(hf_repo_id=model.location, hf_revision=model.revision))
    output = marin_temp_bucket(
        TEMP_TTL_DAYS,
        prefix=f"curriculum-sft/snowball/base-hf/{model.revision}",
        source_prefix=COREWEAVE_PREFIX,
    )
    return dataclasses.replace(step, override_path=output)


def _dataset(condition: CurriculumCondition) -> ArtifactStep[AblationDataset]:
    generation = ArtifactStep.adopt(
        user_owned_name("documents/curriculum-sft/ablation/generation"),
        DATA_VERSION,
        source=DEFAULT_GENERATION_URI,
        kind=Artifact,
    )
    cell = AblationCell(condition, GenerationSpec.WEAK)
    return dataset_step(generation, cell, version=DATA_VERSION)


def _store(
    condition: CurriculumCondition,
    staged_model: ArtifactStep[LevanterCheckpoint],
) -> ArtifactStep[AblationStore]:
    cell = AblationCell(condition, GenerationSpec.WEAK)
    return store_step(_dataset(condition), staged_model, cell, version=DATA_VERSION)


def _sft_spec(
    condition: CurriculumCondition,
    staged_model: ArtifactStep[LevanterCheckpoint],
    *,
    version: str,
) -> SFTSpec:
    return SFTSpec(
        name=user_owned_name(f"checkpoints/curriculum-sft/snowball/{condition.value}"),
        version=version,
        model=PreparedModel(
            step=staged_model,
            model_type="snowball",
            eos_token_ids=SNOWBALL_EOS_TOKEN_IDS,
        ),
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=(),
        optimizer=AdamConfig(
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=1.0,
            weight_decay=0.0,
            lr_schedule="constant",
            warmup=0.0,
            min_lr_ratio=0.0,
        ),
        mesh=MeshConfig(
            axes={"expert": EXPERT_AXIS_SIZE, "replica": 1, "model": 1},
            dcn_axes={"data": DATA_AXIS_SIZE, "replica_dcn": 1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
        ),
        seq_len=TRAIN_SEQUENCE_LENGTH,
        pack=True,
        batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=TRAIN_STEPS,
        wandb_project="marin-curriculum-sft-snowball",
    )


def _training_data(cache_path: str, tokenizer: str, arm: str) -> LmDataConfig:
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={
            arm: DatasetComponent(
                source=None,
                cache_dir=cache_path,
                format=TextLmDatasetFormat(),
                pack=True,
            )
        },
        train_weights={arm: 1.0},
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )


def _sft_step(
    condition: CurriculumCondition,
    staged_model: ArtifactStep[LevanterCheckpoint],
    *,
    version: str,
) -> ArtifactStep[LevanterCheckpoint]:
    arm = condition.value
    store = _store(condition, staged_model)
    spec = _sft_spec(condition, staged_model, version=version)
    source = spec.model

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        tokenizer = source.resolve_tokenizer(ctx)
        data = _training_data(prefix_join(ctx.artifact_path(store), STORE_RELATIVE_PATH), tokenizer, arm)
        pod_config = source.build_train_config(ctx, spec, data, ctx.runtime_arg(_TRAIN_RESOURCES), TRAIN_STEPS)
        train_config = dataclasses.replace(
            pod_config.train_config,
            z_loss_weight=1e-4,
            hf_save_dtype="bfloat16",
        )
        return dataclasses.replace(pod_config, train_config=train_config)

    output = marin_temp_bucket(
        TEMP_TTL_DAYS,
        prefix=f"curriculum-sft/snowball/{version}/{arm}",
        source_prefix=COREWEAVE_PREFIX,
    )
    return ArtifactStep(
        name=spec.name,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=source.run,
        build_config=build_config,
        deps=(store, *source.init_deps()),
        runtime_args={_TRAIN_RESOURCES: _training_resources()},
        override_path=output,
    )


def build_pipeline(
    version: str,
) -> tuple[dict[str, ArtifactStep[LevanterCheckpoint]], dict[str, ArtifactStep[EvaluationResult]]]:
    staged_model = _staged_model()
    trainings = {
        condition.value: _sft_step(condition, staged_model, version=version)
        for condition in (CurriculumCondition.TASK_ONLY, CurriculumCondition.CURRICULUM_CONDITIONED)
    }
    evaluations = {
        arm: eval_step(
            ProducedEvaluationModel(step, _evaluation_model(f"snowball-curriculum-sft-{arm}")),
            evalchemy_config_path=FINANCEBENCH_CONFIG,
            version=version,
            limit=None,
            submission_cluster=COREWEAVE_CLUSTER,
            federated_cluster=COREWEAVE_CLUSTER,
        )
        for arm, step in trainings.items()
    }
    return trainings, evaluations


@click.command(help=__doc__)
@click.option(
    "--stage",
    type=click.Choice(("sft", "eval", "all")),
    default="all",
    show_default=True,
)
@build_options
def main(stage: str) -> dict[str, ArtifactStep]:
    version = resolve_version("curriculum-sft/snowball", None)
    trainings, evaluations = build_pipeline(version)
    if stage == "sft":
        return trainings
    if stage == "eval":
        return evaluations
    return evaluations


if __name__ == "__main__":
    main()
