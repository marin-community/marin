# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-update Snowball SFT comparison followed by FinanceBench.

The two arms reuse the oracle-verified weak-specification datasets from the
Qwen diagnostic. They differ only in whether GLM received the pinned finance
curriculum section while generating those examples. Training uses Snowball's
own tokenizer and chat template, one fixed optimizer update, and no sequence
packing. Model staging, checkpoints, and HF exports live in the CoreWeave
region's lifecycle-managed temporary bucket.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.models.snowball import SnowballConfig
from levanter.optim.config import AdamConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.export import ConvertCheckpointStepConfig, convert_checkpoint_to_hf
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem.cluster_config import marin_temp_bucket

from experiments.evaluation.pipeline import EvaluationModelSource, EvaluationResult, evalchemy_config_step
from experiments.models import ModelConfig as DownloadModelConfig
from experiments.models import download_model
from experiments.post_training.curriculum_sft.ablation.matrix import (
    AblationCell,
    CurriculumCondition,
    GenerationSpec,
    SftDose,
)
from experiments.post_training.curriculum_sft.ablation.pipeline import (
    DEFAULT_GENERATION_URI,
    TRAIN_FILENAME,
    AblationDataset,
    dataset_step,
)
from experiments.sft.launcher import ArtifactDatasetSpec, ModelSource, PreparedModel, SFTSpec, sft_step

SNOWBALL_REPO = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.20"
SNOWBALL_REVISION = "9f2ee50f3d4a12c79b0808bb2414ddba2cdf0098"
SNOWBALL_TOKENIZER = "marin-community/marin-tokenizer"
SNOWBALL_EOS_TOKEN_IDS = (128001, 128009)

DATA_VERSION = "2026.09.21.7"
FINANCEBENCH_CONFIG = Path("experiments/evaluation/configs/evalchemy/financebench.yaml")
COREWEAVE_CLUSTER = "cw-rno2a"
COREWEAVE_PREFIX = "s3://marin-us-east-02a/marin"
TEMP_TTL_DAYS = 7

TRAIN_STEPS = 1
TRAIN_BATCH_SIZE = 64
TRAIN_SEQUENCE_LENGTH = 4096
DATA_AXIS_SIZE = 8
EXPERT_AXIS_SIZE = 8
MAX_COMPILED_MEMORY_BYTES = 70 * 1024**3
RECOVERY_SOURCE_VERSION = "2026.09.21.11"


@dataclass(frozen=True)
class SnowballModelSource(ModelSource):
    """Pinned staged Snowball checkpoint with the proven 64-H100 mesh."""

    staged_model: ArtifactStep[LevanterCheckpoint]

    def _prepared(self) -> PreparedModel:
        return PreparedModel(
            step=self.staged_model,
            model_type="snowball",
            eos_token_ids=SNOWBALL_EOS_TOKEN_IDS,
        )

    def tokenizer_cache_key(self) -> str:
        return SNOWBALL_TOKENIZER

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return SNOWBALL_TOKENIZER

    @property
    def run(self) -> Callable[..., None]:
        return self._prepared().run

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return self._prepared().init_deps()

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config: LmDataConfig,
        resources: ResourceConfig,
        num_train_steps: int,
    ) -> TrainLmOnPodConfig:
        pod_config = self._prepared().build_train_config(
            ctx,
            spec,
            data_config,
            resources,
            num_train_steps,
        )
        mesh = MeshConfig(
            axes={"data": 1, "expert": EXPERT_AXIS_SIZE, "replica": 1, "model": 1},
            dcn_axes={"data": DATA_AXIS_SIZE, "replica_dcn": 1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
        )
        trainer = dataclasses.replace(
            pod_config.train_config.trainer,
            mesh=mesh,
            use_explicit_mesh_axes=True,
            per_device_parallelism=1,
            max_compiled_memory_bytes=MAX_COMPILED_MEMORY_BYTES,
            log_jaxprs=False,
            log_xla_hlo=False,
        )
        train_config = dataclasses.replace(
            pod_config.train_config,
            trainer=trainer,
            z_loss_weight=1e-4,
            hf_save_dtype="bfloat16",
            # The staged base is a data artifact, not an HF repository. Snowball needs no
            # remote model code, so copying the tokenizer and generated config is sufficient.
            hf_save_reference_code=False,
        )
        return dataclasses.replace(pod_config, train_config=train_config)


@dataclass(frozen=True)
class ProducedSnowballEvaluationModel(EvaluationModelSource):
    """Resolve a Snowball SFT artifact's final HF export for evaluation."""

    step: ArtifactStep[Artifact]
    name: str

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ModelConfig:
        if ctx.is_fingerprint:
            location = f"artifact://{self.step.name}@{self.step.version}"
        else:
            checkpoints = discover_hf_checkpoints(ctx.artifact_path(self.step))
            if not checkpoints:
                raise FileNotFoundError(f"no HF checkpoint found under {ctx.artifact_path(self.step)}")
            location = checkpoints[-1]
        return ModelConfig(
            name=self.name,
            location=location,
            tokenizer=SNOWBALL_TOKENIZER,
            apply_chat_template=True,
            resource_hint=ResourceHint(gpu={"H100": 8}, memory="512g"),
            serve=ServeConfig(
                tensor_parallel_size=1,
                data_parallel_size=8,
                max_model_len=73728,
                max_num_seqs=32,
                auto_overrides=False,
                vllm_extra_args=(
                    "--enable-expert-parallel",
                    "--model-loader-extra-config",
                    '{"distributed":true}',
                ),
            ),
            generation=GenerationConfig(extra_gen_kwargs={"skip_special_tokens": "false"}),
        )


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


def _export_resources() -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "H100",
        count=8,
        cpu=32,
        ram="512g",
        disk="256g",
        preemptible=False,
    )


def _staged_model() -> ArtifactStep[LevanterCheckpoint]:
    step = download_model(DownloadModelConfig(hf_repo_id=SNOWBALL_REPO, hf_revision=SNOWBALL_REVISION))
    output = marin_temp_bucket(
        TEMP_TTL_DAYS,
        prefix=f"curriculum-sft/snowball/base-hf/{SNOWBALL_REVISION}",
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
    cell = AblationCell(condition, GenerationSpec.WEAK, SftDose.LOW)
    return dataset_step(generation, cell, version=DATA_VERSION)


def _sft_step(
    condition: CurriculumCondition,
    staged_model: ArtifactStep[LevanterCheckpoint],
    *,
    version: str,
) -> ArtifactStep[LevanterCheckpoint]:
    arm = condition.value
    step = sft_step(
        SFTSpec(
            name=user_owned_name(f"checkpoints/curriculum-sft/snowball/{arm}"),
            version=version,
            model=SnowballModelSource(staged_model),
            chat_template=MARIN_CHAT_TEMPLATE,
            datasets=(
                ArtifactDatasetSpec(
                    slug=f"snowball-curriculum-{arm}",
                    artifact=_dataset(condition),
                    relative_pattern=TRAIN_FILENAME,
                    weight=1.0,
                ),
            ),
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
            seq_len=TRAIN_SEQUENCE_LENGTH,
            pack=False,
            batch_size=TRAIN_BATCH_SIZE,
            num_train_steps=TRAIN_STEPS,
            wandb_project="marin-curriculum-sft-snowball",
        ),
        _training_resources(),
    )
    output = marin_temp_bucket(
        TEMP_TTL_DAYS,
        prefix=f"curriculum-sft/snowball/{version}/{arm}",
        source_prefix=COREWEAVE_PREFIX,
    )
    return dataclasses.replace(step, override_path=output)


def build_pipeline(
    version: str,
) -> tuple[dict[str, ArtifactStep[LevanterCheckpoint]], dict[str, ArtifactStep[EvaluationResult]]]:
    staged_model = _staged_model()
    trainings = {
        condition.value: _sft_step(condition, staged_model, version=version)
        for condition in (CurriculumCondition.TASK_ONLY, CurriculumCondition.CURRICULUM_CONDITIONED)
    }
    evaluations = {
        arm: evalchemy_config_step(
            ProducedSnowballEvaluationModel(step, f"snowball-curriculum-sft-{arm}"),
            FINANCEBENCH_CONFIG,
            version=version,
            limit=None,
            submission_cluster=COREWEAVE_CLUSTER,
            federated_cluster=COREWEAVE_CLUSTER,
        )
        for arm, step in trainings.items()
    }
    return trainings, evaluations


def _recovery_export_step(
    condition: CurriculumCondition,
    staged_model: ArtifactStep[LevanterCheckpoint],
    *,
    version: str,
) -> ArtifactStep[Artifact]:
    """Export a completed native checkpoint after the original final hook failed."""
    arm = condition.value
    source_root = "s3://marin-us-east-02a/tmp/ttl=7d/curriculum-sft/snowball"
    source = f"{source_root}/{RECOVERY_SOURCE_VERSION}/{arm}/checkpoints"
    resources = _export_resources()

    def build_config(ctx: StepContext) -> ConvertCheckpointStepConfig:
        tokenizer = ctx.artifact_path(staged_model)
        return ConvertCheckpointStepConfig(
            checkpoint_path=source,
            discover_latest=True,
            job_name=f"export-snowball-{arm}",
            trainer=TrainerConfig(
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                mesh=MeshConfig(
                    axes={"data": 1, "expert": EXPERT_AXIS_SIZE, "replica": 1, "model": 1},
                    dcn_axes={"replica_dcn": 1},
                    compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
                ),
                use_explicit_mesh_axes=True,
                log_jaxprs=False,
                log_xla_hlo=False,
            ),
            model=SnowballConfig(
                hidden_dim=2560,
                intermediate_dim=1280,
                shared_expert_intermediate_dim=2560,
                num_experts=256,
                num_experts_per_token=4,
                num_layers=26,
                num_heads=20,
                num_kv_heads=5,
                head_dim=128,
                max_seq_len=262144,
                sliding_window=2048,
                layer_norm_eps=1e-5,
                initializer_std=0.009882117688026186,
                qk_mult=1.75,
                reference_checkpoint=f"{SNOWBALL_REPO}@{SNOWBALL_REVISION}",
                tokenizer=tokenizer,
            ),
            resources=ctx.runtime_arg("resources"),
            output_path=ctx.output_path,
            tokenizer=tokenizer,
            save_dtype="bfloat16",
            config_overrides={
                "bos_token_id": 128000,
                "eos_token_id": 128001,
                "decoder_start_token_id": 128000,
                "begin_suppress_tokens": [128000, 128001],
            },
        )

    step = ArtifactStep(
        name=user_owned_name(f"exports/curriculum-sft/snowball/{arm}"),
        version=version,
        artifact_type=Artifact,
        run=convert_checkpoint_to_hf,
        build_config=build_config,
        deps=(staged_model,),
        runtime_args={"resources": resources},
    )
    output = marin_temp_bucket(
        TEMP_TTL_DAYS,
        prefix=f"curriculum-sft/snowball/{version}/recovered/{arm}",
        source_prefix=COREWEAVE_PREFIX,
    )
    return dataclasses.replace(step, override_path=output)


def build_recovery_pipeline(
    version: str,
) -> tuple[dict[str, ArtifactStep[Artifact]], dict[str, ArtifactStep[EvaluationResult]]]:
    staged_model = _staged_model()
    exports = {
        condition.value: _recovery_export_step(condition, staged_model, version=version)
        for condition in (CurriculumCondition.TASK_ONLY, CurriculumCondition.CURRICULUM_CONDITIONED)
    }
    evaluations = {
        arm: evalchemy_config_step(
            ProducedSnowballEvaluationModel(step, f"snowball-curriculum-sft-{arm}"),
            FINANCEBENCH_CONFIG,
            version=version,
            limit=None,
            submission_cluster=COREWEAVE_CLUSTER,
            federated_cluster=COREWEAVE_CLUSTER,
        )
        for arm, step in exports.items()
    }
    return exports, evaluations


@click.command(help=__doc__)
@click.option(
    "--stage",
    type=click.Choice(("sft", "eval", "all", "recover-export", "recover-eval")),
    default="all",
    show_default=True,
)
@build_options
def main(stage: str) -> dict[str, ArtifactStep]:
    version = resolve_version("curriculum-sft/snowball", None)
    if stage in {"recover-export", "recover-eval"}:
        exports, recovery_evaluations = build_recovery_pipeline(version)
        return exports if stage == "recover-export" else recovery_evaluations
    trainings, evaluations = build_pipeline(version)
    if stage == "sft":
        return trainings
    if stage == "eval":
        return evaluations
    return evaluations


if __name__ == "__main__":
    main()
