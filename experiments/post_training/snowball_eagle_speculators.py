# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit a Snowball EAGLE-3 draft on direct evaluation rollouts.

The workflow uses Marin's ordinary evaluation path for both data collection and
the matched control/initial/trained measurements:

1. serve the target with vLLM and collect Evalchemy plus Harbor rollouts;
2. convert the sealed FineStore archives into Speculators conversations;
3. capture target hidden states and train the draft offline;
4. run source-disjoint evaluations with no draft, the starting draft, and the
   trained draft.

Run the complete bounded experiment on the US East 02A controller::

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
      --enable-extra-resources --target-cluster cw-us-east-02a \
      -- python experiments/post_training/snowball_eagle_speculators.py \
      --version 2026.09.23.3 --stage evaluations --run
"""

from __future__ import annotations

from dataclasses import dataclass

import click
from fray.types import ResourceConfig
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.external_dependencies import SPECULATORS
from marin.inference.config import SpeculativeMethod
from marin.training.speculators import (
    SPECULATORS_DATA_FILENAME,
    DraftTrainingConfig,
    EagleDraftArtifact,
    HfSnapshotConfig,
    HiddenStateCaptureConfig,
    RolloutConversationConfig,
    VerifierViewConfig,
    build_verifier_view,
    capture_hidden_states,
    mirror_hf_snapshot,
    train_draft,
    write_rollout_conversations,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import username_segment

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import (
    ArtifactEvaluationModel,
    ArtifactSpeculativeModel,
    EvaluationResult,
    eval_step,
)

TARGET_MODEL_NAME = "snowball-67b-a2b-sft-s3-agentic-step1903"
TARGET_MODEL_URI = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s3-agentic/" "step-1903/hf-bf16-vllm/"
TARGET_TOKENIZER = "marin-community/marin-tokenizer"
TARGET_TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
TARGET_TOKENIZER_EXPORT = TARGET_MODEL_URI
TARGET_MODEL = ArtifactStep.adopt(
    f"models/{TARGET_MODEL_NAME}",
    "2026.09.21",
    TARGET_MODEL_URI,
    kind=LevanterCheckpoint,
)
INITIAL_DRAFT_REPO = "laion/snowball-64k-eagle3-draft-r2egym"
INITIAL_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
TARGET_LAYER_IDS = (2, 13, 23)
VERIFIER_NUM_HIDDEN_LAYERS = 26
SEQUENCE_LENGTH = 32_768
CLUSTER = "cw-us-east-02a"
GPU_VARIANT = "H100"
GPU_COUNT = 8

# This 2,036-trajectory pilot matches the example-presentation budget of the
# strongest prior Snowball adaptation while making a quarter of the corpus
# tool-use behavior that the earlier corpus omitted. MATH-500 has 500 distinct
# documents; OlympiadBench has only 30, regardless of a larger launcher limit.
CORPUS_EVALS = (("gsm8k", 1024), ("math500", 500), ("bfcl", 512))
CORPUS_EXPECTED_CONVERSATIONS = sum(limit for _, limit in CORPUS_EVALS)
CORPUS_MAX_GENERATION_TOKENS = 4096
CORPUS_MINIMUM_VALID_TOKENS = 32

# Evaluation sources are disjoint from the SFT sources. The smaller agentic arm
# bounds Daytona cost while still measuring multi-turn drafting behavior.
BENCHMARK_EVALS = (("aime24", 64), ("tb2", 32))
BENCHMARK_MAX_GENERATION_TOKENS = 8192
NUM_SPECULATIVE_TOKENS = 3

DRAFT_EPOCHS = 8
DRAFT_TASK_CPU = 96
DRAFT_TASK_MEMORY = "512g"
DRAFT_TASK_DISK = "1t"
TORCHAUDIO_CU128_REQUIREMENT = (
    "torchaudio @ https://download.pytorch.org/whl/cu128/"
    "torchaudio-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
    "#sha256=78b86a17f164bdaabdcee93fdfde2587fc43b9ebf15cd61dcf730b4f8615176b"
)


def _artifact_name(suffix: str) -> str:
    return user_owned_name(f"snowball-eagle3-{suffix}")


def _target_config(name: str, generation: GenerationConfig) -> ModelConfig:
    return ModelConfig(
        name=f"{username_segment()}-{name}",
        location=TARGET_MODEL_URI,
        tokenizer=TARGET_TOKENIZER,
        tokenizer_revision=TARGET_TOKENIZER_REVISION,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={GPU_VARIANT: GPU_COUNT}, memory="512g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=GPU_COUNT,
            max_model_len=SEQUENCE_LENGTH,
            max_num_batched_tokens=16_384,
            max_num_seqs=16,
            vllm_extra_args=SNOWBALL_VLLM_ARGS,
        ),
        generation=generation,
    )


def _evaluation_model(name: str, generation: GenerationConfig) -> ArtifactEvaluationModel:
    return ArtifactEvaluationModel(
        step=TARGET_MODEL,
        model=_target_config(name, generation),
    )


def _eval_step(
    *,
    model: ArtifactEvaluationModel,
    evaluation: str,
    limit: int,
    label: str,
    draft: ArtifactStep[EagleDraftArtifact] | None = None,
) -> ArtifactStep[EvaluationResult]:
    speculative = None
    if draft is not None:
        speculative = ArtifactSpeculativeModel(
            step=draft,
            method=SpeculativeMethod.EAGLE3,
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        )
    name = f"evals/{model.model.name}/{evaluation}-{label}"
    return eval_step(
        model,
        evaluation,
        version=resolve_version(name, None),
        speculative=speculative,
        limit=limit,
        accelerator=f"{GPU_VARIANT}x{GPU_COUNT}",
        federated_cluster=CLUSTER,
    )


def _corpus_rollouts() -> tuple[ArtifactStep[EvaluationResult], ...]:
    model = _evaluation_model(
        "snowball-eagle3-corpus-target",
        GenerationConfig(
            max_gen_toks=CORPUS_MAX_GENERATION_TOKENS,
            extra_gen_kwargs={"temperature": "1.0", "top_p": "1.0"},
        ),
    )
    return tuple(
        _eval_step(model=model, evaluation=evaluation, limit=limit, label="corpus") for evaluation, limit in CORPUS_EVALS
    )


def _conversation_step(
    rollouts: tuple[ArtifactStep[EvaluationResult], ...],
) -> ArtifactStep[Artifact]:
    name = _artifact_name("conversations")

    def build_config(ctx: StepContext) -> RolloutConversationConfig:
        if ctx.is_fingerprint:
            archives = tuple(ctx.artifact_path(step) for step in rollouts)
        else:
            archives = tuple(path for step in rollouts for path in ctx.resolved(step).results_paths)
        return RolloutConversationConfig(
            source_archives=archives,
            output_path=ctx.output_path,
            expected_conversations=CORPUS_EXPECTED_CONVERSATIONS,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            write_rollout_conversations,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="16g"),
        ),
        build_config=build_config,
        deps=rollouts,
    )


def _initial_draft_step() -> ArtifactStep[EagleDraftArtifact]:
    name = _artifact_name("initial-draft")
    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=EagleDraftArtifact,
        run=remote(
            mirror_hf_snapshot,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="64g"),
        ),
        build_config=lambda ctx: HfSnapshotConfig(
            repo_id=INITIAL_DRAFT_REPO,
            revision=INITIAL_DRAFT_REVISION,
            output_path=ctx.output_path,
        ),
    )


def _verifier_step() -> ArtifactStep[Artifact]:
    name = _artifact_name("verifier-view")
    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            build_verifier_view,
            resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="32g"),
        ),
        build_config=lambda ctx: VerifierViewConfig(
            source_model=ctx.artifact_path(TARGET_MODEL),
            transformers_model_type="llama",
            output_path=ctx.output_path,
        ),
        deps=(TARGET_MODEL,),
    )


def _capture_step(conversations: ArtifactStep[Artifact]) -> ArtifactStep[Artifact]:
    name = _artifact_name("hidden-states")
    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            capture_hidden_states,
            resources=ResourceConfig.with_gpu(
                GPU_VARIANT,
                count=GPU_COUNT,
                cpu=DRAFT_TASK_CPU,
                ram=DRAFT_TASK_MEMORY,
                disk=DRAFT_TASK_DISK,
            ),
            pip_packages=[SPECULATORS.requirement(), TORCHAUDIO_CU128_REQUIREMENT],
        ),
        build_config=lambda ctx: HiddenStateCaptureConfig(
            dataset_path=prefix_join(ctx.artifact_path(conversations), SPECULATORS_DATA_FILENAME),
            target_model=ctx.artifact_path(TARGET_MODEL),
            tokenizer=TARGET_TOKENIZER_EXPORT,
            output_path=ctx.output_path,
            target_layer_ids=TARGET_LAYER_IDS,
            verifier_num_hidden_layers=VERIFIER_NUM_HIDDEN_LAYERS,
            sequence_length=SEQUENCE_LENGTH,
            data_parallel_size=GPU_COUNT,
            concurrency=64,
            max_samples=CORPUS_EXPECTED_CONVERSATIONS,
            minimum_valid_tokens=CORPUS_MINIMUM_VALID_TOKENS,
            gpu_memory_utilization=0.9,
            vllm_extra_args=SNOWBALL_VLLM_ARGS,
        ),
        deps=(conversations, TARGET_MODEL),
    )


def _draft_step(
    captured_data: ArtifactStep[Artifact],
    verifier: ArtifactStep[Artifact],
    initial_draft: ArtifactStep[EagleDraftArtifact],
) -> ArtifactStep[EagleDraftArtifact]:
    name = _artifact_name("trained-draft")
    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=EagleDraftArtifact,
        run=remote(
            train_draft,
            resources=ResourceConfig.with_gpu(
                GPU_VARIANT,
                count=GPU_COUNT,
                cpu=DRAFT_TASK_CPU,
                ram=DRAFT_TASK_MEMORY,
                disk=DRAFT_TASK_DISK,
            ),
            pip_packages=[SPECULATORS.requirement(), TORCHAUDIO_CU128_REQUIREMENT],
        ),
        build_config=lambda ctx: DraftTrainingConfig(
            captured_data_path=ctx.artifact_path(captured_data),
            verifier_path=ctx.artifact_path(verifier),
            initial_draft_path=ctx.artifact_path(initial_draft),
            output_path=ctx.output_path,
            target_layer_ids=TARGET_LAYER_IDS,
            sequence_length=SEQUENCE_LENGTH,
            epochs=DRAFT_EPOCHS,
            learning_rate=1e-5,
            muon_learning_rate=0.02,
            num_processes=GPU_COUNT,
            train_data_ratio=0.9,
            save_best=True,
        ),
        deps=(captured_data, verifier, initial_draft),
    )


@dataclass(frozen=True)
class DraftSftPipeline:
    rollouts: tuple[ArtifactStep[EvaluationResult], ...]
    conversations: ArtifactStep[Artifact]
    initial_draft: ArtifactStep[EagleDraftArtifact]
    verifier: ArtifactStep[Artifact]
    captured_data: ArtifactStep[Artifact]
    draft: ArtifactStep[EagleDraftArtifact]


@dataclass(frozen=True)
class SnowballDraftPipeline:
    sft: DraftSftPipeline
    evaluations: dict[str, ArtifactStep[EvaluationResult]]


def _benchmark_steps(
    *,
    label: str,
    draft: ArtifactStep[EagleDraftArtifact] | None,
) -> dict[str, ArtifactStep[EvaluationResult]]:
    model = _evaluation_model(
        f"snowball-eagle3-{label}",
        GenerationConfig(
            max_gen_toks=BENCHMARK_MAX_GENERATION_TOKENS,
            extra_gen_kwargs={"temperature": "0.0"},
        ),
    )
    return {
        f"{label}-{evaluation}": _eval_step(
            model=model,
            evaluation=evaluation,
            limit=limit,
            label=label,
            draft=draft,
        )
        for evaluation, limit in BENCHMARK_EVALS
    }


def build_pipeline() -> SnowballDraftPipeline:
    rollouts = _corpus_rollouts()
    conversations = _conversation_step(rollouts)
    initial_draft = _initial_draft_step()
    verifier = _verifier_step()
    captured_data = _capture_step(conversations)
    draft = _draft_step(captured_data, verifier, initial_draft)
    sft = DraftSftPipeline(
        rollouts=rollouts,
        conversations=conversations,
        initial_draft=initial_draft,
        verifier=verifier,
        captured_data=captured_data,
        draft=draft,
    )
    evaluations = {
        **_benchmark_steps(label="control", draft=None),
        **_benchmark_steps(label="initial", draft=initial_draft),
        **_benchmark_steps(label="trained", draft=draft),
    }
    return SnowballDraftPipeline(sft=sft, evaluations=evaluations)


_STAGES = (
    "rollouts",
    "conversations",
    "initial_draft",
    "verifier",
    "captured_data",
    "draft",
    "evaluation-control",
    "evaluation-initial",
    "evaluation-trained",
    "evaluations",
)


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(_STAGES), default="evaluations", show_default=True)
@build_options
def main(stage: str) -> ArtifactStep | dict[str, ArtifactStep] | tuple[ArtifactStep, ...]:
    pipeline = build_pipeline()
    if stage == "rollouts":
        return pipeline.sft.rollouts
    if stage == "evaluations":
        return pipeline.evaluations
    if stage.startswith("evaluation-"):
        label = stage.removeprefix("evaluation-")
        return {name: step for name, step in pipeline.evaluations.items() if name.startswith(f"{label}-")}
    return getattr(pipeline.sft, stage)


if __name__ == "__main__":
    main()
