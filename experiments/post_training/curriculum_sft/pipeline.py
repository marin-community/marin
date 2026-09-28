# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate and prepare curriculum conversations for the shared SFT launcher."""

from collections.abc import Sequence
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name

from experiments.post_training.curriculum_sft.chat_preparation import PrepareConfig, prepare_generated_chat
from experiments.post_training.curriculum_sft.generation import generate_curriculum_sft
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM, TaskCurriculumCatalogArtifact


@dataclass(frozen=True)
class CurriculumGenerationSpec:
    requested_examples: int
    accepted_examples: int
    seed: int
    max_completion_tokens: int
    task_specification: str


def curriculum_generation_steps(
    curriculum_ids: Sequence[str],
    *,
    version: str,
    generation: CurriculumGenerationSpec,
    catalog: ArtifactStep[TaskCurriculumCatalogArtifact] = TASK_CURRICULUM,
) -> dict[str, ArtifactStep[Artifact]]:
    return {
        capability_id: generate_curriculum_sft(
            capability_id,
            catalog=catalog,
            version=version,
            requested_examples=generation.requested_examples,
            accepted_examples=generation.accepted_examples,
            seed=generation.seed,
            max_completion_tokens=generation.max_completion_tokens,
            task_specification=generation.task_specification,
        )
        for capability_id in curriculum_ids
    }


def prepare_curriculum_chat_step(
    generated: ArtifactStep[Artifact], *, capability_id: str, version: str
) -> ArtifactStep[Artifact]:
    """Convert one generated chat artifact to canonical OpenAI messages Parquet."""

    def build_config(ctx: StepContext) -> PrepareConfig:
        return PrepareConfig(ctx.artifact_path(generated), ctx.output_path)

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/prepared-chat"),
        version=version,
        artifact_type=Artifact,
        run=prepare_generated_chat,
        build_config=build_config,
        deps=(generated,),
    )
