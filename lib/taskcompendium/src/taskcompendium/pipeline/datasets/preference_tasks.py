# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve public preference prompts and private candidate labels for curation."""

import re
from collections.abc import Callable

from pydantic import ValidationError

from taskcompendium.harbor.protocol import chat_conversation
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.pipeline.datasets.source_definitions import SourceDefinition
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat, hub_inputs
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)
from taskcompendium.verifiers.base import VerifierKind
from taskcompendium.verifiers.preference import BinaryPreference, PairwisePreference, PreferenceEvidenceVerifier

PREFERENCE_CRITERIA = (
    "The public context contains every shared prior turn and the final user request; final candidates stay private.",
    "Chosen and rejected responses are relative preference evidence, not a unique exact-answer key.",
    "Flag missing context and contradictory requirements without treating disagreement with one candidate as a defect.",
    "The reward model is unbound; unavailable execution alone is not a content-quality defect.",
)

HH_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
HH_TURN = re.compile(r"\n\n(Human|Assistant):")
HH_ROLES = {"Human": "user", "Assistant": "assistant"}


def hh_conversation(text: str) -> tuple[TextMessage, ...]:
    """Read HH's documented Human/Assistant transcript delimiters without dropping prior turns."""
    segments = HH_TURN.split(text)
    if segments[0].strip() or len(segments) < 3:
        raise ValueError("Expected an HH transcript starting with a Human or Assistant delimiter")
    return tuple(
        TextMessage(role=HH_ROLES[segments[index]], content=segments[index + 1]) for index in range(1, len(segments), 2)
    )


def preference_task(row: RawRow, context: ConversationInput, verifier: PreferenceEvidenceVerifier) -> TaskSpec:
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=context,
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.PREFERENCE_EVIDENCE, parameters_json=verifier.model_dump_json()),
    )


def normalize_hh(row: RawRow) -> TaskSpec | ImportRejection:
    try:
        chosen = hh_conversation(row.data["chosen"])
        rejected = hh_conversation(row.data["rejected"])
    except (ValueError, KeyError, TypeError) as error:
        return ImportRejection(reason="invalid_preference_transcript", detail=str(error))
    if not chosen or not rejected or chosen[-1].role != "assistant" or rejected[-1].role != "assistant":
        return ImportRejection(
            reason="missing_preference_completion", detail="Both transcripts must end in assistant turns"
        )
    if chosen[:-1] != rejected[:-1]:
        return ImportRejection(reason="preference_prompt_conflict", detail="Candidates have different public histories")
    if not chosen[:-1] or chosen[-2].role != "user":
        return ImportRejection(reason="invalid_preference_prompt", detail="Public history must end in a user request")
    verifier = PreferenceEvidenceVerifier(
        evidence=PairwisePreference(chosen=(chosen[-1],), rejected=(rejected[-1],)),
        source_metadata={key: value for key, value in row.data.items() if key not in {"chosen", "rejected", "path"}},
    )
    return preference_task(row, ConversationInput(events=chosen[:-1]), verifier)


def normalize_binary(row: RawRow) -> TaskSpec | ImportRejection:
    """Preserve KTO's unpaired boolean label instead of manufacturing a rejected/chosen partner."""
    try:
        prompt = row.data["prompt"]
        completion = row.data["completion"]
        transcript = chat_conversation(prompt + completion)
        context = ConversationInput(events=transcript.events[: len(prompt)])
        verifier = PreferenceEvidenceVerifier(
            evidence=BinaryPreference(response=transcript.events[len(prompt) :], preferred=row.data["label"]),
            source_metadata={
                key: value for key, value in row.data.items() if key not in {"prompt", "completion", "label", "path"}
            },
        )
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(reason="invalid_binary_preference", detail=str(error))
    if not context.events or not completion:
        return ImportRejection(
            reason="missing_preference_messages", detail="Public prompt and labeled completion required"
        )
    return preference_task(row, ConversationInput(events=context.events), verifier)


def recipe(
    name: str,
    *,
    dataset: str,
    revision: str,
    config: str,
    rubric: ReviewRubric,
    files: SourceFiles,
    normalize: Callable[[RawRow], TaskSpec | ImportRejection],
) -> DatasetRecipe:
    return DatasetRecipe(
        name=name,
        version=f"{name}-v1",
        source=HFSource(dataset, revision, config, "train"),
        inputs=hub_inputs(dataset, revision, files),
        normalize=normalize,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
    )


SOURCES = {
    "hh_harmless_base": SourceDefinition(
        dataset="Anthropic/hh-rlhf",
        revision=HH_REVISION,
        config="harmless-base",
        split="train",
        files=SourceFiles(patterns=("harmless-base/train.jsonl.gz",), format=SourceFormat.JSONL),
        rubric=ReviewRubric(
            id="hh_harmless_base-answerability",
            version="1",
            criteria=(
                (
                    "Assess whether the public request is intelligible and whether a safe response is possible; "
                    "harmful requests can have meaningful refusal responses."
                ),
                *PREFERENCE_CRITERIA,
            ),
        ),
    ),
    "hh_helpful_base": SourceDefinition(
        dataset="Anthropic/hh-rlhf",
        revision=HH_REVISION,
        config="helpful-base",
        split="train",
        files=SourceFiles(patterns=("helpful-base/train.jsonl.gz",), format=SourceFormat.JSONL),
        rubric=ReviewRubric(
            id="hh_helpful_base-answerability",
            version="1",
            criteria=(
                (
                    "Assess the helpfulness task using the full conversation, including earlier assistant turns and "
                    "any missing requested inputs."
                ),
                *PREFERENCE_CRITERIA,
            ),
        ),
    ),
    "hh_helpful_online": SourceDefinition(
        dataset="Anthropic/hh-rlhf",
        revision=HH_REVISION,
        config="helpful-online",
        split="train",
        files=SourceFiles(patterns=("helpful-online/train.jsonl.gz",), format=SourceFormat.JSONL),
        rubric=ReviewRubric(
            id="hh_helpful_online-answerability",
            version="1",
            criteria=(
                (
                    "Assess the full online-feedback conversation; source preference alone does not certify factual "
                    "accuracy or completeness."
                ),
                *PREFERENCE_CRITERIA,
            ),
        ),
    ),
    "hh_helpful_rejection_sampled": SourceDefinition(
        dataset="Anthropic/hh-rlhf",
        revision=HH_REVISION,
        config="helpful-rejection-sampled",
        split="train",
        files=SourceFiles(patterns=("helpful-rejection-sampled/train.jsonl.gz",), format=SourceFormat.JSONL),
        rubric=ReviewRubric(
            id="hh_helpful_rejection_sampled-answerability",
            version="1",
            criteria=(
                (
                    "Assess the underlying public task independently of the rejection-sampled candidate ranking and "
                    "any candidate errors."
                ),
                *PREFERENCE_CRITERIA,
            ),
        ),
    ),
    "kto_mix": SourceDefinition(
        dataset="trl-lib/kto-mix-14k",
        revision="4470f033f33364e7d064c9f920c3df54d0cce767",
        config="default",
        split="train",
        files=SourceFiles(patterns=("data/train-00000-of-00001.parquet",), format=SourceFormat.PARQUET),
        rubric=ReviewRubric(
            id="kto-mix-answerability",
            version="1",
            criteria=(
                "Read the complete public prompt messages; the labeled candidate completion remains private.",
                "The boolean label is an unpaired preference observation; do not invent a chosen/rejected counterpart.",
                "Assess public task coherence separately from candidate quality or the source preference label.",
                "The pinned mixture has no contributor column; do not claim a sampled row belongs to a named "
                "contributor.",
                "Missing inputs and contradictions are task defects; an unbound reward model alone is not.",
            ),
        ),
    ),
}


def recipe_for_source(
    name: str,
) -> DatasetRecipe:
    source = SOURCES[name]
    return recipe(
        name,
        dataset=source.dataset,
        revision=source.revision,
        config=source.config,
        rubric=source.rubric,
        files=source.files,
        normalize=normalize_binary if name == "kto_mix" else normalize_hh,
    )
