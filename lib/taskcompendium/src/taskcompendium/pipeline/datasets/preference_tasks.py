# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve public preference prompts and private candidate labels for curation."""

import re
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from taskcompendium.harbor.protocol import chat_conversation
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SnapshotSource,
)
from taskcompendium.verifiers.preference import BinaryPreference, PairwisePreference, PreferenceEvidenceVerifier

PREFERENCE_CRITERIA = (
    "The public context contains every shared prior turn and the final user request; final candidates stay private.",
    "Chosen and rejected responses are relative preference evidence, not a unique exact-answer key.",
    "Flag missing context and contradictory requirements without treating disagreement with one candidate as a defect.",
    "The reward model is unbound; unavailable execution alone is not a content-quality defect.",
)

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
        source_metadata={
            key: value
            for key, value in row.data.items()
            if key not in {"chosen", "rejected", "path", "source_byte_offset"} and not key.startswith("sample_")
        },
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
                key: value
                for key, value in row.data.items()
                if key not in {"prompt", "completion", "label", "path", "source_byte_offset"}
                and not key.startswith("sample_")
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
    snapshot: Path,
    *,
    dataset: str,
    revision: str,
    config: str,
    rubric: ReviewRubric,
    normalize: Callable[[RawRow], TaskSpec | ImportRejection],
) -> DatasetRecipe:
    return DatasetRecipe(
        name=name,
        version=f"{name}-v1",
        source=SnapshotSource(dataset, revision, config, "train", str(snapshot)),
        normalize=normalize,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
    )
