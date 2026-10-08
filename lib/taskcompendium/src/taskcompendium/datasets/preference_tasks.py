# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve public preference prompts and private candidate labels for curation."""

import re

from pydantic import ValidationError

from taskcompendium.chat import chat_conversation
from taskcompendium.datasets.direct_contracts import source_contract_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
)

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


def preference_task(row: RawRow, context: ConversationInput, evidence: dict) -> TaskSpec:
    package = source_contract_package(
        "source preference reward model",
        row.source.revision,
        evidence,
        ("Source preference reward model binding",),
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=context,
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
    )


def normalize_hh(row: RawRow) -> TaskSpec | ImportRejection:
    try:
        chosen = hh_conversation(row.data["chosen"])
        rejected = hh_conversation(row.data["rejected"])
    except (ValueError, KeyError, TypeError) as error:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED, reason="invalid_preference_transcript", detail=str(error)
        )
    if not chosen or not rejected or chosen[-1].role != "assistant" or rejected[-1].role != "assistant":
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_preference_completion",
            detail="Both transcripts must end in assistant turns",
        )
    if chosen[:-1] != rejected[:-1]:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="preference_prompt_conflict",
            detail="Candidates have different public histories",
        )
    if not chosen[:-1] or chosen[-2].role != "user":
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="invalid_preference_prompt",
            detail="Public history must end in a user request",
        )
    evidence = {
        "kind": "pairwise",
        "chosen": [chosen[-1].model_dump(mode="json")],
        "rejected": [rejected[-1].model_dump(mode="json")],
    }
    return preference_task(row, ConversationInput(events=chosen[:-1]), evidence)


def normalize_binary(row: RawRow) -> TaskSpec | ImportRejection:
    """Preserve KTO's unpaired boolean label instead of manufacturing a rejected/chosen partner."""
    try:
        prompt = row.data["prompt"]
        completion = row.data["completion"]
        transcript = chat_conversation(prompt + completion)
        context = ConversationInput(events=transcript.events[: len(prompt)])
        label = row.data["label"]
        if not isinstance(label, bool):
            raise ValueError("Preference label must be a boolean")
        evidence = {
            "kind": "binary",
            "response": [event.model_dump(mode="json") for event in transcript.events[len(prompt) :]],
            "preferred": label,
        }
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_binary_preference", detail=str(error)
        )
    if not context.events or not completion:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_preference_messages",
            detail="Public prompt and labeled completion required",
        )
    return preference_task(row, ConversationInput(events=context.events), evidence)


def hh_policy(rubric: ReviewRubric) -> TaskPolicy:
    return TaskPolicy(normalize=normalize_hh, rubric=rubric)


def binary_policy(rubric: ReviewRubric) -> TaskPolicy:
    return TaskPolicy(normalize=normalize_binary, rubric=rubric)


HH_HARMLESS_BASE_RUBRIC = ReviewRubric(
    id="hh_harmless_base-answerability",
    version="1",
    criteria=(
        (
            "Assess whether the public request is intelligible and whether a safe response is possible; "
            "harmful requests can have meaningful refusal responses."
        ),
        *PREFERENCE_CRITERIA,
    ),
)

HH_HELPFUL_BASE_RUBRIC = ReviewRubric(
    id="hh_helpful_base-answerability",
    version="1",
    criteria=(
        (
            "Assess the helpfulness task using the full conversation, including earlier assistant turns and "
            "any missing requested inputs."
        ),
        *PREFERENCE_CRITERIA,
    ),
)

HH_HELPFUL_ONLINE_RUBRIC = ReviewRubric(
    id="hh_helpful_online-answerability",
    version="1",
    criteria=(
        (
            "Assess the full online-feedback conversation; source preference alone does not certify factual "
            "accuracy or completeness."
        ),
        *PREFERENCE_CRITERIA,
    ),
)

HH_HELPFUL_REJECTION_SAMPLED_RUBRIC = ReviewRubric(
    id="hh_helpful_rejection_sampled-answerability",
    version="1",
    criteria=(
        (
            "Assess the underlying public task independently of the rejection-sampled candidate ranking and "
            "any candidate errors."
        ),
        *PREFERENCE_CRITERIA,
    ),
)

KTO_MIX_RUBRIC = ReviewRubric(
    id="kto-mix-answerability",
    version="1",
    criteria=(
        "Read the complete public prompt messages; the labeled candidate completion remains private.",
        "The boolean label is an unpaired preference observation; do not invent a chosen/rejected counterpart.",
        "Assess public task coherence separately from candidate quality or the source preference label.",
        "The pinned mixture has no contributor column; do not claim a sampled row belongs to a named " "contributor.",
        "Missing inputs and contradictions are task defects; an unbound reward model alone is not.",
    ),
)
