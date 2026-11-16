# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert non-agentic evaluation samples to ordered rollout parts."""

import json

from finestore.eval import ConversationType, EvalSample, ParticipantType, RolloutContentType, RolloutRecord, SampleKind

_ASSISTANT_ROLES = frozenset({"agent", "assistant", "model"})
_SYSTEM_ROLES = frozenset({"context", "developer", "system"})


def participant_type_for_role(role: str) -> ParticipantType:
    normalized = role.lower()
    if normalized in _ASSISTANT_ROLES:
        return ParticipantType.ASSISTANT
    if normalized in _SYSTEM_ROLES:
        return ParticipantType.SYSTEM
    if normalized == "user":
        return ParticipantType.USER
    if normalized == "tool":
        return ParticipantType.TOOL
    if normalized == "environment":
        return ParticipantType.ENVIRONMENT
    return ParticipantType.OTHER


def _sample_output_and_metadata(sample: EvalSample) -> tuple[str, str]:
    if sample.kind is not SampleKind.MULTIPLE_CHOICE or sample.model_choice is None:
        return sample.output or "", "{}"
    choice = (
        sample.choices[sample.model_choice] if sample.choices and sample.model_choice < len(sample.choices) else None
    )
    if choice is None:
        return "", json.dumps({"choice_index": sample.model_choice})
    return choice.text, json.dumps({"choice_index": sample.model_choice, "choice_label": choice.label})


def rollout_records_from_sample(sample: EvalSample, *, trial_id: str = "") -> list[RolloutRecord]:
    """Convert one non-agentic evaluated sample into ordered prompt and response parts."""
    if sample.kind is SampleKind.AGENTIC:
        return []
    conversation_type = ConversationType.CHAT if sample.prompt_messages is not None else ConversationType.COMPLETION
    records = []
    if sample.prompt_messages is not None:
        for turn_id, message in enumerate(sample.prompt_messages):
            records.append(
                RolloutRecord(
                    task=sample.task,
                    doc_id=sample.doc_id,
                    trial_id=trial_id,
                    turn_id=turn_id,
                    part_id=0,
                    conversation_type=conversation_type,
                    participant_type=participant_type_for_role(message.role),
                    participant_id=message.role,
                    content_type=RolloutContentType.MESSAGE,
                    content=message.content,
                )
            )
        response_turn = len(sample.prompt_messages)
    else:
        records.append(
            RolloutRecord(
                task=sample.task,
                doc_id=sample.doc_id,
                trial_id=trial_id,
                turn_id=0,
                part_id=0,
                conversation_type=conversation_type,
                participant_type=ParticipantType.USER,
                participant_id="user",
                content_type=RolloutContentType.MESSAGE,
                content=sample.prompt_text or "",
            )
        )
        response_turn = 1

    output, metadata_json = _sample_output_and_metadata(sample)
    records.append(
        RolloutRecord(
            task=sample.task,
            doc_id=sample.doc_id,
            trial_id=trial_id,
            turn_id=response_turn,
            part_id=0,
            conversation_type=conversation_type,
            participant_type=ParticipantType.ASSISTANT,
            participant_id="assistant",
            content_type=RolloutContentType.MESSAGE,
            content=output,
            metadata_json=metadata_json,
        )
    )
    return records
