# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the shared FineStore rollout table from normalized evaluation rows."""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Iterable, Iterator

from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    ROLLOUT_SCHEMA_VERSION,
    ConversationType,
    EvalSample,
    EvaluationStore,
    ParticipantType,
    RolloutContentType,
    RolloutRecord,
    SampleKind,
    StepRecord,
    sample_from_archive_row,
)
from finestore.reader import ReadView

_ASSISTANT_ROLES = frozenset({"agent", "assistant", "model"})
_SYSTEM_ROLES = frozenset({"context", "developer", "system"})
_COMPLETE_OBJECT = f"_marin/rollouts-v{ROLLOUT_SCHEMA_VERSION}-complete.json"
_SOURCE_TABLES = (ARCHIVE_SAMPLES_TABLE, ARCHIVE_STEPS_TABLE)


def _participant(role: str) -> ParticipantType:
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
                    participant_type=_participant(message.role),
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


def rollout_records_from_steps(steps: Iterable[StepRecord]) -> Iterator[RolloutRecord]:
    """Convert normalized Harbor steps into ordered agentic conversation parts."""
    next_turn: dict[tuple[str, str, str], int] = {}
    for step in steps:
        rollout_key = (step.task, step.doc_id, step.trial_id)
        turn_id = step.step_id if step.step_id is not None else next_turn.get(rollout_key, 0)
        next_turn[rollout_key] = turn_id + 1
        participant_type = _participant(step.source or "")
        participant_id = (
            step.model_name
            if participant_type is ParticipantType.ASSISTANT and step.model_name
            else step.source or "unknown"
        )
        parts = [
            (RolloutContentType.REASONING, step.reasoning_content, participant_type, participant_id),
            (RolloutContentType.MESSAGE, step.message, participant_type, participant_id),
            (RolloutContentType.TOOL_CALL, step.tool_calls_json, participant_type, participant_id),
            (RolloutContentType.TOOL_RESULT, step.observation_json, ParticipantType.ENVIRONMENT, "environment"),
        ]
        model_metrics_written = False
        part_id = 0
        for content_type, content, part_participant, part_participant_id in parts:
            if content is None:
                continue
            write_metrics = participant_type is ParticipantType.ASSISTANT and not model_metrics_written
            if write_metrics:
                model_metrics_written = True
            yield RolloutRecord(
                task=step.task,
                doc_id=step.doc_id,
                trial_id=step.trial_id,
                turn_id=turn_id,
                part_id=part_id,
                conversation_type=ConversationType.AGENTIC,
                participant_type=part_participant,
                participant_id=part_participant_id,
                content_type=content_type,
                content=content,
                prompt_tokens=step.prompt_tokens if write_metrics else None,
                completion_tokens=step.completion_tokens if write_metrics else None,
                cost_usd=step.cost_usd if write_metrics else None,
                prompt_token_ids=step.prompt_token_ids if write_metrics else None,
                completion_token_ids=step.completion_token_ids if write_metrics else None,
                logprobs=step.logprobs if write_metrics else None,
            )
            part_id += 1


def _unique_sample_attempts(reader: ReadView) -> Iterator[tuple[EvalSample, str]]:
    previous_key: tuple[str, str, str] | None = None
    for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE):
        key = (row["task"], row["doc_id"], row.get("trial_id") or "")
        if key == previous_key:
            continue
        previous_key = key
        yield sample_from_archive_row(row), key[2]


def _step_rows(reader: ReadView) -> Iterator[StepRecord]:
    fields = tuple(field.name for field in dataclasses.fields(StepRecord))
    for row in reader.iter_rows(ARCHIVE_STEPS_TABLE):
        yield StepRecord(**{name: row.get(name) for name in fields})


def _source_snapshot(reader: ReadView) -> dict[str, int]:
    return {table: reader.max_seq(table) for table in _SOURCE_TABLES}


def normalize_rollouts(root: str, *, writer_id: str) -> None:
    """Idempotently derive the shared ``rollouts`` table from an evaluation archive."""
    reader = ReadView(root)
    source_snapshot = _source_snapshot(reader)
    complete = reader.read_blob(_COMPLETE_OBJECT)
    if complete is not None and json.loads(complete) == {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "source_max_seq": source_snapshot,
    }:
        return
    records = itertools.chain(
        (
            record
            for sample, trial_id in _unique_sample_attempts(reader)
            for record in rollout_records_from_sample(sample, trial_id=trial_id)
        ),
        rollout_records_from_steps(_step_rows(reader)),
    )
    first = next(records, None)
    if first is None:
        return
    with EvaluationStore.open(root, writer_id=writer_id) as store:
        store.add_rollouts((first,))
        for record in records:
            store.add_rollouts((record,))
        store.add_artifact(
            _COMPLETE_OBJECT,
            json.dumps(
                {
                    "schema_version": ROLLOUT_SCHEMA_VERSION,
                    "source_max_seq": source_snapshot,
                },
                sort_keys=True,
            ).encode(),
            metadata={"content_type": "application/json"},
        )
        store.seal()
