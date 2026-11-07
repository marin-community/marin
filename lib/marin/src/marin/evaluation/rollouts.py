# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the shared FineStore rollout table from normalized evaluation rows."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator

from finestore.admin import drop_table
from finestore.eval import (
    ARCHIVE_ROLLOUTS_TABLE,
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    ROLLOUT_SCHEMA_VERSION,
    ROLLOUTS_MERGE_KEY,
    ConversationType,
    EvalSample,
    ParticipantType,
    RolloutContentType,
    RolloutRecord,
    StepRecord,
    rollouts_schema,
    sample_from_archive_row,
)
from finestore.eval_rollouts import participant_type_for_role, rollout_records_from_sample
from finestore.reader import ReadView
from finestore.store import DataStore

_COMPLETE_OBJECT = f"_marin/rollouts-v{ROLLOUT_SCHEMA_VERSION}-complete.json"
_SOURCE_TABLES = (ARCHIVE_SAMPLES_TABLE, ARCHIVE_STEPS_TABLE)


def rollout_records_from_steps(steps: Iterable[StepRecord]) -> Iterator[RolloutRecord]:
    """Convert normalized Harbor steps into ordered agentic conversation parts."""
    next_turn: dict[tuple[str, str, str], int] = {}
    for step in steps:
        rollout_key = (step.task, step.doc_id, step.trial_id)
        turn_id = step.step_id if step.step_id is not None else next_turn.get(rollout_key, 0)
        next_turn[rollout_key] = turn_id + 1
        participant_type = participant_type_for_role(step.source or "")
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


def _source_fingerprints(reader: ReadView) -> dict[str, str]:
    fingerprints = {}
    for table in _SOURCE_TABLES:
        source = {
            "metadata_path": reader.table_metadata_path(table),
            # Appends and rebuilt tables get fresh row commits; compaction
            # preserves them even when sealing rewrites the physical shards.
            "row_commit_upper_bound": max(
                (shard.row_commit_upper_bound for shard in reader.list_shards(table)), default=0
            ),
        }
        fingerprints[table] = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return fingerprints


def normalize_rollouts(root: str, *, writer_id: str) -> None:
    """Idempotently derive the shared ``rollouts`` table from an evaluation archive."""
    reader = ReadView(root)
    completion = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "source_fingerprints": _source_fingerprints(reader),
    }
    complete = reader.read_blob(_COMPLETE_OBJECT)
    if complete is not None and json.loads(complete) == completion:
        return
    records = itertools.chain(
        (
            record
            for sample, trial_id in _unique_sample_attempts(reader)
            for record in rollout_records_from_sample(sample, trial_id=trial_id)
        ),
        rollout_records_from_steps(_step_rows(reader)),
    )
    # The evaluator owns its source tables and their conflict policy; register only our output.
    with DataStore.open(root, writer_id=writer_id) as store:
        if ARCHIVE_ROLLOUTS_TABLE in reader.table_names():
            drop_table(root, ARCHIVE_ROLLOUTS_TABLE)
        table = store.table(
            ARCHIVE_ROLLOUTS_TABLE,
            schema=rollouts_schema(),
            primary_key=ROLLOUTS_MERGE_KEY,
            schema_version=ROLLOUT_SCHEMA_VERSION,
        )
        for record in records:
            table.append(record.model_dump(mode="json"))
        store.write_object(
            _COMPLETE_OBJECT,
            json.dumps(completion, sort_keys=True).encode(),
            metadata={"content_type": "application/json"},
        )
        store.seal()
