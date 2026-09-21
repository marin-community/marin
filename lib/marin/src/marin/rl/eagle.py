# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable rollout corpora for offline EAGLE draft distillation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import fsspec
from finestore.eval import ARCHIVE_ROLLOUTS_TABLE, ConversationType, ParticipantType
from finestore.reader import ReadView
from rigging.filesystem.storage_path import prefix_join

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext

_CORPUS_FILENAME = "corpus.jsonl"


@dataclass(frozen=True)
class EagleReplayMessage:
    """One provider-neutral message used to render a replay sequence."""

    role: str
    content: str


@dataclass(frozen=True)
class EagleReplayRecord:
    """One assistant response, represented as exact tokens or text."""

    group_id: str
    prompt: str | tuple[EagleReplayMessage, ...] | None = None
    response: str | None = None
    prompt_token_ids: tuple[int, ...] | None = None
    response_token_ids: tuple[int, ...] | None = None

    def to_json(self) -> str:
        value = asdict(self)
        return json.dumps({key: item for key, item in value.items() if item is not None}, separators=(",", ":"))


@dataclass(frozen=True)
class EagleRolloutCorpusConfig:
    """Resolved evaluation archives and output path for corpus extraction."""

    source_archives: tuple[str, ...]
    output_path: str


class EagleRolloutCorpus(Artifact):
    """A durable, target-independent prompt/response corpus."""

    uri: str
    sequences: int
    exact_token_sequences: int
    source_archives: tuple[str, ...]


def _rollout_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["task"]), str(row["doc_id"]), str(row.get("trial_id") or "")


def _group_id(key: tuple[str, str, str]) -> str:
    return "/".join(key)


def _role(participant: str) -> str:
    if participant == ParticipantType.SYSTEM:
        return "system"
    if participant == ParticipantType.USER:
        return "user"
    if participant == ParticipantType.ASSISTANT:
        return "assistant"
    return "tool"


def _turn_text(rows: Sequence[Mapping[str, Any]], participant: str) -> str:
    content = []
    for row in rows:
        if row.get("participant_type") != participant:
            continue
        value = row.get("content")
        if isinstance(value, str) and value:
            content.append(value)
    return "\n".join(content)


def _exact_record(rows: Sequence[Mapping[str, Any]], group_id: str) -> EagleReplayRecord | None:
    for row in rows:
        if row.get("participant_type") != ParticipantType.ASSISTANT:
            continue
        prompt_ids = row.get("prompt_token_ids")
        completion_ids = row.get("completion_token_ids")
        if prompt_ids is None and completion_ids is None:
            continue
        if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
            raise ValueError("rollout rows must carry prompt and completion token IDs together")
        if not prompt_ids or not completion_ids:
            return None
        return EagleReplayRecord(
            group_id=group_id,
            prompt_token_ids=tuple(int(token) for token in prompt_ids),
            response_token_ids=tuple(int(token) for token in completion_ids),
        )
    return None


def _records_for_rollout(rows: Sequence[Mapping[str, Any]]) -> Iterator[EagleReplayRecord]:
    key = _rollout_key(rows[0])
    group_id = _group_id(key)
    conversation_type = str(rows[0]["conversation_type"])
    history: list[EagleReplayMessage] = []
    turns: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        turns.setdefault(int(row["turn_id"]), []).append(row)

    for turn_rows in turns.values():
        exact = _exact_record(turn_rows, group_id)
        assistant_text = _turn_text(turn_rows, ParticipantType.ASSISTANT)
        if exact is not None:
            yield exact
        elif assistant_text and history:
            prompt: str | tuple[EagleReplayMessage, ...]
            if conversation_type == ConversationType.COMPLETION:
                prompt = "\n".join(message.content for message in history)
            else:
                prompt = tuple(history)
            yield EagleReplayRecord(group_id=group_id, prompt=prompt, response=assistant_text)

        for participant in (
            ParticipantType.SYSTEM,
            ParticipantType.USER,
            ParticipantType.ASSISTANT,
            ParticipantType.TOOL,
            ParticipantType.ENVIRONMENT,
            ParticipantType.OTHER,
        ):
            text = _turn_text(turn_rows, participant)
            if text:
                history.append(EagleReplayMessage(role=_role(participant), content=text))


def eagle_replay_records(rows: Iterable[Mapping[str, Any]]) -> Iterator[EagleReplayRecord]:
    """Project primary-key-ordered ``rollouts_v1`` rows into replay records."""
    current_key: tuple[str, str, str] | None = None
    current_rows: list[Mapping[str, Any]] = []
    for row in rows:
        key = _rollout_key(row)
        if current_key is not None and key != current_key:
            yield from _records_for_rollout(current_rows)
            current_rows = []
        current_key = key
        current_rows.append(row)
    if current_rows:
        yield from _records_for_rollout(current_rows)


def build_eagle_rollout_corpus(config: EagleRolloutCorpusConfig) -> EagleRolloutCorpus:
    """Extract one JSONL corpus from normalized Evalchemy or Harbor archives."""
    destination = prefix_join(config.output_path, _CORPUS_FILENAME)
    sequences = 0
    exact_sequences = 0
    with fsspec.open(destination, "wt") as output:
        for archive in config.source_archives:
            reader = ReadView(archive)
            if not reader.list_shards(ARCHIVE_ROLLOUTS_TABLE):
                raise ValueError(f"evaluation archive has no {ARCHIVE_ROLLOUTS_TABLE} table: {archive}")
            for record in eagle_replay_records(reader.iter_rows(ARCHIVE_ROLLOUTS_TABLE)):
                output.write(record.to_json())
                output.write("\n")
                sequences += 1
                exact_sequences += int(record.prompt_token_ids is not None)
    if sequences == 0:
        raise ValueError("evaluation archives contain no assistant responses for EAGLE replay")
    return EagleRolloutCorpus(
        path=config.output_path,
        uri=destination,
        sequences=sequences,
        exact_token_sequences=exact_sequences,
        source_archives=config.source_archives,
    )


def eagle_rollout_corpus_step(
    *,
    name: str,
    version: str,
    evaluations: tuple[ArtifactStep, ...],
) -> ArtifactStep[EagleRolloutCorpus]:
    """Build a target-independent replay corpus from normalized evaluations."""

    def build_config(ctx: StepContext) -> EagleRolloutCorpusConfig:
        return EagleRolloutCorpusConfig(
            source_archives=tuple(ctx.artifact_path(evaluation) for evaluation in evaluations),
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=EagleRolloutCorpus,
        run=build_eagle_rollout_corpus,
        build_config=build_config,
        deps=evaluations,
    )
