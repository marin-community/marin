# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for normalized evaluation rollouts."""

import json
from dataclasses import replace

import pytest
from finestore.admin import drop_table
from finestore.eval import (
    ARCHIVE_ROLLOUTS_TABLE,
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    ROLLOUT_SCHEMA_VERSION,
    ROLLOUTS_MERGE_KEY,
    SAMPLES_MERGE_KEY,
    SCHEMA_VERSION,
    Choice,
    EvalSample,
    EvaluationStore,
    Grading,
    Message,
    SampleKind,
    StepRecord,
    sample_to_archive_row,
    samples_schema,
)
from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.store import DataStore
from marin.evaluation.rollouts import normalize_rollouts


def test_rollouts_preserve_evalchemy_superseding_sample_table(tmp_path):
    root = str(tmp_path / "evalchemy")
    sample = EvalSample(task="gsm8k", doc_id="0", kind=SampleKind.GENERATION, prompt_text="2+2?", output="4")
    with DataStore.open(root, writer_id="evalchemy") as store:
        table = store.table(
            ARCHIVE_SAMPLES_TABLE,
            primary_key=SAMPLES_MERGE_KEY,
            schema=samples_schema(),
            schema_version=SCHEMA_VERSION,
            on_conflict=OnConflict.SUPERSEDE,
        )
        table.append(sample_to_archive_row(sample))
        store.seal()
    original = list(ReadView(root).iter_rows(ARCHIVE_SAMPLES_TABLE))

    normalize_rollouts(root, writer_id="marin-rollouts")

    reader = ReadView(root)
    assert list(reader.iter_rows(ARCHIVE_SAMPLES_TABLE)) == original
    assert [row["content"] for row in reader.iter_rows(ARCHIVE_ROLLOUTS_TABLE)] == ["2+2?", "4"]


def test_evalchemy_samples_normalize_to_one_conversation_across_grading_filters(tmp_path):
    root = str(tmp_path / "evalchemy")
    prompt = [Message(role="system", content="Answer briefly."), Message(role="user", content="2 + 2?")]
    with EvaluationStore.open(root, writer_id="evalchemy") as store:
        for extraction_filter, score in (("strict-match", 0.0), ("flexible-extract", 1.0)):
            store.add_sample(
                EvalSample(
                    task="gsm8k_5shot",
                    doc_id="7",
                    kind=SampleKind.GENERATION,
                    prompt_messages=prompt,
                    output="4",
                    grading=Grading(
                        method="lm-eval:exact_match",
                        metric="exact_match",
                        filter=extraction_filter,
                        score=score,
                        passed=bool(score),
                    ),
                )
            )
        store.add_source_artifact(
            "evalchemy/gsm8k/native/samples.jsonl",
            b'{"doc_id": 7}\n',
            content_type="application/x-ndjson",
        )
        store.seal()

    normalize_rollouts(root, writer_id="marin-rollouts")

    reader = ReadView(root)
    rows = reader.scan(ARCHIVE_ROLLOUTS_TABLE).to_pylist()
    assert reader.primary_key(ARCHIVE_ROLLOUTS_TABLE) == ROLLOUTS_MERGE_KEY
    assert reader.schema_version(ARCHIVE_ROLLOUTS_TABLE) == ROLLOUT_SCHEMA_VERSION
    assert [(row["turn_id"], row["participant_type"], row["participant_id"], row["content"]) for row in rows] == [
        (0, "system", "system", "Answer briefly."),
        (1, "user", "user", "2 + 2?"),
        (2, "assistant", "assistant", "4"),
    ]
    assert {row["conversation_type"] for row in rows} == {"chat"}
    assert reader.read_blob("sources/evalchemy/gsm8k/native/samples.jsonl") == b'{"doc_id": 7}\n'

    normalize_rollouts(root, writer_id="marin-rollouts-retry")
    assert ReadView(root).scan(ARCHIVE_ROLLOUTS_TABLE).num_rows == 3

    with EvaluationStore.open(root, writer_id="evalchemy-resume") as store:
        store.add_sample(
            EvalSample(
                task="gsm8k_5shot",
                doc_id="8",
                kind=SampleKind.GENERATION,
                prompt_text="3 + 3?",
                output="6",
            )
        )
        store.seal()
    normalize_rollouts(root, writer_id="marin-rollouts-resume")
    assert ReadView(root).scan(ARCHIVE_ROLLOUTS_TABLE).num_rows == 5


def test_harbor_steps_normalize_message_parts_and_token_data(tmp_path):
    root = str(tmp_path / "harbor")
    step = StepRecord(
        task="aime",
        doc_id="problem-1",
        trial_id="trial-1",
        step_id=2,
        source="agent",
        model_name="served-model",
        message="I will calculate it.",
        reasoning_content="Need solve.",
        tool_calls_json=json.dumps([{"tool_call_id": "call-1", "function_name": "python"}]),
        observation_json=json.dumps({"results": [{"source_call_id": "call-1", "content": "42"}]}),
        prompt_tokens=8,
        completion_tokens=4,
        cost_usd=0.01,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        logprobs=[-0.1, -0.2],
    )
    with EvaluationStore.open(root, writer_id="harbor") as store:
        store.add_sample(
            EvalSample(
                task="aime",
                doc_id="problem-1",
                kind=SampleKind.AGENTIC,
                trajectory_uri="finestore://blobs/trial-1/trajectory.json",
            ),
            trial_id="trial-1",
        )
        store.add_steps([step])
        store.add_artifact("trial-1/trajectory.json", b'{"steps": []}')
        store.seal()

    normalize_rollouts(root, writer_id="marin-rollouts")

    reader = ReadView(root)
    rows = reader.scan(ARCHIVE_ROLLOUTS_TABLE).to_pylist()
    assert [(row["part_id"], row["participant_type"], row["content_type"]) for row in rows] == [
        (0, "assistant", "reasoning"),
        (1, "assistant", "message"),
        (2, "assistant", "tool_call"),
        (3, "environment", "tool_result"),
    ]
    assert all(row["conversation_type"] == "agentic" for row in rows)
    assert rows[0]["participant_id"] == "served-model"
    assert rows[0]["completion_token_ids"] == [3, 4]
    assert all(row["completion_token_ids"] is None for row in rows[1:])
    assert reader.read_blob("trial-1/trajectory.json") == b'{"steps": []}'


def test_multiple_choice_rollout_records_the_selected_choice(tmp_path):
    root = str(tmp_path / "multiple-choice")
    with EvaluationStore.open(root, writer_id="evalchemy") as store:
        store.add_sample(
            EvalSample(
                task="arc",
                doc_id="2",
                kind=SampleKind.MULTIPLE_CHOICE,
                prompt_text="Choose one.",
                choices=[Choice(label="A", text="first"), Choice(label="B", text="second")],
                model_choice=1,
            )
        )
        store.seal()

    normalize_rollouts(root, writer_id="marin-rollouts")

    rows = ReadView(root).scan(ARCHIVE_ROLLOUTS_TABLE).to_pylist()
    assert {row["conversation_type"] for row in rows} == {"completion"}
    assert rows[1]["content"] == "second"
    assert json.loads(rows[1]["metadata_json"]) == {"choice_index": 1, "choice_label": "B"}


def test_rollouts_replace_same_count_source_shards_and_remove_old_conversations(tmp_path):
    root = str(tmp_path / "archive")
    for iteration, doc_id, output in ((0, "old", "first"), (1, "replacement", "second")):
        if iteration:
            drop_table(root, ARCHIVE_SAMPLES_TABLE)
        with EvaluationStore.open(root, writer_id=f"source-{iteration}") as store:
            store.add_sample(
                EvalSample(task="qa", doc_id=doc_id, kind=SampleKind.GENERATION, prompt_text="Question?", output=output)
            )
            store.add_source_artifact("native/result.json", b'{"preserved":true}', content_type="application/json")
            store.seal()
        normalize_rollouts(root, writer_id=f"rollouts-{iteration}")
        reader = ReadView(root)
        assert [(row["doc_id"], row["content"]) for row in reader.iter_rows(ARCHIVE_ROLLOUTS_TABLE)] == [
            (doc_id, "Question?"),
            (doc_id, output),
        ]
        assert reader.read_blob("sources/native/result.json") == b'{"preserved":true}'

    completed = ReadView(root).token
    normalize_rollouts(root, writer_id="unchanged")
    assert ReadView(root).token == completed


@pytest.fixture
def agent_step():
    return StepRecord(
        task="qa",
        doc_id="same",
        trial_id="same",
        step_id=1,
        source="agent",
        model_name="model",
        message="old answer",
        reasoning_content=None,
        tool_calls_json=None,
        observation_json=None,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=None,
        prompt_token_ids=None,
        completion_token_ids=None,
        logprobs=None,
    )


def test_rollouts_clear_old_parts_after_same_count_step_replacement(tmp_path, agent_step):
    root = str(tmp_path / "archive")
    for iteration, reasoning in ((0, "Old reasoning"), (1, None)):
        if iteration:
            drop_table(root, ARCHIVE_STEPS_TABLE)
        with EvaluationStore.open(root, writer_id=f"source-{iteration}") as store:
            store.add_steps([replace(agent_step, message=f"answer-{iteration}", reasoning_content=reasoning)])
            store.seal()
        normalize_rollouts(root, writer_id=f"rollouts-{iteration}")

    assert [(row["content_type"], row["content"]) for row in ReadView(root).iter_rows(ARCHIVE_ROLLOUTS_TABLE)] == [
        ("message", "answer-1"),
    ]


def test_rollouts_record_empty_replacement_and_clear_stale_rows(tmp_path, agent_step):
    root = str(tmp_path / "archive")
    with EvaluationStore.open(root, writer_id="source") as store:
        store.add_steps([agent_step])
        store.seal()
    normalize_rollouts(root, writer_id="initial-rollouts")
    drop_table(root, ARCHIVE_STEPS_TABLE)

    normalize_rollouts(root, writer_id="empty-rollouts")

    reader = ReadView(root)
    assert list(reader.iter_rows(ARCHIVE_ROLLOUTS_TABLE)) == []
    assert reader.is_sealed()
    completed = reader.token
    normalize_rollouts(root, writer_id="empty-rollouts-retry")
    assert ReadView(root).token == completed
