# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for normalized evaluation rollouts."""

import json

from finestore.eval import (
    ARCHIVE_ROLLOUTS_TABLE,
    ROLLOUT_SCHEMA_VERSION,
    ROLLOUTS_MERGE_KEY,
    Choice,
    EvalSample,
    EvaluationStore,
    Grading,
    Message,
    SampleKind,
    StepRecord,
)
from finestore.reader import ReadView
from marin.evaluation.rollouts import normalize_rollouts


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
