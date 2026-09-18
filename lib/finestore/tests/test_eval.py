# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the evaluation archive contract shipped in marin-finestore."""

from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    EvalSample,
    EvaluationStore,
    SampleKind,
    StepRecord,
    sample_from_archive_row,
)
from finestore.reader import ReadView


def test_evaluation_store_round_trips_a_normalized_sample(tmp_path):
    root = str(tmp_path / "archive")
    with EvaluationStore.open(root, writer_id="evalchemy") as store:
        store.add_sample(
            EvalSample(
                task="gsm8k",
                doc_id="7",
                kind=SampleKind.GENERATION,
                prompt_text="2 + 2?",
                output="4",
                extracted="4",
                metrics={"exact_match": 1.0},
                correct=True,
            )
        )
        store.seal()

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    [row] = table.to_pylist(maps_as_pydicts="strict")
    sample = sample_from_archive_row(row)
    assert (sample.task, sample.doc_id, sample.output, sample.correct) == ("gsm8k", "7", "4", True)


def test_evaluation_store_round_trips_normalized_steps_and_artifacts(tmp_path):
    root = str(tmp_path / "archive")
    step = StepRecord(
        task="aime",
        doc_id="problem-1",
        trial_id="trial-1",
        step_id=0,
        source="agent",
        model_name="model",
        message="answer",
        reasoning_content=None,
        tool_calls_json=None,
        observation_json=None,
        prompt_tokens=4,
        completion_tokens=1,
        cost_usd=None,
        prompt_token_ids=[1, 2, 3, 4],
        completion_token_ids=[5],
        logprobs=[-0.1],
    )
    with EvaluationStore.open(root, writer_id="harbor") as store:
        uri = store.add_artifact("trial-1/trajectory.json", b'{"steps": []}', metadata={"trial": "trial-1"})
        store.add_steps([step])
        store.seal()

    reader = ReadView(root)
    table = reader.scan(ARCHIVE_STEPS_TABLE)
    assert uri.endswith("/trial-1/trajectory.json")
    assert reader.read_blob("trial-1/trajectory.json") == b'{"steps": []}'
    assert table is not None
    [row] = table.to_pylist()
    assert (row["trial_id"], row["message"], row["completion_token_ids"]) == ("trial-1", "answer", [5])
