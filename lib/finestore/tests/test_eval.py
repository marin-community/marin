# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the evaluation archive contract shipped in marin-finestore."""

import pytest
from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    EvalSample,
    EvaluationStore,
    SampleKind,
    SourceArtifact,
    StepRecord,
    sample_from_archive_row,
)
from finestore.reader import ReadView
from finestore.store import DataStore
from rigging.filesystem.conditional_object import LocalConditionalObject


@pytest.mark.parametrize("trajectory_padding", [0, 128 * 1024])
def test_source_pair_commit_failure_publishes_neither_file_and_retry_commits_metadata(
    tmp_path, monkeypatch, trajectory_padding
):
    root = str(tmp_path / "archive")
    sources = [
        SourceArtifact("attempt/result.json", b'{"reward": 1}', "application/json", {"trajectory_present": True}),
        SourceArtifact("attempt/trajectory.json", b'{"steps": []}' + b" " * trajectory_padding, "application/json"),
    ]
    # A native pair larger than the ordinary buffer must retain one commit,
    # including when the trajectory uses FineStore's chunked blob encoding.
    with EvaluationStore(DataStore.open(root, writer_id="harbor", max_buffer_bytes=1024)) as store:
        original = ReadView(root)

        def fail_commit(self, data, *, expected_version):
            raise OSError("interrupted HEAD publication")

        with monkeypatch.context() as failure:
            failure.setattr(LocalConditionalObject, "write", fail_commit)
            with pytest.raises(OSError, match="interrupted HEAD publication"):
                store.add_source_artifacts(sources)
        assert ReadView(root).read_blobs(["sources/attempt/result.json", "sources/attempt/trajectory.json"]) == {}
        store.add_source_artifacts(sources)
        committed = ReadView(root)
        assert committed.read_blobs(["sources/attempt/result.json", "sources/attempt/trajectory.json"]) == {
            "sources/attempt/result.json": sources[0].raw,
            "sources/attempt/trajectory.json": sources[1].raw,
        }
        assert committed.blob_metadata("sources/attempt/result.json")["trajectory_present"] is True
        assert original.blob_metadata("sources/attempt/result.json") is None
        store.add_source_artifacts([SourceArtifact("attempt/result.json", b"new", "text/plain")])
        assert committed.blob_metadata("sources/attempt/result.json")["trajectory_present"] is True
        assert "trajectory_present" not in ReadView(root).blob_metadata("sources/attempt/result.json")


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
