"""Persisted output from an interrupted SkyRL evaluation."""

import json
from pathlib import Path
import shutil

from skyrl_train.partial_evaluation import PartialEvaluationRecorder


def test_interrupted_evaluation_keeps_scored_prefix_and_counts(tmp_path: Path) -> None:
    recorder = PartialEvaluationRecorder(
        str(tmp_path / "partial"), expected_rollouts=3, source="mini840", upload=shutil.copyfile
    )
    recorder.append([{"uid": "second", "score": 0.0, "data_source": "mini840"}], repetition=1)
    recorder.append([{"uid": "first", "score": 1.0, "data_source": "mini840"}], repetition=0)
    recorder.close()

    snapshots = [
        [json.loads(line) for line in path.read_text().splitlines()] for path in tmp_path.glob("partial-*.jsonl")
    ]
    assert len(snapshots) == 2
    latest = max(snapshots, key=lambda lines: lines[0]["completed_rollouts"])
    assert latest[0]["status"] == "partial"
    assert latest[0]["completed_rollouts"] == 2
    assert latest[0]["trajectory_rows"] == 2
    assert [(row["uid"], row["score"], row["repetition_id"]) for row in latest if "uid" in row] == [
        ("second", 0.0, 1),
        ("first", 1.0, 0),
    ]
    assert latest[-1] == {"record_type": "progress", "completed_rollouts": 2}


def test_retry_keeps_prior_attempt_snapshots(tmp_path: Path) -> None:
    prefix = str(tmp_path / "partial")
    first = PartialEvaluationRecorder(prefix, expected_rollouts=3, source="mini840", upload=shutil.copyfile)
    first.append([{"uid": "early", "score": 1.0}], repetition=0)
    first.close()
    prior = {path.name: path.read_bytes() for path in tmp_path.glob("partial-*.jsonl")}

    retry = PartialEvaluationRecorder(prefix, expected_rollouts=3, source="mini840", upload=shutil.copyfile)
    retry.append([{"uid": "retry", "score": 0.0}], repetition=0)
    retry.close()

    assert {path.name: path.read_bytes() for path in tmp_path.glob("partial-*.jsonl") if path.name in prior} == prior
    assert len(list(tmp_path.glob("partial-*.jsonl"))) > len(prior)
