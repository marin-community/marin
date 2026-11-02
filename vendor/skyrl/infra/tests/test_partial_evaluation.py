"""Persisted output from an interrupted SkyRL evaluation."""

import json
from pathlib import Path
import shutil

from skyrl_train.partial_evaluation import PartialEvaluationRecorder


def test_interrupted_evaluation_keeps_scored_prefix_and_counts(tmp_path: Path) -> None:
    targets = (str(tmp_path / "partial-0.jsonl"), str(tmp_path / "partial-1.jsonl"))
    recorder = PartialEvaluationRecorder(targets, expected_rollouts=3, source="mini840", upload=shutil.copyfile)
    recorder.append([{"uid": "second", "score": 0.0, "data_source": "mini840"}], repetition=1)
    recorder.append([{"uid": "first", "score": 1.0, "data_source": "mini840"}], repetition=0)
    recorder.close()

    snapshots = [
        [json.loads(line) for line in Path(target).read_text().splitlines()]
        for target in targets
        if Path(target).exists()
    ]
    assert len(snapshots) == 2
    latest = max(snapshots, key=lambda lines: lines[0]["completed_rollouts"])
    assert latest[0]["status"] == "partial"
    assert latest[0]["expected_rollouts"] == 3
    assert latest[0]["completed_rollouts"] == 2
    assert latest[0]["trajectory_rows"] == 2
    assert [(row["uid"], row["score"], row["repetition_id"]) for row in latest if "uid" in row] == [
        ("second", 0.0, 1),
        ("first", 1.0, 0),
    ]
    assert latest[-1] == {"record_type": "progress", "completed_rollouts": 2}
