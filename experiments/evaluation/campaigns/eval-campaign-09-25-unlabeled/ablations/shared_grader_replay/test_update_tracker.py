# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import fsspec
import pytest
import yaml

MODULE = "experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.shared_grader_replay.update_tracker"


@pytest.mark.parametrize(
    ("benchmark", "trial_count", "expected_count"),
    [("olympiadbench", 300, 300), ("olympiadbench", 299, 300), ("financebench", 50, 50)],
)
def test_subset_replay_preserves_other_scores_and_rejects_incomplete_results(
    tmp_path, monkeypatch, benchmark, trial_count, expected_count
):
    updater = importlib.import_module(MODULE)
    tracker = tmp_path / "TRACKER.md"
    source_cells = {
        "olympiadbench": "0.400 (s3://source/olympiad)",
        "financebench": "0.400 (s3://source/finance)",
    }
    source_uris = {benchmark: cell.split("(", 1)[1][:-1] for benchmark, cell in source_cells.items()}
    original = (
        "| Model | math500 | aime24 | olympiadbench | financebench |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Metric | accuracy | accuracy | accuracy | accuracy |\n"
        f"| org/target | 0.900 (s3://math/results) | 0.700 (s3://aime/results) "
        f"| {source_cells['olympiadbench']} | {source_cells['financebench']} |\n"
        "| org/unrelated | 0.800 (s3://other/math) | 0.600 (s3://other/aime) "
        "| 0.300 (s3://other/olympiad) | 0.200 (s3://other/finance) |\n"
    )
    tracker.write_text(original)
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "model": "org/target",
                        "benchmarks": {benchmark: {"tracker_score": 0.4, "source": source_uris[benchmark]}},
                    }
                ]
            }
        )
    )
    result_file = tmp_path / "replay.json"
    result_file.write_text(
        json.dumps(
            {
                "model": "org/target",
                "benchmark": benchmark,
                "source": source_uris[benchmark],
                "num_trials": trial_count,
                "num_judge_failed": 0,
                "corrected_score": 0.5,
            }
        )
    )
    monkeypatch.setattr(updater, "filesystem_for", lambda _uri: (fsspec.filesystem("file"), str(result_file)))
    if trial_count != expected_count:
        with pytest.raises(ValueError, match="incomplete trial count"):
            updater.update_tracker(tracker, sources, "s3://replay", (benchmark,))
        assert tracker.read_text() == original
        return

    changed = updater.update_tracker(tracker, sources, "s3://replay", (benchmark,))
    replacement = f"0.500 (s3://replay/results/{benchmark}/org--target.json)"
    assert changed == [updater.TrackerChange("org/target", benchmark, replacement)]
    assert tracker.read_text() == original.replace(source_cells[benchmark], replacement)
