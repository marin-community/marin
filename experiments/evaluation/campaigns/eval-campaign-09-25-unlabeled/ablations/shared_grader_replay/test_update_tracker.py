# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import fsspec
import pytest
import yaml

MODULE = "experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.shared_grader_replay.update_tracker"


@pytest.mark.parametrize("trial_count", [300, 299])
def test_subset_replay_preserves_other_scores_and_rejects_incomplete_results(tmp_path, monkeypatch, trial_count):
    updater = importlib.import_module(MODULE)
    tracker = tmp_path / "TRACKER.md"
    original = (
        "| Model | math500 | aime24 | olympiadbench |\n"
        "| --- | --- | --- | --- |\n"
        "| Metric | accuracy | accuracy | accuracy |\n"
        "| org/target | 0.900 (s3://math/results) | 0.700 (s3://aime/results) | 0.400 (s3://source/results) |\n"
        "| org/unrelated | 0.800 (s3://other/math) | 0.600 (s3://other/aime) | 0.300 (s3://other/olympiad) |\n"
    )
    tracker.write_text(original)
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "model": "org/target",
                        "benchmarks": {"olympiadbench": {"tracker_score": 0.4, "source": "s3://source/results"}},
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
                "benchmark": "olympiadbench",
                "source": "s3://source/results",
                "num_trials": trial_count,
                "num_judge_failed": 0,
                "corrected_score": 0.5,
            }
        )
    )
    monkeypatch.setattr(updater, "filesystem_for", lambda _uri: (fsspec.filesystem("file"), str(result_file)))
    if trial_count != 300:
        with pytest.raises(ValueError, match="incomplete trial count"):
            updater.update_tracker(tracker, sources, "s3://replay", ("olympiadbench",))
        assert tracker.read_text() == original
        return

    changed = updater.update_tracker(tracker, sources, "s3://replay", ("olympiadbench",))
    replacement = "0.500 (s3://replay/results/olympiadbench/org--target.json)"
    assert changed == [("org/target", "olympiadbench", replacement)]
    assert tracker.read_text() == original.replace("0.400 (s3://source/results)", replacement)
