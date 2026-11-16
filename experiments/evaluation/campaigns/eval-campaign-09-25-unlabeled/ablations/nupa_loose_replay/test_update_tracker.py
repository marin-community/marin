# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys

import fsspec
import pytest

MODULE = "experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.nupa_loose_replay.update_tracker"


@pytest.mark.parametrize("skip_special_tokens,promoted", [("false", True), ("true", False), (None, False)])
def test_promotion_preserves_unverified_thinking_cells_and_updates_other_models(
    tmp_path, monkeypatch, skip_special_tokens, promoted
):
    updater = importlib.import_module(MODULE)
    filesystem = fsspec.filesystem("memory")
    monkeypatch.setattr(updater, "filesystem_for", lambda uri: (filesystem, f"{tmp_path}/{uri.removeprefix('s3://')}"))
    tracker = tmp_path / "TRACKER.md"
    tracker.write_text(
        "| Model | NUPA |\n| --- | --- |\n"
        "| org/Grug-Snowball | 0.000 (s3://source/thinking/results) |\n"
        "| org/other | 0.000 (s3://source/other/results) |\n"
    )
    if skip_special_tokens is not None:
        filesystem.pipe(
            f"{tmp_path}/source/thinking/record.json",
            json.dumps(
                {"model": {"config": {"generation": {"extra_gen_kwargs": {"skip_special_tokens": skip_special_tokens}}}}}
            ).encode(),
        )
    for model, source in [("org/Grug-Snowball", "thinking"), ("org/other", "other")]:
        filesystem.pipe(
            f"{tmp_path}/replay/results/nupa/{model.replace('/', '--')}.json",
            json.dumps(
                {
                    "model": model,
                    "benchmark": "NUPA",
                    "source": f"s3://source/{source}/results",
                    "raw_source": f"s3://source/{source}/results",
                    "tracker_score": 0.0,
                    "num_trials": 2,
                    "corrected_score": 0.5,
                    "changed_trials": 1,
                    "trials": [{"corrected_correct": True}, {"corrected_correct": False}],
                }
            ).encode(),
        )

    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(updater, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_tracker",
            "--tracker",
            str(tracker),
            "--output-prefix",
            "s3://replay",
            "--lock",
            str(tmp_path / "promotion.lock"),
            "--summary",
            str(summary_path),
        ],
    )
    updater.main()
    summary = json.loads(summary_path.read_text())

    expected = {"org/other", "org/Grug-Snowball"} if promoted else {"org/other"}
    assert {change["model"] for change in summary["updates"]} == expected
    assert summary["regenerate"] == ([] if promoted else ["org/Grug-Snowball"])
    text = tracker.read_text()
    assert "| org/other | 0.500 (s3://replay/results/nupa/org--other.json) |" in text
    if promoted:
        assert "| org/Grug-Snowball | 0.500 (s3://replay/results/nupa/org--Grug-Snowball.json) |" in text
    else:
        assert "| org/Grug-Snowball | 0.000 (s3://source/thinking/results) |" in text
