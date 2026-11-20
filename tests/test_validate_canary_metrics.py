# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.ci.validate_canary_metrics import check_wandb_history, lookup_metric, main, read_summary

HEALTHY_SUMMARY = {
    "train": {"loss": 5.5},
    "_step": 238,
    "_runtime": 2400,
}


def test_lookup_metric():
    assert lookup_metric({"train": {"loss": 5.5}}, "train/loss") == 5.5
    assert lookup_metric({"train/loss": 5.5}, "train/loss") == 5.5
    assert lookup_metric({"_step": 476}, "_step") == 476
    assert lookup_metric({}, "train/loss") is None


def test_main_exits_nonzero_on_failure():
    bad_summary = {**HEALTHY_SUMMARY, "train": {"loss": 10.0}}
    with (
        patch("scripts.ci.validate_canary_metrics.resolve_canary_output_path", return_value="gs://fake"),
        patch("scripts.ci.validate_canary_metrics.read_summary", return_value=bad_summary),
        pytest.raises(SystemExit, match="1"),
    ):
        main()


def test_read_summary(tmp_path):
    summary = {"train": {"loss": 5.5}}
    (tmp_path / "tracker_metrics.jsonl").write_text(json.dumps({"config": {}, "summary": summary}))
    assert read_summary(str(tmp_path)) == summary


def test_check_wandb_history_skipped_outside_ci(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert check_wandb_history(40) is None


@pytest.mark.parametrize(
    ("state", "last_step", "ok"),
    [
        ("finished", 49, True),
        ("running", 49, True),
        ("running", None, False),  # run created, history never uploaded
        ("crashed", 13, False),  # upload stalled mid-run
    ],
)
def test_check_wandb_history(monkeypatch, state, last_step, ok):
    monkeypatch.setenv("RUN_ID", "canary-gpu-1-1")
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    monkeypatch.setattr("scripts.ci.validate_canary_metrics.WANDB_HISTORY_WAIT_SECONDS", 0)
    run = SimpleNamespace(state=state, lastHistoryStep=last_step)
    with patch("scripts.ci.validate_canary_metrics.wandb.Api") as api:
        api.return_value.run.return_value = run
        error = check_wandb_history(40)
    assert (error is None) == ok
