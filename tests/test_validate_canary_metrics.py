# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest

from scripts.canary.validate_canary_metrics import lookup_metric, main, read_summary

HEALTHY_SUMMARY = {
    "train": {"loss": 5.5},
    "_step": 476,
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
        patch("scripts.canary.validate_canary_metrics.resolve_canary_output_path", return_value="gs://fake"),
        patch("scripts.canary.validate_canary_metrics.read_summary", return_value=bad_summary),
        pytest.raises(SystemExit, match="1"),
    ):
        main()


def test_read_summary(tmp_path):
    summary = {"train": {"loss": 5.5}}
    (tmp_path / "tracker_metrics.jsonl").write_text(json.dumps({"config": {}, "summary": summary}))
    assert read_summary(str(tmp_path)) == summary
