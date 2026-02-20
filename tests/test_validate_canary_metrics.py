# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest

from scripts.canary.validate_canary_metrics import lookup_metric, main, read_summary, validate_metrics

HEALTHY_SUMMARY = {
    "train": {"loss": 3.75},
    "throughput": {"mfu": 30.0},
    "_step": 3814,
    "_runtime": 2400,
}


def test_lookup_metric():
    assert lookup_metric({"train": {"loss": 3.75}}, "train/loss") == 3.75
    assert lookup_metric({"train/loss": 3.75}, "train/loss") == 3.75
    assert lookup_metric({"_step": 3814}, "_step") == 3814
    assert lookup_metric({}, "train/loss") is None


@pytest.mark.parametrize(
    "summary, expected_failures",
    [
        (HEALTHY_SUMMARY, []),
        ({**HEALTHY_SUMMARY, "train": {"loss": 4.5}}, ["Final loss"]),
        ({"train": {"loss": 3.75}}, ["MFU (%)", "Steps completed", "Wall-clock (s)"]),
    ],
    ids=["all_pass", "loss_regression", "missing_metrics"],
)
def test_validate_metrics(summary, expected_failures):
    results = validate_metrics(summary)
    failures = [name for name, _, _, passed in results if not passed]
    assert failures == expected_failures


def test_main_exits_nonzero_on_failure():
    bad_summary = {**HEALTHY_SUMMARY, "train": {"loss": 5.0}}
    with (
        patch("scripts.canary.validate_canary_metrics.resolve_canary_output_path", return_value="gs://fake"),
        patch("scripts.canary.validate_canary_metrics.read_summary", return_value=bad_summary),
        pytest.raises(SystemExit, match="1"),
    ):
        main()


def test_read_summary(tmp_path):
    summary = {"train": {"loss": 3.75}}
    (tmp_path / "tracker_metrics.jsonl").write_text(json.dumps({"config": {}, "summary": summary}))
    assert read_summary(str(tmp_path)) == summary
