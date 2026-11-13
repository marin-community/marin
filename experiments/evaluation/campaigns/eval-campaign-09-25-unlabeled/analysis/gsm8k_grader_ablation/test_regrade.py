# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("regrade.py")
SPEC = importlib.util.spec_from_file_location("gsm8k_grader_ablation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
regrade = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = regrade
SPEC.loader.exec_module(regrade)

PROMOTE_PATH = Path(__file__).with_name("promote.py")
PROMOTE_SPEC = importlib.util.spec_from_file_location("gsm8k_grader_ablation_promote", PROMOTE_PATH)
assert PROMOTE_SPEC is not None and PROMOTE_SPEC.loader is not None
promote = importlib.util.module_from_spec(PROMOTE_SPEC)
sys.modules[PROMOTE_SPEC.name] = promote
PROMOTE_SPEC.loader.exec_module(promote)


def test_tracker_cells_selects_only_sealed_gsm8k_scores():
    tracker = """\
| Model | math500 | gsm8k-0shot |
| --- | --- | --- |
| Metric | accuracy | accuracy |
| org/model-a | 0.500 (s3://bucket/math-a/results) | 0.750 (s3://bucket/gsm-a/results) |
| org/model-b | 0.400 (s3://bucket/math-b/results) | RUNNING |
| org/model-c | 0.300 (s3://bucket/math-c/results) | 0.625 (s3://bucket/gsm-c/results) |
"""

    assert regrade.tracker_cells(tracker) == [
        regrade.TrackerCell("org/model-a", 0.75, "s3://bucket/gsm-a/results"),
        regrade.TrackerCell("org/model-c", 0.625, "s3://bucket/gsm-c/results"),
    ]


def test_tracker_cells_rejects_duplicate_models():
    tracker = """\
| Model | gsm8k-0shot |
| --- | --- |
| Metric | accuracy |
| org/model | 0.750 (s3://bucket/a/results) |
| org/model | 0.625 (s3://bucket/b/results) |
"""

    with pytest.raises(ValueError, match="duplicate"):
        regrade.tracker_cells(tracker)


def test_update_tracker_replaces_only_matching_gsm8k_cell():
    tracker = """\
| Model | math500 | gsm8k-0shot |
| --- | --- | --- |
| Metric | accuracy | accuracy |
| org/model | 0.500 (s3://bucket/math/results) | 0.750 (s3://bucket/gsm/results) |
"""
    results = [
        {
            "model": "org/model",
            "tracker_score": 0.750,
            "prior_score": 0.7501,
            "source": "s3://bucket/gsm/results",
            "corrected_score": 0.8125,
            "regrade_result": "s3://bucket/regrade/model.json",
        }
    ]

    updated = promote.update_tracker(tracker, results)

    assert "0.500 (s3://bucket/math/results)" in updated
    assert "0.812 (s3://bucket/regrade/model.json)" in updated


def test_update_tracker_rejects_stale_source():
    tracker = """\
| Model | gsm8k-0shot |
| --- | --- |
| Metric | accuracy |
| org/model | 0.750 (s3://bucket/new/results) |
"""
    results = [
        {
            "model": "org/model",
            "tracker_score": 0.75,
            "prior_score": 0.7501,
            "source": "s3://bucket/old/results",
            "corrected_score": 0.8,
            "regrade_result": "s3://bucket/regrade/model.json",
        }
    ]

    with pytest.raises(ValueError, match="source changed"):
        promote.update_tracker(tracker, results)
