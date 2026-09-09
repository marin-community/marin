# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from experiments.post_training.math_eval.calibration_attempts import claim_measurement
from experiments.post_training.math_eval.calibration_audit import validate_calibration_generation
from tests.rl.test_math_eval_calibration_audit import fixture


def retried_fixture():
    generation, tasks, job, expected = fixture()
    task = tasks["tasks"][0]
    previous = deepcopy(task["attempts"][0])
    previous.update(
        state="TASK_STATE_FAILED",
        exit_code=137,
        attempt_uid="startup-uid",
        started_at={"epoch_ms": "100"},
        finished_at={"epoch_ms": "600"},
    )
    task["attempts"][0]["attempt_id"] = 1
    task["attempts"].insert(0, previous)
    task["current_attempt_id"] = 1
    generation["attempt_id"] = 1
    generation["measurement_start"]["attempt_id"] = 1
    return generation, tasks, job, expected


def test_startup_retry_preserves_actual_cost_of_both_attempts():
    generation, tasks, job, expected = retried_fixture()
    result = validate_calibration_generation(generation, tasks, job, **expected)
    assert result["task_gpu_hours"] == 1 + 500 / 3_600_000
    assert [item["attempt_uid"] for item in result["attempts"]] == ["startup-uid", "fixture-uid"]
    assert result["attempts"][0]["scope"] == "startup before measurement"


@pytest.mark.parametrize(
    "poison", ["marker_attempt", "marker_late", "prior_success", "missing_attempt", "overlap", "uid", "duplicate_uid"]
)
def test_retry_cannot_hide_prior_measurement_or_missing_native_history(poison):
    generation, tasks, job, expected = retried_fixture()
    task = tasks["tasks"][0]
    if poison == "marker_attempt":
        generation["measurement_start"].update(attempt_id=0, attempt_uid="startup-uid")
    elif poison == "marker_late":
        generation["measurement_start"]["started_at_ms"] = 3000
    elif poison == "prior_success":
        task["attempts"][0]["state"] = "TASK_STATE_SUCCEEDED"
    elif poison == "missing_attempt":
        task["attempts"].pop(0)
    elif poison == "overlap":
        task["attempts"][0]["finished_at"]["epoch_ms"] = "1500"
    elif poison == "duplicate_uid":
        task["attempts"][0]["attempt_uid"] = task["attempts"][1]["attempt_uid"]
    else:
        generation["attempt_uid"] = "another"
    with pytest.raises(ValueError):
        validate_calibration_generation(generation, tasks, job, **expected)


def test_unknown_startup_interval_is_explicit_without_invented_total():
    generation, tasks, job, expected = retried_fixture()
    tasks["tasks"][0]["attempts"][0].pop("finished_at")
    result = validate_calibration_generation(generation, tasks, job, **expected)
    assert result["task_gpu_hours"] is None
    assert result["known_task_gpu_hours"] == 1
    assert result["attempts"][0]["task_h100_hours"] is None


def test_measurement_marker_blocks_retry_and_preserves_first_bytes(tmp_path):
    output = str(tmp_path / "calibration")
    kwargs = dict(
        task_id="/atqamar/fixture/0", attempt_id=0, attempt_uid="first", binding_sha256="a" * 64, source_commit="b" * 40
    )
    marker = claim_measurement(output, **kwargs)
    path = tmp_path / "calibration" / "measurement-start.json"
    raw = path.read_bytes()
    assert json.loads(raw) == marker
    with pytest.raises(ValueError, match="already exists"):
        claim_measurement(output, **(kwargs | {"attempt_id": 1, "attempt_uid": "retry"}))
    assert path.read_bytes() == raw
