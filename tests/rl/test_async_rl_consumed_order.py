# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from experiments.post_training.async_rl_audit import consumed_order_comparison, validate_sync_in_async_controls


def test_consumed_order_requires_complete_matching_digests_and_observed_zero_rejections():
    reference = {
        "history": {
            "consumed_uid_digests": {"1": 10, "2": 20},
            "ranges": {"sync/admission/rejected_count": {"count": 2, "max": 0}},
        }
    }
    candidate = {
        "history": {
            "consumed_uid_digests": {"1": 10, "2": 20},
            "ranges": {"async/rejected_count": {"count": 2, "max": 0}},
        }
    }
    assert consumed_order_comparison(reference, candidate, 2)["verified"]
    changed = deepcopy(candidate)
    changed["history"]["consumed_uid_digests"]["2"] = 21
    assert not consumed_order_comparison(reference, changed, 2)["verified"]
    assert consumed_order_comparison(reference, changed, 2)["matched_steps"] == [1]
    missing = deepcopy(candidate)
    missing["history"]["consumed_uid_digests"].pop("2")
    assert not consumed_order_comparison(reference, missing, 2)["verified"]
    for count, maximum in ((1, 0), (2, 1)):
        rejected = deepcopy(candidate)
        rejected["history"]["ranges"]["async/rejected_count"] = {"count": count, "max": maximum}
        assert not consumed_order_comparison(reference, rejected, 2)["verified"]
    assert not consumed_order_comparison({}, {}, 2)["verified"]


def test_runner_comparison_accepts_only_declared_sync_in_async_control():
    controls = {
        "reference": {"entrypoint": "standard", "trainer.fully_async.max_staleness_steps": 0},
        "candidate": {"entrypoint": "fully_async", "trainer.fully_async.max_staleness_steps": 0},
    }
    validate_sync_in_async_controls(controls)
    for key, value in (
        ("entrypoint", "unknown"),
        ("trainer.fully_async.weight_sync_interval", 2),
        ("trainer.fully_async.max_staleness_steps", 1),
    ):
        changed = deepcopy(controls)
        changed["candidate"][key] = value
        with pytest.raises(ValueError, match="Runner comparison requires"):
            validate_sync_in_async_controls(changed)
