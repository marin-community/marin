# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from iris.client.client import IrisClient

from experiments.post_training.snowball_isoflop import guard


def test_preempted_attempt_does_not_reserve_a_gpu_or_trigger_active_time_margin(monkeypatch):
    root = "/benfeuer/snowball-isoflop-B-preempted"
    ledger = {"root": root, "h100_hours": 1.5, "tasks": [{"state": "preempted", "h100_hours": 1.5}]}
    monkeypatch.setattr(guard, "allocation_ledger", lambda client, job: ledger)
    config = guard.GuardConfig((guard.PhaseBudget(guard.Phase.B, (root,), 0, 1.55),), closed=False)

    report = guard.budget_snapshot(Mock(spec=IrisClient), config)["phases"][0]

    assert report["observed_h100_hours"] == 1.5
    assert report["active_gpu_reservation"] == 0
    assert not report["limit_reached"]
