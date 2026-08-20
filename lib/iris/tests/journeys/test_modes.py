# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operating modes at the backend boundary."""

from iris.rpc import job_pb2


def test_dry_run_keeps_work_pending_without_reconcile_or_autoscale(dry_run_journey):
    job = dry_run_journey.submit("plan-only")

    dry_run_journey.step()

    assert dry_run_journey.task(job[0]).summary.state == job_pb2.TASK_STATE_PENDING
    assert dry_run_journey.backend_calls(kind="reconcile") == []
    assert dry_run_journey.backend_calls(kind="autoscale") == []
    assert dry_run_journey.backend_events() == []
