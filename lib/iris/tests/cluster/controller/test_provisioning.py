# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The autoscaler's iris.provisioning emission: failure classification and the
row written for each slice provisioning outcome.

Exercised through a real ScalingGroup + a fake finelog table, so the row's
authoritative resource_type/zone/variant and the ready-only latency rule are
covered without driving a live controller.
"""

import pytest
from iris.cluster.controller.autoscaler.provisioning import classify_create_failure
from iris.cluster.controller.autoscaler.runtime import _ScaleUpOutcome, _ScaleUpRequest
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.platforms.types import InfraError, QuotaExhaustedError
from iris.cluster.stats.tables import ProvisioningOutcome
from iris.cluster.types import CapacityType
from rigging.timing import Timestamp
from tests.cluster.backends.conftest import make_mock_platform, make_mock_slice_handle
from tests.cluster.controller.conftest import make_autoscaler, make_scale_group_config, mark_discovered_ready


class FakeTable:
    """Captures finelog writes in memory."""

    def __init__(self):
        self.rows = []

    def write(self, rows):
        self.rows.extend(rows)


@pytest.fixture
def group():
    config = make_scale_group_config(
        name="tpu_v6e-preemptible_8-us-east5-b",
        accelerator_variant="v6e",
        zones=["us-east5-b"],
        capacity_type=CapacityType.PREEMPTIBLE,
    )
    return ScalingGroup(config, make_mock_platform())


@pytest.fixture
def autoscaler_with_sink(group):
    table = FakeTable()
    autoscaler = make_autoscaler({group.name: group}, provisioning_table=table)
    yield autoscaler, table
    autoscaler.shutdown()


@pytest.mark.parametrize(
    "message,expected",
    [
        ('There is no more capacity in the zone "us-east5-b"', ProvisioningOutcome.STOCKOUT),
        ("TPU operation failed: an internal error", ProvisioningOutcome.ERROR),
        ("", ProvisioningOutcome.ERROR),
    ],
)
def test_classify_create_failure(message, expected):
    assert classify_create_failure(message) == expected


def test_ready_row_carries_identity_and_latency(autoscaler_with_sink, group):
    autoscaler, table = autoscaler_with_sink
    created_at = Timestamp.from_ms(Timestamp.now().epoch_ms() - 5000)
    autoscaler._record_provisioning_outcome(group, ProvisioningOutcome.READY, created_at=created_at, worker_count=2)

    (row,) = table.rows
    assert row.resource_type == "tpu"
    assert row.scale_group == "tpu_v6e-preemptible_8-us-east5-b"
    assert row.zone == "us-east5-b"
    assert row.accelerator_variant == "v6e"
    assert row.outcome == ProvisioningOutcome.READY
    assert row.worker_count == 2
    assert row.provision_latency_ms >= 4000  # ~5s create->ready


def test_nonready_outcome_records_zero_latency(autoscaler_with_sink, group):
    """Latency is create→ready wall time; a slice that never readied records 0,
    not its time-to-failure, even when a created_at is available."""
    autoscaler, table = autoscaler_with_sink
    autoscaler._record_provisioning_outcome(
        group,
        ProvisioningOutcome.STOCKOUT,
        created_at=Timestamp.from_ms(Timestamp.now().epoch_ms() - 5000),
        error_message='There is no more capacity in the zone "us-east5-b"',
    )

    (row,) = table.rows
    assert row.outcome == ProvisioningOutcome.STOCKOUT
    assert row.provision_latency_ms == 0


def test_no_sink_is_noop(group):
    """Without a sink injected, recording must not raise (test/local mode)."""
    autoscaler = make_autoscaler({group.name: group})
    try:
        autoscaler._record_provisioning_outcome(group, ProvisioningOutcome.READY, created_at=Timestamp.now())
    finally:
        autoscaler.shutdown()


def _scale_up_outcome(autoscaler, group, error):
    request = _ScaleUpRequest(
        group=group,
        reason="demand",
        action=autoscaler._log_action("scale_up", group.name, status="pending"),
    )
    return _ScaleUpOutcome(request=request, error=error)


def test_submit_time_stockout_records_stockout(autoscaler_with_sink, group):
    """A create that fails at submit (QuotaExhaustedError — no slice handle, so no
    later describe() outcome) is recorded here; a stockout message classifies as
    STOCKOUT rather than being lost to the success rate."""
    autoscaler, table = autoscaler_with_sink
    error = QuotaExhaustedError('There is no more capacity in the zone "us-east5-b"')
    autoscaler._fold_scale_up(_scale_up_outcome(autoscaler, group, error), Timestamp.now())

    (row,) = table.rows
    assert row.outcome == ProvisioningOutcome.STOCKOUT


def test_submit_time_create_error_records_error(autoscaler_with_sink, group):
    """A non-quota create error at submit is recorded as an ERROR outcome."""
    autoscaler, table = autoscaler_with_sink
    autoscaler._fold_scale_up(_scale_up_outcome(autoscaler, group, InfraError("boom")), Timestamp.now())

    (row,) = table.rows
    assert row.outcome == ProvisioningOutcome.ERROR


def test_runtime_worker_loss_records_preempted():
    """A READY slice whose workers fail liveness (the heartbeat teardown path, not
    probe_health) is recorded as PREEMPTED so it doesn't pollute the create
    success rate."""
    config = make_scale_group_config(name="test-group", zones=["us-central1-a"])
    handle = make_mock_slice_handle("slice-001", all_ready=True)
    group = ScalingGroup(config, make_mock_platform(slices_to_discover=[handle]))
    group.reconcile()
    mark_discovered_ready(group, [handle])
    table = FakeTable()
    autoscaler = make_autoscaler({group.name: group}, provisioning_table=table)
    try:
        autoscaler.terminate_slices_for_workers(["slice-001-vm-0"])
    finally:
        autoscaler.shutdown()

    (row,) = [r for r in table.rows if r.outcome == ProvisioningOutcome.PREEMPTED]
    assert row.scale_group == "test-group"
