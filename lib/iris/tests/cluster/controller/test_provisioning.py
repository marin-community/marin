# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The autoscaler's iris.provisioning emission: failure classification and the
row written for each slice provisioning outcome.

Exercised through a real ScalingGroup + a fake finelog table, so the row's
authoritative resource_type/zone/variant and the ready-only latency rule are
covered without driving a live controller.
"""

import pytest
from iris.cluster.constraints import DeviceType
from iris.cluster.controller.autoscaler.provisioning import classify_create_failure
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.platforms.types import QuotaExhaustedError
from iris.cluster.stats.tables import ProvisioningOutcome
from iris.cluster.types import CapacityType
from iris.testing.backends import make_mock_platform, make_mock_slice_handle
from iris.testing.controller import (
    make_autoscaler,
    make_demand_entries,
    make_scale_group_config,
    mark_discovered_ready,
)
from rigging.timing import Timestamp


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
        max_slices=1,
        accelerator_variant="v6e",
        zones=["us-east5-b"],
        capacity_type=CapacityType.PREEMPTIBLE,
    )
    return ScalingGroup(config, make_mock_platform())


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


def test_ready_row_carries_identity_and_latency(group, monkeypatch):
    now = Timestamp.from_ms(6_000)
    monkeypatch.setattr(Timestamp, "now", classmethod(lambda cls: now))
    handle = make_mock_slice_handle("slice-001", all_ready=True, created_at_ms=1_000)
    platform = make_mock_platform(slices_to_discover=[handle])
    group = ScalingGroup(group.config, platform)
    group.reconcile()
    table = FakeTable()
    autoscaler = make_autoscaler({group.name: group}, provisioning_table=table)

    autoscaler.run_once([], {}, timestamp=now)

    (row,) = table.rows
    assert row.resource_type == "tpu"
    assert row.scale_group == "tpu_v6e-preemptible_8-us-east5-b"
    assert row.zone == "us-east5-b"
    assert row.accelerator_variant == "v6e"
    assert row.outcome == ProvisioningOutcome.READY
    assert row.worker_count == 1
    assert row.provision_latency_ms == 5_000


def test_no_sink_is_noop(group):
    """A scale-up remains operational when local mode configures no provisioning sink."""
    platform = make_mock_platform()
    group = ScalingGroup(group.config, platform)
    autoscaler = make_autoscaler({group.name: group})
    try:
        autoscaler.run_once(
            make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v6e"),
            {},
            timestamp=Timestamp.from_ms(1_000),
        )
        assert autoscaler.get_status().recent_actions[0].action_type == "scale_up"
    finally:
        autoscaler.shutdown()


def test_submit_time_stockout_records_stockout(group):
    """A create that fails at submit (QuotaExhaustedError — no slice handle, so no
    later describe() outcome) is recorded here; a stockout message classifies as
    STOCKOUT rather than being lost to the success rate."""
    platform = make_mock_platform()
    error = QuotaExhaustedError('There is no more capacity in the zone "us-east5-b"')
    platform.create_slice.side_effect = error
    group = ScalingGroup(group.config, platform)
    table = FakeTable()
    autoscaler = make_autoscaler({group.name: group}, provisioning_table=table)

    autoscaler.run_once(
        make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v6e"),
        {},
        timestamp=Timestamp.from_ms(1_000),
    )

    (row,) = table.rows
    assert row.outcome == ProvisioningOutcome.STOCKOUT
    assert row.provision_latency_ms == 0


def test_submit_time_create_error_records_error(group):
    """A non-quota create error at submit is recorded as an ERROR outcome."""
    platform = make_mock_platform()
    platform.create_slice.side_effect = RuntimeError("boom")
    group = ScalingGroup(group.config, platform)
    table = FakeTable()
    autoscaler = make_autoscaler({group.name: group}, provisioning_table=table)

    autoscaler.run_once(
        make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v6e"),
        {},
        timestamp=Timestamp.from_ms(1_000),
    )

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
