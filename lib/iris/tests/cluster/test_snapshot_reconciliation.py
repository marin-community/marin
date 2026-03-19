# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for scaling group snapshot reconciliation.

Exercises the restore_scaling_group() function which reconciles checkpointed
slice state against pre-fetched cloud handles. This is the most critical
snapshot restore logic: bugs here cause orphaned VMs (resource leaks) or
lost slice inventory (capacity gaps).
"""


from iris.cluster.controller.scaling_group import (
    GroupSnapshot,
    SliceLifecycleState,
    SliceSnapshot,
    restore_scaling_group,
)
from iris.cluster.platform.base import Labels
from iris.time_utils import Timestamp
from tests.cluster.platform.fakes import (
    FakeSliceHandle,
    FakeWorkerHandle,
)


def _make_slice_snapshot(
    slice_id: str,
    scale_group: str = "tpu-group",
    lifecycle: str = "ready",
    worker_ids: list[str] | None = None,
    created_at_ms: int = 1000000,
    error_message: str = "",
) -> SliceSnapshot:
    return SliceSnapshot(
        slice_id=slice_id,
        scale_group=scale_group,
        lifecycle=lifecycle,
        worker_ids=worker_ids or [],
        created_at_ms=created_at_ms,
        last_active_ms=created_at_ms,
        error_message=error_message,
    )


def _make_fake_slice(
    slice_id: str,
    scale_group: str = "tpu-group",
    label_prefix: str = "test",
    worker_ids: list[str] | None = None,
) -> FakeSliceHandle:
    """Build a FakeSliceHandle with the right labels for filtering."""
    labels = Labels(label_prefix)
    slice_labels = {
        labels.iris_managed: "true",
        labels.iris_scale_group: scale_group,
    }
    addrs = worker_ids or ["10.0.0.1"]
    vms = [
        FakeWorkerHandle(
            vm_id=f"{slice_id}-vm-{i}",
            address=addr,
            created_at_ms=Timestamp.now().epoch_ms(),
        )
        for i, addr in enumerate(addrs)
    ]
    return FakeSliceHandle(
        slice_id=slice_id,
        scale_group=scale_group,
        zone="us-central1-a",
        vms=vms,
        labels=slice_labels,
    )


# =============================================================================
# Group 1: Slices present in both checkpoint and cloud
# =============================================================================


def test_restore_slice_in_checkpoint_and_cloud_preserves_lifecycle():
    """A READY slice in both checkpoint and cloud keeps its READY lifecycle."""
    slice_snap = _make_slice_snapshot("slice-1", lifecycle="ready", worker_ids=["10.0.0.1"])
    cloud_handle = _make_fake_slice("slice-1", worker_ids=["10.0.0.1"])

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        cloud_handles=[cloud_handle],
        label_prefix="test",
    )

    assert len(result.slices) == 1
    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.READY
    assert result.slices["slice-1"].handle is cloud_handle


def test_restore_booting_slice_that_became_ready_transitions_on_refresh():
    """A BOOTING slice from checkpoint with READY cloud state preserves BOOTING lifecycle.

    The autoscaler's next refresh() cycle will call describe(), see READY,
    and transition the slice. Restore just sets up the state correctly.
    """
    slice_snap = _make_slice_snapshot("slice-1", lifecycle="booting")
    cloud_handle = _make_fake_slice("slice-1")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        cloud_handles=[cloud_handle],
        label_prefix="test",
    )

    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.BOOTING
    assert result.slices["slice-1"].handle is cloud_handle


def test_restore_initializing_slice_with_cloud_ready():
    """An INITIALIZING slice from checkpoint preserves lifecycle regardless of cloud state."""
    slice_snap = _make_slice_snapshot("slice-1", lifecycle="initializing")
    cloud_handle = _make_fake_slice("slice-1")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        cloud_handles=[cloud_handle],
        label_prefix="test",
    )

    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.INITIALIZING
    assert result.slices["slice-1"].handle is cloud_handle


# =============================================================================
# Group 2: Slices in checkpoint but missing from cloud
# =============================================================================


def test_restore_discards_slice_missing_from_cloud():
    """A checkpoint slice not in the cloud is discarded."""
    slice_snap = _make_slice_snapshot("slice-gone", lifecycle="ready", worker_ids=["10.0.0.99"])

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        cloud_handles=[],
        label_prefix="test",
    )

    assert "slice-gone" not in result.slices
    assert result.discarded_count == 1


def test_restore_discards_failed_slice_missing_from_cloud():
    """A FAILED slice that disappeared from cloud is discarded cleanly."""
    slice_snap = _make_slice_snapshot("slice-failed", lifecycle="failed")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        cloud_handles=[],
        label_prefix="test",
    )

    assert "slice-failed" not in result.slices


def test_restore_multiple_slices_some_missing():
    """Present slices are kept; missing slices are discarded without corruption."""
    snap_alive = _make_slice_snapshot("slice-alive", lifecycle="ready")
    snap_gone = _make_slice_snapshot("slice-gone", lifecycle="ready")

    cloud_alive = _make_fake_slice("slice-alive")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(
            name="tpu-group",
            slices=[snap_alive, snap_gone],
        ),
        cloud_handles=[cloud_alive],
        label_prefix="test",
    )

    assert "slice-alive" in result.slices
    assert "slice-gone" not in result.slices
    assert result.slices["slice-alive"].handle is cloud_alive


# =============================================================================
# Group 3: Slices in cloud but NOT in checkpoint
# =============================================================================


def test_restore_adopts_unknown_cloud_slice_as_booting():
    """A cloud slice absent from checkpoint is adopted as BOOTING."""
    orphan = _make_fake_slice("slice-orphan")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[]),
        cloud_handles=[orphan],
        label_prefix="test",
    )

    assert "slice-orphan" in result.slices
    assert result.slices["slice-orphan"].lifecycle == SliceLifecycleState.BOOTING
    assert result.slices["slice-orphan"].handle is orphan
    assert result.adopted_count == 1


def test_restore_adopts_creating_cloud_slice():
    """A CREATING cloud slice is adopted as BOOTING."""
    creating = _make_fake_slice("slice-creating")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[]),
        cloud_handles=[creating],
        label_prefix="test",
    )

    assert "slice-creating" in result.slices
    assert result.slices["slice-creating"].lifecycle == SliceLifecycleState.BOOTING


def test_restore_mixed_known_and_unknown_slices():
    """Checkpoint has slice-a; cloud has slice-a and slice-b. slice-b is adopted."""
    snap_a = _make_slice_snapshot("slice-a", lifecycle="ready")

    cloud_a = _make_fake_slice("slice-a")
    cloud_b = _make_fake_slice("slice-b")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[snap_a]),
        cloud_handles=[cloud_a, cloud_b],
        label_prefix="test",
    )

    assert result.slices["slice-a"].lifecycle == SliceLifecycleState.READY
    assert result.slices["slice-b"].lifecycle == SliceLifecycleState.BOOTING
    assert result.adopted_count == 1


# =============================================================================
# Group 4: Multiple scaling groups
# =============================================================================


def test_restore_multiple_groups_independent_reconciliation():
    """Each scaling group reconciles independently."""
    label_prefix = "test"

    # Group A: slice-a1 alive, slice-a2 gone
    cloud_a1 = _make_fake_slice("slice-a1", scale_group="group-a", label_prefix=label_prefix)

    # Group B: slice-b1 alive, slice-b-orphan appeared during restart
    cloud_b1 = _make_fake_slice("slice-b1", scale_group="group-b", label_prefix=label_prefix)
    cloud_b_orphan = _make_fake_slice("slice-b-orphan", scale_group="group-b", label_prefix=label_prefix)

    result_a = restore_scaling_group(
        group_snapshot=GroupSnapshot(
            name="group-a",
            slices=[
                _make_slice_snapshot("slice-a1", scale_group="group-a"),
                _make_slice_snapshot("slice-a2", scale_group="group-a"),
            ],
        ),
        cloud_handles=[cloud_a1],
        label_prefix=label_prefix,
    )

    result_b = restore_scaling_group(
        group_snapshot=GroupSnapshot(
            name="group-b",
            slices=[_make_slice_snapshot("slice-b1", scale_group="group-b")],
        ),
        cloud_handles=[cloud_b1, cloud_b_orphan],
        label_prefix=label_prefix,
    )

    assert set(result_a.slices.keys()) == {"slice-a1"}
    assert set(result_b.slices.keys()) == {"slice-b1", "slice-b-orphan"}
    assert result_b.slices["slice-b-orphan"].lifecycle == SliceLifecycleState.BOOTING


def test_restore_empty_checkpoint_with_cloud_slices():
    """Empty checkpoint with existing cloud slices: all adopted."""
    cloud_1 = _make_fake_slice("slice-1")
    cloud_2 = _make_fake_slice("slice-2")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[]),
        cloud_handles=[cloud_1, cloud_2],
        label_prefix="test",
    )

    assert len(result.slices) == 2
    assert all(s.lifecycle == SliceLifecycleState.BOOTING for s in result.slices.values())


def test_restore_empty_checkpoint_empty_cloud():
    """Fresh start: no checkpoint, no cloud slices. Clean slate."""
    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[]),
        cloud_handles=[],
        label_prefix="test",
    )

    assert len(result.slices) == 0


# =============================================================================
# Group 8: Timing state
# =============================================================================


def test_restore_preserves_backoff_state():
    """Backoff timers survive checkpoint/restore."""
    # Set backoff_until to 5 minutes in the future
    backoff_ms = Timestamp.now().epoch_ms() + 300_000
    snapshot = GroupSnapshot(
        name="tpu-group",
        consecutive_failures=3,
        backoff_until_ms=backoff_ms,
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        cloud_handles=[],
        label_prefix="test",
    )

    assert result.consecutive_failures == 3
    assert result.backoff_active


def test_restore_expired_backoff_is_inactive():
    """Backoff that expired during the restart window is correctly inactive."""
    # Set backoff_until to 1 minute in the past
    backoff_ms = Timestamp.now().epoch_ms() - 60_000
    snapshot = GroupSnapshot(
        name="tpu-group",
        consecutive_failures=2,
        backoff_until_ms=backoff_ms,
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        cloud_handles=[],
        label_prefix="test",
    )

    assert result.consecutive_failures == 2
    assert not result.backoff_active


def test_restore_preserves_quota_exceeded_state():
    """Quota exceeded state and reason survive restore."""
    # Set quota_exceeded_until to 5 minutes in the future
    quota_ms = Timestamp.now().epoch_ms() + 300_000
    snapshot = GroupSnapshot(
        name="tpu-group",
        quota_reason="RESOURCE_EXHAUSTED: out of v5 TPUs in us-central2",
        quota_exceeded_until_ms=quota_ms,
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        cloud_handles=[],
        label_prefix="test",
    )

    assert result.quota_exceeded_active
    assert "v5 TPUs" in result.quota_reason
