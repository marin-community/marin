# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for scaling group snapshot reconciliation.

Exercises the restore_scaling_group() function which reconciles checkpointed
slice state against live cloud state via platform.list_slices(). This is the
most critical snapshot restore logic: bugs here cause orphaned VMs (resource
leaks) or lost slice inventory (capacity gaps).
"""

import pytest

from iris.cluster.controller.scaling_group import (
    GroupSnapshot,
    SliceLifecycleState,
    SliceSnapshot,
    restore_scaling_group,
)
from iris.cluster.platform.base import Labels
from iris.rpc import config_pb2
from iris.time_utils import Timestamp
from tests.cluster.platform.fakes import (
    FakePlatform,
    FakePlatformConfig,
    FakeSliceHandle,
    FakeWorkerHandle,
)


def _make_scale_group_config(
    name: str = "tpu-group",
    min_slices: int = 0,
    max_slices: int = 10,
) -> config_pb2.ScaleGroupConfig:
    return config_pb2.ScaleGroupConfig(
        name=name,
        min_slices=min_slices,
        max_slices=max_slices,
        slice_template=config_pb2.SliceConfig(
            gcp=config_pb2.GcpSliceConfig(zone="us-central1-a"),
        ),
    )


def _make_fake_platform(config: config_pb2.ScaleGroupConfig) -> FakePlatform:
    return FakePlatform(FakePlatformConfig(config=config))


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
        vm_addresses=vm_addresses or [],
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
    """Build a FakeSliceHandle with the right labels for list_slices filtering."""
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


class ReconciliationEnv:
    """Test environment for snapshot reconciliation tests."""

    def __init__(
        self,
        group_name: str = "tpu-group",
        label_prefix: str = "test",
    ):
        self.config = _make_scale_group_config(name=group_name)
        self.platform = _make_fake_platform(self.config)
        self.label_prefix = label_prefix
        self.group_name = group_name

    def make_slice_snapshot(
        self,
        slice_id: str,
        lifecycle: str = "ready",
        worker_ids: list[str] | None = None,
        created_at_ms: int = 1000000,
    ) -> SliceSnapshot:
        return _make_slice_snapshot(
            slice_id=slice_id,
            scale_group=self.group_name,
            lifecycle=lifecycle,
            worker_ids=worker_ids,
            created_at_ms=created_at_ms,
        )

    def make_fake_slice(
        self,
        slice_id: str,
        worker_ids: list[str] | None = None,
    ) -> FakeSliceHandle:
        return _make_fake_slice(
            slice_id=slice_id,
            scale_group=self.group_name,
            label_prefix=self.label_prefix,
            worker_ids=worker_ids,
        )


@pytest.fixture
def reconciliation_env() -> ReconciliationEnv:
    return ReconciliationEnv()


# =============================================================================
# Group 1: Slices present in both checkpoint and cloud
# =============================================================================


def test_restore_slice_in_checkpoint_and_cloud_preserves_lifecycle(reconciliation_env: ReconciliationEnv):
    """A READY slice in both checkpoint and cloud keeps its READY lifecycle."""
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-1", lifecycle="ready", worker_ids=["10.0.0.1"])
    cloud_handle = env.make_fake_slice("slice-1", worker_ids=["10.0.0.1"])
    env.platform.inject_slice(cloud_handle)

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert len(result.slices) == 1
    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.READY
    assert result.slices["slice-1"].handle is cloud_handle


def test_restore_booting_slice_that_became_ready_transitions_on_refresh(reconciliation_env: ReconciliationEnv):
    """A BOOTING slice from checkpoint with READY cloud state preserves BOOTING lifecycle.

    The autoscaler's next refresh() cycle will call describe(), see READY,
    and transition the slice. Restore just sets up the state correctly.
    """
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-1", lifecycle="booting")
    cloud_handle = env.make_fake_slice("slice-1")
    env.platform.inject_slice(cloud_handle)

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.BOOTING
    assert result.slices["slice-1"].handle is cloud_handle


def test_restore_initializing_slice_with_cloud_ready(reconciliation_env: ReconciliationEnv):
    """An INITIALIZING slice from checkpoint preserves lifecycle regardless of cloud state."""
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-1", lifecycle="initializing")
    cloud_handle = env.make_fake_slice("slice-1")
    env.platform.inject_slice(cloud_handle)

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.slices["slice-1"].lifecycle == SliceLifecycleState.INITIALIZING
    assert result.slices["slice-1"].handle is cloud_handle


# =============================================================================
# Group 2: Slices in checkpoint but missing from cloud
# =============================================================================


def test_restore_discards_slice_missing_from_cloud(reconciliation_env: ReconciliationEnv):
    """A checkpoint slice not in the cloud is discarded."""
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-gone", lifecycle="ready", worker_ids=["10.0.0.99"])

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-gone" not in result.slices
    assert result.discarded_count == 1


def test_restore_discards_failed_slice_missing_from_cloud(reconciliation_env: ReconciliationEnv):
    """A FAILED slice that disappeared from cloud is discarded cleanly."""
    env = reconciliation_env
    slice_snap = env.make_slice_snapshot("slice-failed", lifecycle="failed")

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[slice_snap]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-failed" not in result.slices


def test_restore_multiple_slices_some_missing(reconciliation_env: ReconciliationEnv):
    """Present slices are kept; missing slices are discarded without corruption."""
    env = reconciliation_env
    snap_alive = env.make_slice_snapshot("slice-alive", lifecycle="ready")
    snap_gone = env.make_slice_snapshot("slice-gone", lifecycle="ready")

    cloud_alive = env.make_fake_slice("slice-alive")
    env.platform.inject_slice(cloud_alive)

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(
            name="tpu-group",
            slices=[snap_alive, snap_gone],
        ),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-alive" in result.slices
    assert "slice-gone" not in result.slices
    assert result.slices["slice-alive"].handle is cloud_alive


# =============================================================================
# Group 3: Slices in cloud but NOT in checkpoint
# =============================================================================


def test_restore_adopts_unknown_cloud_slice_as_booting(reconciliation_env: ReconciliationEnv):
    """A cloud slice absent from checkpoint is adopted as BOOTING."""
    env = reconciliation_env
    orphan = env.make_fake_slice("slice-orphan")
    env.platform.inject_slice(orphan)

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-orphan" in result.slices
    assert result.slices["slice-orphan"].lifecycle == SliceLifecycleState.BOOTING
    assert result.slices["slice-orphan"].handle is orphan
    assert result.adopted_count == 1


def test_restore_adopts_creating_cloud_slice(reconciliation_env: ReconciliationEnv):
    """A CREATING cloud slice is adopted as BOOTING."""
    env = reconciliation_env
    creating = env.make_fake_slice("slice-creating")
    env.platform.inject_slice(creating)

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert "slice-creating" in result.slices
    assert result.slices["slice-creating"].lifecycle == SliceLifecycleState.BOOTING


def test_restore_mixed_known_and_unknown_slices(reconciliation_env: ReconciliationEnv):
    """Checkpoint has slice-a; cloud has slice-a and slice-b. slice-b is adopted."""
    env = reconciliation_env
    snap_a = env.make_slice_snapshot("slice-a", lifecycle="ready")

    cloud_a = env.make_fake_slice("slice-a")
    cloud_b = env.make_fake_slice("slice-b")
    env.platform.inject_slice(cloud_a)
    env.platform.inject_slice(cloud_b)

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[snap_a]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
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

    config_a = _make_scale_group_config(name="group-a")
    config_b = _make_scale_group_config(name="group-b")
    # Use a single platform with slices from both groups
    platform = _make_fake_platform(config_a)

    # Group A: slice-a1 alive, slice-a2 gone
    cloud_a1 = _make_fake_slice("slice-a1", scale_group="group-a", label_prefix=label_prefix)
    platform.inject_slice(cloud_a1)

    # Group B: slice-b1 alive, slice-b-orphan appeared during restart
    cloud_b1 = _make_fake_slice("slice-b1", scale_group="group-b", label_prefix=label_prefix)
    cloud_b_orphan = _make_fake_slice("slice-b-orphan", scale_group="group-b", label_prefix=label_prefix)
    platform.inject_slice(cloud_b1)
    platform.inject_slice(cloud_b_orphan)

    result_a = restore_scaling_group(
        group_snapshot=GroupSnapshot(
            name="group-a",
            slices=[
                _make_slice_snapshot("slice-a1", scale_group="group-a"),
                _make_slice_snapshot("slice-a2", scale_group="group-a"),
            ],
        ),
        platform=platform,
        config=config_a,
        label_prefix=label_prefix,
    )

    result_b = restore_scaling_group(
        group_snapshot=GroupSnapshot(
            name="group-b",
            slices=[_make_slice_snapshot("slice-b1", scale_group="group-b")],
        ),
        platform=platform,
        config=config_b,
        label_prefix=label_prefix,
    )

    assert set(result_a.slices.keys()) == {"slice-a1"}
    assert set(result_b.slices.keys()) == {"slice-b1", "slice-b-orphan"}
    assert result_b.slices["slice-b-orphan"].lifecycle == SliceLifecycleState.BOOTING


def test_restore_empty_checkpoint_with_cloud_slices(reconciliation_env: ReconciliationEnv):
    """Empty checkpoint with existing cloud slices: all adopted."""
    env = reconciliation_env
    env.platform.inject_slice(env.make_fake_slice("slice-1"))
    env.platform.inject_slice(env.make_fake_slice("slice-2"))

    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[]),
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert len(result.slices) == 2
    assert all(s.lifecycle == SliceLifecycleState.BOOTING for s in result.slices.values())


def test_restore_empty_checkpoint_empty_cloud(reconciliation_env: ReconciliationEnv):
    """Fresh start: no checkpoint, no cloud slices. Clean slate."""
    result = restore_scaling_group(
        group_snapshot=GroupSnapshot(name="tpu-group", slices=[]),
        platform=reconciliation_env.platform,
        config=reconciliation_env.config,
        label_prefix=reconciliation_env.label_prefix,
    )

    assert len(result.slices) == 0


# =============================================================================
# Group 8: Timing state
# =============================================================================


def test_restore_preserves_backoff_state(reconciliation_env: ReconciliationEnv):
    """Backoff timers survive checkpoint/restore."""
    env = reconciliation_env
    # Set backoff_until to 5 minutes in the future
    backoff_ms = Timestamp.now().epoch_ms() + 300_000
    snapshot = GroupSnapshot(
        name="tpu-group",
        consecutive_failures=3,
        backoff_until_ms=backoff_ms,
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.consecutive_failures == 3
    assert result.backoff_active


def test_restore_expired_backoff_is_inactive(reconciliation_env: ReconciliationEnv):
    """Backoff that expired during the restart window is correctly inactive."""
    env = reconciliation_env
    # Set backoff_until to 1 minute in the past
    backoff_ms = Timestamp.now().epoch_ms() - 60_000
    snapshot = GroupSnapshot(
        name="tpu-group",
        consecutive_failures=2,
        backoff_until_ms=backoff_ms,
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.consecutive_failures == 2
    assert not result.backoff_active


def test_restore_preserves_quota_exceeded_state(reconciliation_env: ReconciliationEnv):
    """Quota exceeded state and reason survive restore."""
    env = reconciliation_env
    # Set quota_exceeded_until to 5 minutes in the future
    quota_ms = Timestamp.now().epoch_ms() + 300_000
    snapshot = GroupSnapshot(
        name="tpu-group",
        quota_reason="RESOURCE_EXHAUSTED: out of v5 TPUs in us-central2",
        quota_exceeded_until_ms=quota_ms,
    )

    result = restore_scaling_group(
        group_snapshot=snapshot,
        platform=env.platform,
        config=env.config,
        label_prefix=env.label_prefix,
    )

    assert result.quota_exceeded_active
    assert "v5 TPUs" in result.quota_reason
