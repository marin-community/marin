# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore-path behavior for scale groups that left config (drain mode)."""

from iris.cluster.backends.types import CloudSliceState, ListedSlice
from iris.cluster.controller.autoscaler.recovery import AutoscalerCheckpoint, restore_autoscaler_state
from iris.cluster.controller.autoscaler.scaling_group import (
    GroupSnapshot,
    ScalingGroup,
    SliceLifecycleState,
    SliceSnapshot,
)
from iris.rpc import config_pb2
from rigging.timing import Duration

from tests.cluster.backends.conftest import make_fake_slice_handle, make_mock_platform


def _draining_group_builder(platform):
    """Mirror factory.create_autoscaler's drain-group construction: scale-to-zero."""

    def make(name: str) -> ScalingGroup:
        return ScalingGroup(
            config=config_pb2.ScaleGroupConfig(name=name, max_slices=0),
            platform=platform,
            idle_threshold=Duration.from_seconds(60),
        )

    return make


def _platform_listing(*handles) -> object:
    platform = make_mock_platform()
    platform.list_all_slices.return_value = [ListedSlice(handle=h, state=CloudSliceState.READY) for h in handles]
    return platform


def _slice_ids(group: ScalingGroup) -> set[str]:
    slices, _ = group.persistable_state()
    return {s.slice_id for s in slices}


def test_retired_group_with_live_vm_adopted_as_draining():
    """A scale group gone from config but with a live cloud slice is adopted scale-to-zero."""
    handle = make_fake_slice_handle("slice-retired", scale_group="retired-group", all_ready=True)
    platform = _platform_listing(handle)
    groups: dict[str, ScalingGroup] = {}  # retired-group is NOT configured
    checkpoint = AutoscalerCheckpoint(group_snapshots={}, tracked_worker_rows=[])

    restore_autoscaler_state(groups, checkpoint, platform, _draining_group_builder(platform))

    assert "retired-group" in groups
    drain = groups["retired-group"]
    assert _slice_ids(drain) == {"slice-retired"}  # live VM adopted so idle scaledown can reclaim it
    assert not drain.can_scale_up()  # max_slices=0: never grows


def test_retired_group_adoption_preserves_checkpoint_lifecycle():
    """First restart after config removal: snapshot + live VM → drained, lifecycle preserved."""
    handle = make_fake_slice_handle("slice-retired", scale_group="retired-group", all_ready=True)
    platform = _platform_listing(handle)
    snapshot = GroupSnapshot(
        name="retired-group",
        slices=[
            SliceSnapshot(
                slice_id="slice-retired",
                scale_group="retired-group",
                lifecycle=SliceLifecycleState.READY.value,
                worker_ids=["w0"],
            )
        ],
    )
    groups: dict[str, ScalingGroup] = {}
    checkpoint = AutoscalerCheckpoint(group_snapshots={"retired-group": snapshot}, tracked_worker_rows=[])

    restore_autoscaler_state(groups, checkpoint, platform, _draining_group_builder(platform))

    drain = groups["retired-group"]
    slices, _ = drain.persistable_state()
    assert [(s.slice_id, s.lifecycle, s.worker_ids) for s in slices] == [
        ("slice-retired", SliceLifecycleState.READY.value, ["w0"])
    ]
    assert not drain.can_scale_up()


def test_retired_group_not_adopted_when_drain_disabled():
    """With no drain builder (test/local mode), a retired group's live VMs are left untouched."""
    handle = make_fake_slice_handle("slice-retired", scale_group="retired-group", all_ready=True)
    platform = _platform_listing(handle)
    groups: dict[str, ScalingGroup] = {}
    checkpoint = AutoscalerCheckpoint(group_snapshots={}, tracked_worker_rows=[])

    restore_autoscaler_state(groups, checkpoint, platform, None)

    assert groups == {}


def test_configured_group_not_replaced_by_drain():
    """The drain pass must skip configured groups even when they have live cloud slices."""
    handle = make_fake_slice_handle("slice-cfg", scale_group="cfg-group", all_ready=True)
    platform = _platform_listing(handle)
    configured = ScalingGroup(config=config_pb2.ScaleGroupConfig(name="cfg-group", max_slices=4), platform=platform)
    groups = {"cfg-group": configured}
    checkpoint = AutoscalerCheckpoint(group_snapshots={}, tracked_worker_rows=[])

    restore_autoscaler_state(groups, checkpoint, platform, _draining_group_builder(platform))

    assert groups["cfg-group"] is configured  # not swapped for a scale-to-zero group
    assert configured.can_scale_up()  # still a normal, growable group
