# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fungible reservation chip accounting and the preemption view."""

import pytest
from iris.cluster.controller.autoscaler.reserved_pool import (
    reserved_pool_usage,
    reserved_pool_view,
)
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.rpc import vm_pb2

from tests.cluster.backends.conftest import make_fake_slice_handle, make_mock_platform

from .conftest import make_scale_group_config


def _ready_group(
    name: str,
    variant: str,
    *,
    quota_pool: str,
    reservation_chips: int,
    slice_ids: list[str],
    vms_per_slice: int = 1,
) -> ScalingGroup:
    """Build a fungible-reservation ScalingGroup with READY slices.

    Each slice is discovered via the mock platform and marked READY so its
    worker ids populate, matching how the autoscaler tracks live slices.
    """
    config = make_scale_group_config(name=name, accelerator_variant=variant, max_slices=64)
    config.quota_pool = quota_pool
    config.reservation_chips = reservation_chips
    handles = [
        make_fake_slice_handle(
            sid,
            scale_group=name,
            vm_states=[vm_pb2.VM_STATE_READY] * vms_per_slice,
        )
        for sid in slice_ids
    ]
    platform = make_mock_platform(slices_to_discover=handles)
    group = ScalingGroup(config, platform)
    group.reconcile()
    for handle in handles:
        worker_ids = [w.worker_id for w in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids)
    return group


class TestReservedPoolUsage:
    def test_consumed_and_free_chips_sum_across_variants_in_a_pool(self):
        # Two v4-8 slices (4 chips each) + one v4-16 slice (8 chips) = 16 chips
        # consumed against a shared 64-chip budget.
        v8 = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=64, slice_ids=["a1", "a2"])
        v16 = _ready_group(
            "v4-16", "v4-16", quota_pool="pool-a", reservation_chips=64, slice_ids=["b1"], vms_per_slice=2
        )

        usage = reserved_pool_usage([v8, v16])

        assert set(usage) == {"pool-a"}
        pool = usage["pool-a"]
        assert pool.reservation_chips == 64
        assert pool.consumed_chips == 4 + 4 + 8
        assert pool.free_chips == 64 - 16
        assert pool.utilization == pytest.approx(16 / 64)

    def test_distinct_pools_bucket_separately(self):
        a = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=32, slice_ids=["a1"])
        b = _ready_group("v5p-8", "v5p-8", quota_pool="pool-b", reservation_chips=16, slice_ids=["b1"])

        usage = reserved_pool_usage([a, b])

        assert usage["pool-a"].consumed_chips == 4
        assert usage["pool-b"].consumed_chips == 4
        assert usage["pool-a"].free_chips == 28
        assert usage["pool-b"].free_chips == 12

    def test_non_fungible_groups_excluded(self):
        fungible = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=32, slice_ids=["a1"])
        # reservation_chips defaults to 0 -> not part of any fungible pool.
        plain_config = make_scale_group_config(name="plain", accelerator_variant="v4-8", max_slices=8)
        plain_config.quota_pool = "pool-a"
        plain = ScalingGroup(plain_config, make_mock_platform())

        usage = reserved_pool_usage([fungible, plain])

        assert set(usage) == {"pool-a"}
        # Only the fungible group's slice counts; the plain group is invisible.
        assert usage["pool-a"].consumed_chips == 4

    def test_conflicting_budgets_in_one_pool_raise(self):
        a = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=64, slice_ids=["a1"])
        b = _ready_group("v4-16", "v4-16", quota_pool="pool-a", reservation_chips=128, slice_ids=["b1"], vms_per_slice=2)

        with pytest.raises(ValueError, match="conflicting reservation_chips"):
            reserved_pool_usage([a, b])

    def test_reservation_chips_without_quota_pool_raises(self):
        config = make_scale_group_config(name="v4-8", accelerator_variant="v4-8", max_slices=8)
        config.reservation_chips = 64
        # quota_pool intentionally left empty.
        group = ScalingGroup(config, make_mock_platform())

        with pytest.raises(ValueError, match="no quota_pool"):
            reserved_pool_usage([group])


class TestReservedPoolView:
    def test_view_maps_workers_variants_and_free_chips(self):
        v8 = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=64, slice_ids=["a1"])
        v16 = _ready_group(
            "v4-16", "v4-16", quota_pool="pool-a", reservation_chips=64, slice_ids=["b1"], vms_per_slice=2
        )

        view = reserved_pool_view([v8, v16])

        assert view.variant_pool == {"v4-8": "pool-a", "v4-16": "pool-a"}
        assert view.chips_per_variant == {"v4-8": 4, "v4-16": 8}
        assert view.free_chips == {"pool-a": 64 - 12}
        # Every worker in both groups maps back to the shared pool.
        assert view.worker_pool == {
            "a1-vm-0": "pool-a",
            "b1-vm-0": "pool-a",
            "b1-vm-1": "pool-a",
        }
        assert view.pools_on_cooldown == frozenset()
        assert not view.is_empty()

    def test_cooldown_passed_through(self):
        v8 = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=32, slice_ids=["a1"])

        view = reserved_pool_view([v8], pools_on_cooldown=frozenset({"pool-a"}))

        assert view.pools_on_cooldown == frozenset({"pool-a"})

    def test_empty_when_no_fungible_groups(self):
        plain_config = make_scale_group_config(name="plain", accelerator_variant="v4-8", max_slices=8)
        plain = ScalingGroup(plain_config, make_mock_platform())

        view = reserved_pool_view([plain])

        assert view.is_empty()
        assert view.variant_pool == {}
        assert view.free_chips == {}
