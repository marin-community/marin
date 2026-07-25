# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import click
import pytest
from iris.cli.job import resolve_multinode_defaults
from iris.cluster.platforms.k8s.coreweave_topology import balanced_rack_slice_size, gpu_gang_coscheduling_level


@pytest.mark.parametrize(
    "tpu,gpu,replicas,expected_replicas,expected_cosched",
    [
        # H100 InfiniBand: multi-node gangs coschedule on the soft leafgroup level.
        (None, "H100x8", 2, 2, "leafgroup"),
        (None, "H100x8", 1, 1, None),
        (None, "H100x8", None, 1, None),
        # GB200 NVL72: 16 nodes (the guaranteed-schedulable slice of a rack) is the largest
        # hard single-domain gang; a valid multi-rack size spills to the sliced level (one rack
        # slice per domain); a single node is not a gang.
        (None, "GB200x4", 2, 2, "nvlink.domain"),
        (None, "GB200x4", 16, 16, "nvlink.domain"),
        (None, "GB200x4", 32, 32, "nvlink.domain.sliced"),
        (None, "GB200x4", 1, 1, None),
        (None, "GB200", None, 1, None),
        (None, None, 2, 2, None),
        (None, None, None, 1, None),
    ],
)
def test_resolve_multinode_defaults_gpu(tpu, gpu, replicas, expected_replicas, expected_cosched):
    actual_replicas, cosched = resolve_multinode_defaults(tpu, gpu, replicas)
    assert actual_replicas == expected_replicas
    if expected_cosched:
        assert cosched.group_by == expected_cosched
    else:
        assert cosched is None


@pytest.mark.parametrize(
    "variant,replicas,expected",
    [
        # NVL72 GPUs: hard nvlink.domain up to 16 (the guaranteed-schedulable rack slice),
        # sliced (rack-sized slices, one per NVLink domain) at 17+ where a single hard domain
        # is not guaranteed all-healthy.
        ("GB200", 2, "nvlink.domain"),
        ("GB200", 16, "nvlink.domain"),
        ("GB200", 17, "nvlink.domain.sliced"),
        ("GB300", 4, "nvlink.domain"),
        ("GB300", 17, "nvlink.domain.sliced"),
        # H100 (and any non-NVL72 GPU) has no nvlink.domain label -> always leafgroup.
        ("H100", 2, "leafgroup"),
        ("H100", 64, "leafgroup"),
        # A bare-count request (empty variant) is not NVL72 -> leafgroup.
        ("", 2, "leafgroup"),
    ],
)
def test_gpu_gang_coscheduling_level(variant, replicas, expected):
    assert gpu_gang_coscheduling_level(variant, replicas) == expected


@pytest.mark.parametrize(
    "num_tasks,slice_size",
    [
        # Spread evenly over the fewest racks (<= 16 nodes each): 24 -> 12+12, 48 -> 16+16+16.
        (0, None),  # non-positive -> rejected (not ZeroDivisionError)
        (17, None),  # ceil(17/16)=2 racks, does not divide evenly -> rejected
        (18, None),  # 2 racks of 9, but two 9-node slices fit one 18-node rack -> rejected
        (216, None),  # ceil(216/16)=14 racks, 216 % 14 != 0 -> rejected (fewest-racks split only)
        (20, 10),
        (24, 12),
        (32, 16),
        (48, 16),
        (64, 16),
    ],
)
def test_balanced_rack_slice_size(num_tasks, slice_size):
    if slice_size is None:
        with pytest.raises(ValueError):
            balanced_rack_slice_size(num_tasks)
    else:
        assert balanced_rack_slice_size(num_tasks) == slice_size


@pytest.mark.parametrize(
    "gpu,replicas",
    [
        ("GB200x4", 17),  # sliced level, but 17 does not split evenly across racks
        ("GB200x4", 18),  # sliced level, but two 9-node slices would share a rack
        ("GB200", 32),  # sliced size is valid but pods are not node-saturating (1 GPU, not 4)
    ],
)
def test_resolve_multinode_defaults_rejects_bad_sliced_gang(gpu, replicas):
    """The CLI rejects a knowably-unplaceable multi-rack NVL72 gang at submit rather than letting
    the controller terminal-fail it after a round-trip."""
    with pytest.raises(click.UsageError):
        resolve_multinode_defaults(None, gpu, replicas)
