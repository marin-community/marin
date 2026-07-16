# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cli.job import resolve_multinode_defaults
from iris.cluster.platforms.k8s.coreweave_topology import gpu_gang_coscheduling_level


@pytest.mark.parametrize(
    "tpu,gpu,replicas,expected_replicas,expected_cosched",
    [
        # H100 InfiniBand: multi-node gangs coschedule on the soft leafgroup level.
        (None, "H100x8", 2, 2, "leafgroup"),
        (None, "H100x8", 1, 1, None),
        (None, "H100x8", None, 1, None),
        # GB200 NVL72: 16 nodes (the guaranteed-schedulable slice of a rack) is the largest
        # hard nvlink.domain gang; 17 spills to the soft nvlink.domain.preferred level; a
        # single node is not a gang.
        (None, "GB200x4", 2, 2, "nvlink.domain"),
        (None, "GB200x4", 16, 16, "nvlink.domain"),
        (None, "GB200x4", 17, 17, "nvlink.domain.preferred"),
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
        # soft nvlink.domain.preferred at 17+ (a full rack is not guaranteed all-healthy).
        ("GB200", 2, "nvlink.domain"),
        ("GB200", 16, "nvlink.domain"),
        ("GB200", 17, "nvlink.domain.preferred"),
        ("GB300", 4, "nvlink.domain"),
        ("GB300", 17, "nvlink.domain.preferred"),
        # H100 (and any non-NVL72 GPU) has no nvlink.domain label -> always leafgroup.
        ("H100", 2, "leafgroup"),
        ("H100", 64, "leafgroup"),
        # A bare-count request (empty variant) is not NVL72 -> leafgroup.
        ("", 2, "leafgroup"),
    ],
)
def test_gpu_gang_coscheduling_level(variant, replicas, expected):
    assert gpu_gang_coscheduling_level(variant, replicas) == expected
