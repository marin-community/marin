# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cli.job import resolve_multinode_defaults
from iris.cluster.platforms.k8s.coreweave_topology import RACK_SIZE, gpu_gang_coscheduling_level


@pytest.mark.parametrize(
    "tpu,gpu,replicas,expected_replicas,expected_cosched",
    [
        # H100 InfiniBand: multi-node gangs coschedule on the soft leafgroup level.
        (None, "H100x8", 2, 2, "leafgroup"),
        (None, "H100x8", 1, 1, None),
        (None, "H100x8", None, 1, None),
        # GB200 NVL72: a multi-node gang that fits one rack binds hard to nvlink.domain,
        # a gang larger than a rack binds soft to nvlink.domain.preferred (pack into whole
        # racks), and a single node is not a gang.
        (None, "GB200x4", 2, 2, "nvlink.domain"),
        (None, "GB200x4", RACK_SIZE, RACK_SIZE, "nvlink.domain"),
        (None, "GB200x4", RACK_SIZE + 1, RACK_SIZE + 1, "nvlink.domain.preferred"),
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
        # NVL72 GPUs: hard nvlink.domain up to one rack, soft nvlink.domain.preferred beyond it.
        ("GB200", 2, "nvlink.domain"),
        ("GB200", RACK_SIZE, "nvlink.domain"),
        ("GB200", RACK_SIZE + 1, "nvlink.domain.preferred"),
        ("GB300", 4, "nvlink.domain"),
        ("GB300", RACK_SIZE + 1, "nvlink.domain.preferred"),
        # H100 (and any non-NVL72 GPU) has no nvlink.domain label -> always leafgroup.
        ("H100", 2, "leafgroup"),
        ("H100", 64, "leafgroup"),
        # A bare-count request (empty variant) is not NVL72 -> leafgroup.
        ("", 2, "leafgroup"),
    ],
)
def test_gpu_gang_coscheduling_level(variant, replicas, expected):
    assert gpu_gang_coscheduling_level(variant, replicas) == expected
