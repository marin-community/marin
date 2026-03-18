# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from fray.v2.types import CpuConfig, GpuConfig, TpuConfig

from fray.v2.iris_backend import resolve_coscheduling


@pytest.mark.parametrize(
    "device,replicas,expected_group_by",
    [
        (GpuConfig(variant="H100", count=8), 2, "pool"),
        (GpuConfig(variant="H100", count=8), 1, None),
        (TpuConfig(variant="v5p-8"), 2, "tpu-name"),
        (TpuConfig(variant="v5p-8"), 1, None),
        (CpuConfig(), 4, None),
    ],
)
def test_resolve_coscheduling(device, replicas, expected_group_by):
    result = resolve_coscheduling(device, replicas)
    if expected_group_by is None:
        assert result is None
    else:
        assert result.group_by == expected_group_by
