# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from iris.cli.job import resolve_multinode_defaults


@pytest.mark.parametrize(
    "tpu,gpu,replicas,expected_replicas,expected_cosched",
    [
        (None, "H100x8", 2, 2, "pool"),
        (None, "H100x8", 1, 1, None),
        (None, "H100x8", None, 1, None),
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
