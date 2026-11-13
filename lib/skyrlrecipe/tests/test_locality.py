# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from skyrlrecipe.locality import plan_locality_groups


@pytest.mark.parametrize(
    "requirements,hosts,roles",
    [
        ((32, 64), (8, 16), (0, 1)),
        ((64, 64, 64), (16, 16, 16), (0, 1, 2)),
        ((4, 6, 6, 6), (6,), (0, 0, 0, 0)),
        ((4, 18, 18, 18, 18, 18, 18, 18), (15, 14, 5), (0, 0, 0, 0, 1, 1, 1, 2)),
    ],
)
def test_local_models_request_whole_hosts_per_domain(requirements, hosts, roles):
    plan = plan_locality_groups(requirements, gpus_per_node=4, max_nodes_per_domain=16)
    assert plan.group_sizes == hosts
    assert plan.role_groups == roles
    assert [plan.group_for_rank(rank) for rank in range(plan.num_nodes)] == [
        index for index, count in enumerate(hosts) for _ in range(count)
    ]
