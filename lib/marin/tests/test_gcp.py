# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.cluster import gcp

TPU_NODES = [
    {
        "name": "projects/hai-gcp-models/locations/us-east5-a/nodes/iris-v5p-slice",
        "networkEndpoints": [
            {
                "ipAddress": "10.202.0.55",
                "accessConfig": {"externalIp": "34.118.10.20"},
            }
        ],
    }
]


@pytest.mark.parametrize("target_ip", ["10.202.0.55", "34.118.10.20"])
def test_find_tpu_by_ip_matches_endpoint(target_ip):
    assert gcp._find_tpu_by_ip_in_nodes(target_ip, TPU_NODES, fallback_zone="-") == (
        "iris-v5p-slice",
        "us-east5-a",
        0,
    )


def test_find_tpu_by_name():
    assert gcp._find_tpu_by_name_in_nodes("iris-v5p-slice", TPU_NODES, fallback_zone="-") == (
        "iris-v5p-slice",
        "us-east5-a",
    )
