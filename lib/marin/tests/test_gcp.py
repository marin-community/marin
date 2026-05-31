# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.cluster import gcp


@pytest.mark.parametrize("target_ip", ["10.202.0.55", "34.118.10.20"])
def test_find_tpu_by_ip_matches_endpoint(monkeypatch, target_ip):
    monkeypatch.setattr(
        gcp,
        "list_tpu_nodes",
        lambda _project, _zone: [
            {
                "name": "projects/hai-gcp-models/locations/us-east5-a/nodes/iris-v5p-slice",
                "networkEndpoints": [
                    {
                        "ipAddress": "10.202.0.55",
                        "accessConfig": {"externalIp": "34.118.10.20"},
                    }
                ],
            }
        ],
    )

    assert gcp.find_tpu_by_ip(target_ip, "hai-gcp-models") == ("iris-v5p-slice", "us-east5-a", 0)


def test_find_tpu_by_name(monkeypatch):
    monkeypatch.setattr(
        gcp,
        "list_tpu_nodes",
        lambda _project, _zone: [
            {
                "name": "projects/hai-gcp-models/locations/us-central1-a/nodes/iris-v5p-slice",
                "networkEndpoints": [],
            }
        ],
    )

    assert gcp.find_tpu_by_name("iris-v5p-slice", "hai-gcp-models") == ("iris-v5p-slice", "us-central1-a")
